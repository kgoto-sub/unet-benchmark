import os
# ファイル・フォルダ操作用。データセットのパスを扱ったり、画像ファイル一覧を取得するときに使う。

import matplotlib.pyplot as plt
import torch
import torch.optim as optim
# PyTorch のオプティマイザ。Adam, SGDなどの学習に必要な最適化アルゴリズムを使う。

from monai.losses import DiceLoss
# MONAI の DiceLoss、セグメンテーションでよく使う「Dice 係数を最大化したい」ための損失関数。

from monai.metrics import DiceMetric
# MONAI の Dice で性能を評価するための指標。

from monai.networks.nets import SwinUNETR, UNet
# 医療画像向けの深層学習ネットワーク。セグメンテーション用の定番 CNN transformer ベースの高性能 U-Net

from torch.utils.data import DataLoader, Dataset
# データをモデルに渡すための仕組み。


class SimpleDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    

'''
SimpleDatasetクラスを作成

__init__：dataの内容を保存。
__len__：データの要素数？を保存。
__getitem__：指定したidx番目のデータを返す。

'''


def get_dummy_data(num_samples=50, img_size=(96, 96)):
    data_list = []
    H, W = img_size

    for _ in range(num_samples):
        # 1. 背景: 弱いノイズ (0.0を中心に分散0.1程度)
        img = torch.randn(1, H, W) * 0.1
        # 2. ラベル: 全て0で初期化
        lbl = torch.zeros(1, H, W)

        # 3. ランダムな四角形を作成
        # 四角のサイズ (10px ~ 40px)
        rh = torch.randint(10, 40, (1,)).item()
        rw = torch.randint(10, 40, (1,)).item()

        # 四角の左上座標 (はみ出さないように計算)
        y = torch.randint(0, H - rh, (1,)).item()
        x = torch.randint(0, W - rw, (1,)).item()

        # 画像: 四角の領域の輝度を上げる (+1.0)
        img[0, y : y + rh, x : x + rw] += 1.0

        # ラベル: 四角の領域を正解(1)にする
        lbl[0, y : y + rh, x : x + rw] = 1.0

        data_list.append({"image": img, "label": lbl})

    return data_list

'''
辞書型で、テンソルがimageとlabelで出力されている。
50個の辞書リスト
例）
{
    "image": torch.Tensor(1, 96, 96),
    "label": torch.Tensor(1, 96, 96)
}
'''




def train_and_evaluate(model_name, model, train_loader, device):

    # モデルの正確性を評価
    loss_function = DiceLoss(to_onehot_y=True, softmax=True)
    # DiceLoss：予測と正解の重なり具合を測る。重なりが多いほどLossが小さくなる。
    # to_onehot_y=True：ラベルをone-hotの形式に変換
    # softmax=True：出力にsoftmax関数化してから計算

    '''
    softmax関数：値の集合（スコア）を 0〜1 の確率に変換し、合計を 1 にする関数。
    値をeの何乗という形に変換し、それを総スコアで割るもの。この値がおこる確率として使用できる。
    '''

    # パラメータの調整
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    # optim：アルゴリズムを最適化
    # Adam：ニューラルネットが学習するために必要な「パラメータ更新の方法」・勾配を見ながら、自動的に学習率を調整してくれる
    # Ir：学習率

    dice_metric = DiceMetric(include_background=False, reduction="mean")
    # DiceMetric：モデルの精度を測るためのツール（評価）
    # include_background=False：背景（0番クラス）はスコアに含めない

    # モデルを学習用の設定に変更？
    model.to(device)
    # モデルを学習用に切り替える

    model.train()
    # モデルを「学習モード」に切り替える。dropout や batch normalization が学習用の動きになる？

    print(f"--- Training {model_name} ---")
    for epoch in range(10):  # デモ用に短く設定
        # 1epochを10回繰返す

        epoch_loss = 0
        # 1epochの損失を記録

        for batch in train_loader:
            # batchは辞書型？ train_loaderってどこで作成されたんだ？

            inputs, labels = batch["image"].to(device), batch["label"].to(device)
            # デバイスにimageとlabelをそれぞれ入れている。

            optimizer.zero_grad()
            # 前回の学習で用いた勾配を0にセット

            outputs = model(inputs)
            # inputs：入力画像 → モデルへ → 出力マスク（確率マップ）？

            loss = loss_function(outputs, labels)
            # クラスのロス計算にいれている

            loss.backward()
            # 逆伝播：損失が小さくなるように、どのパラメータをどう変えればいいか計算

            optimizer.step()
            # 計算した勾配にしたがってモデルを少し賢くする。

            epoch_loss += loss.item()
            # lossを足し込む。.item() で loss を Python の数値に変換。

        print(f"Epoch {epoch+1}: Loss = {epoch_loss/len(train_loader):.4f}")
        # 1epochの平均lossを計算

    model.eval()
    # 評価モードに切り替え

    with torch.no_grad():
        # 勾配を計算しない

        for batch in train_loader:
            inputs, labels = batch["image"].to(device), batch["label"].to(device)
            outputs = model(inputs)
            outputs = torch.argmax(outputs, dim=1, keepdim=True)
            # ピクセルごとに確率を計算し、それに応じて描写を行っている？
            dice_metric(y_pred=outputs, y=labels)
            # 予測と正解の重なり度（diceスコア）を計算、記録

    score = dice_metric.aggregate().item()
    # 渡したスコアの平均値を記録

    dice_metric.reset()
    # 別のモデルを評価して値が混ざらないように

    print(f"{model_name} Final Dice Score: {score:.4f}\n")
    # 1.0→一致　0.0→一致していない



# 学習したモデルが、画像をどのようにセグメンテーションしているかを可視化するための関数
def save_prediction_sample(
    model, loader, device, filename="result.png", title="Prediction"
):
    # model：学習済みモデル、loader：DataLoader（画像とラベルが入っている）、device：CPU or GPU
    # filename：保存する画像のパス、title：表示タイトル（モデル名などを入れられる）

    model.eval()
    batch = next(iter(loader))
    img, label = batch["image"].to(device), batch["label"].to(device)
    # iter(loader) でイテレータを作る。next(...) で最初のバッチを 1 回だけ取り出す

    with torch.no_grad():
        output = model(img)
        pred = torch.argmax(output, dim=1)

    # numpy に変換
    img_np = img[0, 0].cpu().numpy()
    # バッチの 0 番目を可視化

    label_np = label[0, 0].cpu().numpy()
    # チャンネル（0番目）も選択

    pred_np = pred[0].cpu().numpy()
    # CPU に戻して numpy に変換 → Matplotlib が描画できる形式になる

    # 元画像
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(img_np, cmap="gray")
    plt.axis("off")

    # 正解マスク表示
    plt.subplot(1, 3, 2)
    plt.title("Ground Truth Mask")
    plt.imshow(label_np, cmap="gray")
    plt.axis("off")

    # 予測マスク
    plt.subplot(1, 3, 3)
    plt.title(f"{title} Mask")
    plt.imshow(pred_np, cmap="gray")
    plt.axis("off")

    plt.tight_layout()

    # 保存先フォルダを作成
    output_dir = os.path.dirname(filename)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # 画像ファイルとして保存
    plt.savefig(filename)
    plt.close()
    print(f"Saved visualization to {filename}")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # cuda(gpu)が使用できればそれで行う。

    IMG_SIZE = (256, 256)
    BATCH_SIZE = 4
    # 学習画像のサイズとバッチサイズを設定。ミニバッチで読み込む枚数を指定

    # ダミーデータの作成とdataloader
    train_files = get_dummy_data(img_size=IMG_SIZE)
    train_ds = SimpleDataset(train_files)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    # データを「バッチサイズ 4 枚ずつ」「シャッフルしながら」供給する仕組み

    # U-Net モデルの作成
    unet_model = UNet(
        spatial_dims=2,
        # 2Dと設定

        in_channels=1,
        # 入力：1ch の画像

        out_channels=2,
        # 出力：2クラス（背景＋領域）

        channels=(16, 32, 64, 128, 256),
        # 5段階のエンコーダー（channels 参照）

        strides=(2, 2, 2, 2),
        # strides によりダウンサンプリングを行う
        num_res_units=2,
        # Residual Block を 2 個積む
    )

    # SwinUNETR モデル作成
    swin_model = SwinUNETR(
        in_channels=1, out_channels=2, feature_size=24, spatial_dims=2
    )

    # 2 種類のモデルを学習・評価
    train_and_evaluate("Standard U-Net", unet_model, train_loader, device)
    train_and_evaluate("CIS-UNet (SwinUNETR)", swin_model, train_loader, device)
    # DiceLoss で学習・Adam で最適化・DiceMetric で精度評価・結果をログとして表示


    # 画像として保存
    save_prediction_sample(
        unet_model, train_loader, device, "./data/unet_prediction.png", "Standard U-Net"
    )
    save_prediction_sample(
        swin_model, train_loader, device, "./data/cis_unet_prediction.png", "CIS-UNet"
    )


'''
メモ
softmax 関数とは
出力されるデータの形式
つまりepochごとに学習していくのか

'''

'''
(画像生成)
 → image: (1, H, W)
 → label: (1, H, W)

DataLoader
 → (B, 1, H, W)

モデルの出力
 → (B, 2, H, W) ※確率マップ

argmax
 → (B, 1, H, W) ※クラスID

可視化
 → (H, W)
'''

'''
テンソルとは
数を並べたデータの入れ物
スカラー・ベクトル・行列を一般化

テンソルは「多次元の箱」
| 名前            | 次元 | 例               | 内容                  |
| ------------- | -- | --------------- | ------------------- |
| スカラー（0次元テンソル） | 0D | `3`             | ただの数1個              |
| ベクトル（1次元テンソル） | 1D | `[1,2,3]`       | 数の並び                |
| 行列（2次元テンソル）   | 2D | `[[1,2],[3,4]]` | 行と列                 |
| 3次元テンソル       | 3D | 画像（高さ×幅×チャネル）   | `C x H x W`         |
| 4次元テンソル       | 4D | ミニバッチ画像         | `B x C x H x W`     |
| 5次元テンソル       | 5D | 動画              | `B x T x C x H x W` |

'''