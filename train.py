import os
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.networks.nets import SwinUNETR, UNet
from torch.utils.data import DataLoader, Dataset

# train, testの分割をランダムにするために導入
import random

# 画像読み込みに必要なライブラリ
from PIL import Image
from torchvision import transforms


class SimpleDataset(Dataset):
    def __init__(self, data, img_size=(256, 256)):
        self.data = data

        # 画像とマスクをテンソルに変換し、指定サイズにリサイズ
        self.transform = transforms.Compose(
            [
                transforms.Resize(img_size),
                transforms.ToTensor(),  # PIL Imageを[C, H, W]形式のTensorに変換し、値を[0.0, 1.0]に正規化
            ]
        )
        # マスクのバイナリ化は__getitem__内で個別に行う

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # 画像の読み込み: グレースケール("L")で読み込み
        img_path = item["image"]
        image = Image.open(img_path).convert("L")

        # ラベル/マスクの読み込み: グレースケール("L")で読み込み
        label_path = item["label"]
        label = Image.open(label_path).convert("L")

        # 変換を適用
        image_tensor = self.transform(image)
        label_tensor = self.transform(label)

        # マスクはセグメンテーションのため、値を0または1に変換 (二値化)
        label_tensor = (label_tensor > 0.5).float()

        return {"image": image_tensor, "label": label_tensor}


def get_file_paths(root_dir="Dataset_BUSI_with_GT"):
    data_list = []

    for sub_dir in ["benign", "malignant", "normal"]:
        current_dir = os.path.join(root_dir, sub_dir)
        files = os.listdir(current_dir)

        # マスクファイルでない画像ファイル (.png) を抽出
        image_files = sorted(
            [f for f in files if not f.endswith("_mask.png") and f.endswith(".png")]
        )

        for image_name in image_files:
            base_name = image_name.replace(".png", "")
            mask_name = base_name + "_mask.png"

            image_path = os.path.join(current_dir, image_name)
            mask_path = os.path.join(current_dir, mask_name)

            if os.path.exists(mask_path):
                data_list.append({"image": image_path, "label": mask_path})

    return data_list


# データをトレインとテストに分割
def train_test_split(data_list, test_size=0.2, seed=42):
    random.seed(seed)
    random.shuffle(data_list)
    test_count = int(len(data_list) * test_size)
    train_data = data_list[test_count:]
    test_data = data_list[:test_count]
    return train_data, test_data

def train_test_and_evaluate(model_name, model, train_loader, test_loader, device):
    loss_function = DiceLoss(to_onehot_y=True, softmax=True)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    dice_metric = DiceMetric(include_background=False, reduction="mean")

    model.to(device)

    # train
    model.train()
    print(f"--- Training {model_name} ---")
    for epoch in range(50):
        epoch_loss = 0
        for batch in train_loader:
            inputs, labels = batch["image"].to(device), batch["label"].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}: Train Loss = {epoch_loss/len(train_loader):.4f}")

    # test
    model.eval()
    print(f"--- Evaluating {model_name} on Test Set ---")
    with torch.no_grad():
        for batch in test_loader:
            inputs, labels = batch["image"].to(device), batch["label"].to(device)
            outputs = model(inputs)
            outputs = torch.argmax(outputs, dim=1, keepdim=True)
            dice_metric(y_pred=outputs, y=labels)

    score = dice_metric.aggregate().item()
    dice_metric.reset()
    print(f"{model_name} Final Test Dice Score: {score:.4f}\n")
    return score


def save_prediction_sample(
    model, loader, device, filename="result.png", title="Prediction"
):
    model.eval()
    batch = next(iter(loader))
    img, label = batch["image"].to(device), batch["label"].to(device)

    with torch.no_grad():
        output = model(img)
        # out_channels=2から最も確率の高いクラス (0 or 1) を取得
        pred = torch.argmax(output, dim=1)

    img_np = img[0, 0].cpu().numpy()
    label_np = label[0, 0].cpu().numpy()
    pred_np = pred[0].cpu().numpy()

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(img_np, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("Ground Truth Mask")
    plt.imshow(label_np, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title(f"{title} Mask")
    plt.imshow(pred_np, cmap="gray")
    plt.axis("off")

    plt.tight_layout()

    output_dir = os.path.dirname(filename)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    plt.savefig(filename)
    plt.close()
    print(f"Saved visualization to {filename}")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    IMG_SIZE = (256, 256)
    BATCH_SIZE = 16

    # データのロードと分割
    all_files = get_file_paths(root_dir="Dataset_BUSI_with_GT")
    train_files, test_files = train_test_split(all_files, test_size=0.2)

    train_ds = SimpleDataset(train_files, img_size=IMG_SIZE)
    test_ds = SimpleDataset(test_files, img_size=IMG_SIZE)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False
    )  # テストからモデルは学習しないためシャッフル不要とした

    unet_model = UNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=2,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    )

    swin_model = SwinUNETR(
        in_channels=1, out_channels=2, feature_size=24, spatial_dims=2
    )

    unet_score = train_test_and_evaluate(
        "Standard U-Net", unet_model, train_loader, test_loader, device
    )
    swin_score = train_test_and_evaluate(
        "CIS-UNet (SwinUNETR)", swin_model, train_loader, test_loader, device
    )

    # 結果の比較
    print("\n--- Final Test Scores ---")
    print(f"Standard U-Net Dice Score: {unet_score:.4f}")
    print(f"CIS-UNet (SwinUNETR) Dice Score: {swin_score:.4f}")

    # 予測サンプルの生成にtest_loaderを使用
    save_prediction_sample(
        unet_model,
        test_loader,  # テストデータでの予測を表示
        device,
        "./output/unet_prediction.png",
        "Standard U-Net",
    )
    save_prediction_sample(
        swin_model,
        test_loader,
        device,
        "./output/cis_unet_prediction.png",
        "CIS-UNet",  # テストデータでの予測を表示
    )
