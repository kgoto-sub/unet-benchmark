import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.networks.nets import SwinUNETR, UNet
from torch.utils.data import DataLoader, Dataset


class SimpleDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


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


def train_and_evaluate(model_name, model, train_loader, device):
    loss_function = DiceLoss(to_onehot_y=True, softmax=True)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    dice_metric = DiceMetric(include_background=False, reduction="mean")

    model.to(device)
    model.train()

    print(f"--- Training {model_name} ---")
    for epoch in range(10):  # デモ用に短く設定
        epoch_loss = 0
        for batch in train_loader:
            inputs, labels = batch["image"].to(device), batch["label"].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}: Loss = {epoch_loss/len(train_loader):.4f}")

    model.eval()
    with torch.no_grad():
        for batch in train_loader:
            inputs, labels = batch["image"].to(device), batch["label"].to(device)
            outputs = model(inputs)
            outputs = torch.argmax(outputs, dim=1, keepdim=True)
            dice_metric(y_pred=outputs, y=labels)

    score = dice_metric.aggregate().item()
    dice_metric.reset()
    print(f"{model_name} Final Dice Score: {score:.4f}\n")


def save_prediction_sample(
    model, loader, device, filename="result.png", title="Prediction"
):
    model.eval()
    batch = next(iter(loader))
    img, label = batch["image"].to(device), batch["label"].to(device)

    with torch.no_grad():
        output = model(img)
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
    plt.savefig(filename)
    plt.close()
    print(f"Saved visualization to {filename}")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    IMG_SIZE = (256, 256)
    BATCH_SIZE = 4

    train_files = get_dummy_data(img_size=IMG_SIZE)
    train_ds = SimpleDataset(train_files)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

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

    train_and_evaluate("Standard U-Net", unet_model, train_loader, device)
    train_and_evaluate("CIS-UNet (SwinUNETR)", swin_model, train_loader, device)

    save_prediction_sample(
        unet_model, train_loader, device, "./data/unet_prediction.png", "Standard U-Net"
    )
    save_prediction_sample(
        swin_model, train_loader, device, "./data/cis_unet_prediction.png", "CIS-UNet"
    )
