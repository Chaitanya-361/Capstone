"""
train_vision.py: Fine-Tune EfficientNet-B0 on the 140k Real & Fake Faces Benchmark
Trains the vision classification head on Real (FFHQ) vs Synthetic (StyleGAN) face images.

Usage:
  python train_vision.py --epochs 3 --batch_size 16 --limit 200
"""

import os
import sys
import glob
import argparse
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)


class FaceDataset(Dataset):
    def __init__(self, data_dir: str, limit_per_class: int = 200, transform=None):
        self.samples = []
        self.transform = transform

        real_dir = os.path.join(data_dir, "real")
        fake_dir = os.path.join(data_dir, "fake")

        real_files = glob.glob(os.path.join(real_dir, "*.*"))[:limit_per_class]
        fake_files = glob.glob(os.path.join(fake_dir, "*.*"))[:limit_per_class]

        for p in real_files:
            self.samples.append((p, 0))  # 0: Authentic
        for p in fake_files:
            self.samples.append((p, 1))  # 1: Deepfake / Synthetic

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.long)


def fine_tune_vision(epochs: int = 3, batch_size: int = 16, limit: int = 200, lr: float = 1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("============================================================")
    print("   Fine-Tuning Vision Branch (EfficientNet-B0)")
    print(f"   Device: {device} | Epochs: {epochs} | Batch Size: {batch_size}")
    print("============================================================")

    data_dir = os.path.join(BASE_DIR, "data", "image_eval")
    if not os.path.exists(data_dir):
        print(f"Data directory not found at: {data_dir}")
        return

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    full_ds = FaceDataset(data_dir, limit_per_class=limit, transform=train_transform)
    total_len = len(full_ds)
    train_len = int(0.8 * total_len)
    val_len = total_len - train_len

    train_ds, val_ds = torch.utils.data.random_split(
        full_ds, [train_len, val_len], generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    print(f"Dataset Loaded: {total_len} samples (Train = {train_len}, Val = {val_len})")

    # Load EfficientNet-B0 backbone
    model = models.efficientnet_b0(num_classes=2)

    # Freeze convolutional feature layers for transfer learning
    for param in model.features.parameters():
        param.requires_grad = False

    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(1280, 2)
    )
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=lr, weight_decay=1e-2)

    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(labels)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += len(labels)

        train_loss = total_loss / total
        train_acc = correct / total

        # Validation phase
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                preds = torch.argmax(outputs, dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += len(labels)

        val_acc = val_correct / val_total
        print(f"Epoch [{epoch:02d}/{epochs:02d}] - Loss: {train_loss:.4f} | Train Acc: {train_acc:.1%} | Val Acc: {val_acc:.1%}")

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            save_dir = os.path.join(BASE_DIR, "models")
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, "best_vision_finetuned.pth")
            torch.save(model.state_dict(), save_path)

    print("============================================================")
    print(f"Fine-Tuning Complete! Best Val Acc: {best_val_acc:.1%}")
    print(f"Saved weights to: {os.path.join(BASE_DIR, 'models', 'best_vision_finetuned.pth')}")
    print("============================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--limit", type=int, default=100, help="Limit samples per class for quick training")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    args = parser.parse_args()

    fine_tune_vision(epochs=args.epochs, batch_size=args.batch_size, limit=args.limit, lr=args.lr)
