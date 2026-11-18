"""
train_fusion.py: Train & Calibrate Multimodal Cross-Attention Fusion Layer
Fulfills Capstone requirements: Trains the bidirectional cross-modal attention network
connecting Text (RoBERTa 768-d), Vision (EfficientNet 768-d), Frequency (FFT 128-d),
and Consistency (1-d) to classify into 4 veracity categories.

Usage:
  python train_fusion.py --epochs 5 --lr 0.0001 --batch_size 8
"""

import os
import sys
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from fusion.cross_attention import CrossAttentionFusion


class SyntheticMultimodalDataset(Dataset):
    """
    Multimodal dataset representing the 4 veracity categories:
      0: Likely Authentic (Real text, real photo, high consistency)
      1: Potentially Misleading (Real photo, fake/out-of-context headline, low consistency)
      2: Potentially Manipulated (Synthetic visual deepfake, standard claim)
      3: Multimodal Fake Content (Sensational text + synthetic visual artifacts)
    """

    def __init__(self, num_samples: int = 400, seed: int = 42):
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.samples = []

        for _ in range(num_samples):
            label = random.randint(0, 3)

            # 1. Text feature (768-d)
            if label in [1, 3]:  # Fake/Sensational text
                t_feat = np.random.normal(loc=0.8, scale=1.0, size=(768,))
            else:  # Authentic text
                t_feat = np.random.normal(loc=-0.5, scale=1.0, size=(768,))

            # 2. Vision feature (768-d)
            if label in [2, 3]:  # Deepfake / Synthetic visual
                v_feat = np.random.normal(loc=1.2, scale=1.0, size=(768,))
            else:  # Authentic visual
                v_feat = np.random.normal(loc=-0.6, scale=1.0, size=(768,))

            # 3. Frequency feature (128-d)
            if label in [2, 3]:  # High-frequency lattice artifacts
                f_feat = np.random.normal(loc=0.9, scale=0.8, size=(128,))
            else:  # Natural spectrum
                f_feat = np.random.normal(loc=-0.4, scale=0.8, size=(128,))

            # 4. Consistency score (1-d)
            if label in [1, 3]:  # Semantic mismatch
                c_score = np.array([np.random.uniform(0.1, 0.35)])
            else:  # High consistency
                c_score = np.array([np.random.uniform(0.65, 0.95)])

            self.samples.append({
                "t_feat": torch.tensor(t_feat, dtype=torch.float32),
                "v_feat": torch.tensor(v_feat, dtype=torch.float32),
                "f_feat": torch.tensor(f_feat, dtype=torch.float32),
                "c_score": torch.tensor(c_score, dtype=torch.float32),
                "label": torch.tensor(label, dtype=torch.long),
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def train(epochs: int = 5, lr: float = 1e-4, batch_size: int = 16):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"============================================================")
    print(f"   Training Multimodal Cross-Attention Fusion Layer")
    print(f"   Device: {device} | Epochs: {epochs} | Batch Size: {batch_size}")
    print(f"============================================================")

    # 70% Train, 15% Validation, 15% Test
    full_ds = SyntheticMultimodalDataset(num_samples=600)
    train_size = int(0.70 * len(full_ds))
    val_size = int(0.15 * len(full_ds))
    test_size = len(full_ds) - train_size - val_size

    train_ds, val_ds, test_ds = torch.utils.data.random_split(
        full_ds, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    print(f"Dataset split: Train = {len(train_ds)}, Val = {len(val_ds)}, Test = {len(test_ds)}")

    model = CrossAttentionFusion(
        text_dim=768, vision_dim=768, forensic_dim=128, shared_dim=512, num_classes=4
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)

    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch in train_loader:
            t_f = batch["t_feat"].to(device)
            v_f = batch["v_feat"].to(device)
            f_f = batch["f_feat"].to(device)
            c_s = batch["c_score"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            outputs = model(t_f, v_f, f_f, c_s)
            logits = outputs["logits"]

            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(labels)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += len(labels)

        train_loss = total_loss / total
        train_acc = correct / total

        # Validation phase
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch in val_loader:
                t_f = batch["t_feat"].to(device)
                v_f = batch["v_feat"].to(device)
                f_f = batch["f_feat"].to(device)
                c_s = batch["c_score"].to(device)
                labels = batch["label"].to(device)

                outputs = model(t_f, v_f, f_f, c_s)
                preds = torch.argmax(outputs["logits"], dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += len(labels)

        val_acc = val_correct / val_total
        print(f"Epoch [{epoch:02d}/{epochs:02d}] - Loss: {train_loss:.4f} | Train Acc: {train_acc:.1%} | Val Acc: {val_acc:.1%}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_dir = os.path.join(BASE_DIR, "models")
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, "best_cross_attention_weights.pth")
            torch.save(model.state_dict(), save_path)

    # Final Testing Evaluation
    print(f"\n================ Final Evaluation on Holdout Test Set ================")
    model.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for batch in test_loader:
            t_f = batch["t_feat"].to(device)
            v_f = batch["v_feat"].to(device)
            f_f = batch["f_feat"].to(device)
            c_s = batch["c_score"].to(device)
            labels = batch["label"].to(device)

            outputs = model(t_f, v_f, f_f, c_s)
            preds = torch.argmax(outputs["logits"], dim=1)
            test_correct += (preds == labels).sum().item()
            test_total += len(labels)

    test_acc = test_correct / test_total
    print(f"Test Accuracy: {test_acc:.1%} ({test_correct}/{test_total} correct)")
    print(f"Model saved to: {os.path.join(BASE_DIR, 'models', 'best_cross_attention_weights.pth')}")
    print(f"======================================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    args = parser.parse_args()

    train(epochs=args.epochs, lr=args.lr, batch_size=args.batch_size)
