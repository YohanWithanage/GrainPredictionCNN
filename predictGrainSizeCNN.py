"""
cnn_grain_size.py
-----------------
Convolutional Neural Network (CNN) regression to predict mean grain size
from microstructure images.

Input:  Microstructure images (from Images/ folder)
Target: grain_size_um column from the parameters CSV

Usage:
    python cnn_grain_size.py --csv image_parameters.csv --images_dir Images/

Options:
    --img_size    Resize images to NxN (default: 128)
    --epochs      Training epochs (default: 50)
    --batch_size  Batch size (default: 16)
    --lr          Learning rate (default: 1e-3)
    --outdir      Output directory for model + plots (default: cnn_output/)
"""

import os
import sys
import argparse
import csv
import warnings
warnings.filterwarnings('ignore')
import joblib

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Dataset
# ─────────────────────────────────────────────────────────────────────────────

class GrainDataset(Dataset):
    def __init__(self, image_paths, targets, img_size=128, augment=False):
        self.image_paths = image_paths
        self.targets     = torch.tensor(targets, dtype=torch.float32)
        self.img_size    = img_size

        # Base transforms
        base = [
            transforms.Resize((img_size, img_size)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),                          # [0,1]
            transforms.Normalize(mean=[0.5], std=[0.5]),   # [-1,1]
        ]

        # Augmentation for training only
        aug = [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(90),
        ] if augment else []

        self.transform = transforms.Compose(aug + base)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img  = Image.open(self.image_paths[idx]).convert('RGB')
        img  = self.transform(img)
        label = self.targets[idx]
        return img, label


# ─────────────────────────────────────────────────────────────────────────────
# 2.  CNN Architecture
# ─────────────────────────────────────────────────────────────────────────────

class GrainCNN(nn.Module):
    """
    Lightweight CNN architecture for Perlmutter GPU memory constraints.

    Architecture:
        3 convolutional blocks (Conv → BN → ReLU → MaxPool)
        Adaptive average pool to 2x2
        2-layer FC head → scalar grain size output

    Input:  (B, 1, img_size, img_size)
    Output: (B,)  — predicted grain size in µm

    Memory estimate at img_size=64, batch=16: ~400 MB GPU RAM
    """
    def __init__(self, img_size=64):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),           # 64 → 32

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),           # 32 → 16

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),           # 16 → 8
        )

        self.pool = nn.AdaptiveAvgPool2d((2, 2))  # → 128 * 2 * 2 = 512

        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 2 * 2, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = self.regressor(x)
        return x.squeeze(1)   # (B,)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Training & Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, device, accum_steps=2):
    """Gradient accumulation: effective batch = batch_size * accum_steps."""
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()
    for step, (imgs, targets) in enumerate(loader):
        imgs, targets = imgs.to(device), targets.to(device)
        preds = model(imgs)
        loss  = criterion(preds, targets) / accum_steps
        loss.backward()
        if (step + 1) % accum_steps == 0 or (step + 1) == len(loader):
            optimizer.step()
            optimizer.zero_grad()
        total_loss += loss.item() * accum_steps * len(imgs)
    return total_loss / len(loader.dataset)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []
    with torch.no_grad():
        for imgs, targets in loader:
            imgs, targets = imgs.to(device), targets.to(device)
            preds = model(imgs)
            total_loss += criterion(preds, targets).item() * len(imgs)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    avg_loss = total_loss / len(loader.dataset)
    return avg_loss, np.array(all_preds), np.array(all_targets)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_training_curves(train_losses, val_losses, outdir):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(train_losses, label='Train Loss (MSE)', color='#4C72B0', lw=2)
    ax.plot(val_losses,   label='Val Loss (MSE)',   color='#C44E52', lw=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss')
    ax.set_title('CNN Training Curves')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(outdir, 'cnn_training_curves.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved -> {path}")


def plot_predictions(y_true, y_pred, split_name, outdir):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f'CNN Predictions — {split_name} set', fontsize=13, fontweight='bold')

    # Scatter: predicted vs actual
    ax = axes[0]
    ax.scatter(y_true, y_pred, alpha=0.7, color='#4C72B0', edgecolors='white', lw=0.5)
    lims = [min(y_true.min(), y_pred.min()) * 0.95,
            max(y_true.max(), y_pred.max()) * 1.05]
    ax.plot(lims, lims, 'r--', lw=1.5, label='Perfect prediction')
    ax.set_xlabel('Actual Grain Size [µm]')
    ax.set_ylabel('Predicted Grain Size [µm]')
    ax.set_title('Predicted vs Actual')
    ax.legend(fontsize=9)
    ax.set_xlim(lims); ax.set_ylim(lims)
    stats_txt = f"MAE  = {mae:.2f} µm\nRMSE = {rmse:.2f} µm\nR²   = {r2:.4f}"
    ax.text(0.05, 0.95, stats_txt, transform=ax.transAxes,
            fontsize=9, va='top', bbox=dict(boxstyle='round', fc='lightyellow', alpha=0.8))

    # Residuals
    ax = axes[1]
    residuals = y_pred - y_true
    ax.scatter(y_true, residuals, alpha=0.7, color='#55A868', edgecolors='white', lw=0.5)
    ax.axhline(0, color='red', ls='--', lw=1.5)
    ax.set_xlabel('Actual Grain Size [µm]')
    ax.set_ylabel('Residual (Predicted − Actual) [µm]')
    ax.set_title('Residual Plot')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(outdir, f'cnn_predictions_{split_name.lower()}.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved -> {path}")

    return {'MAE': mae, 'RMSE': rmse, 'R2': r2}


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Load data from CSV
# ─────────────────────────────────────────────────────────────────────────────

def load_data(csv_path, images_dir):
    image_paths, targets = [], []
    skipped = 0

    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            gs = row.get('grain_size_um', '').strip()
            fn = row.get('Filename', '').strip()

            if not gs or gs in ('', 'nan', 'ERROR', 'NaN'):
                skipped += 1
                continue
            try:
                gs_val = float(gs)
            except ValueError:
                skipped += 1
                continue

            img_path = os.path.join(images_dir, fn)
            if not os.path.isfile(img_path):
                skipped += 1
                continue

            image_paths.append(img_path)
            targets.append(gs_val)

    print(f"  Loaded  : {len(image_paths)} samples")
    if skipped:
        print(f"  Skipped : {skipped} rows (missing grain_size_um or image file)")

    return image_paths, np.array(targets, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='CNN grain size regression.')
    parser.add_argument('--csv',        required=True)
    parser.add_argument('--images_dir', required=True)
    parser.add_argument('--img_size',   type=int,   default=64, help='Resize to NxN (default: 64). Use 96/128 only if GPU has >16GB free.')
    parser.add_argument('--epochs',     type=int,   default=50)
    parser.add_argument('--batch_size', type=int,   default=8, help='Per-step batch size (default: 8). Keep small to avoid OOM.')
    parser.add_argument('--lr',         type=float, default=1e-3)
    parser.add_argument('--grad_accum', type=int,   default=2,
                        help='Gradient accumulation steps (default: 2)')
    parser.add_argument('--outdir',     default='cnn_output')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if device.type == 'cuda':
        torch.cuda.empty_cache()

    print(f"\n{'='*60}")
    print(f"  CNN Grain Size Regression")
    print(f"{'='*60}")
    print(f"  Device     : {device}")
    print(f"  Image size : {args.img_size}x{args.img_size}")
    print(f"  Epochs     : {args.epochs}")
    print(f"  Batch size : {args.batch_size}\n")

    # ── Load data ─────────────────────────────────────────────────────────────
    image_paths, targets = load_data(args.csv, args.images_dir)

    if len(image_paths) < 10:
        sys.exit("Error: need at least 10 samples. Run batch_grain_analysis.py first.")

    # ── 80/20 split ───────────────────────────────────────────────────────────
    idx = np.arange(len(image_paths))
    train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=42)

    train_paths   = [image_paths[i] for i in train_idx]
    test_paths    = [image_paths[i] for i in test_idx]
    train_targets = targets[train_idx]
    test_targets  = targets[test_idx]

    print(f"  Train samples : {len(train_paths)}")
    print(f"  Test samples  : {len(test_paths)}\n")

    # ── Datasets & loaders ────────────────────────────────────────────────────
    train_ds = GrainDataset(train_paths, train_targets, args.img_size, augment=True)
    test_ds  = GrainDataset(test_paths,  test_targets,  args.img_size, augment=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=0)

    # ── Model ─────────────────────────────────────────────────────────────────
    model     = GrainCNN(img_size=args.img_size).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model parameters : {total_params:,}\n")

    # ── Training loop ─────────────────────────────────────────────────────────
    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    best_model_path = os.path.join(args.outdir, 'best_cnn_model.pth')

    print(f"  {'Epoch':>6}  {'Train MSE':>10}  {'Val MSE':>10}  {'LR':>10}")
    print(f"  {'-'*45}")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, getattr(args, 'grad_accum', 2))
        val_loss, _, _ = evaluate(model, test_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        scheduler.step(val_loss)

        lr_now = optimizer.param_groups[0]['lr']
        if epoch % 5 == 0 or epoch == 1:
            print(f"  {epoch:>6}  {train_loss:>10.4f}  {val_loss:>10.4f}  {lr_now:>10.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)

    print(f"\n  Best val MSE : {best_val_loss:.4f}")
    print(f"  Model saved  -> {best_model_path}\n")

    joblib.dump(model, 'grain_model.pkl')
    # Save weights only
    torch.save(model.state_dict(), 'grain_model_weights.pth')

    # Save full model
    torch.save(model, 'grain_model_full.pth')
    # ── Evaluate best model ───────────────────────────────────────────────────
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    _, train_preds, train_true = evaluate(model, train_loader, criterion, device)
    _, test_preds,  test_true  = evaluate(model, test_loader,  criterion, device)

    plot_training_curves(train_losses, val_losses, args.outdir)
    train_metrics = plot_predictions(train_true, train_preds, 'Train', args.outdir)
    test_metrics  = plot_predictions(test_true,  test_preds,  'Test',  args.outdir)
    # Save weights only
    torch.save(model.state_dict(), 'grain_model_weights.pth')

    # Save full model
    torch.save(model, 'grain_model_full.pth')
    # ── Print final metrics ───────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  FINAL METRICS")
    print(f"{'='*60}")
    for split, m in [('Train', train_metrics), ('Test', test_metrics)]:
        print(f"  {split}")
        print(f"    MAE  = {m['MAE']:.4f} µm")
        print(f"    RMSE = {m['RMSE']:.4f} µm")
        print(f"    R²   = {m['R2']:.4f}")
    print(f"{'='*60}\n")

    # Save metrics to txt
    with open(os.path.join(args.outdir, 'cnn_metrics.txt'), 'w') as f:
        for split, m in [('Train', train_metrics), ('Test', test_metrics)]:
            f.write(f"{split}\n")
            f.write(f"  MAE  = {m['MAE']:.6f} um\n")
            f.write(f"  RMSE = {m['RMSE']:.6f} um\n")
            f.write(f"  R2   = {m['R2']:.6f}\n\n")


if __name__ == '__main__':
    main()