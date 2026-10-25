"""
ASPRON-V Subterranean Root Reconstruction - SIH 2026
Training Pipeline for 1D-to-2D EIT U-Net
"""

import os
import argparse
import time
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from src.dataset import EITRootDataset, get_dataloaders
    from src.unet_model import EITUNet, CombinedEITLoss
except ImportError:
    from dataset import EITRootDataset, get_dataloaders
    from unet_model import EITUNet, CombinedEITLoss


def calculate_dice(probs: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5, eps: float = 1e-6) -> float:
    """Calculates standard Dice similarity coefficient for binary masks."""
    preds = (probs > threshold).float()
    intersection = (preds * targets).sum().item()
    total = preds.sum().item() + targets.sum().item()
    if total == 0:
        return 1.0
    return (2.0 * intersection + eps) / (total + eps)


def train_one_epoch(model, loader, optimizer, criterion, device):
    """Executes a single training epoch."""
    model.train()
    running_loss = 0.0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * x.size(0)

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss


def validate(model, loader, criterion, device):
    """Evaluates validation loss and mean Dice score."""
    model.eval()
    running_loss = 0.0
    total_dice = 0.0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = criterion(logits, y)
            probs = torch.sigmoid(logits)

            running_loss += loss.item() * x.size(0)
            total_dice += calculate_dice(probs, y) * x.size(0)

    val_loss = running_loss / len(loader.dataset)
    val_dice = total_dice / len(loader.dataset)
    return val_loss, val_dice


def evaluate_and_visualize(
    model: nn.Module,
    val_dataset: EITRootDataset,
    device: torch.device,
    save_path: str = "inference_results.png",
    num_samples: int = 3,
    seed: int = 42,
):
    """
    Runs inference on random validation samples and saves an inspection panel:
    Col 1: 1D Boundary Voltage Signature (Delta V, 16 channels)
    Col 2: Ground-Truth 2D Root Mask (64x64)
    Col 3: U-Net Reconstructed Probability Heatmap (64x64)
    """
    model.eval()
    random.seed(seed)
    indices = random.sample(range(len(val_dataset)), num_samples)

    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 4.2 * num_samples), dpi=150)
    fig.patch.set_facecolor("#0F172A")

    if num_samples == 1:
        axes = np.expand_dims(axes, 0)

    theta = np.linspace(0, 2 * np.pi, 200)
    circle_x = np.cos(theta)
    circle_y = np.sin(theta)

    for row, idx in enumerate(indices):
        x_vec, y_true = val_dataset[idx]
        x_tensor = x_vec.unsqueeze(0).to(device)

        with torch.no_grad():
            prob_map = model(x_tensor, return_probs=True).squeeze().cpu().numpy()

        y_true_np = y_true.squeeze().numpy()
        x_vec_np = x_vec.numpy()

        # Compute sample Dice score
        sample_dice = calculate_dice(
            torch.from_numpy(prob_map).unsqueeze(0).unsqueeze(0),
            y_true.unsqueeze(0),
            threshold=0.5,
        )

        # -------------------------------------------------------------
        # Column 1: 1D Voltage Differential Signature
        # -------------------------------------------------------------
        ax_volt = axes[row, 0]
        ax_volt.set_facecolor("#1E293B")
        channels = np.arange(1, len(x_vec_np) + 1)
        ax_volt.bar(
            channels,
            x_vec_np,
            color="#38BDF8",
            edgecolor="#0284C7",
            alpha=0.75,
            label="$\\Delta V$",
        )
        ax_volt.plot(channels, x_vec_np, color="#F59E0B", marker="o", linewidth=1.8, markersize=5)
        ax_volt.axhline(0, color="#64748B", linestyle="--", linewidth=1.0)
        ax_volt.set_title(
            f"Sample #{idx}: 1D Voltage Profile $\\Delta V$ (16 Channels)",
            color="#F8FAFC",
            fontsize=11,
            fontweight="bold",
        )
        ax_volt.set_xlabel("Electrode Channel Index", color="#94A3B8", fontsize=10)
        ax_volt.set_ylabel("Differential Voltage (V)", color="#94A3B8", fontsize=10)
        ax_volt.tick_params(colors="#94A3B8")
        ax_volt.grid(True, linestyle="--", alpha=0.25, color="#94A3B8")

        # -------------------------------------------------------------
        # Column 2: Ground Truth 2D Mask Y (64x64)
        # -------------------------------------------------------------
        ax_gt = axes[row, 1]
        ax_gt.set_facecolor("#1E293B")
        im_gt = ax_gt.imshow(
            y_true_np,
            origin="lower",
            extent=[-1, 1, -1, 1],
            cmap="inferno",
            vmin=0.0,
            vmax=1.0,
        )
        ax_gt.plot(circle_x, circle_y, color="#94A3B8", linestyle=":", linewidth=1.2)
        ax_gt.set_title(
            f"Ground Truth 2D Mask $Y$ (64x64)\n(1.0 = Root Biomass)",
            color="#F8FAFC",
            fontsize=11,
            fontweight="bold",
        )
        ax_gt.set_aspect("equal")
        ax_gt.tick_params(colors="#94A3B8")
        cbar_gt = plt.colorbar(im_gt, ax=ax_gt, fraction=0.046, pad=0.04)
        cbar_gt.ax.tick_params(colors="#94A3B8")

        # -------------------------------------------------------------
        # Column 3: U-Net Reconstructed Predicted Mask
        # -------------------------------------------------------------
        ax_pred = axes[row, 2]
        ax_pred.set_facecolor("#1E293B")
        im_pred = ax_pred.imshow(
            prob_map,
            origin="lower",
            extent=[-1, 1, -1, 1],
            cmap="inferno",
            vmin=0.0,
            vmax=1.0,
        )
        ax_pred.plot(circle_x, circle_y, color="#94A3B8", linestyle=":", linewidth=1.2)
        # Contour at 0.5 decision threshold
        ax_pred.contour(
            np.linspace(-1, 1, 64),
            np.linspace(-1, 1, 64),
            prob_map,
            levels=[0.5],
            colors=["#10B981"],
            linewidths=[1.8],
        )
        ax_pred.set_title(
            f"U-Net Reconstructed Mask $\\hat{{Y}}$\n(Dice Similarity: {sample_dice:.3f})",
            color="#F8FAFC",
            fontsize=11,
            fontweight="bold",
        )
        ax_pred.set_aspect("equal")
        ax_pred.tick_params(colors="#94A3B8")
        cbar_pred = plt.colorbar(im_pred, ax=ax_pred, fraction=0.046, pad=0.04)
        cbar_pred.ax.tick_params(colors="#94A3B8")

    plt.tight_layout()
    plt.savefig(save_path, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"[+] Inference evaluation visual successfully saved to '{save_path}'")


def run_training(
    data_path: str = "data/eit_dataset.pt",
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
    val_split: float = 0.20,
    base_channels: int = 16,
    save_dir: str = "models",
    plot_path: str = "inference_results.png",
    seed: int = 42,
):
    """Main training routine."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print("ASPRON-V: SUBTERRANEAN ROOT RECONSTRUCTION U-NET TRAINING PIPELINE")
    print("Smart India Hackathon 2026 | Subterranean Root MRI Array")
    print("=" * 70)
    print(f"[*] Compute Device          : {device}")
    print(f"[*] Dataset Location        : {data_path}")
    print(f"[*] Train / Val Split       : {int((1-val_split)*100)}% / {int(val_split*100)}%")
    print(f"[*] Epochs                  : {epochs}")
    print(f"[*] Batch Size              : {batch_size}")
    print(f"[*] Learning Rate           : {lr}")
    print(f"[*] Base Channels           : {base_channels}")
    print(f"[*] Model Save Directory    : {save_dir}/")
    print("-" * 70)

    # 1. Load Data
    print("[1/5] Loading paired EIT dataset & creating DataLoaders...")
    train_loader, val_loader = get_dataloaders(
        data_path=data_path,
        batch_size=batch_size,
        val_split=val_split,
        seed=seed,
    )
    print(f"      Train Samples: {len(train_loader.dataset):,} ({len(train_loader)} batches)")
    print(f"      Val Samples  : {len(val_loader.dataset):,} ({len(val_loader)} batches)")

    # 2. Initialize Model, Optimizer, Criterion
    print("[2/5] Initializing EITUNet Architecture & Combined Loss...")
    in_features = train_loader.dataset[0][0].shape[0]
    model = EITUNet(
        in_features=in_features,
        out_channels=1,
        base_channels=base_channels,
        grid_size=64,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"      Model Parameters: {num_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=4, min_lr=1e-6
    )
    criterion = CombinedEITLoss(bce_weight=0.5)

    # 3. Training Loop
    print(f"[3/5] Starting training loop ({epochs} epochs)...")
    best_val_loss = float("inf")
    best_val_dice = 0.0
    best_model_path = os.path.join(save_dir, "best_eit_unet.pth")

    start_training_time = time.time()

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_dice = validate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            best_val_dice = val_dice
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_dice": val_dice,
                    "in_features": in_features,
                    "base_channels": base_channels,
                },
                best_model_path,
            )
            marker = " [* SAVED BEST]"
        else:
            marker = ""

        print(
            f"Epoch [{epoch:>2}/{epochs}] ({elapsed:4.1f}s) | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Dice: {val_dice:.4f} | "
            f"LR: {current_lr:.1e}{marker}"
        )

    total_training_time = time.time() - start_training_time
    print("-" * 70)
    print(f"Training completed in {total_training_time/60:.2f} minutes.")
    print(f"Best Validation Loss : {best_val_loss:.4f}")
    print(f"Best Validation Dice : {best_val_dice:.4f}")
    print(f"Saved Best Model     : {best_model_path}")

    # 4. Load Best Model & Run Inference on 3 Random Validation Samples
    print("[4/5] Loading best checkpoint for validation inference...")
    checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    print(f"[5/5] Generating inference comparison figure -> '{plot_path}'...")
    evaluate_and_visualize(
        model=model,
        val_dataset=val_loader.dataset,
        device=device,
        save_path=plot_path,
        num_samples=3,
        seed=seed,
    )

    print("=" * 70)
    print("SUCCESS: Full ASPRON-V training & inference pipeline completed!")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASPRON-V EIT U-Net Training")
    parser.add_argument("--data_path", type=str, default="data/eit_dataset.pt", help="Path to eit_dataset.pt")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--val_split", type=float, default=0.20, help="Validation split (default: 0.20)")
    parser.add_argument("--base_channels", type=int, default=16, help="Base U-Net channels (default: 16)")
    parser.add_argument("--save_dir", type=str, default="models", help="Model checkpoint directory")
    parser.add_argument("--plot_path", type=str, default="inference_results.png", help="Path for inference comparison figure")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()
    run_training(
        data_path=args.data_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        val_split=args.val_split,
        base_channels=args.base_channels,
        save_dir=args.save_dir,
        plot_path=args.plot_path,
        seed=args.seed,
    )
