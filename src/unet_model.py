"""
ASPRON-V Subterranean Root Reconstruction
Deep Learning Architecture: 1D-to-2D EIT U-Net
SIH 2026 - Problem Statement: Student Innovation / Hardware (AgriTech)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """
    Two consecutive Convolution -> BatchNorm -> GELU blocks with optional residual link.
    """

    def __init__(self, in_channels: int, out_channels: int, mid_channels: int = None):
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.act1 = nn.GELU()

        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act2 = nn.GELU()

        # Residual shortcut if channel dimension changes
        self.shortcut = (
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.shortcut(x)
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.act2(out + res)
        return out


class DownBlock(nn.Module):
    """Downscaling with MaxPool2d then DoubleConv."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class UpBlock(nn.Module):
    """Upscaling with Bilinear Interpolation / ConvTranspose2d followed by DoubleConv."""

    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor = None) -> torch.Tensor:
        x1 = self.up(x1)
        if x2 is not None:
            # Handle possible padding discrepancies
            diff_y = x2.size()[2] - x1.size()[2]
            diff_x = x2.size()[3] - x1.size()[3]
            if diff_y != 0 or diff_x != 0:
                x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
            x = torch.cat([x2, x1], dim=1)
        else:
            x = x1
        return self.conv(x)


class EIT1DTo2DProjection(nn.Module):
    """
    MLP Projection Head:
    Transforms sparse 1D boundary voltage vector (e.g. 16 or 4 channels)
    into a structured 2D spatial feature map (e.g. 64 channels @ 8x8 or 16x16).
    """

    def __init__(
        self,
        in_features: int = 16,
        init_channels: int = 64,
        init_spatial_size: int = 8,
        hidden_dim: int = 512,
    ):
        super().__init__()
        self.init_channels = init_channels
        self.init_spatial_size = init_spatial_size
        self.target_dim = init_channels * init_spatial_size * init_spatial_size

        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(p=0.05),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(p=0.05),
            nn.Linear(hidden_dim * 2, self.target_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, in_features)
        b = x.shape[0]
        feat = self.mlp(x)  # (B, target_dim)
        feat_2d = feat.view(b, self.init_channels, self.init_spatial_size, self.init_spatial_size)
        return feat_2d


class EITUNet(nn.Module):
    """
    Custom 1D-to-2D EIT U-Net for Subterranean Root Reconstruction.

    Pipeline:
    1. Input: 1D voltage array [B, in_features] (e.g., differential voltage Delta V).
    2. Linear Projection: Dense projection -> Reshaped into latent 2D map [B, 64, 8, 8].
    3. Progressive Upsampler: [B, 64, 8, 8] -> [B, 64, 16, 16] -> [B, 32, 32, 32] -> [B, 32, 64, 64].
    4. Multi-Scale U-Net Refinement:
       Encoder-Decoder network with skip connections operating on the 64x64 spatial domain
       to sharply isolate root boundary geometries and eliminate soil background noise.
    5. Output Head: 1x1 Conv producing [B, out_channels, 64, 64] spatial mask.
    """

    def __init__(
        self,
        in_features: int = 16,
        out_channels: int = 1,
        base_channels: int = 32,
        grid_size: int = 64,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_channels = out_channels
        self.grid_size = grid_size

        # 1. 1D-to-2D Projection Head (projects 1D voltage to 8x8 feature map)
        self.projector = EIT1DTo2DProjection(
            in_features=in_features,
            init_channels=base_channels * 2,  # 64 channels
            init_spatial_size=8,
            hidden_dim=512,
        )

        # 2. Latent Upscaling Pathway (8x8 -> 16x16 -> 32x32 -> 64x64)
        self.up_stage1 = UpBlock(base_channels * 2, base_channels * 2)  # 8 -> 16
        self.up_stage2 = UpBlock(base_channels * 2, base_channels)       # 16 -> 32
        self.up_stage3 = UpBlock(base_channels, base_channels)           # 32 -> 64

        # 3. U-Net Spatial Refinement Encoder (64x64 -> 32x32 -> 16x16)
        self.enc1 = DoubleConv(base_channels, base_channels)             # 64x64
        self.down1 = DownBlock(base_channels, base_channels * 2)         # 32x32
        self.down2 = DownBlock(base_channels * 2, base_channels * 4)     # 16x16

        # Bottleneck (16x16)
        self.bottleneck = DoubleConv(base_channels * 4, base_channels * 4)

        # 4. U-Net Decoder with Skip Connections (16x16 -> 32x32 -> 64x64)
        self.dec2 = UpBlock(base_channels * 4 + base_channels * 2, base_channels * 2)  # 16 -> 32
        self.dec1 = UpBlock(base_channels * 2 + base_channels, base_channels)          # 32 -> 64

        # Final Convolutional Head
        self.out_conv = nn.Sequential(
            nn.Conv2d(base_channels, base_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels // 2),
            nn.GELU(),
            nn.Conv2d(base_channels // 2, out_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor, return_probs: bool = False) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (Batch_size, in_features)
            return_probs: If True, applies sigmoid to return [0, 1] probability map.
        Returns:
            Tensor of shape (Batch_size, out_channels, grid_size, grid_size)
        """
        # Step 1: Project 1D voltage array to 2D latent representation (B, 64, 8, 8)
        feat_8x8 = self.projector(x)

        # Step 2: Progressive upsampling to 64x64 initial spatial field
        feat_16x16 = self.up_stage1(feat_8x8)   # (B, 64, 16, 16)
        feat_32x32 = self.up_stage2(feat_16x16) # (B, 32, 32, 32)
        feat_64x64 = self.up_stage3(feat_32x32) # (B, 32, 64, 64)

        # Step 3: Encoder with skip connections
        e1 = self.enc1(feat_64x64)  # (B, 32, 64, 64)
        e2 = self.down1(e1)         # (B, 64, 32, 32)
        e3 = self.down2(e2)         # (B, 128, 16, 16)

        # Step 4: Bottleneck
        b = self.bottleneck(e3)     # (B, 128, 16, 16)

        # Step 5: Decoder with skip connections
        d2 = self.dec2(b, e2)       # (B, 64, 32, 32)
        d1 = self.dec1(d2, e1)      # (B, 32, 64, 64)

        # Step 6: Output logits
        logits = self.out_conv(d1)  # (B, 1, 64, 64)

        if return_probs:
            return torch.sigmoid(logits)
        return logits


class CombinedEITLoss(nn.Module):
    """
    Combined Binary Cross-Entropy (BCE) and Soft Dice Loss
    Optimized for sparse root segmentation under high class imbalance.
    """

    def __init__(self, bce_weight: float = 0.5, smooth: float = 1e-5):
        super().__init__()
        self.bce_weight = bce_weight
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(logits, targets)

        # Dice Loss
        probs = torch.sigmoid(logits)
        p_flat = probs.view(-1)
        t_flat = targets.view(-1)
        intersection = (p_flat * t_flat).sum()
        dice = (2.0 * intersection + self.smooth) / (p_flat.sum() + t_flat.sum() + self.smooth)
        dice_loss = 1.0 - dice

        return self.bce_weight * bce_loss + (1.0 - self.bce_weight) * dice_loss


if __name__ == "__main__":
    print("=" * 60)
    print("ASPRON-V EIT U-Net Architecture Verification Test")
    print("=" * 60)

    # Test with batch of 4 samples, 16 voltage channels
    batch_size = 4
    in_features = 16
    dummy_input = torch.randn(batch_size, in_features)

    model = EITUNet(in_features=in_features, out_channels=1, base_channels=32, grid_size=64)
    model.eval()

    with torch.no_grad():
        out_logits = model(dummy_input)
        out_probs = model(dummy_input, return_probs=True)

    print(f"[+] Input Voltage Shape   : {tuple(dummy_input.shape)}")
    print(f"[+] Output Logits Shape   : {tuple(out_logits.shape)}")
    print(f"[+] Output Probs Range    : [{out_probs.min():.4f}, {out_probs.max():.4f}]")

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[+] Trainable Parameters  : {num_params:,}")

    # Loss test
    criterion = CombinedEITLoss()
    dummy_targets = (torch.rand(batch_size, 1, 64, 64) > 0.85).float()
    loss = criterion(out_logits, dummy_targets)
    print(f"[+] Sample Combined Loss  : {loss.item():.4f}")
    print("=" * 60)
    print("SUCCESS: EIT U-Net forward pass & dimensions verified without errors!")
    print("=" * 60)
