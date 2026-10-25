"""
ASPRON-V EIT Dataset Loader & PyTorch Dataset Interface
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split


class EITRootDataset(Dataset):
    """
    PyTorch Dataset for paired EIT Boundary Voltages and 2D Spatial Root Reconstruction Masks.
    """

    def __init__(
        self,
        data_path: str = "data/eit_dataset.pt",
        voltage_mode: str = "diff",  # 'diff' (tdEIT Delta V), 'raw' (V1), or 'both'
        transform=None,
    ):
        super().__init__()
        if not os.path.exists(data_path):
            raise FileNotFoundError(
                f"Dataset bundle '{data_path}' not found. Please run generate_synthetic_data.py first."
            )

        data_bundle = torch.load(data_path, map_location="cpu", weights_only=False)
        self.voltage_mode = voltage_mode

        if voltage_mode == "diff":
            self.X = data_bundle["X_diff"]
        elif voltage_mode == "raw":
            self.X = data_bundle["X_raw"]
        elif voltage_mode == "both":
            self.X = torch.cat([data_bundle["X_diff"], data_bundle["X_raw"]], dim=1)
        else:
            raise ValueError(f"Unknown voltage_mode '{voltage_mode}'. Use 'diff', 'raw', or 'both'.")

        self.Y = data_bundle["Y"]
        self.metadata = data_bundle.get("metadata", [])
        self.transform = transform

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        x = self.X[idx]
        y = self.Y[idx]

        if self.transform:
            x, y = self.transform(x, y)

        return x, y


def get_dataloaders(
    data_path: str = "data/eit_dataset.pt",
    batch_size: int = 64,
    val_split: float = 0.20,
    seed: int = 42,
    num_workers: int = 0,
):
    """
    Returns train_loader and val_loader for model training.
    """
    full_dataset = EITRootDataset(data_path=data_path, voltage_mode="diff")
    total_len = len(full_dataset)
    val_len = int(total_len * val_split)
    train_len = total_len - val_len

    generator = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(full_dataset, [train_len, val_len], generator=generator)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader
