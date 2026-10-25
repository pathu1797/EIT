"""
ASPRON-V Agritech Platform - SIH 2026
Synthetic EIT Data Generation Script for Subterranean Root Reconstruction
Array Setup: 4-Electrode Circular Array with Adjacent Drive Protocol
Physics Engine: pyEIT (2D Finite Element Method)
"""

import os
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
import torch

import pyeit.mesh as mesh
from pyeit.mesh.wrapper import PyEITAnomaly_Circle
import pyeit.eit.fem as fem
import pyeit.eit.protocol as protocol


# ==============================================================================
# Physical Conductivity Profiles
# References: Basak & Wahid (2022) Plants 11(13):1713,
#             Soil Physics & Agro-Geophysical EIT Surveys
# ==============================================================================
SOIL_CONFIGS = {
    "black_cotton": {
        "name": "Black Cotton Soil (Regur)",
        "description": "High montmorillonite clay content, high CEC, high moisture retention",
        "sigma_range": (0.85, 1.45),  # Normalized conductivity (corresponds to ~0.12 - 0.28 S/m)
    },
    "red_loam": {
        "name": "Red Loam Soil",
        "description": "Porous, low clay/high silt-sand, lower moisture holding capacity",
        "sigma_range": (0.25, 0.60),  # Normalized conductivity (corresponds to ~0.02 - 0.07 S/m)
    },
}

ROOT_CONFIG = {
    "name": "Moist Root Biomass (Taproot / Tuber)",
    "description": "Conductive living tissue containing cellular electrolyte fluids",
    "sigma_range": (2.2, 3.8),     # High relative conductivity contrast (ratio ~ 2.5 - 6x soil)
    "radius_range": (0.08, 0.26),  # Dimension relative to unit circle radius (R=1.0)
}


def build_fem_mesh(n_el: int = 4, h0: float = 0.08):
    """
    Constructs a 2D circular FEM mesh with exactly n_el boundary electrodes.
    """
    mesh_obj = mesh.create(n_el=n_el, h0=h0)
    return mesh_obj


def generate_grid_mask(anomalies, grid_size: int = 64):
    """
    Renders 2D spatial ground-truth mask Y of size (grid_size, grid_size)
    over the domain [-1, 1] x [-1, 1].
    Values: 1.0 inside root anomaly, 0.0 elsewhere within the circular container.
    """
    x = np.linspace(-1.0, 1.0, grid_size)
    y = np.linspace(-1.0, 1.0, grid_size)
    xx, yy = np.meshgrid(x, y)
    
    # Unit circle domain mask
    domain_mask = (xx**2 + yy**2) <= 1.0
    root_mask = np.zeros((grid_size, grid_size), dtype=np.float32)

    for an in anomalies:
        cx, cy = an["center"]
        r = an["radius"]
        dist_sq = (xx - cx)**2 + (yy - cy)**2
        root_mask[dist_sq <= r**2] = 1.0

    # Ensure mask outside bucket domain is zero
    root_mask = root_mask * domain_mask.astype(np.float32)
    return root_mask


def generate_dataset(
    num_samples: int = 10000,
    n_el: int = 4,
    grid_size: int = 64,
    output_dir: str = "data",
    parser_meas: str = "meas_current",
    seed: int = 42,
    save_plot: bool = True,
    plot_path: str = "test_plot.png",
):
    """
    Simulates paired EIT boundary voltage arrays (X) and 2D spatial root masks (Y).
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("ASPRON-V: SUBTERRANEAN ROOT EIT SYNTHETIC DATA GENERATOR")
    print("Smart India Hackathon 2026 | Subterranean Root MRI Array")
    print("=" * 70)
    print(f"[*] Number of samples to generate : {num_samples:,}")
    print(f"[*] Electrode configuration       : {n_el} Boundary Electrodes (Adjacent Protocol)")
    print(f"[*] Measurement protocol parser   : {parser_meas}")
    print(f"[*] 2D Spatial Reconstruction Res : {grid_size}x{grid_size}")
    print(f"[*] Output directory              : {output_dir}")
    print("-" * 70)

    # 1. Build Base Mesh and Protocol
    print("[1/4] Constructing circular FEM mesh with 4 boundary electrodes...")
    base_mesh = build_fem_mesh(n_el=n_el, h0=0.08)
    protocol_obj = protocol.create(
        n_el=n_el, dist_exc=1, step_meas=1, parser_meas=parser_meas
    )
    fwd_engine = fem.EITForward(base_mesh, protocol_obj)

    # Determine voltage measurement dimension
    dummy_v = fwd_engine.solve_eit(base_mesh.perm)
    v_dim = dummy_v.shape[0]
    print(f"      FEM Elements: {base_mesh.element.shape[0]}, Nodes: {base_mesh.node.shape[0]}")
    print(f"      Voltage measurement vector dimension: {v_dim} features")

    # Pre-allocate storage
    X_diff = np.zeros((num_samples, v_dim), dtype=np.float32)
    X_raw = np.zeros((num_samples, v_dim), dtype=np.float32)
    X_baseline = np.zeros((num_samples, v_dim), dtype=np.float32)
    Y_masks = np.zeros((num_samples, 1, grid_size, grid_size), dtype=np.float32)

    soil_types = ["black_cotton", "red loam"]
    sample_metadata = []

    print(f"[2/4] Generating {num_samples} simulated root anomalies across soil profiles...")
    start_time = time.time()
    last_print = start_time

    first_sample_data = None

    for i in range(num_samples):
        # Alternate or randomly sample soil profile
        soil_key = "black_cotton" if (i % 2 == 0) else "red_loam"
        soil_info = SOIL_CONFIGS[soil_key]
        sigma_bg = np.random.uniform(*soil_info["sigma_range"])

        # Solve homogeneous baseline frame (V0)
        # Baseline mesh with uniform soil conductivity
        mesh_bg = mesh.set_perm(base_mesh, anomaly=[], background=sigma_bg)
        v0 = fwd_engine.solve_eit(mesh_bg.perm)

        # Generate randomized root anomalies (1 or 2 root segments)
        num_roots = np.random.choice([1, 2], p=[0.75, 0.25])
        anomalies_list = []
        pyeit_anomalies = []

        for _ in range(num_roots):
            r_root = np.random.uniform(*ROOT_CONFIG["radius_range"])
            # Keep anomaly comfortably within circular boundary (margin of 0.15 from electrodes)
            max_dist = max(0.1, 0.85 - r_root)
            angle = np.random.uniform(0, 2 * np.pi)
            dist = np.random.uniform(0.05, max_dist)
            cx = dist * np.cos(angle)
            cy = dist * np.sin(angle)
            sigma_root = np.random.uniform(*ROOT_CONFIG["sigma_range"])

            pyeit_an = PyEITAnomaly_Circle(center=[cx, cy], r=r_root, perm=sigma_root)
            pyeit_anomalies.append(pyeit_an)
            anomalies_list.append({"center": (cx, cy), "radius": r_root, "sigma": sigma_root})

        # Solve inhomogeneous anomaly frame (V1)
        mesh_anomaly = mesh.set_perm(base_mesh, anomaly=pyeit_anomalies, background=sigma_bg)
        v1 = fwd_engine.solve_eit(mesh_anomaly.perm)

        # Time-Difference EIT signal: Delta V = V1 - V0
        v_diff = v1 - v0

        # Generate continuous 2D spatial ground-truth mask
        mask_2d = generate_grid_mask(anomalies_list, grid_size=grid_size)

        # Store arrays
        X_diff[i] = v_diff.astype(np.float32)
        X_raw[i] = v1.astype(np.float32)
        X_baseline[i] = v0.astype(np.float32)
        Y_masks[i, 0] = mask_2d

        meta = {
            "index": i,
            "soil_type": soil_key,
            "soil_sigma": float(sigma_bg),
            "num_roots": num_roots,
            "anomalies": anomalies_list,
        }
        sample_metadata.append(meta)

        if i == 0:
            first_sample_data = {
                "mesh_anomaly": mesh_anomaly,
                "v0": v0,
                "v1": v1,
                "v_diff": v_diff,
                "mask_2d": mask_2d,
                "soil_info": soil_info,
                "soil_sigma": sigma_bg,
                "anomalies": anomalies_list,
            }

        # Progress reporting
        now = time.time()
        if (i + 1) % max(1, num_samples // 10) == 0 or (now - last_print > 3.0) or (i + 1 == num_samples):
            elapsed = now - start_time
            rate = (i + 1) / max(0.001, elapsed)
            percent = (i + 1) / num_samples * 100.0
            print(f"      Progress: {i+1:>6}/{num_samples} ({percent:5.1f}%) | Speed: {rate:6.1f} samples/s | Elapsed: {elapsed:5.1f}s")
            last_print = now

    total_time = time.time() - start_time
    print(f"      Completed {num_samples} simulations in {total_time:.2f}s ({num_samples/total_time:.1f} samples/sec).")

    # 3. Convert to PyTorch Tensors and Save
    print(f"[3/4] Exporting PyTorch tensor datasets to '{output_dir}/'...")
    X_diff_tensor = torch.from_numpy(X_diff)
    X_raw_tensor = torch.from_numpy(X_raw)
    X_baseline_tensor = torch.from_numpy(X_baseline)
    Y_masks_tensor = torch.from_numpy(Y_masks)

    diff_path = os.path.join(output_dir, "eit_diff_voltages_X.pt")
    raw_path = os.path.join(output_dir, "eit_raw_voltages_X.pt")
    baseline_path = os.path.join(output_dir, "eit_baseline_voltages_X.pt")
    masks_path = os.path.join(output_dir, "eit_root_masks_Y.pt")
    dataset_bundle_path = os.path.join(output_dir, "eit_dataset.pt")

    torch.save(X_diff_tensor, diff_path)
    torch.save(X_raw_tensor, raw_path)
    torch.save(X_baseline_tensor, baseline_path)
    torch.save(Y_masks_tensor, masks_path)

    # Save consolidated bundle for easy PyTorch Dataset loading
    torch.save(
        {
            "X_diff": X_diff_tensor,
            "X_raw": X_raw_tensor,
            "X_baseline": X_baseline_tensor,
            "Y": Y_masks_tensor,
            "metadata": sample_metadata,
            "n_el": n_el,
            "grid_size": grid_size,
            "parser_meas": parser_meas,
        },
        dataset_bundle_path,
    )

    print(f"      [+] Difference Voltages X (tdEIT) : {diff_path} {tuple(X_diff_tensor.shape)}")
    print(f"      [+] Raw Voltages X                : {raw_path} {tuple(X_raw_tensor.shape)}")
    print(f"      [+] 2D Spatial Root Masks Y       : {masks_path} {tuple(Y_masks_tensor.shape)}")
    print(f"      [+] Unified Dataset Bundle        : {dataset_bundle_path}")

    # 4. Generate Matplotlib Review Plot
    if save_plot and first_sample_data is not None:
        print(f"[4/4] Rendering validation visualization -> '{plot_path}'...")
        render_validation_plot(first_sample_data, base_mesh, plot_path)
        print(f"      [+] Visualization successfully saved to '{plot_path}'")

    print("=" * 70)
    print("SUCCESS: Data generation pipeline verified and ready for model training!")
    print("=" * 70)
    return dataset_bundle_path


def render_validation_plot(sample, base_mesh, save_path: str = "test_plot.png"):
    """
    Renders an engineering inspection panel showing:
    1. 2D Circular FEM Mesh with 4 Electrodes & Root Anomaly Permittivity
    2. Reconstructed Ground-Truth 2D Spatial Root Mask (64x64)
    3. 1D Differential Voltage Profile (Delta V) across 4 Electrodes
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=150)
    fig.patch.set_facecolor("#0F172A")

    # Panel 1: FEM Mesh & Conductivity Field
    ax1 = axes[0]
    ax1.set_facecolor("#1E293B")
    pts = sample["mesh_anomaly"].node
    tri = sample["mesh_anomaly"].element
    perm = sample["mesh_anomaly"].perm

    # Triplot with conductivity coloring
    tpc = ax1.tripcolor(
        pts[:, 0],
        pts[:, 1],
        tri,
        facecolors=perm,
        cmap="viridis",
        edgecolors="none",
        alpha=0.95,
    )
    # Outline circular boundary
    theta = np.linspace(0, 2 * np.pi, 200)
    ax1.plot(np.cos(theta), np.sin(theta), color="#94A3B8", linestyle="--", linewidth=1.5)

    # Plot the 4 electrodes
    el_pos = base_mesh.el_pos
    el_coords = pts[el_pos]
    for idx, (ex, ey) in enumerate(el_coords[:, :2]):
        ax1.scatter(ex, ey, s=180, color="#EF4444", edgecolors="white", linewidths=1.8, zorder=5)
        ax1.text(
            ex * 1.15,
            ey * 1.15,
            f"E{idx+1}",
            color="#F8FAFC",
            fontsize=12,
            fontweight="bold",
            ha="center",
            va="center",
            zorder=6,
        )

    ax1.set_title(
        f"2D Circular FEM Mesh (4 Electrodes)\nSoil: {sample['soil_info']['name']} ($\\sigma$={sample['soil_sigma']:.2f})",
        color="#F8FAFC",
        fontsize=12,
        pad=10,
        fontweight="bold",
    )
    ax1.set_xlim(-1.35, 1.35)
    ax1.set_ylim(-1.35, 1.35)
    ax1.set_aspect("equal")
    ax1.tick_params(colors="#94A3B8")
    cbar1 = plt.colorbar(tpc, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.set_label("Conductivity $\\sigma$ (S/m relative)", color="#F8FAFC")
    cbar1.ax.tick_params(colors="#94A3B8")

    # Panel 2: Ground-Truth 2D Spatial Mask Y (64x64)
    ax2 = axes[1]
    ax2.set_facecolor("#1E293B")
    mask = sample["mask_2d"]
    im2 = ax2.imshow(
        mask,
        origin="lower",
        extent=[-1, 1, -1, 1],
        cmap="inferno",
        vmin=0.0,
        vmax=1.0,
    )
    ax2.plot(np.cos(theta), np.sin(theta), color="#64748B", linestyle=":", linewidth=1.2)
    ax2.set_title(
        "Ground-Truth 2D Spatial Mask $Y$ (64x64)\n(1.0 = Root Biomass, 0.0 = Background Soil)",
        color="#F8FAFC",
        fontsize=12,
        pad=10,
        fontweight="bold",
    )
    ax2.set_xlim(-1.15, 1.15)
    ax2.set_ylim(-1.15, 1.15)
    ax2.set_aspect("equal")
    ax2.tick_params(colors="#94A3B8")
    cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cbar2.set_label("Root Probability / Binary Mask", color="#F8FAFC")
    cbar2.ax.tick_params(colors="#94A3B8")

    # Panel 3: 1D Differential Voltage Profile Delta V
    ax3 = axes[2]
    ax3.set_facecolor("#1E293B")
    v_diff = sample["v_diff"]
    channels = np.arange(1, len(v_diff) + 1)
    
    # Plot bars and spline line
    ax3.bar(
        channels,
        v_diff,
        color="#38BDF8",
        edgecolor="#0284C7",
        alpha=0.7,
        label="$\\Delta V = V_{anomaly} - V_{baseline}$",
    )
    ax3.plot(channels, v_diff, color="#F59E0B", marker="o", linewidth=2.0)
    ax3.axhline(0, color="#64748B", linestyle="--", linewidth=1.0)
    ax3.set_title(
        f"1D Voltage Differential Signature $X$ ({len(v_diff)} Channels)\nAdjacent Stimulation Pattern",
        color="#F8FAFC",
        fontsize=12,
        pad=10,
        fontweight="bold",
    )
    ax3.set_xlabel("Boundary Measurement Channel Index", color="#F8FAFC", labelpad=8)
    ax3.set_ylabel("$\\Delta V$ (Volts)", color="#F8FAFC", labelpad=8)
    ax3.tick_params(colors="#94A3B8")
    ax3.grid(True, linestyle="--", alpha=0.25, color="#94A3B8")
    ax3.legend(facecolor="#1E293B", edgecolor="#334155", labelcolor="#F8FAFC", loc="upper right")

    plt.tight_layout()
    plt.savefig(save_path, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASPRON-V EIT Synthetic Data Generator")
    parser.add_argument("--num_samples", type=int, default=10000, help="Total synthetic iterations")
    parser.add_argument("--n_el", type=int, default=4, help="Number of electrodes (default: 4)")
    parser.add_argument("--grid_size", type=int, default=64, help="2D mask dimension (default: 64)")
    parser.add_argument("--output_dir", type=str, default="data", help="Output directory")
    parser.add_argument("--parser_meas", type=str, default="meas_current", choices=["meas_current", "std"], help="Measurement protocol parser")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--plot_path", type=str, default="test_plot.png", help="Path to save verification plot")
    parser.add_argument("--no_plot", action="store_true", help="Disable plotting")

    args = parser.parse_args()
    generate_dataset(
        num_samples=args.num_samples,
        n_el=args.n_el,
        grid_size=args.grid_size,
        output_dir=args.output_dir,
        parser_meas=args.parser_meas,
        seed=args.seed,
        save_plot=not args.no_plot,
        plot_path=args.plot_path,
    )
