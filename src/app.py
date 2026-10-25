"""
ASPRON-V Subterranean Root MRI & AI Agronomist Platform
Smart India Hackathon 2026 - Problem Statement ID: SIH26210
Interactive Real-Time EIT Reconstruction Dashboard (Streamlit)
"""

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import streamlit as st
import torch

# Ensure src modules are resolvable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from src.unet_model import EITUNet
    from src.esp32_interface import ESP32EITInterface
except ImportError:
    from unet_model import EITUNet
    from esp32_interface import ESP32EITInterface


# ==============================================================================
# Page Configuration & Modern Agritech Theme
# ==============================================================================
st.set_page_config(
    page_title="ASPRON-V Subterranean Root MRI",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom High-End Styling
st.markdown(
    """
    <style>
    .main {
        background-color: #0B1120;
        color: #F8FAFC;
    }
    .stApp {
        background: radial-gradient(circle at 10% 20%, #0F172A 0%, #020617 90%);
    }
    div[data-testid="stMetric"] {
        background-color: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(51, 65, 85, 0.8);
        border-radius: 12px;
        padding: 12px 18px;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.25);
    }
    div[data-testid="stMetricLabel"] {
        color: #94A3B8 !important;
        font-size: 0.82rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }
    div[data-testid="stMetricValue"] {
        color: #38BDF8 !important;
        font-size: 1.6rem !important;
        font-weight: 700 !important;
    }
    .badge-live {
        background-color: rgba(16, 185, 129, 0.15);
        color: #10B981;
        border: 1px solid #10B981;
        border-radius: 20px;
        padding: 4px 12px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
    }
    .badge-sim {
        background-color: rgba(245, 158, 11, 0.15);
        color: #F59E0B;
        border: 1px solid #F59E0B;
        border-radius: 20px;
        padding: 4px 12px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==============================================================================
# Model Loading with Caching
# ==============================================================================
@st.cache_resource(show_spinner="Loading Subterranean U-Net Weights...")
def load_trained_model(model_path: str = "models/best_eit_unet.pth"):
    """Loads and compiles the PyTorch EITUNet model from disk."""
    if not os.path.exists(model_path):
        st.error(f"Model checkpoint '{model_path}' not found. Please train the model first.")
        return None, None

    try:
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        in_features = checkpoint.get("in_features", 16)
        base_channels = checkpoint.get("base_channels", 16)

        model = EITUNet(
            in_features=in_features,
            out_channels=1,
            base_channels=base_channels,
            grid_size=64,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        return model, checkpoint
    except Exception as e:
        st.error(f"Error loading model checkpoint: {e}")
        return None, None


# Custom Agri-Tech Colormaps
def get_custom_colormap(choice: str):
    if choice == "Bio-Luminescence (Neon Mint)":
        colors = ["#020617", "#064E3B", "#059669", "#10B981", "#6EE7B7", "#ECFDF5"]
        return LinearSegmentedColormap.from_list("bio_mint", colors)
    elif choice == "Agro-Thermal (Deep Amber)":
        colors = ["#020617", "#451A03", "#9A3412", "#D97706", "#F59E0B", "#FEF3C7"]
        return LinearSegmentedColormap.from_list("agro_amber", colors)
    elif choice == "Electric Violet":
        colors = ["#020617", "#2E1065", "#6D28D9", "#8B5CF6", "#C4B5FD", "#FAF5FF"]
        return LinearSegmentedColormap.from_list("electric_violet", colors)
    elif choice == "Inferno":
        return "inferno"
    elif choice == "Viridis":
        return "viridis"
    else:
        return "magma"


# ==============================================================================
# Main Application Dashboard
# ==============================================================================
def main():
    # Header Banner
    st.markdown(
        """
        <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #334155; padding-bottom: 12px; margin-bottom: 20px;">
            <div>
                <h1 style="margin: 0; font-size: 2.1rem; color: #F8FAFC; font-weight: 800;">
                    🌱 ASPRON-V <span style="font-weight: 300; color: #38BDF8;">Subterranean Root MRI</span>
                </h1>
                <p style="margin: 4px 0 0 0; color: #94A3B8; font-size: 0.95rem;">
                    Non-Invasive 4-Electrode Electrical Impedance Tomography (EIT) Array | SIH 2026 (SIH26210)
                </p>
            </div>
            <div>
                <span class="badge-live">● AI Engine Active</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------------------------
    # Sidebar: Hardware Controls & Visualization Parameters
    # --------------------------------------------------------------------------
    with st.sidebar:
        st.header("⚙️ Hardware Interface")
        
        # COM Ports
        detected_ports = ESP32EITInterface.list_available_ports()
        mode_options = ["Synthetic pyEIT Simulation"] + [f"ESP32 on {p}" for p in detected_ports]
        selected_mode = st.selectbox("Signal Input Source:", mode_options, index=0)

        force_sim = "Simulation" in selected_mode
        port_name = None if force_sim else selected_mode.split("on ")[-1]

        # Scan Mode
        st.subheader("📡 Scan Protocol")
        live_stream = st.toggle("Continuous Live Streaming", value=True)
        refresh_delay = st.slider("Scan Refresh Period (sec):", 0.2, 2.0, 0.5, 0.1)

        # Baseline Calibration
        if st.button("🔄 Calibrate Soil Baseline V0", use_container_width=True):
            st.session_state["calibrate_trigger"] = True

        st.subheader("🎨 Colormap & Display")
        cmap_choice = st.selectbox(
            "Reconstruction Palette:",
            [
                "Bio-Luminescence (Neon Mint)",
                "Agro-Thermal (Deep Amber)",
                "Electric Violet",
                "Inferno",
                "Viridis",
            ],
            index=0,
        )
        threshold = st.slider("Root Detection Sensitivity (Threshold):", 0.1, 0.9, 0.45, 0.05)
        show_contours = st.checkbox("Overlay Root Boundary Contours", value=True)

        st.markdown("---")
        st.markdown(
            r"""
            **Hardware Specs:**
            - Electrodes: 4 Stainless Steel Pegs ($90^\circ$ ring)
            - Stimulation: Adjacent Protocol ($10\text{ kHz}$, $2\text{ mA}$)
            - Microcontroller: ESP32-S3 (INA128 + CD74HC4067)
            - Spatial Resolution: $64 \times 64$ Mesh Grid
            """
        )

    # Initialize Interface in Session State
    if "interface" not in st.session_state:
        st.session_state["interface"] = ESP32EITInterface(
            port=port_name,
            simulate=force_sim,
        )
    else:
        # Update simulation mode if user changed selection
        st.session_state["interface"].simulate = force_sim
        if not force_sim and st.session_state["interface"].port != port_name:
            st.session_state["interface"].port = port_name
            st.session_state["interface"].connect()

    interface: ESP32EITInterface = st.session_state["interface"]

    # Handle Baseline Calibration
    if st.session_state.get("calibrate_trigger", False):
        with st.spinner("Recording homogeneous soil baseline frame (V0)..."):
            interface.calibrate_baseline(num_frames=8)
            st.success("Baseline V0 calibrated! Differential mode enabled.")
            st.session_state["calibrate_trigger"] = False

    # Load Model
    model, checkpoint_info = load_trained_model("models/best_eit_unet.pth")
    if model is None:
        st.stop()

    # Dynamic Containers for Live Rendering
    metrics_placeholder = st.empty()
    display_placeholder = st.empty()

    # Streaming Loop or Single Snapshot
    run_loop = live_stream

    while True:
        # 1. Fetch live 16-channel normalized voltage frame
        v_live, status_meta = interface.get_live_frame(normalize=True)

        # 2. Run inference through PyTorch U-Net
        v_tensor = torch.from_numpy(v_live).unsqueeze(0)  # Shape: (1, 16)
        with torch.no_grad():
            prob_map = model(v_tensor, return_probs=True).squeeze().numpy()  # (64, 64)

        # Apply circular container mask (points outside bucket domain R=1.0 are 0)
        grid_axis = np.linspace(-1.0, 1.0, 64)
        xx, yy = np.meshgrid(grid_axis, grid_axis)
        domain_mask = (xx**2 + yy**2) <= 1.0
        prob_map_masked = prob_map * domain_mask.astype(np.float32)

        # 3. Calculate Agronomic RSA Metrics
        root_binary = (prob_map_masked >= threshold).astype(np.float32)
        total_bucket_pixels = np.sum(domain_mask)
        root_pixel_count = np.sum(root_binary)
        biomass_density = (root_pixel_count / max(1, total_bucket_pixels)) * 100.0

        # Centroid calculation
        if root_pixel_count > 0:
            cx = float(np.sum(xx * root_binary) / root_pixel_count)
            cy = float(np.sum(yy * root_binary) / root_pixel_count)
        else:
            cx, cy = 0.0, 0.0

        peak_contrast = float(np.max(prob_map_masked))

        # Agronomic Diagnostic logic
        if biomass_density < 0.5:
            diag_label = "Uniform Soil (No Root Mass)"
            diag_color = "#94A3B8"
            diag_msg = "Baseline scan indicates uniform soil domain. Ready for taproot insertion."
        elif biomass_density > 18.0:
            diag_label = "Excess Moisture / Waterlogging"
            diag_color = "#EF4444"
            diag_msg = "Anomalously large conductive zone detected. Possible drainage failure or early root rot."
        else:
            diag_label = "Healthy Root Structure Detected"
            diag_color = "#10B981"
            diag_msg = f"Root cluster identified at ({cx:+.2f}, {cy:+.2f}). Normal cell sap conductivity."

        # 4. Render Metrics Cards
        with metrics_placeholder.container():
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            with col_m1:
                src_badge = "Hardware COM" if not interface.simulate else "pyEIT Synthetic"
                st.metric("Source Mode", src_badge, delta="Streaming" if live_stream else "Paused")
            with col_m2:
                st.metric("Root Biomass Density", f"{biomass_density:.1f} %", delta=f"{root_pixel_count:.0f} px")
            with col_m3:
                st.metric("Root Centroid", f"({cx:+.2f}, {cy:+.2f})", delta=f"Peak: {peak_contrast:.2f}")
            with col_m4:
                st.metric("U-Net Validation Dice", f"{checkpoint_info.get('val_dice', 0.67):.3f}", delta=f"Epoch {checkpoint_info.get('epoch', 3)}")

        # 5. Render Dynamic Visual Panels
        with display_placeholder.container():
            col_left, col_right = st.columns([1.1, 1.2])

            # Left Panel: 16-Channel Voltage Differential Graph
            with col_left:
                fig_v, ax_v = plt.subplots(figsize=(6.5, 5.2), dpi=130)
                fig_v.patch.set_facecolor("#0F172A")
                ax_v.set_facecolor("#1E293B")

                channels = np.arange(1, 17)
                colors = ["#38BDF8" if val >= 0 else "#F43F5E" for val in v_live]
                bars = ax_v.bar(channels, v_live, color=colors, alpha=0.85, width=0.65, edgecolor="#0284C7")
                ax_v.plot(channels, v_live, color="#F59E0B", marker="o", linewidth=1.8, markersize=5)
                ax_v.axhline(0, color="#64748B", linestyle="--", linewidth=1.0)

                ax_v.set_title("16-Channel Differential Voltage Array ($\\Delta V$)", color="#F8FAFC", fontsize=11, fontweight="bold", pad=8)
                ax_v.set_xlabel("Electrode Pair Channel Index", color="#94A3B8", fontsize=9)
                ax_v.set_ylabel("Normalized Voltage Drop", color="#94A3B8", fontsize=9)
                ax_v.set_xticks(channels)
                ax_v.tick_params(colors="#94A3B8", labelsize=8)
                ax_v.grid(True, linestyle="--", alpha=0.2, color="#94A3B8")
                ax_v.set_ylim(-0.35, 0.45)
                fig_v.tight_layout()
                st.pyplot(fig_v)
                plt.close(fig_v)

            # Right Panel: 2D Spatial Subterranean Root Cross-Section
            with col_right:
                fig_m, ax_m = plt.subplots(figsize=(6.5, 5.2), dpi=130)
                fig_m.patch.set_facecolor("#0F172A")
                ax_m.set_facecolor("#1E293B")

                cmap = get_custom_colormap(cmap_choice)
                im = ax_m.imshow(
                    prob_map_masked,
                    origin="lower",
                    extent=[-1, 1, -1, 1],
                    cmap=cmap,
                    vmin=0.0,
                    vmax=1.0,
                )

                # Boundary circle
                theta = np.linspace(0, 2 * np.pi, 200)
                ax_m.plot(np.cos(theta), np.sin(theta), color="#94A3B8", linestyle="--", linewidth=1.5)

                # Overlay 4 Electrodes
                el_coords = [(-1.0, 0.0), (0.0, 1.0), (1.0, 0.0), (0.0, -1.0)]
                for i, (ex, ey) in enumerate(el_coords):
                    ax_m.scatter(ex, ey, s=150, color="#EF4444", edgecolors="white", linewidths=1.5, zorder=5)
                    ax_m.text(ex * 1.15, ey * 1.15, f"E{i+1}", color="#F8FAFC", fontsize=10, fontweight="bold", ha="center", va="center")

                # Optional Contour Overlay
                if show_contours and np.max(prob_map_masked) >= threshold:
                    ax_m.contour(
                        grid_axis,
                        grid_axis,
                        prob_map_masked,
                        levels=[threshold],
                        colors=["#10B981"],
                        linewidths=[2.0],
                    )

                ax_m.set_title("2D Subterranean Root Cross-Section", color="#F8FAFC", fontsize=11, fontweight="bold", pad=8)
                ax_m.set_xlim(-1.3, 1.3)
                ax_m.set_ylim(-1.3, 1.3)
                ax_m.set_aspect("equal")
                ax_m.axis("off")

                cbar = fig_m.colorbar(im, ax=ax_m, fraction=0.046, pad=0.04)
                cbar.set_label("Root Mass Probability Density", color="#F8FAFC", fontsize=9)
                cbar.ax.tick_params(colors="#94A3B8", labelsize=8)

                fig_m.tight_layout()
                st.pyplot(fig_m)
                plt.close(fig_m)

            # Diagnostic Alert Banner
            st.markdown(
                f"""
                <div style="background-color: #1E293B; border-left: 4px solid {diag_color}; border-radius: 8px; padding: 12px 18px; margin-top: 10px;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="color: {diag_color}; font-weight: 700; font-size: 1.05rem;">{diag_label}</span>
                        <span style="color: #94A3B8; font-size: 0.85rem;">ASPRON-V Agro-Diagnostic AI</span>
                    </div>
                    <p style="margin: 6px 0 0 0; color: #E2E8F0; font-size: 0.92rem;">{diag_msg}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        if not run_loop:
            break

        time.sleep(refresh_delay)


if __name__ == "__main__":
    main()
