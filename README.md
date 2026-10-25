# ASPRON-V Subterranean Root MRI & AI Agronomist Platform
## Smart India Hackathon 2026 (Problem Statement ID: SIH26210)
### Subterranean Root Reconstruction using 4-Electrode Electrical Impedance Tomography (EIT)

---

## 1. System Overview & Physical Grounding
The **ASPRON-V** platform non-invasively reconstructs underground root system architecture (RSA) to detect root decay, Fusarium wilt, and waterlogging before visible canopy damage occurs.

### Soil & Root Conductivity Baselines
Derived from research publications (*Basak & Wahid, Plants 2022, 11(13), 1713*) and agro-geophysical field surveys:
* **Black Cotton Soil (Regur):** $\sigma \approx 0.12 - 0.28\text{ S/m}$ (normalized baseline $0.85 - 1.45$). High montmorillonite clay content with heavy water holding capacity.
* **Red Loam Soil:** $\sigma \approx 0.02 - 0.07\text{ S/m}$ (normalized baseline $0.25 - 0.60$). Porous, low clay/high silt-sand texture.
* **Root Biomass Anomaly (Taproot / Tuber):** $\sigma \approx 0.35 - 0.85\text{ S/m}$ (normalized anomaly $2.20 - 3.80$). High electrolyte content in cellular sap yields marked positive electrical conductivity contrast ($\sigma_{root} / \sigma_{soil} \approx 2.5\times - 6\times$).

---

## 2. Mathematical Modeling

### Forward Problem
The electrical potential field $\varphi(r)$ in domain $\Omega$ with conductivity distribution $\sigma(r)$ is governed by the generalized Laplace equation:
$$\nabla \cdot (\sigma(r) \nabla \varphi(r)) = 0 \quad \text{in } \Omega$$
with Complete Electrode Model (CEM) boundary conditions:
$$\int_{e_l} \sigma \frac{\partial \varphi}{\partial n} \, dS = I_l, \quad \varphi + z_l \sigma \frac{\partial \varphi}{\partial n} = V_l \quad \text{on } e_l$$

### Adjacent Stimulation Protocol
With $N_e = 4$ electrodes:
1. Low-frequency AC current ($10\text{ kHz}, 2\text{ mA}$) is sequentially injected across adjacent electrode pairs $(E_1, E_2), (E_2, E_3), (E_3, E_4), (E_4, E_1)$.
2. Boundary voltage drops are measured across adjacent pairs for each excitation, generating 16 differential voltage measurements per frame.

### Time-Difference EIT (tdEIT)
To reject contact impedance drift and hardware static, the network is trained on the difference signature:
$$\Delta V = V_{\text{anomaly}} - V_{\text{baseline}}$$

---

## 3. Directory Layout
```
EIT/
├── .venv/                      # Python 3.13 virtual environment with PyTorch & pyEIT
├── data/                       # Generated PyTorch tensor datasets
│   ├── eit_dataset.pt          # Consolidated bundle (X_diff, X_raw, X_baseline, Y, metadata)
│   ├── eit_diff_voltages_X.pt  # (N, 16) Time-Difference voltage tensors
│   ├── eit_raw_voltages_X.pt   # (N, 16) Raw anomaly voltage tensors
│   ├── eit_baseline_voltages_X.pt
│   └── eit_root_masks_Y.pt     # (N, 1, 64, 64) Ground-truth spatial masks
├── src/
│   ├── __init__.py
│   ├── generate_synthetic_data.py # pyEIT 2D circular FEM forward simulator
│   ├── unet_model.py           # 1D-to-2D Projection Head + Spatial U-Net
│   ├── dataset.py              # PyTorch Dataset & DataLoader utilities
│   ├── esp32_interface.py      # ESP32 serial interface + simulation fallback
│   ├── train.py                # Model training, checkpointing & evaluation
│   └── app.py                  # Real-time Streamlit reconstruction dashboard
├── models/
│   └── best_eit_unet.pth       # Optimal trained model weights
├── test_plot.png               # Validation inspection plot (FEM mesh, mask, Delta V)
├── inference_results.png       # 3-sample validation comparison visual
└── requirements.txt            # Dependency specification
```

---

## 4. Usage Commands

### Launch Real-Time Streamlit Dashboard
```powershell
.venv\Scripts\streamlit.exe run src/app.py
```

### Run Model Training Pipeline
```powershell
.venv\Scripts\python.exe src/train.py --epochs 50 --batch_size 128
```

### Test Hardware Interface (Standalone)
```powershell
.venv\Scripts\python.exe src/esp32_interface.py
```
