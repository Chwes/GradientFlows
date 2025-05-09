# Particle-Based Gradient Flow Simulator

A Python framework for simulating particle-based gradient flows, supporting both SVGD and KFRFlow dynamics. This toolkit allows researchers to sample complex distributions, visualize particle evolution, and compute self-normalized importance sampling estimates.

---

## 🔧 Features

- **Gradient Flow Solvers**:
  - Stein Variational Gradient Descent (SVGD)
  - Kernelized Fisher-Rao Flow (KFRFlow)

- **Custom Distributions**:
  - Standard Gaussian, Mixtures, Donut, Spaceships, Butterfly

- **Flexible Integration**:
  - RK45 (adaptive) and Euler methods

- **Kernel Support**:
  - RBF and Inverse Multiquadric

- **Diagnostics**:
  - 1D/2D density plots, SNIS estimation, divergence visualizations

---

## 🧩 Project Structure

```
.
├── Distributions.py     # Custom distributions
├── GradientFlow.py      # Core solvers (SVGD, KFRFlow)
├── Kernel.py            # Kernel definitions
├── Integrators.py       # Numerical ODE integrators
├── Integrands.py        # Integrands for SNIS
├── utils.py             # Plotting, integration, and SNIS
├── main.py              # Experiment entry point using Hydra
├── example.py           # Example run (1D, SVGD)
├── example_kfrflow.py   # Example run (2D, KFRFlow)
├── configs/             # YAML config files (not uploaded)
└── outputs/             # Auto-created during experiments
```

---

## 🚀 Installation

```bash
git clone <your-repo-url>
cd <your-repo-name>
pip install -r requirements.txt
```

**Note:** You’ll also need [Hydra](https://hydra.cc/) and PyTorch installed.

---

## 🧪 Usage

Run an example SVGD experiment:

```bash
python example_svgd.py
```

Run a 2D KFRFlow experiment:

```bash
python example_kfrflow.py
```

Each run will generate diagnostic plots and save results to the `outputs/` directory.

---

## ⚙️ Configuration

All experiment settings are managed via Hydra YAML config files (e.g., `config_example_svgd.yaml`). These control:

- Distribution types
- Sampling methods
- Solver parameters
- Output directories

---

## 📈 Visualization

The framework provides built-in visualization for:

- Particle trajectories
- Evolved densities
- SNIS weights
- Relative distance changes

Plots are automatically saved as `.png` in the output directory.

---
