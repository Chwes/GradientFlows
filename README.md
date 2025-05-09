# Particle-Based Gradient Flow

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
├── src/
│   ├── distributions/       # Custom distributions
│   ├── flows/               # Core solvers: SVGD, KFRFlow
│   ├── kernels/             # Kernel definitions
│   ├── integrators/         # Numerical ODE solvers
│   ├── integrands/          # Functions for SNIS estimation
│   └── utils/               # Plotting, metrics, and general utilities
│  
├── example.py               # 1D SVGD example
├── example_kfrflow.py       # 2D KFRFlow example
├── configs/                 # YAML config files for experiments (not uploaded)
├── outputs/                 # Auto-created directory for results
└── main.py              # Hydra-based experiment entry point
```

---

## 🚀 Installation

```bash
git clone https://github.com/Chwes/GradientFlows.git
cd GradientFlows
pip install -r requirements.txt
pip install hydra-core torch  # If not already included
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

## 🧠 Core Pipeline (`main.py`)

The main experiment runner, powered by **Hydra**, is defined in `main.py`. It manages the complete simulation pipeline from sampling to result saving.

### 🔄 Pipeline Overview

1. **Initialize Prior**  
   Instantiates a custom prior distribution using the Hydra config.

2. **Sample Constraint & Evaluation Particles**  
   Samples particles used in the solver and for diagnostics (or loads from file if specified).

3. **Compute Initial Densities**  
   Evaluates prior density at sampled points.

4. **Define Target Distribution**  
   Sets up the target’s log-likelihood and score function for gradient-based flows.

5. **Solver Instantiation**  
   Dynamically builds the solver (SVGD or KFRFlow) with kernels, integrators, and particle data.

6. **Run Dynamics**  
   Evolves particles and computes divergence fields over time.

7. **Evolve Densities**  
   Integrates the divergence to produce evolved densities.

8. **SNIS Estimation**  
   Applies self-normalized importance sampling for final target estimation.

9. **Save Results**  
   Outputs all arrays (positions, densities, weights, etc.) to disk with timestamped folders.

### 🗂 Saved Outputs

Each experiment stores the following as `.npy` files:

- Particle trajectories (Gradient Flow particles = `x_constr`, ODE particles = `x_eval`)
- Evolved densities
- Divergence fields
- SNIS weights and estimates
- Time steps (`t_values`)

These are saved to the `outputs/` directory by default, organized per run.

---
