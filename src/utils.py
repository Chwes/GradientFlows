from typing import Callable
import numpy as np
import scipy
import seaborn as sns

sns.set_theme(style="whitegrid", palette="deep")
palette = sns.color_palette("deep", n_colors=10)

def integrate(t_values, initial_density, div):
    if initial_density.ndim == 1:
        initial_density = initial_density[:, np.newaxis]

    log_density = np.log(initial_density)
    log_density_evolved = log_density - scipy.integrate.cumulative_trapezoid(div, t_values, axis=-1, initial=0.0)
    return np.exp(log_density_evolved)


def self_normalized_importance_sampling(x_samples: np.ndarray,
                                        evolved_density: np.ndarray,
                                        target_pdf: Callable[[np.ndarray], np.ndarray],
                                        integrand_fn: Callable[[np.ndarray], np.ndarray]) -> dict:
    """
    Perform self-normalized importance sampling (SNIS) in potentially high-dimensional settings.

    Parameters
    ----------
    x_samples : np.ndarray
        Evaluation particles of shape (N, D), where N is the number of particles and D the dimension.
    evolved_density : np.ndarray
        Density values at x_samples (shape: (N,) or (N, 1)).
    target_pdf : Callable
        Function to evaluate the target density: R^D -> R.
    integrand_fn : Callable
        Function to evaluate under the target distribution: R^D -> R or R^D -> R^K.

    Returns
    -------
    dict
        Dictionary with keys:
        - "unnormalized_weights": Raw importance weights.
        - "normalized_weights": Weights normalized to sum to 1.
        - "snis_estimate": Estimated integral(s).
    """
    # Ensure 1D shape for evolved_density
    evolved_density = evolved_density.squeeze()
    assert evolved_density.shape[0] == x_samples.shape[0], "Shape mismatch between samples and density"

    # Evaluate target pdf and integrand
    target_vals = target_pdf(x_samples)  # shape: (N,)
    f_eval = integrand_fn(x_samples)     # shape: (N,) or (N, K)

    # Compute raw weights
    weights = np.nan_to_num(target_vals / evolved_density, nan=0.0, posinf=0.0, neginf=0.0)

    # Normalize weights
    sum_weights = np.sum(weights)
    if sum_weights == 0:
        raise ValueError("All weights are zero — check densities or sampling quality.")
    normalized_weights = weights / sum_weights

    # Weighted estimate (support vector-valued integrands)
    estimate = np.sum(normalized_weights[:, None] * f_eval, axis=0) if f_eval.ndim > 1 else np.sum(normalized_weights * f_eval)

    return {
        "unnormalized_weights": weights,
        "normalized_weights": normalized_weights,
        "snis_estimate": estimate
    }



import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde, norm
from scipy.integrate import trapezoid

sns.set_theme(style="whitegrid", palette="deep")
palette = sns.color_palette("deep", n_colors=10)

def gaussian_pdf(x, mean, variance):
    return 1 / np.sqrt(2 * np.pi * variance) * np.exp(-0.5 * ((x - mean) ** 2) / variance)

def gaussian_mixture(x, means, variances, weights):
    pdf = np.zeros_like(x)
    for m, v, w in zip(means, variances, weights):
        pdf += w * gaussian_pdf(x, m, v)
    return pdf

def run_1d_plots(run_dir, sampling_method="Halton", method="eval", highlight_indices=None):
    """
    Run all diagnostic plots for a saved experiment.

    Parameters:
    - run_dir : str
        Path to the saved output directory.
    - sampling_method : str
        The sampling method used (e.g. 'Halton', 'Uniform').
    - method : str
        Whether to use 'eval' or 'constr' data.
    - highlight_indices : list[int] or None
        Indices to highlight in AUC and KDE plots.
    """
    if highlight_indices is None:
        highlight_indices = []

    print(f"📊 Generating plots for: {run_dir}")

    try:
        # Plot SNIS densities and weights
        plot_snis_densities(
            run_dir,
            sampling_method=sampling_method,
            method=method,
            integrand_fn=lambda x: x,
            save_path=os.path.join(run_dir, "plot_snis_densities.png")
        )
    except Exception as e:
        print(f"[WARNING] SNIS plot failed: {e}")

    try:
        # Plot AUC comparisons with optional region highlighting
        plot_auc_comparison(
            run_dir,
            sampling_method=sampling_method,
            method=method,
            highlight_indices=highlight_indices,
            save_path=os.path.join(run_dir, "plot_auc_comparison.png")
        )
    except Exception as e:
        print(f"[WARNING] AUC plot failed: {e}")

    try:
        # Plot relative change in spacing between particles
        x_eval = np.load(os.path.join(run_dir, f"{sampling_method}_x_eval.npy"))
        x_constr = np.load(os.path.join(run_dir, f"{sampling_method}_x_constr.npy"))
        x0 = x_eval[:, 0, 0]
        xT = x_eval[:, -1, 0]

        plot_relative_distance_change(
            x0,
            xT,
            x_constr=x_constr,
            save_path=os.path.join(run_dir, f"plot_relative_distance_change.png")
        )
    except Exception as e:
        print(f"[WARNING] Relative distance plot failed: {e}")

    try:
        # Plot particle trajectories over time
        plot_trajectories(
            run_dir,
            sampling_method=sampling_method,
            method="eval",
            save_path=os.path.join(run_dir, f"plot_trajectories.png")
        )
    except Exception as e:
        print(f"[WARNING] Trajectory plot failed: {e}")


def run_2d_plot(run_dir, sampling_method="Halton"):
    """
    Run all 2D visualization plots for a given experiment directory.

    Parameters:
    - run_dir : str
        Path to the experiment output folder.
    - sampling_method : str
        Sampling method used (e.g., 'Halton').
    """
    try:
        plot_2d_posterior_with_particles(
            run_dir,
            sampling_method=sampling_method,
            step=-1,
            save_path=os.path.join(run_dir, "posterior_2d_particles.png")
        )
    except Exception as e:
        print(f"[WARNING] 2D posterior plot failed for {run_dir}: {e}")




# === Plot Functions ===

def plot_snis_densities(run_dir, sampling_method="Halton", method="eval", integrand_fn=None, save_path=None):
    """
    Plot evolved density, reweighted density, and importance weights.
    """
    x = np.load(os.path.join(run_dir, f"{sampling_method}_x_{method}.npy"))[:, -1, 0]
    evolved_density = np.load(os.path.join(run_dir, f"{sampling_method}_evolved_density_{method}.npy"))[:, -1]
    target_pdf_fn = lambda x: gaussian_pdf(x, 4, 1)
    if integrand_fn is None:
        integrand_fn = lambda x: x

    snis_result = self_normalized_importance_sampling(x, evolved_density, target_pdf_fn, integrand_fn)

    weights = snis_result["unnormalized_weights"]
    normalized_weights = snis_result["normalized_weights"]
    reweighted_density = evolved_density * weights
    snis = snis_result["snis_estimate"]
    ess = 1 / np.sum(normalized_weights**2)

    print(f"SNIS: {snis:.4f}, ESS: {ess:.2f}")

    x_plot = np.linspace(-10, 10, 10000)
    target_plot = target_pdf_fn(x_plot)

    color_density = palette[0]
    color_evolved = palette[2]
    color_target = palette[1]
    color_weights = "gray"
    bar_base = 1e-4

    fig, ax2 = plt.subplots(figsize=(8, 4))
    ax1 = ax2.twinx()
    ax2.bar(x, weights, width=0.1, bottom=bar_base, color=color_weights, alpha=0.3, label="Importance Weights")
    ax1.plot(x, evolved_density, label="Evolved Density", color=color_evolved, zorder=4)
    ax1.plot(x, reweighted_density, label="Reweighted Density", color=color_density, linestyle="--")
    ax1.plot(x_plot, target_plot, label="Target Density", color=color_target, linestyle="dotted")
    ax1.scatter(x, np.zeros_like(x), color=color_target, alpha=0.8, label="ODE Particles", marker="o")
    ax2.axhline(1, color=color_target, linestyle='--', linewidth=1, alpha=0.5)
    ax1.set_xlabel("Position")
    ax1.set_ylabel("Density")
    ax2.set_ylabel("Importance Weights")
    ax1.axhline(0, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center')
    ax1.grid(False)
    ax2.grid(False)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_relative_distance_change(x0, xT, x_constr=None, save_path=None):
    """
    Plot relative change in distance between consecutive particles.
    """
    palette = sns.color_palette("deep")
    rel_changes = [(abs(xT[i+1] - xT[i]) - abs(x0[i+1] - x0[i])) / abs(x0[i+1] - x0[i]) if x0[i+1] != x0[i] else 0
                   for i in range(len(x0) - 1)]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axhline(0, linestyle="--", color="gray")
    for i in range(len(rel_changes)):
        ax.plot([x0[i], x0[i + 1]], [rel_changes[i]] * 2, color=palette[0])
        ax.plot([x0[i], x0[i]], [0, rel_changes[i]], linestyle='dashed', color="gray")
        ax.plot([x0[i + 1], x0[i + 1]], [0, rel_changes[i]], linestyle='dashed', color="gray")

    if x_constr is not None:
        ax.scatter(x_constr[:, 0, 0], np.zeros_like(x_constr[:, 0, 0]), marker="^", alpha=0.6, label="SVGD Init", color=palette[2])

    ax.scatter(x0, np.zeros_like(x0), marker="o", alpha=0.8, label="ODE Init", color=palette[1])
    ax.set_xlabel("Initial Position")
    ax.set_ylabel("Relative Δ Distance")
    ax.set_yscale("symlog", linthresh=1e-1)
    ax.set_ylim(bottom=-1.5)
    ax.set_title("Relative Change in Particle Spacing")
    ax.legend()
    ax.grid(True)
    sns.despine()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_trajectories(run_dir, sampling_method="Halton", method="eval", save_path=None):
    """
    Plot particle trajectories over time and compare to initial/target densities.
    """
    positions = np.load(os.path.join(run_dir, f"{sampling_method}_x_{method}.npy")).squeeze()
    t_arr = np.load(os.path.join(run_dir, f"{sampling_method}_t_values.npy"))
    x_constr = np.load(os.path.join(run_dir, f"{sampling_method}_x_constr.npy")).squeeze()

    prior_pdf = lambda x: gaussian_pdf(x, 0, 1)
    target_pdf = lambda x: gaussian_pdf(x, 4, 1)
    x_plot = np.linspace(-10, 10, 1000)
    prior_curve = prior_pdf(x_plot)
    target_curve = target_pdf(x_plot)

    norm_factor = np.max(t_arr) * 0.1
    prior_curve *= norm_factor / np.max(prior_curve)
    target_curve *= norm_factor / np.max(target_curve)

    fig, ax = plt.subplots(figsize=(10, 6))
    t_flipped = np.max(t_arr) - t_arr
    for i in range(positions.shape[0]):
        ax.plot(positions[i, :], t_flipped, color="gray", alpha=0.4, linewidth=0.8)

    ax.plot(x_plot, np.max(t_arr) + prior_curve, color=palette[0], label="Initial Density")
    ax.fill_between(x_plot, np.max(t_arr), np.max(t_arr) + prior_curve, color=palette[0], alpha=0.2)

    ax.plot(x_plot, -target_curve, color=palette[4], label="Target Density")
    ax.fill_between(x_plot, -target_curve, 0, color=palette[4], alpha=0.2)

    ax.scatter(positions[:, 0], [t_flipped[0]] * positions.shape[0], color=palette[3], s=24, alpha=0.6, label="ODE Init")
    ax.scatter(positions[:, -1], [t_flipped[-1]] * positions.shape[0], color=palette[3], s=24, alpha=0.6, label="ODE Final")

    ax.scatter(x_constr[:, 0], [t_flipped[0]] * x_constr.shape[0], color=palette[2], marker='^', s=24, alpha=0.6, label="SVGD Init")
    ax.scatter(x_constr[:, -1], [t_flipped[-1]] * x_constr.shape[0], color=palette[2], marker='^', s=24, alpha=0.6, label="SVGD Final")

    ax.set_xlabel("Position")
    ax.set_ylabel("Time (↓)")
    ax.set_xlim(-10, 10)
    ax.set_ylim(-1.5 * norm_factor, np.max(t_arr) + 1.5 * norm_factor)
    ax.get_yaxis().set_ticks([])
    ax.text(10, np.max(t_arr) + 0.01, "t=0", ha="right")
    ax.text(10, -norm_factor + 0.01, f"t={np.max(t_arr):.2f}", ha="right")

    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)
    sns.despine(left=True)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

def plot_auc_comparison(run_dir, sampling_method="Halton", method="eval", highlight_indices=None, save_path=None):
    """
    Plot initial and final particle densities with highlighted AUC regions between particles.

    Parameters:
    - run_dir : str
        Directory containing the simulation outputs.
    - sampling_method : str
        Sampling method used in the simulation (e.g., "Halton").
    - method : str
        Whether to use "eval" or "constr" data.
    - highlight_indices : list[int]
        Indices for manually highlighting AUC regions.
    - save_path : str or None
        Path to save the figure.
    """
    if highlight_indices is None:
        highlight_indices = []

    # === Load data ===
    x = np.load(os.path.join(run_dir, f"{sampling_method}_x_{method}.npy"))
    x0 = x[:, 0, 0]
    xT = x[:, -1, 0]
    initial_density = np.load(os.path.join(run_dir, f"{sampling_method}_prior_density_{method}.npy"))
    final_density = np.load(os.path.join(run_dir, f"{sampling_method}_evolved_density_{method}.npy"))[:, -1]

    x_min = min(x0.min(), xT.min()) - 1
    x_max = max(x0.max(), xT.max()) + 1
    x_grid = np.linspace(x_min, x_max, 10000)
    target_pdf_fn = lambda x: gaussian_pdf(x, 4, 1)

    fig, (ax_i, ax_f) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    ax_i.plot(x_grid, norm.pdf(x_grid), '--', label="Standard Normal", color=palette[0])
    ax_i.plot(x0, initial_density, label="Initial", color=palette[2])
    ax_i.scatter(x0, np.zeros_like(x0), alpha=0.4)

    for idx in highlight_indices:
        fill_x = np.linspace(x0[idx], x0[idx + 1], 20)
        fill_y = np.interp(fill_x, x0, initial_density)
        ax_i.fill_between(fill_x, fill_y, alpha=0.3, color=palette[idx % len(palette)])

    ax_f.plot(xT, final_density, label="Final", color=palette[2])
    ax_f.plot(x_grid, target_pdf_fn(x_grid), '--', label="Target PDF", color=palette[0])
    ax_f.scatter(xT, np.zeros_like(xT), alpha=0.4)

    for idx in highlight_indices:
        fill_x = np.linspace(xT[idx], xT[idx + 1], 20)
        fill_y = np.interp(fill_x, xT, final_density)
        ax_f.fill_between(fill_x, fill_y, alpha=0.3, color=palette[idx % len(palette)])

    ax_i.set_ylabel("Initial Density")
    ax_f.set_ylabel("Final Density")
    ax_f.set_xlabel("Position")
    ax_i.legend()
    ax_f.legend()

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()



def plot_2d_posterior_with_particles(run_dir, sampling_method="Halton", step=-1, save_path=None):
    """
    Plot 2D posterior PDF as contours and overlay particles.

    Parameters:
    - run_dir : str
        Directory containing output data (expects <sampling_method>_x_constr.npy).
    - sampling_method : str
        Prefix used for saved particle files (e.g., 'Halton').
    - step : int
        Which time step to plot (-1 = final step).
    - save_path : str or None
        If provided, saves the plot to this path.
    """
    def base_pdf(pos, mu=np.array([0, 0]), sigma=np.array([[1, 0], [0, 1]])):
        diff = pos - mu
        inv_sigma = np.linalg.inv(sigma)
        det_sigma = np.linalg.det(sigma)
        norm_const = 1 / (2 * np.pi * np.sqrt(det_sigma))
        exponent = np.einsum('ij,jk,ik->i', diff, inv_sigma, diff)
        return norm_const * np.exp(-0.5 * exponent)

    def G(x): return np.linalg.norm(x, axis=-1)
    def likelihood(x): return np.exp(-0.5 * ((G(x) - 2) ** 2) / (0.25 ** 2))
    def get_pdf(x): return base_pdf(x) * likelihood(x)

    # Grid
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    pos = np.stack([X.flatten(), Y.flatten()], axis=-1)
    pdf_vals = get_pdf(pos).reshape(X.shape)

    # Load particles
    particle_path = os.path.join(run_dir, f"{sampling_method}_x_constr.npy")
    particles = np.load(particle_path)
    if particles.ndim == 3:
        particles = particles[:, step, :]  # e.g., final time step

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    contour = ax.contourf(X, Y, pdf_vals, levels=20, cmap="viridis", alpha=0.8)
    fig.colorbar(contour, ax=ax, label="Posterior PDF")
    ax.scatter(particles[:, 0], particles[:, 1], alpha=0.9, s=10, color=sns.color_palette("deep")[1])
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_aspect("equal")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
