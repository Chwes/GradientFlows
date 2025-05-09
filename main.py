"""
main.py

Entry point for executing gradient flow solvers via Hydra configuration.
Handles sampling, evaluation, solver execution, and data saving.

Functions
---------
save_outputs(output_data, prefix)
    Save dictionary of output arrays to disk.

main(config)
    Main Hydra-configured experiment runner.
"""

from omegaconf import OmegaConf, DictConfig
import hydra
from hydra.utils import instantiate, get_method
import gc
import os
from src.utils import *
import logging

log = logging.getLogger(__name__)

def save_outputs(output_data: dict, prefix: str, run_dir: str = "."):
    """
    Save numpy arrays to disk using a common prefix.

    Parameters
    ----------
    output_data : dict
        Dictionary mapping filenames to numpy arrays.
    prefix : str
        Prefix for all output filenames.
    """
    os.makedirs(run_dir, exist_ok=True)
    for name, data in output_data.items():
        np.save(os.path.join(run_dir, f"{prefix}_{name}.npy"), data)

@hydra.main(config_path="configs", config_name="config_main.yaml", version_base=None)
def main(config: DictConfig):
    """
    Main Hydra-configured experiment pipeline. Handles:
    - Instantiating prior and target distributions
    - Sampling particles
    - Executing gradient flow solver
    - Saving output arrays

    Parameters
    ----------
    config : DictConfig
        Configuration object parsed by Hydra.
    """
    log.info("Configuration:\n%s", OmegaConf.to_yaml(config))

    try:
        # Step 1: Initialize the prior distribution from config
        prior = instantiate(config.prior)

        # Step 2: Load or sample particles for constraint
        if config.get("load_data"):
            x_prior_constr = np.load(config.load_data)[:, -1, :]
        else:
            x_prior_constr = prior.sample(config.num_samples_constr, config.prior_sampling_constr, seed=config["seed"])
            x_prior_constr = x_prior_constr[np.argsort(x_prior_constr[:, 0])]

        # Step 3: Sample evaluation particles
        x_prior_eval = prior.sample(config.num_samples_eval, config.prior_sampling_eval)
        x_prior_eval = x_prior_eval[np.argsort(x_prior_eval[:, 0])]

        # Step 4: Compute initial densities under the prior
        prior_pdf = prior.get_pdf()
        initial_density_constr = prior_pdf(x_prior_constr)
        initial_density_eval = prior_pdf(x_prior_eval)

        # Step 5: Initialize target distribution and compute score/log-likelihood
        target = instantiate(config.target)
        log_likelihood = lambda x: target.get_log_prob()(x) - prior.get_log_prob()(x)
        target_score = target.get_grad_log_prob()

        # Step 6: Instantiate solver with appropriate arguments
        solver = instantiate(
                config.solver,
                x_prior_constr, x_prior_eval,
                config.kernel, config.integrator, config.solver,
                target_score=target_score if config.solver.type == "SVGD" else None,
                log_likelihood=log_likelihood if config.solver.type == "KFRFlow" else None
        )

        # Step 7: Run solver to get particle evolution and divergence
        t_values, x_constr, div_x_constr, x_eval, div_x_eval, score_values = solver()

        # Step 8: Integrate divergence fields for density evolution
        evolved_density_constr = integrate(t_values, initial_density_constr[:, np.newaxis].copy(), div_x_constr)
        evolved_density_eval = integrate(t_values, initial_density_eval[:, np.newaxis].copy(), div_x_eval)

        # Step 8.5: Self-normalized importance sampling (SNIS)
        integrand_fn = get_method(config.integrand._target_)

        snis_result = self_normalized_importance_sampling(
            x_eval[:,-1,:], evolved_density_eval[:,-1], target.get_pdf(), integrand_fn
        )


        # Step 9: Package outputs and save to disk
        output_data = {
            "t_values": t_values,
            "x_constr": x_constr,
            "div_x_constr": div_x_constr,
            "x_eval": x_eval,
            "div_x_eval": div_x_eval,
            "evolved_density_constr": evolved_density_constr,
            "evolved_density_eval": evolved_density_eval,
            "prior_density_constr": initial_density_constr,
            "prior_density_eval": initial_density_eval,
            "score_values": score_values,
            "unnormalized_weights": snis_result["unnormalized_weights"],
            "normalized_weights": snis_result["normalized_weights"],
            "snis_estimate": snis_result["snis_estimate"],
        }
        save_outputs(output_data, config.prior_sampling_constr, run_dir=config.run_dir)

        # Step 10: Free memory manually (optional)
        del t_values, x_constr, div_x_constr, x_eval, div_x_eval
        gc.collect()

    except RuntimeError as e:
        # Handle runtime errors
        print(f"[SKIPPED] {e}")


if __name__ == "__main__":
    main()
