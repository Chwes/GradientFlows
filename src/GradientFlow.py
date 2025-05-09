"""
GradientFlow.py

Implements SVGD and KFR-based particle evolution under gradient flows.
Includes the core BaseGradientFlow class and problem-specific subclasses.

Classes
-------
BaseGradientFlow
    Abstract class for time-evolving particles.

SVGD
    Stein Variational Gradient Descent.

KFRFlow
    Kernelized Fisher-Rao Flow.
"""

import numpy as np
from hydra.utils import instantiate
from typing import Tuple, Any, Dict, List
from hydra.core.hydra_config import HydraConfig
import json
import sys
import os

class BaseGradientFlow:
    """
    Base class for implementing gradient flow solvers.

    Parameters
    ----------
    x_prior_constr : np.ndarray
        Initial particles for constraint evaluation.
    x_prior_eval : np.ndarray
        Initial particles for evaluation purposes.
    kernel_dict : dict
        Configuration dictionary for kernel instantiation.
    integrator_dict : dict
        Configuration dictionary for the ODE integrator.
    **kwargs : dict
        Additional optional keyword arguments.
    """

    def __init__(self,
                 x_prior_constr: np.ndarray,
                 x_prior_eval: np.ndarray,
                 kernel_dict: Any,
                 integrator_dict: Any,
                 solver_dict: Any,
                 **kwargs):
        """
        Args:
            x_prior_constr (np.ndarray): Initial particles for constraint evaluation.
            x_prior_eval (np.ndarray): Initial particles for evaluation purposes.
            kernel_dict (dict): Hydra config dictionary for kernel instantiation.
            integrator_dict (dict): Hydra config dictionary for the ODE integrator.
            **kwargs: Additional optional arguments.
        """
        self.x_constr = x_prior_constr
        self.x_eval = x_prior_eval
        self.kernel_dict = kernel_dict
        self.integrator_dict = integrator_dict
        self.stopping_time = solver_dict["stopping_time"]
        self.n_samples_constr, self.n_dims = x_prior_constr.shape
        self.n_samples_eval = x_prior_eval.shape[0]

    def _reshape_particles(self, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Reshape the flat particle array into constraint and evaluation sets.

        Parameters
        ----------
        y : np.ndarray
            Flattened particle array.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Reshaped constraint and evaluation particle arrays.
        """
        x_constr = y[:self.n_samples_constr * self.n_dims].reshape(self.n_samples_constr, self.n_dims)
        x_eval = y[self.n_samples_constr * self.n_dims:].reshape(self.n_samples_eval, self.n_dims)
        return x_constr, x_eval

    def _init_kernel(self, x: np.ndarray, **kwargs):
        """
        Instantiate a kernel object for the given particle array.

        Parameters
        ----------
        x : np.ndarray
            Particle array.
        **kwargs : dict
            Additional arguments passed to the kernel.

        Returns
        -------
        Any
            Instantiated kernel object.
        """
        return instantiate(self.kernel_dict, x.copy(), self.__class__.__name__,**kwargs)

    def _system_odes(self, t: float, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute the ODE derivatives for the particles.

        Parameters
        ----------
        t : float
            Current time.
        x : np.ndarray
            Flattened particle array.

        Returns
        -------
        Tuple
            Derivatives, divergence at constraint particles, divergence at eval particles, score values.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def __call__(self)-> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Any]:
        """
        Execute the integration procedure.

        Returns
        -------
        Tuple
            Time steps, constraint particles, constraint divergence,
            evaluation particles, evaluation divergence, score values.
        """
        x0_constr = self.x_constr.flatten()
        x0_eval = self.x_eval.flatten()
        self.integrator = instantiate(self.integrator_dict,t1=self.stopping_time)
        y0 = np.hstack((x0_constr, x0_eval))
        sol = self.integrator(self._system_odes, y0, self.n_samples_constr, self.n_samples_eval, self.n_dims)

        return sol.t, sol.x_constr, sol.div_x_constr, sol.x_eval, sol.div_x_eval, sol.score


class SVGD(BaseGradientFlow):
    """
    Stein Variational Gradient Descent (SVGD) gradient flow implementation.

    Parameters
    ----------
    x_prior_constr : np.ndarray
        Constraint particles.
    x_prior_eval : np.ndarray
        Evaluation particles.
    kernel_dict : dict
        Kernel configuration dictionary.
    integrator_dict : dict
        Integrator configuration dictionary.
    solver_dict : dict
        Solver configuration dictionary.
    **kwargs : dict
        Must contain 'target_score'.
    """
    def __init__(self,
                 x_prior_constr: np.ndarray,
                 x_prior_eval: np.ndarray,
                 kernel_dict: Dict[str, Any],
                 integrator_dict: Dict[str, Any],
                 solver_dict: Dict[str, Any],
                 **kwargs):
        """
        Initialize the SVGD solver with target score.

        Parameters
        ----------
        x_prior_constr : np.ndarray
            Constraint particles.
        x_prior_eval : np.ndarray
            Evaluation particles.
        kernel_dict : dict
            Kernel configuration dictionary.
        integrator_dict : dict
            Integrator configuration dictionary.
        solver_dict : dict
            Solver configuration dictionary.
        **kwargs : dict
            Must contain 'target_score'.
        """
        super().__init__(x_prior_constr, x_prior_eval, kernel_dict, integrator_dict, solver_dict)
        self.target_score = kwargs["target_score"]
        self.threshold = integrator_dict.get("threshold", 1e-3)

    def _compute_vt(self, kernel_values: np.ndarray, kernel_grad_x: np.ndarray, score_values:np.ndarray) -> np.ndarray:
        """
        Compute velocity field using kernel and score values.

        Returns
        -------
        np.ndarray
            Velocity vector.
        """

        return (kernel_values.T @ score_values + np.sum(kernel_grad_x, axis=0)) / self.n_samples_constr

    def _compute_div_vt(self, div_kernel: np.ndarray, div_grad_kernel: np.ndarray, score_values: np.ndarray) -> np.ndarray:
        """
        Compute divergence of the velocity field.

        Returns
        -------
        np.ndarray
            Divergence values.
        """

        div_vt = (np.matmul(div_kernel[:, :, 0].T, score_values).squeeze() + np.sum(div_grad_kernel,
                                                                                    axis=0)) / self.n_samples_constr
        return div_vt

    def _system_odes(self, t: float, y: np.ndarray)-> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Define ODE system for SVGD.

        Returns
        -------
        Tuple
            Flattened derivatives, constraint divergence, eval divergence, score.
        """
        x_constr, x_eval = self._reshape_particles(y)

        score_values = self.target_score(x_constr)
        kernel_constr = self._init_kernel(x_constr)
        kernel_eval = self._init_kernel(x_constr)

        k_val_c, k_grad_c, k_div_c, k_div_grad_c = kernel_constr(x_constr)
        k_val_e, k_grad_e, k_div_e, k_div_grad_e = kernel_eval(x_eval)

        vt_constr = self._compute_vt(k_val_c, k_grad_c, score_values).flatten()
        vt_eval = self._compute_vt(k_val_e, k_grad_e, score_values).flatten()

        div_vt_constr = self._compute_div_vt(k_div_c, k_div_grad_c, score_values)
        div_vt_eval = self._compute_div_vt(k_div_e, k_div_grad_e, score_values)

        return np.hstack((vt_constr, vt_eval)), div_vt_constr, div_vt_eval, score_values


class KFRFlow(BaseGradientFlow):
    """
    Kernelized Fisher-Rao Flow (KFRFlow) implementation.

    Parameters
    ----------
    x_prior_constr : np.ndarray
        Constraint particles.
    x_prior_eval : np.ndarray
        Evaluation particles.
    kernel_dict : dict
        Kernel configuration.
    integrator_dict : dict
        Integrator configuration.
    solver_dict : dict
        Solver config (must contain 'regularization_parameter').
    **kwargs : dict
        Must contain 'log_likelihood'.
    """

    def __init__(
        self,
        x_prior_constr: np.ndarray,
        x_prior_eval: np.ndarray,
        kernel_dict: Dict[str, Any],
        integrator_dict: Dict[str, Any],
        solver_dict: Dict[str, Any],
        **kwargs
    ):
        """
        Initialize the KFR solver with log-likelihood.

        Parameters
        ----------
        x_prior_constr : np.ndarray
            Constraint particles.
        x_prior_eval : np.ndarray
            Evaluation particles.
        kernel_dict : dict
            Kernel configuration.
        integrator_dict : dict
            Integrator configuration.
        solver_dict : dict
            Solver config (must contain 'regularization_parameter').
        **kwargs : dict
            Must contain 'log_likelihood'.
        """
        super().__init__(x_prior_constr, x_prior_eval, kernel_dict, integrator_dict,solver_dict)
        self.log_likelihood = kwargs["log_likelihood"]
        self.reg_param = solver_dict["regularization_parameter"]

    def _compute_weighted_kernel(self, x: np.ndarray, kernel_values: np.ndarray) -> np.ndarray:
        """
        Weight the kernel values by the log-likelihood.

        Returns
        -------
        np.ndarray
            Weighted kernel vector.
        """
        log_likelihood = self.log_likelihood(x)[:,np.newaxis]
        weights = log_likelihood - np.mean(log_likelihood)
        return kernel_values @ weights

    def _compute_mt(self, kernel_grad: np.ndarray) -> np.ndarray:
        """
        Construct the regularized mass matrix.

        Returns
        -------
        np.ndarray
            Mass matrix.
        """
        mt = np.einsum('ijd, imd -> jm', kernel_grad, kernel_grad)
        return mt + self.reg_param * np.eye(mt.shape[0])

    def _compute_div_vt(self, div_kernel: np.ndarray, ft: np.ndarray) -> np.ndarray:
        """
        Compute divergence of the velocity field.

        Returns
        -------
        np.ndarray
            Divergence values.
        """
        return div_kernel.T @ft

    def _system_odes(
            self,
            t: float,
            y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List]:
        """
        Define ODE system for KFRFlow.

        Returns
        -------
        Tuple
            Flattened derivatives, constraint divergence, eval divergence, empty list.
        """
        x_constr, x_eval = self._reshape_particles(y)

        kernel = self._init_kernel(x_constr)
        if kernel.bandwidth >100:
            parent_folder = os.path.dirname(os.getcwd())
            with open(os.path.join(parent_folder, "failure_bw.txt"), "a") as f:
                f.write(f"{HydraConfig.get().sweep.subdir} did not converge.\n")
            raise RuntimeError(f"[SKIPPED] bandwidth is {kernel.bandwidth}")
        k_val_c, k_grad_c, k_div_grad_c = kernel(x_constr)
        _, k_grad_e, k_div_grad_e = kernel(x_eval)

        weighted_kernel = self._compute_weighted_kernel(x_constr, k_val_c)
        mt = self._compute_mt(k_grad_c)
        log_path = os.path.join(os.getcwd(), "condMt.jsonl")

        try:
            cond = np.linalg.cond(mt)
            mt_value = {"step": t, "result": cond}
            with open(log_path, "a") as f:
                f.write(json.dumps(mt_value) + "\n")

        except np.linalg.LinAlgError as e:
            # Optional: log failure
            parent_folder = os.path.dirname(os.getcwd())
            with open(os.path.join(parent_folder, "failure.txt"), "a") as f:
                f.write(f"{HydraConfig.get().sweep.subdir} did not converge.\n")
            raise RuntimeError(f"[Hydra Sweep Warning] Step {t}: Matrix error — {str(e)}")

        ft = np.linalg.solve(mt, weighted_kernel).squeeze()

        k_grad_c = k_grad_c.transpose(1, 0, 2)  # -> (n_samples_constr, n_ft, n_dims)
        k_grad_e = k_grad_e.transpose(1, 0, 2)
        vt_constr = np.einsum('ijd,j->id', k_grad_c, ft)  # (n_samples_constr, n_dims)
        vt_eval = np.einsum('ijd,j->id', k_grad_e, ft)  # (n_samples_eval, n_dims)

        div_vt_constr = self._compute_div_vt(k_div_grad_c,ft)
        div_vt_eval = self._compute_div_vt(k_div_grad_e,ft)


        return np.hstack((vt_constr.flatten(), vt_eval.flatten())), div_vt_constr, div_vt_eval, []
