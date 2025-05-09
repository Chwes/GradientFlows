"""
Kernel.py

High-dimensional kernel definitions for particle-based gradient flows.

Classes
-------
Kernel
    Abstract base class for kernels used in SVGD/KFR-based particle systems.

RBF
    Radial Basis Function (Gaussian) kernel implementation.

InverseMultiquadric
    Inverse multiquadric kernel for smoother particle interactions.
"""

import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import kv, gamma
from typing import Callable


class Kernel:
    """
    Base class for implementing kernels for particle-based flows.

    Parameters
    ----------
    data_array : np.ndarray
        Constraint particle array of shape (n_samples, n_dim).
    flow : str
        The type of flow ('SVGD' or 'KFRFlow').
    **kwargs : dict
        Additional keyword arguments, such as bandwidth (bw).

    Attributes
    ----------
    flow_method : str
        The gradient flow method.
    x_constr : np.ndarray
        Constraint particles.
    n_samples_constr : int
        Number of constraint particles.
    n_dim : int
        Dimensionality of each particle.
    bandwidth : float
        Bandwidth used in the kernel.
    bw_sq : float
        Squared bandwidth for efficiency.
    """

    def __init__(self, data_array: np.ndarray, flow: str, **kwargs):
        self.flow_method = flow
        self.x_constr = data_array
        self.n_samples_constr, self.n_dim = data_array.shape
        self.bandwidth = kwargs.get("bw", self._compute_median(self.x_constr))
        # Handle bw = None (null in YAML)
        bw_from_config = kwargs.get("bw")
        self.bandwidth = self._compute_median(self.x_constr) if bw_from_config is None else bw_from_config
        self.bw_sq = self.bandwidth ** 2
        self.nu = kwargs.get("nu",None)


    def _compute_median(self, x: np.ndarray) -> float:
        """
        Compute the median-based bandwidth using pairwise distances.

        Parameters
        ----------
        x : np.ndarray
            Input particle array.

        Returns
        -------
        float
            Median-based bandwidth.
        """
        dMat = cdist(x, x, metric='sqeuclidean')
        med = np.median(dMat)
        return np.sqrt(0.5 * med / np.log(self.n_samples_constr - 1))


    def __call__(self, x: np.ndarray):
        """
        Evaluate the kernel and its derivatives on evaluation particles.

        Parameters
        ----------
        x : np.ndarray
            Evaluation particles.

        Returns
        -------
        tuple
            Kernel values and their gradients/divergences.
        """
        dist = self._get_distance(x)  # shape: (n_constr, n_eval, dim)
        sqdist = np.sum(dist ** 2, axis=2)  # (n_constr, n_eval)
        weighted_dist = dist / self.bw_sq  # (n_constr, n_eval, dim)
        kernel_values = self._kernel_function()(sqdist)  # (n_constr, n_eval)

        grads1 = self._grad_1()(weighted_dist, kernel_values)
        div_grad1 = self._div_grad_1()(weighted_dist, kernel_values)

        if self.flow_method == "SVGD":
            grads2 = self._grad_2()(weighted_dist, kernel_values)
            return kernel_values, grads1, grads2, div_grad1

        return kernel_values, grads1, div_grad1

    def _get_distance(self, x: np.ndarray) -> np.ndarray:
        """
        Compute pairwise particle distances depending on flow method.

        Parameters
        ----------
        x : np.ndarray
            Evaluation particles.

        Returns
        -------
        np.ndarray
            Pairwise difference tensor.
        """
        return self.x_constr[:, None, :] - x[None, :, :] if self.flow_method == "SVGD" else x[None, :, :] - self.x_constr[:, None, :]

    def _kernel_function(self) -> Callable:
        """
        Kernel function definition (to be implemented by subclass).

        Returns
        -------
        Callable
            Kernel function.
        """
        raise NotImplementedError

    def _grad_1(self) -> Callable:
        """
        Gradient of the kernel w.r.t. constraint particles.

        Returns
        -------
        Callable
            Gradient function.
        """
        raise NotImplementedError

    def _grad_2(self) -> Callable:
        """
        Gradient of the kernel w.r.t. evaluation particles.

        Returns
        -------
        Callable
            Gradient function.
        """
        raise NotImplementedError

    def _div_grad_1(self) -> Callable:
        """
        Divergence of gradient w.r.t. constraint particles.

        Returns
        -------
        Callable
            Divergence function.
        """
        raise NotImplementedError

    def _div_grad_2(self) -> Callable:
        """
         Divergence of gradient w.r.t. evaluation particles.

         Returns
         -------
         Callable
             Divergence function.
         """
        raise NotImplementedError


class RBF(Kernel):
    """
    Radial Basis Function (Gaussian) kernel.
    """

    def _kernel_function(self) -> Callable:
        """
        RBF kernel: exp(-||x - y||^2 / (2 * bw^2))

        Returns
        -------
        Callable
            Kernel function.
        """
        return lambda sq_dist: np.exp(-sq_dist / (2 * self.bw_sq))

    def _grad_1(self) -> Callable:
        """
        Gradient w.r.t. constraint particles.

        Returns
        -------
        Callable
            Gradient function.
        """
        return lambda weighted_dist, kernel_values: -weighted_dist * kernel_values[..., None]

    def _grad_2(self) -> Callable:
        """
        Gradient w.r.t. evaluation particles.

        Returns
        -------
        Callable
            Gradient function.
        """
        return lambda weighted_dist, kernel_values: weighted_dist * kernel_values[..., None]

    def _div_grad_1(self) -> Callable:
        """
        Divergence of gradient w.r.t. constraint particles.

        Returns
        -------
        Callable
            Divergence function.
        """
        return lambda weighted_dist, kernel_values: (
            (kernel_values[..., None] / self.bw_sq - weighted_dist ** 2 * kernel_values[..., None]).sum(axis=-1)
        )

    def _div_grad_2(self) -> Callable:
        """
        Divergence of gradient w.r.t. evaluation particles.

        Returns
        -------
        Callable
            Divergence function.
        """
        return lambda weighted_dist, kernel_values: (
            (-kernel_values[..., None] / self.bw_sq - weighted_dist ** 2 * kernel_values[..., None]).sum(axis=2)
        )


class InverseMultiquadric(Kernel):
    """
    Inverse Multiquadric kernel: (1 + ||x - y||^2 / bw^2)^(-1/2)
    """

    def _kernel_function(self) -> Callable:
        """
        Inverse multiquadric function.

        Returns
        -------
        Callable
            Kernel function.
        """
        return lambda sq_dist: (1 + sq_dist / self.bw_sq) ** (-1 / 2)

    def _grad_1(self) -> Callable:
        """
        Gradient w.r.t. constraint particles.

        Returns
        -------
        Callable
            Gradient function.
        """
        return lambda weighted_dist, kernel_values: -weighted_dist * kernel_values[..., None] ** 3

    def _grad_2(self) -> Callable:
        """
        Gradient w.r.t. evaluation particles.

        Returns
        -------
        Callable
            Gradient function.
        """
        return lambda weighted_dist, kernel_values: weighted_dist * kernel_values[..., None] ** 3

    def _div_grad_1(self) -> Callable:
        """
        Divergence of gradient w.r.t. constraint particles.

        Returns
        -------
        Callable
            Divergence function.
        """
        return lambda weighted_dist, kernel_values: (
            -kernel_values[..., None] ** 3 / self.bw_sq + 3 * weighted_dist ** 2 * kernel_values[..., None] ** 5
        ).sum(axis=2)

    def _div_grad_2(self) -> Callable:
        """
        Divergence of gradient w.r.t. evaluation particles.

        Returns
        -------
        Callable
            Divergence function.
        """
        return lambda weighted_dist, kernel_values: (
            kernel_values[..., None] ** 3 / self.bw_sq - 3 * weighted_dist ** 2 * kernel_values[..., None] ** 5
        ).sum(axis=2)
