"""
Distributions.py

Defines base and custom probability distributions for use in gradient flow simulations.

Classes
-------
Distribution
    Abstract base distribution class.

BaseNormal
    Multivariate Gaussian with optional Halton sampling.

GaussianMixture
    Simple mixture of Gaussians using PyTorch.

LogRatioBased
    Base class for log-density ratio models.

Donut
    Distribution centered around a ring.

Spaceships
    Toy problem using sine/cosine of product terms.

Butterfly
    Toy problem using sine/cosine of additive terms.
"""

import numpy as np
from abc import abstractmethod
from typing import Callable, List
from scipy.stats.qmc import Halton
from scipy.special import erfinv
import torch


class Distribution:
    """
    Abstract base class for probability distributions.
    """

    @abstractmethod
    def get_pdf(self) -> Callable:
        """
        Return a function that computes the probability density function.

        Returns
        -------
        Callable
            Function that computes the PDF.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def sample(self, n_samples: int, method: str, seed: int) -> np.ndarray:
        """
        Sample from the distribution.

        Parameters
        ----------
        n_samples : int
            Number of samples to draw.
        method : str
            Sampling method (e.g., "Random", "Halton").
        seed : int
            Random seed for reproducibility.

        Returns
        -------
        np.ndarray
            Samples from the distribution.
        """
        raise NotImplementedError("Subclasses must implement this method.")

class BaseNormal(Distribution):
    """
    Multivariate normal distribution with diagonal covariance support.

    Parameters
    ----------
    mean : np.ndarray
        Mean vector of the Gaussian.
    cov_matrix : np.ndarray or float
        Covariance matrix or scalar (isotropic case).
    """

    def __init__(self, mean: np.ndarray, cov_matrix: np.ndarray, **kwargs):
        """
        Args:
            mean (np.ndarray): Mean vector of the distribution.
            cov_matrix (np.ndarray): Covariance matrix or diagonal values.
        """

        self.mean = np.atleast_1d(mean)
        self.cov_matrix = np.diag(cov_matrix) if np.isscalar(cov_matrix) else np.array(cov_matrix)
        self.n_dims = self.mean.shape[0]
        self.inv_cov_matrix = np.linalg.inv(self.cov_matrix)
        self.norm_const = np.sqrt((2 * np.pi) ** self.n_dims * np.linalg.det(self.cov_matrix))

    def get_pdf(self) -> Callable:
        """
        Return the probability density function.

        Returns
        -------
        Callable
            Function computing the PDF.
        """
        def pdf(x: np.ndarray) -> float:
            """
            Args:
                x (np.ndarray): Points to evaluate the PDF at.

            Returns:
                float: PDF values.
            """
            x = np.atleast_2d(x)
            diff = x - self.mean
            exponent = -0.5 * np.sum(diff @ self.inv_cov_matrix * diff, axis=1)
            return np.exp(exponent) / self.norm_const
        return pdf

    def get_log_prob(self) -> Callable:
        """
        Return the log-probability function.

        Returns
        -------
        Callable
            Function computing log probabilities.
        """
        def log_prob(x: np.ndarray):
            """
            Args:
                x (np.ndarray): Input samples.

            Returns:
                np.ndarray: Log-probability values.
            """
            x = np.atleast_2d(x)
            diff = x - self.mean
            constants = -0.5 * (self.n_dims * np.log(2 * np.pi) + np.log(np.linalg.det(self.cov_matrix)))
            return constants - 0.5 * np.sum(diff @ self.inv_cov_matrix * diff, axis=1)
        return log_prob

    def get_grad_log_prob(self) -> Callable:
        """
        Return the gradient of the log-probability function.

        Returns
        -------
        Callable
            Function computing gradients of log probabilities.
        """
        def grad_log_prob(x: np.ndarray) -> np.ndarray:
            """
            Args:
                x (np.ndarray): Input samples.

            Returns:
                np.ndarray: Gradient of the log-probability.
            """
            x = np.atleast_2d(x)
            return -(x - self.mean) @ self.inv_cov_matrix
        return grad_log_prob

    def sample(self, n_samples: int, method: str, **kwargs) -> np.ndarray:
        """
        Sample from the Gaussian using the specified method.

        Parameters
        ----------
        n_samples : int
            Number of samples to generate.
        method : str
            Sampling method ("Random" or "Halton").
        **kwargs : dict
            Additional sampling options such as random seed.

        Returns
        -------
        np.ndarray
            Sampled points of shape (n_samples, n_dims).
        """

        seed = kwargs.get("seed")
        if seed is not None:
            np.random.seed(seed)

        if method == "Random":
            return np.random.multivariate_normal(self.mean, self.cov_matrix, size=n_samples)
        elif method == "Halton":
            # -- Step 1: Halton sequence --
            sampler = Halton(d=self.n_dims, scramble=True)
            base_samples = sampler.random(n_samples + 1)[1:]

            # -- Step 2: Convert to standard normal using erfinv --
            z = np.sqrt(2) * erfinv(2 * base_samples - 1)  # shape: (1000, 2)

            # -- Step 3: Apply covariance and mean via Cholesky --
            L = np.linalg.cholesky(self.cov_matrix)  # L such that cov = L @ L.T
            x = self.mean + z @ L.T
            return x
        else:
            raise NotImplementedError(f"Sampling method '{method}' not implemented.")

    def eval_pdf(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate the PDF over a batched time-series-like structure.

        Parameters
        ----------
        x : np.ndarray
            Array of shape (batch_size, time, dims).

        Returns
        -------
        np.ndarray
            PDF evaluated for each element in time.
        """
        pdf = self.get_pdf()
        return np.stack([pdf(x[:, i, :]) for i in range(x.shape[1])], axis=1)[..., None]


class GaussianMixture(Distribution):
    """
    Mixture of Gaussians using PyTorch tensors.

    Parameters
    ----------
    means : List[float]
        List of mean vectors for each component.
    variances : List[float]
        List of variance values per component.
    weights : List[float]
        Mixing weights for each component.
    """


    def __init__(self, means: List[float], variances: List[float], weights: List[float]):
        """
        Args:
            means (List[float]): List of mean vectors for each component.
            variances (List[float]): List of variances per component.
            weights (List[float]): Weights for each component (must sum to 1).
        """
        self.means = torch.tensor(means)
        self.variances = torch.tensor(variances)
        self.weights = torch.tensor(weights)
        self.n_dims = self.means.shape[1]
        self.n_components = len(means)

    def get_pdf(self) -> Callable:
        """
        Return the probability density function of the Gaussian mixture.

        Returns
        -------
        Callable
            Function computing the PDF.
        """
        def pdf(x: np.ndarray) -> torch.Tensor:
            """
            Args:
                x (np.ndarray): Input points.

            Returns:
                torch.Tensor: PDF values.
            """
            if isinstance(x, np.ndarray):
                x = torch.from_numpy(x.copy())
            pdf_values = torch.zeros((x.shape[0], self.n_dims))
            for i in range(self.n_components):
                coeff = 1.0 / torch.sqrt(2 * torch.pi * self.variances[i])
                exponent = torch.exp(-0.5 * ((x - self.means[i]) ** 2) / self.variances[i])
                pdf_values += self.weights[i] * coeff * exponent
            return pdf_values
        return pdf

    def get_log_prob(self) -> Callable:
        """
        Return the log-probability function of the Gaussian mixture.

        Returns
        -------
        Callable
            Function computing log-probabilities.
        """
        def log_prob(x: np.ndarray) -> np.ndarray:
            """
            Args:
                x (np.ndarray): Input data.

            Returns:
                np.ndarray: Log-probability values.
            """
            return torch.log(self.get_pdf()(x)).numpy().squeeze()
        return log_prob

    def get_grad_log_prob(self) -> Callable:
        """
        Return gradient of the log-probability for the Gaussian mixture.

        Returns
        -------
        Callable
            Function computing gradients.
        """
        def grad_log_prob(x: np.ndarray) -> np.ndarray:
            """
            Args:
                x (np.ndarray): Input points.

            Returns:
                np.ndarray: Gradients.
            """
            x_tensor = torch.tensor(x, dtype=torch.float32, requires_grad=True)
            log_p = torch.log(self.get_pdf()(x_tensor))
            log_p.backward(torch.ones_like(log_p))
            return x_tensor.grad.numpy()
        return grad_log_prob

    def sample(self, n_samples: int, method: str, **kwargs) -> np.ndarray:
        """
        Not implemented.

        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError("Sampling for GaussianMixture is not implemented.")

    def eval_pdf(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate PDF for a sequence of particles.

        Parameters
        ----------
        x : np.ndarray
            Shape (batch, time, dim).

        Returns
        -------
        np.ndarray
            PDF evaluated per timestep.
        """
        pdf = self.get_pdf()
        return np.stack([pdf(x[:, i, :]).numpy() for i in range(x.shape[1])], axis=1)[..., None]


class LogRatioBased(Distribution):
    """
    Abstract base for distributions defined by a log-density ratio with respect to a base distribution.

    Parameters
    ----------
    forward_model : Callable
        Mapping from inputs to scalar observations.
    grad_forward : Callable
        Gradient of the forward model.
    ystar : float
        Target output value.
    sigma_eps : float
        Standard deviation of observation noise.
    base_dist : Distribution, optional
        Underlying prior distribution.
    """
    def __init__(self,forward_model: Callable, grad_forward: Callable, ystar: float = 2.0, sigma_eps: float = 0.25, base_dist: Distribution = None, **kwargs):
        """
        Args:
            forward_model (Callable): Function mapping input to observable.
            grad_forward (Callable): Gradient of the forward model.
            ystar (float): Target output.
            sigma_eps (float): Observation noise.
            base_dist (Distribution): Base prior distribution.
        """
        self.ystar = ystar
        self.sigma_eps = sigma_eps
        self.base_dist = base_dist if base_dist else BaseNormal(mean=np.zeros(2), cov_matrix=np.eye(2))
        self.forward_model = forward_model
        self.grad_forward = grad_forward

    def get_log_ratio(self) -> Callable:
        """
        Compute log-density ratio.

        Returns
        -------
        Callable
            Function computing log-density ratio.
        """
        def log_ratio(x: np.ndarray) -> np.ndarray:
            G = self.forward_model(x)
            return -0.5 * ((G - self.ystar) ** 2) / (self.sigma_eps ** 2)

        return log_ratio

    def get_grad_log_ratio(self) -> Callable:
        """
        Gradient of the log-density ratio.

        Returns
        -------
        Callable
            Function computing gradient of log ratio.
        """
        def grad_forward(x: np.ndarray) -> np.ndarray:
            """
            Computes the gradient of the log-density ratio for the Donut-type distribution.

            Args:
                x (np.ndarray): Input array of shape (n_samples, n_dims).

            Returns:
                np.ndarray: Gradient array of shape (n_samples, n_dims).
            """
            Gx = self.forward_model(x)
            grad_f = self.grad_forward(x)
            return -((Gx - self.ystar)[:, None]) / self.sigma_eps ** 2 * grad_f
        return grad_forward

    def get_density_ratio(self) -> Callable:
        """
        Compute density ratio (unnormalized).

        Returns
        -------
        Callable
            Function computing the density ratio.
        """
        log_ratio = self.get_log_ratio()

        def density_ratio(x: np.ndarray) -> np.ndarray:
            """
            Computes the density ratio from the log-density ratio.

            Args:
                x (np.ndarray): Input array of shape (n_samples, n_dims).

            Returns:
                np.ndarray: Density ratio values of shape (n_samples,).
            """
            return np.exp(log_ratio(x))

        return density_ratio

    def get_pdf(self) -> Callable:
        """
        Return full PDF from base and density ratio.

        Returns
        -------
        Callable
            Function computing PDF.
        """
        base_pdf = self.base_dist.get_pdf()
        density_ratio = self.get_density_ratio()

        def pdf(x: np.ndarray) -> np.ndarray:
            """
            Computes the PDF of the LogRatioBased distribution.

            Args:
                x (np.ndarray): Input array of shape (n_samples, n_dims).

            Returns:
                np.ndarray: PDF values.
            """
            return base_pdf(x) * density_ratio(x)

        return pdf

    def get_log_prob(self) -> Callable:
        """
        Return log-probability of the distribution.

        Returns
        -------
        Callable
            Function computing log PDF.
        """

        pdf = self.get_pdf()

        def log_prob(x:np.ndarray) -> np.ndarray:
            """
            Computes the log-PDF of the LogRatioBased distribution.

            Args:
                x (np.ndarray): Input array of shape (n_samples, n_dims).

            Returns:
                np.ndarray: Log-probability values.
            """
            return np.log(pdf(x))
        return log_prob

    def get_grad_log_prob(self) -> Callable:
        """
        Return gradient of log-probability.

        Returns
        -------
        Callable
            Function computing gradient.
        """
        grad_log_ratio = self.get_grad_log_ratio()
        grad_log_prior = self.base_dist.get_grad_log_prob()

        def grad_log_prob(x:np.ndarray) -> np.ndarray:
            """
            Computes the total gradient of the log PDF by summing the gradient of the prior and the log-density ratio.

            Args:
                x (np.ndarray): Input array of shape (n_samples, n_dims).

            Returns:
                np.ndarray: Gradient array.
            """
            return grad_log_prior(x) + grad_log_ratio(x)

        return grad_log_prob

    def sample(self, n_samples: int, method: str, **kwargs) -> np.ndarray:
        """
        Not implemented.

        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError("Sampling from this distribution is not implemented.")


class Donut(LogRatioBased):
    """
    Donut-shaped radial distribution defined by a forward model
    based on the Euclidean norm.

    The probability mass is concentrated around a fixed radius (ystar).
    """

    def __init__(self, ystar: float = 2.0, sigma_eps: float = 0.25, base_dist: Distribution = None, **kwargs):
        """
        Initialize the Donut distribution.

        Args:
            ystar (float): Target radius around which the density is concentrated.
            sigma_eps (float): Standard deviation of the radial Gaussian kernel.
            base_dist (Distribution, optional): Base distribution to use. Defaults to standard Gaussian.
            **kwargs: Additional keyword arguments for flexibility.
        """
        self.G = self._get_model()
        self.grad_G = self._get_grad_model()
        super().__init__(self.G, self.grad_G, ystar, sigma_eps, base_dist)

    @staticmethod
    def _get_model() -> Callable:
        """
        Returns:
            Callable: Function that computes the Euclidean norm of each point.
        """
        def model(x: np.ndarray) -> np.ndarray:
            """
            Compute Euclidean norm (L2 distance) for each row in x.

            Args:
                x (np.ndarray): Input array of shape (n_samples, n_dims).

            Returns:
                np.ndarray: Norms of shape (n_samples,).
            """
            return np.linalg.norm(x, axis=-1)
        return model

    @staticmethod
    def _get_grad_model() -> Callable:
        """
        Returns:
            Callable: Function that computes gradient of Euclidean norm.
        """
        def grad(x: np.ndarray) -> np.ndarray:
            """
            Compute gradient of the norm function.

            Args:
                x (np.ndarray): Input array of shape (n_samples, n_dims).

            Returns:
                np.ndarray: Normalized direction vectors (unit gradients).
            """
            return x / np.linalg.norm(x, axis=-1, keepdims=True)
        return grad


class Spaceships(LogRatioBased):
    """
    Toy 2D distribution shaped using sine and cosine patterns
    of the product of coordinates.
    """
    def __init__(self, ystar: float = 2.0, sigma_eps: float = 0.25, base_dist: Distribution = None, **kwargs):
        """
        Initialize the Spaceships distribution.

        Args:
            ystar (float): Target output value around which the density is concentrated.
            sigma_eps (float): Noise standard deviation.
            base_dist (Distribution, optional): Base distribution. Defaults to standard normal.
            **kwargs: Optional extra arguments.
        """
        self.G = self._get_model()
        self.grad_G = self._get_grad_model()
        super().__init__(self.G, self.grad_G, ystar, sigma_eps, base_dist)

    @staticmethod
    def _get_model() -> Callable:
        """
        Returns:
            Callable: Function that computes sin(x₀ * x₁) + cos(x₀ * x₁).
        """
        def model(x: np.ndarray) -> np.ndarray:
            """
            Compute the Spaceships forward model.

            Args:
                x (np.ndarray): Input of shape (n_samples, 2).

            Returns:
                np.ndarray: Output of the function.
            """
            return np.sin(x[:, 0] * x[:, 1]) + np.cos(x[:, 0] * x[:, 1])
        return model

    @staticmethod
    def _get_grad_model() -> Callable:
        """
        Returns:
            Callable: Function that computes the gradient of the Spaceships model.
        """

        def grad(x: np.ndarray) -> np.ndarray:
            """
            Compute the gradient of sin(x₀ * x₁) + cos(x₀ * x₁).

            Args:
                x (np.ndarray): Input array of shape (n_samples, 2).

            Returns:
                np.ndarray: Gradients of shape (n_samples, 2).
            """
            x0, x1 = x[:, 0], x[:, 1]
            term = np.cos(x0 * x1) - np.sin(x0 * x1)
            grad_x0 = x1 * term
            grad_x1 = x0 * term
            return np.vstack((grad_x0, grad_x1)).T

        return grad



class Butterfly(LogRatioBased):
    """
    A sine-cosine-based 2D distribution where the log-density
    is shaped by sin(x₁) + cos(x₀), creating a butterfly-like pattern.
    """

    def __init__(self, ystar: float = 2.0, sigma_eps: float = 0.25, base_dist: Distribution = None, **kwargs):
        """
        Initialize the Butterfly distribution.

        Args:
            ystar (float): Target output for the forward model.
            sigma_eps (float): Gaussian noise around the target.
            base_dist (Distribution, optional): Base prior distribution. Defaults to standard normal.
            **kwargs: Additional unused arguments for compatibility.
        """
        self.G = self._get_model()
        self.grad_G = self._get_grad_model()
        super().__init__(self.G, self.grad_G, ystar, sigma_eps, base_dist)

    @staticmethod
    def _get_model() -> Callable:
        """
        Returns:
            Callable: Function that computes sin(x₁) + cos(x₀).
        """
        def model(x: np.ndarray) -> np.ndarray:
            """
            Compute the Butterfly forward model.

            Args:
                x (np.ndarray): Input of shape (n_samples, 2).

            Returns:
                np.ndarray: Output values.
            """
            return np.sin(x[:, 1]) + np.cos(x[:, 0])
        return model

    @staticmethod
    def _get_grad_model() -> Callable:
        """
        Returns:
            Callable: Function that computes gradient of the Butterfly model.
        """
        def grad(x: np.ndarray) -> np.ndarray:
            """
            Compute gradients of sin(x₁) + cos(x₀) with respect to each input component.

            Args:
                x (np.ndarray): Input array of shape (n_samples, 2).

            Returns:
                np.ndarray: Gradient of shape (n_samples, 2).
            """
            x0, x1 = x[:, 0], x[:, 1]
            grad_x0 = -np.sin(x0)
            grad_x1 = np.cos(x1)
            return np.vstack((grad_x0, grad_x1)).T
        return grad