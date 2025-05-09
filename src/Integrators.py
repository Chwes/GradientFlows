
"""
Integrators.py

Numerical integration routines for time-evolving particle dynamics.

Classes
-------
Integrator
    Base interface for integrators.

ExplicitEuler
    Fixed-step Euler integrator.

RK45Integrator
    Adaptive Runge-Kutta 4/5 integrator using SciPy.
"""

import numpy as np
from collections import namedtuple
from scipy.integrate import solve_ivp
from typing import Callable, Any, Tuple, Dict, Union


class Integrator:
    """
    Abstract base class for integrator schemes.

    Parameters
    ----------
    type : str
        Integration method type.
    t0 : float
        Initial time.
    t1 : float
        Final time.
    threshold : float, optional
        Termination threshold based on ODE derivative norm.
    verbose : bool, optional
        If True, print integration progress.
    """
    def __init__(self, **kwargs: Any):
        """
        Initialize the base integrator with common settings.

        Args:
            type (str): Type of integrator.
            t0 (float): Start time.
            t1 (float): End time.
            threshold (float, optional): Termination threshold.
            verbose (bool, optional): If True, print debug information.
        """
        self.type = kwargs["type"]
        self.t0 = kwargs["t0"]
        self.t1 = kwargs["t1"]
        self.termination_threshold = kwargs.get("threshold", None)
        self.verbose = kwargs.get("verbose", True)
        self.result = namedtuple('sol', [
            't', 'x_constr', 'div_x_constr', 'x_eval', 'div_x_eval', 'score', 'success'
        ])

    def __call__(self, *args: Any, **kwargs: Any):
        """
        Abstract call method to execute integration.

        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError("Subclasses must implement this method.")


class ExplicitEuler(Integrator):
    """
    Explicit Euler integrator with fixed step size.

    Parameters
    ----------
    step_size : float
        Step size for the integration.
    """


    def __init__(self, **kwargs: Any):
        """
        Initialize the Explicit Euler integrator.

        Args:
            step_size (float): Time step for Euler updates.
        """
        super().__init__(**kwargs)
        self.step_size = kwargs["step_size"]

    def __call__(
            self,
            f: Callable[[float, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray, Any]],
            x0: np.ndarray,
            n_samples_constr: int,
            n_samples_eval: int,
            n_dims: int
    ) -> namedtuple:
        """
        Integrate using the explicit Euler method.

        Parameters
        ----------
        f : Callable
            ODE system to integrate.
        x0 : np.ndarray
            Initial flattened particle array.
        n_samples_constr : int
            Number of constraint particles.
        n_samples_eval : int
            Number of evaluation particles.
        n_dims : int
            Dimensionality of particles.

        Returns
        -------
        namedtuple
            Integration results including time values, states, divergences, scores, and success flag.
        """

        if self.t1/self.step_size > 1000000:
            raise ValueError(f"Too many time steps: t1={self.t1}, step_size={self.step_size}")

        n_steps = int((self.t1 - self.t0) / self.step_size)
        t_values = np.linspace(self.t0, self.t1, n_steps + 1)

        x_constr = np.zeros((n_samples_constr, n_dims, n_steps + 1))
        x_constr[:, :, 0] = x0[:n_samples_constr * n_dims].reshape(n_samples_constr, n_dims)

        x_eval = np.zeros((n_samples_eval, n_dims, n_steps + 1))
        x_eval[:, :, 0] = x0[n_samples_constr * n_dims:].reshape(n_samples_eval, n_dims)

        div_x_constr = np.zeros((n_samples_constr, n_steps + 1))
        div_x_eval = np.zeros((n_samples_eval, n_steps + 1))

        x = x0.copy()
        for i in range(1, n_steps + 1):
            t = t_values[i - 1]
            print(f"t={t:.4f}")
            dxdt, div_c, div_e, _ = f(t, x)

            x_c = dxdt[:n_samples_constr * n_dims].reshape(n_samples_constr, n_dims)
            x_e = dxdt[n_samples_constr * n_dims:].reshape(n_samples_eval, n_dims)

            x_constr[:, :, i] = x_constr[:, :, i - 1] + self.step_size * x_c
            x_eval[:, :, i] = x_eval[:, :, i - 1] + self.step_size * x_e

            div_x_constr[:, i] = div_c
            div_x_eval[:, i] = div_e

            x = np.hstack((x_constr[:, :, i].flatten(), x_eval[:, :, i].flatten()))

        return self.result(
            t=t_values,
            x_constr=x_constr.transpose(0, 2, 1),
            div_x_constr=div_x_constr,
            x_eval=x_eval.transpose(0, 2, 1),
            div_x_eval=div_x_eval,
            score=None,
            success=True
        )


class RK45Integrator(Integrator):
    """
    Adaptive RK45 integrator using SciPy's `solve_ivp`.

    Parameters
    ----------
    atol : float
        Absolute tolerance.
    rtol : float
        Relative tolerance.
    """

    def __init__(self, **kwargs):
        """
        Initialize the RK45 integrator.

        Args:
            atol (float): Absolute tolerance for solver.
            rtol (float): Relative tolerance for solver.
        """
        super().__init__(**kwargs)
        self.atol = kwargs["atol"]
        self.rtol = kwargs["rtol"]

    def __call__(
            self,
            f: Callable[[float, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray, Any]],
            x0: np.ndarray,
            n_samples_constr: int,
            n_samples_eval: int,
            n_dims: int
    ) -> namedtuple:
        """
        Integrate using adaptive Runge-Kutta 4(5).

        Parameters
        ----------
        f : Callable
            ODE function.
        x0 : np.ndarray
            Initial state.
        n_samples_constr : int
            Number of constraint particles.
        n_samples_eval : int
            Number of evaluation particles.
        n_dims : int
            Number of dimensions.

        Returns
        -------
        namedtuple
            Integration result with states and diagnostics.
        """
        extra: Dict[str, list] = {"div_c": [], "div_e": [], "scores": []}

        def wrapped_f(t: float, y: np.ndarray) -> np.ndarray:
            dydt, _, _, _ = f(t, y)
            return dydt

        def termination_event(t: float, y: np.ndarray) -> float:
            dydt, _, _, _ = f(t, y)
            norm = np.max(np.abs(dydt[:n_samples_constr * n_dims]))
            if self.verbose:
                print(f"[t={t:.4f}] Max norm: {norm:.8f}")
            return norm - self.termination_threshold

        termination_event.terminal = True
        termination_event.direction = -1

        sol = solve_ivp(
            fun=wrapped_f,
            t_span=(self.t0, self.t1),
            y0=x0,
            method='RK45',
            atol=self.atol,
            rtol=self.rtol,
            events=termination_event,
            dense_output=True
        )

        for ti, yi in zip(sol.t, sol.y.T):
            _, div_c, div_e, scores = f(ti, yi)
            extra["div_c"].append(div_c)
            extra["div_e"].append(div_e)
            extra["scores"].append(scores)

        x_constr = sol.y[:n_samples_constr * n_dims].reshape(n_samples_constr, n_dims, len(sol.t))
        x_eval = sol.y[n_samples_constr * n_dims:].reshape(n_samples_eval, n_dims, len(sol.t))

        score_values: Union[np.ndarray, list] = (
            np.stack(extra["scores"], axis=0).transpose(1, 0, 2)
            if extra["scores"] and hasattr(extra["scores"][0], "__len__") and len(extra["scores"][0]) > 0
            else np.array([])
        )

        return self.result(
            t=sol.t,
            x_constr=x_constr.transpose(0, 2, 1),
            div_x_constr=np.array(extra["div_c"]).T,
            x_eval=x_eval.transpose(0, 2, 1),
            div_x_eval=np.array(extra["div_e"]).T,
            score=score_values,
            success=sol.success
        )
