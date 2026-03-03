# lorenz63.py
import numpy as np
from Models.ssm import StateSpaceModel
from Models.transition import GaussianTransition
from Models.observation import GaussianObservation
from Models.priors import GaussianPrior



class Lorenz63SSM(StateSpaceModel):
    """
    Discrete-time Lorenz-63 system via Euler discretization.
    """

    def __init__(
        self,
        sigma=10.0,
        rho=28.0,
        beta=8.0 / 3.0,
        dt=0.01,
        m0=np.array([1.0, 1.0, 1.0]),
        P0=0.01 * np.eye(3),
        Sigma=1e-4 * np.eye(3),
        Gamma=1e-2 * np.eye(3),
        H=None,
    ):

        H = np.eye(3) if H is None else np.atleast_2d(H)

        d = 3
        m = H.shape[0]

        Sigma = np.atleast_2d(Sigma)
        Gamma = np.atleast_2d(Gamma)

        # ---------- Transition ----------
        def f(x, theta=None):
            x1, x2, x3 = x
            dx1 = sigma * (x2 - x1)
            dx2 = x1 * (rho - x3) - x2
            dx3 = x1 * x2 - beta * x3
            return x + dt * np.array([dx1, dx2, dx3])

        def f_x(x, theta=None):
            x1, x2, x3 = x
            J = np.array([
                [-sigma, sigma, 0.0],
                [rho - x3, -1.0, -x1],
                [x2, x1, -beta]
            ])
            return np.eye(3) + dt * J

        def Q_func(x, theta=None):
            return Sigma

        transition = GaussianTransition(
            f=f,
            Q_func=Q_func,
            d=d,
            f_x=f_x
        )

        # ---------- Observation ----------
        def h(x, theta=None):
            return H @ x

        def h_x(x, theta=None):
            return H

        def R_func(x, theta=None):
            return Gamma

        observation = GaussianObservation(
            h=h,
            R_func=R_func,
            m=m,
            h_x=h_x
        )

        # ---------- Prior ----------
        prior = GaussianPrior(
            m0=np.atleast_1d(m0),
            P0=np.atleast_2d(P0)
        )

        # ---------- Build full SSM ----------
        super().__init__(transition, observation, prior)