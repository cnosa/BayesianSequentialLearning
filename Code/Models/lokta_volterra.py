# lokta_volterra.py
import numpy as np
from Models.ssm import StateSpaceModel
from Models.transition import GaussianTransition
from Models.observation import GaussianObservation, NonlinearProductObservation
from Models.priors import GaussianPrior, LogNormalPrior



class LotkaVolterraSSM(StateSpaceModel):
    """
    Discrete-time Lotka–Volterra model via Euler discretization.
    State: x = (prey, predator)
    """

    def __init__(
        self,
        alpha=1.0,
        beta=0.5,
        delta=0.5,
        gamma=1.0,
        dt=0.01,
        m0=np.array([1.0, 1.0]),
        P0=0.01 * np.eye(2),
        Sigma=1e-4 * np.eye(2),
        Gamma=1e-2 * np.eye(2),
        H=None,
    ):

        self.alpha = alpha
        self.beta = beta
        self.delta = delta
        self.gamma = gamma
        self.dt = dt

        Sigma = np.atleast_2d(Sigma)
        Gamma = np.atleast_2d(Gamma)

        H = np.eye(2) if H is None else np.atleast_2d(H)

        d = 2
        m = H.shape[0]

        # ---------- Transition ----------
        def f(x, theta=None):
            x1, x2 = x
            dx1 = alpha * x1 - beta * x1 * x2
            dx2 = delta * x1 * x2 - gamma * x2
            return x + dt * np.array([dx1, dx2])

        def f_x(x, theta=None):
            x1, x2 = x
            J = np.array([
                [alpha - beta * x2, -beta * x1],
                [delta * x2, delta * x1 - gamma]
            ])
            return np.eye(2) + dt * J

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


class LotkaVolterraProductSSM(StateSpaceModel):
    """
    Lotka–Volterra with nonlinear product observation
    and LogNormal prior.
    """

    def __init__(
        self,
        alpha=1.0,
        beta=0.5,
        delta=0.5,
        gamma=1.0,
        dt=0.01,
        mean_log=np.array([0.0, 0.0]),
        cov_log=0.25 * np.eye(2),
        Sigma=1e-4 * np.eye(2),
        sigma_obs=0.1,
    ):

        d = 2

        Sigma = np.atleast_2d(Sigma)

        # ---------- Transition ----------
        def f(x, theta=None):
            x1, x2 = x
            dx1 = alpha * x1 - beta * x1 * x2
            dx2 = delta * x1 * x2 - gamma * x2
            return x + dt * np.array([dx1, dx2])

        def f_x(x, theta=None):
            x1, x2 = x
            J = np.array([
                [alpha - beta * x2, -beta * x1],
                [delta * x2, delta * x1 - gamma]
            ])
            return np.eye(2) + dt * J

        def Q_func(x, theta=None):
            return Sigma

        transition = GaussianTransition(
            f=f,
            Q_func=Q_func,
            d=d,
            f_x=f_x
        )

        # ---------- Observation ----------
        observation = NonlinearProductObservation(
            sigma=sigma_obs
        )

        # ---------- LogNormal Prior ----------
        prior = LogNormalPrior(
            mean=mean_log,
            cov=cov_log
        )

        super().__init__(
            transition=transition,
            observation=observation,
            prior=prior)