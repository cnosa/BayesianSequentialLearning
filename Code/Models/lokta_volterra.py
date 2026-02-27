import numpy as np
from Models.ssm import StateSpaceModel

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
        super().__init__(d=2, m=2 if H is None else H.shape[0])

        self.alpha = alpha
        self.beta = beta
        self.delta = delta
        self.gamma = gamma
        self.dt = dt

        self.m0 = np.asarray(m0)
        self.P0 = np.asarray(P0)

        self.Sigma = np.atleast_2d(Sigma)
        self.Gamma = np.atleast_2d(Gamma)

        self.H = np.eye(2) if H is None else np.atleast_2d(H)

    # ---------- Dynamics ----------
    def f(self, x, theta=None):
        x1, x2 = x
        dx1 = self.alpha * x1 - self.beta * x1 * x2
        dx2 = self.delta * x1 * x2 - self.gamma * x2
        return x + self.dt * np.array([dx1, dx2])

    def h(self, x, theta=None):
        return self.H @ x

    # ---------- Jacobians ----------
    def f_x(self, x, theta=None):
        x1, x2 = x
        J = np.array([
            [self.alpha - self.beta * x2, -self.beta * x1],
            [self.delta * x2, self.delta * x1 - self.gamma]
        ])
        return np.eye(2) + self.dt * J

    def h_x(self, x, theta=None):
        return self.H

    # ---------- Covariances ----------
    def Q(self, x=None, theta=None):
        return self.Sigma

    def R(self, x=None, theta=None):
        return self.Gamma

    # ---------- Prior ----------
    def prior(self):
        return self.m0.copy(), self.P0.copy()

