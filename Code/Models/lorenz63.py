import numpy as np
from Models.ssm import StateSpaceModel

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
        super().__init__(d=3, m=3 if H is None else H.shape[0])

        self.sigma = sigma
        self.rho = rho
        self.beta = beta
        self.dt = dt

        self.m0 = np.asarray(m0)
        self.P0 = np.asarray(P0)

        self.Sigma = np.atleast_2d(Sigma)
        self.Gamma = np.atleast_2d(Gamma)

        self.H = np.eye(3) if H is None else np.atleast_2d(H)

    # ---------- Dynamics ----------
    def f(self, x, theta=None):
        x1, x2, x3 = x
        dx1 = self.sigma * (x2 - x1)
        dx2 = x1 * (self.rho - x3) - x2
        dx3 = x1 * x2 - self.beta * x3
        return x + self.dt * np.array([dx1, dx2, dx3])

    def h(self, x, theta=None):
        return self.H @ x

    # ---------- Jacobians ----------
    def f_x(self, x, theta=None):
        x1, x2, x3 = x
        J = np.array([
            [-self.sigma, self.sigma, 0.0],
            [self.rho - x3, -1.0, -x1],
            [x2, x1, -self.beta]
        ])
        return np.eye(3) + self.dt * J

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

