import numpy as np
from Models.ssm import StateSpaceModel

class LinearGaussianSSM(StateSpaceModel):
    def __init__(self, m0, P0, A, H, Sigma, Gamma):
        A = np.atleast_2d(A)
        H = np.atleast_2d(H)

        super().__init__(d=A.shape[0], m=H.shape[0])

        self.m0 = np.atleast_1d(m0)
        self.P0 = np.atleast_2d(P0)

        self.A = A
        self.H = H
        self.Sigma = np.atleast_2d(Sigma)
        self.Gamma = np.atleast_2d(Gamma)

    # Evolution and observation (deterministic)
    def f(self, x, theta=None):
        return self.A @ x

    def h(self, x, theta=None):
        return self.H @ x

    # Jacobians
    def f_x(self, x=None, theta=None):
        return self.A

    def h_x(self, x=None, theta=None):
        return self.H

    # Covariances
    def Q(self, x=None, theta=None):
        return self.Sigma

    def R(self, x=None, theta=None):
        return self.Gamma

    # Prior
    def prior(self):
        return self.m0.copy(), self.P0.copy()

