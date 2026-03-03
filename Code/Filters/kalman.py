# kalman.py
from Filters.bayesian import BayesianFilter
import numpy as np
from scipy import linalg


class KalmanFilter(BayesianFilter):
    """
    Kalman Filter / Extended Kalman Filter
    """

    @staticmethod
    def log_gaussian_density(y, mu, S):
        d = len(y)
        diff = y - mu

        # jitter for numerical stability
        jitter = 1e-8
        S = 0.5 * (S + S.T) + jitter * np.eye(d)

        c, lower = linalg.cho_factor(S, check_finite=False)
        alpha = linalg.cho_solve((c, lower), diff)
        logdet = 2.0 * np.sum(np.log(np.diag(c)))

        return -0.5 * (diff @ alpha + logdet + d * np.log(2 * np.pi))

    def predict(self):
        try:
            F = self.model.f_x(self.m, self.theta)
            Q = self.model.Q(self.m, self.theta)
        except NotImplementedError:
            raise ValueError("KalmanFilter requires Gaussian transition model")

        self.m_minus = self.model.f(self.m, self.theta)
        self.P_minus = F @ self.P @ F.T + Q

    def update(self, y):
        H = self.model.h_x(self.m_minus, self.theta)
        R = self.model.R(self.m_minus, self.theta)

        mu = self.model.h(self.m_minus, self.theta)
        S = H @ self.P_minus @ H.T + R

        # log likelihood 
        self.log_likelihood += self.log_gaussian_density(y, mu, S)

        # Cholesky with jitter 
        d = S.shape[0]
        jitter = 1e-8
        S = 0.5 * (S + S.T) + jitter * np.eye(d)

        c, lower = linalg.cho_factor(S, check_finite=False)

        PHt = self.P_minus @ H.T
        K = linalg.cho_solve((c, lower), PHt.T).T

        innovation = y - mu
        self.m = self.m_minus + K @ innovation

        # Joseph form
        I = np.eye(self.P_minus.shape[0])
        self.P = (
            (I - K @ H) @ self.P_minus @ (I - K @ H).T
            + K @ R @ K.T
        )

        self.P = 0.5 * (self.P + self.P.T)
        self.P += 1e-10 * np.eye(self.P.shape[0])