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
        try:
            c, lower = linalg.cho_factor(S, check_finite=False)
            alpha = linalg.cho_solve((c, lower), diff)
            logdet = 2 * np.sum(np.log(np.diag(c)))
        except np.linalg.LinAlgError:
            alpha = np.linalg.solve(S, diff)
            logdet = np.log(np.linalg.det(S))

        return -0.5 * (diff @ alpha + logdet + d * np.log(2 * np.pi))

    def predict(self):
        F = self.model.f_x(self.m, self.theta)
        Q = self.model.Q(self.m, self.theta)

        self.m_minus = self.model.f(self.m, self.theta)
        self.P_minus = F @ self.P @ F.T + Q

    def update(self, y):
        H = self.model.h_x(self.m_minus, self.theta)
        R = self.model.R(self.m_minus, self.theta)

        mu = self.model.h(self.m_minus, self.theta)
        S = H @ self.P_minus @ H.T + R

        self.log_likelihood += self.log_gaussian_density(y, mu, S)

        try:
            c, lower = linalg.cho_factor(S, check_finite=False)
            S_inv = linalg.cho_solve((c, lower), np.eye(S.shape[0]))
        except np.linalg.LinAlgError:
            S_inv = np.linalg.inv(S)

        K = self.P_minus @ H.T @ S_inv
        innovation = y - mu

        self.m = self.m_minus + K @ innovation
        self.P = self.P_minus - K @ H @ self.P_minus

        # numerical stability
        self.P = 0.5 * (self.P + self.P.T)
        self.P += 1e-8 * np.eye(self.P.shape[0])