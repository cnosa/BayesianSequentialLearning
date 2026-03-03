# ukf.py
from Filters.bayesian import BayesianFilter
import numpy as np
from scipy import linalg

class UnscentedKalmanFilter(BayesianFilter):
    """
    Unscented Kalman Filter for StateSpaceModel
    """

    def __init__(self, model, theta=None,
                 alpha=0.1, beta=2.0, kappa=0.0):

        super().__init__(model, theta)

        self.d = model.d
        self.m_obs = model.m

        # parameters UKF
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa

        self.lambd = alpha**2 * (self.d + kappa) - self.d
        self.denom = self.d + self.lambd
        self.sqrt_d_lamb = np.sqrt(self.denom)

        self.w0m = self.lambd / self.denom
        self.w0c = self.w0m + (1.0 - alpha**2 + beta)
        self.wim = 1.0 / (2.0 * self.denom)

    # --------------------------------------------------
    @staticmethod
    def log_gaussian_density(y, mu, S):
        d = len(y)
        diff = y - mu

        jitter = 1e-8
        S = 0.5 * (S + S.T) + jitter * np.eye(d)

        c, lower = linalg.cho_factor(S, check_finite=False)
        alpha = linalg.cho_solve((c, lower), diff)
        logdet = 2.0 * np.sum(np.log(np.diag(c)))

        return -0.5 * (diff @ alpha + logdet + d * np.log(2 * np.pi))

    # --------------------------------------------------
    def _sigma_points(self, m, P):
        P = 0.5 * (P + P.T)
        jitter = 1e-10
        A = np.linalg.cholesky(P + jitter * np.eye(self.d))

        X = np.empty((self.d, 2 * self.d + 1))
        X[:, 0] = m

        for i in range(self.d):
            col = A[:, i]
            X[:, 1 + i] = m + self.sqrt_d_lamb * col
            X[:, 1 + self.d + i] = m - self.sqrt_d_lamb * col

        return X

    # --------------------------------------------------
    def predict(self):
               
        X = self._sigma_points(self.m, self.P)

        X_hat = np.zeros_like(X)
        for i in range(2 * self.d + 1):
            X_hat[:, i] = self.model.f(X[:, i], self.theta)

        m_minus = (
            self.w0m * X_hat[:, 0]
            + self.wim * np.sum(X_hat[:, 1:], axis=1)
        )

        try:
            Q = self.model.Q(m_minus, self.theta).astype(float)
        except NotImplementedError:
            raise ValueError("UKF requires Gaussian transition model")

        P_minus = Q.copy()
        for i in range(2 * self.d + 1):
            diff = X_hat[:, i] - m_minus
            w = self.w0c if i == 0 else self.wim
            P_minus += w * np.outer(diff, diff)

        self.m_minus = m_minus
        self.P_minus = 0.5 * (P_minus + P_minus.T)

    # --------------------------------------------------
    def update(self, y):
        X = self._sigma_points(self.m_minus, self.P_minus)

        Y_hat = np.zeros((self.m_obs, 2 * self.d + 1))
        for i in range(2 * self.d + 1):
            Y_hat[:, i] = self.model.h(X[:, i], self.theta)

        mu = (
            self.w0m * Y_hat[:, 0]
            + self.wim * np.sum(Y_hat[:, 1:], axis=1)
        )

        S = self.model.R(self.m_minus, self.theta).astype(float)
        for i in range(2 * self.d + 1):
            dy = Y_hat[:, i] - mu
            w = self.w0c if i == 0 else self.wim
            S += w * np.outer(dy, dy)

        S = 0.5 * (S + S.T) + 1e-8 * np.eye(self.m_obs)

        self.log_likelihood += self.log_gaussian_density(y, mu, S)

        C = np.zeros((self.d, self.m_obs))
        for i in range(2 * self.d + 1):
            dx = X[:, i] - self.m_minus
            dy = Y_hat[:, i] - mu
            w = self.w0c if i == 0 else self.wim
            C += w * np.outer(dx, dy)

        c, lower = linalg.cho_factor(S, check_finite=False)
        K = linalg.cho_solve((c, lower), C.T).T

        self.m = self.m_minus + K @ (y - mu)

        self.P = self.P_minus - K @ S @ K.T
        self.P = 0.5 * (self.P + self.P.T)
        self.P += 1e-10 * np.eye(self.d)
        