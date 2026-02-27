from Filters.bayesian import BayesianFilter
import numpy as np
from scipy import linalg
from scipy.special import logsumexp

class ParticleFilter(BayesianFilter):
    """
    Particle Filter for StateSpaceModel
    """

    def __init__(self, model, N, theta=None, resample_threshold=0.15):
        super().__init__(model, theta)

        self.N = N
        self.resample_threshold = resample_threshold

        self.d = model.d
        self.m_obs = model.m

        # inicialization of particules from prior
        m0, P0 = model.prior()
        self.particles = np.random.multivariate_normal(
            mean=m0, cov=P0, size=N
        )

        self.logw = np.zeros(N) - np.log(N)

    # --------------------------------------------------
    def _store(self):
        self.history_m.append(self.m.copy())
        self.history_P.append(self.P.copy())
        self.history_ll.append(self.log_likelihood)

        self.history_particles.append(self.particles.copy())
        self.history_weights.append(np.exp(self.logw).copy())


    # --------------------------------------------------
    @staticmethod
    def log_gaussian_density(y, mu, S):
        d = len(y)
        diff = y - mu
        try:
            c = np.linalg.cholesky(S)
            alpha = np.linalg.solve(c.T, np.linalg.solve(c, diff))
            logdet = 2 * np.sum(np.log(np.diag(c)))
        except np.linalg.LinAlgError:
            alpha = np.linalg.solve(S, diff)
            logdet = np.log(np.linalg.det(S))

        return -0.5 * (diff @ alpha + logdet + d * np.log(2 * np.pi))

    # --------------------------------------------------
    def predict(self):
        for i in range(self.N):
            self.particles[i] = self.model.sample_transition(
                self.particles[i], self.theta
            )

    # --------------------------------------------------
    def update(self, y):
        logw_new = np.zeros(self.N)

        R = self.model.R(None, self.theta)

        for i in range(self.N):
            y_pred = self.model.h(self.particles[i], self.theta)
            logw_new[i] = self.logw[i] + self.log_gaussian_density(
                y, y_pred, R
            )

        # normalización estable
        logZ = logsumexp(logw_new)
        self.logw = logw_new - logZ
        self.log_likelihood += logZ

    # --------------------------------------------------
    def effective_sample_size(self):
        w = np.exp(self.logw)
        return 1.0 / np.sum(w**2)

    # --------------------------------------------------
    def resample(self):
        w = np.exp(self.logw)
        idx = np.random.choice(self.N, size=self.N, p=w)

        self.particles = self.particles[idx]
        self.logw[:] = -np.log(self.N)

    # --------------------------------------------------
    def estimate(self):
        w = np.exp(self.logw)

        mean = np.sum(self.particles * w[:, None], axis=0)

        diff = self.particles - mean
        cov = diff.T @ (diff * w[:, None])

        self.m = mean
        self.P = 0.5 * (cov + cov.T)

    # --------------------------------------------------
    def update_step(self, y):
        self.predict()
        self.update(y)

        if self.effective_sample_size() < self.resample_threshold * self.N:
            self.resample()

        self.estimate()

    #--------------------------------------------------

    def filter(self, Y):

        for y in Y:
            self.update_step(y)
            self._store()

        return {
            "filtering_mean": self.m,
            "filtering_cov": self.P,
            "particles": self.particles,
            "weights": np.exp(self.logw),
            "log_likelihood": self.log_likelihood,
            "history": {
                "mean": self.history_m,
                "cov": self.history_P,
                "log_likelihood": self.history_ll,
                "particles": self.history_particles,
                "weights": self.history_weights
            }
        }