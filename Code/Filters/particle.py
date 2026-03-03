# particle.py
from Filters.bayesian import BayesianFilter
import numpy as np
from scipy import linalg
from scipy.special import logsumexp

class ParticleFilter(BayesianFilter):
    """
    Bootstrap Particle Filter for StateSpaceModel
    """

    def __init__(self, model, N, theta=None, resample_threshold=0.15):
        super().__init__(model, theta)

        self.N = N
        self.resample_threshold = resample_threshold

        self.d = model.d
        self.m_obs = model.m

        # inicialization of particules from prior
        self.particles = model.sample_prior(N, theta)


        self.logw = np.zeros(N) - np.log(N)

    # --------------------------------------------------
    def _store(self):
        self.history_m.append(self.m.copy())
        self.history_P.append(self.P.copy())
        self.history_ll.append(self.log_likelihood)

        self.history_particles.append(self.particles.copy())
        self.history_weights.append(np.exp(self.logw).copy())


    # --------------------------------------------------
    def predict(self):
        for i in range(self.N):
            self.particles[i] = self.model.sample_transition(
                self.particles[i], self.theta
            )

    # --------------------------------------------------
    def update(self, y):
        logw_new = np.zeros(self.N)

        for i in range(self.N):
            logw_new[i] = self.logw[i] + self.model.log_observation_density(y, self.particles[i], self.theta)

        # stable normalization
        logZ = logsumexp(logw_new)

        if not np.isfinite(logZ):
            # reset weights if logZ is not finite (e.g., all particles have zero likelihood)
            self.logw[:] = -np.log(self.N)
            return
        
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
        self.P += 1e-10 * np.eye(self.d)

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