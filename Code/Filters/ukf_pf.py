from Filters.bayesian import BayesianFilter
from Filters.ukf import UnscentedKalmanFilter
from Filters.particle import ParticleFilter
import numpy as np
from scipy import linalg


class UKF_PF(BayesianFilter):
    """
    UKF with periodic PF-based correction.
    """

    def __init__(
        self,
        model,
        L,
        N,
        theta=None,
        alpha=1e-3,
        beta=2.0,
        kappa=0.0,
        pf_resample_threshold=0.15,
    ):
        super().__init__(model, theta)

        self.L = L
        self.N = N
        self.t = 0  # global time step

        # UKF 
        self.ukf = UnscentedKalmanFilter(
            model,
            theta=theta,
            alpha=alpha,
            beta=beta,
            kappa=kappa,
        )

        # synchronize UKF
        self.ukf.m = self.m.copy()
        self.ukf.P = self.P.copy()

        self.pf_resample_threshold = pf_resample_threshold

    # --------------------------------------------------
    def predict(self):
        self.ukf.predict()

    # --------------------------------------------------
    def update(self, y):
        self.ukf.update(y)

        # UKF
        self.m = self.ukf.m.copy()
        self.P = self.ukf.P.copy()
        self.log_likelihood = self.ukf.log_likelihood

        self.t += 1

        # checkpoint
        if self.t % self.L == 0:
            self._pf_correction(y)

    # --------------------------------------------------
    def _pf_correction(self, y_last):
        """
        Run a short PF initialized from the current Gaussian posterior.
        """

        pf = ParticleFilter(
            self.model,
            N=self.N,
            theta=self.theta,
            resample_threshold=self.pf_resample_threshold,
        )

        pf.particles = np.random.multivariate_normal(
            mean=self.m,
            cov=self.P,
            size=self.N,
        )
        pf.logw[:] = -np.log(self.N)

        pf.update(y_last)
        pf.estimate()

        # project PF → Gaussian
        self.m = pf.m.copy()
        self.P = pf.P.copy()

        # restart UKF with the corrected posterior
        self.ukf.m = self.m.copy()
        self.ukf.P = self.P.copy()

        self.history_particles[-1] = pf.particles.copy()
        self.history_weights[-1] = np.exp(pf.logw).copy()
