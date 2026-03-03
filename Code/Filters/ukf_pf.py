# ukf_pf.py
from Filters.bayesian import BayesianFilter
from Filters.ukf import UnscentedKalmanFilter
from Filters.particle import ParticleFilter
import numpy as np
from scipy import linalg
from scipy.special import logsumexp


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
        alpha=0.1,
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

        self.m_minus = self.ukf.m_minus.copy()
        self.P_minus = self.ukf.P_minus.copy()

    # --------------------------------------------------
    def update(self, y):

        ll_before = self.ukf.log_likelihood

        self.t += 1

        if self.t % self.L == 0:
            # use PF instead UKF
            self._pf_correction(y, ll_before)

        else:
            # use only UKF
            self.ukf.update(y)

            self.m = self.ukf.m.copy()
            self.P = self.ukf.P.copy()
            self.log_likelihood = self.ukf.log_likelihood

    # --------------------------------------------------
    def _pf_correction(self, y_last, ll_before):
        """
        Run a short PF initialized from the current Gaussian posterior.
        """

        pf = ParticleFilter(
            self.model,
            N=self.N,
            theta=self.theta,
            resample_threshold=self.pf_resample_threshold,
        )

        P_minus = 0.5 * (self.P_minus + self.P_minus.T)
        P_minus += 1e-10 * np.eye(P_minus.shape[0])  

        pf.particles = np.random.multivariate_normal(
            mean=self.m_minus,
            cov=P_minus,
            size=self.N,
        )
        pf.logw[:] = -np.log(self.N)

        pf.update(y_last)
        pf.estimate()
        
        ll_pf_increment = pf.log_likelihood
        self.log_likelihood = ll_before + ll_pf_increment

        # project PF → Gaussian
        self.m = pf.m.copy()
        self.P = 0.5 * (pf.P + pf.P.T)
        self.P += 1e-10 * np.eye(self.P.shape[0])

        # restart UKF with the corrected posterior
        self.ukf.m = self.m.copy()
        self.ukf.P = self.P.copy()
        self.ukf.log_likelihood = self.log_likelihood

        # history storage
        self.history_particles[-1] = pf.particles.copy()
        self.history_weights[-1] = np.exp(pf.logw).copy()
