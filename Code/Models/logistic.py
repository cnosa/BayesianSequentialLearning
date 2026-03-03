# lorenz63.py
import numpy as np
from Models.ssm import StateSpaceModel
from Models.transition import GaussianTransition
from Models.observation import GaussianObservation
from Models.priors import GaussianPrior



class LogisticGrowthSSM(StateSpaceModel):
    """
    Discrete-time Logistic Growth model via Euler discretization.
    State dimension: 1
    """

    def __init__(
        self,
        r=1.0,              # growth rate
        K=10.0,             # carrying capacity
        dt=0.01,
        m0=np.array([1.0]),
        P0=np.array([[0.1]]),
        Sigma=np.array([[1e-4]]),
        Gamma=np.array([[1e-2]]),
        H=None,
    ):

        d = 1
        H = np.array([[1.0]]) if H is None else np.atleast_2d(H)
        m = H.shape[0]

        Sigma = np.atleast_2d(Sigma)
        Gamma = np.atleast_2d(Gamma)

        # ---------- Transition ----------
        def f(x, theta=None):
            x_val = x[0]
            dx = r * x_val * (1.0 - x_val / K)
            return np.array([x_val + dt * dx])

        def f_x(x, theta=None):
            x_val = x[0]
            derivative = r * (1.0 - 2.0 * x_val / K)
            return np.array([[1.0 + dt * derivative]])

        def Q_func(x, theta=None):
            return Sigma

        transition = GaussianTransition(
            f=f,
            Q_func=Q_func,
            d=d,
            f_x=f_x
        )

        # ---------- Observation ----------
        def h(x, theta=None):
            return H @ x

        def h_x(x, theta=None):
            return H

        def R_func(x, theta=None):
            return Gamma

        observation = GaussianObservation(
            h=h,
            R_func=R_func,
            m=m,
            h_x=h_x
        )

        # ---------- Prior ----------
        prior = GaussianPrior(
            m0=np.atleast_1d(m0),
            P0=np.atleast_2d(P0)
        )

        super().__init__(transition, observation, prior)



from Models.transition import TransitionModel
from Models.observation import ObservationModel, PoissonObservation
from Models.priors import GammaPrior
from scipy.special import gammaln


class MultiplicativeLogisticTransition(TransitionModel):

    def __init__(self, r, K, dt, sigma_q):
        super().__init__(d=1)
        self.r = r
        self.K = K
        self.dt = dt
        self.sigma_q = sigma_q

    # deterministic part (for UKF mean propagation)
    def f(self, x, theta=None):
        x_val = x[0]
        dx = self.r * x_val * (1.0 - x_val / self.K)
        return np.array([x_val + self.dt * dx])

    def f_x(self, x, theta=None):
        x_val = x[0]
        derivative = self.r * (1.0 - 2.0 * x_val / self.K)
        return np.array([[1.0 + self.dt * derivative]])

    # multiplicative lognormal noise
    def sample(self, x, theta=None):
        mean_det = self.f(x)
        eta = np.random.normal(0.0, self.sigma_q)
        return mean_det * np.exp(eta)
    
    def Q(self, x=None, theta=None):
        mean_det = self.f(x)
        var = (mean_det[0]**2) * (np.exp(self.sigma_q**2) - 1)
        return np.array([[var]])

    def log_density(self, x_next, x_prev, theta=None):
        mean_det = self.f(x_prev)[0]

        if x_next[0] <= 0:
            return -np.inf

        # lognormal density
        log_ratio = np.log(x_next[0] / mean_det)

        return (
            - np.log(x_next[0])
            - 0.5 * np.log(2 * np.pi * self.sigma_q**2)
            - (log_ratio**2) / (2 * self.sigma_q**2)
        )
    
class LogisticGammaPoissonSSM(StateSpaceModel):

    def __init__(
        self,
        r=1.0,
        K=20.0,
        dt=0.1,
        sigma_q=0.2,
        alpha=2.0,
        beta=1.0,
    ):

        transition = MultiplicativeLogisticTransition(
            r=r,
            K=K,
            dt=dt,
            sigma_q=sigma_q
        )

        def rate_func(x, theta=None):
            return max(x[0], 1e-8) 

        observation = PoissonObservation(rate_func=rate_func)

        prior = GammaPrior(
            shape=alpha,
            rate=beta
        )

        super().__init__(
            transition=transition,
            observation=observation,
            prior=prior
        )