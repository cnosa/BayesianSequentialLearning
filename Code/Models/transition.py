# transition.py
import numpy as np
from Utils.functions import log_gaussian_density


class TransitionModel:

    def __init__(self, d):
        self.d = d

    # deterministic part (optional for UKF)
    def f(self, x, theta=None):
        raise NotImplementedError

    def f_x(self, x, theta=None):
        raise NotImplementedError

    # probabilistic part
    def sample(self, x, theta=None):
        raise NotImplementedError

    def log_density(self, x_next, x_prev, theta=None):
        raise NotImplementedError


class GaussianTransition(TransitionModel):

    def __init__(self, f, Q_func, d, f_x=None):
        super().__init__(d)
        self._f = f
        self._Q = Q_func
        self._f_x = f_x

    def f(self, x, theta=None):
        return self._f(x, theta)
    
    def f_x(self, x, theta=None):
        if self._f_x is None:
            raise NotImplementedError("Jacobian not provided")
        return self._f_x(x, theta)

    def sample(self, x, theta=None):
        mean = self._f(x, theta)
        Q = self._Q(x, theta)
        noise = np.random.multivariate_normal(
            mean=np.zeros(self.d),
            cov=Q
        )
        return mean + noise

    def log_density(self, x_next, x_prev, theta=None):
        mean = self._f(x_prev, theta)
        Q = self._Q(x_prev, theta)
        return log_gaussian_density(x_next, mean, Q)


class LogNormalTransition(TransitionModel):

    def __init__(self, f, sigma, d, f_x=None):
        super().__init__(d)
        self._f = f
        self.sigma = sigma
        self._f_x = f_x

    def f(self, x, theta=None):
        return self._f(x, theta)

    def f_x(self, x, theta=None):
        if self._f_x is None:
            raise NotImplementedError
        return self._f_x(x, theta)

    def sample(self, x, theta=None):
        mean_det = self._f(x, theta)
        eta = np.random.normal(0.0, self.sigma, size=self.d)
        return mean_det * np.exp(eta)

    def log_density(self, x_next, x_prev, theta=None):
        mean_det = self._f(x_prev, theta)

        if np.any(x_next <= 0):
            return -np.inf

        log_ratio = np.log(x_next / mean_det)

        return (
            -np.sum(np.log(x_next))
            - self.d * 0.5 * np.log(2 * np.pi * self.sigma**2)
            - np.sum(log_ratio**2) / (2 * self.sigma**2)
        )
    

from scipy.stats import t

class StudentTTransition(TransitionModel):

    def __init__(self, f, scale, df, d, f_x=None):
        super().__init__(d)
        self._f = f
        self.scale = scale
        self.df = df
        self._f_x = f_x

    def f(self, x, theta=None):
        return self._f(x, theta)

    def f_x(self, x, theta=None):
        if self._f_x is None:
            raise NotImplementedError
        return self._f_x(x, theta)

    def sample(self, x, theta=None):
        mean = self._f(x, theta)
        noise = t.rvs(df=self.df, scale=self.scale, size=self.d)
        return mean + noise

    def log_density(self, x_next, x_prev, theta=None):
        mean = self._f(x_prev, theta)
        return np.sum(
            t.logpdf(x_next - mean, df=self.df, scale=self.scale)
        )
    
from scipy.special import gammaln

class GammaTransition(TransitionModel):

    def __init__(self, shape_func, rate_func, d=1):
        super().__init__(d)
        self._shape = shape_func
        self._rate = rate_func

    def sample(self, x, theta=None):
        alpha = self._shape(x, theta)
        beta = self._rate(x, theta)
        return np.random.gamma(alpha, 1.0 / beta, size=self.d)

    def log_density(self, x_next, x_prev, theta=None):
        if np.any(x_next <= 0):
            return -np.inf

        alpha = self._shape(x_prev, theta)
        beta = self._rate(x_prev, theta)

        return (
            alpha * np.log(beta)
            - gammaln(alpha)
            + (alpha - 1) * np.log(x_next[0])
            - beta * x_next[0]
        )
    

class GaussianMixtureTransition(TransitionModel):

    def __init__(self, f, Q1, Q2, d, weight=0.5):
        super().__init__(d)
        self._f = f
        self.Q1 = Q1
        self.Q2 = Q2
        self.weight = weight

    def sample(self, x, theta=None):
        mean = self._f(x, theta)
        if np.random.rand() < self.weight:
            noise = np.random.multivariate_normal(
                np.zeros(self.d), self.Q1
            )
        else:
            noise = np.random.multivariate_normal(
                np.zeros(self.d), self.Q2
            )
        return mean + noise

    def log_density(self, x_next, x_prev, theta=None):
        mean = self._f(x_prev, theta)

        log1 = log_gaussian_density(
            x_next, mean, self.Q1
        )
        log2 = log_gaussian_density(
            x_next, mean, self.Q2
        )

        # log-sum-exp
        m = max(log1, log2)
        return m + np.log(
            self.weight * np.exp(log1 - m)
            + (1 - self.weight) * np.exp(log2 - m)
        )
    
class HeteroscedasticGaussianTransition(TransitionModel):

    def __init__(self, f, sigma, d):
        super().__init__(d)
        self._f = f
        self.sigma = sigma

    def sample(self, x, theta=None):
        mean = self._f(x, theta)
        var = (self.sigma * np.abs(mean))**2
        noise = np.random.normal(0, np.sqrt(var), size=self.d)
        return mean + noise

    def log_density(self, x_next, x_prev, theta=None):
        mean = self._f(x_prev, theta)
        var = (self.sigma * np.abs(mean))**2
        return log_gaussian_density(
            x_next,
            mean,
            np.diag(var)
        )