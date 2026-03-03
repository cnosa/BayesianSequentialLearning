# observation.py
import numpy as np
from Utils.functions import log_gaussian_density
from scipy.special import gammaln
from scipy.stats import t


class ObservationModel:

    def __init__(self, m):
        self.m = m

    def h(self, x, theta=None):
        raise NotImplementedError

    def h_x(self, x, theta=None):
        raise NotImplementedError

    def sample(self, x, theta=None):
        raise NotImplementedError

    def log_density(self, y, x, theta=None):
        raise NotImplementedError
    
class GaussianObservation(ObservationModel):

    def __init__(self, h, R_func, m, h_x=None):
        super().__init__(m)
        self._h = h
        self._R = R_func
        self._h_x = h_x

    def h(self, x, theta=None):
        return self._h(x, theta)
    
    def h_x(self, x, theta=None):
        if self._h_x is None:
            raise NotImplementedError("Jacobian not provided")
        return self._h_x(x, theta)

    def sample(self, x, theta=None):
        mean = self._h(x, theta)
        R = self._R(x, theta)
        noise = np.random.multivariate_normal(
            mean=np.zeros(self.m),
            cov=R
        )
        return mean + noise

    def log_density(self, y, x, theta=None):
        mean = self._h(x, theta)
        R = self._R(x, theta)
        return log_gaussian_density(y, mean, R)
    


class PoissonObservation(ObservationModel):

    def __init__(self, rate_func, m=1):
        super().__init__(m)
        self._rate = rate_func  # lambda(x)

    def h(self, x, theta=None):
        return np.array([self._rate(x, theta)])

    def sample(self, x, theta=None):
        lam = self._rate(x, theta)
        return np.random.poisson(lam, size=self.m)

    def log_density(self, y, x, theta=None):
        lam = self._rate(x, theta)
        y = y[0]
        if lam <= 0:
            return -np.inf
        return y * np.log(lam) - lam - gammaln(y + 1)
    

class BernoulliObservation(ObservationModel):

    def __init__(self, prob_func):
        super().__init__(m=1)
        self._prob = prob_func

    def sample(self, x, theta=None):
        p = self._prob(x, theta)
        return np.array([np.random.binomial(1, p)])

    def log_density(self, y, x, theta=None):
        p = self._prob(x, theta)
        y_val = y[0]
        if y_val == 1:
            return np.log(p)
        return np.log(1 - p)
    


class StudentTObservation(ObservationModel):

    def __init__(self, h, scale, df):
        super().__init__(m=1)
        self._h = h
        self.scale = scale
        self.df = df

    def sample(self, x, theta=None):
        mean = self._h(x, theta)
        noise = t.rvs(df=self.df, scale=self.scale)
        return np.array([mean + noise])

    def log_density(self, y, x, theta=None):
        mean = self._h(x, theta)
        return t.logpdf(
            y[0] - mean,
            df=self.df,
            scale=self.scale
        )
    
class GammaObservation(ObservationModel):

    def __init__(self, shape, rate_func):
        """
        shape: alpha > 0
        rate_func: beta(x) > 0
        """
        super().__init__(m=1)
        self.shape = shape
        self._rate = rate_func

    def h(self, x, theta=None):
        # media = alpha / beta
        beta = self._rate(x, theta)
        return np.array([self.shape / beta])

    def sample(self, x, theta=None):
        beta = self._rate(x, theta)
        return np.random.gamma(
            shape=self.shape,
            scale=1.0 / beta,
            size=1
        )

    def log_density(self, y, x, theta=None):
        y = y[0]
        if y <= 0:
            return -np.inf

        alpha = self.shape
        beta = self._rate(x, theta)

        return (
            alpha * np.log(beta)
            - gammaln(alpha)
            + (alpha - 1) * np.log(y)
            - beta * y
        )
    

class LogNormalObservation(ObservationModel):

    def __init__(self, h, sigma, h_x=None):
        super().__init__(m=1)
        self._h = h
        self.sigma = sigma
        self._h_x = h_x

    def h(self, x, theta=None):
        return np.array([np.exp(self._h(x, theta))])

    def h_x(self, x, theta=None):
        if self._h_x is None:
            raise NotImplementedError("Jacobian not provided")
        base = self._h(x, theta)
        return np.exp(base) * self._h_x(x, theta)

    def sample(self, x, theta=None):
        mean_log = self._h(x, theta)
        noise = np.random.normal(0, self.sigma)
        return np.array([np.exp(mean_log + noise)])

    def log_density(self, y, x, theta=None):
        y = y[0]
        if y <= 0:
            return -np.inf

        mean_log = self._h(x, theta)

        return (
            -np.log(y)
            -0.5 * np.log(2 * np.pi * self.sigma**2)
            - ((np.log(y) - mean_log)**2) / (2 * self.sigma**2)
        )
    
class MultiplicativeGaussianObservation(ObservationModel):

    def __init__(self, h, sigma, h_x=None):
        super().__init__(m=1)
        self._h = h
        self.sigma = sigma
        self._h_x = h_x

    def h(self, x, theta=None):
        return np.array([self._h(x, theta)])

    def h_x(self, x, theta=None):
        if self._h_x is None:
            raise NotImplementedError("Jacobian not provided")
        return self._h_x(x, theta)

    def sample(self, x, theta=None):
        mean = self._h(x, theta)
        noise = np.random.normal(0, self.sigma)
        return np.array([mean * (1 + noise)])

    def log_density(self, y, x, theta=None):
        mean = self._h(x, theta)
        var = (self.sigma * mean)**2

        return log_gaussian_density(
            y,
            np.array([mean]),
            np.array([[var]])
        )
    
class NonlinearProductObservation(ObservationModel):

    def __init__(self, sigma):
        super().__init__(m=1)
        self.sigma = sigma

    def h(self, x, theta=None):
        return np.array([x[0] * x[1]])

    def h_x(self, x, theta=None):
        return np.array([[x[1], x[0]]])

    def sample(self, x, theta=None):
        mean = x[0] * x[1]
        noise = np.random.normal(0, self.sigma)
        return np.array([mean + noise])

    def log_density(self, y, x, theta=None):
        mean = x[0] * x[1]
        return log_gaussian_density(
            y,
            np.array([mean]),
            np.array([[self.sigma**2]])
        )