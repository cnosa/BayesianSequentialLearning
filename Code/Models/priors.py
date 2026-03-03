# priors.py
import numpy as np
from scipy.special import gammaln
from Utils.functions import log_gaussian_density

class PriorModel:

    def sample(self, N=1, theta=None):
        raise NotImplementedError

    def log_density(self, x, theta=None):
        raise NotImplementedError


class GaussianPrior(PriorModel):

    def __init__(self, m0, P0):
        self.m0 = m0
        self.P0 = P0

    def sample(self, N=1, theta=None):
        return np.random.multivariate_normal(self.m0, self.P0, size=N)

    def log_density(self, x, theta=None):
        return log_gaussian_density(x, self.m0, self.P0)
    

class UniformPrior(PriorModel):

    def __init__(self, low, high):
        self.low = np.asarray(low)
        self.high = np.asarray(high)
        self.d = len(self.low)

    def sample(self, N=1, theta=None):
        return np.random.uniform(self.low, self.high, size=(N, self.d))

    def log_density(self, x, theta=None):
        x = np.atleast_1d(x)
        if np.all((x >= self.low) & (x <= self.high)):
            volume = np.prod(self.high - self.low)
            return -np.log(volume)
        return -np.inf
    

class LogNormalPrior(PriorModel):

    def __init__(self, mean, cov):
        self.mean = np.asarray(mean)
        self.cov = np.asarray(cov)

    def sample(self, N=1, theta=None):
        z = np.random.multivariate_normal(self.mean, self.cov, size=N)
        return np.exp(z)

    def log_density(self, x, theta=None):
        x = np.atleast_1d(x)
        if np.any(x <= 0):
            return -np.inf
        logx = np.log(x)
        log_gauss = log_gaussian_density(
            logx, self.mean, self.cov
        )
        return log_gauss - np.sum(np.log(x))
    

class GaussianMixturePrior(PriorModel):

    def __init__(self, weights, means, covs):
        self.weights = np.asarray(weights) / np.sum(weights)
        self.means = means
        self.covs = covs
        self.K = len(weights)

    def sample(self, N=1, theta=None):
        idx = np.random.choice(self.K, size=N, p=self.weights)
        samples = []
        for k in idx:
            samples.append(
                np.random.multivariate_normal(self.means[k], self.covs[k])
            )
        return np.array(samples)

    def log_density(self, x, theta=None):
        from scipy.special import logsumexp
        log_terms = []
        for w, m, S in zip(self.weights, self.means, self.covs):
            log_terms.append(
                np.log(w)
                + log_gaussian_density(x, m, S)
            )
        return logsumexp(log_terms)
    
from scipy.stats import multivariate_t

class StudentTPrior(PriorModel):

    def __init__(self, mean, shape, df):
        self.mean = mean
        self.shape = shape
        self.df = df

    def sample(self, N=1, theta=None):
        return multivariate_t.rvs(
            loc=self.mean,
            shape=self.shape,
            df=self.df,
            size=N
        )

    def log_density(self, x, theta=None):
        return multivariate_t.logpdf(
            x,
            loc=self.mean,
            shape=self.shape,
            df=self.df
        )
    
class ExponentialPrior(PriorModel):

    def __init__(self, rate):
        self.rate = rate  # lambda

    def sample(self, N=1, theta=None):
        samples = np.random.exponential(
            scale=1.0 / self.rate,
            size=(N, 1)
        )
        return samples

    def log_density(self, x, theta=None):
        x = np.atleast_1d(x)
        if np.any(x <= 0):
            return -np.inf
        return np.log(self.rate) - self.rate * x[0]
    


class GammaPrior(PriorModel):

    def __init__(self, shape, rate):
        self.shape = shape      # alpha
        self.rate = rate        # beta

    def sample(self, N=1, theta=None):
        samples = np.random.gamma(
            shape=self.shape,
            scale=1.0 / self.rate,
            size=(N, 1)
        )
        return samples

    def log_density(self, x, theta=None):
        x = np.atleast_1d(x)

        if np.any(x <= 0):
            return -np.inf

        alpha = self.shape
        beta = self.rate

        return (
            alpha * np.log(beta)
            - gammaln(alpha)
            + (alpha - 1) * np.log(x[0])
            - beta * x[0]
        )