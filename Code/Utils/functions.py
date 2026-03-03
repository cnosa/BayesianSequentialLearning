import numpy as np

def log_gaussian_density(x, mu, S):
    x = np.atleast_1d(x)
    mu = np.atleast_1d(mu)

    d = len(x)
    diff = x - mu

    S = 0.5 * (S + S.T)
    jitter = 1e-10
    S = S + jitter * np.eye(d)

    c = np.linalg.cholesky(S)
    alpha = np.linalg.solve(c.T, np.linalg.solve(c, diff))
    logdet = 2.0 * np.sum(np.log(np.diag(c)))

    return -0.5 * (diff @ alpha + logdet + d * np.log(2 * np.pi))