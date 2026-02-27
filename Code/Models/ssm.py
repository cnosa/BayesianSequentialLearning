import numpy as np

class StateSpaceModel:
    """
    Class for discrete space state models:
        x_{k+1} = f(x_{k}, theta) + q_{k+1}
        y_{k+1}     = h(x_{k+1}, theta) + r_{k+1}
        q_{k+1} ~ Normal(0, Q(theta))
        r_{k+1} ~ Normal(0, R(theta))
    """

    def __init__(self, d, m):
        self.d = d  # state dimension
        self.m = m  # observation dimension

    # Evolution and observation (deterministic)
    def f(self, x, theta=None):
        raise NotImplementedError

    def h(self, x, theta=None):
        raise NotImplementedError

    # Jacobians ( KF / EKF)
    def f_x(self, x, theta=None):
        raise NotImplementedError

    def h_x(self, x, theta=None):
        raise NotImplementedError

    # Covariances
    def Q(self, x=None, theta=None):
        raise NotImplementedError

    def R(self, x=None, theta=None):
        raise NotImplementedError

    # Prior
    def prior(self):
        """
        Returns (m0, P0) such that x0 ~ Normal(m0, P0)
        """
        raise NotImplementedError

    # Simulation
    def sample_process_noise(self, x=None, theta=None):
        return np.random.multivariate_normal(
            mean=np.zeros(self.d),
            cov=self.Q(x, theta)
        )

    def sample_observation_noise(self, x=None, theta=None):
        return np.random.multivariate_normal(
            mean=np.zeros(self.m),
            cov=self.R(x, theta)
        )

    def sample_transition(self, x, theta=None):
        return self.f(x, theta) + self.sample_process_noise(x, theta)

    def sample_observation(self, x, theta=None):
        return self.h(x, theta) + self.sample_observation_noise(x, theta)

    def simulate(self, T, theta=None):
        """
        Simulate a trajectory (X, Y)
        """
        X = np.zeros((T + 1, self.d))
        Y = np.zeros((T, self.m))

        m0, _ = self.prior()
        X[0] = m0

        for t in range(T):
            X[t + 1] = self.sample_transition(X[t], theta)
            Y[t] = self.sample_observation(X[t + 1], theta)

        return X, Y
    

import numpy as np