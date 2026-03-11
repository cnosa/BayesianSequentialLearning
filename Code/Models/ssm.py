# ssm.py
import numpy as np
from Models.transition import TransitionModel
from Models.observation import ObservationModel
from Models.priors import PriorModel



class StateSpaceModel:

    """
    General discrete-time state space model:

        x_{k+1} ~ p(x_{k+1} | x_k, theta)
        y_k     ~ p(y_k | x_k, theta)
    """

    def __init__(self, transition, observation, prior):
        self.transition = transition
        self.observation = observation
        self.prior = prior
        if not hasattr(transition, "d"):
            raise ValueError("Transition model must define state dimension d")
        if not hasattr(observation, "m"):
            raise ValueError("Observation model must define observation dimension m")
        self.d = transition.d
        self.m = observation.m

    @staticmethod
    def _log_gaussian_density(x, mu, S):
        x = np.atleast_1d(x)
        mu = np.atleast_1d(mu)

        d = len(x)
        diff = x - mu

        S = 0.5 * (S + S.T)
        jitter = 1e-10
        S = S + jitter * np.eye(d)

        try:
            c = np.linalg.cholesky(S)
        except np.linalg.LinAlgError:
            S = S + 1e-8 * np.eye(d)
            c = np.linalg.cholesky(S)
        alpha = np.linalg.solve(c.T, np.linalg.solve(c, diff))
        logdet = 2.0 * np.sum(np.log(np.diag(c)))

        return -0.5 * (diff @ alpha + logdet + d * np.log(2 * np.pi))

    # wrapper methods for compatibility

    def f(self, x, theta=None):
        return self.transition.f(x, theta)

    def h(self, x, theta=None):
        return self.observation.h(x, theta)
    
    def f_x(self, x, theta=None):
        return self.transition.f_x(x, theta)

    def h_x(self, x, theta=None):
        return self.observation.h_x(x, theta)

    def Q(self, x=None, theta=None):
        if hasattr(self.transition, "_Q"):
            return self.transition._Q(x, theta)
        raise NotImplementedError("Covariance transition is not implemented")

    def R(self, x=None, theta=None):
        if hasattr(self.observation, "_R"):
            return self.observation._R(x, theta)
        raise NotImplementedError("Covariance observation is not implemented")

    def sample_transition(self, x, theta=None):
        return self.transition.sample(x, theta)

    def sample_observation(self, x, theta=None):
        return self.observation.sample(x, theta)

    def log_transition_density(self, x_next, x_prev, theta=None):
        return self.transition.log_density(x_next, x_prev, theta)

    def log_observation_density(self, y, x, theta=None):
        return self.observation.log_density(y, x, theta)

    def sample_prior(self, N=1, theta=None):
        return self.prior.sample(N, theta)

    def log_prior_density(self, x, theta=None):
        return self.prior.log_density(x, theta)
    


    def _validate_state(self, x):

        if not np.all(np.isfinite(x)):
            raise FloatingPointError("State contains NaN or Inf")

        if np.linalg.norm(x) > 1e6:
            raise FloatingPointError("State exploded numerically")

        return x
    

    def simulate(self, T, theta=None):

        """
        Simulate trajectory (X, Y) with numerical stability checks
        """

        X = np.zeros((T + 1, self.d))
        Y = np.zeros((T, self.m))

        x0 = self.sample_prior(N=1, theta=theta)
        x = np.atleast_1d(x0)[0]

        x = self._validate_state(x)

        X[0] = x

        for t in range(T):

            try:

                x = self.sample_transition(x, theta)

                x = self._validate_state(x)

                y = self.sample_observation(x, theta)

            except FloatingPointError:

                # restart trajectory if unstable
                return self.simulate(T, theta)

            X[t + 1] = x
            Y[t] = y

        return X, Y

