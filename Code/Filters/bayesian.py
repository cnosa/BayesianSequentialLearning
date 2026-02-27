import numpy as np
import matplotlib.pyplot as plt
np.random.seed(123)



class BayesianFilter:
    """
    Bayesian filters in state space models.
    """

    def __init__(self, model, theta=None):
        self.model = model
        self.theta = theta

        self.m, self.P = model.prior()
        self.log_likelihood = 0.0

        self.history_m = []
        self.history_P = []
        self.history_ll = []
        self.history_particles = []
        self.history_weights = []


        

    def _store(self):
        self.history_m.append(None if self.m is None else self.m.copy())
        self.history_P.append(None if self.P is None else self.P.copy())
        self.history_ll.append(self.log_likelihood)

        self.history_particles.append(None)
        self.history_weights.append(None)


    def predict(self):
        raise NotImplementedError

    def update(self, y):
        raise NotImplementedError

    def filter(self, Y):
        self._store()

        for y in Y:
            self.predict()
            self.update(y)
            self._store()

        return {
            "filtering_mean": self.m,
            "filtering_cov": self.P,
            "particles": None,
            "weights": None,
            "log_likelihood": self.log_likelihood,
            "history": {
                "mean": self.history_m,
                "cov": self.history_P,
                "log_likelihood": self.history_ll,
                "particles": self.history_particles,
                "weights": self.history_weights,
            }
        }
    
    # History 
    def get_means(self):
        return np.array(self.history_m)

    def get_covariances(self):
        return np.array(self.history_P)

    def get_loglikelihood(self):
        return np.array(self.history_ll)
    
    def plot_state(self, dim=0, X_true=None, sigma=2.0):
        """
        Plot the component dim of the filter state.
        """
        m = self.get_means()
        P = self.get_covariances()

        t = np.arange(len(m))
        mean = m[:, dim]
        std = np.sqrt(P[:, dim, dim])

        plt.figure()
        plt.plot(t, mean, label="Filtered")
        plt.fill_between(
            t,
            mean - sigma * std,
            mean + sigma * std,
            alpha=0.3,
            label=f"±{sigma}σ"
        )

        if X_true is not None:
            plt.plot(t, X_true[1:, dim], "--", label="True")

        plt.xlabel("Time")
        plt.ylabel(f"State x[{dim}]")
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def plot_variance(self, dim=0):
        P = self.get_covariances()
        t = np.arange(len(P))

        plt.figure()
        plt.plot(t, P[:, dim, dim])
        plt.xlabel("Time")
        plt.ylabel(f"Var(x[{dim}])")
        plt.grid(True)
        plt.show()
    
    def plot_loglikelihood(self):
        ll = self.get_loglikelihood()

        plt.figure()
        plt.plot(ll)
        plt.xlabel("Time")
        plt.ylabel("Cumulative log-likelihood")
        plt.grid(True)
        plt.show()

    # Metrics

    def rmse(self, X_true):
        """
        RMSE per time.
        """
        m = self.get_means()
        err = m - X_true[1:]
        return np.sqrt(np.mean(err**2, axis=1))
    
    def rmse_total(self, X_true):
        rmse_t = self.rmse(X_true)
        return np.mean(rmse_t)
    
    def mae(self, X_true):
        m = self.get_means()    
        return np.mean(np.abs(m - X_true[1:]), axis=1)

    def negative_log_likelihood(self):
        return -self.get_loglikelihood()
  
    def effective_sample_size(self):
        """
        ESS per time (only for PF).
        """
        ess = []

        for w in self.history_weights:
            if w is None:
                ess.append(np.nan)
            else:
                ess.append(1.0 / np.sum(w**2))

        return np.array(ess)
    
    def plot_ess(self):
        ess = self.effective_sample_size()

        plt.figure()
        plt.plot(ess)
        plt.xlabel("Tiempo")
        plt.ylabel("ESS")
        plt.grid(True)
        plt.show()