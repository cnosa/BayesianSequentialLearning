# linear.py
import numpy as np
from Models.ssm import StateSpaceModel
from Models.transition import GaussianTransition
from Models.observation import GaussianObservation
from Models.priors import GaussianPrior


class LinearGaussianSSM(StateSpaceModel):

    def __init__(self, m0, P0, A, H, Sigma, Gamma):

        A = np.atleast_2d(A)
        H = np.atleast_2d(H)
        Sigma = np.atleast_2d(Sigma)
        Gamma = np.atleast_2d(Gamma)

        d = A.shape[0]
        m = H.shape[0]

        # ---------- Transition ----------
        def f(x, theta=None):
            return A @ x

        def f_x(x, theta=None):
            return A

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

        # ---------- Build SSM ----------
        super().__init__(transition, observation, prior)