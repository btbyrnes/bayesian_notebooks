from gibbs.distributions import Distribution
from gibbs.parameters import Parameter
import numpy as np


def sampler(mu:Distribution, tau:Distribution, y:np.ndarray, samples:int=2000, thinning:int=20) -> tuple:
    mu_ = Parameter(mu, samples)
    tau_ = Parameter(tau, samples)

    for i in range(1,samples):
        tau_.estimates[i] = tau.sample(mu_.estimates[i-1], y)
        mu_.estimates[i] = mu.sample(tau.estimates[i], y)

    mu_.thin_estimates(thinning)
    tau.thin_estimates(thinning)
    
    return mu_, tau_


def sampler_2(mu:Distribution, sigma:float, y:np.ndarray, samples:int=2000):
    mu_ = Parameter(mu, samples)
    for i in range(1, samples):
        mu_.estimates[i] = mu.sample(sigma, y)

    return mu_