from abc import ABC
from ast import Param
import scipy.stats, numpy as np
from distributions import Normal, InverseGamma, Distribution
from parameters import Parameter

# Using phi[i-1] draw theta[i] from p(theta | phi = phi[i-1], y)
# Using theta[i] draw phi[i] from p(phi | theta = theta[i], y)

# Normal likelihood, unknown mean (theta) uknown variance (phi)

def sampler(mu:Distribution, sigma:Distribution, y:np.ndarray, samples:int=2000) -> tuple:
    mu_ = Parameter(mu, samples)
    sigma_ = Parameter(sigma, samples)

    for i in range(1,samples):
        sigma_.estimates[i] = sigma.update(mu_.estimates[i-1], y)
        mu_.estimates[i] = mu.update(sigma_.estimates[i], y)

    return mu_, sigma_





class GibbsSampler:
    def __init__(self, parameters:list):
        self.parameters = parameters
        self.n_parameters = len(parameters)

    def sample(self, n=1000):
        parameters = self.parameters

        for i in range(1,1000):
            pass
        pass      


if __name__ == "__main__":
    y = np.array([1.2, 1.4, -0.5, 0.3, 0.9, 2.3, 1.0, 0.1, 1.3, 1.9])
    # mu, sigma = sampler(y)
    print(np.mean(y), np.std(y))
    # print(np.mean(mu), np.mean(sigma))