from abc import ABC
import scipy.stats, numpy as np
from sklearn.preprocessing import scale


"""
These are equivalent

sns.distplot( 1 / scipy.stats.invgamma.rvs(a=5, scale=10, size=1000), label="Inverse Gamma")
sns.distplot(scipy.stats.gamma.rvs(a=5, scale=1/10, size=1000))
"""

class Distribution(ABC):
    @classmethod
    def sample():
        pass

    def init_estimates(size:int=1000):
        pass


class Normal(Distribution):
    def __init__(self, mu_prior:float, sigma_prior:float):
        self.mu_prior = mu_prior
        self.sigma_prior = sigma_prior
        self.name = "Normal"


    def sample(self, sigma, y:np.ndarray):
        mu_prior = self.mu_prior
        sigma_prior = self.sigma_prior

        y_bar = np.mean(y)
        n = len(y)
        
        # print("mu sample:    ", ((n * y_bar / (sigma*sigma)) + (mu_prior / sigma_prior*sigma_prior)) / ((n*sigma) + (1/sigma_prior)))
        # print("sigma sample: ", np.sqrt((1 / ((n/(sigma*sigma)) + (1/(sigma_prior*sigma_prior))))))

        mu_sample = ((n * y_bar / (sigma*sigma)) + (mu_prior / sigma_prior*sigma_prior)) / ((n/(sigma*sigma)) + (1/(sigma_prior*sigma_prior)))
        sigma_sample = np.sqrt(1 / ((n/(sigma*sigma)) + (1/(sigma_prior*sigma_prior))))
        
        return scipy.stats.norm.rvs(loc=mu_sample, scale=sigma_sample, size=1)[0]


    def init_estimates(self, size=1000):
        return np.zeros(size) + self.mu_prior


    def __repr__(self) -> str:
        return f"{self.name}({self.mu_prior}, {self.sigma_prior}^2)"


class Gamma(Distribution):
    def __init__(self, shape_prior:float, beta_prior:float=None, scale_prior:float=None):
        self.shape_prior = shape_prior
        if beta_prior is not None: self.scale_prior = 1 / beta_prior
        elif self.scale_prior is not None: self.scale_prior = scale_prior
        self.name = "Gamma"


    def sample(self, mu:float=0.0, y:np.ndarray=0.0):
        shape_prior = self.shape_prior
        scale_prior = self.scale_prior

        n = len(y)

        shape_sample = shape_prior + (n / 2)
        scale_sample = 1 / ((1 / scale_prior) + (np.sum( ( y - mu)**2 ) / 2))
    
        return np.sqrt(scipy.stats.gamma.rvs(a=shape_sample, scale=scale_sample, size=1))[0]


    def init_estimates(self, size=1000):
        return np.ones(size) * self.shape_prior


    def __repr__(self) -> str:
        return f"{self.name}({self.shape_prior}, {self.scale_prior})"


class InvGamma(Gamma):
    def __init__(self, shape_prior:float, beta_prior:float=None, scale_prior:float=None):
        super().__init__(shape_prior, beta_prior, scale_prior)
        self.name = "InverseGamma"


    def sample(self, mu:float=0.0, y:np.ndarray=0.0):
        return 1 / super().sample(mu, y)


class InverseGamma_old(Distribution):
    def __init__(self, shape_prior=2.0, scale_prior=1.0, beta_prior=1.0):
        self.shape_prior = shape_prior
        self.scale_prior = scale_prior
        self.name = "InverseGamma"


    def sample(self, mu:float=0.0, y:np.ndarray=0.0):
        shape_prior = self.shape_prior
        scale_prior = self.scale_prior

        n = len(y)

        shape_sample = shape_prior + (n / 2)
        scale_sample = scale_prior + (np.sum( ( y - mu)**2 ) / 2)
    
        return np.sqrt(scipy.stats.invgamma.rvs(a=shape_sample, scale=scale_sample, size=1))[0]


    def init_estimates(self, size=1000):
        return np.ones(size)


    def __repr__(self) -> str:
        return f"{self.name}({self.shape_prior}, {self.scale_prior})"