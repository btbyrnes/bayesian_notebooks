from abc import ABC
from gibbs.distributions import Distribution

class Parameter(ABC):
    def __init__(self, distribution:Distribution, size=1000):
        self.distribution = distribution
        self.estimates = self.distribution.init_estimates(size)

    def update_estimate(self, value, i:int):
        self.estimates[i] = value

    def __repr__(self) -> str:
        return str(self.estimates)

    def thin_estimates(self, thinning=10):
        self.estimates = self.estimates[0::thinning]
    
    
class Parameters:
    def __init__(self, parameters:list):
        self.parameters = parameters
    
