import numpy as np
from primitive.parameters import ParameterInterface


# Measurement noise objects:

class Noise(ParameterInterface):
    parameter_keys = None

    def get_parameter_values(self):
        parameters = super().get_parameter_values()
        # Remove the shape key as it will not be changed after initialisation.
        return {key: value for key, value in parameters.items() if key not in ["shape"]}

    def sample(self, t=None, n_particles=1):
        pass

    def __call__(self, t=None, n_particles=1):
        return self.sample(t=t, n_particles=1)
    
class GaussianNoise(Noise):
    parameter_keys = ["shape", "sigma_eps"]

    def covariance(self, t, n_particles=1):
        return self.sigma_eps**2 * np.eye(self.shape[-2])

    def sample(self, t=None, n_particles=1):
        return self.sigma_eps * np.random.randn(self.shape[0], self.shape[1])
    
class EnsembleGaussianNoise(Noise):
    parameter_keys = ["shape", "sigma_eps"]

    def covariance(self, t, n_particles=1):
        return self.sigma_eps**2 * np.broadcast_to(np.eye(self.shape[-2]), (n_particles, self.shape[-2], self.shape[-2]))

    def sample(self, t=None, n_particles=1):
        return self.sigma_eps * np.random.randn(n_particles, self.shape[0], self.shape[1])
    