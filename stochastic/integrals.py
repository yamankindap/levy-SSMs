import numpy as np

from primitive.parameters import ParameterInterface

# Base stochastic integral object.

class StochasticIntegral(ParameterInterface):
    """The StochasticIntegral class provides the basic template of a forcing (driving) function in a stochastic differential equation (SDE).

    A StochasticIntegral can be interpreted as a random variable generator whose scale is defined through a time interval (s, t). 
    Random samples with specified shapes can be generated by calling the instantiated object with keyword arguments 's', 't' and 'n_particles'. 
    
    By convention random points must be numpy arrays with shape (D, 1) or (Np, D, 1) where D is the state dimension and Np is the number of particles.

    The (possibly random) moments associated with a time interval (s, t) can be accessed. In this case, the mean should have shape (D, 1) or (Np, D, 1),
    and the covariance has shape (D, D) or (Np, D, D).
    """
    parameter_keys = None

    def get_parameter_values(self):
        """A StochasticIntegral is a parameterised class but the shape parameter is excluded from the resulting dictionary since it is assumed to not change
        during sampling procedures. This may not be general enough, e.g. reversible-jump MCMC.
        """
        parameters = super().get_parameter_values()
        # Remove the shape key as it will not be changed after initialisation.
        return {key: value for key, value in parameters.items() if key not in ["shape"]}

    def sample(self, s=None, t=None, n_particles=1):
        pass

    def __call__(self, s=None, t=None, n_particles=1):
        return self.sample(s=s, t=t, n_particles=n_particles)
    

# Brownian motion classes.
    
class BrownianMotion(StochasticIntegral):
    """The BrownianMotion class provides functionality for sampling a 1-dimensional zero mean Gaussian random variable approximating the change in a Brownian motion
    in a time interval (s, t).

    The Gaussian random variable must by convention have shape (D, 1). All dimensions are assumed to be independent but all particles are identical.

    Additionally, the mean and covariance associated with a Brownian motion iteration may be accessed. These objects have shapes (D, 1) and (D, D).

    Both the shape and scale parameter 'sigma' are considered as parameters of the object. 
    """
    parameter_keys = ["shape", "sigma"]

    def moments(self, s, t, n_particles=1):
        """Returns the means and covariances associated with a time interval (s,t) for 'n_particles' independent particles. 
        These objects have shapes (D, 1) and (D, D).
        """
        means = np.zeros((self.shape[0], self.shape[1]))
        covs = (t - s) * self.sigma**2 * np.eye(self.shape[0])
        return means, covs

    def sample(self, s, t, n_particles=1):
        """Generates random samples from a Gaussian with independent dimensions with a scale proportional to the length of an interval (s, t).
        The shape of the random point by convention is (D, 1).

        DEVNOTE: This function does not implement any further discretisation of the interval (s, t) and simply generates a single independent homoscedastic Gaussian
        as the driving noise term for each particle. There may be an alternative sample function that discretises (s,t) further for increased definition.
        """
        dW = np.sqrt(t - s) * self.sigma * np.random.randn(self.shape[0], self.shape[1])
        return dW
    
class EnsembleBrownianMotion(StochasticIntegral):
    """The EnsembleBrownianMotion class provides functionality for sampling an ensemble of zero mean Gaussian random variables approximating the change in a Brownian motion
    in a time interval (s, t) with fixed scale parameter 'sigma'.

    The parameter 'sigma' can be a scalar value or it can have shape (Np, 1, 1) which represents Np independent Brownian motions with fixed scale.

    The Gaussian random variable must by convention have shape (Np, D, 1). All dimensions and particles are assumed to be independent.

    Additionally, the mean and covariance associated with a Brownian motion iteration may be accessed. These objects have shapes (Np, D, 1) and (Np, D, D).

    Both the shape and scale parameter 'sigma' are considered as parameters of the object. 
    """
    parameter_keys = ["shape", "sigma"]

    def moments(self, s, t, n_particles=1):
        """Returns the means and covariances associated with a time interval (s,t) for 'n_particles' independent particles. 
        These objects have shapes (Np, D, 1) and (Np, D, D).
        """
        means = np.zeros((n_particles, self.shape[0], self.shape[1]))
        covs = (t - s) * self.sigma**2 * np.broadcast_to(np.eye(self.shape[0]), (n_particles, self.shape[0], self.shape[0]))
        return means, covs

    def sample(self, s, t, n_particles=1):
        """Generates random samples from a Gaussian with independent dimensions with a scale proportional to the length of an interval (s, t).
        The shape of the random point by convention is (Np, D, 1).

        DEVNOTE: This function does not implement any further discretisation of the interval (s, t) and simply generates a single independent homoscedastic Gaussian
        as the driving noise term for each particle. There may be an alternative sample function that discretises (s,t) further for increased definition.
        """
        dW = np.sqrt(t - s) * self.sigma * np.random.randn(n_particles, self.shape[0], self.shape[1])
        return dW
    

# Brownian motion driven SDE classes.

class BrownianMotionDrivenIntegral(BrownianMotion):
    parameter_keys = ["shape", "sigma"]

    def __init__(self, **kwargs):
        # Set variable parameters using the ParameterInterface class.
        super().__init__(**kwargs)

        # Set covariance matrix associated with the integral directly.
        self.Q = kwargs.get("Q", None)
        self.L = kwargs.get("L", None)

        if self.Q is None:
            self.sample = self.sample_euler
        else:
            self.sample = self.sample_analytical

    def moments(self, s, t, n_particles=1):
        """Returns the means and covariances associated with a time interval (s,t). 
        These objects have shapes (D, 1) and (D, D).
        """
        means = np.zeros((self.shape[0], self.shape[1]))
        covs = self.sigma**2 * self.Q(t-s)
        return means, covs

    def sample_analytical(self, s, t, n_particles=1):
        """Generates random samples from a Gaussian defined by a (D, D) covariance matrix Q with scale proportional to the length of an interval (s, t).
        The shape of the random point by convention is (D, 1).
        """
        dW = np.random.multivariate_normal(mean=np.zeros((self.Q.shape[0])), cov=self.sigma**2 * self.Q(t-s), size=1).T
        return dW
    
    def sample_euler(self, s, t, n_particles=1):
        pass

class EnsembleBrownianMotionDrivenIntegral(EnsembleBrownianMotion):
    parameter_keys = ["shape", "sigma"]

    def __init__(self, **kwargs):
        # Set variable parameters using the ParameterInterface class.
        super().__init__(**kwargs)

        # Set covariance matrix associated with the integral directly.
        self.Q = kwargs.get("Q", None)
        self.L = kwargs.get("L", None)

        if self.Q is None:
            self.sample = self.sample_euler
        else:
            self.sample = self.sample_analytical

    def moments(self, s, t, n_particles=1):
        """Returns the means and covariances associated with a time interval (s,t). 
        These objects have shapes (Np, D, 1) and (Np, D, D). A scale mixture can be generated by setting 'sigma' to shape (Np, 1, 1).
        """
        means = np.zeros((self.shape[0], self.shape[1]))
        covs = self.sigma**2 * np.broadcast_to(self.Q(t-s), (n_particles, self.shape[0], self.shape[0]))
        return means, covs

    def sample_analytical(self, s, t, n_particles=1):
        """Generates random samples from a Gaussian ensemble defined by a (D, D) covariance matrix Q with scale proportional to the length of an interval (s, t).
        The shape of the random point by convention is (Np, D, 1). A scale mixture can be generated by setting 'sigma' to shape (Np, 1, 1).
        """
        dW = self.sigma * np.random.multivariate_normal(mean=np.zeros((self.Q.shape[0])), cov=self.Q(t-s), size=n_particles).reshape(n_particles, self.shape[0], self.shape[1])
        return dW
    
    def sample_euler(self, s, t, n_particles=1):
        pass


# NVM process driven SDE classes.

class NormalVarianceMeanProcessDrivenIntegral(StochasticIntegral):
    parameter_keys = ["shape", "mu", "sigma"]

    def __init__(self, **kwargs):
        # Set parameters using the ParameterInterface class.
        super().__init__(**kwargs)

        # Set SDE model terms.
        self.expA = kwargs.get("expA", None)
        self.L = kwargs.get("L", None)
        self.ft = lambda dt: self.expA(dt) @ self.L()

        # Set stochastic process generator and sampling method.
        self.subordinator = kwargs.get("subordinator", None)
        if self.subordinator is None:
            self.process = kwargs.get("process", None)

            if self.process is None:
                raise ValueError("The generator is not initialised. Arguments must contain a 'subordinator' or a 'process'.")
    
    def moments(self, s, t, n_particles, t_series, x_series):
        """Returns the means and covariances associated with a time interval (s,t). 
        These objects have shapes (Np, D, 1) and (Np, D, D).

        The function assumes that the given t_series only contains jumps in (s, t). Such a series can be created from a t_series
        object using the get_jumps_between method of LevyProcess objects.

        It also assumes that the driving NVM process is one dimensional.
        """
        mean = np.zeros((n_particles, self.L.shape[0], 1))
        cov = np.zeros((n_particles, self.expA.shape[0], self.expA.shape[1]))
        for i in range(n_particles):
            for j in range(x_series[i].size):
                mat = self.ft(t-t_series[i][j])
                mean[i] += mat @ np.array([[self.mu]]) @ np.array([[x_series[i][j]]])
                cov[i] += mat @ mat.T * np.array([[self.sigma**2]]) * np.array([[x_series[i][j]]])
        return mean, cov

    def sample(self, s=None, t=None, n_particles=1):
        t_series, x_series = self.subordinator.simulate_points(rate=(t-s), low=s, high=t, n_particles=n_particles)
        mean, cov = self.moments(s, t, n_particles=n_particles, t_series=t_series, x_series=x_series)
        dW = np.array([np.random.multivariate_normal(mean=mean[i].flatten(), cov=cov[i], size=1).T for i in range(n_particles)])
        return dW
    
    def _sample(self, s=None, t=None, n_particles=1):
        """The alternative sample method also returns the underlying jumps.
        """
        t_series, x_series = self.subordinator.simulate_points(rate=(t-s), low=s, high=t, n_particles=n_particles)
        mean, cov = self.moments(s, t, n_particles=n_particles, t_series=t_series, x_series=x_series)
        dW = np.array([np.random.multivariate_normal(mean=mean[i].flatten(), cov=cov[i], size=1).T for i in range(n_particles)])
        return dW, t_series, x_series