import numpy as np

from primitive.linalg import LinearOperator

from stochastic.integrals import BrownianMotionDrivenIntegral, EnsembleBrownianMotionDrivenIntegral, NormalVarianceMeanProcessDrivenIntegral
from stochastic.variables import GaussianNoise, EnsembleGaussianNoise

# Custom linear operator components for a state-space model.

class VelocitySelector(LinearOperator):
    parameter_keys = ["shape"]

    def get_parameter_values(self):
        # There are no parameters in the L vector for Constant velocity and Langevin dynamics models.
        return {}
    
    def compute_matrix(self, dt=None):
        L = np.zeros(self.shape)
        L[-1, 0] = 1.
        return L

class expA_ConstantVelocity(LinearOperator):
    parameter_keys = ["shape"]

    def compute_matrix(self, dt):
        expA = np.zeros(self.shape)
        expA[0][0] = 1.
        expA[0][1] = dt
        expA[1][1] = 1.
        return expA

class Q_ConstantVelocity(LinearOperator):
    parameter_keys = ["shape"]

    def compute_matrix(self, dt):
        Q = np.zeros(self.shape)
        Q[0][0] = dt**3 / 3
        Q[0][1] = dt**2 / 2
        Q[1][0] = dt**2 / 2
        Q[1][1] = dt
        return Q
    
class expA_Langevin(LinearOperator):
    parameter_keys = ["shape", "theta"]

    def compute_matrix(self, dt):
        expA = np.zeros(self.shape)
        expA[0][0] = 1.
        expA[0][1] = (np.exp(self.theta*dt) - 1) / self.theta
        expA[1][1] = np.exp(self.theta*dt)
        return expA
    
class Q_Langevin(LinearOperator):
    parameter_keys = ["shape", "theta"]

    def compute_matrix(self, dt):
        Q = np.zeros(self.shape)
        K = -1. * self.theta
        Q[0][0] = ( dt - (2/K) * (1 - np.exp(self.theta*dt)) + (1/(2*K)) * (1 - np.exp(2*self.theta*dt)) ) / K**2
        Q[0][1] = ( (1/K) * (1 - np.exp(self.theta*dt)) - (1/(2*K)) * (1 - np.exp(2*self.theta*dt)) ) / K
        Q[1][0] = Q[0][1]
        Q[1][1] = (1 - np.exp(2*self.theta*dt)) / (2*K)
        return Q


# Base state-space model object:

class BaseStateSpaceModel:

    def __init__(self, **kwargs):
        """The BaseStateSpaceModel contains all essential functionality and definitions of a state-space model (SSM) for a linear system defined as

            X(t) = F X(s) + I(s, t)
            Y(t) = H X(t) + eps(t)

        To initialise an SSM, the required keyword arguments are 'F', 'I', 'H', and 'eps'. 

        The state X at each t is assumed to have shape (D,1) or (Np, D, 1) where Np is the number of particles and D is the state dimensions.
        The observations Y at each t is assumed to have shape (D_prime, 1) where D_prime is the observation dimensions.

        - Linear dynamical model.
        The argument 'F' is the state transition matrix. It must be a LinearOperator object that has a valid compute_matrix method which returns a
        matrix of shape (D, D). 

        The argument 'I' is a StochasticIntegral object defined in stochastic.integrals.py which is a high level wrapper for random driving forces.
        It can be used to sample random points and (possibly random) moments for a time interval (s, t) which returns a numpy array with shape (D, 1) or 
        (Np, D, 1) for points and means. The covariance will have shape (D, D) or (Np, D, D).

        - Linear observation model.
        The argument 'H' is the linear observation model expressed as a LinearOperator or a standard matrix with dimensions (D_prime, D). It is alternatively
        referred as the feature matrix in standard linear models.

        The argument eps is the observation noise that is implemented as a RandomVariable object defined in stochastic.variables.py. It may in general be time dependent.
        It can be used to sample random variables with shape (D_prime, 1) or (Np, D_prime, D_prime). For (D_prime, 1), it is assumed that the noise term is shared
        between particles if Np > 1.
        It can also be used to sample the associated mean with shape (D_prime, 1) or (Np, D_prime, 1) and covariance with shape (D_prime, D_prime) or (Np, D_prime, D_prime).
        """
        # System transition matrix
        self.F = kwargs.get("F", None)

        # System forcing function (stochastic integral)
        self.I = kwargs.get("I", None)

        # Measurement matrix
        self.H = kwargs.get("H", None)

        # Measurement noise:
        self.eps = kwargs.get("eps", None)

    def set_configuration(self, **kwargs):
        """This method is provided to set the main SSM attributes. 

        REQUIRED EDIT: Instead of setting all attributes, only change given attributes.
        """
        # System transition matrix
        self.F = kwargs.get("F", None)

        # System forcing function
        self.I = kwargs.get("I", None)

        # Measurement matrix
        self.H = kwargs.get("H", None)

        # Measurement noise:
        self.eps = kwargs.get("eps", None)

    def get_parameter_values(self):
        """The BaseStateSpaceModel does not have any parameters. Classes that inherit from BaseStateSpaceModel should implement a custom method.

        REQUIRED EDIT: The base model can return all parameters instead of not doing anything.
        """
        pass
    
    def set_parameter_values(self, **kwargs):
        """The BaseStateSpaceModel does not have any parameters. Classes that inherit from BaseStateSpaceModel should implement a custom method.

        REQUIRED EDIT: The base model can return all parameters instead of not doing anything.
        """
        pass

    def propose(self, times, x_init, n_particles=1):
        """The propose method implements forward simulation of the state dynamics.
        """
        # Ensure times is 1-dimensional.
        times = times.flatten()

        # Initialise the state array with zeros.
        X = np.zeros(shape=(times.shape[0], n_particles, x_init.shape[-2], 1))

        X[0,:] = x_init

        for i in range(1, times.shape[0]):
            dt = (times[i] - times[i-1])
            X[i,:] = self.F(dt) @ X[i-1,:] + self.I(s=times[i-1], t=times[i], n_particles=n_particles)
        return X[1:,:]

    def sample(self, times, x_init, n_particles=1):
        """The sample method implements forward simulation of the SSM.

        The keyword argument 'times' is assumed to have shape (N,1), 'x_init' has shape (D,1) or (Np, D, 1) where Np is equal to 'n_particles'. 

        If 'x_init' has shape (Np, D, 1) 'n_particles' must be equal to Np. The method currently does NOT assert this.

        REQUIRED EDIT: remove the particles dimension if n_particles=1 before returning.
        """

        # Ensure times is 1-dimensional.
        times = times.flatten()

        # Initialise the state and observation arrays with zeros.
        X = np.zeros(shape=(times.shape[0], n_particles, x_init.shape[-2], 1))
        Y = np.zeros(shape=(times.shape[0], n_particles, self.H.shape[-2], 1))

        # Assign the given initial array of points as the initial state. 
        ## If 'x_init' has shape (D, 1) it will be broadcast to (Np, D, 1).
        X[0,:] = x_init

        # Generate observations associated with the initial state.
        Y[0,:] = self.H @ X[0,:] + self.eps(t=times[0], n_particles=n_particles)

        # Forward simulation of states and observation generation.
        for i in range(1, times.shape[0]):
            dt = (times[i] - times[i-1])

            X[i,:] = self.F(dt) @ X[i-1,:] + self.I(s=times[i-1], t=times[i], n_particles=n_particles)
            Y[i,:] = self.H @ X[i,:] + self.eps(t=times[i], n_particles=n_particles)

        return X, Y
    

class BaseLevyStateSpaceModel(BaseStateSpaceModel):

    def propose(self, times, x_init, n_particles=1):
        """The propose method implements forward simulation of the state dynamics.
        """
        # Ensure times is 1-dimensional.
        times = times.flatten()

        # Initialise the state array with zeros.
        X = np.zeros(shape=(times.shape[0], n_particles, x_init.shape[-2], 1))

        # Initialise jump times and sizes.
        _t_series = np.zeros((n_particles, 1))
        _x_series = np.zeros((n_particles, 1))

        X[0,:] = x_init

        for i in range(1, times.shape[0]):
            dt = (times[i] - times[i-1])
            dW, t_series, x_series = self.I._sample(s=times[i-1], t=times[i], n_particles=n_particles)

            X[i,:] = self.F(dt) @ X[i-1,:] + dW

            _t_series = np.concatenate((_t_series, t_series), axis=1)
            _x_series = np.concatenate((_x_series, x_series), axis=1)

        return X[1:,:], _t_series, _x_series

    def sample(self, times, x_init, n_particles=1):
        """The sample method implements forward simulation of the SSM.

        The keyword argument 'times' is assumed to have shape (N,1), 'x_init' has shape (D,1) or (Np, D, 1) where Np is equal to 'n_particles'. 

        If 'x_init' has shape (Np, D, 1) 'n_particles' must be equal to Np. The method currently does NOT assert this.

        REQUIRED EDIT: remove the particles dimension if n_particles=1 before returning.
        """

        # Ensure times is 1-dimensional.
        times = times.flatten()

        # Initialise the state and observation arrays with zeros.
        X = np.zeros(shape=(times.shape[0], n_particles, x_init.shape[-2], 1))
        Y = np.zeros(shape=(times.shape[0], n_particles, self.H.shape[-2], 1))

        # Initialise jump times and sizes.
        t_series, x_series = self.I.subordinator.initialise_jumps(n_particles)

        # Assign the given initial array of points as the initial state. 
        ## If 'x_init' has shape (D, 1) it will be broadcast to (Np, D, 1).
        X[0,:] = x_init

        # Generate observations associated with the initial state.
        Y[0,:] = self.H @ X[0,:] + self.eps(t=times[0], n_particles=n_particles)

        # Forward simulation of states and observation generation.
        for i in range(1, times.shape[0]):
            dt = (times[i] - times[i-1])
            dW, t_series_extension, x_series_extension = self.I._sample(s=times[i-1], t=times[i], n_particles=n_particles)

            X[i,:] = self.F(dt) @ X[i-1,:] + dW
            Y[i,:] = self.H @ X[i,:] + self.eps(t=times[i], n_particles=n_particles)

            t_series, x_series = self.I.subordinator.add_jumps(t_series, x_series, t_series_extension, x_series_extension)

        return X, Y, t_series, x_series


# Custom state-space model objects:

## Constant velocity model objects:

class BrownianConstantVelocityModel(BaseStateSpaceModel):

    def __init__(self, sigma=1., sigma_eps=0.1, D=2, D_prime=1):

        # Define state transition dynamics:
        F = expA_ConstantVelocity(**{"shape":(D, D)})
        Q = Q_ConstantVelocity(**{"shape":(D, D)})
        I = EnsembleBrownianMotionDrivenIntegral(**{"shape":(D, 1), "sigma":sigma, "Q":Q})

        # Define observation model:
        H = np.zeros((D_prime,D))
        H[0][0] = 1
        eps = EnsembleGaussianNoise(**{"shape":(D_prime,1), "sigma_eps":sigma_eps})

        config = {"F":F, "I":I, "H":H, "eps":eps}
        super().__init__(**config)

    def get_parameter_values(self):
        return self.I.get_parameter_values() | self.eps.get_parameter_values()
    
    def set_parameter_values(self, **kwargs):
        self.I.set_parameter_values(**kwargs)
        self.eps.set_parameter_values(**kwargs)

class NVMConstantVelocityModel(BaseLevyStateSpaceModel):

    def __init__(self, subordinator, mu=0., sigma=1., sigma_eps=0.1, D=2, D_prime=1):

        # Define state transition dynamics:
        F = expA_ConstantVelocity(**{"shape":(D, D)})
        L = VelocitySelector(shape=(D,1))
        I = NormalVarianceMeanProcessDrivenIntegral(**{"shape":(D, 1), "mu":mu, "sigma":sigma, "subordinator":subordinator, "expA":F, "L":L})

        # Define observation model:
        H = np.zeros((D_prime,D))
        H[0][0] = 1
        eps = EnsembleGaussianNoise(**{"shape":(D_prime,1), "sigma_eps":sigma_eps})

        config = {"F":F, "I":I, "H":H, "eps":eps}
        super().__init__(**config)

    def get_parameter_values(self):
        return self.I.get_parameter_values() | self.eps.get_parameter_values()
    
    def set_parameter_values(self, **kwargs):
        self.I.set_parameter_values(**kwargs)
        self.eps.set_parameter_values(**kwargs)


## Langevin model objects:

class BrownianLangevinModel(BaseStateSpaceModel):

    def __init__(self, theta, sigma=1., sigma_eps=0.1, D=2, D_prime=1):

        # Define state transition dynamics:
        F = expA_Langevin(**{"shape":(D, D), "theta":theta})
        Q = Q_Langevin(**{"shape":(D, D), "theta":theta})
        I = EnsembleBrownianMotionDrivenIntegral(**{"shape":(D, 1), "sigma":sigma, "Q":Q})

        # Define observation model:
        H = np.zeros((D_prime,D))
        H[0][0] = 1
        eps = EnsembleGaussianNoise(**{"shape":(D_prime,1), "sigma_eps":sigma_eps})

        config = {"F":F, "I":I, "H":H, "eps":eps}
        super().__init__(**config)

    def get_parameter_values(self):
        return self.F.get_parameter_values() | self.I.get_parameter_values() | self.eps.get_parameter_values()
    
    def set_parameter_values(self, **kwargs):
        self.F.set_parameter_values(**kwargs)
        self.I.set_parameter_values(**kwargs)
        self.eps.set_parameter_values(**kwargs)

class NVMLangevinModel(BaseLevyStateSpaceModel):

    def __init__(self, subordinator, theta, mu=0., sigma=1., sigma_eps=0.1, D=2, D_prime=1):

        # Define state transition dynamics:
        F = expA_Langevin(**{"shape":(D, D), "theta":theta})
        L = VelocitySelector(shape=(D,1))
        I = NormalVarianceMeanProcessDrivenIntegral(**{"shape":(D, 1), "mu":mu, "sigma":sigma, "subordinator":subordinator, "expA":F, "L":L})

        # Define observation model:
        H = np.zeros((D_prime,D))
        H[0][0] = 1
        eps = EnsembleGaussianNoise(**{"shape":(D_prime,1), "sigma_eps":sigma_eps})

        config = {"F":F, "I":I, "H":H, "eps":eps}
        super().__init__(**config)

    def get_parameter_values(self):
        return self.F.get_parameter_values() | self.I.get_parameter_values() | self.eps.get_parameter_values()
    
    def set_parameter_values(self, **kwargs):
        self.F.set_parameter_values(**kwargs)
        self.I.set_parameter_values(**kwargs)
        self.eps.set_parameter_values(**kwargs)

