import numpy as np
from primitive.parameters import ParameterInterface

class LevyProcess(ParameterInterface):
    """The LevyProcess class is a base (parent) class for jump process simulators. The class defines the protocols to operate on data structures that contain the 
    jump times and sizes of multiple independent Levy processes.

    The adaptive truncation of a series representation of a Levy process results in a random number of jumps. Hence independent realisations of a Levy process will 
    likely have different number of jumps. To store the jumps corresponding to different realisations the data structure should allow the shape to be variable across
    a specified dimension. This class attempts to offer a solution by forming a Python list of NumPy arrays that separately correspond to the jumps of an individual
    realisation.
     
    Each realisation may represent an independent dimension in space or a random particle in a Monte Carlo algorithm.

    Helper methods are provided to get all jumps for each independent dimension within a specified time interval (or space coordinates).
    """
    parameter_keys = None

    def __len__(self):
        """The length of the LevyProcess object is set to 1. This can be helpful when multiple instances are stored in a Python list.
        """
        return 1

    def set_name(self, name):
        """Specific instances can be named for debugging.
        """
        if not hasattr(self, 'name'):
            self.name = name
        else:
            print('The process is already named as {}.'.format(self.name))

    def initialise_jumps(self, n_particles):
        """The empty array object is broadcast to multiple dimensions. The resulting Python list holds a reference to the same array in each of its dimensions.
        Since the size of an array cannot be changed in place, a new array object overwrites on the previous object when new jumps will be added.
        """
        t_series = n_particles*[np.array([], dtype=np.float32)]
        x_series = n_particles*[np.array([], dtype=np.float32)]
        return t_series, x_series
    
    def initialise_jump_sizes(self, n_particles):
        """The empty array object is broadcast to multiple dimensions. The resulting Python list holds a reference to the same array in each of its dimensions.
        Since the size of an array cannot be changed in place, a new array object overwrites on the previous object when new jumps will be added.
        """
        x_series = n_particles*[np.array([], dtype=np.float32)]
        return x_series
    
    def add_jump_sizes(self, x_series, x_series_extension, indices):
        """Update the jump sizes with the new jumps. Jumps are structured as (N, r) where N is the number of realisations and r is a random number of
        points. 
        """
        for i, idx in enumerate(indices):
            x_series[idx] = np.concatenate((x_series[idx], x_series_extension[i]))
        return x_series
    
    def add_jumps(self, t_series, x_series, t_series_extension, x_series_extension):
        """Update the jump times and sizes with the new jumps. Jumps are structured as (N, r) where N is the number of realisations and r is a random number of
        points. 
        """
        for i in range(len(t_series)):
            t_series[i] = np.concatenate((t_series[i], t_series_extension[i]))
            x_series[i] = np.concatenate((x_series[i], x_series_extension[i]))
        return t_series, x_series
    
    def sum(self, x_series):
        """Returns the sum of all jumps for each realisation.
        """
        return np.array([series.sum() for series in x_series])
    
    def get_largest_poisson_epoch(self, gamma_series):
        """Returns the largest Poisson epoch for later use in the computation of truncation level.
        """
        return np.array([series[-1] for series in gamma_series])

    def get_jumps_between(self, s, t, t_series, x_series):
        """Return t_series and x_series that only contains the jumps in (s, t).
        """
        t_series_interval = []
        x_series_interval = []
        for i in range(len(x_series)):
            mask = (s <= t_series[i]) & (t_series[i] <= t)
            t_series_interval.append(t_series[i][mask])
            x_series_interval.append(x_series[i][mask])
        return t_series, x_series

    def get_jumps_outside(self, s, t, t_series, x_series):
        """Return t_series and x_series that only contains the jumps outside (s, t).
        """
        t_series_interval = []
        x_series_interval = []
        for i in range(len(x_series)):
            mask = (s <= t_series[i]) & (t_series[i] <= t)
            t_series_interval.append(t_series[i][~mask])
            x_series_interval.append(x_series[i][~mask])
        return t_series, x_series
    
    def simulate_jump_times(self, x_series, low, high):
        """Simulate random jump times between 'low' and 'high' given a set of jumps 'x_series'.
        """
        t_series = []
        for i in range(len(x_series)):
            t_series.append(np.random.uniform(low=low, high=high, size=x_series[i].shape))
        return t_series

    def stochastic_integral(self, times, t_series, x_series):
        """Return the position of the process evaluated at 'times' for each realisation.
        """
        W = []
        for i in range(len(x_series)):
            W_i = np.array([x_series[i][t_series[i]<point].sum() for point in times]).reshape(times.shape[0], -1)
            W.append(W_i)
        return W
    
    # The following methods are required for the implementation of sampling algorithms based on jump likelihoods.

    def likelihood(self, x_prime, x):
        """ The density g( \dot | \theta ) evaluated at \theta^\prime.
        """
        pass

    def log_likelihood(self, x_prime, x):
        """ The log density g( \dot | \theta ) evaluated at \theta^\prime.
        """
        pass