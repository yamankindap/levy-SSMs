import numpy as np

from scipy.special import kv
from scipy.special import gamma as gammafnc
from scipy.special import gammainc, gammaincc, gammaincinv
from scipy.special import hankel1, hankel2

from primitive.utils import incgammal
from primitive.utils import incgammau

from primitive.stochastics import LevyProcess

class GammaProcess(LevyProcess):
    """The GammaProcess class produces random jump times and sizes of a gamma Levy process. It is designed to produce multiple realisations with a single call.
     
    The main method is .simulate_points() which returns random jump times and sizes for a given number of realisations. The default single realisation shape is
    (1, M) where M is the number of jump times and sizes. This shape is generalised to (n_realisations, M*) for multiple realisation versions where M* is variable
    across the realisations.
    
    Note that both times and sizes are always a Python list.
    """
    parameter_keys = ["beta", "C"]

    def __init__(self, **kwargs):
        # Set parameters using the ParameterInterface parent. 
        super().__init__(**kwargs)

        # Set sampling hyperparameters. For more information see "Point process simulation of generalised hyperbolic Lévy processes" by Kındap and Godsill, 2023.
        self.set_hyperparameters(M=10, tolerance=0.01, pt=0.05)

    def set_hyperparameters(self, M, tolerance, pt):
        """The simulation parameters are treated as hyperparameters for subordinator simulation.
        """
        self.M = M
        self.tolerance = tolerance
        self.pt = pt
    
    def h_gamma(self, gamma):
        """Shot noise method jump size function for a dominating process which is thinned at a later stage. The definition is based on "Series Representations of Levy
        Processes from the Perspective of Point Processes" by Rosinski, 2001. Specifically, the "fourth representation" in Section 6.
        """
        return 1/(self.beta*(np.exp(gamma/self.C)-1))

    def unit_expected_residual(self, c):
        """The unit expected residual mass for a series truncation with level 'c', i.e. the unit mass for all jumps smaller than 'c'. The derivation can be found in
        "Point process simulation of generalised hyperbolic Lévy processes" by Kındap and Godsill, 2023.
        """
        return (self.C/self.beta)*incgammal(1, self.beta*c)

    def unit_variance_residual(self, c):
        """The unit expected residual variance for a series truncation with level 'c', i.e. the unit variance for all jumps smaller than 'c'. The derivation can be found in
        "Point process simulation of generalised hyperbolic Lévy processes" by Kındap and Godsill, 2023.
        """
        return (self.C/self.beta**2)*incgammal(2, self.beta*c)
    
    def simulate_from_series_representation(self, rate=1.0, M=100, gamma_0=0.0, size=1):
        """Simulates from the series representation given the largest Poisson process point gamma or a list of gamma values that correspond to each realisation.
        In case 'size' > 1, the length of 'gamma_0' should match 'size' or 'gamma_0' can be a scalar.

        To support multiple realisations, a NumPy array must have the same dimensions across realisations. Rejected jumps change the dimension of the corresponding realisation.
        Instead these jumps are set to zero which is equivalent to rejecting them since they won't have any effect on the process. The computational burden of storing
        rejected jumps may be high when size is large.
        """
        gamma_sequence = np.random.exponential(scale=1/rate, size=(size,M))
        gamma_sequence[:,0] += gamma_0
        gamma_sequence = gamma_sequence.cumsum(axis=1)

        x_series = self.h_gamma(gamma_sequence)
        thinning_function = (1+self.beta*x_series)*np.exp(-self.beta*x_series)
        u = np.random.uniform(low=0.0, high=1.0, size=x_series.shape)
        x_series[u > thinning_function] = 0.
        return gamma_sequence, x_series
    
    def _simulate_from_series_representation(self, rate=1.0, M=10, gamma_0=0.0, size=1):
        """Simulates from the series representation given the largest Poisson process point gamma or a list of gamma values that correspond to each realisation.
        In case 'size' > 1, the length of 'gamma_0' should match 'size' or 'gamma_0' can be a scalar.
        """
        # The gamma sequence can be sampled in standard NumPy.
        gamma_sequence = np.random.exponential(scale=1/rate, size=(size,M))
        gamma_sequence[:,0] += gamma_0
        gamma_sequence = gamma_sequence.cumsum(axis=1)

        # Rejection sampling may result in different numbers of jumps for each realisation. Hence the final x_series is a list of NumPy arrays.
        x_sequence = self.h_gamma(gamma_sequence)
        thinning_function = (1+self.beta*x_sequence)*np.exp(-self.beta*x_sequence)

        x_series = size*[None]
        gamma_series = size*[None]
        for i in range(size):
            u = np.random.uniform(low=0.0, high=1.0, size=M)
            x_series[i] = x_sequence[i][u < thinning_function[i]]
            gamma_series[i] = gamma_sequence[i]

        return gamma_series, x_series
    
    def simulate_adaptively_truncated_jump_series(self, rate=1.0, size=1):
        """Simulate jumps from the Levy process using adaptive determination of the series truncation level. For details see "Point process simulation of
        generalised hyperbolic Lévy processes" by Kındap and Godsill, 2023.
        """
        gamma_sequence, x_series = self.simulate_from_series_representation(rate, M=self.M, gamma_0=0., size=size)

        truncation_level = self.h_gamma(gamma_sequence[:,-1])

        residual_expected_value = rate*self.unit_expected_residual(truncation_level)
        residual_variance = rate*self.unit_variance_residual(truncation_level)
        E_c = self.tolerance*x_series.sum(axis=1)

        condition1 = (residual_variance/((E_c - residual_expected_value)**2) > self.pt)
        condition2 = (E_c < residual_expected_value)

        while condition1.any() or condition2.any():
            gamma_sequence_extension, x_series_extension = self.simulate_from_series_representation(rate=rate, M=self.M, gamma_0=gamma_sequence[:,-1], size=size)
            gamma_sequence = np.concatenate((gamma_sequence, gamma_sequence_extension), axis=1)
            x_series = np.concatenate((x_series, x_series_extension), axis=1)
            truncation_level = self.h_gamma(gamma_sequence[:,-1])
            residual_expected_value = rate*self.unit_expected_residual(truncation_level)
            residual_variance = rate*self.unit_variance_residual(truncation_level)
            E_c = self.tolerance*x_series.sum(axis=1)

            condition1 = (residual_variance/((E_c - residual_expected_value)**2) > self.pt)
            condition2 = (E_c < residual_expected_value)
            
        return x_series, truncation_level
    
    def _simulate_adaptively_truncated_jump_series(self, rate=1.0, size=1):
        """Simulate jumps from the Levy process using adaptive determination of the series truncation level. For details see "Point process simulation of
        generalised hyperbolic Lévy processes" by Kındap and Godsill, 2023.
        """
        # Start simulating a fixed number of jumps.
        gamma_sequence, x_series = self.simulate_from_series_representation(rate, M=self.M, gamma_0=0., size=size)

        # Determine current truncation level for each realisation.
        truncation_level = self.h_gamma(gamma_sequence[:,-1])

        # Compute moments of the residual jumps.
        residual_expected_value = rate*self.unit_expected_residual(truncation_level)
        residual_variance = rate*self.unit_variance_residual(truncation_level)

        # Set tolerance level.
        E_c = self.tolerance*self.sum(x_series)

        # Set probabilistic upper bounds on the residual error and determine rejection conditions.
        condition1 = (residual_variance/((E_c - residual_expected_value)**2) > self.pt)
        condition2 = (E_c < residual_expected_value)

        # Continue adding new jumps until all realisations are sufficiently converged.
        while condition1.any() or condition2.any():

            # Simulate new jumps based on the last values of the gamma sequence for each realisation.
            gamma_sequence_extension, x_series_extension = self.simulate_from_series_representation(rate=rate, M=self.M, gamma_0=gamma_sequence[:,-1], size=size)

            # Extend gamma sequence for all realisations.
            gamma_sequence = np.concatenate((gamma_sequence, gamma_sequence_extension), axis=1)

            # Extend x_series for all realisations using the LevyProcess functionality.
            x_series = self.add_jump_sizes(x_series, x_series_extension, indices=np.arange(size))

            # Determine current truncation level for each realisation.
            truncation_level = self.h_gamma(gamma_sequence[:,-1])

            # Compute moments of the residual jumps.
            residual_expected_value = rate*self.unit_expected_residual(truncation_level)
            residual_variance = rate*self.unit_variance_residual(truncation_level)

            # Set tolerance level.
            E_c = self.tolerance*x_series.sum(axis=1)

            # Update conditions.
            condition1 = (residual_variance/((E_c - residual_expected_value)**2) > self.pt)
            condition2 = (E_c < residual_expected_value)
            
        return x_series, truncation_level
    
    def simulate_points(self, rate, low, high, n_particles=1):
        """Returns the times and jump sizes associated with the point process representation of a Levy process. 
        The truncation level is required for making a Gaussian approximation of the residual process. This type of approximation is not valid for the Gaussian process.
        """
        x_series, truncation_level = self.simulate_adaptively_truncated_jump_series(rate=rate, size=n_particles)
        # t_series = self.simulate_jump_times(x_series, low, high)
        t_series = np.random.uniform(low=low, high=high, size=x_series.shape)
        return t_series, x_series  

    # These will be derived in future publications.
    def likelihood(self, x_prime, x):
        """ The likelihood should consider both the poisson process likelihood of the times and the joint jump density. 
        """
        pass

    def log_likelihood(self, x_prime, x):
        """ The log density g( \dot | \theta ) evaluated at \theta^\prime.
        """
        pass


class StableProcess(LevyProcess):
    parameter_keys = ["alpha", "C"]

    def __init__(self, **kwargs):
        # Set parameters using the ParameterInterface parent.
        super().__init__(**kwargs)

        # Set sampling hyperparameters.
        self.check_parameter_constraints()

    def check_parameter_constraints(self):
        if (self.alpha >= 1):
            raise ValueError('The alpha parameter is set to greater than or equal to 1.')

    def h_stable(self, gamma):
        return np.power((self.alpha/self.C)*gamma, np.divide(-1,self.alpha))
    
    def unit_expected_residual(self, c):
        return (self.C/(1-self.alpha))*(c**(1-self.alpha))

    def unit_variance_residual(self, c):
        return (self.C/(2-self.alpha))*(c**(2-self.alpha))
    
    def simulate_from_series_representation(self, rate=1.0, M=1000, gamma_0=0.0, size=1):
        gamma_sequence = np.random.exponential(scale=1/rate, size=(size,M))
        gamma_sequence[:,0] += gamma_0
        gamma_sequence = gamma_sequence.cumsum(axis=1)
        x_series = self.h_stable(gamma_sequence)
        return gamma_sequence, x_series
    
    def simulate_points(self, rate, low, high, n_particles=1):
        """Returns the times and jump sizes associated with the point process representation of a Levy process. Returns the points.
        """
        gamma_sequence, x_series = self.simulate_from_series_representation(rate=rate, size=n_particles)
        t_series = np.random.uniform(low=low, high=high, size=x_series.shape)
        return t_series, x_series  


class TemperedStableProcess(LevyProcess):
    parameter_keys = ["alpha", "beta", "C"]

    def __init__(self, **kwargs):
        # Set parameters using the ParameterInterface parent.
        super().__init__(**kwargs)

        # Set sampling hyperparameters.
        self.set_hyperparameters(M=100, tolerance=0.01, pt=0.05)

    def set_hyperparameters(self, M, tolerance, pt):
        """The simulation parameters are treated as hyperparameters for subordinator simulation.
        """
        self.M = M
        self.tolerance = tolerance
        self.pt = pt
        self.set_residual_approximation_method(mode="Gaussian")

    def h_stable(self, gamma):
        """Shot noise method jump size function for a stable process which is thinned at a later stage.
        """
        return np.power((self.alpha/self.C)*gamma, np.divide(-1,self.alpha))

    def unit_expected_residual(self, c):
        """The unit expected residual mass for a series truncation with level 'c', i.e. the unit mass for all jumps smaller than 'c'.
        """
        return (self.C*self.beta**(self.alpha-1))*incgammal(1-self.alpha, self.beta*c)

    def unit_variance_residual(self, c):
        """The unit expected residual variance for a series truncation with level 'c', i.e. the unit variance for all jumps smaller than 'c'.
        """
        return (self.C*self.beta**(self.alpha-2))*incgammal(2-self.alpha, self.beta*c)

    # Residual approximation: 
    def set_residual_approximation_method(self, mode):
        if mode is None:
            print('Residual approximation mode is set to add the expected residual value.')
            self.simulate_residual = self.simulate_residual_drift
        elif mode == 'mean-only':
            print('Residual approximation mode is set to add the expected residual value.')
            self.simulate_residual = self.simulate_residual_drift
        elif mode == 'Gaussian':
            print('Residual approximation mode is set to Gaussian approximation.')
            self.simulate_residual = self.simulate_residual_gaussians
        else:
            raise ValueError('The mode can only be set to `mean-only` or `Gaussian`.')
        
    def residual_stats(self, rate, truncation_level):
        R_mu = rate*self.unit_expected_residual(truncation_level)
        R_var = rate*self.unit_variance_residual(truncation_level)
        return R_mu, R_var

    def simulate_residual_gaussians(self, low, high, truncation_level, n_realisations):
        n_samples = 50
        # n_samples = shape[1]

        R_mu = (high-low)*self.unit_expected_residual(truncation_level)
        R_var = (high-low)*self.unit_variance_residual(truncation_level)

        t_series = np.linspace(low, high, num=n_samples+1) # This series includes 0, which is later removed.
        delta = t_series[1] - t_series[0]

        residual_jumps = np.random.normal(loc=delta * R_mu, scale=np.sqrt(delta * R_var), size=(n_samples, R_mu.shape[0])).T

        # Broadcast linspaced times to number of particles.
        t_series = np.broadcast_to(t_series[1:][np.newaxis], shape=(n_realisations, n_samples))

        return t_series, residual_jumps

    def simulate_residual_drift(self, low, high, truncation_level, n_realisations):
        n_samples = 50

        R_mu = (high-low)*self.unit_expected_residual(truncation_level)

        t_series = np.linspace(low, high, num=n_samples+1) # This series includes 0, which is later removed.
        delta = t_series[1] - t_series[0]

        residual_jumps = np.random.normal(loc=delta * R_mu, scale=0, size=(n_samples, R_mu.shape[0])).T

        # Broadcast linspaced times to number of particles.
        t_series = np.broadcast_to(t_series[1:][np.newaxis], shape=(n_realisations, n_samples))

        return t_series, residual_jumps

    # Simulation functions:

    def _simulate_from_series_representation(self, rate=1.0, M=100, gamma_0=0.0, size=1):
        """Simulates from the series representation given the largest Poisson process point gamma or a list of gamma values that correspond to each realisation.
        In case 'size' > 1, the length of 'gamma_0' should match 'size' or 'gamma_0' can be a scalar.

        To support multiple realisations, a NumPy array must have the same dimensions across realisations. Rejected jumps change the dimension of the corresponding realisation.
        Instead these jumps are set to zero which is equivalent to rejecting them since they won't have any effect on the process. The computational burden of storing
        rejected jumps may be high when size is large.
        """
        gamma_sequence = np.random.exponential(scale=1/rate, size=(size,M))
        gamma_sequence[:,0] += gamma_0
        gamma_sequence = gamma_sequence.cumsum(axis=1)

        x_series = self.h_stable(gamma_sequence)
        thinning_function = np.exp(-self.beta*x_series)
        u = np.random.uniform(low=0.0, high=1.0, size=x_series.shape)
        x_series[u > thinning_function] = 0.
        return gamma_sequence, x_series
    
    def simulate_from_series_representation(self, rate=1.0, M=100, gamma_0=0.0, size=1):
        """Simulates from the series representation given the largest Poisson process point gamma or a list of gamma values that correspond to each realisation.
        In case 'size' > 1, the length of 'gamma_0' should match 'size' or 'gamma_0' can be a scalar.
        """
        # The gamma sequence can be sampled in standard NumPy.
        gamma_sequence = np.random.exponential(scale=1/rate, size=(size,M))
        gamma_sequence[:,0] += gamma_0
        gamma_sequence = gamma_sequence.cumsum(axis=1)

        # Rejection sampling may result in different numbers of jumps for each realisation. Hence the final x_series is a list of NumPy arrays.
        x_sequence = self.h_stable(gamma_sequence)
        thinning_function = np.exp(-self.beta*x_sequence)

        x_series = size*[None]
        gamma_series = size*[None]
        for i in range(size):
            u = np.random.uniform(low=0.0, high=1.0, size=M)
            x_series[i] = x_sequence[i][u < thinning_function[i]]
            gamma_series[i] = gamma_sequence[i]
        return gamma_series, x_series
    
    def _simulate_adaptively_truncated_jump_series(self, rate=1.0, size=1):
        """Simulate jumps from the Levy process using adaptive determination of the series truncation level. For details see "Point process simulation of
        generalised hyperbolic Lévy processes" by Kındap and Godsill, 2023.
        """
        gamma_sequence, x_series = self.simulate_from_series_representation(rate, M=self.M, gamma_0=0., size=size)

        truncation_level = self.h_stable(gamma_sequence[:,-1])

        residual_expected_value = rate*self.unit_expected_residual(truncation_level)
        residual_variance = rate*self.unit_variance_residual(truncation_level)

        E_c = self.tolerance*x_series.sum(axis=1) + residual_expected_value

        max_iter = 50
        idx = 0
        condition = (residual_variance/((E_c - residual_expected_value)**2) > self.pt) 

        while condition.any():
            gamma_sequence_extension, x_series_extension = self.simulate_from_series_representation(rate=rate, M=self.M, gamma_0=gamma_sequence[:,-1], size=size)
            gamma_sequence = np.concatenate((gamma_sequence, gamma_sequence_extension), axis=1)
            x_series = np.concatenate((x_series, x_series_extension), axis=1)
            truncation_level = self.h_stable(gamma_sequence[:,-1])
            residual_expected_value = rate*self.unit_expected_residual(truncation_level)
            residual_variance = rate*self.unit_variance_residual(truncation_level)
            E_c = self.tolerance*x_series.sum(axis=1) + residual_expected_value

            condition = (residual_variance/((E_c - residual_expected_value)**2) > self.pt)

            idx += 1
            if idx > max_iter:
                print('Max iter reached.')
                break
            
        return x_series, truncation_level

    def simulate_adaptively_truncated_jump_series(self, rate=1.0, size=1):
        """Simulate jumps from the Levy process using adaptive determination of the series truncation level. For details see "Point process simulation of
        generalised hyperbolic Lévy processes" by Kındap and Godsill, 2023.
        """
        # Start simulating a fixed number of jumps.
        gamma_series, x_series = self.simulate_from_series_representation(rate, M=self.M, gamma_0=0., size=size)

        # Determine current truncation level for each realisation.
        epochs = self.get_largest_poisson_epoch(gamma_series)
        truncation_level = self.h_stable(epochs)

        # Compute moments of the residual jumps.
        residual_expected_value = rate*self.unit_expected_residual(truncation_level)
        residual_variance = rate*self.unit_variance_residual(truncation_level)

        # Set tolerance level.
        E_c = self.tolerance*self.sum(x_series) + residual_expected_value

        # Set probabilistic upper bounds on the residual error and determine rejection conditions.
        max_iter = 50
        idx = 0
        condition = (residual_variance/((E_c - residual_expected_value)**2) > self.pt)
        active_indices = np.where(condition)[0]

        # Continue adding new jumps until all realisations are sufficiently converged.
        while (active_indices.size > 0):

            # Simulate new jumps based on the last values of the gamma sequence for each realisation.
            gamma_series_extension, x_series_extension = self.simulate_from_series_representation(rate=rate, M=self.M, gamma_0=epochs[active_indices], size=active_indices.size)

            # Extend gamma sequence for all realisations.
            gamma_series = self.add_jump_sizes(gamma_series, gamma_series_extension, indices=active_indices)

            # Extend x_series for all realisations using the LevyProcess functionality.
            x_series = self.add_jump_sizes(x_series, x_series_extension, indices=active_indices)

            # Determine current truncation level for each realisation.
            epochs = self.get_largest_poisson_epoch(gamma_series)
            truncation_level = self.h_stable(epochs)

            # Compute moments of the residual jumps.
            residual_expected_value = rate*self.unit_expected_residual(truncation_level)
            residual_variance = rate*self.unit_variance_residual(truncation_level)

            # Set tolerance level.
            E_c = self.tolerance*self.sum(x_series) + residual_expected_value

            # Update conditions.
            condition = (residual_variance/((E_c - residual_expected_value)**2) > self.pt)
            active_indices = np.where(condition)[0]

            idx += 1
            if idx > max_iter:
                print('Max iter reached.')
                break
            
        return x_series, truncation_level

    def simulate_points(self, rate, low, high, n_particles=1):
        """Returns the times and jump sizes associated with the point process representation of a Levy process.
        The truncation level is required for making a Gaussian approximation of the residual process.
        """
        x_series, truncation_level = self.simulate_adaptively_truncated_jump_series(rate=rate, size=n_particles)
        t_series = self.simulate_jump_times(x_series, low, high)

        residual_t_series, residual_jumps = self.simulate_residual(low=low, high=high, truncation_level=truncation_level, n_realisations=n_particles)

        t_series, x_series = self.add_jumps(t_series, x_series, residual_t_series, residual_jumps)
        return t_series, x_series

    # These will be derived in future publications.
    def likelihood(self, x_prime, x):
        """ The likelihood should consider both the poisson process likelihood of the times and the joint jump density. 
        """
        pass

    def log_likelihood(self, x_prime, x):
        """ The log density g( \dot | \theta ) evaluated at \theta^\prime.
        """
        pass

    def propose(self, x, shape):
        pass


class GeneralisedInverseGaussianProcess(LevyProcess):
    parameter_keys = ["lam", "gamma", "delta"]

    def __init__(self, **kwargs):
        # Set parameters using the ParameterInterface parent.
        super().__init__(**kwargs)

        self.abs_lam = np.abs(kwargs["lam"])

        # Set sampling hyperparameters.
        self.set_hyperparameters(M_gamma=10, M_stable=100, tolerance=0.01, pt=0.05)
        self.max_iter = 50 # This can also be given as a keyword argument.

        # The parameter values assigned here are temporary.
        self.gamma_process = GammaProcess(**{"beta":np.float64(1.), "C":np.float64(1.)})
        self.gamma_process2 = GammaProcess(**{"beta":np.float64(1.), "C":np.float64(1.)})
        self.tempered_stable_process = TemperedStableProcess(**{"alpha":np.float64(0.6), "beta":np.float64(1.), "C":np.float64(1.)})

        # Define a third gamma process for the positive lam extension
        if (self.lam > 0):
            C = np.float64(self.lam)
            beta = np.float64(0.5*self.gamma**2)
            self.pos_ext_gamma_process = GammaProcess(**{"beta":beta, "C":C})

        self.set_simulation_method()
        self.set_residual_approximation_method(mode="Gaussian")

    def set_hyperparameters(self, M_gamma, M_stable, tolerance, pt):
        """The simulation parameters are treated as hyperparameters for subordinator simulation.
        """
        self.M_gamma = M_gamma
        self.M_stable = M_stable
        self.tolerance = tolerance
        self.pt = pt

    # Residual approximation module:
    def _exact_residual_stats(self, rate, truncation_level_gamma, truncation_level_TS):
        residual_expected_value_GIG = rate*self.tempered_stable_process.unit_expected_residual(truncation_level_TS)
        residual_variance_GIG = rate*self.tempered_stable_process.unit_variance_residual(truncation_level_TS)
        return residual_expected_value_GIG, residual_variance_GIG
    
    def _lower_bound_residual_stats(self, rate, truncation_level_gamma, truncation_level_TS):
        residual_expected_value_GIG = rate*self.lb_gamma_process.unit_expected_residual(truncation_level_gamma) + rate*self.lb_tempered_stable_process.unit_expected_residual(truncation_level_TS)
        residual_variance_GIG = rate*self.lb_gamma_process.unit_variance_residual(truncation_level_gamma) + rate*self.lb_tempered_stable_process.unit_variance_residual(truncation_level_TS)
        return residual_expected_value_GIG, residual_variance_GIG

    # Define the related function using mean or gaussian here....
    def set_residual_approximation_method(self, mode):
        if (self.abs_lam >= 0.5):
            if (self.abs_lam == 0.5):
                print('Residual approximation method is set to exact method.')
                self.residual_stats = self._exact_residual_stats
            else:
                # Initialise the lower bounding point processes for residual approximation
                print('Residual approximation method is set to lower bounding method.')
                z0 = self.cornerpoint()
                H0 = z0*self.H_squared(z0)
                C_gamma_B = np.float64(z0/((np.pi**2)*H0*self.abs_lam))
                beta_gamma_B = np.float64(0.5*self.gamma**2 + (self.abs_lam/(1+self.abs_lam))*(z0**2)/(2*self.delta**2))
                self.lb_gamma_process = GammaProcess(**{"beta":beta_gamma_B, "C":C_gamma_B})
                beta_0 = np.float64(1.95) # This parameter value can be optimised further in the future...
                C_TS_B = np.float64((2*self.delta*np.sqrt(np.e)*np.sqrt(beta_0-1))/((np.pi**2)*H0*beta_0))
                beta_TS_B = np.float64(0.5*self.gamma**2 + (beta_0*z0**2)/(2*self.delta**2))
                self.lb_tempered_stable_process = TemperedStableProcess(**{"alpha":0.5, "beta":beta_TS_B, "C":C_TS_B})
                # Select the appropriate residual_gaussian_sequence() function
                self.residual_stats = self._lower_bound_residual_stats
        else:
            print('Residual approximation method is set to lower bounding method.')
            z1 = self.cornerpoint()
            C_gamma_A = np.float64(z1/(2*np.pi*self.abs_lam))
            beta_gamma_A = np.float64(0.5*self.gamma**2 + (self.abs_lam/(1+self.abs_lam))*(z1**2)/(2*self.delta**2))
            self.lb_gamma_process = GammaProcess(**{"beta":beta_gamma_A, "C":C_gamma_A})
            beta_0 = np.float64(1.95) # This parameter value can be optimised further in the future...
            C_TS_A = np.float64((self.delta*np.sqrt(np.e)*np.sqrt(beta_0-1))/(np.pi*beta_0))
            beta_TS_A = np.float64(0.5*self.gamma**2 + (beta_0*z1**2)/(2*self.delta**2))
            self.lb_tempered_stable_process = TemperedStableProcess(**{"alpha":0.5, "beta":beta_TS_A, "C":C_TS_A})
            self.residual_stats = self._lower_bound_residual_stats

        if mode is None:
            print('Residual approximation mode is set to add the expected residual value.')
            self.simulate_residual = self.simulate_residual_drift
        elif mode == 'mean-only':
            print('Residual approximation mode is set to add the expected residual value.')
            self.simulate_residual = self.simulate_residual_drift
        elif mode == 'Gaussian':
            print('Residual approximation mode is set to Gaussian approximation.')
            self.simulate_residual = self.simulate_residual_gaussians
        else:
            raise ValueError('The mode can only be set to `mean-only` or `Gaussian`.')

    def simulate_residual_gaussians(self, low, high, truncation_level_gamma, truncation_level_TS, n_realisations):
        n_samples = 50
        # n_samples = shape[1]

        R_mu, R_var = self.residual_stats((high-low), truncation_level_gamma, truncation_level_TS)

        t_series = np.linspace(low, high, num=n_samples+1) # This series includes 0, which is later removed.
        delta = t_series[1] - t_series[0]

        residual_jumps = np.random.normal(loc=delta * R_mu, scale=np.sqrt(delta * R_var), size=(n_samples, R_mu.shape[0])).T

        # Broadcast linspaced times to number of particles.
        t_series = np.broadcast_to(t_series[1:][np.newaxis], shape=(n_realisations, n_samples))
        
        return t_series, residual_jumps

    def simulate_residual_drift(self, low, high, truncation_level_gamma, truncation_level_TS, n_realisations):
        n_samples = 50
        # n_samples = shape[1]

        R_mu, R_var = self.residual_stats((high-low), truncation_level_gamma, truncation_level_TS)

        t_series = np.linspace(low, high, num=n_samples+1)
        delta = t_series[1] - t_series[0]

        residual_jumps = np.random.normal(loc=delta * R_mu, scale=0, size=(n_samples, R_mu.shape[0])).T

        # Broadcast linspaced times to number of particles.
        t_series = np.broadcast_to(t_series[1:][np.newaxis], shape=(n_realisations, n_samples))

        return t_series, residual_jumps

    # Auxiliary functionality
    def cornerpoint(self):
        return np.power(np.power(float(2), 1-2*self.abs_lam)*np.pi/np.power(gammafnc(self.abs_lam), 2), 1/(1-2*self.abs_lam))

    def H_squared(self, z):
        return np.real(hankel1(self.abs_lam, z)*hankel2(self.abs_lam, z))

    def probability_density(self, x):
        return np.power(self.gamma/self.delta, self.lam)*(1/(2*kv(self.lam, self.delta*self.gamma))*np.power(x, self.lam-1)*np.exp(-(self.gamma**2*x+self.delta**2/x)/2))

    def random_sample(self, size):
        def thinning_function(delta, x):
            return np.exp(-(1/2)*(np.power(delta, 2)*(1/x)))
        def reciprocal_sample(x, i):
            return x**(i)
        def random_GIG(lam, gamma, delta, size=1):
            i = 1
            if lam < 0:
                tmp = gamma
                gamma = delta
                delta = tmp
                lam = -lam
                i = -1
            shape = lam
            scale = 2/np.power(gamma, 2)
            gamma_rv = np.random.gamma(shape=shape, scale=scale, size=size)
            u = np.random.uniform(low=0.0, high=1.0, size=size)
            sample = gamma_rv[u < thinning_function(delta, gamma_rv)]
            return reciprocal_sample(sample, i)
        sample = np.array([])
        while sample.size < size:
            sample = np.concatenate((sample, random_GIG(self.lam, self.gamma, self.delta, size=size)))
        return sample[np.random.randint(low=0, high=sample.size, size=size)]

    # Extend the gig process density with a gamma process for lambda > 0:
    def simulate_with_positive_extension(self, rate, size=1):
        x_series, truncation_level = self.simulate_Q_GIG(rate=rate, size=size)
        x_P_series, _ = self.pos_ext_gamma_process.simulate_adaptively_truncated_jump_series(rate=rate, size=size)
        x_series = self.add_jump_sizes(x_series, x_P_series, indices=np.arange(size))
        return x_series, truncation_level
    
    # Select simulation method and set corresponding parameters   
    def set_simulation_method(self, method=None):
        # Automatically select a method for simulation
        if method is None:
            if (self.abs_lam >= 0.5):
                if (self.gamma == 0) or (self.abs_lam == 0.5):
                    print('Simulation method is set to GIG paper version.')
                    # Set parameters of the tempered stable process...
                    alpha = np.float64(0.5)
                    C = np.float64(self.delta*gammafnc(0.5)/(np.sqrt(2)*np.pi))
                    beta = np.float64(0.5*self.gamma**2)

                    if (self.gamma == 0):
                        print('The dominating point process is set as a stable process.')
                        self.tempered_stable_process = StableProcess(**{"alpha":alpha, "C":C})
                    else:
                        self.tempered_stable_process.set_parameter_values(**{"alpha":alpha, "beta":beta, "C":C})

                    self.simulate_Q_GIG = self.simulate_adaptive_series_setting_1
                    if (self.lam > 0):
                        print('An independent gamma process extension will be made.')
                        self.simulate_jumps = self.simulate_with_positive_extension
                    else:
                        self.simulate_jumps = self.simulate_Q_GIG
                else:
                    print('Simulation method is set to improved version.')
                    # Set parameters of the two gamma and one TS processes...
                    z1 = self.cornerpoint()
                    C1 = np.float64(z1/(np.pi*self.abs_lam*2*(1+self.abs_lam)))
                    beta1 = np.float64(0.5*self.gamma**2)
                    self.gamma_process.set_parameter_values(**{"beta":beta1, "C":C1})

                    C2 = np.float64(z1/(np.pi*2*(1+self.abs_lam)))
                    beta2 = np.float64(0.5*self.gamma**2 + (z1**2)/(2*self.delta**2))
                    self.gamma_process2.set_parameter_values(**{"beta":beta2, "C":C2})

                    C = np.float64(self.delta/(np.sqrt(2*np.pi)))
                    alpha = np.float64(0.5)
                    beta = np.float64(0.5*self.gamma**2 + (z1**2)/(2*self.delta**2))
                    self.tempered_stable_process.set_parameter_values(**{"alpha":alpha, "beta":beta, "C":C})

                    self.simulate_Q_GIG = self.simulate_adaptive_combined_series_setting_1
                    if (self.lam > 0):
                        print('An independent gamma process extension will be made.')
                        self.simulate_jumps = self.simulate_with_positive_extension
                    else:
                        self.simulate_jumps = self.simulate_Q_GIG
            else:
                print('Simulation method is set to improved version for 0 < |lam| < 0.5.')
                # Set parameters of the two gamma and one TS processes...
                z0 = self.cornerpoint()
                H0 = z0*self.H_squared(z0)
                C1 = np.float64(z0/((np.pi**2)*H0*self.abs_lam*(1+self.abs_lam)))
                beta1 = np.float64(0.5*self.gamma**2)
                self.gamma_process.set_parameter_values(**{"beta":beta1, "C":C1})

                C2 = np.float64(z0/((np.pi**2)*(1+self.abs_lam)*H0))
                beta2 = np.float64(0.5*self.gamma**2 + (z0**2)/(2*self.delta**2))
                self.gamma_process2.set_parameter_values(**{"beta":beta2, "C":C2})

                C = np.float64(np.sqrt(2*self.delta**2)*gammafnc(0.5)/(H0*np.pi**2))
                alpha = np.float64(0.5)
                beta = np.float64(0.5*self.gamma**2)
                self.tempered_stable_process.set_parameter_values(**{"alpha":alpha, "beta":beta, "C":C})

                self.simulate_Q_GIG = self.simulate_adaptive_combined_series_setting_2
                if (self.lam > 0):
                    print('An independent gamma process extension will be made.')
                    self.simulate_jumps = self.simulate_with_positive_extension
                else:
                    self.simulate_jumps = self.simulate_Q_GIG
        else:
            raise ValueError('The manual selection functionality for simulation method is NOT implemented.')

    
    # Jump magnitude simulation:
    ## GIG-paper:
    def simulate_adaptive_series_setting_1(self, rate=1.0, size=1):

        # Start simulating a fixed number of jumps.
        gamma_series, x_series = self.simulate_series_setting_1(rate=rate, M=self.M_stable, gamma_0=0., size=size)

        # Determine current truncation level for each realisation.
        epochs = self.get_largest_poisson_epoch(gamma_series)
        truncation_level = self.tempered_stable_process.h_stable(epochs)

        # Compute moments of the residual jumps.
        residual_expected_value = rate*self.tempered_stable_process.unit_expected_residual(truncation_level)
        residual_variance = rate*self.tempered_stable_process.unit_variance_residual(truncation_level)

        # Set tolerance level.
        E_c = self.tolerance*self.sum(x_series) + residual_expected_value
        
        # Set probabilistic upper bounds on the residual error and determine rejection conditions.
        max_iter = 50
        idx = 0
        condition = (residual_variance/((E_c - residual_expected_value)**2) > self.pt)
        active_indices = np.where(condition)[0]

        # Continue adding new jumps until all realisations are sufficiently converged.
        while (active_indices.size > 0):

            # Simulate new jumps based on the last values of the gamma sequence for each realisation.
            gamma_series_extension, x_series_extension = self.simulate_series_setting_1(rate=rate, M=self.M_stable, gamma_0=epochs[active_indices], size=active_indices.size)

            # Extend gamma sequence for all realisations.
            gamma_series = self.add_jump_sizes(gamma_series, gamma_series_extension, indices=active_indices)

            # Extend x_series for all realisations using the LevyProcess functionality.
            x_series = self.add_jump_sizes(x_series, x_series_extension, indices=active_indices)

            # Determine current truncation level for each realisation.
            epochs = self.get_largest_poisson_epoch(gamma_series)
            truncation_level = self.tempered_stable_process.h_stable(epochs)

            # Compute moments of the residual jumps.
            residual_expected_value = rate*self.tempered_stable_process.unit_expected_residual(truncation_level)
            residual_variance = rate*self.tempered_stable_process.unit_variance_residual(truncation_level)

            # Set tolerance level.
            E_c = self.tolerance*self.sum(x_series) + residual_expected_value

            # Update conditions.
            condition = (residual_variance/((E_c - residual_expected_value)**2) > self.pt)
            active_indices = np.where(condition)[0]

            idx += 1
            if idx > max_iter:
                print('Max iter reached.')
                break

        return x_series, truncation_level

    def simulate_series_setting_1(self, rate, M=100, gamma_0=0., size=1):
        gamma_sequence, x_series = self.tempered_stable_process.simulate_from_series_representation(rate=rate, M=M, gamma_0=gamma_0, size=size)

        # Simulate auxiliary variables z and thin x_series based on acceptance probabilities.
        for i in range(size):
            z_series = np.sqrt(np.random.gamma(shape=0.5, scale=np.power(x_series[i]/(2*self.delta**2), -1.0)))
            hankel_squared = self.H_squared(z_series)
            acceptance_prob = 2/(hankel_squared*z_series*np.pi)
            u = np.random.uniform(low=0.0, high=1.0, size=x_series[i].size)
            x_series[i] = x_series[i][u < acceptance_prob]

        return gamma_sequence, x_series

    ## GH-paper:
    def simulate_adaptive_combined_series_setting_1(self, rate=1.0, size=1):

        # Start simulating a fixed number of jumps from each process.
        gamma_sequence_N_Ga_1, x_series_N_Ga_1 = self.simulate_left_bounding_series_setting_1(rate=rate, M=self.M_gamma, size=size)
        gamma_sequence_N_Ga_2, x_series_N_Ga_2 = self.simulate_left_bounding_series_setting_1_alternative(rate=rate, M=self.M_gamma, size=size)
        gamma_sequence_N2, x_series_N2 = self.simulate_right_bounding_series_setting_1(rate=rate, M=self.M_stable, size=size)

        # Extend x_series for all realisations using the LevyProcess functionality.
        x_series = self.add_jump_sizes(x_series_N_Ga_1, x_series_N_Ga_2, indices=np.arange(size))
        x_series = self.add_jump_sizes(x_series, x_series_N2, indices=np.arange(size))

        # Determine current truncation level for each realisation.
        epochs_N_Ga_1 = self.get_largest_poisson_epoch(gamma_sequence_N_Ga_1)
        truncation_level_N_Ga_1 = self.gamma_process.h_gamma(epochs_N_Ga_1)

        epochs_N_Ga_2 = self.get_largest_poisson_epoch(gamma_sequence_N_Ga_2)
        truncation_level_N_Ga_2 = self.gamma_process2.h_gamma(epochs_N_Ga_2)

        epochs_N2 = self.get_largest_poisson_epoch(gamma_sequence_N2)
        truncation_level_N2 = self.tempered_stable_process.h_stable(epochs_N2)

        # Compute moments of the residual jumps.
        residual_expected_value_N_Ga_1 = rate*self.gamma_process.unit_expected_residual(truncation_level_N_Ga_1)
        residual_variance_N_Ga_1 = rate*self.gamma_process.unit_variance_residual(truncation_level_N_Ga_1)
        residual_expected_value_N_Ga_2 = rate*self.gamma_process2.unit_expected_residual(truncation_level_N_Ga_2)
        residual_variance_N_Ga_2 = rate*self.gamma_process2.unit_variance_residual(truncation_level_N_Ga_2)
        residual_expected_value_N2 = rate*self.tempered_stable_process.unit_expected_residual(truncation_level_N2)
        residual_variance_N2 = rate*self.tempered_stable_process.unit_variance_residual(truncation_level_N2)
        residual_expected_value = residual_expected_value_N_Ga_1 + residual_expected_value_N_Ga_2 + residual_expected_value_N2
        residual_variance = residual_variance_N_Ga_1 + residual_variance_N_Ga_2 + residual_variance_N2

        # For each realisation select process with largest (worst) truncation level.
        selections = np.argmax(np.array([truncation_level_N_Ga_1, truncation_level_N_Ga_2, truncation_level_N2]), axis=0)
        ## Then randomly select a process to simulate based on empirical counts of requirement.
        selection = np.random.choice(selections)

        # Set tolerance level.
        _mean_lower_bound, _var_lower_bound = self.residual_stats(rate, truncation_level_gamma=truncation_level_N2, truncation_level_TS=truncation_level_N2)
        E_c = self.tolerance*self.sum(x_series) + _mean_lower_bound
        
        # Set probabilistic upper bounds on the residual error and determine rejection conditions.
        max_iter = 50
        idx = 0
        condition1 = (residual_variance/((E_c - residual_expected_value)**2) > self.pt)
        condition2 = (E_c < residual_expected_value)
        active_indices = np.where(np.logical_or(condition1, condition2))[0]

        # Continue adding new jumps until all realisations are sufficiently converged.
        while (active_indices.size > 0):
            
            if (selection == 2):

                # Simulate new jumps based on the last values of the gamma sequence for each realisation.
                gamma_series_extension, x_series_extension = self.simulate_right_bounding_series_setting_1(
                    rate=rate,
                    M=self.M_stable,
                    gamma_0=epochs_N2[active_indices],
                    size=active_indices.size
                )

                # Extend gamma sequence for all realisations.
                gamma_sequence_N2 = self.add_jump_sizes(gamma_sequence_N2, gamma_series_extension, indices=active_indices)

                # Extend x_series for all realisations using the LevyProcess functionality.
                x_series_N2 = self.add_jump_sizes(x_series_N2, x_series_extension, indices=active_indices)

                # Determine current truncation level for each realisation.
                epochs_N2 = self.get_largest_poisson_epoch(gamma_sequence_N2)
                truncation_level_N2 = self.tempered_stable_process.h_stable(epochs_N2)

                # Compute moments of the residual jumps.
                residual_expected_value_N2 = rate*self.tempered_stable_process.unit_expected_residual(truncation_level_N2)
                residual_variance_N2 = rate*self.tempered_stable_process.unit_variance_residual(truncation_level_N2)
                
            elif (selection == 0):

                # Simulate new jumps based on the last values of the gamma sequence for each realisation.
                gamma_sequence_extension, x_series_extension = self.simulate_left_bounding_series_setting_1(
                    rate=rate,
                    M=self.M_gamma,
                    gamma_0=epochs_N_Ga_1[active_indices],
                    size=active_indices.size
                )

                # Extend gamma sequence for all realisations.
                gamma_sequence_N_Ga_1 = self.add_jump_sizes(gamma_sequence_N_Ga_1, gamma_sequence_extension, indices=active_indices)

                # Extend x_series for all realisations using the LevyProcess functionality.
                x_series_N_Ga_1 = self.add_jump_sizes(x_series_N_Ga_1, x_series_extension, indices=active_indices)

                # Determine current truncation level for each realisation.
                epochs_N_Ga_1 = self.get_largest_poisson_epoch(gamma_sequence_N_Ga_1)
                truncation_level_N_Ga_1 = self.gamma_process.h_gamma(epochs_N_Ga_1)

                # Compute moments of the residual jumps.
                residual_expected_value_N_Ga_1 =  rate*self.gamma_process.unit_expected_residual(truncation_level_N_Ga_1)
                residual_variance_N_Ga_1 = rate*self.gamma_process.unit_variance_residual(truncation_level_N_Ga_1)
                
            else:

                gamma_sequence_extension, x_series_extension = self.simulate_left_bounding_series_setting_1_alternative(
                    rate=rate,
                    M=self.M_gamma,
                    gamma_0=epochs_N_Ga_2[active_indices],
                    size=active_indices.size
                )
                
                # Extend gamma sequence for all realisations.
                gamma_sequence_N_Ga_2 = self.add_jump_sizes(gamma_sequence_N_Ga_2, gamma_sequence_extension, indices=active_indices)

                # Extend x_series for all realisations using the LevyProcess functionality.
                x_series_N_Ga_2 = self.add_jump_sizes(x_series_N_Ga_2, x_series_extension, indices=active_indices)

                # Determine current truncation level for each realisation.
                epochs_N_Ga_2 = self.get_largest_poisson_epoch(gamma_sequence_N_Ga_2)
                truncation_level_N_Ga_2 = self.gamma_process2.h_gamma(epochs_N_Ga_2)

                # Compute moments of the residual jumps.
                residual_expected_value_N_Ga_2 =  rate*self.gamma_process2.unit_expected_residual(truncation_level_N_Ga_2)
                residual_variance_N_Ga_2 = rate*self.gamma_process2.unit_variance_residual(truncation_level_N_Ga_2)
            
            # Form the final jump sequence.
            x_series = self.add_jump_sizes(x_series, x_series_extension, indices=active_indices)

            # Compute moments of the residual jumps.
            residual_expected_value = residual_expected_value_N_Ga_1 + residual_expected_value_N_Ga_2 + residual_expected_value_N2
            residual_variance = residual_variance_N_Ga_1 + residual_variance_N_Ga_2 + residual_variance_N2

            # For each realisation select process with largest (worst) truncation level.
            selections = np.argmax(np.array([truncation_level_N_Ga_1, truncation_level_N_Ga_2, truncation_level_N2]), axis=0)
            ## Then randomly select a process to simulate based on empirical counts of requirement.
            selection = np.random.choice(selections)

            # Set tolerance level.
            _mean_lower_bound, _var_lower_bound = self.residual_stats(rate, truncation_level_gamma=truncation_level_N2, truncation_level_TS=truncation_level_N2)
            E_c = self.tolerance*self.sum(x_series) + _mean_lower_bound

            # Update conditions.
            condition1 = (residual_variance/((E_c - residual_expected_value)**2) > self.pt)
            condition2 = (E_c < residual_expected_value)
            active_indices = np.where(np.logical_or(condition1, condition2))[0]

            idx += 1
            if idx > max_iter:
                print('Max iter reached.')
                break

        truncation_level = truncation_level_N2
        return x_series, truncation_level

    def simulate_left_bounding_series_setting_1(self, rate, M=10, gamma_0=0.0, size=1):
        z1 = self.cornerpoint()
        gamma_sequence, x_series = self.gamma_process._simulate_from_series_representation(rate=rate, M=M, gamma_0=gamma_0, size=size)

        # Thin x_series further. Simulate auxiliary variables z and thin x_series based on acceptance probabilities.
        for i in range(size):
            envelope_fnc = (((2*self.delta**2)**self.abs_lam)*incgammal(self.abs_lam, (z1**2)*x_series[i]/(2*self.delta**2))*self.abs_lam*(1+self.abs_lam)
                        /((x_series[i]**self.abs_lam)*(z1**(2*self.abs_lam))*(1+self.abs_lam*np.exp(-(z1**2)*x_series[i]/(2*self.delta**2)))))
            u = np.random.uniform(low=0.0, high=1.0, size=x_series[i].size)
            x_series[i] = x_series[i][u < envelope_fnc]
            u_z = np.random.uniform(low=0.0, high=1.0, size=x_series[i].size) 
            z_series = np.sqrt(((2*self.delta**2)/x_series[i])*gammaincinv(self.abs_lam, u_z*gammainc(self.abs_lam, (z1**2)*x_series[i]/(2*self.delta**2))))
            hankel_squared = self.H_squared(z_series)
            acceptance_prob = 2/(hankel_squared*np.pi*((z_series**(2*self.abs_lam))/(z1**(2*self.abs_lam-1))))
            u = np.random.uniform(low=0.0, high=1.0, size=x_series[i].size)
            x_series[i] = x_series[i][u < acceptance_prob]
        return gamma_sequence, x_series

    def simulate_left_bounding_series_setting_1_alternative(self, rate, M=10, gamma_0=0.0, size=1):
        z1 = self.cornerpoint()
        gamma_sequence, x_series = self.gamma_process2._simulate_from_series_representation(rate=rate, M=M, gamma_0=gamma_0, size=size)

        # Thin x_series further. Simulate auxiliary variables z and thin x_series based on acceptance probabilities.
        for i in range(size):
            envelope_fnc = (((2*self.delta**2)**self.abs_lam)*incgammal(self.abs_lam, (z1**2)*x_series[i]/(2*self.delta**2))*self.abs_lam*(1+self.abs_lam)
                        /((x_series[i]**self.abs_lam)*(z1**(2*self.abs_lam))*(1+self.abs_lam*np.exp(-(z1**2)*x_series[i]/(2*self.delta**2)))))
            u = np.random.uniform(low=0.0, high=1.0, size=x_series[i].size)
            x_series[i] = x_series[i][u < envelope_fnc]
            u_z = np.random.uniform(low=0.0, high=1.0, size=x_series[i].size) 
            z_series = np.sqrt(((2*self.delta**2)/x_series[i])*gammaincinv(self.abs_lam, u_z*gammainc(self.abs_lam, (z1**2)*x_series[i]/(2*self.delta**2))))
            hankel_squared = self.H_squared(z_series)
            acceptance_prob = 2/(hankel_squared*np.pi*((z_series**(2*self.abs_lam))/(z1**(2*self.abs_lam-1))))
            u = np.random.uniform(low=0.0, high=1.0, size=x_series[i].size)
            x_series[i] = x_series[i][u < acceptance_prob]
        return gamma_sequence, x_series

    def simulate_right_bounding_series_setting_1(self, rate, M=100, gamma_0=0.0, size=1):
        z1 = self.cornerpoint()
        gamma_sequence, x_series = self.tempered_stable_process.simulate_from_series_representation(rate=rate, M=M, gamma_0=gamma_0, size=size)

        # Thin x_series further. Simulate auxiliary variables z and thin x_series based on acceptance probabilities.
        for i in range(size):
            envelope_fnc = incgammau(0.5, (z1**2)*x_series[i]/(2*self.delta**2))/(np.sqrt(np.pi)*np.exp(-(z1**2)*x_series[i]/(2*self.delta**2)))
            u = np.random.uniform(low=0.0, high=1.0, size=x_series[i].size)
            x_series[i] = x_series[i][u < envelope_fnc]
            u_z = np.random.uniform(low=0.0, high=1.0, size=x_series[i].size) 
            z_series = np.sqrt(((2*self.delta**2)/x_series[i])*gammaincinv(0.5, u_z*(gammaincc(0.5, (z1**2)*x_series[i]/(2*self.delta**2)))
                                                            + gammainc(0.5, (z1**2)*x_series[i]/(2*self.delta**2))))
            hankel_squared = self.H_squared(z_series)
            acceptance_prob = 2/(hankel_squared*z_series*np.pi)
            u = np.random.uniform(low=0.0, high=1.0, size=x_series[i].size)
            x_series[i] = x_series[i][u < acceptance_prob]
        return gamma_sequence, x_series

    def simulate_adaptive_combined_series_setting_2(self, rate=1.0, size=1):

        # Start simulating a fixed number of jumps from each process.
        gamma_sequence_N_Ga_1, x_series_N_Ga_1 = self.simulate_left_bounding_series_setting_2(rate=rate, M=self.M_gamma, size=size)
        gamma_sequence_N_Ga_2, x_series_N_Ga_2 = self.simulate_left_bounding_series_setting_2_alternative(rate=rate, M=self.M_gamma, size=size)
        gamma_sequence_N2, x_series_N2 = self.simulate_right_bounding_series_setting_2(rate=rate, M=self.M_stable, size=size)

        # Extend x_series for all realisations using the LevyProcess functionality.
        x_series = self.add_jump_sizes(x_series_N_Ga_1, x_series_N_Ga_2, indices=np.arange(size))
        x_series = self.add_jump_sizes(x_series, x_series_N2, indices=np.arange(size))

        # Determine current truncation level for each realisation.
        epochs_N_Ga_1 = self.get_largest_poisson_epoch(gamma_sequence_N_Ga_1)
        truncation_level_N_Ga_1 = self.gamma_process.h_gamma(epochs_N_Ga_1)

        epochs_N_Ga_2 = self.get_largest_poisson_epoch(gamma_sequence_N_Ga_2)
        truncation_level_N_Ga_2 = self.gamma_process2.h_gamma(epochs_N_Ga_2)

        epochs_N2 = self.get_largest_poisson_epoch(gamma_sequence_N2)
        truncation_level_N2 = self.tempered_stable_process.h_stable(epochs_N2)

        # Compute moments of the residual jumps.
        residual_expected_value_N_Ga_1 = rate*self.gamma_process.unit_expected_residual(truncation_level_N_Ga_1)
        residual_variance_N_Ga_1 = rate*self.gamma_process.unit_variance_residual(truncation_level_N_Ga_1)
        residual_expected_value_N_Ga_2 = rate*self.gamma_process2.unit_expected_residual(truncation_level_N_Ga_2)
        residual_variance_N_Ga_2 = rate*self.gamma_process2.unit_variance_residual(truncation_level_N_Ga_2)
        residual_expected_value_N2 = rate*self.tempered_stable_process.unit_expected_residual(truncation_level_N2)
        residual_variance_N2 = rate*self.tempered_stable_process.unit_variance_residual(truncation_level_N2)
        residual_expected_value = residual_expected_value_N_Ga_1 + residual_expected_value_N_Ga_2 + residual_expected_value_N2
        residual_variance = residual_variance_N_Ga_1 + residual_variance_N_Ga_2 + residual_variance_N2

        # For each realisation select process with largest (worst) truncation level.
        selections = np.argmax(np.array([truncation_level_N_Ga_1, truncation_level_N_Ga_2, truncation_level_N2]), axis=0)
        ## Then randomly select a process to simulate based on empirical counts of requirement.
        selection = np.random.choice(selections)

        # Set tolerance level.
        _mean_lower_bound, _var_lower_bound = self.residual_stats(rate, truncation_level_gamma=truncation_level_N2, truncation_level_TS=truncation_level_N2)
        E_c = self.tolerance*self.sum(x_series) + _mean_lower_bound

        # Set probabilistic upper bounds on the residual error and determine rejection conditions.
        max_iter = 50
        idx = 0
        condition1 = (residual_variance/((E_c - residual_expected_value)**2) > self.pt)
        condition2 = (E_c < residual_expected_value)
        active_indices = np.where(np.logical_or(condition1, condition2))[0]

        # Continue adding new jumps until all realisations are sufficiently converged.
        while (active_indices.size > 0):

            if (selection == 2):

                # Simulate new jumps based on the last values of the gamma sequence for each realisation.
                gamma_series_extension, x_series_extension = self.simulate_right_bounding_series_setting_2(
                    rate=rate,
                    M=self.M_stable,
                    gamma_0=epochs_N2[active_indices],
                    size=active_indices.size
                )

                # Extend gamma sequence for all realisations.
                gamma_sequence_N2 = self.add_jump_sizes(gamma_sequence_N2, gamma_series_extension, indices=active_indices)

                # Extend x_series for all realisations using the LevyProcess functionality.
                x_series_N2 = self.add_jump_sizes(x_series_N2, x_series_extension, indices=active_indices)

                # Determine current truncation level for each realisation.
                epochs_N2 = self.get_largest_poisson_epoch(gamma_sequence_N2)
                truncation_level_N2 = self.tempered_stable_process.h_stable(epochs_N2)

                # Compute moments of the residual jumps.
                residual_expected_value_N2 = rate*self.tempered_stable_process.unit_expected_residual(truncation_level_N2)
                residual_variance_N2 = rate*self.tempered_stable_process.unit_variance_residual(truncation_level_N2)

            elif (selection == 0):

                # Simulate new jumps based on the last values of the gamma sequence for each realisation.
                gamma_sequence_extension, x_series_extension = self.simulate_left_bounding_series_setting_2(
                    rate=rate,
                    M=self.M_gamma,
                    gamma_0=epochs_N_Ga_1[active_indices],
                    size=active_indices.size
                )

                # Extend gamma sequence for all realisations.
                gamma_sequence_N_Ga_1 = self.add_jump_sizes(gamma_sequence_N_Ga_1, gamma_sequence_extension, indices=active_indices)

                # Extend x_series for all realisations using the LevyProcess functionality.
                x_series_N_Ga_1 = self.add_jump_sizes(x_series_N_Ga_1, x_series_extension, indices=active_indices)

                # Determine current truncation level for each realisation.
                epochs_N_Ga_1 = self.get_largest_poisson_epoch(gamma_sequence_N_Ga_1)
                truncation_level_N_Ga_1 = self.gamma_process.h_gamma(epochs_N_Ga_1)

                # Compute moments of the residual jumps.
                residual_expected_value_N_Ga_1 =  rate*self.gamma_process.unit_expected_residual(truncation_level_N_Ga_1)
                residual_variance_N_Ga_1 = rate*self.gamma_process.unit_variance_residual(truncation_level_N_Ga_1)

            else:

                gamma_sequence_extension, x_series_extension = self.simulate_left_bounding_series_setting_2_alternative(
                    rate=rate,
                    M=self.M_gamma,
                    gamma_0=epochs_N_Ga_2[active_indices],
                    size=active_indices.size
                )
                
                # Extend gamma sequence for all realisations.
                gamma_sequence_N_Ga_2 = self.add_jump_sizes(gamma_sequence_N_Ga_2, gamma_sequence_extension, indices=active_indices)

                # Extend x_series for all realisations using the LevyProcess functionality.
                x_series_N_Ga_2 = self.add_jump_sizes(x_series_N_Ga_2, x_series_extension, indices=active_indices)

                # Determine current truncation level for each realisation.
                epochs_N_Ga_2 = self.get_largest_poisson_epoch(gamma_sequence_N_Ga_2)
                truncation_level_N_Ga_2 = self.gamma_process2.h_gamma(epochs_N_Ga_2)

                # Compute moments of the residual jumps.
                residual_expected_value_N_Ga_2 =  rate*self.gamma_process2.unit_expected_residual(truncation_level_N_Ga_2)
                residual_variance_N_Ga_2 = rate*self.gamma_process2.unit_variance_residual(truncation_level_N_Ga_2)

            # Form the final jump sequence.
            x_series = self.add_jump_sizes(x_series, x_series_extension, indices=active_indices)

            # Compute moments of the residual jumps.
            residual_expected_value = residual_expected_value_N_Ga_1 + residual_expected_value_N_Ga_2 + residual_expected_value_N2
            residual_variance = residual_variance_N_Ga_1 + residual_variance_N_Ga_2 + residual_variance_N2

            # For each realisation select process with largest (worst) truncation level.
            selections = np.argmax(np.array([truncation_level_N_Ga_1, truncation_level_N_Ga_2, truncation_level_N2]), axis=0)
            ## Then randomly select a process to simulate based on empirical counts of requirement.
            selection = np.random.choice(selections)

            # Set tolerance level.
            _mean_lower_bound, _var_lower_bound = self.residual_stats(rate, truncation_level_gamma=truncation_level_N2, truncation_level_TS=truncation_level_N2)
            E_c = self.tolerance*self.sum(x_series) + _mean_lower_bound

            # Update conditions.
            condition1 = (residual_variance/((E_c - residual_expected_value)**2) > self.pt)
            condition2 = (E_c < residual_expected_value)
            active_indices = np.where(np.logical_or(condition1, condition2))[0]

            idx += 1
            if idx > max_iter:
                print('Max iter reached.')
                break

        truncation_level = truncation_level_N2
        return x_series, truncation_level

    def simulate_left_bounding_series_setting_2(self, rate, M, gamma_0=0.0, size=1):
        z0 = self.cornerpoint()
        H0 = z0*self.H_squared(z0)
        gamma_sequence, x_series = self.gamma_process._simulate_from_series_representation(rate=rate, M=M, gamma_0=gamma_0, size=size)

        # Thin x_series further. Simulate auxiliary variables z and thin x_series based on acceptance probabilities.
        for i in range(size):
            envelope_fnc = (((2*self.delta**2)**self.abs_lam)* incgammal(self.abs_lam, (z0**2)*x_series[i]/(2*self.delta**2))*self.abs_lam*(1+self.abs_lam)/
                            ((x_series[i]**self.abs_lam)*(z0**(2*self.abs_lam))*(1+self.abs_lam*np.exp(-(z0**2)*x_series[i]/(2*self.delta**2)))))
            u = np.random.uniform(low=0.0, high=1.0, size=x_series[i].size)
            x_series[i] = x_series[i][u < envelope_fnc]
            u_z = np.random.uniform(low=0.0, high=1.0, size=x_series[i].size) 
            z_series = np.sqrt(((2*self.delta**2)/x_series[i])*gammaincinv(self.abs_lam, u_z*(gammainc(self.abs_lam, (z0**2)*x_series[i]/(2*self.delta**2)))))
            hankel_squared = self.H_squared(z_series)
            acceptance_prob = H0/(hankel_squared*((z_series**(2*self.abs_lam))/(z0**(2*self.abs_lam-1))))
            u = np.random.uniform(low=0.0, high=1.0, size=x_series[i].size)
            x_series[i] = x_series[i][u < acceptance_prob]
        return gamma_sequence, x_series
    
    def simulate_left_bounding_series_setting_2_alternative(self, rate, M, gamma_0=0.0, size=1):
        z0 = self.cornerpoint()
        H0 = z0*self.H_squared(z0)
        gamma_sequence, x_series = self.gamma_process2._simulate_from_series_representation(rate=rate, M=M, gamma_0=gamma_0, size=size)

        # Thin x_series further. Simulate auxiliary variables z and thin x_series based on acceptance probabilities.
        for i in range(size):
            envelope_fnc = (((2*self.delta**2)**self.abs_lam)* incgammal(self.abs_lam, (z0**2)*x_series[i]/(2*self.delta**2))*self.abs_lam*(1+self.abs_lam)/
                ((x_series[i]**self.abs_lam)*(z0**(2*self.abs_lam))*(1+self.abs_lam*np.exp(-(z0**2)*x_series[i]/(2*self.delta**2)))))
            u = np.random.uniform(low=0.0, high=1.0, size=x_series[i].size)
            x_series[i] = x_series[i][u < envelope_fnc]
            u_z = np.random.uniform(low=0.0, high=1.0, size=x_series[i].size) 
            z_series = np.sqrt(((2*self.delta**2)/x_series[i])*gammaincinv(self.abs_lam, u_z*(gammainc(self.abs_lam, (z0**2)*x_series[i]/(2*self.delta**2)))))
            hankel_squared = self.H_squared(z_series)
            acceptance_prob = H0/(hankel_squared*((z_series**(2*self.abs_lam))/(z0**(2*self.abs_lam-1))))
            u = np.random.uniform(low=0.0, high=1.0, size=x_series[i].size)
            x_series[i] = x_series[i][u < acceptance_prob]
        return gamma_sequence, x_series

    def simulate_right_bounding_series_setting_2(self, rate, M, gamma_0=0.0, size=1):
        z0 = self.cornerpoint()
        H0 = z0*self.H_squared(z0)
        gamma_sequence, x_series = self.tempered_stable_process.simulate_from_series_representation(rate=rate, M=M, gamma_0=gamma_0, size=size)

        # Thin x_series further. Simulate auxiliary variables z and thin x_series based on acceptance probabilities.
        for i in range(size):
            envelope_fnc = gammaincc(0.5, (z0**2)*x_series[i]/(2*self.delta**2))
            u = np.random.uniform(low=0.0, high=1.0, size=x_series[i].size)
            x_series[i] = x_series[i][u < envelope_fnc]
            u_z = np.random.uniform(low=0.0, high=1.0, size=x_series[i].size)
            z_series = np.sqrt(((2*self.delta**2)/x_series[i])*gammaincinv(0.5, u_z*(gammaincc(0.5, (z0**2)*x_series[i]/(2*self.delta**2)))
                                                                +gammainc(0.5, (z0**2)*x_series[i]/(2*self.delta**2))))
            hankel_squared = self.H_squared(z_series)
            acceptance_prob = H0/(hankel_squared*z_series)
            u = np.random.uniform(low=0.0, high=1.0, size=x_series[i].size)
            x_series[i] = x_series[i][u < acceptance_prob]
        return gamma_sequence, x_series

    def simulate_points(self, rate, low, high, n_particles=1):
        """Returns the times and jump sizes associated with the point process representation of a Levy process.
        The truncation level is required for making a Gaussian approximation of the residual process.
        """
        x_series, truncation_level = self.simulate_jumps(rate=rate, size=n_particles)
        t_series = self.simulate_jump_times(x_series, low, high)

        residual_t_series, residual_jumps = self.simulate_residual(
            low=low,
            high=high,
            truncation_level_gamma=truncation_level,
            truncation_level_TS=truncation_level,
            n_realisations=n_particles
        )

        t_series, x_series = self.add_jumps(t_series, x_series, residual_t_series, residual_jumps)
        return t_series, x_series
