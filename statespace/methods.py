import numpy as np

from primitive.linalg import invert_covariance
from primitive.methods import InferenceModule

from scipy.special import logsumexp

# Kalman filtering recursion functions:

def kalman_prediction(mu_init, C_init, F, m, Q):
    """Kalman prediction algorithm for a linear dynamical model.

    - Representation of the initial distribution.
    The initial distribution of the states may be a single Gaussian represented by its mean 'mu_init' and covariance 'C_init' with shapes (Dx1) and (DxD) respectively.
    Additionally, it may be represented by Gaussian particles by an extended array with shapes (NpxDx1) and (NpxDxD) respectively where Np is the number of particles.

    - Linear dynamical model.
    The parameters defining the linear dynamical model are 'F' which is the transition matrix (DxD), and 'm', 'Q' are the mean (Dx1) and covariance (DxD) of the driving Gaussian process. 
    For multiple particles, the parameters 'F', 'm' and 'Q' can be shared or each particle may have a separate setting. In this case the shapes must be extended to (NpxDxD), (NpxDx1) 
    and (NpxDxD) respectively.

    Returns mu_pred and C_pred which are the predictive moments of the dynamically evolved density.
    """
    mu_ts = F @ mu_init + m
    C_ts = F @ C_init @ F.T + Q
    return mu_ts, C_ts

def kalman_correction(y, mu_pred, C_pred, H, Omega_eps):
    """Kalman correction algorithm for a linear model.

    - Representation of the initial distribution.
    The initial distribution of the states may be a single Gaussian represented by its mean 'mu_pred' and covariance 'C_pred' with shapes (Dx1) and (DxD) respectively.
    Additionally, it may be represented by Gaussian particles by an extended array with shapes (NpxDx1) and (NpxDxD) respectively where Np is the number of particles.

    - Linear dynamical model.
    The parameters defining the linear model are 'H' which is a fixed observation model (D_primexD), and 'Omega_eps' (D_primexD_prime) is the observation noise covariance matrix.
    This inherently assumes the observation noise is Gaussian. A particle representation of the observation noise can be applied by extending 'Omega_eps' to (NpxD_primexD_prime).

    - Log marginal likelihood.
    The log likelihood of the model given observation 'y' and the initial distribution represented by 'mu_pred', 'C_pred'.

    Returns mu_est, C_est and log_marginal_likelihood which are the filtering mean, covariance and the log marginal likelihood.
    """

    # Kalman correction:
    residual_pred = y - H @ mu_pred
    residual_pred_cov = H @ C_pred @ H.T + Omega_eps

    K_t = C_pred @ H.T @ np.linalg.inv(residual_pred_cov)
    mu_tt = mu_pred + K_t @ residual_pred
    C_tt = C_pred - K_t @ H @ C_pred

    # Calculate log marginal likelihood:
    log_det = np.linalg.slogdet(residual_pred_cov)[1].reshape(-1, 1, 1)
    log_marginal_likelihood = -0.5 * (log_det + np.swapaxes(residual_pred, -1, -2) @ np.linalg.inv(residual_pred_cov) @ residual_pred + y.shape[-2] * np.log(2 * np.pi))

    return mu_tt, C_tt, log_marginal_likelihood


# Base Kalman filtering class:

class KalmanFilter(InferenceModule):

    def initialise_memory(self, mu_prior, C_prior, mu_est, C_est, log_marginal_likelihood):

        D = mu_prior.shape[-2]
        
        field_names = ['predictive_mean', 'predictive_cov', 'filtered_mean', 'filtered_cov', 'log_marginal_likelihood']
        field_dtypes = ['f4', 'f4', 'f4', 'f4', 'f4']
        field_shapes = [(D, 1), (D, D), (D, 1), (D, D), (1,1,1)]

        datatype = np.dtype([(field_names[i], field_dtypes[i], field_shapes[i]) for i in range(len(field_names))])
        memory = np.array([(mu_prior, C_prior, mu_est, C_est, log_marginal_likelihood)], dtype=datatype)

        return memory, datatype

    def filter(self, times, y, mu_prior, C_prior):

        mu_est, C_est, log_marginal_likelihood = kalman_correction(y=y[0], mu_pred=mu_prior, C_pred=C_prior, H=self.model.H, Omega_eps=self.model.eps.covariance(t=times[0]))
        memory, datatype = self.initialise_memory(mu_prior, C_prior, mu_est, C_est, log_marginal_likelihood)

        for i in range(1, times.shape[0]):

            dt = (times[i] - times[i-1])
            
            F = self.model.F(dt)
            m, Q = self.model.I.moments(s=times[i-1], t=times[i], n_particles=1)

            mu_pred, C_pred = kalman_prediction(mu_init=memory['filtered_mean'][-1], C_init=memory['filtered_cov'][-1], F=F, m=m, Q=Q)
            mu_est, C_est, log_marginal_likelihood = kalman_correction(y=y[i], mu_pred=mu_pred, C_pred=C_pred, H=self.model.H, Omega_eps=self.model.eps.covariance(times[i]))

            # Save variables:
            memory = np.concatenate((memory, np.array([(mu_pred, C_pred, mu_est, C_est, log_marginal_likelihood)], dtype=datatype)))

        return memory
    
# Sequential Collapsed Gaussian MCMC class for normal variance-mean processes.

class SequentialCollapsedGaussianMCMCFilter(InferenceModule):

    def initialise_memory(self, mu_est, C_est, log_marginal_likelihood):

        D = mu_est.shape[-2]
        
        field_names = ['filtered_mean', 'filtered_cov', 'log_marginal_likelihood']
        field_dtypes = ['f4', 'f4', 'f4']
        field_shapes = [(D, 1), (D, D), (1,1,1)]

        datatype = np.dtype([(field_names[i], field_dtypes[i], field_shapes[i]) for i in range(len(field_names))])
        memory = np.array([(mu_est, C_est, log_marginal_likelihood)], dtype=datatype)

        return memory, datatype
    
    def initialise_chain(self, conditional_mean, conditional_cov, log_conditional_likelihood):

        D = conditional_mean.shape[-2]

        field_names = ['conditional_mean', 'conditional_cov', 'log_conditional_likelihood']
        field_dtypes = ['f4', 'f4', 'f4']
        field_shapes = [(D, 1), (D, D), (1,1,1)]

        datatype = np.dtype([(field_names[i], field_dtypes[i], field_shapes[i]) for i in range(len(field_names))])
        chain = np.array([(conditional_mean, conditional_cov, log_conditional_likelihood)], dtype=datatype)

        return chain, datatype
    
    def get_Gaussian_mixture_moments(self, chain, burn_in=0):
        """This method computes the Gaussian mixture mean and covariance of the MCMC samples in the current iteration.
        """

        means = chain['conditional_mean'][burn_in:]
        covs = chain['conditional_cov'][burn_in:]

        post_mix_mean = means.mean(axis=0)

        residual_mean = (means - post_mix_mean)
        mixture_adjustment = residual_mean @ np.transpose(residual_mean, axes=(0,2,1))

        post_mix_cov = (covs + mixture_adjustment).mean(axis=0)

        return post_mix_mean, post_mix_cov

    def filter(self, times, y, mu_prior, C_prior, n_samples=10, burn_in=0):

        # Acceptance probs.
        acceptance_probs = []

        # Flatten times.
        times = times.flatten()

        # Initialise filtering density and memory.
        mu_est, C_est, log_marginal_likelihood = kalman_correction(y=y[0], mu_pred=mu_prior, C_pred=C_prior, H=self.model.H, Omega_eps=self.model.eps.covariance(t=times[0]))
        memory, datatype = self.initialise_memory(mu_est, C_est, log_marginal_likelihood)
        
        chain_memory = []

        for i in range(1, times.shape[0]):
            
            # Set fixed model parameters for the time interval.
            dt = (times[i] - times[i-1])
            F = self.model.F(dt)

            # Propose model transition:
            t_series, x_series = self.model.I.subordinator.simulate_points(rate=dt, low=times[i-1], high=times[i], n_particles=1)
            m, Q = self.model.I.moments(s=times[i-1], t=times[i], n_particles=1, t_series=t_series, x_series=x_series)

            mu_pred, C_pred = kalman_prediction(mu_init=memory['filtered_mean'][-1], C_init=memory['filtered_cov'][-1], F=F, m=m, Q=Q)
            mu_est, C_est, log_marginal_likelihood = kalman_correction(y=y[i], mu_pred=mu_pred, C_pred=C_pred, H=self.model.H, Omega_eps=self.model.eps.covariance(times[i]))

            chain, chain_datatype = self.initialise_chain(conditional_mean=mu_est, conditional_cov=C_est, log_conditional_likelihood=log_marginal_likelihood)

            # Start sampling.
            for _ in range(n_samples):
                
                # Propose model transition:
                t_series, x_series = self.model.I.subordinator.simulate_points(rate=dt, low=times[i-1], high=times[i], n_particles=1)
                m, Q = self.model.I.moments(s=times[i-1], t=times[i], n_particles=1, t_series=t_series, x_series=x_series)

                mu_pred, C_pred = kalman_prediction(mu_init=memory['filtered_mean'][-1], C_init=memory['filtered_cov'][-1], F=F, m=m, Q=Q)
                mu_est, C_est, log_marginal_likelihood = kalman_correction(y=y[i], mu_pred=mu_pred, C_pred=C_pred, H=self.model.H, Omega_eps=self.model.eps.covariance(times[i]))

                # Metropolis-Hastings:
                likelihood_ratio = np.exp(log_marginal_likelihood[0,0,0] - chain['log_conditional_likelihood'][-1][0, 0, 0])
                
                acceptance_prob = np.min([1., likelihood_ratio])
                u = np.random.uniform(low=0.0, high=1.0)

                acceptance_probs.append(acceptance_prob)

                if u < acceptance_prob:

                    # Save proposed variables:
                    chain = np.concatenate((chain, np.array([(mu_est, C_est, log_marginal_likelihood)], dtype=chain_datatype)))

                else:

                    # Save previous variables:
                    chain = np.concatenate((chain, np.array([(chain['conditional_mean'][-1], chain['conditional_cov'][-1], chain['log_conditional_likelihood'][-1])], dtype=chain_datatype)))

            # Compute Gaussian mixture moments.
            mix_log_likelihood = chain['log_conditional_likelihood'].mean(axis=0)
            post_mix_mean, post_mix_cov = self.get_Gaussian_mixture_moments(chain, burn_in=burn_in)

            # Save variables:
            memory = np.concatenate((memory, np.array([(post_mix_mean, post_mix_cov, mix_log_likelihood)], dtype=datatype)))
            chain_memory.append(chain)

        self.acceptance_rates = acceptance_probs

        return memory, chain_memory
