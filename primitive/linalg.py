import numpy as np
from primitive.parameters import ParameterInterface

# General linear algebra operation implementations:

def invert_covariance(cov, alpha=1e-9):
    """Inverts a covariance matrix that is not necessarily positive-definite. The alpha parameter is added to the diagonal before invertion.
    """
    cov = cov + alpha*np.eye(cov.shape[-2])
    # Compute the Cholesky decomposition of the covariance matrix
    L = np.linalg.cholesky(cov)
    # Invert the lower triangular matrix using forward substitutions
    Linv = np.linalg.solve(L, np.array([np.eye(cov.shape[-2]) for i in range(cov.shape[0])]))
    # Invert the upper triangular matrix using backward substitutions
    cov_inv = np.transpose(Linv, axes=[0, 2, 1]) @ Linv

    return cov_inv


# Parameterised linear algebra operator classes:

class LinearOperator(ParameterInterface):
    parameter_keys = ["shape"]

    def get_parameter_values(self):
        parameters = super().get_parameter_values()
        # Remove the shape key as it will not be changed after initialisation.
        return {key: value for key, value in parameters.items() if key not in ["shape"]}

    def compute_matrix(self, dt):
        pass

    def __call__(self, dt=None):
        return self.compute_matrix(dt=dt)