import numpy as np

# Base inference module class:

class InferenceModule:

    def __init__(self, model):
        """A model defines the parameters to be learned and returns a log likelihood value (and possibly a vector of gradients) given a set of parameters and measurements.
        A prior is a PriorModule instance or a dictionary of parameter keys and PriorModule instances.
        A proposal is a ProposalModule instance or a dictionary of parameter keys and ProposalModule instances.
        """

        # Set model:
        self.model = model

        # Initalise instance parameters to be learned.
        self.parameters = self.model.get_parameter_values()

    def set_training_variables(self, y, X, Xeval):
        self.y = y
        self.X = X

        if Xeval is None:
            self.Xeval = self.X
        else:
            self.Xeval = Xeval

    def initialise(self):
        pass

    def sample(self):
        pass
    
    def fit(self, y, X, n_samples=10):
        pass