# Parameterised abstract class or protocol:

import numpy as np

class ParameterInterface:
    parameter_keys = None

    def __init__(self, **kwargs):

        if self.__class__.parameter_keys is not None:
            
            # Set parameter configuration.
            self.parameters = {}
            self.initialise_parameter_values(**kwargs)

            # Binary variable to check if parameters are initialised.
            self.valid = self.check_parameter_key_initialisation()

            # Set parameter schema:
            self.set_parameter_schema()

    def get_parameter_values(self):
        # Return a dictionary of parameters.
        return self.parameters
    
    def initialise_parameter_values(self, **kwargs):
        # This method implements the same functionality as set_parameter_values, but does not validate parameter schema.

        # Filter any parameter key value pairs not related to specific module:
        new_parameters = {key: kwargs[key] for key in self.__class__.parameter_keys if key in kwargs}

        for key, value in new_parameters.items():

            # Create/update dictionary of parameters.
            self.parameters[key] = value

            # Create/modify instance parameters named after the key and stores value.
            setattr(self, key, value)
    
    def set_parameter_values(self, **kwargs):
        
        # Filter any parameter key value pairs not related to specific module:
        new_parameters = {key: kwargs[key] for key in self.__class__.parameter_keys if key in kwargs}

        for key, value in new_parameters.items():

            # Validate parameters.
            self.validate_parameters(key, value)

            # Create/update dictionary of parameters.
            self.parameters[key] = value

            # Create/modify instance parameters named after the key and stores value.
            setattr(self, key, value)

    def set_parameter_schema(self):
        self.parameter_schema = {}

        for key in self.parameters.keys():

            _type = type(self.parameters[key])
            _shape = np.shape(self.parameters[key])

            if (len(_shape) == 0):
                _shape = None

            self.parameter_schema[key] = {"type":_type, "shape":_shape}

    def check_parameter_key_initialisation(self):
        # If any key in parameter_keys are not found in the initialised parameters dictionary:
        if [parameter_key for parameter_key in self.__class__.parameter_keys if parameter_key not in self.parameters.keys()]:
            print("Parameter values are not initialised.")
            return False
        else:
            return True

    def validate_parameters(self, key, value):
        
        schema = self.parameter_schema[key]

        _type = type(value)
        _shape = np.shape(value)

        if (len(_shape) == 0):
            _shape = None

        new_schema = {"type":_type, "shape":_shape}

        for schema_key in schema.keys():

            assert schema[schema_key] == new_schema[schema_key], f"schema_key is {schema_key}."
