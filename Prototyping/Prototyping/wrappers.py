

import numpy as np
import pandas as pd
import time

import sys
# set up libraries directory
sys.path.append("C:/Users/User/dropbox/Prototyping")


def Precision(target_dtype):
    """
    Decorator to convert output result to a specified data type.
    
    This decorator is applied to a function and converts its output result
    to the specified data type, while preserving the original structure.
    
    Args:
        target_dtype (str): The target data type to which the output result will be converted.
        
    Returns:
        decorator: A decorator function that performs the conversion.
        
    """
    
    def decorator(function):
        def wrapper(*args, **kwargs):
            # Call the original function to obtain the result
            result = function(*args, **kwargs)

            if isinstance(result, (pd.DataFrame, pd.Series)):
                # Convert DataFrame or Series to the specified dtype
                result = result.astype(target_dtype)
            elif isinstance(result, (np.ndarray, np.number)):
                # Convert list or NumPy numeric value to the specified dtype
                result = np.array(result, dtype=target_dtype)
            else:
                if isinstance(result, (int, float)):
                    if target_dtype == "float32":
                        result = np.float32(result)
                    elif target_dtype == "float64":
                        result = np.float64(result)
                    elif target_dtype == "int32":
                        result = np.int32(result)            
                    elif target_dtype == "int64":
                        result = np.int64(result)
                else:
                    raise TypeError(f"Cannot convert dtype {type(result)} to {target_dtype}.")
            
            return result
        return wrapper
    return decorator

def timeit(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        hours, remainder = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        print(f" completed in {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}")
        return result
    return wrapper


def DataType(type):
    """
    Decorator that converts function output to the specified data type.

    Args:
        type (str): The target data type. Must be either "dataframe" or "array".

    Returns:
        decorator: A decorator function that performs the conversion.
    """
    def decorator(function):
        def wrapper(*args, **kwargs):
            """
            Wrapper function that converts function output to the specified data type.

            Args:
                *args: Positional arguments passed to the wrapped function.
                **kwargs: Keyword arguments passed to the wrapped function.

            Returns:
                object: Resulting data in the specified data type.
            """
            # Call the original function to obtain the result
            result = function(*args, **kwargs)

            if type == "dataframe":
                if isinstance(result, pd.Series):
                    # Convert Series to DataFrame with a single column
                    result = result.to_frame(result.name)
                elif isinstance(result, (pd.DataFrame, np.number)):
                    # Already a DataFrame, no conversion needed
                    pass
                elif isinstance(result, np.ndarray):
                    # Convert NumPy array to DataFrame
                    result = pd.DataFrame(result)
                else:
                    raise TypeError(f"Cannot convert dtype {type(result)} to {type}.")

            elif type == "array":
                if isinstance(result, (pd.Series, pd.DataFrame)):
                    # Convert Pandas object to NumPy array
                    result = result.values
                elif isinstance(result, np.number):
                    # Convert single NumPy value to array
                    result = np.array(result)
                elif isinstance(result, np.ndarray):
                    # Already a NumPy array, no conversion needed
                    pass
                else:
                    raise TypeError(f"Cannot convert dtype {type(result)} to {type}.")

            else:
                raise ValueError(f"Type value must be either 'dataframe' or 'array'")

            return result
        return wrapper
    return decorator

def dataframe_inputs(func):
    """
    A decorator that converts Pandas Series to DataFrame and checks for DataFrame inputs.

    This decorator takes a function as input and returns a new function (wrapper).
    The new function converts Pandas Series inputs into DataFrames, and it also checks
    if the other arguments are DataFrames. If the inputs are not valid, it raises an error.

    Args:
        func (function): The original function to be decorated.

    Returns:
        function: The wrapped function that handles Series and DataFrame inputs.
    """
    def wrapper(self, *args, **kwargs):
        """
        The wrapper function that handles Series and DataFrame inputs.

        This function converts Pandas Series inputs to DataFrames while preserving the column name.
        It also checks if the other arguments are DataFrames. If the inputs are not valid, it raises an error.

        Args:
            *args: Positional arguments passed to the wrapped function.
            **kwargs: Keyword arguments passed to the wrapped function.

        Returns:
            The result of the wrapped function with converted DataFrame inputs.
        """
        
        new_args = []
        for arg in args:
            if isinstance(arg, pd.Series):
                # Convert Series to DataFrame and preserve the column name
                if arg.name is None:
                    arg.name = "column"
                new_args.append(arg.to_frame(arg.name))
            elif isinstance(arg, pd.DataFrame):
                new_args.append(arg)
            else:
                raise TypeError(f"Inputs should be a pandas.DataFrame or pandas.Series, not {type(arg)}.")
        
        # Call the original function with the new DataFrame inputs
        return func(self, *new_args, **kwargs)
    
    return wrapper

def series_to_dataframe(func):
    def wrapper(self, inputs):
        if isinstance(inputs, pd.Series):
               inputs = inputs.to_frame(inputs.name)
        result = func(self, inputs)
        return result
    return wrapper


