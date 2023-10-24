

import numpy as np
import pandas as pd
import time
from typing import Union
from functools import wraps


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


def downstream(output_type:Union[np.ndarray,
                                 pd.Series,
                                 pd.DataFrame],
               dtype:str):
    """
    DESCRIPTION
    ---------------------------------------------------------------------------
    This decorator allows you to specify the desired downstream output type for
    a function. It automatically performs the conversion as needed to ensure
    the output matches the specified type.
    
    wrapper assumes 3 possible type for the function's output:
        {np.array, pd.DataFrame, pd.Series}.
    
    If the function is not one of these type, no transformation will be
    applied.
    ---------------------------------------------------------------------------
    
    PARAMETERS
    ----------
    output_type : The desired output type, which can be one of the following:
                  {np.ndarray, pd.Series, pd.DataFrame}.
                  
    dtype       : the output data type expected by pandas or numpy.
                  {int32, int64, float32, float64}.

    RETURNS
        A wrapped function with the specified output type.
    """
    
    if output_type not in [np.ndarray, pd.DataFrame, pd.Series]:
        raise ValueError("Specified output type must be one of " +
                         "{numpy.ndarray, pandas.DataFrame, pandas.Series}. " +
                         f"input received : {output_type}.")
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            output = func(*args, **kwargs)

            # if we want numpy array as output.
            if output_type == np.ndarray:
                if isinstance(output, pd.Series) or isinstance(output,
                                                               pd.DataFrame):
                    return output.values

            # If we want a pandas Serie as output
            elif output_type == pd.Series:
                if isinstance(output, np.ndarray):
                    return pd.Series(np.squeeze(output))
                elif isinstance(output, pd.DataFrame):
                    return output.squeeze()
              
            # If we want a pandas DataFrame as output
            elif output_type == pd.DataFrame:
                if isinstance(output, pd.Series):
                    return output.to_frame(output.name)
                elif isinstance(output, np.ndarray):
                    return pd.DataFrame(output)
                
            if dtype is not None:
                output = output.astype(dtype)

            return output # return output with the specified type.

        return wrapper

    return decorator