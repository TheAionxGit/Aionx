

import numpy as np
from typing import Union, List, Iterable
import pandas as pd
from Prototyping import base

    
class TimeSeriesBlockBootstrap(base.Bootstrapper):
    """
    -----------------------------------------------------------------------------------------
    Perform block bootstrap on time-dependent data to preserve temporal structure.

    This class inherits from the Bootstrapper parent class and is designed for time series
    data.
    -----------------------------------------------------------------------------------------
    
    PARAMETERS
    ----------
    block_size : int
        The size of time-dependent blocks to be bootstrapped.

    sampling_rate (float): An amount of 'sampling_rate' will be bootstrapped from the
                            data. Default to 0.8.
                            
    replace (bool): If the True, sampling will be done with replacement. Default to
                    True.
                    
    **kwargs:
        extra parameters callable by the constructor.

    __call__
    --------
    Perform block bootstrap resampling on the input time series data.

    PARAMETERS
    ----------
    data : iterable with a __len__ method
        The input time series data.

    Returns
    -------
    tuple
        A tuple containing two arrays:
        - bootstrap_idx (numpy.ndarray): Indices of the bootstrap samples.
        - oob_idx (numpy.ndarray): Indices of the out-of-bag samples.
    """
    
    def __init__(self,
                 block_size:int,
                 sampling_rate:float=0.8,
                 replace:bool=True,
                 **kwargs)->None:
        
        super().__init__(sampling_rate=sampling_rate,
                         replace=replace,**kwargs)
        self._block_size = block_size
        
    
    def __call__(self, data:Iterable, return_indices:bool=True)->tuple:
        """
        PARAMETERS:
        -----------
            return_indices (bool): If true will return indices of the of the
                bootstrapped data. otherwise, will return the sampled data.
                Default to True.
        """
        
        # compute data length. bootstrap will be performed across the length dimension
        # of the data
        N = len(data)

        # compute number of blocks
        vec = np.arange(0, N / self._block_size, 1)

        # draw random blocks
        groups = np.sort(np.random.choice(vec, size=N, replace=self.replace).astype(int))
        rando_vec = np.random.exponential(scale=1, size=(N // self._block_size) + 1)[groups]
        rando_vec = np.where(rando_vec > np.quantile(rando_vec, 1 - self.sampling_rate))
        chosen_one_plus = rando_vec

        # reshape and split between bootstrap and out-of-bag samples
        bootstrap_idx = np.sort(chosen_one_plus).reshape(-1, )
        oob_idx = np.delete(np.arange(0, N, 1), bootstrap_idx)

        if return_indices:
            return bootstrap_idx, oob_idx
        else:
            return np.array(data)[bootstrap_idx], np.array(data)[oob_idx]


