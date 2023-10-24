"""
aionx bootstrap.py is a file for storing data bootstrapping classes.

The module contains:

    - The 'TimeSeriesBlockBootstrap' class:
        A class for sampling blocks of time-dependent data. 
        
"""

# Author: Mikael Frenette (mik.frenette@gmail.com)
# Created: September 2023


import numpy as np
from typing import Iterable
from aionx import base

    
class TimeSeriesBlockBootstrap(base.Bootstrapper):
    """
    DESCRIPTION
    ---------------------------------------------------------------------------
    Perform block sampling on time-dependent data to preserve temporal
    structure. This class inherits from the Bootstrapper parent class and is
    designed exclusively for time series data.
    ---------------------------------------------------------------------------

    PARAMETERS
    ----------
    block_size    : The size of time-dependent blocks to be bootstrapped.

    sampling_rate : An amount of 'sampling_rate' will be bootstrapped from the
                            data. Default to 0.8.
                            
    replace       : If the True, sampling will be done with replacement.
                    Default to True.
                    
    **kwargs      : extra parameters inheriting from parent class.

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
        PARAMETERS
        ----------
        data           : the data to sample from. the sampling will be done on
                         the length of data(len(data)). so the input must be an
                         iterable with a length.
        
        return_indices : If true will return indices of the of the
                         bootstrapped data. otherwise, will return the sampled
                         data. Default to True.
                         
        RETURNS
            A tuple or a list of tuple containing two arrays:
                -bootstrap_idx : Indices of the bootstrap samples.
                -oob_idx       : Indices of the out-of-bag samples.
        """
        
        # compute data length. bootstrap will be performed across the length dimension
        # of the data
        if isinstance(data, tuple):
            N = len(data[0])
        else:
            N = len(data)
            data = tuple(data)

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
            sampled = [(np.array(dta)[bootstrap_idx], np.array(dta)[oob_idx]) for dta in data]
            return sampled[0] if len(sampled) == 1 else sampled


