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
    X             : A matrix-like containing regressors.

    y             : A matrix-like containing target variables. Default to None.
    
    block_size    : The size of time-dependent blocks to be bootstrapped.
                    Default to 8.

    sampling_rate : An amount of 'sampling_rate' will be bootstrapped from the
                            data. Default to 0.8.
                            
    replace       : If the True, sampling will be done with replacement.
                    Default to True.
                    
    **kwargs      : extra parameters inheriting from parent class.

    USAGE
    -----
    sampler = TimeSeriesBlockBootstrap(
            X=your_X, y=your_y, block_size=24, sampling_rate=0.8
    )
    X_train, y_train, X_val, y_val = next(sampler)

    """
    
    def __init__(self,
                 X,
                 y = None,
                 block_size:int=8,
                 sampling_rate:float=0.8,
                 replace:bool=True,
                 **kwargs)->None:
        
        super().__init__(sampling_rate=sampling_rate,
                         replace=replace,**kwargs)
        self.block_size = block_size
        self.X, self.y = self._configure(X), self._configure(y) 
        self.N = self._compute_length(self.X, self.y)
     
    def generate_indices(self):
        """
        DESCRIPTION
        ---------------------------------------------------------------------------
        Generates a subset of training and validation samples.
        ---------------------------------------------------------------------------

        RETURNS
            list containining a mapping indices for the bootstrapped and out-of-bag
            values.
        """
        self.iterations+=1
        # compute number of blocks
        vec = np.arange(0, self.N / self.block_size, 1)

        # draw random blocks
        groups = np.sort(np.random.choice(vec, size=self.N, replace=self.replace).astype(int))
        rando_vec = np.random.exponential(scale=1, size=(self.N // self.block_size) + 1)[groups]
        rando_vec = np.where(rando_vec > np.quantile(rando_vec, 1 - self.sampling_rate))
        chosen_one_plus = rando_vec

        bootstrap_idx = np.sort(chosen_one_plus).reshape(-1, )
        oob_idx = np.delete(np.arange(0, self.N, 1), bootstrap_idx)
        return bootstrap_idx, oob_idx
        
    def __iter__(self):
        return self

    def __next__(self):
        bootstrap_indices, oob_indices = self.generate_indices()
        self.state[f"{self.iterations}"] = {"bootstrap_indices":bootstrap_indices,
                                            "oob_indices":oob_indices}
        
        if self.X is not None:
            X_train = tuple([np.array(X)[bootstrap_indices] for X in self.X])            
            X_val = tuple([np.array(X)[oob_indices] for X in self.X])
        else:
            X_train, X_val = (None,), (None,)
            
        if self.y is not None:
            y_train = tuple([np.array(y)[bootstrap_indices] for y in self.y])      
            y_val = tuple([np.array(y)[oob_indices] for y in self.y])
        else:
            y_train, y_val = (None,), (None,)
            
        return [X_train[0] if len(X_train) == 1 else X_train,
                y_train[0] if len(y_train) == 1 else y_train,
                X_val[0] if len(X_val) == 1 else X_val,
                y_val[0] if len(y_val) == 1 else y_val]
    
    @property
    def bootstrap_indices(self):
        return self._state_history("bootstrap_indices")

    @property
    def oob_indices(self):
        return self._state_history("oob_indices")


