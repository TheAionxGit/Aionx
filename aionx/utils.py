"""
aionx utils.py is a file for storing/dumping utility classes.

The module contains:

    - The 'add_trends' function:
        A function for adding trends to the provided dataframe.
        
    - The 'WindowEstimationHandler' class:
        generates expanding windows from the provided pandas DataFrame.

"""

# Author: Mikael Frenette (mik.frenette@gmail.com)
# Created: August 2023


import pandas as pd
import numpy as np
from datetime import datetime
from typing import Union
from aionx.wrappers import downstream

@downstream(
    pd.DataFrame, dtype="float32"
)
def add_trends(data: pd.DataFrame, trends: int) -> pd.DataFrame:
    """
    DESCRIPTION
    ---------------------------------------------------------------------------
    Add trend columns to a given pandas DataFrame.
    ---------------------------------------------------------------------------

    PARAMETERS
    ----------
    data   : The original dataframe.

    trends : The number of trend columns to add.

    RETURNS
    -------
        The dataframe with added trend columns.
    """

    try:
        # Create trend columns as separate dataframes with values ranging
        #from 0 to len(data)
        ntrends = [pd.DataFrame(np.arange(0, len(data), 1),
                                columns=[f"trend_{i}"]) for i in range(trends)]

        # Concatenate the trend columns horizontally
        ntrends = pd.concat(ntrends, axis=1)

        # Set the index of ntrends to match the index of the original dataset
        ntrends.index = data.index

        # Concatenate the original dataset and the trend columns vertically
        data = pd.concat([data, ntrends], axis=1)
    except Exception as e:
        print(e)
    
    return data




class ExpandingWindowGenerator:
    """
    DESCRIPTION
    ---------------------------------------------------------------------------
    This class is designed for managing the estimation windows in time series
    data. it generates expanding windows for the training as well as providing
    the remaining out-of-sample data.
    ---------------------------------------------------------------------------

    PARAMETERS
    ----------
    data            : The input time series data.
    
    expanding_start : The index where out-of-sample (OOS) data begins.
    
    timesteps       : The number of time steps in each window.
    
    last_window     : The index of the last window to consider.
    
    verbose         : the level of verbosity

    EXAMPLE
    -------
    # Initialize a WindowEstimationHandler
    expanding_window = WindowEstimationHandler(your_data,
                                        expanding_start=2007-01-01,
                                        timesteps = 12,
                                        last_window="2020-01-01")
    for step, train, oos in expanding_window:
        ...
        your code here
        ...
        
    """
    def __init__(self, 
                 data:pd.DataFrame,
                 expanding_start:Union[str, datetime],
                 timesteps:int,
                 last_window:Union[str, datetime],
                 verbose:int=0) -> None:

        self.data = data
        self.expanding_start=expanding_start
        self.timesteps = timesteps
        self.last_window = last_window
        self.verbose=verbose
        self._it = 0    

        if isinstance(self.last_window, str):
            self.last_window = datetime.strptime(self.last_window, "%Y-%m-%d")
        if isinstance(self.expanding_start, str):
            self.expanding_start = datetime.strptime(self.expanding_start, "%Y-%m-%d")
        self.starting_idx = len(self.data.loc[
            :self.data.index[self.data.index < self.expanding_start].max()
        ])  
        self.max_it = len(self.data.loc[self.expanding_start:self.last_window]) / self.timesteps
        
    def __iter__(self):
        return self
    
    def __update_state(self):
        if self._it >= self.max_it:
            raise StopIteration
        else:
            self._start_point = 0
            self._end_point = self.starting_idx+(self.timesteps*(self._it))
        self._it += 1
            
    def __next__(self):           
        self.__update_state()
        train_slice = self.data.iloc[self._start_point:self._end_point]
        poos_slice  = self.data.iloc[self._end_point:]
        if self.verbose == 1:
            print(f"Expanding[{self._it}/{int(self.max_it)}]")
        if self.verbose > 1:
            print(f"Expanding[{self._it}/{int(self.max_it)}] : \n", 
                    f"    Estimation start: {train_slice.index[0].strftime('%Y-%m-%d')}",
                    f" - Estimation end: {train_slice.index[-1].strftime('%Y-%m-%d')}") 
        return self._it, train_slice, poos_slice
