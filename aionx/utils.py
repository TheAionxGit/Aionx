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



class WindowEstimationHandler(object):
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

    EXAMPLE
    -------
    # Initialize a WindowEstimationHandler
    expanding_window = WindowEstimationHandler(your_data,
                                        expanding_start=2007-01-01,
                                        timesteps = 12,
                                        last_window="2020-01-01")
    for train, oos in expanding_window:
        ...
        your code here
        ...
        
    """
    def __init__(self, data:pd.DataFrame,
                 expanding_start:Union[str, datetime],
                 timesteps:int,
                 last_window=Union[str, datetime]
                 )->None:
        
        if isinstance(last_window, str):
            last_window = datetime.strptime(last_window, "%Y-%m-%d")

        # protected attributes
        self.data = data
        self.timesteps = timesteps
        self.last_window = last_window

        # private attribute
        self.expanding_start = expanding_start

        # target values
        self.estimation_blocs = []

        # compute the number of expansions that will occur
        n_training_periods = len(self.data.loc[:self.expanding_start])
        n_oos_periods = len(self.data.loc[self.expanding_start:])
        n_expansions = int(np.ceil(n_oos_periods / timesteps))

        for expansion in range(n_expansions):
            training_end_idx = n_training_periods + (
                self.timesteps * expansion)
            forecast_end_idx = n_training_periods + (
                self.timesteps * (expansion + 1))

            # Check if last_window is reached
            if self.last_window is not None:
                if self.data.iloc[training_end_idx, :].name > self.last_window:
                    break
                else:
                    self.estimation_blocs.append(
                        (
                            self.data.iloc[:training_end_idx],
                            self.data.iloc[training_end_idx:]
                        )
                    )
            else:
                self._estimation_blocs.append(
                    (
                        self.data.iloc[:training_end_idx],
                        self.data.iloc[training_end_idx:]
                    )
                )

    def __getitem__(self, idx):
        return self.estimation_blocs[idx]

    def __len__(self):
        return len(self.estimation_blocs)
