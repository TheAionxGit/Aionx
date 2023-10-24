"""
aionx metrics.py is a file for storing metric classes.

The module contains:
    
    - The 'WindowDataset' class:
        A class for generating window pairs of (X, y). 
        
    - The 'Seq2SeqDataset' class:
        A class for generating sequence-to-sequence window pairs of (X, y)
        
    - The 'RegressionMatrix' class:
        A class for generating 2d regression-ready pairs of (X, y)
        
"""

# Author: Mikael Frenette (mik.frenette@gmail.com)
# Created: September 2023

import pandas as pd
import numpy as np
from typing import Union, List
from aionx import base

class WindowDataset(base.WindowHandler):
    
    """
    DESCRIPTION
    ---------------------------------------------------------------------------
    This callable class transforms a pandas dataframe into pairs of (Xs, ys)
    windows for time series forecasting. The shapes of features and labels are
    as follows: (batch, in_steps, dim) and (batch, out_steps, len(targets)),
    respectively. The data types are numpy arrays.
    
    Also, please note that if a sampler is provided (optional), calling the 
    class will return a set of training data (i.e, X_train, y_train) and a set
    of validation data (i.e, X_val, y_val). If no sampler is provided and
    targets is set to None, calling the class will only return regression
    inputs (i.e X_train).
    ---------------------------------------------------------------------------

    PARAMETERS
    ----------
    
    in_steps  : The number of time steps to consider for input sequences.
    
    out_steps : The number of time steps to predict for output sequences.
    
    horizon   : The time horizon for prediction.
    
    targets   : A list of column names to be used as target variables.
    
    sampler   : A sampler object that splits data between training and 
                validation. Defaults to None.
                             
    RETURNS
        None
        
    USAGE
    -----
    # without sampler
    generator = WindowDataset(in_steps=6, out_steps=1,
                              horizon=1, targets=["your_targets_name"])
    X_train, y_train = generator(your_training_data)
    
    #with sampler
    sampler = TimeSeriesBlockBootstrap(block_size=8, sampling_rate=0.8,
                                       replace=True)
    
    generator = WindowDataset(in_steps=6, out_steps=1,
                              horizon=1, targets=["your_targets_name"],
                              sampler=sampler)
    X_train, y_train, X_val, y_val = generator(your_training_data)
    
    """
    
    def __init__(self, targets:list, in_steps:int, horizon:int, out_steps:int=1,
                 sampler:base.Bootstrapper=None, **kwargs)->None:
        super().__init__(targets=targets,
                         in_steps=in_steps,
                         horizon=horizon,
                         out_steps=out_steps,
                         **kwargs)
        self.sampler = sampler
    
    def __call__(self, data:pd.DataFrame)->list:
        """
        PARAMETERS:
        -----------
        data : The input Pandas DataFrame containing time series data.
               The data must have the targets.
               
        RETURNS
            list containing the regression-ready window data.
        """
        # step1: create the windows
        windows = self.create_windows(data)
        
        if self.sampler is not None:
            # perform sampling and return sampled and out-of-bag indices
            self.train_idx, self.oob_idx = self.sampler(
                windows, return_indices=True
            )

            # split X_train/y_train and X_val/y_val
            train_windows = [windows[i] for i in self.train_idx]
            val_windows = [windows[i] for i in self.oob_idx]
            X_train = np.stack([x for x, _ in train_windows], axis=0)
            y_train = np.stack([y for _, y in train_windows], axis=0)
            X_val = np.stack([x for x, _ in val_windows], axis=0)
            y_val = np.stack([y for _, y in val_windows], axis=0)
            
            if self.targets is None:
                return [X_train, X_val]
            else:
                return [X_train, y_train, X_val, y_val]
        
        else:
            # split X_train/y_train and X_val/y_val
            X_train = np.stack([x for x, _ in windows], axis=0)
            y_train = np.stack([y for _, y in windows], axis=0)
            
            if self.targets is None:
                return X_train
            else:
                return [X_train, y_train]



class Seq2SeqDataset(base.TimeSeriesDataset):
    """
    DESCRIPTION
    ---------------------------------------------------------------------------
    This callable class transforms a pandas dataframe into pairs of (Xs, ys)
    windows for time series sequence-to-sequence forecasting.
    The shapes of features and labels are as follows: (batch, in_steps, dim)
    and (batch, in_steps, len(targets)), respectively. The data types are
    numpy arrays. The particularity of the Seq2SeqDataset is that the input
    steps must be the same as the output steps.
    
    Also, please note that if a sampler is provided (optional), calling the 
    class will return a set of training data (i.e, X_train, y_train) and a set
    of validation data (i.e, X_val, y_val). If no sampler is provided and
    targets is set to None, calling the class will only return regression
    inputs (i.e X_train).
    ---------------------------------------------------------------------------

    PARAMETERS
    ----------
    
    in_steps  : The number of time steps to consider for input sequences.
    
    out_steps : The number of time steps to predict for output sequences.
    
    horizon   : The time horizon for prediction.
    
    targets   : A list of column names to be used as target variables.
    
    sampler   : A sampler object that splits data between training and 
                validation. Defaults to None.
                             
    RETURNS
        None
        
    USAGE
    -----
    # without sampler
    generator = Seq2SeqDataset(in_steps=6, out_steps=6,
                              horizon=1, targets=["your_targets_name"])
    X_train, y_train = generator(your_training_data)
    
    #with sampler
    sampler = TimeSeriesBlockBootstrap(block_size=8, sampling_rate=0.8,
                                       replace=True)
    
    generator = WindowDataset(in_steps=6, out_steps=6,
                              horizon=1, targets=["your_targets_name"],
                              sampler=sampler)
    X_train, y_train, X_val, y_val = generator(your_training_data)
    
    RAISES
    ------
    ValueError : If the in_steps and out_steps do not match.
    """
    
    def __init__(self, targets:list, in_steps:int, horizon:int, out_steps:int=1,
                 sampler:base.Bootstrapper=None, **kwargs)->None:
        
        super().__init__(targets=targets,
                         in_steps=in_steps,
                         horizon=horizon,
                         out_steps=out_steps,
                         **kwargs)
        
        if in_steps!=out_steps:
            raise ValueError("In a sequence-to-sequence exercice, the output"+
                             "sequence must be the same length as the input." +
                             f"Input received : {in_steps},"+
                             f"output received:{out_steps}")
        
        self.sampler = sampler
        
        
    def _create_windows(self, data:pd.DataFrame)->list:

        # Get column indices
        column_indices = {name: i for i, name in enumerate(data.columns)}
        labels = [column_indices[i] for i in self.targets]
        windows = []
        
        i = 0 
        while True: # loop over first axis
            X = data.iloc[i:i + self.in_steps, :].values
            y = data.iloc[i + self.horizon : i + self.horizon + self.out_steps,
                          labels].values
            if (X.shape[0] < self.in_steps) or (y.shape[0] < self.out_steps):
                break
            else:
                windows.append((X, y)) # store pair of (x, y)
            i+=1
        return windows
    
    
    def __call__(self, data:pd.DataFrame)->list:
        """
        PARAMETERS:
        -----------
        data : The input Pandas DataFrame containing time series data.
               The data must have the targets.
               
        RETURNS
            list containing the regression-ready window data.
        """
        # step1: create the windows
        windows = self._create_windows(data)
        
        if self.sampler is not None:
            # perform sampling and return sampled and out-of-bag indices
            self.train_idx, self.oob_idx = self.sampler(windows)

            # split X_train/y_train and X_val/y_val
            train_windows = [windows[i] for i in self.train_idx]
            val_windows = [windows[i] for i in self.oob_idx]
            X_train = np.stack([x for x, _ in train_windows], axis=0)
            y_train = np.stack([y for _, y in train_windows], axis=0)
            X_val = np.stack([x for x, _ in val_windows], axis=0)
            y_val = np.stack([y for _, y in val_windows], axis=0)
            
            if self.targets is None:
                return [X_train, X_val]
            else:
                return [X_train, y_train, X_val, y_val]
        
        else:
            # split X_train/y_train and X_val/y_val
            X_train = np.stack([x for x, _ in windows], axis=0)
            y_train = np.stack([y for _, y in windows], axis=0)
                
            if self.targets is None:
                return [X_train, X_val]
            else:
                return [X_train, y_train, X_val, y_val]
            
            
            
class RegressionMatrix(base.TimeSeriesDataset):
    """
    DESCRIPTION
    ---------------------------------------------------------------------------
    This callable class transforms a pandas dataframe into pairs of (Xs, ys)
    2d numpy arrays for time series forecasting. The shapes of features and
    labels are as follows: (batch, dim) and (batch, len(targets)),
    respectively.
    ---------------------------------------------------------------------------

    PARAMETERS
    ----------
    
    in_steps  : The number of time steps to consider for input sequences.
    
    horizon   : The time horizon for prediction.
    
    targets   : A list of column names to be used as target variables.
                             
    RETURNS
        None
        
    USAGE
    -----
    generator = RegressionMatrix(in_steps=6, horizon=1,
                                 targets=["your_targets_name"])
    
    X_train, y_train = generator(your_training_data)
    """
    
    def __init__(self,
                 targets:List[str],
                 in_steps:int,
                 horizon:int)->None:
        super().__init__(targets=targets, in_steps=in_steps, horizon=horizon)
        
    def __call__(self,
                 data:Union[pd.DataFrame, pd.Series],
                 return_dataframe:bool=False
                 ):
        """
        PARAMETERS:
        -----------
        data             : The input Pandas DataFrame containing time series
                           data. The data must have the targets.
               
        return_dataframe : If True, the class call returns a pandas
                           dataframe containing the regressors and the targets.
                           the targets will be in the first columns.
               
        RETURNS
            list containing the regression-ready window data. or a pandas
            DataFrame.
        """
        # iterate over columns
        var_lags = [data.loc[:, self.targets]]
        for col in data:
            # build lag for each columns
            lags = []
            for lag in range(1, self.in_steps+1):
                sublag = data[[col]].shift(lag+self.horizon-1)
                sublag.columns = [f'{col}_L{lag}']
                lags.append(sublag)
            var_lags.append(pd.concat(lags, axis=1))
        var_lags = pd.concat(var_lags, axis=1).dropna()
        
        if return_dataframe:
            return var_lags
        else:
            return var_lags.iloc[:, len(self.targets):].values, var_lags.iloc[:, :len(self.targets)].values
            
