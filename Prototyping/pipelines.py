

import pandas as pd
import numpy as np
import sys
from typing import Union, List, Iterable

sys.path.append("C:/Users/User/dropbox/Prototyping")
from Prototyping import base

class WindowDataset(base.WindowHandler):
    """
    ---------------------------------------------------------------------------------------
    This callable class will transform a pandas dataframe into pairs of (Xs, ys) windows 
    for time series forecasting. The shapes of features and labels are : 
    (batch, in_steps, dim) and (batch, out_steps, len(targets)) respectively. Types 
    are numpy arrays.
    --------------------------------------------------------------------------------------

    PARAMETERS
    ----------
    in_steps (int): The number of time steps to consider for input sequences.
    out_steps (int): The number of time steps to predict for output sequences.
    horizon (int): The horizon to predict.
    target (list): List of column names to be used as target variables.
    sampler (Booststrapper, optional): sampler object which will split between training and
                             validation data. Default to None.
    """
    
    def __init__(self, targets:list, in_steps:int, horizon:int, out_steps:int=1,
                 sampler:base.Bootstrapper=None, **kwargs)->None:
        super().__init__(targets=targets,
                         in_steps=in_steps,
                         horizon=horizon,
                         out_steps=out_steps,
                         **kwargs)
        self.sampler = sampler
    
    def __call__(self, data:pd.DataFrame, inference_mode:bool=False):
        """
        PARAMETERS:
        -----------
        data (pd.DataFrame): The input Pandas DataFrame containing time series data.

        inference_mode (bool, optional): If True, the function operates in inference mode,
                                        where output sequences are not provided,
                                        suitable for predictions on unseen data.
        
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
                return X_train
            else:
                return [X_train, y_train]



class Seq2SeqDataset(base.TimeSeriesDataset):
    """
    ---------------------------------------------------------------------------------------
    This callable class will transform a pandas dataframe into pairs of (Xs, ys) windows 
    for time series forecasting. The shapes of features and labels are : 
    (batch, in_steps, dim) and (batch, out_steps, len(targets)) respectively. Types 
    are numpy arrays.
    --------------------------------------------------------------------------------------

    PARAMETERS
    ----------
    in_steps (int): The number of time steps to consider for input sequences.
    out_steps (int): The number of time steps to predict for output sequences.
    horizon (int): The horizon to predict.
    target (list): List of column names to be used as target variables.
    sampler (Booststrapper, optional): sampler object which will split between training and
                             validation data. Default to None.
    """
    
    def __init__(self, targets:list, in_steps:int, horizon:int, out_steps:int=1,
                 sampler:base.Bootstrapper=None, **kwargs)->None:
        super().__init__(targets=targets,
                         in_steps=in_steps,
                         horizon=horizon,
                         out_steps=out_steps,
                         **kwargs)
        self.sampler = sampler
        
        
    def _create_windows(self, data:pd.DataFrame)->list:
        """
        This private method will loop over the dataframe on the first axis and will store 
        pairs of inputs and outputs
        """
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
    
    
    def __call__(self, data:pd.DataFrame, inference_mode:bool=False)->list:
        """
        PARAMETERS:
        -----------
        data (pd.DataFrame): The input Pandas DataFrame containing time series data.

        inference_mode (bool, optional): If True, the function operates in inference mode,
                                        where output sequences are not provided,
                                        suitable for predictions on unseen data.
        
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
    This class is used for creating regression matrix from time series data.
    It starts by defining configurations such as lags, horizon and targets 
    then generate a matrix ready for regression. 
    
    IMPORTANT NOTE:
        - the first column in the matrix is the target data.

    PARAMETERS:
    -----------
        targets (List[str]):
            List of target variable names.
        in_steps (int):
            Number of input steps (lags).
        horizon (int):
            Number of time steps to forecast into the future.
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
        
        dataset = pd.DataFrame(index = data.index)
        
        # keep target's data as it is.
        dataset.loc[:, self.targets] = data.loc[:, self.targets].values
        
        # iterate over columns
        for col in data:
            # build lag for each columns
            for lag in range(1, self.in_steps+1):
                dataset.loc[:, f"{col}_L{lag}"] = data[col].shift(
                    self.in_steps+self.horizon-1)
        if return_dataframe:
            return dataset
        else:
            return dataset.values
            