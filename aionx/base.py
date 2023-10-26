"""
aionx base.py is a file for storing base classes exclusively used in the
in the entire prototyping ecosystem.

The module contains:

    - The 'TimeSeriesDataset' class:
        parent class for time series data intput formatting. 
        
    - The 'WindowHandler' class:
        inherits from TimeSeriesDataset. parent class for window base formatting 
        child class.
        
    - The 'Bootstrapper' class:
        parent class for bootstrapping classes.
        
    - The 'Metric' class:
        parent class for metrics computation.
        
    - The 'RegressionMetric' class:
        parent class for computing regression evaluation metrics. Inherits from
        Metric
        
    - The 'DensityMetric class':
        Parent class for computin density evaluation metrics Inherits from
        Metric.
        
    - The 'Scaler' class:
        A base class for Scaling / unscaling pandas data.
"""

# Author: Mikael Frenette (mik.frenette@gmail.com)
# Created: September 2023

import numpy as np 
import pandas as pd
from typing import Union, List

class TimeSeriesDataset:
    """
    DESCRIPTION
    ---------------------------------------------------------------------------
    Time Series Dataset base class.
    ---------------------------------------------------------------------------
    
    PARAMETERS
    ----------
    targets  : The name of the target(s). Must be column's name(s).
    
    in_steps : the number of lags.
    
    horizon  : the horizon in the future to forecast.
    
    out_steps: number of timesteps to forecast at the same time.
    
    
    """
    def __init__(self,
                 targets:List[str],
                 in_steps:int,
                 horizon:int,
                 out_steps:int=1)->None:
        """
        initialize
        """
        self._targets  = targets
        self._in_steps    = in_steps
        self._horizon = horizon
        self._out_steps   = out_steps
        
    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self._in_steps + self._out_steps}',
            f'Input indices: {list(np.arange(0, self._in_steps))}',
            f'Label indices: {list(np.arange(self._in_steps+self._horizon-1, self._in_steps+self._horizon+self._out_steps-1))}',
            f'Label column name(s): {self._targets}'])
    
    @property
    def targets(self):
        return self._targets
    @property
    def in_steps(self):
        return self._in_steps
    @property
    def horizon(self):
        return self._horizon
    @property
    def out_steps(self):
        return self._out_steps
    
    @targets.setter
    def targets(self, value):
        self._targets = value
    @in_steps.setter
    def in_steps(self, value):
        self._in_steps = value
    @horizon.setter
    def horizon(self, value):
        self._horizon = value
    @out_steps.setter
    def out_steps(self, value):
        self._out_steps = value



class WindowHandler(TimeSeriesDataset):
    """
    Window construction base class.
    
    PARAMETERS:
    -----------
    targets  : list of str. The name of the target(s).
                    Must be column's name(s).
                    
    in_steps : the number of lags
    
    horizon  : the horizon in the future to forecast
    
    out_steps: number of timesteps to forecast at the same time.     
    
    """
    def __init__(self,
                 targets:List[str],
                 in_steps:int,
                 horizon:int,
                 out_steps:int=1,
                 **kwargs):
        
        super().__init__(targets=targets,
                         in_steps=in_steps,
                         horizon=horizon,
                         out_steps=out_steps,
                         **kwargs)
        
    def create_windows(self, data:Union[pd.DataFrame, pd.Series])->list:
        """
        PARAMETERS
        ----------
        data : data to format into windows.

        RETURNS
            list of windows. contains tuples of pairing (X, y)

        """
        if isinstance(data, pd.Series):
            data = data.to_frame(data.name)
            
        if self.targets is None:
            self.targets = data.columns
            
        # Get column indices
        column_indices = {name: i for i, name in enumerate(data.columns)}
        labels = [column_indices[i] for i in self.targets]
        windows = []
        
        i = 0 
        while True: # loop over first axis
            X = data.iloc[i:i + self.in_steps, :].values
            y = data.iloc[
                i + self.in_steps + self.horizon - 1:i + self.in_steps +
                self.out_steps + self.horizon - 1, labels].values
            if (X.shape[0] < self.in_steps) or (y.shape[0] < self.out_steps):
                break
            else:
                windows.append((X, y)) # store pair of (x, y)
            i+=1
        return windows
    

    
class Bootstrapper(object):
    """
    DESCRIPTION
    ---------------------------------------------------------------------------
    Bootstrapping Parent class. Classes inheriting from Bootstrapper will gain
    2 default parameters: sampling rate and sampling with replacement.
    
    If the sampling rate is below 0.0 and higher than 1.0, an exception will be
    raised.
    ---------------------------------------------------------------------------
    
    PARAMETERS
    ----------
    sampling_rate: An amount of 'sampling_rate' will be bootstrapped from the
                            data. Default to 0.8.
                            
    replace      : If the True, sampling will be done with replacement.
                   Default to True.
        
    RETURNS
        None
    """

    def __init__(self, sampling_rate: float=0.8,
                 replace:bool=False)->None:
        
        if not (0.0 < sampling_rate < 1.0):
            raise ValueError("sampling_rate value must be a float >0.0 and <1.0")
        else:
            self._sampling_rate = sampling_rate
        self._replace = replace
        
        self.iterations = 0
        self.reset_state() 
        
    def reset_state(self):
        """
        DESCRIPTION
        -------------------------------------------------------------------------
        resets internal history
        -------------------------------------------------------------------------
        """
        self._state = {}
        
    def _configure(self, inputs):
        if inputs is None:
            return None
        
        else:
            return (inputs, ) if not isinstance(inputs, tuple) else inputs

    def _compute_length(self, X, y):
        if X is not None:
            return len(X[0])
        else:
            if y is not None:
                return len(y[0])
            else:
                raise ValueError("Input for X or y or both must be provided. Found None.")
        
    def _state_history(self, subset):
        history = []
        for it, values in self.state.items():
            for key, vals in values.items():
                if key == subset:
                    history.append(vals)
        return history
        
    @property
    def sampling_rate(self):
        return self._sampling_rate
    
    @property
    def replace(self):
        return self._replace

    @property
    def state(self):
        return self._state
  
    
class Metric(object):
    """
    DESCRIPTION
    ---------------------------------------------------------------------------
    Parent class for metrics computation
    ---------------------------------------------------------------------------
    
    PARAMETERS
    ----------
    precision  : The data type that will be outputed from child's __call__
                method. One of { float32,  float64 }
    
    tolerance : The numerical tolerance parameter. this value will be used
                to control for potential underflow issues coming from the
                computations
                    
    RETURNS
        None
    
    RAISES
    ------
    ValueError : if the input for the precision parameter is invalid. 
    """
    def __init__(self, precision:str="float32",
                       tolerance:float=1e-7,
                       name:str=None)->None:
        self.precision = precision
        self.tolerance = tolerance
        self.name = name
        if self.precision not in ["float32", "float64"]:
            raise ValueError("Input for precision received {precision}. Valid" +
                             "inputs are float32 or float64")
    def _configure(self, y_true:np.ndarray, y_pred:np.ndarray)->tuple:
        if not isinstance(y_true, np.ndarray):
            y_true = np.squeeze(np.array(y_true))
        if not isinstance(y_pred, np.ndarray):
            y_pred = np.squeeze(np.array(y_pred))   
        return y_true, y_pred


class RegressionMetric(Metric):
    """
    DESCRIPTION
    ---------------------------------------------------------------------------
    Parent class for Regression metrics. Inherits from Metric.
    ---------------------------------------------------------------------------
    
    RAISES
    ------
    ValueError : if y_true and y_pred do not have the same shape
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def __call__(self, y_true:np.ndarray, y_pred:np.ndarray):
        """
        PARAMETERS
        ----------
        y_true : True values.
        
        y_pred : Predicted value

        RETURNS
            tuple (y_true, y_hat)
        """
        y_true, y_pred = self._configure(y_true, y_pred)
        if y_true.shape != y_pred.shape:
            raise ValueError("Input arrays must have the same shape.")
            
            
class DensityMetric(Metric):
    """
    DESCRIPTION
    ---------------------------------------------------------------------------
    Parent class for Density metrics. Inherits from Metric.
    ---------------------------------------------------------------------------
    
    RAISES
    ------
    ValueError : if y_true and y_pred do not have the same shape
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def __call__(self, y_true:np.ndarray, y_pred:Union[List[np.ndarray],
                                                       np.ndarray]):
        """
        PARAMETERS
        ----------
        y_true : True values.
        
        y_pred : Predicted value

        RETURNS
            tuple (y_true, y_hat)
        """
        y_true, y_pred = self._configure(y_true, y_pred)
        y_pred = np.split(y_pred,
                          indices_or_sections=y_pred.shape[-1], axis=1)    
        if y_true.shape != y_pred[0].shape !=y_pred[1].shape:
            raise ValueError("Input arrays must have the same shape.")
        return y_true, y_pred
    
    
    
class Scaler(object):
    """
    DESCRIPTION
    ---------------------------------------------------------------------------
    The  base class for Scalers. child inheriting from this class will have a
    get_stats method which will be able to gather information about the
    summary statistics of a provided pandas DataFrame.
    ---------------------------------------------------------------------------
    """

    def __init__(self) -> None:
        self._stats = {}
        
    def get_stats(self, data: pd.DataFrame) -> None:
        """
        Calculate and store summary statistics for data columns.

        PARAMETERS
        ----------
        data : The input data for which to calculate statistics.

        RETURNS
            None
        """
        
        if isinstance(data, pd.Series):
            data = data.to_frame(data.name)
        
        for col in data:
            summarystat = {}
            summarystat["mean"] = data[col].mean()
            summarystat["std"] = data[col].std()
            self._stats[col] = summarystat
        
    @property
    def stats(self) -> dict:
        return self._stats
    
