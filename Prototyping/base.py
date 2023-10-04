import numpy as np 
import pandas as pd
from typing import Union, List, Iterable, Callable

class TimeSeriesDataset:
    """
    Time Series Dataset base class
    
    PARAMETERS:
    -----------
        targets (list): list of str. The name of the target(s).
                        Must be column's name(s).
        in_steps (int): the number of lags
        horizon (int): the horizon in the future to forecast
        steps   (int): number of timesteps to forecast at the same time.
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
        targets (list): list of str. The name of the target(s).
                        Must be column's name(s).
        in_steps (int): the number of lags
        horizon (int): the horizon in the future to forecast
        steps   (int): number of timesteps to forecast at the same time.     
    
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
        
    def _create_windows(self, data:Union[pd.DataFrame, pd.Series])->list:
        """
        This private method will loop over the dataframe on the first axis and will store 
        pairs of inputs and outputs
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
    ------------------------------------------------------------------------------------
    Bootstrapping Parent class. Classes inheriting from Bootstrapper will gain 2 default
    parameters: sampling rate and sampling with replacement.
    
    If the sampling rate is below 0.0 and higher than 1.0, an exception will be raised.
    ------------------------------------------------------------------------------------
    
    PARAMETERS
    ----------
    
    sampling_rate (float): An amount of 'sampling_rate' will be bootstrapped from the
                            data. Default to 0.8.
                            
    replace (bool): If the True, sampling will be done with replacement. Default to
                    True.

    """

    def __init__(self, sampling_rate: float=0.8, replace:bool=False)->None:
        if not (0.0 < sampling_rate < 1.0):
            raise ValueError("sampling_rate value must be a float >0.0 and <1.0")
        else:
            self._sampling_rate = sampling_rate
        self._replace = replace

    @property
    def sampling_rate(self):
        return self._sampling_rate
    
    @property
    def replace(self):
        return self._replace
    
    
    