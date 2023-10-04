import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import sys


class LogsTracker(object):
    """
    -------------------------------------------------------------------------
    The LogsTracker class is used for storing losses and metrics value 
    during training. It does so by updating a dict() which contains lists
    associated to each metric.
    
    One can have access to the log history by the .history property. Also, 
    we must call the reset_state() method before using it.
    -------------------------------------------------------------------------
    """

    def __init__(self):
        self._logs = {}
        
    def __getitem__(self, metric):
        return self._logs[metric][-1]

    def __setitem__(self, metric, value):
        if metric not in self._logs.keys():
            self._logs[metric] = []
            
        if tf.is_tensor(value):
            self._logs[metric].append(value)
        
    def __len__(self):
        return len(self._logs)
    
    def last_log(self):
        last_log = {}
        for key, val in self._logs.items():
            last_log[key] = val[-1]
        return last_log
    
    def reset_state(self):
        self._logs = {}
        
    @property
    def logs(self):
        return self._logs
    
    @property
    def history(self):
        return pd.DataFrame.from_dict(self._logs)
    
    
class ProgressBar(object):
    
    def __init__(self, 
                 name:str,
                 total_epochs:int,
                 steps_per_epoch:int,
                 number_of_trainings:int=1,
                 length:int=20, fill='█')->None:

        self.name = name
        self.length = length
        self.fill = fill
        self.total_epochs = total_epochs
        self.steps_per_epoch = steps_per_epoch
        self.number_of_trainings = number_of_trainings
        
    def __call__(self, prefix:str, epoch:int, step:int, logs:dict):
        
        """
        Parameters:
        -----------
            prefix (str):
                Prefix string displayed before the progress bar.
            epoch (int):
                Current epoch.
            step (int):
                the substep in the entire epoch. also called the batch size
                step.
            logs (dict):
                a dict containing the losses/metrics which will be printed out.
                The dict must contain float values and the keys must be the 
                desired name to be displayed.
        """
        
        epoch_progress = epoch + 1
        epoch_percent = round(100 * (epoch_progress / int(self.total_epochs)))

        filled_length = int(self.length * (epoch_progress - 1) // self.total_epochs) + 1
        bar = self.fill * filled_length + '-' * (self.length - filled_length)
            
        base_template = ["\r"
            f"{self.name} {prefix}",
            f"|{bar}| Step: {step}/{self.steps_per_epoch}",
            f"| Epoch: {epoch_progress}/{self.total_epochs} ({epoch_percent}%)",
        ]
        
        for name, val in logs.items():     
            metric_format = f"| {name}: {val:.4f}"   
            base_template.append(metric_format)
        
        progress = " ".join(base_template)
        sys.stdout.write(progress)
        sys.stdout.flush()
        
        

        
    
def add_trends(data, trends):
    """
    This function adds trend columns to a given dataset.

    Parameters:
    data (pd.DataFrame): The original dataset.
    trends (int): The number of trend columns to add.

    Returns:
    data (pd.DataFrame): The dataset with added trend columns.
    """

    try:
        # Create trend columns as separate dataframes with values ranging from 0 to len(data)
        ntrends = [pd.DataFrame(np.arange(0, len(data), 1), columns=[f"trend_{i}"]) for i in range(trends)]

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
    A class for handling window estimation in time series data.

    Attributes:
        data (pandas.DataFrame): The input time series data.
        expanding_start (int): The index where out-of-sample (OOS) data begins.
        timesteps (int): The number of time steps in each window.
        last_window (optional): The index of the last window to consider.

    Methods:
        __init__(self, data, expanding_start, timesteps, last_window=None): Initializes the WindowEstimationHandler object.
        __getitem__(self, idx): Returns the estimation block at the given index.
        __len__(self): Returns the number of estimation blocks.
    """

    def __init__(self, data, expanding_start, timesteps, last_window=None):
        """
        Initializes the WindowEstimationHandler object.

        Args:
            data (pandas.DataFrame): The input time series data.
            expanding_start (int): The index where out-of-sample (OOS) data begins.
            timesteps (int): The number of time steps in each window.
            last_window (optional): The index of the last window to consider.
        """

        # protected attributes
        self.__data = data
        self.__timesteps = timesteps
        self.__last_window = last_window

        # private attribute
        self._expanding_start = expanding_start

        # target values
        self._estimation_blocs = []

        # compute the number of expansions that will occur
        n_training_periods = len(self.__data.loc[:self._expanding_start])
        n_oos_periods = len(self.__data.loc[self._expanding_start:])
        n_expansions = int(np.ceil(n_oos_periods / timesteps))

        for expansion in range(n_expansions):
            training_end_idx = n_training_periods + (self.__timesteps * expansion)
            forecast_end_idx = n_training_periods + (self.__timesteps * (expansion + 1))

            # Check if last_window is reached
            if self.__last_window is not None:
                if self.__data.iloc[training_end_idx, :].name > self.__last_window:
                    break
                else:
                    self._estimation_blocs.append(
                        (
                            self.__data.iloc[:training_end_idx],
                            self.__data.iloc[training_end_idx:]
                        )
                    )
            else:
                self._estimation_blocs.append(
                    (
                        self.__data.iloc[:training_end_idx],
                        self.__data.iloc[training_end_idx:]
                    )
                )

    def __getitem__(self, idx):
        """
        Returns the estimation block at the given index.

        Args:
            idx (int): Index of the estimation block.

        Returns:
            tuple: A tuple containing the training and testing data of the estimation block.
        """
        return self._estimation_blocs[idx]

    def __len__(self):
        """
        Returns the number of estimation blocks.

        Returns:
            int: Number of estimation blocks.
        """
        return len(self._estimation_blocs)
