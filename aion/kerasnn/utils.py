"""
utils.py is a file for storing utility classes exclusively used in the
TensorFlow/Keras ecosystem.

The module contains:

    - The 'TensorFlowDatasetConfigurer' class:
        Used for configuring Keras and transforming data into tf.data.Dataset
        objects ready to be used.
        
    - The 'LogsTracker' class:
        An class for storing/caching metrics on the fly.
        
    - The 'ProgressBar' class:
        A class for printing a progressbar.
"""

# Author: Mikael Frenette (mik.frenette@gmail.com)
# Created: September 2023

import numpy as np
import pandas as pd
import sys
import tensorflow as tf
from typing import Union

class TensorFlowDatasetConfigurer(object):
    """
    Author: Mikael Frenette
    
    DESCRIPTION:
    ---------------------------------------------------------------------------
    A class for configuring TensorFlow datasets.
    ---------------------------------------------------------------------------
    
    PARAMETERS
    ----------
    batch_size  : The batch size for the dataset.
    
    repeat      : The number of times to repeat the dataset. Default is 1.
    
    shuffle     : Whether to shuffle the dataset. Default is False.
    
    buffer_size : The buffer size for shuffling the dataset. Default is 32.
    
    prefetch    : The number of elements to prefetch in the dataset.
                  Default is 1.
    
    USAGE
    -----
    X, y = numpy arrays
    
    configurer = DatasetConfigurer(batch_size=5, repeat=1, prefetch=1,
                                   shuffle=True, buffer_size=32)
    dataset = configurer(X=X, y=y)
    ---------------------------------------------------------------------------
    """

    def __init__(self, 
                 batch_size: int = None,
                 repeat: int = 1,
                 shuffle: bool = False,
                 buffer_size: int = tf.data.AUTOTUNE,
                 prefetch: int = 1) -> None:
        
        self.batch_size = batch_size
        self.repeat = repeat
        self.shuffle = shuffle
        self.buffer_size = buffer_size
        self.prefetch = prefetch

    def __call__(self, 
                 X: Union[tuple[Union[np.ndarray, tf.Tensor],
                                Union[np.ndarray, tf.Tensor]],
                          np.ndarray, tf.Tensor],
                 y: Union[tuple[Union[np.ndarray, tf.Tensor],
                                Union[np.ndarray, tf.Tensor]],
                          np.ndarray, tf.Tensor] = None) -> tf.data.Dataset:
        """
        PARAMETERS
        ----------
        X : The input data.
            
        y : The target data. Default is None.

        RETURNS
            tf.data.Dataset: The configured TensorFlow dataset.
        """

        if not isinstance(X, tuple):
            X = (X)
        Xs = tf.data.Dataset.from_tensor_slices(X)
        
        if y is not None:
            if not isinstance(y, tuple):
                y = (y)
            ys = tf.data.Dataset.from_tensor_slices(y)
            dataset = tf.data.Dataset.zip((Xs, ys))
            del Xs, ys
        else:
            dataset = Xs
            del Xs

        dataset = dataset.repeat(self.repeat)
        dataset = dataset.prefetch(self.prefetch)  # Corrected prefetch argument
        
        if self.shuffle:
            dataset = dataset.shuffle(self.buffer_size)
            
        if self.batch_size is None:
            dataset = dataset.batch(len(X))
        else:
            dataset = dataset.batch(self.batch_size) 
            
        return dataset
    
    
    @staticmethod
    def from_default_config(X: Union[tuple[Union[np.ndarray, tf.Tensor],
                                       Union[np.ndarray, tf.Tensor]],
                                    np.ndarray, tf.Tensor],
                           y: Union[tuple[Union[np.ndarray, tf.Tensor],
                                      Union[np.ndarray, tf.Tensor]],
                                    np.ndarray, tf.Tensor] = None) -> tf.data.Dataset:
        """
        Create and configure a TensorFlow dataset.
    
        PARAMETERS
        ----------
        X : The input data.
        
        y : The target data. Default is None.
    
        RETURNS
            tf.data.Dataset: The configured TensorFlow dataset.
        """
        if not isinstance(X, tuple):
            X = (X)
        Xs = tf.data.Dataset.from_tensor_slices(X)
        
        if y is not None:
            if not isinstance(y, tuple):
                y = (y)
            ys = tf.data.Dataset.from_tensor_slices(y)
            dataset = tf.data.Dataset.zip((Xs, ys))
        else:
            dataset = Xs

        dataset = dataset.repeat(1)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)  # Corrected prefetch argument
        dataset = dataset.shuffle(len(X))
        dataset = dataset.batch(32)

        return dataset


class LogsTracker(object):
    """
    DESCRIPTION
    ---------------------------------------------------------------------------
    The LogsTracker class is designed for the purpose of storing loss and
    metric values during training. This is achieved by maintaining a dictionary
    where each metric is associated with a list.

    Access to the log history can be obtained using the .history property.
    It's important to note that the reset_state() method must be called before
    utilizing it.
    ---------------------------------------------------------------------------
    """

    def __init__(self):
        self._logs = {}
        
    def __getitem__(self, metric:str):
        return self._logs[metric][-1]

    def __setitem__(self, metric:str, value:float):
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
    """
    DESCRIPTION
    ---------------------------------------------------------------------------
    A custom progress bar for displaying training progress. This class allows
    for displaying a progress bar during training, indicating the progress of
    epochs and steps.
    ---------------------------------------------------------------------------

    PARAMETERS
    ----------
    total_epochs        : Total number of epochs.
    
    steps_per_epoch     : Total number of steps per epoch.
    
    number_of_trainings : Total number of training processes. Defaults to 1.
    
    length              : Length of the progress bar. Defaults to 20.
    
    fill                : Character used to fill the progress bar.
                          Defaults to '█'.
    """
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
        PARAMETERS
        ----------
        prefix : Prefix string displayed before the progress bar.
        
        epoch  : Current epoch.
        
        step   : the substep in the entire epoch. also called the batch size
                 step.
                 
        logs   : a dict containing the losses/metrics which will be printed out.
                 The dict must contain float values and the keys must be the 
                 desired name to be displayed.
                 
        RETURNS
            None.
        """
        epoch_progress = epoch + 1
        epoch_percent = round(100 * (epoch_progress / int(self.total_epochs)))

        filled_length = int(
            self.length * (epoch_progress - 1) // self.total_epochs) + 1
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