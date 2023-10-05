

import pandas as pd
import numpy as np
import sys
import tensorflow as tf
from typing import Union, List, Iterable
sys.path.append("C:/Users/User/dropbox/Prototyping")
from Prototyping import base


class WindowPipeline(base.WindowHandler):
    """
    ---------------------------------------------------------------------------------------
    This callable class will transform a pandas dataframe into a TensorFlow tf.data.Dataset
    for time series forecasting.
    
    The pipeline will generate pairs of inputs/outputs with the shape
    (batch, in_steps, dim) and (batch, out_steps, len(targets)) respectively.
    --------------------------------------------------------------------------------------

    PARAMETERS
    ----------
    in_steps (int): The number of time steps to consider for input sequences.
    out_steps (int): The number of time steps to predict for output sequences.
    horizon (int): The horizon to predict.
    target (list): List of column names to be used as target variables.
    sampler (Booststrapper): sampler object which will split between training and
                             validation data.
        
    The __call__ method
    -------------------
    
    The __call__ method will do 4 things:

    1) create the windows.
    2) will perform bootstrap on the windows if a sampler is provided.
    3) Will make further dataset preparation such as batch size and window shuffling.
    4) Returns a data.Dataset object which can be iterated on. Also, One can use this
       data.Dataset object on model.fit()

    Also, the inference mode must be set to True when making final predictions. This
    will remove the targets from the windowing which will allow the while loop to store
    additional input windows. 

    Example:
    --------
    if one wants to predict the target "y" one period in the future by using 12 past
    time steps:

    pipe = WindowPipeline(in_steps = 12, out_steps = 1, horizon = 1, targets = ["y"],
                            sampler=None)
    dataset = pipe(dataframe, batch_size=32, shuffle=False)

    network = model()
    network.compile(...)
    network.fit(dataset, ...)
    
    """
    
    def __init__(self, targets:list, in_steps:int, horizon:int, out_steps:int=1,
                 sampler:base.Bootstrapper=None, **kwargs)->None:
        
        super().__init__(targets=targets,
                         in_steps=in_steps,
                         horizon=horizon,
                         out_steps=out_steps,
                         **kwargs)
        self.sampler = sampler

    def __call__(self, data:pd.DataFrame, batch_size:int=None, shuffle:bool=False,
                 inference_mode:bool=False)->tf.data.Dataset:
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
            
            # map indices to windows index
            train_windows = [windows[i] for i in self.train_idx]
            val_windows = [windows[i] for i in self.oob_idx]
            
            # build data.Dataset.from_generator object.
            train_generator = self.__BuildBluePrints(train_windows,
                                                     batch_size, shuffle)
            val_generator   = self.__BuildBluePrints(val_windows,
                                                     batch_size, shuffle=False)
            
            return [train_generator, val_generator]
        
        else:
            # build data.Dataset.from_generator object. 
            train_generator = self.__BuildBluePrints(windows, batch_size, shuffle)

            return train_generator


    def __BuildBluePrints(self, windows, batch_size, shuffle)->tf.data.Dataset:
        """
        Build a TensorFlow dataset generator from the windowed dataset.

        Parameters:
            windows (list): List of input-output pairs representing the windowed dataset.
            batch_size (int): Size of each batch in the dataset.
            shuffle (bool): Whether to shuffle the dataset.
            buffer_size (int): Size of the shuffle buffer.

        Returns:
            dataset (tf.data.Dataset): TensorFlow dataset generator.
        """
        def generator_fn(windows):
            """
            Generator function that yields batches of input-output pairs.
            """
            for x, y in windows:
                yield x, y

        dataset = tf.data.Dataset.from_generator(
                lambda: generator_fn(windows),
                output_signature=(
                    tf.TensorSpec(shape=(self.in_steps, windows[0][0].shape[-1]),
                                  dtype=tf.float32),
                    tf.TensorSpec(shape=(self.out_steps, len(self.targets)),
                                  dtype=tf.float32)
                    ))
        
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE) 
        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(windows))
        if batch_size is None:
            dataset = dataset.repeat(1).batch(len(windows))
        else:
            dataset = dataset.repeat(1).batch(batch_size)
        return dataset
    
    
class TensorFlowDatasetConfigurer(object):
    """
    A class for configuring TensorFlow datasets.

    PARAMETERS:
    -----------
        batch_size (int):
            The batch size for the dataset.
        repeat (int):
            The number of times to repeat the dataset. Default is 1.
        shuffle (bool):
            Whether to shuffle the dataset. Default is False.
        buffer_size (int):
            The buffer size for shuffling the dataset. Default is 32.
        prefetch (int):
            The number of elements to prefetch in the dataset. Default is 1.
            
    USAGE:
    ------
        X, y = numpy arrays
    
        configurer = DatasetConfigurer(batch_size=5, repeat=1, prefetch=1,
                                       shuffle=True, buffer_size=32)
        dataset = configurer(X=X, y=y)
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
        Create and configure a TensorFlow dataset.

        PARAMETERS:
        -----------
            X: (Union[tuple[Union[np.ndarray, tf.Tensor],
                            Union[np.ndarray, tf.Tensor]],
                    np.ndarray, tf.Tensor]):
                
                The input data.
                
            y: (Union[tuple[Union[np.ndarray, tf.Tensor],
                            Union[np.ndarray, tf.Tensor]],
                    np.ndarray, tf.Tensor]):
                
                The target data. Default is None.

        Returns:
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

        PARAMETERS:
        -----------
            X: (Union[tuple[Union[np.ndarray, tf.Tensor],
                            Union[np.ndarray, tf.Tensor]],
                    np.ndarray, tf.Tensor]):
                
                The input data.
                
            y: (Union[tuple[Union[np.ndarray, tf.Tensor],
                            Union[np.ndarray, tf.Tensor]],
                    np.ndarray, tf.Tensor]):
                
                The target data. Default is None.

        Returns:
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
