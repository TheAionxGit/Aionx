"""
 base.py file for storing parent classes useful entirely in the Tensorflow/
 keras Ecosystem.
 
 The module contains:
     
     the 'NetworkTrainer' base class. Used for training keras models only.
"""

# Author: Mikael Frenette (mik.frenette@gmail.com)
# Created: September 2023

import tensorflow as tf
import numpy as np 
from typing import Union, List, Callable

from aionx.kerasnn.utils import (
    TensorFlowDatasetConfigurer,
    LogsTracker,
    ProgressBar
    )


class NetworkTrainer(object):
    """
    Author: Mikael Frenette
    
    DESCRIPTION
    --------------------------------------------------------------------------
    Base class used for training a keras.models.Model. Its purpose is to mimic
    TensorFlow's model.compile() and model.fit() functionalities as closely
    as possible.
    
    The class is instantiated by specifying a keras.optimizers instance and
    a callable loss function with a signature similar to (y_true, y_pred).
    --------------------------------------------------------------------------
    
    PARAMETERS
    ----------
    optimizer : The network optimizer.
    
    loss      : The loss function used for training the network.
    
    metrics   : A list of metrics that will be tracked and printed during training.
                Default to None.
              
    n_trainings : Number of models that will be trained. Only affects the 
                  the progress bar. Default to 1.
              
    METHODS
    -------
    train : The train method begins by transforming input data into a tf.data.Dataset.
            It then iterates over the dataset for a specified number of epochs and
            applies a training and validation step as defined by the child class.
    
    PROPERTIES
    ----------
    One can access the training history logs by using the .history property
    on the trainer instance, which contains information tracked by LogsTracker.

    """
    
    def __init__(self,
                 optimizer:Union[tf.keras.optimizers.Optimizer,
                                 List[tf.keras.optimizers.Optimizer]],
                 loss:Union[Callable, tf.keras.losses.Loss,
                            List[tf.keras.losses.Loss]],
                 metrics:Union[tf.keras.metrics.Metric,
                               List[tf.keras.metrics.Metric]]=None,
                 n_trainings:int=1,
                 )->None:
        
        self.optimizer = optimizer
        self.loss_fn = loss
        self.n_trainings = n_trainings
        
        if isinstance(metrics, list):
            self.metrics = metrics
        elif metrics is None:
            self.metrics = {}
        else:
            self.metrics = [metrics]
            
        self.logger = LogsTracker()
        
        self._model_trained = 0
        
    @property
    def history(self):
        return self.logger.history()

    def train(self,
              model:tf.keras.models.Model,
              X:Union[tuple[Union[np.ndarray, tf.Tensor],
                            Union[np.ndarray, tf.Tensor]],
                              np.ndarray, tf.Tensor],
              y:Union[tuple[Union[np.ndarray, tf.Tensor],
                            Union[np.ndarray, tf.Tensor]],
                              np.ndarray, tf.Tensor],
              epochs:int,
              shuffle:bool=False,
              validation_data:tuple[Union[np.ndarray, tf.Tensor],
                                    Union[np.ndarray, tf.Tensor]]=None,
              batch_size:int=32,
              verbose:int=1,
              early_stopping:tf.keras.callbacks.EarlyStopping=None,
              validation_batch_size:int=None,
              )->None:
        
        """
        
        DESCRIPTION
        ----------------------------------------------------------------------
        The `train` method follows a structure similar to keras.fit and is
        designed for training a Keras model.
        ----------------------------------------------------------------------
        
        PARAMETERS
        ----------
        model   : The model to be trained. It does not need to be compiled.
        
        X       : The input data for the network.
        
        y       : The target data.
        
        epochs  : The number of training epochs for the model.
        
        shuffle : If set to True, only the training data will be shuffled.
                  Default to False.
                  
        validation_data : Validation data to be monitored during training.
                          Default to None
        
        batch_size      : The batch size for applying a gradient descent step. 
                          Default to 32.
                     
        verbose         : The verbosity level. A higher value prints more
                          logs during training. If set to 0, the progress bar
                          will be suppressed.
                  
        early_stopping  : Performs early stopping based on the parameters
                          of the Keras EarlyStopping callback. Default to None.
                        
        validation_batch_size : The batch size for validation steps. Default
                                to None.
        ----------------------------------------------------------------------

        """
        
        # add the model as attribute since we need it in other methods
        self.model = model
        
        # this needs to be set to false for the early stopping
        self.model.stop_training=False
        
        self._model_trained += 1
        
        # get the shape input shape
        if isinstance(X, tuple):
            # look at first instance and compute the first and last dimension
            # parameters
            N, K = X[0].shape[0], X[0].shape[-1]
            # in the case where Xs in the tuple do not have the same shape,
            # an error will be raised by tensorflow while configuring the 
            # dataset
        else:
            N, K = X.shape[0], X.shape[-1]
            
        # training dataset configuration
        train_configurer = TensorFlowDatasetConfigurer(
            batch_size=batch_size,
            repeat=1, prefetch=tf.data.AUTOTUNE,
            shuffle=shuffle,
            buffer_size=N
            )
        train_set = train_configurer(X, y)
        
        if validation_data is not None:
            # if validation_data is provided, then configure the validation
            # dataset
            X_val, y_val = validation_data
            
            if validation_batch_size is None:
                validation_batch_size = X_val.shape[0]

            val_configurer = TensorFlowDatasetConfigurer(
                batch_size=validation_batch_size,
                repeat=1, prefetch=tf.data.AUTOTUNE,
                shuffle=shuffle,
                buffer_size=N
                )
            val_set = val_configurer(X_val, y_val)

        num_batches = tf.data.experimental.cardinality(train_set).numpy()
        
        progress_bar = ProgressBar(name=self.model.name,
                                   total_epochs=epochs,
                                   steps_per_epoch=num_batches)
        
        
        # earlystopping requires on_train_begin() initialization.
        # see keras.callbacks documentations
        if early_stopping is not None:
            early_stopping.model = self.model
            early_stopping.on_train_begin()
        
        # reset logger's state before training
        self.logger.reset_state()
        
        # main loop over epochs
        for epoch in range(epochs): 
            train_loss = 0.0 
            # a regular train step
            for train_step, (x_batch_train, y_batch_train) in enumerate(train_set):
                batch_loss = self.training_step(x_batch_train, y_batch_train)
                train_loss+= batch_loss
                train_loss = train_loss / (train_step + 1)
                self.logger["loss"] = train_loss
                
                
                if verbose>0: # print out progress_bar
                    progress_bar(f"[{self._model_trained}/{self.n_trainings}]",
                                 epoch, train_step+1, self.logger.last_log())
                                   
            if validation_data is not None:
                val_loss = 0.0
                # a regular validation step
                for val_step, (x_batch_val, y_batch_val) in enumerate(val_set):
                    batch_vloss = self.validation_step(x_batch_val, y_batch_val)   
                    val_loss+=batch_vloss
                    val_loss = val_loss / (val_step+1)
                self.logger["val_loss"] = val_loss
                
            for metric in self.metrics:
                metric.reset_state()
            
            # watch early stopping status.
            if early_stopping is not None:
                early_stopping.on_epoch_end(epoch, self.logger.last_log())
                if early_stopping.model.stop_training:
                    break # force training stop
                    
        
    
