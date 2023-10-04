import tensorflow as tf
import numpy as np 
from typing import Union, List, Iterable, Callable
from Prototyping.utils import LogsTracker, ProgressBar
from Prototyping.tensorflow.pipelines import TensorFlowDatasetConfigurer


class NetworkTrainer(object):
    """
    --------------------------------------------------------------------------
    Base class used to train a keras.models.Model. Its purpose is to mimic 
    tensorflow's model.compile() and model.fit() as much as possible.
    
    The class is instantiated by specifying a keras.optimizers, a callable
    loss function with the signature similar to (y_true, y_train).
    
    parameters:
    -----------
        optimizer (keras.optimizers):
            the network optimizer.
        loss (callable, keras.losses):
            the loss function used for training the network.
        metrics (list of keras.metrics):
            The list of metrics which will be printed during training.

    methods:
    --------
        train:
            first, The train method will start by transforming data input into
            a tf.data.Dataset. second, it will loop over the dataset for a number
            of specified epoch. 
            
        child classes will have to specify a self.training_step() and
        self.validation_step()

            
    properties:
    -----------
        One can use .hisotry on the trainer instance to have access to 
        LogsTracker logs history.

    --------------------------------------------------------------------------
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
        
        # create attributes
        self.optimizer = optimizer
        self.loss_fn = loss
        self.n_trainings = n_trainings
        
        if isinstance(metrics, list):
            self.metrics = metrics
        elif metrics is None:
            self.metrics = {}
        else:
            self.metrics = [metrics]
            
        self.logger = LogsTracker() # instantiate LogsTracker
        
        self._model_trained = 0

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
        ---------------------------------------------------------------------
        The train method keeps the same intuitive structure used by tensorflow.
        
        parameters:
        -----------
            model (keras.models):
                the model that will be trained. doesn't need to be compiled.
            X (np.ndarray, tf.Tensor):
                The inputs to the network.
            y (np.ndarray, tf.Tensor):
                The target data.
            epochs (int):
                The number of training epochs to train the model.
            shuffle (bool, default:False):
                if True, only the training data will be shuffled.
            validation_data (tuple, default:None):
                The validation data to monitor during training.
            batch_size (int, default:32):
                the sizes of batches a gradient descent step will be applied 
                on.
            verbose (int, default:1):
                The level of verborsity. the higher the more logs will be 
                printed out during training. if 0, progress bar will be
                suppressed.
            early_stopping (keras.callbacks.EarlyStopping):
                will perform early stopping based on the keras EarlyStopping's
                callback parameters.
            validation_batch_size (int, default:None):
                the sizes of batches on which a validation step will be 
                performed on.
                
        other considerations:
        ---------------------
            -The dataset will be looped over once every epoch.
            ...
            
        ---------------------------------------------------------------------
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
                # after each epoch, the metric's state must be reset.
                metric.reset_state()
            
            # watch early stopping status.
            if early_stopping is not None:
                early_stopping.on_epoch_end(epoch, self.logger.last_log())
                if early_stopping.model.stop_training:
                    break # force training stop
                    
        
    