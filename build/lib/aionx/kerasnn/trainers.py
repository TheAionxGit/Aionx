"""
trainers.py is a module designed to store training algorithms for keras models.
The classes defined in this module are exclusively useful within the
TensorFlow/Keras ecosystem.

The module contains:

    - The 'Trainer' class: A class for standard neural network training.
      
"""

# Author: Mikael Frenette (mik.frenette@gmail.com)
# Created: September 2023

import tensorflow as tf
from aionx.kerasnn import base
from typing import Union, List, Callable

class Trainer(base.NetworkTrainer):
    
    """
    Author: Mikael Frenette
    
    DESCRIPTION
    ---------------------------------------------------------------------------
    This class provides a custom yet standardized way of training a Keras model.
    It inherits from the parent class 'NetworkTrainer,' which offers the 
    standard training loop. Training steps and validation steps are defined
    here, performing the standard procedures. Please note that the
    '@tf.function' decorators are used by default when training the network
    in graph mode (not eagerly). For more details about training in graph mode,
    please refer to TensorFlow's documentation.
    ---------------------------------------------------------------------------
    
    PARAMETERS
    ----------
    optimizer : Optimizer used for training the model.
    
    loss      : Loss function(s) used for training the model.
    
    metrics   : Metric(s) used for evaluation during training. Defaults to None.
                Also, please note that using metrics when training in graph mode
                will raise an error.
    
    **kwargs  : Additional keyword arguments to pass to the parent class
                constructor.
    
    RETURNS
        None
    
    USAGE
    -----
    One can use a Keras models object without compiling it.
    
    model = create_model()  # returns a Keras model
    
    trainer = NetworkTrainer(optimizer=keras.optimizers.Adam(1e-3),
                             loss=keras.losses.MeanSquaredError(),
                             metrics=[keras.metrics.RootMeanSquaredError()])
    
    trainer.train(model, X, y, epochs=20, ...)
    """
    
    def __init___(self,
                 optimizer:Union[tf.keras.optimizers.Optimizer,
                                 List[tf.keras.optimizers.Optimizer]],
                 loss:Union[Callable, tf.keras.losses.Loss,
                            List[tf.keras.losses.Loss]],
                 metrics:Union[tf.keras.metrics.Metric,
                               List[tf.keras.metrics.Metric]]=None,
                 **kwargs)->None:
        
        super().__init__(
                     optimizer=optimizer,
                     loss=loss,
                     metrics=metrics,
                     **kwargs
                     )

    @tf.function
    def training_step(self, x, y):
        with tf.GradientTape() as tape: # auto diff
            pred = self.model(x, training=True)
            loss = self.loss_fn(y, pred)  # compute loss
        grads = tape.gradient(loss, self.model.trainable_weights) # compute gradients
        # back propagate
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))  

        for i, metric in enumerate(self.metrics):
            #update logger with metrics values
            metric.update_state(y, pred)
            self.logger["train_"+metric.name] = metric.result().numpy()
        return loss

    @tf.function
    def validation_step(self, x, y):
        pred = self.model(x, training=False)       
        val_loss = self.loss_fn(y, pred) # compute validation loss
        for i, metric in enumerate(self.metrics):
            metric.update_state(y, pred)
            #update logger with metrics values
            self.logger["val_"+metric.name] = metric.result().numpy()
        return val_loss
    
    def history(self):
        return self.logger.history
        
