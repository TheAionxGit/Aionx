import sys
import tensorflow as tf
import numpy as np

sys.path.append("C:/Users/User/dropbox/Prototyping")
from Prototyping.tensorflow import base
from typing import Union, List, Iterable, Callable


class Trainer(base.NetworkTrainer):
    
    """
    
    usage:
    ------
        One could use a keras.models object without compiling it.
        
        model = create_model() # instiantiate the model
        
        #instantiate the trainer
        trainer = NetworkTrainer(optimizer=keras.optimizers.Adam(1e-3),
                                 loss=keras.losses.MeanSquaredError(),
                                 metrics=[keras.metrics.RootMeanSquaredError()])
        
        # train the network
        trainer.train(model, X, y, epochs=20, ... )
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
            self.logger["train_"+metric.name] = metric.result()
        return loss

    @tf.function
    def validation_step(self, x, y):
        pred = self.model(x, training=False)       
        val_loss = self.loss_fn(y, pred) # compute validation loss
        for i, metric in enumerate(self.metrics):
            metric.update_state(y, pred)
            #update logger with metrics values
            self.logger["val_"+metric.name] = metric.result()   
        return val_loss
    
    def history(self):
        return self.logger.history
        