"""
losses.py is a module designed to store loss functions that inherit
from keras.losses.loss. The classes defined in this module are exclusively
useful within the TensorFlow/Keras ecosystem.

The module contains:

    - The 'GaussianLogLikelihood' class: This class computes the negative of the
      log-likelihood on a provided batch and is used as a keras.losses.loss.
      
"""

# Author: Mikael Frenette (mik.frenette@gmail.com)
# Created: September 2023

import tensorflow as tf
import numpy as np
from typing import Union

class GaussianLogLikelihood(tf.keras.losses.Loss):
    """
    @Author: Mikael Frenette
    
    DESCRIPTION:
    ---------------------------------------------------------------------------
    This function computes the negative log-likelihood loss between the true
    and predicted values using a Gaussian distribution. The loss is calculated
    as the negative sum of the log-probability of the true values given the
    predicted mean and volatility.
    
    Alternatively, you can use the function provided by TensorFlow:
    tf.reduce_sum(tfp.distributions.Normal(loc=μ, scale=σ).log_prob(y))
    
    In this implementation, we have customized the loss function to gain more
    control over its behavior. Specifically, we have added a clip operation
    on the volatility value to avoid numerical instability. 
    This helps with early stopping, as the convergence of the validation loss
    is expected to be more stable.
    ---------------------------------------------------------------------------
    
    PARAMETERS:
    -----------
    volatility_clip : The minimum value for the volatility. Default is 0.05.
        
     See keras.losses.loss documentation for more details. 
    ---------------------------------------------------------------------------
    """
    def __init__(self, volatility_clip:float=0.05, **kwargs):
        
        super().__init__(**kwargs) # inherits from tensorflow.losses.Loss
        
        # the minimum value for the volatility
        self.clip = volatility_clip
    

    @tf.function
    def call(self, y_true: Union[tf.Tensor, np.ndarray],
                    y_pred: Union[tf.Tensor, np.ndarray]):
        """
        PARAMETERS
        ----------
        y_true : The true target values, with a shape of (batch_size, num_outputs)
                 or (batch_size, num_timesteps, num_outputs)
                 in the case of sequence-to-sequence modeling.
                
        y_pred : The predicted target values, consisting of mean and volatility,
                 each with a shape of (batch_size, num_outputs*2) or
                 (batch_size, num_timesteps, num_outputs*2)
                 in the case of sequence-to-sequence modeling.
    
        RETURNS
            loss : The computed negative log-likelihood loss.
        """
        
        mu, sigma = tf.split(y_pred,
                             num_or_size_splits=2,
                             axis=-1)

        mu = tf.cast(tf.squeeze(mu), tf.float32)  # Extract the predicted mean value.
        sigma = tf.cast(tf.squeeze(sigma), tf.float32)  # Extract the predicted volatility value.        
        y = tf.cast(tf.squeeze(y_true), tf.float32)
        # Squeeze the tensors to remove any extra dimensions.
        
        # Clip the volatility to prevent numerical instability.
        sigma = tf.where(sigma > self.clip, sigma, self.clip)  
        
        # Compute the loss using the log-likelihood of the normal distribution.
        loss = tf.reduce_mean(
            tf.math.square(y - mu) / tf.math.square(sigma) + tf.math.log(tf.math.square(sigma))
            )
        
        #loss = -tf.reduce_sum(tfp.distributions.Normal(loc=μ, scale=σ).log_prob(y))
                                                    
        # Return the computed loss as a double-precision float
        return tf.cast(loss, tf.double)  