
import tensorflow as tf
import numpy as np
from typing import Union, List



class GaussianLogLikelihood(tf.keras.losses.Loss):
    """
    @Author : Mikael Frenette
    
    Computes the negative log-likelihood loss between the true and predicted values
    using a Gaussian distribution. The loss is calculated as the negative sum of the 
    log-probability of the true values given the predicted mean and volatility.

    Can also use the function provided by tensorflow:
    -tf.reduce_sum(tfp.distributions.Normal(loc=μ, scale=σ).log_prob(y))

    In this implementation, we have customized the loss function to gain more
    control over its behavior. Specifically, we have added a clip operation
    on the volatility value to avoid numerical instability.
    Additionally, this helps with early stopping, as the convergence of
    the validation loss is expected to be more stable.

    PARAMETERS:
    ----------
    volatility_clip (float): the minimum value for the volatility. Default to 0.05

    """
    def __init__(self, volatility_clip:float=0.05, **kwargs):
        
        super().__init__(**kwargs) # inherits from tensorflow.losses.Loss
        
        # the minimum value for the volatility
        self.clip = volatility_clip
    
    @tf.function
    def call(self, y_true:Union[tf.Tensor, np.ndarray],
                   y_pred:Union[tf.Tensor, np.ndarray]):
        """
        PARAMETERS:
        -----------
        y_true : tensor
            The true target values, of shape
            (batch_size, num_outputs) or (batch_size, num_timesteps, num_outputs)
            in the case of sequence-to-sequence modeling.
            
        y_pred : list of tensors. Must contain two values. μ and σ respectively
            The predicted target values, consisting of mean and volatility, each of shape
            (batch_size, num_outputs) or (batch_size, num_timesteps, num_outputs)
            in the case of sequence-to-sequence modeling.

        RETURNS:
        --------
        loss : float
            The computed negative log-likelihood loss.
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