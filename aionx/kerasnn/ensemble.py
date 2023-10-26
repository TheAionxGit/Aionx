"""
ensemble.py is a module designed for storing ensemble-based models that are
useful within the TensorFlow/Keras ecosystem.

The module contains:
    
    - The 'ModelGenerator' class: Instantiate a generator object that efficiently
      stores Keras models before training.
    
    - The 'DeepEnsemble' class: Instantiate a deep ensemble object with a
     '.fit()' method for efficient and sequential training of Keras models.
     The class also provides a '.predict()' method that separates the model
     outputs into lists.
    
    - The 'OutOfBagPredictor' class: Instantiate an ensemble predictor used
      exclusively for bagging models.
     
"""

# Author: Mikael Frenette (mik.frenette@gmail.com)
# Created: September 2023

import tensorflow as tf
from tensorflow import keras
from IPython.display import clear_output
import types
import numpy as np
import pandas as pd
from typing import Union, List

from aionx.kerasnn import base as knnbase
from aionx import base
from aionx.wrappers import timeit, downstream
from aionx.bootstrap import TimeSeriesBlockBootstrap

class ModelGenerator:
    """
    Author: Mikael Frenette
    
    DESCRIPTION
    ---------------------------------------------------------------------
    A generator class for efficiently storing Keras models. If the provided
    model to the constructor is a direct instance of a Keras model, then 
    the generator will simply generate the same instance with shuffled weights.
    ---------------------------------------------------------------------
    
    PARAMETERS
    ----------
    model        : The Keras model or a callable function that generates the
                   model.
    
    n_estimators : The number of models to generate.
    
    USAGE
    -----
    generator = ModelGenerator(model, n_estimators=100)
    for i, generated_model in enumerate(generator):
        # ... your code here
    
    ---------------------------------------------------------------------
    """

    def __init__(self, model: keras.models.Model, n_estimators: int = 100):

        self.model = model
        self.n_estimators = n_estimators
        self.current = 0

        # Check if the provided model is an instance of keras.models.Model
        self.iskerasmodel = isinstance(self.model, keras.models.Model)
        if self.iskerasmodel:
            self.original_weights = self.model.get_weights()

    def __iter__(self):
        return self

    def __next__(self):
        if self.current < self.n_estimators:
            if self.iskerasmodel:
                # Modify weights of the same model
                weights = [
                    np.random.permutation(w.flat).reshape(w.shape) for w in self.original_weights
                ]
                self.current += 1
                self.model.set_weights(weights)
                return self.model
                #return weights
            else:
                # Generate a new model using the provided callable function
                model = self.model()
                self.current += 1
                return model
        else:
            raise StopIteration
            
class DeepEnsemble:

    """
    Author: Mikael Frenette
    
    DESCRIPTION
    ---------------------------------------------------------------------------
    This class is used for creating and training an ensemble of neural networks
    within a bagging framework. It is operational only within a TensorFlow/Keras
    ecosystem.
    ---------------------------------------------------------------------------
    
    PARAMETERS
    ----------
    n_estimators  : The number of estimators (neural networks) in the ensemble.
    
    network       : The base neural network(s) to use in the ensemble.
    
    trainer       : An optional network trainer for training the ensemble.
                    Default is None.
              
    sampler       : An optional bootstrapping sampler for generating training
                    samples. Default is None, which uses
                    TimeSeriesBlockBootstrap.
        
    block_size    : The block size for bootstrapping. Default is 8.
    
    sampling_rate : The sampling rate for bootstrapping. Default is 0.8.
    
    replace       : Whether to sample with replacement during bootstrapping.
                    Default is True.
    
    USAGE
    -----
    model = yourmodel(...)
    ensemble = BaggingNetwork(n_estimators=100, network=model)
    ensemble.fit(*args, **tfkwargs)
    ensemble.predict(*args, **tfkwargs)
    
    IMPORTANT NOTES
    ---------------
    - If a function is provided, no trainers should be used as it may raise
    an error by TensorFlow in later versions. The reason is that the trainer
    is using only one optimizer, so one could end up training multiple models
    with only one optimizer. This is a problem for those with internal states,
    such as keras.optimizers.Adam.
    
    - In the case where only a model is provided and, since most Keras models
    cannot be cloned (subclassing API), we need to copy the weights and simply
    shuffle them. During training, only one model will be used but with
    different sets of shuffled weight initialization.
    ---------------------------------------------------------------------------
    """

    def __init__(self,
                 n_estimators: int,
                 network: Union[tf.keras.models.Model,
                                List[tf.keras.models.Model]],
                 trainer: knnbase.NetworkTrainer = None,
                 sampler: base.Bootstrapper = None,
                 block_size: int = 8,
                 sampling_rate: float = 0.8,
                 replace: bool = True) -> None:

        self._n_estimators = n_estimators
        self._trainer = trainer
        self.network = network
        self.block_size = block_size
        self.sampling_rate = sampling_rate
        self.replace = replace
        
        self.model_generator = ModelGenerator(
            model = self.network,
            n_estimators=self._n_estimators
            )
        
        self._sampler = sampler
        
        # out of bag indices will be stored here
        self._oob_idx = {}
        
        # models will be stored here
        self._estimators = []

    @classmethod
    def from_function(cls,
                      n_estimators: int,
                      func: types.FunctionType,
                      sampler: base.Bootstrapper = None,
                      block_size: int = 8,
                      sampling_rate: float = 0.8,
                      replace: bool = True):
        """
        PARAMETERS
        ----------
        n_estimators  : The number of estimators (neural networks) in the
                        ensemble.
        
        func          : A Python function that returns a Keras model.
        
        sampler       : An optional bootstrapping sampler for generating
                        training samples.
                        Default is None, which uses TimeSeriesBlockBootstrap.
                   
        block_size    : The block size for bootstrapping. Default is 8.
        
        sampling_rate : sampling rate for bootstrapping. Default is 0.8.
        
        replace       : Whether to sample with replacement during bootstrapping.
                        Default is True.
            
        RETURNS:
            None
        """

        return cls(n_estimators, func, None, sampler, block_size, sampling_rate)
    
    
    @timeit
    def fit(self,
            X: Union[tuple[Union[np.ndarray, tf.Tensor], Union[np.ndarray, tf.Tensor]],
                     np.ndarray, tf.Tensor],
            y: Union[tuple[Union[np.ndarray, tf.Tensor], Union[np.ndarray, tf.Tensor]],
                     np.ndarray, tf.Tensor],
            epochs: int,
            batch_size: int = 32,
            validation_batch_size: int = 32,
            verbose: int = 1,
            **tfkwargs
            ) -> None:
        """
        PARAMETERS:
        -----------
        X      : The input data. It can be a single array/tensor or a tuple of
                 arrays/tensors.
            
        y      : The target data. It can be a single array/tensor or a tuple of
                 arrays/tensors.
            
        epochs : The number of training epochs.
        
        batch_size            : The batch size for training. Default is 32.
        
        validation_batch_size : The batch size for validation. Default is 32.
        
        verbose               : The verbosity mode (0, 1, or 2). Default is 1.
        
        **tfkwargs            : Additional Keras keyword arguments to pass to
                                the model.fit() call.
    
        RETURNS:
            None
        """
        
        if self._sampler is None:
            self._sampler = TimeSeriesBlockBootstrap(
                X, y, block_size=self.block_size,
                sampling_rate=self.sampling_rate, replace=True
            )
        
        for e, model in enumerate(self.model_generator):
            keras.backend.clear_session()
            X_train, y_train, X_val, y_val = next(self._sampler)

            if self._trainer is not None:
                self._trainer.train(
                    model,
                    X_train,
                    y_train,
                    epochs=epochs,
                    validation_data=(X_val, y_val),
                    batch_size=batch_size,
                    validation_batch_size=validation_batch_size,
                    **tfkwargs
                    )
                if verbose>1:
                    print(" ")
            else:
                model.fit(X_train, y_train,
                                      validation_data=(X_val, y_val),
                                      epochs=epochs,
                                      batch_size=batch_size,
                                      validation_batch_size=validation_batch_size,
                                      verbose=verbose,
                                      **tfkwargs)
                clear_output(wait=True)
                
            self._estimators.append(model.get_weights() if self.model_generator.iskerasmodel else model)
            self._oob_idx[f"Estimator_{e}"] = self._sampler.oob_indices[-1]
            
            
    def predict(self, X: Union[tuple[Union[np.ndarray, tf.Tensor],
                                     Union[np.ndarray, tf.Tensor]],
                              np.ndarray, tf.Tensor], **tfkwargs) -> list:
        """
        PARAMETERS
        ----------
        X        : Input data for predictions.
        
        tfkwargs : Additional keyword arguments for TensorFlow prediction.
        
        RETURNS
            List of predictions from individual estimators.
        """
        
        for e, model in enumerate(self._estimators):
            
            # make raw predictions
            if self.model_generator.iskerasmodel:
                self.network.set_weights(model)       
                output = self.network.predict(X, **tfkwargs)
            else:
                output = model.predict(X, **tfkwargs)
            
            if isinstance(output, list):
                output = tf.concat(output, axis=-1)
                n_outputs = output.shape[-1]
                
            if tf.rank(output) < 2:
                output = tf.expand_dims(output, axis=-1)
            n_outputs = output.shape[-1]
            
            if e == 0:
                predictions = [[] for _ in range(n_outputs)]
            
            if output.shape[-1] > 1:
                output = tf.split(output, num_or_size_splits=n_outputs, axis=-1)
                for k, out in enumerate(output):
                    predictions[k].append(out)         
            else:
                predictions[0].append(output)
                
        predictions = [
            tf.squeeze(tf.concat(pred, axis=-1)).numpy() for pred in predictions
            ]
                
        return predictions[0] if len(predictions)==1 else predictions
    
    @property
    def oob_indices(self):
        return self._oob_idx
    
    @property
    def estimators(self):
        return self._estimators



class OutOfBagPredictor(object):
    """
    Author : Mikael Frenette
    
    DESCRIPTION
    ---------------------------------------------------------------------------
    This class creates a blueprint of out-of-bag (oob) indices for each estimator
    and uses the call method to map each set of oob indices to its corresponding
    estimator."
    
    The function computes:
        oob_forecast = (1 / ((1 - sampling_rate) * n_bootstraps)) * sum(bootstraps)
        
    rows with all NaN values will produce a NaN in the oob_forecast defined
    above.
    ---------------------------------------------------------------------------  
        
    PARAMETERS
    ----------
    oob_idx : A dictionnary mapping Each estimator to its out-of-bag indices.
              each element in the dictionnary must contain an array-like value
              with out-of-bag indices.  
    ---------------------------------------------------------------------------
    """
    
    def __init__(self, oob_indices:dict)->None:
        self.oob_indices = oob_indices
        
    @downstream(
        np.ndarray, dtype="float32"
    )
    def __call__(self,
                 forecasts:np.ndarray,
                 sampling_rate:float=0.8
                 )->pd.DataFrame():
        
        """
        PARAMETERS
        ----------
        forecasts : the prediction from the models. the dataframe should have a
                    shape of (T, B) where T is the number of periods
                    and B the number of Bootstraps (i.e number of estimators).
                    The first column should be the first bootstrap while the last
                    should be the last bootstrap forecast.
            
        sampling_rate : The sampling rate parameter used by the sampler for
                        bootstrapping
                
        RETURNS
            full_forecasts : The out-of-bag as well as out-of-sample predictions
                             if the provided forecasts go beyond the training
                             set.
        """
        for i, (estimator, oob_idx) in enumerate(self.oob_indices.items()):
            rows_to_nan = np.where(~np.isin(np.arange(forecasts.shape[0]), oob_idx))[0]
            forecasts[rows_to_nan, i] = np.nan
        row_sums = np.nansum(forecasts, axis=1)
        fcst = (1 / ((1 - sampling_rate) * forecasts.shape[1])) * row_sums
        fcst[fcst==0] = np.nan
        return fcst
        
        
        
