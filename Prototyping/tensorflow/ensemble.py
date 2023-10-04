import tensorflow as tf
from tensorflow import keras
from IPython.display import clear_output
import types
import sys
import numpy as np
import pandas as pd
from typing import Union, List, Iterable

sys.path.append("C:/Users/User/dropbox/Prototyping")
import Prototyping.tensorflow as tfproto
from Prototyping import base
from Prototyping.wrappers import timeit
from Prototyping.bootstrap import TimeSeriesBlockBootstrap



class DeepEnsemble:

    """
    A class for an ensemble of neural networks trained using bagging.
    
    This class is used to create an ensemble of neural networks for bagging purposes.
    
    PARAMETERS:
    -----------
        n_estimators (int):
            The number of estimators (neural networks) in the ensemble.
        network (Union[tf.keras.models.Model, List[tf.keras.models.Model]]):
            The base neural network(s) to use in the ensemble.
        trainer (tfproto.base.NetworkTrainer, optional):
            An optional network trainer for training the ensemble. Default is None.
        sampler (base.Bootstrapper, optional):
            An optional bootstrapping sampler
            for generating training samples. Default is None, which uses
            TimeSeriesBlockBootstrap.
        block_size (int, optional):
            The block size for bootstrapping. Default is 8.
        sampling_rate (float, optional):
            The sampling rate for bootstrapping. Default is 0.8.
        replace (bool, optional):
            Whether to sample with replacement during bootstrapping.
            Default is True.
        
    USAGE:
    ------
        model = yourmodel(...)
        
        ensemble = BaggingNetwork(n_estimators=100, network=model)
        ensemble.fit(**tfkwargs)
        ensemble.predict(**tfkwargs)
        
    Important notes:
    ----------------
        - If a function is provided, no trainers should be used as it may
        raise an error by tensorflow in later versions. The reason is that
        the trainer is using only one optimizer so one could end up training
        multiple models with only one optimizer. This is a problem for those
        with internal states such as keras.optimizers.Adam.
        
        - In the case where only a model is provided and since most keras models
        cannot be cloned (subclassing api), we need to copies the weights 
        and simply shuffle them. During training, only one model will be used
        but with different weights initialization.
        
        
    """

    def __init__(self,
                 n_estimators: int,
                 network: Union[tf.keras.models.Model,
                                List[tf.keras.models.Model]],
                 trainer: tfproto.base.NetworkTrainer = None,
                 sampler: base.Bootstrapper = None,
                 block_size: int = 8,
                 sampling_rate: float = 0.8,
                 replace: bool = True) -> None:

        # attributes
        self._n_estimators = n_estimators
        self._trainer = trainer
        
        # check inputs and set the private attribute "__has_to_set_weight"
        # to true if a model is provided.
        if isinstance(network, tf.keras.models.Model):
            self.network = network
            self.estimators = [
                DeepEnsemble.shuffle_weights(
                    self.network, weights=self.network.get_weights()
                ).get_weights() for _ in range(self._n_estimators)
            ]
            self.__has_to_set_weights = True
            
        elif isinstance(network, list):
            self.network = network[0]
            self.estimators = network
            self.__has_to_set_weights = False
            
        if sampler is None:
            self._sampler = TimeSeriesBlockBootstrap(
                block_size=block_size,
                sampling_rate=sampling_rate,
                replace=replace
            )
        else:
            self._sampler = sampler
        
        # out of bag indices will be stored here
        self._oob_idx = {}

    @classmethod
    def from_function(cls,
                      n_estimators: int,
                      networks: types.FunctionType = None,
                      trainer: tfproto.base.NetworkTrainer = None,
                      sampler: base.Bootstrapper = None,
                      block_size: int = 8,
                      sampling_rate: float = 0.8,
                      replace: bool = True):
        """
        PARAMETERS:
        -----------
            n_estimators (int):
                The number of estimators (neural networks) in the ensemble.
            network (function):
                A python function that builds a keras.models.Model.
            trainer (tfproto.base.NetworkTrainer, optional):
                An optional network trainer for training the ensemble. Default is None.
            sampler (base.Bootstrapper, optional):
                An optional bootstrapping sampler
                for generating training samples. Default is None, which uses
                TimeSeriesBlockBootstrap.
            block_size (int, optional):
                The block size for bootstrapping. Default is 8.
            sampling_rate (float, optional):
                The sampling rate for bootstrapping. Default is 0.8.
            replace (bool, optional):
                Whether to sample with replacement during bootstrapping.
                Default is True.
                
        RETURNS:
            class constructor using PARAMETERS
        """
        estimators = [
            DeepEnsemble.model_builder(networks)(
                name=f"Estimator_{e}") for e in range(n_estimators)
        ]
        return cls(n_estimators, estimators, trainer, sampler)
    
    
    @timeit
    def fit(self,
            X:Union[tuple[Union[np.ndarray, tf.Tensor],
                          Union[np.ndarray, tf.Tensor]],
                            np.ndarray, tf.Tensor],
            y:Union[tuple[Union[np.ndarray, tf.Tensor],
                          Union[np.ndarray, tf.Tensor]],
                            np.ndarray, tf.Tensor],
            epochs:int,
            batch_size:int=32,
            validation_batch_size:int=32,
            verbose:int=1,
            **tfkwargs
            )->None:
        
        """
        Fit the ensemble of estimators to the provided training data.

        PARAMETERS:
        -----------
            X (Union[tuple[Union[np.ndarray, tf.Tensor], Union[np.ndarray, tf.Tensor]],
                  np.ndarray, tf.Tensor]): The input data. It can be a single array/tensor
                  or a tuple of arrays/tensors.
            y (Union[tuple[Union[np.ndarray, tf.Tensor], Union[np.ndarray, tf.Tensor]],
                  np.ndarray, tf.Tensor]): The target data. It can be a single array/tensor
                  or a tuple of arrays/tensors.
            epochs (int):
                The number of training epochs.
            batch_size (int, optional):
                The batch size for training. Default is 32.
            validation_batch_size (int, optional):
                The batch size for validation. Default is 32.
            verbose (int, optional):
                The verbosity mode (0, 1, or 2). Default is 1.
            **tfkwargs:
                Additional keras keyword arguments to pass to the 
                         fit() call.
                        
        Returns:
            None
        """
        
        for e, model in enumerate(self.estimators):
            keras.backend.clear_session()
            bootstrap_idx, oob_idx = self._sampler(X, return_indices=True)
            X_train, y_train = X[bootstrap_idx], y[bootstrap_idx]
            X_val, y_val = X[oob_idx], y[oob_idx]
            
            if self.__has_to_set_weights:
                self.network.set_weights(model)
                if self._trainer is not None:
                    self._trainer.train(
                        self.network,
                        X_train,
                        y_train,
                        validation_data=(X_val, y_val),
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_batch_size=validation_batch_size,
                        **tfkwargs
                        )
                    if verbose>1:
                        print(" ")
                else:
                    self.network.fit(X_train, y_train,
                                          validation_data=(X_val, y_val),
                                          verbose=verbose, **tfkwargs)

                    clear_output(wait=True)
                self.estimators[e] = self.network.get_weights()
            else:
                model.fit(X_train, y_train, validation_data=(X_val, y_val),
                                                 verbose=verbose,
                                                 **tfkwargs)
                clear_output(wait=True)
                
            self._oob_idx[f"Estimator_{e}"] = oob_idx
            
    def predict(self, X:Union[tuple[Union[np.ndarray, tf.Tensor],
                              Union[np.ndarray, tf.Tensor]],
                                np.ndarray, tf.Tensor], **tfkwargs)->list:
        """
        Make predictions using the ensemble of estimators.

        PARAMETERS:
        -----------
        - X (Union[Tuple[Union[np.ndarray, tf.Tensor],
                         Union[np.ndarray, tf.Tensor]],
                   np.ndarray, tf.Tensor]):
            Input data for predictions.
        - tfkwargs (dict): Additional keyword arguments for TensorFlow prediction.

        Returns:
        --------
        - List[np.ndarray]: List of predictions from individual estimators.
        """
        
        predictions = []
        for e, model in enumerate(self.estimators):
            if self.__has_to_set_weights:
                self.network.set_weights(model)       
                output = self.network.predict(X, **tfkwargs)
                if isinstance(output, list):
                    output = tf.concat(output, axis=-1)
                output = tf.squeeze(output)
            else:
                output = model.predict(X, **tfkwargs)
                if isinstance(output, list):
                    output = tf.concat(output, axis=-1)
                output = tf.squeeze(output)
            if tf.rank(output) < 2:
                output = tf.expand_dims(output, axis=-1)                
            predictions.append(output)      
        predictions = tf.squeeze(predictions)
        predictions = tf.split(predictions,
                               num_or_size_splits=predictions.shape[-1],
                               axis=-1)
        predictions = [
            tf.transpose(tf.squeeze(pred)).numpy() for pred in predictions
            ]
        return predictions            
        
        
            
    @staticmethod
    def shuffle_weights(model:keras.models.Model,
                        weights:List[Union[tf.Tensor, np.ndarray]]=None):
        """
        Shuffle the weights of a Keras model.

        PARAMETERS:
        -----------
        - model (keras.models.Model):
            Keras model to shuffle the weights of.
        - weights (List[Union[tf.Tensor, np.ndarray]], optional):
            List of weights to shuffle. 
          If None, the model's current weights will be used.

        Returns:
        --------
        - keras.models.Model: The model with shuffled weights.
        """

        if weights is None:
            weights = model.get_weights()
        weights = [np.random.permutation(w.flat).reshape(w.shape) for w in weights]
        model.set_weights(weights)
        return model
        
    @staticmethod
    def model_builder(build_function):
        """
        Create a model builder function.

        PARAMETERS:
        -----------
        - build_function (callable): A function that builds a Keras model.

        Returns:
        --------
        - callable: A function that builds a Keras model and sets its name.
        """
        def build(name=None):
            model = build_function()
            model.Name=name
            return model
        return build
    
    @property
    def oob_indices(self):
        return self._oob_idx


class OutOfBagPredictor(object):
    """
    This class starts by creating a blueprint of out-of-bag (oob) indices for each
    estimator. Then, we can use the call method which will map each set of oob
    indices to it's estimator.
    
    The function computes:
        oob_forecast = (1 / ((1 - sampling_rate) * n_bootstraps)) * sum(bootstraps)
        
    rows with all NaN values will produce a NaN in the oob_forecast defined
    above.
    
    PARAMETERS:
    -----------
    
    oob_idx (dict):
        A dictionnary mapping Each estimator to its out-of-bag indices.
        each element in the dictionnary must contain an array-like value with out-of-bag
        indices.  
        
        
    """
    
    def __init__(self, oob_indices:dict)->None:
        self.oob_indices = oob_indices
        
    def __call__(self,
                 forecasts:Union[pd.DataFrame, np.ndarray],
                 sampling_rate:float=0.8
                 )->pd.DataFrame():
        
        """
        PARAMETERS:
        -----------
            forecasts (pd.DataFrame):
                the prediction from the models. the dataframe should have a
                shape of (T, B) where T is the number of periods
                and B the number of Bootstraps (i.e number of estimators)
                
            sampling_rate (float):
                The sampling rate parameter used by the sampler for bootstrapping
                
        RETURNS
        ------
            full_forecasts (pd.DataFrame):
                The out-of-bag as well as out-of-sample predictions
                if the provided forecasts go beyond the training set.
        """
        
        if isinstance(forecasts, np.ndarray): 
            forecasts = pd.DataFrame(
                forecasts,
                columns=[f"Estimator_{e}" for e in range(forecasts.shape[1])]
                )
        else:
            forecasts.columns = [
                f"Estimator_{e}" for e in range(forecasts.shape[1])
                ]
            
        df = forecasts.copy()

        for estimator, oob_idx in self.oob_indices.items():
            df.loc[~df.reset_index().index.isin(oob_idx), estimator] = np.nan

        row_sums = df.sum(axis=1)
        fcst = (1 / ((1 - sampling_rate) * df.shape[1])) * row_sums
        fcst[df.isnull().all(axis=1)] = np.nan  # Set predictions as NaN for rows with all NaN values
        oob_forecast =  pd.DataFrame(fcst, columns=["forecast"])
        
        return oob_forecast
        
        