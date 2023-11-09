"""
aionx models.py is a file for storing model's classes.

The module contains:
    
    - The 'DensityHNN' class:
        A class implementation of the model proposed in
        'From Reactive to Proactive Volatility Modeling with
        Hemisphere Neural Networks' by Goulet-Coulombe (2023).
"""

# Author: Mikael Frenette (mik.frenette@gmail.com)
# Created: october 2023

import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential, Model
from keras.initializers import RandomNormal
from keras.layers import Flatten, Dense, Input, Dropout
import numpy as np
from datetime import datetime
from typing import Union
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

from aionx.scalers import StandardScaler
from aionx.pipelines import WindowDataset
from aionx.kerasnn.trainers import Trainer
from aionx.utils import (add_trends, WindowEstimationHandler)
from aionx.bootstrap import TimeSeriesBlockBootstrap
from aionx.kerasnn.ensemble import DeepEnsemble, OutOfBagPredictor
from aionx.kerasnn.losses import GaussianLogLikelihood

import warnings

class DensityHNN(object):
    """"
    Author : Mikael Frenette (mik.frenette@gmail.com)
    
    DESCRIPTION
    ---------------------------------------------------------------------------
    Initialize a Density Hemisphere Neural Network (DensityHNN) model.
    DensityHNN is an ensemble of neural network trained by minimizing the
    negative log likelihood in order to produce forecasts along with associated
    uncertainty estimates. 
    
                                  IMPORTANT
    ***************************************************************************

    THE 'DensityHNN' CLASS IS DESIGNED FOR SMALL DATASETS SUCH AS FRED-MD OR
    FRED-QD. DO NOT USE THIS CLASS WITH DATA SUBJECTED TO IREGULAR TIMESTAMPS
    SUCH AS STOCK MARKET DATA (WHICH HAVE HOLIDAYS), BECAUSE PANDAS CANNOT
    INFER FROM THEIR TIMESTAMP EASILY. ADDITIONALY, I WOULD RECOMMEND NOT USING
    THE CLASS ON LARGE DATASET SINCE THIS IMPLEMENTATION DOES NOT SUPPORT A
    MINIBATCH LEARNING AND LARGE BATCHES MAY QUICKY LEAD TO MEMORY ERROR.
    
    ***************************************************************************
    
    ---------------------------------------------------------------------------
                   
    PARAMETERS
    ----------
    target           : The target variable name.
    
    lags             : Number of lags used for each regressor provided.
    
    horizon          : The number of steps ahead to forecast. For example,
                       if horizon=1, the network will be trained to predict
                       y_{t+1} using y_{t-lags:t}.
    
    bootstraps       : Number of weak learners (estimators) to train. Their
                       forecasts will be averaged out. The default is 200.
    
    prior_bootstraps : Number of weak learners to use for the prior DNN. This
                       is used to compute the volatility emphasis parameter
                       (estimating a prior for the importance of volatility).
                       The default is 100.
    
    trends           : Number of trend regressors to include in the model.
                       The default is 100.
    
    sampling_rate    : The fraction of data to use for training each epoch.
                       The default is 0.8.
    
    block_size       : Size of blocks used for bootstrapping. The default is 8.
    
    epochs           : Number of training epochs. The default is 100.
    
    learning_rate    : Learning rate for training. The default is 1e-3.
    
    patience         : Number of epochs with no improvement after which
                       training will stop early. The default is 15.
    
    shuffle          : Whether to shuffle the training data before each epoch.
                       The default is False.
    
    verbose          : Verbosity mode (0, 1, or 2). The default is 1.
    
    STATIC METHODS
    --------------
    base_architecture : The base architecture method returns the architecture
                        used in the paper. One must force a volatility emphasis
                        parameter to use the model instead of estimating it
                        with a prior neural net.
    
    prior_dnn         : The prior_dnn method returns a Keras model
                        architecture. The model is used in an ensemble of
                        networks with identical architecture, and its 
                        out-of-bag error is used as volatility emphasis
                        for the DensityHNN.
    
    volatility_rescaling_algorithm : The rescaling function. It uses blocked
                                     out-of-bag residuals to recalibrate the
                                     ensemble's volatility estimation.
    
    METHODS
    -------
    run : The run method is used to instantiate, fit, and predict the data. One
          can also use expanding windows fitting, though this takes some time.
    
    USAGE
    -----
    HNN = DensityHNN(
        target="your_target_name",
        lags=6,
        horizon=1,
        bootstraps=100,
        prior_bootstraps=50,
    )
    
    results = HNN.run(your_data,
                      train_start="1961-01-01",
                      train_end="2006-12-01",
                      expanding=False)
    
    dnn_pred = results["dnn_forecast"]
    HNN_preds = results["hnn_forecast"]
    vol_emphasis = results["volatility_emphasis"]
    """

    def __init__(self,
                 target:str,
                 lags:int,
                 horizon:int,
                 bootstraps:int=200,
                 prior_bootstraps:int=100,
                 trends:int=100,
                 sampling_rate:float=0.8,
                 block_size:int=8,
                 epochs:int=100,
                 learning_rate:float=1e-3,
                 patience:int=15,
                 verbose:int=1,
                 )->None:
        
        # accessible attributes
        self.target = target
        self.lags = lags
        self.horizon = horizon
        self.bootstraps = bootstraps
        self.prior_bootstraps = prior_bootstraps
        self.trends = trends
        self.epochs=epochs
        self.sampling_rate = sampling_rate
        self.block_size = block_size
        self.learning_rate = learning_rate
        self.patience = patience
        self.verbose = verbose

        self.__pipeline = WindowDataset(
            in_steps = self.lags, out_steps = 1, horizon = self.horizon,
            targets = [self.target], sampler = None
            )
        
        self.__pred_pipeline = WindowDataset(
            in_steps = self.lags, out_steps = 0, horizon = 0,
            targets = [self.target], sampler = None
            )
        
    @staticmethod
    def base_architecture(
            input_shape:tuple,
            hemisphere_outputs:int=1,
            vol_emphasis:float=1.0
            )->Model:
        
        """
        PARAMETERS
        ----------
        input_shape        : format expected by keras models input.
        
        hemisphere_outputs : the number of output units for each hemisphere.
                             Default to 1.
                             
        vol_emphasis       : the vol emphasis parameter. must be between
                             0.01 and 1.0. Default to 1.0. If above 1.0, 
                             value will automatically be set to 0.99.
        
        RETURNS
            keras.models.Model
            
        """
        
        #The architecture is as following:
            
        #    - The network has 2 seperate hemispheres:
        #        One that models the conditional mean (hm).
        #        One that models the conditional std (hv).
        
        #    - Both hemispheres share a common block of layers at the entrance
        #      of the network (hz).
            
        #               -- hm -> mu
        #    X -> -- hz 
        #               -- hv -> sigma
                   
        if (hemisphere_outputs <=0):
            raise ValueError(
                "Each hemisphere must have atleast one output unit."+
                f"Number received  {hemisphere_outputs}."
                )
        
        if not (0.01 <= vol_emphasis <= 1.0):
            warnings.warn(
                "vol_emphasis input must be between 0.01 and 1.0 "+
                f"input received {vol_emphasis:.4f}. vol_emphasis will be set"+
                " to 0.99."
                )
            vol_emphasis = 0.99
            
        inputs = Input(shape=input_shape)
    
        shared = Sequential([
            Flatten(),
            Dense(units=400,
                activation="relu",
                kernel_initializer=RandomNormal(mean=0, stddev=0.03),
                ),
            Dropout(0.2),
            Dense(units=400,
                activation="relu",
                kernel_initializer=RandomNormal(mean=0, stddev=0.03),
                ),
            keras.layers.Dropout(0.2),
        ], name="shared_hemisphere")
        
        mean = Sequential([
            Dense(units=400,
                activation="relu",
                kernel_initializer=RandomNormal(mean=0, stddev=0.03),
                ),
            Dropout(0.2),
            Dense(units=400,
                activation="relu",
                kernel_initializer=RandomNormal(mean=0, stddev=0.03),
                ),
            Dropout(0.2),
            Dense(hemisphere_outputs, activation='linear'),
        ], name="conditional_mean_hemisphere")
        
        vol = Sequential([
            Dense(units=400,
                activation="relu",
                kernel_initializer=RandomNormal(mean=0, stddev=0.03),
                ),
            Dropout(0.2),
            Dense(units=400,
                activation="relu",
                kernel_initializer=RandomNormal(mean=0, stddev=0.03),
                ),
            Dropout(0.2),
            Dense(hemisphere_outputs, activation='softplus'),
        ], name="conditional_vol_hemisphere")
    
        z = shared(inputs)
        mean_output = mean(z)
        vol_output = vol(z)
        
        vol_output = tf.math.scalar_mul(
            vol_emphasis, tf.math.divide(vol_output,
                                         tf.reduce_mean(vol_output))
            )
    
        output = tf.concat([mean_output, vol_output], axis=-1)
    
        HNN = Model(inputs=[inputs], outputs=[output], name="density_hnn")
        
        HNN.compile(loss=GaussianLogLikelihood(),
                    optimizer=keras.optimizers.Adam(1e-3))
        
        return HNN
    
    @staticmethod
    def prior_dnn_architecture(input_shape:tuple)->Model:
        
        """
        PARAMETERS
        ----------
        input_shape : format expected by keras models input.
        
        RETURNS
            keras.models.Model.
        """
        
        inputs = Input(shape=input_shape)
        DNN = Sequential([
            Flatten(),
            Dense(units=400,
                  activation="tanh",
                  kernel_initializer=RandomNormal(mean=0, stddev=0.03)),
            Dropout(0.2),
            Dense(units=400,
                  activation="tanh",
                  kernel_initializer=RandomNormal(mean=0, stddev=0.03)),
            Dropout(0.2),
            Dense(units=400,
                  activation="tanh",
                  kernel_initializer=RandomNormal(mean=0, stddev=0.03)),
            Dropout(0.2),
            Dense(units=400,
                  activation="tanh",
                  kernel_initializer=RandomNormal(mean=0, stddev=0.03)),
            Dropout(0.2),
            Dense(1, activation="linear")
        ], name="dnn_block")
        
        output = DNN(inputs)
        
        DNN = Model(inputs=[inputs], outputs=[output], name="prior_dnn")
        
        DNN.compile(loss="MSE", optimizer=keras.optimizers.Adam(1e-3)) 
        
        return DNN
    
    @staticmethod
    def volatility_rescaling_algorithm(y_true:pd.Series,
                                     y_pred:pd.Series,
                                     vol_pred:pd.Series,
                                     oob_start:Union[str, datetime],
                                     oob_end:Union[str, datetime])->pd.Series:
        """
        PARAMETERS
        ----------
        y_true    : the true values of the target variable.
        
        y_pred    : the ensemble prediction of the conditional mean.
                    predictions on the training set must be out-of-bag.
                    
        vol_pred  : the ensemble prediction conditional volatility.
                    predictions on the training set must be out-of-bag.
                    
        oob_start : the start index of the out-of-bag (OOB) samples.
        
        oob_end   : the end index of the out-of-bag (OOB) samples.
    
        RETURNS
            VolatilityRescaled : the rescaled predicted volatility.
    
        """
    
        # concatenate the true and predicted values and drop any missing data
        merged = pd.concat([y_true, y_pred], axis=1).dropna()
    
        # calculate the model residuals
        HNN_residuals = merged.iloc[:, 0] - merged.iloc[:, 1]
    
        # concatenate the squared residuals and estimated volatility squared,
        # and only keep the
        # forecast on the train set (i.e., OOB)
        merged_oob = pd.concat([HNN_residuals**2,
                                vol_pred**2], axis=1).loc[oob_start:oob_end]
    
        # use the squared estimated volatility as the input (X) for the
        # linear regression
        X_train = merged_oob.iloc[:, 1].values.reshape(-1, 1)
    
        # use the squared residuals as the output (y) for the linear regression
        y_train = merged_oob.iloc[:, 0].values.reshape(-1, 1)
    
        # create a linear regression object and fit the training data
        linreg = LinearRegression()
        linreg.fit(np.log(X_train), np.log(y_train))
    
        # use the full predicted volatility as the input for the linear
        # regression
        full_X = np.log(vol_pred**2)
    
        # predict the log of the residuals based on the log of the estimated
        # volatility
        projection = linreg.predict(full_X.values.reshape(-1, 1))
    
        # store the predictions in a dataframe
        pred = pd.DataFrame(projection, columns=["vol"], index=y_pred.index)
    
        # merge the log squared errors of the model and the predictions and
        # drop any missing data
        merged2 = pd.concat([np.log(merged_oob.iloc[:, 0]), pred],
                            axis=1).dropna()
    
        # calculate the residuals between the two merged dataframes
        res = merged2.iloc[:, 0].values - merged2.iloc[:, 1].values
    
        # calculate the scaler by taking the square root of the mean
        # exponential of the residuals
        scaler = np.sqrt(np.mean(np.exp(res)))
    
        # adjust the predicted volatility based on the scaler
        VolatilityRescaled = np.sqrt(np.exp(pred)) * scaler
    
        return VolatilityRescaled
        
    def run(self,
              data:Union[pd.DataFrame, pd.Series],
              train_start = Union[str, datetime],
              train_end = Union[str, datetime],
              expanding:bool=False,
              expanding_start:Union[str, datetime]=None,
              expanding_steps:int=8,
              last_expanding_window:Union[str, datetime]=None
              )->dict:
        
        """
        PARAMETERS
        ----------
        data            : The input data to be used.
        
        train_start     : The starting date or timestamp for the training
                          data.
        
        train_end       : The ending date or timestamp for the training
                          data.
        
        expanding       : If True, use an expanding training window.
                           Default is False.
                    
        expanding_start : The starting date or timestamp for the expanding
                          training window.
                          
        expanding_steps : The number of steps to expand the training window.
                          Only relevant when 'expanding' is True.
                          Default is 8.
                          
        last_expanding_window : The end date or timestamp for the last
                                expanding training window. Only relevant
                                when 'expanding' is True.

        RETURNS
            results : a python dictionary the ensemble forecast for the
                      prior_dnn and the hnn. Also, the dictionary contains the
                      estimated volatility emphasis parameter.
        """

        # making sure the data respects the index constraints.
        data = self._validate_dataset(data)
                  
        if expanding: # performs __static_fit() for each expanding window.
            windows = WindowEstimationHandler(data,
                                              expanding_start=expanding_start,
                                              last_window=last_expanding_window,
                                              timesteps=expanding_steps)
            pred_idx = pd.date_range(
                data.index[0+self.lags+self.horizon-1],
                periods=len(data)-self.lags+1,
                freq=pd.infer_freq(data.index)
            )
            dnn_rolling_outputs = {}
            hnn_rolling_outputs = {}
            vol_emphasis        = {}
            
            # main estimation loop.
            for step, (train, oos) in enumerate(windows):
                if self.verbose > 0:
                    print(" ")
                    print(
                        f"Expanding[{step+1}/{len(windows)}] : \n", 
                        f"    Estimation start: {train.index[0].strftime('%Y-%m-%d')}",
                        f" - Estimation end: {train.index[-1].strftime('%Y-%m-%d')}"
                        )
                    print(" ")
                
                window_results = self._static_fit(data,
                                                  train_start = train.index[0],
                                                  train_end = train.index[-1])   
                dnn_rolling_outputs[f"{oos.index[0]}"] = window_results[
                    "dnn_forecast"]
                hnn_rolling_outputs[f"{oos.index[0]}"] = window_results[
                    "hnn_forecast"]
                vol_emphasis[f"{oos.index[0]}"] = window_results[
                    "volatility_emphasis"]
                
            dnn_forecast = pd.DataFrame(columns=["dnn_forecast"],
                            index=pred_idx)

            for i, (step, output) in enumerate(dnn_rolling_outputs.items()):
                if i == 0:
                    dnn_forecast.loc[
                        :train_end, "dnn_forecast"] = output.loc[
                            :train_end, "forecast"]             
                dnn_forecast.loc[
                    step:, "dnn_forecast"] = output.loc[step:, "forecast"]
            dnn_forecast = dnn_forecast.astype("float32")
                
            
            hnn_forecast = pd.DataFrame(columns=["conditional_mean",
                                                 "conditional_vol"],
                                        index=pred_idx) 
            for i, (step, output) in enumerate(hnn_rolling_outputs.items()):
                if i == 0:
                    hnn_forecast.loc[
                        :train_end, "conditional_mean"] = output.loc[
                            :train_end, "conditional_mean"]
                    hnn_forecast.loc[
                        :train_end, "conditional_vol"] = output.loc[
                            :train_end, "conditional_vol"]
                    
                hnn_forecast.loc[
                    step:, "conditional_mean"] = output.loc[
                        step:, "conditional_mean"]
                hnn_forecast.loc[
                    step:, "conditional_vol"] = output.loc[
                        step:, "conditional_vol"]
                        
            hnn_forecast = hnn_forecast.astype("float32")
                        
            # to keep track of the volatility emphasis across each expanding
            # window.
            volatility_emphasis = pd.DataFrame.from_dict(vol_emphasis,
                                                         orient="index")
            volatility_emphasis.columns = ["volatility_emphasis"]
            volatility_emphasis.index.name = "date"
            # the resulting dict.
            results = {"hnn_forecast":hnn_forecast,
                       "dnn_forecast":dnn_forecast,
                       "volatility_emphasis":volatility_emphasis}
        else:
            results = self._static_fit(data, train_start=train_start, 
                                      train_end=train_end)
        return results
    
    
    
    def _static_fit(self, data:Union[pd.DataFrame, pd.Series],
                   train_start:Union[str, datetime],
                   train_end:Union[str, datetime]):

        # adding trends to the data
        data = add_trends(data, self.trends)
                       
        #instantiate a scaler
        scaler = StandardScaler()
                       
        # get the statistical parameters needed on the train set (mean, std)
        scaler.get_stats(data.loc[train_start:train_end])
                       
        # scale using the computed mean and std
        full_scaled = scaler.scale(data)
                       
        # split
        train_scaled = full_scaled.loc[train_start:train_end]
        # create data ready to be used by the models
        X_train, y_train = self.__pipeline(train_scaled)
        X_full, y_full = self.__pred_pipeline(full_scaled)
        input_shape=(self.lags, X_train.shape[-1]) # defining input shapes.

        # instantiate sampler
        self.__sampler = TimeSeriesBlockBootstrap(
            X=X_train, y=y_train,
            sampling_rate=self.sampling_rate,
            block_size=self.block_size,
            replace=True
        )

        # build prior fully connected network
        prior_dnn = DensityHNN.prior_dnn_architecture(input_shape)
    
        pred_idx = pd.date_range(
            data.index[0+self.lags+self.horizon-1],
            periods=len(data)-self.lags+1,
            freq=pd.infer_freq(data.index)
            )
        tf.keras.backend.clear_session() # this is important so we do not
        # get overlapping graphs or tensorflow session during estimations.

        # only one tensorflow callback is used for the model.
        callbacks = [keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=self.patience)]

        # custom trainer
        dnn_trainer = Trainer(
            optimizer = keras.optimizers.Adam(self.learning_rate),
            loss      = keras.losses.MeanSquaredError(),
            metrics   = None,
            n_trainings = self.prior_bootstraps
        )
                       
        # instantiate and fit
        prior_dnn = DensityHNN.prior_dnn_architecture(input_shape)
        ensemble_dnn = DeepEnsemble( # wrap into a deep ensemble.
            n_estimators=self.prior_bootstraps,
            network = prior_dnn,
            trainer = dnn_trainer,
            sampler = self.__sampler
            )
        ensemble_dnn.fit(
            X_train, y_train, epochs=self.epochs,
            early_stopping=callbacks[0],
            batch_size=None,
            validation_batch_size=None,
            verbose=self.verbose
            )               
        # By default, the predict method will return all estimators forecasts.
        dnn_prediction = ensemble_dnn.predict(X_full,
                                   batch_size=X_full.shape[0],
                                   verbose=0)
        # gather and create bluprints for out-of-bag forecasts.
        oob_idx = ensemble_dnn.oob_indices
        oob_predictor = OutOfBagPredictor(oob_idx)

        # oob and oos forecasts
        dnn_oob = oob_predictor(np.copy(dnn_prediction[:len(y_train)]),
                                sampling_rate=self.sampling_rate)
        dnn_oos = np.mean(np.copy(dnn_prediction[len(y_train):]), axis=-1)
        dnn_forecast = np.concatenate([dnn_oob, dnn_oos], axis=0)
    
        # compute volatility emphasis using the MSE on the out-of-bag forecast.
        volatility_emphasis = mean_squared_error(dnn_oob, np.squeeze(y_train))
        
        dnn_pred = scaler.unscale(
            pd.DataFrame(dnn_forecast, columns=["forecast"], index=pred_idx),
            target=self.target, use_mean=True
        )
                       
        tf.keras.backend.clear_session()
        # instantiate and fit
        densityhnn = DensityHNN.base_architecture(
            input_shape, hemisphere_outputs=1, 
            vol_emphasis=tf.constant(volatility_emphasis, dtype=tf.float32)
            )
        # custom trainer
        hnn_trainer = Trainer(
            optimizer = keras.optimizers.Adam(self.learning_rate),
            loss      = GaussianLogLikelihood(),
            metrics   = None,
            n_trainings=self.bootstraps,
        )
        self.ensemble_hnn = DeepEnsemble( # wrap in ensemble.
            n_estimators=self.bootstraps,
            network = densityhnn,
            trainer = hnn_trainer,
            sampler = self.__sampler
        )
        self.ensemble_hnn.fit(
            X_train, y_train, epochs=self.epochs,
            early_stopping=callbacks[0],
            batch_size=None,
            validation_batch_size=None,
            verbose=self.verbose
            )
        # make prediction
        mean_preds, vol_preds = self.ensemble_hnn.predict(
            X_full, batch_size=X_full.shape[0], verbose=0
        )
        # gather and create bluprints for out-of-bag forecasts.
        oob_idx = self.ensemble_hnn.oob_indices
        oob_predictor = OutOfBagPredictor(oob_idx)
        hnn_mean_oob = oob_predictor(np.copy(mean_preds[:len(y_train)]),
                                     sampling_rate=self.sampling_rate)        
        hnn_mean_oos = np.mean(np.copy(mean_preds[len(y_train):]), axis=-1)
        hnn_mean = np.concatenate([hnn_mean_oob, hnn_mean_oos], axis=0)
    
        hnn_mean = scaler.unscale(
            pd.DataFrame(hnn_mean, columns=["conditional_mean"], index=pred_idx),
            self.target, use_mean=True
        )
        hnn_vol_oob = oob_predictor(np.copy(vol_preds[:len(y_train)]),
                                    sampling_rate=self.sampling_rate)
        hnn_vol_oos = np.mean(np.copy(vol_preds[len(y_train):]), axis=-1)
        hnn_vol = np.concatenate([hnn_vol_oob, hnn_vol_oos], axis=0)
        hnn_vol = scaler.unscale(
            pd.DataFrame(hnn_vol, columns=["conditional_vol"], index=pred_idx),
            self.target, use_mean=False
        )
        hnn_preds = pd.concat([hnn_mean, hnn_vol], axis=1)
        hnn_preds.columns =["conditional_mean", "conditional_vol"]
        hnn_preds.index = pred_idx
                       
        # rescaling the volatility
        hnn_preds["conditional_vol"] = DensityHNN.volatility_rescaling_algorithm(
                                y_true    = data[[self.target]],
                                y_pred    = hnn_preds["conditional_mean"],
                                vol_pred  = hnn_preds["conditional_vol"],
                                oob_start = train_start,
                                oob_end   = train_end)
        
        return {"hnn_forecast":hnn_preds,
                "dnn_forecast":dnn_pred,
                "dnn_bootstraps":dnn_prediction,
                "hnn_bootstraps":{"conditional_mean":mean_preds,
                                  "conditional_vol":vol_preds},
                "volatility_emphasis":volatility_emphasis}
    
    def _validate_dataset(self, data):
        try: 
            pd.infer_freq(data.index)
        except TypeError:
            raise ValueError("Pandas cannot infer frequency from the provided "+
                             "data's index. Please make sure that your data "+
                             "has a valid datetime index. See panda's " +
                             "documentation for more information.")
        else:
            return data
        
    
    
