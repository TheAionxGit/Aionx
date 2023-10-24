"""
aionx metrics.py is a file for storing metric classes.

The module contains:
    
    - The 'MeanSquaredError' class:
        A class for computing the mean squared error. 
        
    - The 'MeanAbsoluteError' class:
        A class for computing the mean absolute error.
        
    - The 'RootMeanSquaredError' class:
        A class for computing the root of the mean squared error.
        
    - The 'LogScore' class:
        A class for computing the Gaussian log score.
        
    - The 'NominalCoverage' class:
        A class for computing the norminal coverage.
"""

# Author: Mikael Frenette (mik.frenette@gmail.com)
# Created: September 2023

import numpy as np
from typing import List
from aionx import base

class MeanSquaredError(base.RegressionMetric):
    """
    DESCRIPTION
    ---------------------------------------------------------------------------
    Class to compute the Mean squared error between a provided true target
    (y_true) and predicted values (y_pred). Inherits from RegressionMetrics
    which controls for inputs formatting.
    ---------------------------------------------------------------------------
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self,
                 y_true:np.ndarray,
                 y_pred:np.ndarray) -> float:
        """
        PARAMETERS
        ----------
        y_true : True values.
        
        y_pred : Predicted value

        RETURNS
            float : the computed metric with the specified precision type.
        """
        super().__call__(y_true, y_pred)
        # Calculate the mean squared error
        metric = np.mean((y_true - y_pred) ** 2)
        return np.array(metric, dtype=self.precision)



class MeanAbsoluteError(base.RegressionMetric):
    """
    DESCRIPTION
    ---------------------------------------------------------------------------
    Class to compute the Mean absolute error between a provided true target
    (y_true) and predicted values (y_pred). Inherits from RegressionMetrics
    which controls for inputs formatting.
    ---------------------------------------------------------------------------
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self,
                 y_true:np.ndarray,
                 y_pred:np.ndarray) -> float:
        """
        PARAMETERS
        ----------
        y_true : True values.
        
        y_pred : Predicted value

        RETURNS
            float : the computed metric with the specified precision type.
        """
        super().__call__(y_true, y_pred)
        
        # Calculate the mean absolute error
        metric = np.mean(np.abs(y_true - y_pred))
        
        return np.array(metric, dtype=self.precision)


class RootMeanSquaredError(base.RegressionMetric):
    """
    DESCRIPTION
    ---------------------------------------------------------------------------
    Class to compute the Root Mean squared error between a provided true target
    (y_true) and predicted values (y_pred). Inherits from RegressionMetrics
    which controls for inputs formatting.
    ---------------------------------------------------------------------------
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self,
                 y_true:np.ndarray,
                 y_pred:np.ndarray) -> float:
        """
        PARAMETERS
        ----------
        y_true : True values.
        
        y_pred : Predicted value

        RETURNS
            float : the computed metric with the specified precision type.
        """
        super().__call__(y_true, y_pred)
        
        # Calculate the root mean squared error
        metric = np.sqrt(np.mean((y_true - y_pred) ** 2))
        
        return np.array(metric, dtype=self.precision)
    

class LogScore(base.DensityMetric):
    """
    DESCRIPTION
    ---------------------------------------------------------------------------
    Class to compute the Gaussian Log Score between a provided true target
    (y_true) and a list containing predictions for the mean and standard
    deviation (y_pred). Inherits from DensityMetrics which controls for inputs
    formatting.
    ---------------------------------------------------------------------------
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def __call__(self,
                 y_true:np.ndarray,
                 y_pred:List[np.ndarray]) -> float:
        """
        PARAMETERS
        ----------
        y_true : True values.
        
        y_pred : predicted mean and standard deviationb values

        RETURNS
            float : the computed metric with the specified precision type.
        """
        y_true, y_pred = super().__call__(y_true, y_pred)
        
        # Convert inputs to numpy arrays for ease of calculation
        y, mu, sigma = y_true, np.squeeze(y_pred[0]), np.squeeze(y_pred[1])

        # Calculate the error between true values and mean predictions
        eps = y - mu

        # Calculate the log probability density function using the error and standard deviation
        log_pdf = -0.5 * np.log(2 * np.pi * sigma ** 2) - 0.5 * (eps ** 2) / ( (sigma ** 2) + self.tolerance )

        # Return the negative of the log likelihood as the log score
        return np.array(-np.mean(log_pdf), dtype=self.precision)



class NominalCoverage(base.DensityMetric):
    """
    DESCRIPTION
    ---------------------------------------------------------------------------
    Class to compute the Gaussian Log Score between a provided true target
    (y_true) and a list containing predictions for the mean and standard
    deviation (y_pred). Inherits from DensityMetrics which controls for inputs
    formatting.
    
    computes the ratio where the residuals fall into the standard deviation
    brackets
    
    if std == 1:
        computes eps +/- 1*sigma
        
    if std == 2:
        computes eps +/- 2*sigma
        
    ...        
    ---------------------------------------------------------------------------
    
    PARAMETERS
    ----------
    std : The number of standard deviations to use for the coverage
          interval. Defaults to 1.
    """
    def __init__(self, std:int=1, **kwargs):
        super().__init__(**kwargs)
        self.std = std

    def __call__(self,
                 y_true:np.ndarray,
                 y_pred:List[np.ndarray]) -> float:
        """
        PARAMETERS
        ----------
        y_true : True values.
        
        y_pred : predicted mean and standard deviationb values

        RETURNS
            float : the computed metric with the specified precision type.
        """
        y_true, y_pred = super().__call__(y_true, y_pred)
        y, mu, sigma = y_true, np.squeeze(y_pred[0]), np.squeeze(y_pred[1])

        # Compute the boolean masks for values outside the coverage interval
        above_interval = y - mu > (self.std * sigma)
        below_interval = y - mu < (-self.std * sigma)
        outside_interval = np.logical_or(above_interval, below_interval)

        # Calculate the number of values outside the interval
        not_covered = np.sum(outside_interval)

        # Calculate the coverage ratio
        coverage = 1 - (not_covered / len(y))

        return coverage

