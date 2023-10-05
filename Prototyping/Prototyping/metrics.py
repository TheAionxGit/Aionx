import numpy as np
from typing import Union, List, Iterable, Callable

class Metric(object):
    def __init__(self, precision="float32", tolerance=1e-7, name=None):
        self.precision = precision
        self.tolerance = tolerance
        self.name = name
        if self.precision not in ["float32", "float64"]:
            raise ValueError("Input for precision received {precision}. Valid" +
                             "inputs are float32 or float64")
    def _configure(self, y_true, y_pred):
        if not isinstance(y_true, np.ndarray):
            y_true = np.squeeze(np.array(y_true))
        if not isinstance(y_pred, np.ndarray):
            y_pred = np.squeeze(np.array(y_pred))   
        return y_true, y_pred


class RegressionMetric(Metric):
    """
    Parent class for calculating regression metrics
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def __call__(self, y_true:np.ndarray, y_pred:np.ndarray):
        y_true, y_pred = self._configure(y_true, y_pred)
        if y_true.shape != y_pred.shape:
            raise ValueError("Input arrays must have the same shape.")
            
            
class DensityMetric(Metric):
    """
    Parent class for calculating density metrics
    
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def __call__(self, y_true:np.ndarray, y_pred:Union[List[np.ndarray],
                                                       np.ndarray]):
        if isinstance(y_pred, np.ndarray):
            y_true, y_pred = self._configure(y_true, y_pred)
            y_pred = np.split(y_pred,
                              indices_or_sections=y_pred.shape[-1], axis=1)
        if y_true.shape != y_pred[0].shape !=y_pred[1].shape:
            raise ValueError("Input arrays must have the same shape.")
        return y_true, y_pred

class MSE(RegressionMetric):
    """
    Mean Squared Error (MSE) metric class.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self,
                 y_true:np.ndarray,
                 y_pred:np.ndarray) -> float:
        """
        PARAMETERS
        ----------
            y_true (array-like):
                True values.
            y_pred (array-like):
                Predicted values.
        RETURNS
        -------
            metric (float):
                The Mean squared error.
        """
        super().__call__(y_true, y_pred)
        # Calculate the mean squared error
        metric = np.mean((y_true - y_pred) ** 2)
        return np.array(metric, dtype=self.precision)



class MAE(RegressionMetric):
    """
    Mean Absolute Error (MAE) metric class.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self,
                 y_true:np.ndarray,
                 y_pred:np.ndarray) -> float:
        """
        PARAMETERS
        ----------
            y_true (array-like):
                True values.
            y_pred (array-like):
                Predicted values.

        RETURNS
        -------
        
            metric (float):
                The Mean absolute Error.
        """
        
        super().__call__(y_true, y_pred)
        
        # Calculate the mean absolute error
        metric = np.mean(np.abs(y_true - y_pred))
        
        return np.array(metric, dtype=self.precision)



class RMSE(RegressionMetric):
    """
    Root Mean Squared Error (RMSE) metric class.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self,
                 y_true:np.ndarray,
                 y_pred:np.ndarray) -> float:
        """
        PARAMETERS
        ----------
            y_true (array-like):
                True values.
            y_pred (array-like):
                Predicted values.

        RETURNS
        -------
            metric (float):
                The Root Mean absolute Error.
        """
        
        super().__call__(y_true, y_pred)
        
        # Calculate the root mean squared error
        metric = np.sqrt(np.mean((y_true - y_pred) ** 2))
        
        return np.array(metric, dtype=self.precision)
    

class LogScore(DensityMetric):
    """
    Log score for a Gaussian distribution metric class.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def __call__(self,
                 y_true:np.ndarray,
                 y_pred:List[np.ndarray]) -> float:
        """
        PARAMETERS
        ----------
            y_true (array-like):
                True values.
            y_pred (list of array-like):
                Predicted values
        RETURNS
        -------
            metric (float):
                The Root Mean absolute Error.
        """
        
        y_true, y_pred = super().__call__(y_true, y_pred)
        
        # Convert inputs to numpy arrays for ease of calculation
        y, mu, sigma = y_true, y_pred[0], y_pred[1]
        

        # Calculate the error between true values and mean predictions
        eps = y - mu

        # Calculate the log probability density function using the error and standard deviation
        log_pdf = -0.5 * np.log(2 * np.pi * sigma ** 2) - 0.5 * (eps ** 2) / ( (sigma ** 2) + self.tolerance )

        # Return the negative of the log likelihood as the log score
        return np.array(-np.mean(log_pdf), dtype=self.precision)



class NominalCoverage(DensityMetric):
    """
    Nominal coverage metric class
        
    PARAMETERS
    ----------
        std (int, optional):
            The number of standard deviations to use for the coverage interval.
            Defaults to 1.
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
            y_true (array-like):
                True values.
            y_pred (list of array-like):
                Predicted values
        RETURNS
        -------
            metric (float):
                The Root Mean absolute Error.
        """
        
        y_true, y_pred = super().__call__(y_true, y_pred)
        y, mu, sigma = y_true, y_pred[0], y_pred[1]

        # Compute the boolean masks for values outside the coverage interval
        above_interval = y - mu > (self.std * sigma)
        below_interval = y - mu < (-self.std * sigma)
        outside_interval = np.logical_or(above_interval, below_interval)

        # Calculate the number of values outside the interval
        not_covered = np.sum(outside_interval)

        # Calculate the coverage ratio
        coverage = 1 - (not_covered / len(y))

        return coverage

