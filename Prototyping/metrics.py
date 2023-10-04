import numpy as np



class StatisticalMetric:
    """
    A base class for statistical metrics.
    """

    def __init__(self, precision: str = "float32", tolerance:float=1e-8):
        """
        Initialize the StatisticalMetric class.

        Args:
            precision (str): Data type precision for calculations (default: "float32").
            tolerance (float): numerical tolerance (default: 1e-7).
        """
        self.precision = precision
        self.tolerance = tolerance




class MSE(StatisticalMetric):
    """
    Mean Squared Error (MSE) metric class, derived from StatisticalMetric.
    """

    def __init__(self, **kwargs):
        """
        Initialize the MSE class by calling the parent class's constructor.

        Args:
            **kwargs: Additional keyword arguments.
        """
        super().__init__(**kwargs)

    def __call__(self, y, mu) -> float:
        """
        Compute the mean squared error.

        Args:
            y_true (array-like): True values.
            y_pred (array-like): Predicted values.

        Returns:
            float: Mean squared error.
        """
        # Convert inputs to numpy arrays with specified precision
        y_true = np.array(y, dtype=self.precision)
        y_pred = np.array(mu, dtype=self.precision)
        
        # Check if shapes of input arrays match
        if y_true.shape != y_pred.shape:
            raise ValueError("Input arrays must have the same shape.")
        
        # Calculate the mean squared error
        metric = np.mean((y_true - y_pred) ** 2)
        
        return metric



class MAE(StatisticalMetric):
    """
    Mean Absolute Error (MAE) metric class, derived from StatisticalMetric.
    """

    def __init__(self, **kwargs):
        """
        Initialize the MAE class by calling the parent class's constructor.

        Args:
            **kwargs: Additional keyword arguments.
        """
        super().__init__(**kwargs)

    def __call__(self, y, mu) -> float:
        """
        Compute the mean absolute error.

        Args:
            y (array-like): True values.
            mu (array-like): Predicted values.

        Returns:
            float: Mean absolute error.
        """
        # Convert inputs to numpy arrays with specified precision
        y = np.array(y, dtype=self.precision)
        mu = np.array(mu, dtype=self.precision)

        # Calculate the mean absolute error
        metric = np.mean(np.abs(y - mu))
        
        return metric



class RMSE(StatisticalMetric):
    """
    Root Mean Squared Error (RMSE) metric class, derived from StatisticalMetric.
    """

    def __init__(self, **kwargs):
        """
        Initialize the RMSE class by calling the parent class's constructor.

        Args:
            **kwargs: Additional keyword arguments.
        """
        super().__init__(**kwargs)

    def __call__(self, y, mu) -> float:
        """
        Compute the root mean squared error.

        Args:
            y (array-like): True values.
            mu (array-like): Predicted values.

        Returns:
            float: Root mean squared error.
        """
        # Convert inputs to numpy arrays with specified precision
        y = np.array(y, dtype=self.precision)
        mu = np.array(mu, dtype=self.precision)
        
        # Check if shapes of input arrays match
        if y.shape != mu.shape:
            raise ValueError("Input arrays must have the same shape.")
        
        # Calculate the root mean squared error
        metric = np.sqrt(np.mean((y - mu) ** 2))
        
        return metric



class LogScore(StatisticalMetric):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def __call__(self, y, mu, sigma):
        """
        Calculates the log score for a given set of true values, mean predictions, and volatility.

        Args:
            y (array-like): True values.
            mu (array-like): Mean predictions.
            sigma (array-like): STD prediction.

        Returns:
            Negative of the log likelihood (log PDF) averaged over all observations.
        """
        # Convert inputs to numpy arrays for ease of calculation
        y = np.array(y, dtype=self.precision)
        mu = np.array(mu, dtype=self.precision)
        sigma = np.array(sigma, dtype=self.precision)

        # Calculate the error between true values and mean predictions
        eps = y - mu

        # Calculate the log probability density function using the error and standard deviation
        log_pdf = -0.5 * np.log(2 * np.pi * sigma ** 2) - 0.5 * (eps ** 2) / ( (sigma ** 2) + self.tolerance )

        # Return the negative of the log likelihood as the log score
        return -np.mean(log_pdf)



class NominalCoverage(StatisticalMetric):
    
    def __init__(self, std:int=1, **kwargs):
        super().__init__(**kwargs)
        """
        Initialize the NominalCoverageCalculator.

        Args:
            std (int, optional): The number of standard deviations to use for the coverage interval.
                                 Defaults to 1.
        """
        self.std = std

    def __call__(self, y, mu, sigma):
        """
        Computes the nominal coverage for a given set of true values, mean predictions, and standard deviations.

        Args:
            true (array-like): True values.
            pred (array-like): Predicted conditional mean.
            vol (array-like): Predicted conditional volatility.

        Returns:
            float: The nominal coverage.
        """
        # Convert the inputs to NumPy arrays
        y = np.array(y, dtype=self.precision)
        mu = np.array(mu, dtype=self.precision)
        sigma = np.array(sigma, dtype=self.precision)

        # Compute the boolean masks for values outside the coverage interval
        above_interval = y - mu > (self.std * sigma)
        below_interval = y - mu < (-self.std * sigma)
        outside_interval = np.logical_or(above_interval, below_interval)

        # Calculate the number of values outside the interval
        not_covered = np.sum(outside_interval)

        # Calculate the coverage ratio
        coverage = 1 - (not_covered / len(y))

        return coverage

