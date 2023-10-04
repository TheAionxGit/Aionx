import pandas as pd
import sys
sys.path.append("C:/Users/User/dropbox/Prototyping")
from typing import Union, List, Iterable
from Prototyping.wrappers import series_to_dataframe

class MissingStatsError(Exception):
    """
    An exception raised when summary statistics are missing for a required operation.

    Args:
        message (str): The error message to be displayed.
    """
    def __init__(self, message:str)->None:
        """
        Initialize a MissingStatsError instance with a custom error message.

        Args:
            message (str): The error message to be displayed.
        """
        self.message = message
        super().__init__(self.message)


class Scaler(object):
    """
    A base class for data scaling operations.

    This class provides methods to calculate and store summary statistics (mean and standard deviation) of data columns.
    It also offers a property to access the stored statistics.

    Methods:
        get_stats(data): Calculate and store summary statistics for the data.
    """

    def __init__(self)->None:
        """
        Initialize a Scaler instance.
        """
        self._stats = {}
        
    @series_to_dataframe
    def get_stats(self, data:Union[pd.DataFrame, pd.Series])->None:
        """
        Calculate and store summary statistics (mean and standard deviation) for data columns.

        Args:
            data (pd.DataFrame): The input data for which to calculate statistics.
        """
        
        for col in data:
            summarystat = {}
            summarystat["mean"] = data[col].mean()
            summarystat["std"] = data[col].std()
            self._stats[col] = summarystat
        
    @property
    def stats(self)->dict:
        """
        Access the stored summary statistics.

        Returns:
            dict: A dictionary containing mean and standard deviation statistics for data columns.
        """
        return self._stats


class StandardScaler(Scaler):
    """
    A class for performing standard scaling and unscaling of data.

    This class inherits from the Scaler class and provides methods to perform
    standard scaling and unscaling operations on data using mean and standard deviation statistics.

    Args:
        **kwargs: Additional keyword arguments to pass to the parent class.

    Attributes:
        _stats (dict): A dictionary containing mean and standard deviation statistics for the data.

    Methods:
        scale(data, target=None): Scale the input data using mean and standard deviation statistics.
        unscale(data, target=None, use_mean=True): Unscale the scaled data using mean and standard deviation statistics.
    """

    def __init__(self, **kwargs):
        """
        Initialize a StandardScaler instance.

        Args:
            **kwargs: Additional keyword arguments to pass to the parent class.
        """
        super().__init__(**kwargs)
        
    def scale(self, data, target: str = None) -> pd.DataFrame:
        """
        Scale the input data using mean and standard deviation statistics.

        Args:
            data (pd.DataFrame): The input data to be scaled.
            target (str): The target variable name for which scaling should be applied. If None,
                         scaling is applied to all columns.

        Returns:
            pd.DataFrame: The scaled data.
        """
        df = data.copy()

        # Check if summary stats are available
        if not self._stats:
            raise MissingStatsError(
                " ".join(["No summary stats found.",
                          "Call the get_stats() method before scaling or unscaling the data."])
            )

        if target is not None:
            for col in df:
                df[col] = (df[col] - self.stats["mean"].loc[target]) / self.stats["std"].loc[target]
        else:
            for col in df:
                df[col] = (df[col] - self.stats[col]["mean"]) / self.stats[col]["std"]

        return df
    
    def unscale(self, data, target: str = None, use_mean: bool = True) -> pd.DataFrame:
        """
        Unscale the scaled data using mean and standard deviation statistics.

        Args:
            data (pd.DataFrame): The scaled data to be unscaled.
            target (str): The target variable name for which unscaling should be applied. If None,
                         unscaling is applied to all columns.
            use_mean (bool): Whether to add the mean back during unscaling (default is True).

        Returns:
            pd.DataFrame: The unscaled data.
        """
        df = data.copy()

        # Check if summary stats are available
        if not self._stats:
            raise MissingStatsError(
                " ".join(["No summary stats found.",
                          "Call the get_stats() method before scaling or unscaling the data."])
            )       
        if target is not None:
            if use_mean:
                for col in df:
                    df[col] = (df[col] * self.stats[target]["std"]) + self.stats[target]["mean"]
            else:
                for col in df:
                    df[col] = df[col] * self.stats[target]["std"]
        else:
            if use_mean:
                for col in df:
                    df[col] = (df[col] * self.stats[col]["std"]) + self.stats[col]["mean"]
            else:
                for col in df:
                    df[col] = df[col] * self.stats[col]["std"]
                    
        return df

