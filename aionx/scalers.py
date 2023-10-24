"""
aionx scalers.py is a file for storing metric classes.

The module contains:
    
    - The 'MissingStatsError' class:
        An Exception class inheriting from base Exception.
        
    - The 'Scaler' class:
        A base class for Scaling / unscaling pandas data.
        
    - The 'StandardScaler' class:
        A class for scaling/unscaling dataframes values.

"""

# Author: Mikael Frenette (mik.frenette@gmail.com)
# Created: August 2023

import pandas as pd
from aionx.base import Scaler
from aionx.wrappers import downstream

class MissingStatsError(Exception):
    """
    DESCRIPTION
    ---------------------------------------------------------------------------
    An exception raised when summary statistics are missing.
    ---------------------------------------------------------------------------
    
    PARAMETERS
    ----------
    message : The error message to be displayed.
    """
    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(self.message)


class StandardScaler(Scaler):
    """
    DESCRIPTION
    ---------------------------------------------------------------------------
    This class can perform Standard scaling as well as unscaling operations
    on pandas DataFrame object. 
    
    X_hat = (X - X_mean) / X_std
    ---------------------------------------------------------------------------

    USAGE
    -----
    # Create a StandardScaler instance
    scaler = StandardScaler()

    # Calculate and store statistics for a DataFrame
    data = pd.DataFrame({'feature1': [1, 2, 3, 4, 5],
                         'feature2': [2, 3, 4, 5, 6]})
    scaler.get_stats(data)

    # Scale the data
    scaled_data = scaler.scale(data)

    # Unscale the data
    feature1_unscaled = scaler.unscale(scaled_data, target='feature1')
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    
    @downstream(
        pd.DataFrame, dtype = "float32"
    )
    def scale(self, data: pd.DataFrame, target: str = None) -> pd.DataFrame:
        """
        PARAMETERS
        ----------
        data   : The input data to be scaled.
        
        target : The target variable name for which scaling should be applied.
                 If None, scaling is applied to all columns.

        RETURNS
            pd.DataFrame: The scaled data.
        """
        
        if isinstance(data, pd.Series):
            data = data.to_frame(data.name)
        
        df = data.copy()

        # Check if summary stats are available
        if not self._stats:
            raise MissingStatsError(
                " ".join([
                    "No summary stats found.",
        "Call the get_stats() method before scaling or unscaling the data."])
            )

        if target is not None:
            for col in df:
                df[col] = (df[col] - self.stats["mean"].loc[
                    target]) / self.stats["std"].loc[target]
        else:
            for col in df:
                df[col] = (df[col] - self.stats[col][
                    "mean"]) / self.stats[col]["std"]

        return df
    
    @downstream(
        pd.DataFrame, dtype="float32"
    )
    def unscale(self,
                data: pd.DataFrame,
                target: str = None,
                use_mean: bool = True) -> pd.DataFrame:
        """
        PARAMETERS
        ----------
        data     : The scaled data to be unscaled.
        
        target   : The target variable name for which unscaling should be
                   applied. If None, unscaling is applied to all columns
                   with respect to their original stats.
        
        use_mean : Whether to add the mean back during unscaling
                  (default is True).

        RETURNS
            pd.DataFrame: The unscaled data.
        """
        if isinstance(data, pd.Series):
            data = data.to_frame(data.name)
        
        df = data.copy()

        # Check if summary stats are available
        if not self._stats:
            raise MissingStatsError(
                " ".join([
                    "No summary stats found.",
        "Call the get_stats() method before scaling or unscaling the data."])
            )       
        if target is not None:
            if use_mean:
                for col in df:
                    df[col] = (df[col] * self.stats[
                        target]["std"]) + self.stats[target]["mean"]
            else:
                for col in df:
                    df[col] = df[col] * self.stats[target]["std"]
        else:
            if use_mean:
                for col in df:
                    df[col] = (df[col] * self.stats[
                        col]["std"]) + self.stats[col]["mean"]
            else:
                for col in df:
                    df[col] = df[col] * self.stats[col]["std"]
                    
        return df




