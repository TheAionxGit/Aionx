import pandas as pd
import numpy as np

def _has_multi_level_datetime_index(df : pd.DataFrame) -> bool:
    """
    ------------------------------------------------------------------------------------------------
    Returns True if the DataFrame has a multi-level index and one of the levels is a datetime index.
    ------------------------------------------------------------------------------------------------
    
    PARAMETERS:
    df : pandas.DataFrame. The DataFrame to check.
    
    RETURNS:
    bool: True if the DataFrame has a multi-level index and one of the levels is a datetime index, False otherwise.
    """
    
    index_levels = df.index.nlevels
    if index_levels > 1:
        for level in range(index_levels):
            if isinstance(df.index.get_level_values(level), pd.DatetimeIndex):
                return True
    return False

def AutoRegressionMatrix(df : pd.DataFrame, target:str, lags=1, h=1):
    """
    ----------------------------------------------------------------------------------------
    Builds regression matrix for time dependant data. Will use the entire provided data.
    ----------------------------------------------------------------------------------------
    PARAMS
    
    df: Pandas DataFrame object. The dataset that will be used to build the regression matrix.
    
    target: String. The name of the target variable. Must be a column in the provided df.
    
    lags : Int. The number of lags to use. Will be applied to all columns in the provided df.
                Default to 1.
    
    horizon: Int. The horizon in the futur to forecast. Default to 1.
    --------------------------------------------------------------------------------------
    
    RETURN
        pandas DataFrame ready for regression.
        
    RAISES
        None.    
    """
    
    # check if data is panel.
    is_panel_data = _has_multi_level_datetime_index(df)
    
    if is_panel_data:
        
        def build():
            
            datasets = []
            
            for i, (level_name, data) in enumerate(df.groupby(level=0)):
            
                # create a dataset with the good indexes
                dataset = pd.DataFrame(index = data.index)

                # keep target's data as it is.
                dataset.loc[:, target] = data.loc[:, target].values
                
                for col in data:
                    # build lag for each columns
                    for lag in range(1, lags+1):
                        dataset.loc[:, f"{col}_L{lag}"] = data[col].shift(lag+h-1)
                        
                datasets.append(dataset)

            return pd.concat(datasets, axis=0) # return the dataset.
        
    else:
        
        def build():
            # create a dataset with the good indexes
            dataset = pd.DataFrame(index = df.index)

            # keep target's data as it is.
            dataset.loc[:, target] = df.loc[:, target].values

            # iterate over columns
            for col in df:
                # build lag for each columns
                for lag in range(1, lags+1):
                    dataset.loc[:, f"{col}_L{lag}"] = df[col].shift(lag+h-1)

            return dataset # return the dataset.
        
    return build()
          
    
def reshape_to_2d(arr) -> np.ndarray:
    """
    Reshapes a Pandas DataFrame or NumPy array into a 2D array.

    Parameters:
    arr (Pandas DataFrame or NumPy array): The input data to be reshaped.

    Returns:
    NumPy array: A 2D NumPy array with the same data as the input.

    Raises:
    ValueError: If the input is not a Pandas DataFrame or NumPy array,
                or if the input is not 1D or 2D.
    """
    
    # transform to numpy array
    if isinstance(arr, pd.DataFrame):
        arr = arr.to_numpy()
            
    if not isinstance(arr, np.ndarray):
        # if other than numpy array or pandas dataframe.
        raise ValueError("Input must be a Pandas DataFrame or NumPy array.")
        
    # reshape only if dim == 1. 
    if len(arr.shape) == 1:
        arr = arr.reshape((-1, 1))
    
    # if dim > 2. input is not valid.
    elif len(arr.shape) > 2:
        raise ValueError("Input must be 1D or 2D.")
    return arr
