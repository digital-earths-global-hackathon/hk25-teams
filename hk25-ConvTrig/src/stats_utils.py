#%%
import xarray as xr
import numpy as np



# function to remove the daily mean from the data
def remove_daily_mean(ds, var):

    """
    remove the daily mean from the hourly data
    Parameters
    ----------
    ds : xarray.Dataset
        The dataset containing the variable to be detrended.
    var : str
        The name of the variable to be detrended.
    Returns
    -------
    xarray.Dataset
        A dataset containing the variable with the daily mean removed.

    """
    anomalies = xr.apply_ufunc(
        lambda x, mean: x - mean,
        ds[var].groupby("time.hour"),
        ds[var].groupby("time.hour").mean(),
        vectorize=True,
        dask="allowed",
    )

    # drop the time.hour dimension
    anomalies = anomalies.drop("hour")

    # return as dataset
    return xr.Dataset({var: anomalies}, coords=ds.coords)

