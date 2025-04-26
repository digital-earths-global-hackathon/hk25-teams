#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
(c) 2025 under a MIT License (https://mit-license.org)

Authors:
- Lukas Brunner || lukas.brunner@uni-hamburg.de

"""

import numpy as np
import xarray as xr
import healpy as hp


def aggregate_grid(arr: np.ndarray, z_out: int, method: str='mean') -> np.ndarray:
    """Spatially aggregate to a coarser grid.

    Parameters
    ----------
    arr : np.ndarray, shape (M,)
        The length of the array M has to be M = 12 * (2**zoom)**2
    z_out : int
        Healpix zoom level of the output grid. Needs to be smaller than the input zoom level.
    method : str, optional by default 'mean'
        Spatial aggregation method. 
        - 'mean': Mean of sub-grid cells -> conservative regridding
                  This is equivalent to `healpy.ud_grade`
        - 'std': Standard deviation of sub-grid cells
        - 'min': Minimum of sub-grid cells
        - 'max': Maximum of sub-grid cells

    Returns
    -------
    np.ndarray, shape (N < M,)

    Info
    ----
    Zoom levels in healpix (Hierarchical Equal Area isoLatitude Pixelization of a sphere)
    https://healpy.readthedocs.io/en/latest/index.html
    https://healpix.jpl.nasa.gov/index.shtml

    Zoom level 0 divides the globe into 12 grid cells, each zoom level
    increase increases the number of cells by 4 and half the resolution
    in kilometer

    nside = 2**zoom
    nr. cells = 12 * nside**2

    | zoom | nside | res. (km) | nr. cells  |
    | ---- | ----- | --------- | ---------- |
         0 |     1 |    6519.6 | 12
         1 |     2 |    3259.8 | 48
         2 |     4 |    1629.9 | 192
         3 |     8 |     815.0 | 768
         4 |    16 |     407.5 | 3,072
         5 |    32 |     203.7 | 12,288
         6 |    64 |     101.9 | 49,152
         7 |   128 |      50.9 | 196,608
         8 |   256 |      25.5 | 786,432
         9 |   512 |      12.7 | 3,145,728
        10 |  1024 |       6.4 | 12,582,912
        11 |  2048 |       3.2 | 50,331,648
        12 |  4096 |       1.6 | 201,326,592
    """
    npix_in = arr.size
    npix_out = hp.nside2npix(2**z_out)
    
    if npix_out >= npix_in:
        raise ValueError('Outuput zoom level needs to be smaller than input zoom level')

    ratio = npix_in / npix_out
    if not ratio.is_integer():  # this should never happen
        raise ValueError(f'{ratio=}')
    else:
        ratio = int(ratio)
    
    if method == 'mean':
        return arr.reshape(npix_out, ratio).mean(axis=-1)
    if method == 'std':
        return arr.reshape(npix_out, ratio).std(axis=-1)
    if method == 'min':
        return arr.reshape(npix_out, ratio).min(axis=-1)
    if method == 'max':
        return arr.reshape(npix_out, ratio).max(axis=-1)
        
    raise ValueError(f'{method=}')


def guess_gridn(da: xr.DataArray) -> str:
    """Try to gess the name of the spatial coordinate name from a list of frequent options."""
    dims = list(da.dims)
    gridn = []
    if 'values' in dims:
        gridn.append('values')
    if 'value' in dims:
        gridn.append('value')
    if 'cell' in dims:
        gridn.append('cell')
    if 'x' in dims:
        gridn.append('x')
    if len(gridn) == 1:
        return gridn[0]

    raise ValueError('gridn needs to be set manually to one of: {}'.format(', '.join(dims)))


def attach_grid_info(da: xr.DataArray, gridn=None, return_latlon=False) -> xr.Dataset:
    """Attach to longitude and latitude values of each grid cell to the Dataset.

    Parameters
    ----------
    da : xr.DataArray
    gridn : string, optional, by default None
        String specifying the name of the grid variable. If None, try to guess it from frequent options
    return_latlon: bool, optional, by default False
        If True, return the grid values as xr.DataArrays instead of creating a xr.Dataset and attaching them.

    Returns
    -------
    xr.Dataset
        Dataset with the grid information attached as two new variables.
    """
    if gridn is None:  # try to guess grid name from frequent options
        gridn = guess_gridn(da)
        
    lon, lat = hp.pix2ang(hp.npix2nside(da[gridn].size), da[gridn].values, nest=True, lonlat=True) 
    lon = xr.DataArray(
        lon, 
        coords={gridn: da[gridn].values},
        attrs={'units': 'degree_east', 'long_name': 'longitude'},
    )

    lat = xr.DataArray(
        lat, 
        coords={gridn: da[gridn].values},
        attrs={'units': 'degree_north', 'long_name': 'latitude'},
    )

    if return_latlon:
        return lat, lon

    da.attrs.update({
        'coordinates': 'lat lon',
        'gridType': 'healpix',
    })
    ds = da.to_dataset()
    ds['lon'] = lon
    ds['lat'] = lat
    return ds
    

def aggregate_grid_xr(da: xr.DataArray, z_out: int, method: str='mean', gridn=None) -> xr.DataArray:
    """Thin xarray wrapper for `aggregate_grid'."""
    
    if gridn is None:  # try to guess grid name from frequent options
        gridn = guess_gridn(da)
            
    return xr.apply_ufunc(
        aggregate_grid,
        da, z_out,
        input_core_dims=[[gridn], []],
        output_core_dims=[['tmp']],
        vectorize=True,
        kwargs={'method': method},
    ).rename({'tmp': gridn})


def subgrid_anomaly(fine: np.ndarray, z_coarse=None, coarse: np.ndarray=None) -> np.ndarray:
    """Calculate the sub grid anomaly as difference of a fine grid minus a coarse grid. Output is on the fine grid.

    Parameters
    ----------
    fine : np.ndarray, shape (M,)
    z_coarse : int, optional
    coarse : np.ndarray, optional, shape (N < M,)

    Returns
    -------
    np.ndarray, shape (M,)

    Info
    ----
    The simplest way of using this function is to give only a dataset on a fine grid and a coarse zoom level `z_coarse`. The data on the fine grid will then be evaluated against itself on the coarse grid (which is calculated on the fly). 

    Alternatively data on a coarse grid can be provided (e.g., by first calculating them using `aggregate_grid`). This can also be used to calculate the difference between two different datasets on different zoom levels (e.g., an extreme index calculated on a fine grid against an extreme index calculated on a coarse grid)
    """
    npix_fine = fine.size

    if coarse is None:
        if z_coarse is None:
            raise ValueError('Either `coarse` or `z_coarse` needs to be set')
        npix_coarse = hp.order2npix(z_coarse)
    elif z_coarse is None:
        npix_coarse = coarse.size
    else:
        if hp.order2npix(z_coarse) != coarse.size:
            raise ValueError('If `coarse` and `z_coarse` are given, theny need to be consistent')
        npix_coarse = coarse.size


    if npix_coarse > npix_fine:
        raise ValueError('`fine` needs to have a higher zoom levels than `coarse`')

    ratio = npix_fine / npix_coarse 
    if not ratio.is_integer():  # this should never happen
        raise ValueError(f'{ratio=}')
    else:
        ratio = int(ratio)

    if coarse is not None:
        return (fine.reshape(npix_coarse, ratio) - coarse.reshape(npix_coarse, 1)).ravel()
    return (fine.reshape(npix_coarse, ratio) - aggregate_grid(fine, z_coarse).reshape(npix_coarse, 1)).ravel()



def subgrid_anomaly_xr(da_fine, z_coarse=None, da_coarse=None, gridn=None):
    """Thin xarray wrapper for `subgrid_anomaly`."""
    if gridn is None:  # try to guess grid name from frequent options
        gridn = guess_gridn(da_fine)

    if da_coarse is not None:
        return xr.apply_ufunc(
            subgrid_anomaly,
            da_fine, da_coarse.rename({gridn: 'tmp'}),
            input_core_dims=[[gridn], ['tmp']],
            output_core_dims=[[gridn]],
            vectorize=True,
            kwargs={'z_coarse': z_coarse},
        )
    return xr.apply_ufunc(
        subgrid_anomaly,
        da_fine,
        input_core_dims=[[gridn]],
        output_core_dims=[[gridn]],
        vectorize=True,
        kwargs={'z_coarse': z_coarse},
    )

