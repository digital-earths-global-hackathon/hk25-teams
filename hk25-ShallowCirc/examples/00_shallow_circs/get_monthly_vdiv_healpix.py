"""
Script for calculating monthly mean wind variables for different models on healpix grid
Each month is processed separately and saved locally as a netCDF.
If enough compute is available, will be more efficient to resample all output to time='1M' and 
process altogether.
NOTE: requires windspharm package to be installed in the virtual environment
"""

# Standard Stuff
from scipy.interpolate import NearestNDInterpolator
from tqdm import tqdm
from windspharm.xarray import VectorWind
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import healpix as hp
import easygems.healpix as egh
import healpy
import intake     # For catalogs

import os
import warnings
warnings.filterwarnings('ignore')


# SPECIFY LOCAL STORAGE
save_to = '/path/'

# PATH TO INTAKE CATALOG
catfn = '/home/tmerlis/hackathon/hackathon_cat_may14_main.yaml'


model = 'ifs_fesom'

months = ['01', '02', '03', '04', '05',
          '06', '07', '08', '09', '10', '11', '12']


def interpolate_field_lon_lat_all_levels(field, lon_coord="lon", lat_coord="lat",
                                         dlon=0.5, dlat=0.5, relative_resolution=2):
    """
        Interpolates a field defined on irregular (lon, lat) grid to a regular 2D grid
        over all time and pressure levels using nearest-neighbor interpolation.
    Parameters:
        field (xarray.DataArray): Input field with dimensions (time, plev, cell)
        lon_coord (str): Name of longitude coordinate
        lat_coord (str): Name of latitude coordinate
        dlon (float): new lon interval
        dlat (float): new lat interval
        relative_resolution (float): Controls output grid resolution
    Returns:
        xarray.DataArray: Interpolated field with dims (time, plev, lat, lon)
    """
    # Extract lon/lat from coordinates
    lon_points = field[lon_coord].values
    lat_points = field[lat_coord].values

    # nlon = nlat = int(np.sqrt(len(lon_points) * relative_resolution))
    lon = np.arange(dlon/2, 360, dlon)
    lat = np.arange(-90+dlat/2, 90, dlat)
    lon2, lat2 = np.meshgrid(lon, lat)
    points = np.stack((lon_points, lat_points), axis=1)

    def _interp_single_level(values_1d):
        if values_1d.size == 0:
            return np.full_like(lon2, np.nan, dtype=values_1d.dtype)
        return NearestNDInterpolator(points, values_1d)(lon2, lat2)

    interpolated = xr.apply_ufunc(
        _interp_single_level,
        field,
        input_core_dims=[['cell']],
        output_core_dims=[['lat', 'lon']],
        vectorize=True,
        dask='parallelized',
        dask_gufunc_kwargs={"output_sizes": {"lat": len(lat), "lon": len(lon)},
                            "allow_rechunk": True},
        output_dtypes=[field.dtype],
    )
    # Assign new coordinates
    interpolated = interpolated.assign_coords({"lat": lat, "lon": lon})
    interpolated = interpolated.transpose(..., "lat", "lon")
    return interpolated


def run_month(do_month='2020-01', model='xsh24_coarse'):

    # open catalog
    cat = intake.open_catalog(catfn)
    zoom_select = 7

    # load output -- options must be added for new models
    if 'xsh24' in model:
        ds = cat[model](zoom=zoom_select).to_dask()
        ds = ds.drop_vars('cell').pipe(egh.attach_coords)
    if 'scream' in model:
        ds = cat[model](zoom=zoom_select).to_dask().pipe(egh.attach_coords)
        ds = ds.rename({'level': 'plev'})
        ds['plev'] = (('plev'),
                      np.array([1, 5, 10, 20, 30, 50, 70, 100, 150, 200, 250, 300, 400, 500,
                                600, 700, 750, 800, 850, 875, 900, 925, 950, 975, 1000]))
    if 'icon' in model:
        ds = cat[model].to_dask().pipe(egh.attach_coords).rename({'pressure':'plev'})
    if 'ifs' in model:
        ds = cat[model].to_dask().pipe(egh.attach_coords).rename({'level':'plev'})
    ds['lat'] = ds['lat'].compute()
    ds['lon'] = ds['lon'].compute()
    
    # select month
    month = ds.sel(time=do_month).mean('time')

    # interpolate to lat/lon
    ua = interpolate_field_lon_lat_all_levels(month.ua)
    va = interpolate_field_lon_lat_all_levels(month.va)

    # use windspharm to get divergent (chi) and rotational (psi) wind components
    w = VectorWind(ua, va)
    uchi, vchi, upsi, vpsi = w.helmholtz(truncation=42)

    # make dataset to save
    out = xr.Dataset()
    out['u'] = ua
    out['v'] = va
    out['udiv'] = uchi
    out['vdiv'] = vchi
    out['urot'] = upsi
    out['vrot'] = vpsi
    out = out.expand_dims(dim='time')
    out['time'] = (('time'), [do_month])

    fname = f'{model}_{do_month}.nc'
    fpath = save_to + fname
    out.to_netcdf(fpath)

if __name__ == "__main__":
    for mm in months:
        if 'scream' in model:
            if mm in ['09', '10', '11', '12']:
                do_month = '2019-' + mm
            else:
                do_month = '2020-' + mm
        elif 'icon' in model or 'ifs' in model:
            if mm in ['01','02']:
                do_month = '2021-' + mm
            else:
                do_month = '2020-' + mm
        else:
            do_month = '2020-' + mm
        run_month(do_month, model)
        print('finished {}'.format(do_month))
