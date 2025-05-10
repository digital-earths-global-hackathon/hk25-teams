import healpy as hp  
from scipy.interpolate import NearestNDInterpolator 
import numpy as np
import xarray as xr


def ocean(ds):
    """
    Returns a mask for ocean grid cells.
    
    Parameters:
        ds (xarray.Dataset): Dataset containing 'ocean_fraction_surface'
    
    Returns:
        xarray.DataArray: Boolean mask where ocean_fraction_surface == 1
    """
    return ds.ocean_fraction_surface == 1


def tropics(ds, lat_min=-40, lat_max=40):
    """
    Returns a mask for a latitude band, by default covering the tropics 
    (between -40° and 40° latitude).

    Parameters:
        ds (xarray.Dataset): Dataset with a 'lat' coordinate
        lat_min (float): Lower latitude boundary (default: -40)
        lat_max (float): Upper latitude boundary (default: 40)

    Returns:
        xarray.DataArray: Boolean mask where latitudes are within [lat_min, lat_max]
    """
    return (ds.lat > lat_min) & (ds.lat < lat_max)


def attach_coords(ds, nside, nest_tf):
    """
    Adds latitude and longitude coordinates to a dataset using Healpix indexing.
    
    Parameters:
        ds (xarray.Dataset): Dataset with 'cell' dimension containing Healpix indices
        nside (int): Healpix resolution
        nest_tf (bool): Whether Healpix indexing is nested

    Returns:
        xarray.Dataset: Dataset with added 'lat' and 'lon' coordinates
    """
    lons, lats = hp.pix2ang(nside, ds.cell.values, nest=nest_tf, lonlat=True)
    return ds.assign_coords(
        lat=(("cell",), lats, {"units": "degrees_north"}),
        lon=(("cell",), lons, {"units": "degrees_east"}),
    )


def interpolate_field_lon_lat(field, lon_coord="lon", lat_coord="lat", relative_resolution=2):
    """
    Interpolates a 1D spatial field to a regular 2D lon-lat grid using nearest-neighbor.

    Parameters:
        field (xarray.DataArray): Field with coordinates (lon, lat)
        lon_coord (str): Name of the longitude coordinate
        lat_coord (str): Name of the latitude coordinate
        relative_resolution (float): Controls output grid resolution (higher = finer)

    Returns:
        xarray.DataArray: Interpolated 2D field on regular lon-lat grid
    """
    nlon = nlat = int(np.sqrt(len(field) * relative_resolution))

    lon_points = field[lon_coord].values
    lat_points = field[lat_coord].values

    lon = np.linspace(np.min(lon_points), np.max(lon_points), nlon)
    lat = np.linspace(np.min(lat_points), np.max(lat_points), nlat)
    lon2, lat2 = np.meshgrid(lon, lat)

    points = np.stack((lon_points, lat_points), axis=1)

    interpolated = NearestNDInterpolator(points, field.values)(lon2, lat2)

    return xr.DataArray(
        interpolated,
        dims=["lat", "lon"],
        coords={"lon": lon, "lat": lat},
    )
