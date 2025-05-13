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

def nest2ring_index(ds, nside):
    """
        ds (xarray:Dataset): dataset with cell as dimensions
        nside (int): nside of the zoom level 
    """
    return np.array([hp.ring2nest(nside, i) for i in ds.cell.values])

def compute_hder(var, nside):
        """
        computes the horizontal derivatives of any variable (1 vertical level, 1 time) using spherical harmonics
        
        Parameters:
            var (xarray.DataArray): Name of the variable on which derivatives will be computed
            nside (int): nside of the zoom level 

        Returns:
            numpy array: derivative with respect to co-latitude
            numpy array: derivative with respect to longitude
        
        """
        var_alm = hp.sphtfunc.map2alm(var)
        der_arr = hp.sphtfunc.alm2map_der1(var_alm, nside)
        return der_arr[1, :], der_arr[2, :] # dvar_dtheta (lat), dvar_dphi (lon)


def compute_conv(ua, va, ring_index, nside):
        """
        computes the horizontal wind convergence using spherical harmonics
        
        Parameters:
            ua (xarray.DataArray): zonal wind
            va (xarray.DataArray): meridional wind
            ring_index (numpy array): indices to convert from nest to ring
            nside (int): nside of the zoom level 

        Returns:
            convergence (xarray.DataArray)
        
        """
        ua = ua.isel(cell = ring_index)
        va = va.isel(cell = ring_index)
        lat = ua.lat
    
        def _compute_conv(ua, va, lat, nside):
            _, dua_dphi = compute_hder(ua, nside)
            dva_dtheta, _  = compute_hder(va, nside)
            va_tanlat = va * np.tan(np.deg2rad(lat))
            return -(dua_dphi - dva_dtheta - va_tanlat) / 6371/1000 #+ 2*7.2921e-5 *np.sin(np.deg2rad(lat))
    
        conv_time = xr.apply_ufunc(_compute_conv,
                            ua, va, lat, nside,
                            input_core_dims=[['cell'],['cell'],['cell'],[]],
                            dask = "parallelized",
                            vectorize = True,
                            output_core_dims= [['cell']],
                            dask_gufunc_kwargs = {"output_sizes": {"cell": len(ua.cell)}},
                            output_dtypes = ["f8"],)
        return conv_time
