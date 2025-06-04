# This script applies the preprocessing required for the TC tracking with TempestExtremes.

import sys
sys.path.insert(1, '../tools')

from get_data import get_data_polytope
import earthkit.regrid

def store_nc(data_latlon_sfc, data_latlon_pl,ncout, ncpath):
    """Format data and store it as netcdf suitable for TempestExtremes.
    """
    data_pl_sfc = data_latlon_sfc.to_xarray()
    data_pl_sfc["zg500"]=data_latlon_pl.to_xarray().z.sel(levelist=500)
    data_pl_sfc["zg250"]=data_latlon_pl.to_xarray().z.sel(levelist=250)

    # get rid of weird metadata. Sadly attr_drop fails continuously. 
    variables=["zg500", "zg250", '10u', '10v', 'msl']
    for var in variables:
        data_pl_sfc[var].attrs['_earthkit']="none"

    # Renaming to avoid errors due to format in nc file
    data_pl_sfc = data_pl_sfc.rename({"forecast_reference_time":"time"})
    data_pl_sfc =  data_pl_sfc.rename({"10u":"uas"})
    data_pl_sfc =  data_pl_sfc.rename({"10v":"vas"})

    print(ncpath+ncout)
    data_pl_sfc.to_netcdf(ncpath+ncout)

def get_data_as_nc(date, ncout, ncpath):
    """Get the data from the data-bridge, apply prerpocessing and store as netcdf.
    """
    
    request_basic={
    "activity": ["scenariomip"],
    "class": "d1",
    "dataset": "climate-dt",
    "date": date,
    "experiment": "ssp3-7.0",
    "expver": "0001",
    "generation": "1",
    "model": "icon",
    "realization": "1",
    "stream": "clte",
    "resolution": "high",
    "type": "fc",
    "time": "0000/to/2300/by/0300"
    }

    request_pl=request_basic | {
        "levtype": ["pl"],
        "param": ["129"],
        "levelist": ["500", "250"],
    }

    request_sfc=request_basic | {
        "levtype": ["sfc"],
        "param": ["166", "165", "151"],
    }

    # remapping to 0.2 degree lonlat grid
    data_pl = get_data_polytope(request_pl)
    data_latlon_pl = earthkit.regrid.interpolate(data_pl, out_grid={"grid": [0.2,0.2]}, method="linear")

    data_sfc = get_data_polytope(request_sfc)
    data_latlon_sfc = earthkit.regrid.interpolate(data_sfc, out_grid={"grid": [0.2,0.2]}, method="linear")

    ncout = "tmp_icon_"+str(date)+".nc"
    store_nc(data_latlon_sfc, data_latlon_pl, ncout, ncpath)
