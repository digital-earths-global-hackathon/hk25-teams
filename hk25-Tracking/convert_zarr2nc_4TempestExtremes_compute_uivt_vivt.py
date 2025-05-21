#!/usr/bin/env python
# coding: utf-8

# # Convert HEALPix Zarr data to netCDF with variable subsets for TempestExtremes
# 
# ## This code compute the uivt and vivt based on ua, va, and hus
# 
# ## Author:
# - Zhe Feng || zhe.feng@pnnl.gov

# In[1]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import xarray as xr
import os

import easygems.healpix as egh
import intake

plt.rcParams['figure.dpi'] = 72


# ### Load the catalog

# In[2]:


list(intake.open_catalog("https://digital-earths-global-hackathon.github.io/catalog/catalog.yaml"))


# In[3]:


# Load the NERSC catalog
current_location = "NERSC" # "online" # 
cat = intake.open_catalog("https://digital-earths-global-hackathon.github.io/catalog/catalog.yaml")[current_location]
list(cat)


# ### Pick a Dataset

# In[4]:

s_Model = "nicam_gl11" # "icon_d3hp003" # "um_glm_n2560_RAL3p3" # "casesm2_10km_nocumulus" # "icon_ngc4008" # 
s_TimeRes = "PT6H"
#
zoom = 8
pd.DataFrame(cat[s_Model].describe()["user_parameters"])


# ### Load Data into a Data Set
# most datasets have a `zoom` parameter. We will use `zoom` level 8 [(~24km)](https://easy.gems.dkrz.de/Processing/healpix/index.html#healpix-spatial-resolution)

# In[5]:

ds0  = cat[s_Model](zoom=zoom, time=s_TimeRes).to_dask()
ds0  = ds0.pipe(egh.attach_coords)
ds0


# In[ ]:


# Variables to output
### RawVarName: output_name
varout_dict       = { 'time' : 'time',
                      'lat' : 'lat',
                      'lon' : 'lon',
                      'pressure': 'lev',
                      'ua': 'ua',
                      'va': 'va',
                      'hus': 'hus',
                      'orog': 'ELEV',
                      'pr' : 'pr',
                      'prs': 'prs',
                      'ps' : 'ps',
                      'psl' : 'psl',
                      'uas' : 'uas',
                      'vas' : 'vas',
                      'sfcWind' : 'sfcWind',
                      'zg' : 'zg',
                      'ta' : 'ta',
                      'tas': 'tas',
                      'rlut': 'rlut',
                     }
print(ds0.data_vars)


# In[22]:


# Subset variables and rename
varout = []
varout_RawVarName = []
ds     = {}
for var in varout_dict:
    if var in ds0.data_vars: #  or var in ds0.coords
        ds[varout_dict[var]] = ds0[var]
        #
    else:
        print(f"No {var}: {varout_dict[var]}")


ds    = xr.Dataset(ds)
print(ds)
ds.attrs = ds0.attrs
for var in ds0.coords:
    if var in varout_dict and var != varout_dict[var]:
        ds    = ds.rename_dims({var: varout_dict[var]})
        ds    = ds.rename_vars({var: varout_dict[var]})

print(ds.data_vars)

## === Ziming Chen 05/11/12 ===
ds['time'] = xr.decode_cf(ds).indexes['time'] #.to_datetimeindex()
ds

# %%
ds.data_vars

# In[ ]:


# Compute the uivt and vivt

# Python function for vertical mass integration using xarray and numpy
import numpy as np
import xarray as xr

def vertical_mass_integration(hus: xr.DataArray, ps: xr.DataArray, plev: xr.DataArray) -> xr.DataArray:
    """
    Perform vertical integration of specific humidity (hus) in pressure coordinates.

    Parameters:
    - hus: xr.DataArray with dimensions (time, lev, cell)
    - ps: xr.DataArray with dimensions (time, cell), surface pressure in hPa
    - plev: xr.DataArray with dimension (lev), pressure levels in hPa

    Returns:
    - xr.DataArray with dimensions (time, cell) representing vertically integrated hus
    """
    # Ensure pressure levels are sorted from top (min) to bottom (max)
    if not np.all(np.diff(plev.values) > 0):
        hus = hus.sel(lev=plev[::-1])
        plev = plev[::-1]
    #
    if ps.max() < 1200: # hPa to Pa for ps
        ps                    = ps * 100
        ps.name               = "Pa"
        ps.attrs["units"]     = "Pa"
        ps.attrs["long_name"] = "Estimated surface pressure"
    # Mask hus where pressure level > surface pressure
    plev_3d = plev * xr.ones_like(hus)
    ps_3d = ps * xr.ones_like(hus)
    hus_masked = hus.where(plev_3d <= ps_3d)

    # Integrate using trapezoidal rule in pressure coordinates (in Pa)
    dp = np.gradient(plev.values) * 100.0  # convert hPa to Pa
    dp = xr.DataArray(dp, coords={"lev": plev}, dims=["lev"])
    dp_3d = dp * xr.ones_like(hus)

    g = 9.8  # gravity
    integrand = hus_masked * dp_3d / g
    result = integrand.sum(dim="lev")

    return result

# In[ ]:
out_dir = f'/pscratch/sd/w/wcmca1/scream-cess-healpix/data4TE/{s_Model}_{s_TimeRes}/'

# Optional: create output directory
os.makedirs(out_dir, exist_ok=True)

# Clean up attributes before writing to NetCDF
def clean_attrs_for_netcdf(ds):
    # Make a copy to avoid modifying the original
    attrs = ds.attrs.copy()
    
    # Handle dictionary and boolean attributes
    for key, value in list(attrs.items()):
        if isinstance(value, dict):
            # Remove dictionary attributes
            ds.attrs.pop(key, None)
        elif isinstance(value, bool):
            # Convert boolean to integer (1 for True, 0 for False)
            ds.attrs[key] = int(value)
    
    # Check all variables too
    if hasattr(ds, 'variables'):
        for var in ds.variables:
            var_attrs = ds[var].attrs.copy()
            for key, value in list(var_attrs.items()):
                if isinstance(value, dict):
                    import json
                    ds[var].attrs[key] = json.dumps(value)
                elif isinstance(value, bool):
                    # Convert boolean to integer
                    ds[var].attrs[key] = int(value)
    
    return ds

original_history = ds.attrs.get('history', '')
original_title   = ds.attrs.get('title', '')
timestamp        = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
new_history      = f"{timestamp}: Data processed for TempestExtremes - Tracking Variables Preparation"
# Group by month and write each to a separate file
for month, ds_month in ds.groupby('time.month'):
    for year, ds_month_year in ds_month.groupby('time.year'):
        #
        year     = ds_month_year.coords['time'].dt.year
        year     = year[0].values
        #
        if year == 2020 or (year == 2021 and month < 1):
            # Skip 2020 for now
            print(f"Skipping {year}-{str(month).zfill(2)} for now")
            continue
        #
        if "sfcWind" not in ds_month_year.data_vars and \
           "uas" in ds_month_year.data_vars and "vas" in ds_month_year.data_vars:
            ds_month_year["sfcWind"]  = np.sqrt(ds_month_year["uas"]**2 + ds_month_year["vas"]**2)
            ds_month_year["sfcWind"].attrs["long_name"] = "surface Wind Speed"
            ds_month_year["sfcWind"].attrs["units"]     = "m s-1"

        ## compute uivt and vivt
        if "hus" in ds_month_year.data_vars and \
           "ua" in ds_month_year.data_vars and \
            "va" in ds_month_year.data_vars:
            lev        = ds["lev"]
            if "ps" not in ds_month_year.data_vars:
                mask_valid = ~np.isnan(ds_month_year["hus"])
                # Use argmax to find the *first valid level from top* (lev should be sorted from high to low)
                first_valid_idx = mask_valid.argmax(dim="lev")

                # Convert the index to the actual pres value
                ps         = lev[first_valid_idx]
                #
            else:
                ps         = ds["ps"]
            #
            ## compute day by day
            ds_uivt_tmp= vertical_mass_integration(
                        ds_month_year["hus"]*ds_month_year["ua"], ps, ds_month_year['lev'] )

            ds_vivt_tmp= vertical_mass_integration(
                        ds_month_year["hus"]*ds_month_year["va"], ps, ds_month_year['lev'] )
            # print(ds_uivt_tmp)

            ds_month_year["uivt"]     = ds_uivt_tmp 
            ds_month_year["vivt"]     = ds_vivt_tmp 
            ds_month_year["uivt"].attrs["long_name"] = "ZonalVapFlux"
            ds_month_year["vivt"].attrs["long_name"] = "MeridionalVapFlux"
            ds_month_year["uivt"].attrs["units"]  = "m^-1 s^-1 kg"
            ds_month_year["vivt"].attrs["units"]  = "m^-1 s^-1 kg"
        #
        # Format output filename
        ## Check if file exists and append a number if it does
        ds_month_year.attrs       = ds.attrs
        out_file = os.path.join(out_dir, f"{s_Model}_ivt_hp{zoom}_{s_TimeRes}.{str(year)}{str(month).zfill(2)}.nc")
        
        if os.path.exists(out_file):
            # Load existing file to check for duplicates
            existing_ds = xr.open_dataset(out_file)
            
            # Check for new variables
            existing_vars = set(existing_ds.data_vars)
            new_vars = set(ds_month_year.data_vars)
            new_variables = new_vars - existing_vars
            has_new_vars = len(new_variables) > 0
            
            if has_new_vars:
                #
                # Just adding new variables to existing times
                # combined_ds = existing_ds.copy()
                existing_ds.close()
                for var in new_variables:
                    print(f"Adding new variable: {var}")
                    new_vars_only = ds_month_year[var]
                    new_vars_only = clean_attrs_for_netcdf(new_vars_only)
                    #
                    # Add back history with timestamp of this processing
                    if original_history:
                        new_vars_only.attrs['history'] = f"{new_history}; {original_history}"
                    else:
                        new_vars_only.attrs['history'] = new_history
                    # Add additional metadata
                    new_vars_only.attrs['source_model'] = s_Model
                    new_vars_only.attrs['time_resolution'] = s_TimeRes
                    new_vars_only.attrs['healpix_zoom'] = zoom
                    new_vars_only.attrs['processing_script'] = "convert_zarr2nc_4TempestExtremes_compute_uivt_vivt.py"
                    #
                    new_vars_only.to_netcdf(out_file, mode='a', engine='h5netcdf')
                
                # Close to free up resources
            else:
                print(f"No new data or variables to add to {out_file}")
                existing_ds.close()
        else:
            # File doesn't exist, create a new one
            ds_month_year = clean_attrs_for_netcdf(ds_month_year)
            #
            # Add back history with timestamp of this processing
            if original_history:
                new_vars_only.attrs['history'] = f"{new_history}; {original_history}"
            else:
                new_vars_only.attrs['history'] = new_history
            # Add additional metadata
            new_vars_only.attrs['source_model'] = s_Model
            new_vars_only.attrs['time_resolution'] = s_TimeRes
            new_vars_only.attrs['healpix_zoom'] = zoom
            new_vars_only.attrs['processing_script'] = "convert_zarr2nc_4TempestExtremes_compute_uivt_vivt.py"
            #
            ds_month_year.to_netcdf(out_file)
            print(f"Created new file: {out_file}")

