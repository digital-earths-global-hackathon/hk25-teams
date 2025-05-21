#!/usr/bin/env python
# coding: utf-8

# # Download variables for Snapshot retrieval with TempestExtremes
# By Stella Bourdin


import os, intake, datetime
import xarray as xr
import numpy as np
from tqdm import tqdm


# Load catalogue
cat = intake.open_catalog('https://digital-earths-global-hackathon.github.io/catalog/catalog.yaml')['UK']

# TODO: Define the run you want to track, run-specific info + working directories
run='um_glm_n1280_GAL9'
# TODO: run-specific info
time_name ='PT1H' # Select the time name for the dataset containing the variables you are looking for

# TODO: Define working directories
scr_dir = '/work/scratch-nopw2/sbourdin/'
run_dir = os.path.join(scr_dir,run)
if not os.path.isdir(run_dir):
    os.makedirs(run_dir)
output_dir = os.path.join(run_dir,'data_healpix/')
if not os.path.isdir(output_dir):
    os.makedirs(output_dir)

# TODO: Script parameters: Select zoom level, variables
zoom = 9 # Select zoom level closest to the model's actual resolution and/or highest zoom available
variables = ["uas", "vas"] 

for var in variables:

    # Load data
    ds = cat[run](zoom=zoom, time=time_name).to_dask()[var]
    
    # Select 6-hourly
    ds = ds.isel(time=(ds.time.dt.hour % 6 == 0))

    # Delete spurious attributes if present
    if "bounds" in list(ds.attrs.keys()):
        del ds.attrs["bounds"]
    if "regional" in list(ds.attrs.keys()):
        del ds.attrs["regional"]

    # Save one file per month
    mth_list = np.unique(ds.time.astype(str).str.slice(0,7))
    for mth in tqdm(mth_list):
        fname = output_dir+mth+"_"+var+"_zoom_"+str(zoom)+".nc"
        if not os.path.exists(fname):
            ds.sel(time = mth).to_netcdf(fname)
        else:
            print(mth, "File already exists")
