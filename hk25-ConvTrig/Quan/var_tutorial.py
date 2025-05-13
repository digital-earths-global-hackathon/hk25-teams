# %% [markdown]
# # Introduction
# This notebook is a simple example of how to read in the mcs_mask.

# %%
import xarray as xr
import numpy as np
import intake
import easygems.healpix as egh
import healpy as hp

import matplotlib.pyplot as plt

import sys

sys.path.append("../src")
import mcs_utils

# %% [markdown]
# ### Set some variables

# %%
MCS_TRACK_FILES = {
    "icon_ngc4008": "./../data/icon_ngc4008/mcs_tracks_final_20200101.0000_20201231.2330.nc",
    "icon_d3hp003": "",
    "scream-cess": "./../data/scream-cess/mcs_tracks_final_20190901.0000_20200901.0000.nc",
}

CATALOG = "https://digital-earths-global-hackathon.github.io/catalog/catalog.yaml"
LOCATION = "EU"  # Other possibility: 'online', but 'EU' ensures local download
PRODUCT = "icon_d3hp003"

ZOOM = 9
TIME = "PT3H"
ANALYSIS_TIME = (
    np.datetime64("2020-02-01T00:00:00"),
    np.datetime64("2020-02-15T00:00:00"),
)

# analysis-specific user seetings
TROPICAL_BELT = (-15.0, 15.0)
RADII = np.arange(1.5, 0.0, -0.1)  # Radius around trigger location in degrees

# %% [markdown]
# ### Read in simulation data

# %%
# Read in simulation data for analysis
cat = intake.open_catalog(CATALOG)[LOCATION]
simu_data = (
    cat[PRODUCT](zoom=ZOOM, time=TIME, time_method = 'inst', chunks="auto").to_dask().pipe(egh.attach_coords)
)

# Subsample simulation data to relevant time frame
data_field = simu_data.sel(time=slice(*ANALYSIS_TIME))
data_field = data_field.where(
    (data_field["lat"] > TROPICAL_BELT[0] - RADII.max())
    & (data_field["lat"] < TROPICAL_BELT[1] + RADII.max()),
    drop=True,
)

# Get the lat/lon coordinates of the healpix grid
hp_grid = data_field[["lat", "lon"]].compute()

# Get land-sea-mask
# Get land-sea-mask
ofs = data_field["sftlf"]
land_mask = ofs.where(ofs >0).compute()

ocean_mask = np.isnan(land_mask)
ocean_mask = ocean_mask.where(ocean_mask == 1)

# %% [markdown]
# ### Read in and subsample MCS tracking data

# %%
# Read in MCS tracks
mcs_tracks = xr.open_dataset(
    "/work/mh0033/m221071/works/hk25/d3hp003/hk25-mcs/PyFLEXTRKR/icon_d3hp003/mcs_tracking_hp9/stats/mcs_tracks_final_20200102.0000_20201231.2330.nc"
)
cat_tracks = mcs_tracks[
    ["start_split_cloudnumber", "start_basetime", "meanlat", "meanlon"]
]
cat_tracks = cat_tracks.compute().to_dataframe()

# %%

# Subsample relevant information
mcs_tracks = mcs_tracks[
    ["start_split_cloudnumber", "start_basetime", "meanlat", "meanlon"]
].compute()

# Subsample MCS tracks to relevant time frame
mcs_tracks = mcs_tracks.where(
    (mcs_tracks["start_basetime"] > ANALYSIS_TIME[0])
    & (mcs_tracks["start_basetime"] < ANALYSIS_TIME[1]),
    drop=True,
)

# Select all tracks that don't start as a splitter but are triggered
mcs_tracks_triggered = mcs_tracks.where(
    np.isnan(mcs_tracks["start_split_cloudnumber"]),
    drop=True,
)

# Keep only the start location of the tracks
mcs_tracks_triggered["start_lat"] = mcs_tracks_triggered["meanlat"].isel(times=0)
mcs_tracks_triggered["start_lon"] = mcs_tracks_triggered["meanlon"].isel(times=0)
mcs_trigger_locs = mcs_tracks_triggered.drop_vars(
    ["meanlat", "meanlon", "start_split_cloudnumber", "times"]
)

# Select only tropical start locations of MCSs
mcs_trigger_locs = mcs_trigger_locs.where(
    (mcs_trigger_locs["start_lat"] > TROPICAL_BELT[0])
    & (mcs_trigger_locs["start_lat"] < TROPICAL_BELT[1]),
    drop=True,
)

# Assign the healpix cell index to each trigger location
mcs_trigger_locs["trigger_idx"] = (
    "tracks",
    hp.ang2pix(
        egh.get_nside(hp_grid),
        mcs_trigger_locs["start_lon"].values,
        mcs_trigger_locs["start_lat"].values,
        nest=True,
        lonlat=True,
    ),
)

# %% [markdown]
# ### Determine triggering area of MCS and remove MCSs whose trigger region includes land

# %%
import importlib as implib

implib.reload(mcs_utils)
# Select increasingly smaller circular region around trigger location
mcs_trigger_locs = mcs_utils.add_circular_trigger_areas(
    mcs_trigger_locs, RADII, hp_grid
)

# Generate mask of MCS tracks with triggering region entirely over ocean
mcs_trigger_locs_ocean = mcs_utils.remove_land_triggers(mcs_trigger_locs, ocean_mask)

# %% [markdown]
# ### Get variables in trigger region

# %%
vars = ["prw"]
data_sample = data_field[vars].compute()
time_before_trigger = np.timedelta64(24*3, "h")
var_in_trigger_area = {
    var: mcs_utils.get_var_in_trigger_area(
        mcs_trigger_locs_ocean,
        data_sample[var],
        times_before_trigger=time_before_trigger,
        analysis_time=ANALYSIS_TIME,
    )
    for var in vars
}

# %%
mcs_trigger_locs_ocean

# %%
for var in vars:
    plt.pcolormesh(
        var_in_trigger_area[var]["radius"],
        var_in_trigger_area[var]["time"],
        var_in_trigger_area[var].mean(dim=["tracks", "cell"]).transpose(),
    )
    plt.xlabel("radius around trigger location / degree")
    plt.ylabel("timesteps before triggering / h")
    plt.colorbar(label="prw / mm")
    plt.title(f"Spatiotemporal variability of {var} before MCS triggering")

# %%
time = -1
var = "prw"
for track in var_in_trigger_area["prw"]["tracks"]:
    plt.plot(
        var_in_trigger_area[var]["radius"],
        var_in_trigger_area[var]
        .sel(tracks=track)
        .isel(time=time)
        .mean(dim="cell", skipna=True),
        lw=1,
    )
plt.plot(
    var_in_trigger_area[var]["radius"],
    var_in_trigger_area[var].mean(dim=["tracks", "cell"]).isel(time=time),
    color="black",
    linewidth=2,
    ls="--",
    label="mean",
)
plt.xlabel("radius around trigger location / degree")
plt.ylabel("prw / mm")
plt.title(f"Spatial variability of {var} at last saved timestep before MCS triggering")
plt.xlim([0.1, 1.5])
plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)

# %%
