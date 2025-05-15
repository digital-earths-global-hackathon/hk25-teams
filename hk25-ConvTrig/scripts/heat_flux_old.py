# %%
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
import stats_utils
import intake
import slurm_cluster as scluster


# %%
import importlib as implib

implib.reload(mcs_utils)
implib.reload(stats_utils)

# %%
# %%
client, cluster = scluster.init_dask_slurm_cluster(
    scale = 5, processes = 20, walltime="07:00:00", memory="200GB", dash_address=8282
)


# %%
MCS_TRACK_FILES = {
    "icon_ngc4008": "./../data/icon_ngc4008/mcs_tracks_final_20200101.0000_20201231.2330.nc",
    "icon_d3hp003": "",
    "scream-cess": "./../data/scream-cess/mcs_tracks_final_20190901.0000_20200901.0000.nc",
    }

CATALOG = "https://digital-earths-global-hackathon.github.io/catalog/catalog.yaml"
LOCATION = "EU"     #Other possibility: 'online', but 'EU' ensures local download
PRODUCT = "icon_ngc4008"

ZOOM = 9
TIME = "PT3H"
ANALYSIS_TIME = (
    np.datetime64("2020-02-01T00:00:00"), np.datetime64("2021-01-01T00:00:00")
    )

# analysis-specific user seetings
TROPICAL_BELT = (-15., 15.)
RADII = np.arange(1.5, 0., -0.1)    # Radius around trigger location in degrees
# %%
# Read in simulation data for analysis
cat = intake.open_catalog(CATALOG)[LOCATION]
simu_data = cat[PRODUCT](zoom=ZOOM, time=TIME, chunks="auto").to_dask().\
    pipe(egh.attach_coords)

# Subsample simulation data to relevant time frame
data_field = simu_data
data_field = data_field.where(
    (data_field['lat'] > TROPICAL_BELT[0]-RADII.max()) &
    (data_field['lat'] < TROPICAL_BELT[1]+RADII.max()), drop=True,
    )

data_field = data_field.sel(time=slice(*ANALYSIS_TIME))


# %%
# Get the lat/lon coordinates of the healpix grid
hp_grid = data_field[['lat', 'lon']].compute()

# Get land-sea-mask
ofs = data_field['ocean_fraction_surface']
ocean_mask = ofs.where(ofs==1).compute()
# %%
MCS_TRACK_FILES = {
    "icon_ngc4008": \
        "/work/mh0731/m300738/project_hk25-ConvTrig/data/icon_ngc4008/mcs_tracks_final_20200101.0000_20201231.2330.nc",
    "icon_d3hp003": \
        "./../data/icon_d3hp003/mcs_tracks_final_20200102.0000_20201231.2330.nc",
    "scream-dkrz": \
        "./../data/scream-dkrz/mcs_tracks_final_20190901.0000_20200901.0000.nc",
    "um_glm_n2560_RAL3p3": \
        "./../data/um_glm_n2560_RAL3p3/mcs_tracks_final_20200201.0000_20210301.0000.nc",
    }

# Read in MCS tracks
mcs_tracks = xr.open_dataset(MCS_TRACK_FILES[PRODUCT], chunks={})

# Subsample relevant information
mcs_tracks = mcs_tracks[
    ['start_split_cloudnumber', 'start_basetime', 'meanlat', 'meanlon']
    ].compute()

# Subsample MCS tracks to relevant time frame
mcs_tracks = mcs_tracks.where(
    (mcs_tracks['start_basetime'] > ANALYSIS_TIME[0]) &
    (mcs_tracks['start_basetime'] < ANALYSIS_TIME[1]),
    drop=True,
    )

# Select all tracks that don't start as a splitter but are triggered
mcs_tracks_triggered = mcs_tracks.where(
    np.isnan(mcs_tracks["start_split_cloudnumber"]), drop=True,
    )

# Keep only the start location of the tracks
mcs_tracks_triggered['start_lat'] = mcs_tracks_triggered['meanlat'].isel(times=0)
mcs_tracks_triggered['start_lon'] = mcs_tracks_triggered['meanlon'].isel(times=0)
mcs_trigger_locs = mcs_tracks_triggered.drop_vars(
    ['meanlat', 'meanlon', 'start_split_cloudnumber', 'times']
    )

# Select only tropical start locations of MCSs
mcs_trigger_locs = mcs_trigger_locs.where(
    (mcs_trigger_locs['start_lat'] > TROPICAL_BELT[0]) &
    (mcs_trigger_locs['start_lat'] < TROPICAL_BELT[1]),
    drop=True,
    )

# Assign the healpix cell index to each trigger location
mcs_trigger_locs['trigger_idx'] = (
    'tracks',
    hp.ang2pix(
        egh.get_nside(hp_grid),
        mcs_trigger_locs['start_lon'].values,
        mcs_trigger_locs['start_lat'].values,
        nest=True, lonlat=True,
        ),
)


# %%
# %% [markdown]
# ### Determine triggering area of MCS and remove MCSs whose trigger region includes land

# %%

# Select increasingly smaller circular region around trigger location
mcs_trigger_locs = mcs_utils.add_circular_trigger_areas(
    mcs_trigger_locs, RADII, hp_grid
)

# Generate mask of MCS tracks with triggering region entirely over ocean
mcs_trigger_locs_ocean = mcs_utils.remove_land_triggers(mcs_trigger_locs, ocean_mask)
# %%

# calculate the anomalies
vars = ["hfls", "hfss", "prw"]# latent heat, sensible heat, precipitation

# data_ano= stats_utils.remove_daily_mean(data_field, var)
data_ano = data_field[vars]
# data_ano = data_ano.sel(time=slice(*ANALYSIS_TIME))

# data_ano = data_ano.compute()


# %%
time_before_trigger = np.timedelta64(24, "h")

# %%
# original data without anomaly
# var_in_trigger_area_origin = {
#     var: mcs_utils.get_var_in_trigger_area(
#         mcs_trigger_locs_ocean,
#         data_sample[var],
#         times_before_trigger=time_before_trigger,
#         analysis_time=ANALYSIS_TIME,
#     )
#     for var in vars
# }

# %%


var_in_trigger_area_ano = {
    var: mcs_utils.get_var_in_trigger_area(
        mcs_trigger_locs_ocean,
        data_ano[var],
        times_before_trigger=time_before_trigger,
        analysis_time=ANALYSIS_TIME,
    )
    for var in vars
}
#%%
# dic to xarray
var_in_trigger_area_ano = xr.Dataset(var_in_trigger_area_ano)

#%%
# save
var_in_trigger_area_ano.to_netcdf("/work/mh0033/m300883/hk25_data/PT3H_var_before_oldICON.nc")


# %%
var = "hflsd" # latent heat flux


var_in_trigger_area_ano_plot= var_in_trigger_area_ano[var]
# show only plot for the anomaly, with the profile and contourf
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(2, 4, width_ratios=[2, 0.7, 2, 0.7], height_ratios=[2, 0.5])
# Anomaly data contourf
ax2 = fig.add_subplot(gs[0, 2])
cf1 = (
    var_in_trigger_area_ano_plot  # # convert to mm/day
    .mean(dim=["tracks", "cell"])
    .T.plot.contourf(
        ax=ax2,
        add_colorbar=False,
        levels=20,
    )
)

ax2.set_title("Anomalies precipitation before MCS triggering")
ax2.set_ylabel("timesteps before triggering / h")
# Profile for x=0.4 (radius=0.4) for anomalies
x_idx = 0.4
ax3 = fig.add_subplot(gs[0, 3], sharey=ax2)
profile1 = (
    var_in_trigger_area_ano_plot
    .mean(dim=["tracks", "cell"])
    .sel(radius=x_idx, method="nearest")
)
profile1.plot(ax=ax3, y="time", marker="o")
ax3.set_title("Profile at radius=0.4")
ax3.set_xlabel(var)
ax3.set_ylabel("time relative to triggering / h")
ax3.invert_yaxis()

for ax in [ax2, ax3]:
    ax.set_ylim(-1, -8)


# Line plot below ax2: mean along time (anomaly)
ax5 = fig.add_subplot(gs[1, 2], sharex=ax2)
mean_time_ano = var_in_trigger_area_ano_plot.mean(dim=["tracks", "cell", "time"])
ax5.plot(var_in_trigger_area_ano_plot["radius"], mean_time_ano)
ax5.set_xlabel("radius from trigger location / degree")
ax5.set_ylabel(rf"Mean {var} / W m-2")
ax5.set_title("Mean along time (Anomaly)")
# Empty below ax3 for alignment
fig.add_subplot(gs[1, 3]).axis("off")

# add colorbar below the ax3
cbar_ax = fig.add_axes([0.55, -0.05, 0.2, 0.03])
cbar = fig.colorbar(cf1, cax=cbar_ax, orientation="horizontal")
cbar.set_label(f"{var} / W m-2")

fig.tight_layout()
plt.savefig("/work/mh0033/m300883/hk25_plots/latent_ano_compoiste.png", dpi=300)

# %%
# same plot for the sensible heat flux
var = "hfssd" # sensible heat flux
var_in_trigger_area_ano_plot= var_in_trigger_area_ano[var]
# show only plot for the anomaly, with the profile and contourf
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(2, 4, width_ratios=[2, 0.7, 2, 0.7], height_ratios=[2, 0.5])
# Anomaly data contourf
ax2 = fig.add_subplot(gs[0, 2])
cf1 = (
    var_in_trigger_area_ano_plot  # # convert to mm/day
    .mean(dim=["tracks", "cell"])
    .T.plot.contourf(
        ax=ax2,
        add_colorbar=False,
        levels=20,
    )
)
ax2.set_title("Anomalies precipitation before MCS triggering")
ax2.set_ylabel("timesteps before triggering / h")
# Profile for x=0.4 (radius=0.4) for anomalies
x_idx = 0.4
ax3 = fig.add_subplot(gs[0, 3], sharey=ax2)
profile1 = (
    var_in_trigger_area_ano_plot
    .mean(dim=["tracks", "cell"])
    .sel(radius=x_idx, method="nearest")
)
profile1.plot(ax=ax3, y="time", marker="o")
ax3.set_title("Profile at radius=0.4")
ax3.set_xlabel(var)
ax3.set_ylabel("time relative to triggering / h")
ax3.invert_yaxis()
for ax in [ax2, ax3]:
    ax.set_ylim(-1, -8)
# Line plot below ax2: mean along time (anomaly)
ax5 = fig.add_subplot(gs[1, 2], sharex=ax2)
mean_time_ano = var_in_trigger_area_ano_plot.mean(dim=["tracks", "cell", "time"])
ax5.plot(var_in_trigger_area_ano_plot["radius"], mean_time_ano)
ax5.set_xlabel("radius from trigger location / degree")
ax5.set_ylabel(rf"Mean {var} / W m-2")
ax5.set_title("Mean along time (Anomaly)")
# Empty below ax3 for alignment
fig.add_subplot(gs[1, 3]).axis("off")   
# add colorbar below the ax3
cbar_ax = fig.add_axes([0.55, -0.05, 0.2, 0.03])
cbar = fig.colorbar(cf1, cax=cbar_ax, orientation="horizontal")
cbar.set_label(f"{var} / W m-2")
fig.tight_layout()
plt.savefig("/work/mh0033/m300883/hk25_plots/sensible_ano_compoiste.png", dpi=300)
# %%
