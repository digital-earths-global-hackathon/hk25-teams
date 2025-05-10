#%%
import pathlib
import healpy as hp
import numpy as np
import xarray as xr
import pandas as pd
import intake
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

import multiprocessing
import dask
from dask.distributed import Client, progress, LocalCluster

import easygems.healpix as egh
from matplotlib.patches import Rectangle

#%%
## Functions
# Load Catalog
cat = intake.open_catalog(
    "https://data.nextgems-h2020.eu/catalog.yaml"
)
cat_hera = intake.open_catalog("https://tcodata.mpimet.mpg.de/internal.yaml")
print(f"Experiments in Catalog: {list(cat.keys())}")

#%%
# Healpix
# Example data
zoom_use = 8

t1 = "2020-03-01"
t2 = "2020-03-30"

t1_AMIP = "1979-03-01"
t2_AMIP = "1979-03-30"

test_data_ngc4008 = (
    cat["ICON"]["ngc4008"](time="P1D", zoom=zoom_use, chunks="auto")
    .to_dask()
    .sel(time=slice(t1, t2))
    .mean(dim="time")
)
test_data_era5 = (
    cat_hera["HERA5"](time="P1D")
    .to_dask()
    .sel(time=slice(t1, t2))
    .mean(dim="time")
)
crs_for_ERA5 = (
    cat["ICON"]["ngc4008"](time="P1D", zoom=7, chunks="auto")
    .to_dask()
)

#%%
## Get Lon-Lat Coordinates
def attach_coords(ds, nside, nest_tf):
    lons, lats = hp.pix2ang(nside, ds.cell.values, nest=nest_tf, lonlat=True)
    return ds.assign_coords(
        lat=(("cell",), lats, {"units": "degrees_north"}),
        lon=(("cell",), lons, {"units": "degrees_east"}),
    )
ngc4008 = attach_coords(test_data_ngc4008, test_data_ngc4008.crs.healpix_nside, True)
era5 = test_data_era5.rename({"latitude": "lat", "longitude": "lon"})
crs_for_ERA5 = attach_coords(crs_for_ERA5, crs_for_ERA5.crs.healpix_nside, True)
era5_q_MAR = xr.open_dataset("/work/mh0731/m300868/00_DATA/ERA5/133/E5pl00_1D_1979-03_133.nc").rename({'cells':'cell'}).mean(dim="time")
era5_T_MAR = xr.open_dataset("/work/mh0731/m300868/00_DATA/ERA5/130/E5pl00_1D_1979-03_130.nc").rename({'cells':'cell'}).mean(dim="time")
era5_geoHeight_MAR = xr.open_dataset("/work/mh0731/m300868/00_DATA/ERA5/129/E5pl00_1D_1979-03_129.nc").rename({'cells':'cell'}).mean(dim="time")
era5_w_MAR = xr.open_dataset("/work/mh0731/m300868/00_DATA/ERA5/135/E5pl00_1D_1979-03_135.nc").rename({'cells':'cell'}).mean(dim="time")
era5_u_MAR = xr.open_dataset("/work/mh0731/m300868/00_DATA/ERA5/131/E5pl00_1D_1979-03_131.nc").rename({'cells':'cell'}).mean(dim="time")
era5_v_MAR = xr.open_dataset("/work/mh0731/m300868/00_DATA/ERA5/132/E5pl00_1D_1979-03_132.nc").rename({'cells':'cell'}).mean(dim="time")
era5_cli_MAR = (
    xr.open_dataset("/work/mh0731/m300868/00_DATA/ERA5/201033/E5pl00_1D_1979-JAN_FEB_MAR_201033_plev.nc")
    .ciwc
    .sel(time=slice('1979-03-01','1979-03-31'))
    .mean(dim='time')
    .rename({'cells':'cell'})
    .rename({'level':'plev'})
) #ciwc in ERA5 langugage
era5_cli_MAR["plev"] = era5_cli_MAR.plev * 100
era5_cli_MAR
era5_clw_MAR = (
    xr.open_dataset("/work/mh0731/m300868/00_DATA/ERA5/201031/E5pl00_1D_1979-JAN_FEB_MAR_201031_plev.nc")
    .clwc
    .sel(time=slice('1979-03-01','1979-03-31'))
    .mean(dim='time')
    .rename({'cells':'cell'})
    .rename({'level':'plev'})
) #clwc in ERA5 langugage
era5_clw_MAR["plev"] = era5_clw_MAR.plev * 100
era5_clw_MAR


#%%
# Native Grid
import glob
grid_atm = xr.open_dataset(
    "/pool/data/ICON/grids/public/mpim/0025/icon_grid_0025_R02B08_G.nc"
).rename({"clon": "lon", "clat": "lat"})
grid_oce = xr.open_dataset(
    "/pool/data/ICON/grids/public/mpim/0026/icon_grid_0026_R02B08_O.nc"
).rename({"clon": "lon", "clat": "lat"})

grid_atm["lat"] = np.rad2deg(grid_atm.lat)
grid_atm["lon"] = np.rad2deg(grid_atm.lon)
grid_atm["lon"] = (grid_atm["lon"] + 360) % 360
## ngcMBE2922
import re
import glob
import os
grid_atm = xr.open_dataset(
    "/pool/data/ICON/grids/public/mpim/0025/icon_grid_0025_R02B08_G.nc"
).rename({"clon": "lon", "clat": "lat"})
grid_oce = xr.open_dataset(
    "/pool/data/ICON/grids/public/mpim/0026/icon_grid_0026_R02B08_O.nc"
).rename({"clon": "lon", "clat": "lat"})

grid_atm["lat"] = np.rad2deg(grid_atm.lat)
grid_atm["lon"] = np.rad2deg(grid_atm.lon)
grid_atm["lon"] = (grid_atm["lon"] + 360) % 360
directory = "/work/mh0287/m214002/GIT/icon-mpim-master_dbf6dad5/experiments/mbe2922"
all_files = glob.glob(f"{directory}/*")

pattern_MAR = re.compile(r'mbe2922_atm_3d_P1D_4_\d{4}03(0[1-9]|[12][0-9]|3[01])T.*\.nc')

files_MAR = sorted([file for file in all_files if pattern_MAR.match(os.path.basename(file))])
ngcMBE2922_MAR = (xr.open_mfdataset(files_MAR, combine="by_coords")
                    .rename({"ncells": "cell"})
                    .sel(time=slice('1979-03-01','1979-03-31'))
                    .mean(dim="time")
                   )
directory = "/work/mh0287/m214002/GIT/icon-mpim-master_dbf6dad5/experiments/mbe2922"
all_files = glob.glob(f"{directory}/*")

pattern_MAR = re.compile(r'mbe2922_atm_3d_P1D_2_\d{4}03(0[1-9]|[12][0-9]|3[01])T.*\.nc')

files_MAR = sorted([file for file in all_files if pattern_MAR.match(os.path.basename(file))])
ngcMBE2922_2_MAR = (xr.open_mfdataset(files_MAR, combine="by_coords")
                    .rename({"ncells": "cell"})
                    .sel(time=slice('1979-03-01','1979-03-31'))
                    .mean(dim="time")
                   )
directory = "/work/mh0287/m214002/GIT/icon-mpim-master_dbf6dad5/experiments/mbe2922"
all_files = glob.glob(f"{directory}/*")

pattern_MAR = re.compile(r'mbe2922_atm_3d_P1D_1_\d{4}03(0[1-9]|[12][0-9]|3[01])T.*\.nc')

files_MAR = sorted([file for file in all_files if pattern_MAR.match(os.path.basename(file))])
ngcMBE2922_3_MAR = (xr.open_mfdataset(files_MAR, combine="by_coords")
                    .rename({"ncells": "cell"})
                    .sel(time=slice('1979-03-01','1979-03-31'))
                    .mean(dim="time")
                   )
directory = "/work/mh0287/m214002/GIT/icon-mpim-master_dbf6dad5/experiments/mbe2922"
all_files = glob.glob(f"{directory}/*")

pattern_MAR = re.compile(r'mbe2922_atm_3d_P1D_3_\d{4}03(0[1-9]|[12][0-9]|3[01])T.*\.nc')

files_MAR = sorted([file for file in all_files if pattern_MAR.match(os.path.basename(file))])
ngcMBE2922_4_MAR = (xr.open_mfdataset(files_MAR, combine="by_coords")
                    .rename({"ncells": "cell"})
                    .sel(time=slice('1979-03-01','1979-03-31'))
                    .mean(dim="time")
                   )
directory = "/work/mh0287/m214002/GIT/icon-mpim-master_dbf6dad5/experiments/mbe2922"
all_files = glob.glob(f"{directory}/*")

pattern_MAR = re.compile(r'mbe2922_atm_2d_P1D_ml_\d{4}03(0[1-9]|[12][0-9]|3[01])T.*\.nc')

files_MAR = sorted([file for file in all_files if pattern_MAR.match(os.path.basename(file))])
ngcMBE2922_2D_MAR = (xr.open_mfdataset(files_MAR, combine="by_coords")
                    .rename({"ncells": "cell"})
                    .sel(time=slice('1979-03-01','1979-03-31'))
                    .mean(dim="time")
                   )
## win0000
win_exp = 'win0030'
directory = f"/work/mh0287/m300868/00_ICON/icon-mpim-master_dbf6dad5/experiments/{win_exp}"
all_files = glob.glob(f"{directory}/*")

pattern_MAR = re.compile(f'{win_exp}'+r'_atm_3d_P1D_4_\d{4}03(0[1-9]|[12][0-9]|3[01])T.*\.nc')

files_MAR = sorted([file for file in all_files if pattern_MAR.match(os.path.basename(file))])
ngcWIN0000_MAR = (xr.open_mfdataset(files_MAR, combine="by_coords")
                    .rename({"ncells": "cell"})
                    .sel(time=slice('1979-03-01','1979-03-31'))
                    .mean(dim="time")
                   )
directory = f"/work/mh0287/m300868/00_ICON/icon-mpim-master_dbf6dad5/experiments/{win_exp}"
all_files = glob.glob(f"{directory}/*")

pattern_MAR = re.compile(f'{win_exp}'+r'_atm_3d_P1D_2_\d{4}03(0[1-9]|[12][0-9]|3[01])T.*\.nc')

files_MAR = sorted([file for file in all_files if pattern_MAR.match(os.path.basename(file))])
ngcWIN0000_2_MAR = (xr.open_mfdataset(files_MAR, combine="by_coords")
                    .rename({"ncells": "cell"})
                    .sel(time=slice('1979-03-01','1979-03-31'))
                    .mean(dim="time")
                   )
directory = f"/work/mh0287/m300868/00_ICON/icon-mpim-master_dbf6dad5/experiments/{win_exp}"
all_files = glob.glob(f"{directory}/*")

pattern_MAR = re.compile(f'{win_exp}'+r'_atm_3d_P1D_1_\d{4}03(0[1-9]|[12][0-9]|3[01])T.*\.nc')

files_MAR = sorted([file for file in all_files if pattern_MAR.match(os.path.basename(file))])
ngcWIN0000_3_MAR = (xr.open_mfdataset(files_MAR, combine="by_coords")
                    .rename({"ncells": "cell"})
                    .sel(time=slice('1979-03-01','1979-03-31'))
                    .mean(dim="time")
                   )
directory = f"/work/mh0287/m300868/00_ICON/icon-mpim-master_dbf6dad5/experiments/{win_exp}"
all_files = glob.glob(f"{directory}/*")

pattern_MAR = re.compile(f'{win_exp}'+r'_atm_2d_P1D_ml_\d{4}03(0[1-9]|[12][0-9]|3[01])T.*\.nc')

files_MAR = sorted([file for file in all_files if pattern_MAR.match(os.path.basename(file))])
ngcWIN0000_2D_MAR = (xr.open_mfdataset(files_MAR, combine="by_coords")
                    .rename({"ncells": "cell"})
                    .sel(time=slice('1979-03-01','1979-03-31'))
                    .mean(dim="time")
                   )
# Tropical Pacific
def ocean(ds):
    return ds.ocean_fraction_surface == 1
def ocean_native(ds):
    return ds.cell_sea_land_mask == -2
TOP = 10
BOTTOM = -10
wPcf_LEFT  = 150
wPcf_RIGHT = 170
ePcf_LEFT  = 240
ePcf_RIGHT = 260

def west_pacific(ds):
    return (ds.lat > BOTTOM) & (ds.lat < TOP) & (ds.lon > wPcf_LEFT) & (ds.lon < wPcf_RIGHT)

def east_pacific(ds):
    return (ds.lat > BOTTOM) & (ds.lat < TOP) & (ds.lon > ePcf_LEFT) & (ds.lon < ePcf_RIGHT)
### west_pacific
rho
ngcMBE2922_3_MAR_wPcf = ngcMBE2922_3_MAR.where(west_pacific(ocean_native(grid_atm)), drop=True)
ngcWIN0000_3_MAR_wPcf = ngcWIN0000_3_MAR.where(west_pacific(ocean_native(grid_atm)), drop=True) 
ngcMBE2922_3_MAR_wPcf_rho = ngcMBE2922_3_MAR_wPcf.rho.mean(dim='cell').compute()
ngcWIN0000_3_MAR_wPcf_rho = ngcWIN0000_3_MAR_wPcf.rho.mean(dim='cell').compute()
R_d = 287.05
g = 9.80665

# From ideal gas law: rho = P / (R*Tv)

# Tv = T * (1 + 0.61 * q), where q is specific humidity
T_v = era5_T_MAR.T * (1 + 0.61 * era5_q_MAR.Q)

# Calculate Density
# rho = p / (R_d * Tv)

era5_MAR_rho = (era5_geoHeight_MAR.plev / (R_d * T_v)).compute()  # Example computation
era5_MAR_rho;
era5_MAR_wPcf_rho = era5_MAR_rho.where(west_pacific(ocean(ngc4008)), drop=True).mean(dim='cell').compute()
T
ngcMBE2922_MAR_wPcf = ngcMBE2922_MAR.where(west_pacific(ocean_native(grid_atm)), drop=True)
ngcWIN0000_MAR_wPcf = ngcWIN0000_MAR.where(west_pacific(ocean_native(grid_atm)), drop=True) 
ngcMBE2922_MAR_wPcf_T = ngcMBE2922_MAR_wPcf.ta.mean(dim='cell').compute()
ngcWIN0000_MAR_wPcf_T = ngcWIN0000_MAR_wPcf.ta.mean(dim='cell').compute()
era5_MAR_wPcf_T = era5_T_MAR.T.where(west_pacific(ocean(ngc4008)), drop=True).mean(dim='cell').compute()
### east_pacific
rho
ngcMBE2922_3_MAR_ePcf = ngcMBE2922_3_MAR.where(east_pacific(ocean_native(grid_atm)), drop=True)
ngcWIN0000_3_MAR_ePcf = ngcWIN0000_3_MAR.where(east_pacific(ocean_native(grid_atm)), drop=True) 
ngcMBE2922_3_MAR_ePcf_rho = ngcMBE2922_3_MAR_ePcf.rho.mean(dim='cell').compute()
ngcWIN0000_3_MAR_ePcf_rho = ngcWIN0000_3_MAR_ePcf.rho.mean(dim='cell').compute()
era5_MAR_ePcf_rho = era5_MAR_rho.where(east_pacific(ocean(ngc4008)), drop=True).mean(dim='cell').compute()
T
ngcMBE2922_MAR_ePcf = ngcMBE2922_MAR.where(east_pacific(ocean_native(grid_atm)), drop=True)
ngcWIN0000_MAR_ePcf = ngcWIN0000_MAR.where(east_pacific(ocean_native(grid_atm)), drop=True) 
ngcMBE2922_MAR_ePcf_T = ngcMBE2922_MAR_ePcf.ta.mean(dim='cell').compute()
ngcWIN0000_MAR_ePcf_T = ngcWIN0000_MAR_ePcf.ta.mean(dim='cell').compute()
era5_MAR_ePcf_T = era5_T_MAR.T.where(east_pacific(ocean(ngc4008)), drop=True).mean(dim='cell').compute()
### Get Heights
g_at_equator = 9.78
#### 1979 - ERA5
era5_geoHeight_MAR_ePcf = era5_geoHeight_MAR.where(east_pacific(ocean(ngc4008)), drop=True)
era5_geoHeight_MAR_wPcf = era5_geoHeight_MAR.where(west_pacific(ocean(ngc4008)), drop=True)
era5_MAR_wPcf_height_in_meter = ((era5_geoHeight_MAR_wPcf.Z / g_at_equator)).mean(dim="cell")
era5_MAR_ePcf_height_in_meter = ((era5_geoHeight_MAR_ePcf.Z / g_at_equator)).mean(dim="cell")
#### 2020 - ICON AMIP
ngc4008_wPcf = ngc4008.where(west_pacific(ocean(ngc4008)), drop=True)
ngc4008_ePcf = ngc4008.where(east_pacific(ocean(ngc4008)), drop=True)
AMIP_1979_MAR_wPcf_height_in_meter = ngc4008_wPcf.zg.mean(dim="cell").sel(level_full=ngcMBE2922_MAR_wPcf.height_2).compute()
AMIP_1979_MAR_ePcf_height_in_meter = ngc4008_ePcf.zg.mean(dim="cell").sel(level_full=ngcMBE2922_MAR_ePcf.height_2).compute()
# Interpolate ERA5 to ICON Vertical Grid
west_pacific
era5_MAR_wPcf_T_with_height = era5_MAR_wPcf_T.assign_coords(height=("plev", era5_MAR_wPcf_height_in_meter.data))
era5_MAR_wPcf_T_with_height = era5_MAR_wPcf_T_with_height.swap_dims({"plev": "height"})
era5_MAR_wPcf_T_with_height_interp = (
    era5_MAR_wPcf_T_with_height.interp(height=AMIP_1979_MAR_wPcf_height_in_meter, method="linear")
    .drop_vars({'plev','height'})
    .rename({'height_2':'height'})
)
east_pacific
era5_MAR_ePcf_T_with_height = era5_MAR_ePcf_T.assign_coords(height=("plev", era5_MAR_ePcf_height_in_meter.data))
era5_MAR_ePcf_T_with_height = era5_MAR_ePcf_T_with_height.swap_dims({"plev": "height"})
era5_MAR_ePcf_T_with_height_interp = (
    era5_MAR_ePcf_T_with_height.interp(height=AMIP_1979_MAR_ePcf_height_in_meter, method="linear")
    .drop_vars({'plev','height'})
    .rename({'height_2':'height'})
)
# Plotting
import matplotlib.lines as mlines
SIZE = 20
plt.rcParams["axes.labelsize"] = SIZE
plt.rcParams["legend.fontsize"] = SIZE
plt.rcParams["xtick.labelsize"] = SIZE
plt.rcParams["ytick.labelsize"] = SIZE
plt.rcParams["font.size"] = SIZE
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.major.size'] = 6
plt.rcParams['ytick.major.size'] = 6
plt.rc('text', usetex=False)
plt.rc('font', family='serif')
fig = plt.figure(figsize=(14,8), facecolor="w", edgecolor="k")
#fig.suptitle(f"{win_exp}: March 1979 \n Moist Static Energy \n"+r"$\mathrm{MSE} = c_{\mathrm{p}} \cdot T + L_{\mathrm{v}} \cdot q + g \cdot z$", y=1.2)

G = gridspec.GridSpec(1, 2, hspace=0.7, wspace=0.5)

XLOW  = -3
XHIGH = 3
NUM_TICKS = 6

YLOW  = 0
YHIGH = 15000

##################################################################################################################
# 1st Row
LW=3
BUCHSTABI = 27
##################################################################################################################
##################################################################################################################

ax = plt.subplot(G[0, 0])
ax.axvline(0, color='grey', ls='dotted')

ax.plot(ngcMBE2922_MAR_wPcf_T-ngcMBE2922_MAR_ePcf_T, AMIP_1979_MAR_wPcf_height_in_meter, label='Control', color="#009E73", ls='solid', lw=LW)
ax.plot(ngcWIN0000_MAR_wPcf_T-ngcWIN0000_MAR_ePcf_T, AMIP_1979_MAR_wPcf_height_in_meter, label='OptiFlux', color="#D55E00", ls='solid', lw=LW)
ax.plot(era5_MAR_wPcf_T-era5_MAR_ePcf_T, era5_MAR_wPcf_height_in_meter, label='ERA5', color='black', lw=LW)

#ax.plot(ngcMBE2922_3_MAR_wPcf_rho, AMIP_1979_MAR_wPcf_height_in_meter, label='Control', color="black", ls='solid', lw=LW)

ax.set_title(r"Western Pacific $-$ Eastern Pacific", pad=30)
ax.set_ylabel(r"Height / m ")
ax.set_xlabel(r"$T$ / K")

ax.set_xlim(XLOW, XHIGH)
XTICKS = np.linspace(XLOW, XHIGH, NUM_TICKS)
ax.set_xticks(XTICKS)
ax.set_ylim(YLOW, YHIGH)

ax.spines[["bottom", "left"]].set_position(("outward", 20))
ax.spines[["right", "top"]].set_visible(False)

ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), fancybox=True, shadow=False, ncol=4)

filename = f"figs/fig_08.pdf"
#plt.savefig(filename, facecolor='white', bbox_inches='tight', dpi=800)

plt.show()
fig = plt.figure(figsize=(12,8), facecolor="w", edgecolor="k")
#fig.suptitle(f"{win_exp}: March 1979 \n Moist Static Energy \n"+r"$\mathrm{MSE} = c_{\mathrm{p}} \cdot T + L_{\mathrm{v}} \cdot q + g \cdot z$", y=1.2)

G = gridspec.GridSpec(1, 2, hspace=0.7, wspace=0.5)

XLOW  = 280
XHIGH = 301
NUM_TICKS = 6

YLOW  = 0
YHIGH = 4000

##################################################################################################################
# 1st Row
LW=3
BUCHSTABI = 27
##################################################################################################################
##################################################################################################################

ax = plt.subplot(G[0, 0])
ax.axvline(0, color='grey', ls='dotted')

ax.plot(ngcMBE2922_MAR_wPcf_T, AMIP_1979_MAR_wPcf_height_in_meter, label='Control', color="#009E73", ls='solid', lw=LW)
ax.plot(ngcWIN0000_MAR_wPcf_T, AMIP_1979_MAR_wPcf_height_in_meter, label='OptiFlux', color="#D55E00", ls='solid', lw=LW)
ax.plot(era5_MAR_wPcf_T, era5_MAR_wPcf_height_in_meter, label='ERA5', color='black', lw=LW)

#ax.plot(ngcMBE2922_3_MAR_wPcf_rho, AMIP_1979_MAR_wPcf_height_in_meter, label='Control', color="black", ls='solid', lw=LW)

ax.set_title(r"Western Pacific", pad=30)
ax.set_ylabel(r"Height / m ")
ax.set_xlabel(r"$T$ / K")

ax.set_xlim(XLOW, XHIGH)
XTICKS = np.linspace(XLOW, XHIGH, NUM_TICKS)
ax.set_xticks(XTICKS)
ax.set_ylim(YLOW, YHIGH)

ax.spines[["bottom", "left"]].set_position(("outward", 20))
ax.spines[["right", "top"]].set_visible(False)

#ax.text(-0.3, 1.1, '(a)', fontsize=BUCHSTABI, transform=ax.transAxes, color='black')

#ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), fancybox=True, shadow=False, ncol=4)

##################################################################################################################
##################################################################################################################

ax = plt.subplot(G[0, 1])
ax.axvline(0, color='grey', ls='dotted')

ax.plot(ngcMBE2922_MAR_ePcf_T, AMIP_1979_MAR_ePcf_height_in_meter, label=r'Control$-$ERA5', color="#009E73", ls='solid', lw=LW)
ax.plot(ngcWIN0000_MAR_ePcf_T, AMIP_1979_MAR_ePcf_height_in_meter, label=r'OptiFlux$-$ERA5', color="#D55E00", ls='solid', lw=LW)
ax.plot(era5_MAR_ePcf_T, era5_MAR_wPcf_height_in_meter, label='ERA5', color='black', lw=LW)

#ax.plot(ngcMBE2922_3_MAR_wPcf_rho, AMIP_1979_MAR_wPcf_height_in_meter, label='Control', color="black", ls='solid', lw=LW)

ax.set_title(r"Eastern Pacific", pad=30)
ax.set_ylabel(r"Height / m ")
ax.set_xlabel(r"$T$ / K")

ax.set_xlim(XLOW, XHIGH)
XTICKS = np.linspace(XLOW, XHIGH, NUM_TICKS)
ax.set_xticks(XTICKS)
ax.set_ylim(YLOW, YHIGH)

ax.spines[["bottom", "left"]].set_position(("outward", 20))
ax.spines[["right", "top"]].set_visible(False)

#ax.text(-0.3, 1.1, '(a)', fontsize=BUCHSTABI, transform=ax.transAxes, color='black')

ax.legend(loc='upper center', bbox_to_anchor=(-0.3, -0.2), fancybox=True, shadow=False, ncol=4)

filename = f"figs/fig_08.pdf"
#plt.savefig(filename, facecolor='white', bbox_inches='tight', dpi=800)

plt.show()
fig = plt.figure(figsize=(6,8), facecolor="w", edgecolor="k")
#fig.suptitle(f"{win_exp}: March 1979 \n Moist Static Energy \n"+r"$\mathrm{MSE} = c_{\mathrm{p}} \cdot T + L_{\mathrm{v}} \cdot q + g \cdot z$", y=1.2)

G = gridspec.GridSpec(1, 1, hspace=0.7, wspace=0.5)

XLOW  = -12
XHIGH = 12
YLOW  = 0
YHIGH = 15000

##################################################################################################################
# 1st Row
LW=3
BUCHSTABI = 27
##################################################################################################################

ax = plt.subplot(G[0, 0])
ax.axvline(0, color='grey', ls='dotted')

ax.plot((ngcMBE2922_3_MAR_wPcf_rho-ngcMBE2922_3_MAR_ePcf_rho)*1000, AMIP_1979_MAR_wPcf_height_in_meter, label='Control', color="#009E73", ls='solid', lw=LW)
ax.plot((ngcWIN0000_3_MAR_wPcf_rho-ngcWIN0000_3_MAR_ePcf_rho)*1000, AMIP_1979_MAR_wPcf_height_in_meter, label='OptiFlux', color="#D55E00", ls='solid', lw=LW)
ax.plot((era5_MAR_wPcf_rho-era5_MAR_ePcf_rho)*1000, (era5_MAR_ePcf_height_in_meter+era5_MAR_wPcf_height_in_meter)/2, label='ERA5', color='black', lw=LW)

#ax.plot(ngcMBE2922_3_MAR_wPcf_rho, AMIP_1979_MAR_wPcf_height_in_meter, label='Control', color="black", ls='solid', lw=LW)

ax.set_title(r"Western Pacific $-$ Eastern Pacific", pad=30)
ax.set_ylabel(r"Height / m ")
ax.set_xlabel(r"$\Delta \rho$ / $\cdot10^{-3}$ kg$\cdot$m$^{-3}$")

ax.set_xlim(XLOW, XHIGH)
ax.set_ylim(YLOW, YHIGH)

ax.spines[["bottom", "left"]].set_position(("outward", 20))
ax.spines[["right", "top"]].set_visible(False)

#ax.text(-0.3, 1.1, '(a)', fontsize=BUCHSTABI, transform=ax.transAxes, color='black')

ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), fancybox=True, shadow=False, ncol=4)

filename = f"figs/fig_08.pdf"
#plt.savefig(filename, facecolor='white', bbox_inches='tight', dpi=800)

plt.show()
# All Together
SIZE = 35
plt.rcParams["axes.labelsize"] = SIZE
plt.rcParams["legend.fontsize"] = SIZE
plt.rcParams["xtick.labelsize"] = SIZE
plt.rcParams["ytick.labelsize"] = SIZE
plt.rcParams["font.size"] = SIZE
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.major.size'] = 12
plt.rcParams['ytick.major.size'] = 12
# Enable LaTeX font
plt.rc('text', usetex=False)
plt.rc('font', family='sans-serif')
fig = plt.figure(figsize=(24,18), facecolor="w", edgecolor="k")
#fig.suptitle(f"{win_exp}: March 1979 \n Moist Static Energy \n"+r"$\mathrm{MSE} = c_{\mathrm{p}} \cdot T + L_{\mathrm{v}} \cdot q + g \cdot z$", y=1.2)

G = gridspec.GridSpec(2, 8, hspace=0.45, wspace=0.4, height_ratios=[2, 3])

#####################################################################################################################
BUCHSTABI = 45
#####################################################################################################################

ax1 = plt.subplot(G[0, :], projection=ccrs.PlateCarree(central_longitude=180))
ax1.set_extent([120, 290, -30, 30], crs=ccrs.PlateCarree())

ax1.coastlines()
# Add degree values to x and y ticks
gl = ax1.gridlines(draw_labels=True, xlocs=[150, 170, -120, -100], ylocs=[-10, 10])
gl.xlabel_style = {'size': SIZE-5}  
gl.ylabel_style = {'size': SIZE-5} 
gl.top_labels = False
gl.right_labels = False

ax1.add_patch(Rectangle((-30,-10),20,20,edgecolor='black', facecolor='None', lw=5, alpha=1, zorder=10))
ax1.add_patch(Rectangle((60,-10),20,20,edgecolor='black', facecolor='None', lw=5, alpha=1, zorder=10))

ax1.text(-20, 30, "Western Pacific", ha="center", va="bottom", fontsize=SIZE+5, color="black", zorder=11)
ax1.text(70, 30, "Eastern Pacific", ha="center", va="bottom", fontsize=SIZE+5, color="black", zorder=11)

ax1.text(-0.1, 1.2, '(a)', fontsize=BUCHSTABI, transform=ax1.transAxes)

#####################################################################################################################
#####################################################################################################################

XLOW  = -10
XHIGH = 10
YLOW  = 0
YHIGH = 15000

##################################################################################################################
LW=5
NUM_TICKS = 8
##################################################################################################################

ax = plt.subplot(G[1, 0:3])
ax.axvline(0, color='grey', ls='dotted', lw=2)

ax.plot((ngcMBE2922_3_MAR_wPcf_rho-ngcMBE2922_3_MAR_ePcf_rho)*1000, AMIP_1979_MAR_wPcf_height_in_meter, label='Control', color="#009E73", ls='solid', lw=LW)
ax.plot((ngcWIN0000_3_MAR_wPcf_rho-ngcWIN0000_3_MAR_ePcf_rho)*1000, AMIP_1979_MAR_wPcf_height_in_meter, label='OptiFlux', color="#D55E00", ls='solid', lw=LW)
ax.plot((era5_MAR_wPcf_rho-era5_MAR_ePcf_rho)*1000, (era5_MAR_ePcf_height_in_meter+era5_MAR_wPcf_height_in_meter)/2, label='ERA5', color='black', lw=LW)

#ax.plot(ngcMBE2922_3_MAR_wPcf_rho, AMIP_1979_MAR_wPcf_height_in_meter, label='Control', color="black", ls='solid', lw=LW)

ax.set_title(r"West. P.$-$East. P.", pad=30)
ax.set_ylabel(r"Height / m ")
ax.set_xlabel(r"$\Delta \rho$ / $\cdot10^{-3}$ kg$\cdot$m$^{-3}$")

ax.set_xlim(XLOW, XHIGH)
ax.set_yticks([0,2000,4000,6000,8000,10000,12000,14000])
ax.set_ylim(YLOW, YHIGH)

# Configure the left y-axis
ax.spines[["bottom", "left"]].set_position(("outward", 20))
ax.spines[["top", "right"]].set_visible(False)  # Hide both top and right spines

# Set up the left y-axis with label and tick labels
ax.set_ylabel("Height / m")  # Set the y-axis label on the left
ax.tick_params(axis="y", which="both", labelleft=True)  # Show tick labels on the left y-axis

# Annotate the subplot
ax.text(-0.3, 1.15, '(b)', fontsize=BUCHSTABI, transform=ax.transAxes, color='black')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), fancybox=True, shadow=False, ncol=1)


##################################################################################################################
##################################################################################################################
XLOW  = -3
XHIGH = 1
NUM_TICKS = 5

YLOW  = 0
YHIGH = 4000
##################################################################################################################
##################################################################################################################

ax = plt.subplot(G[1, 4:6])
ax.axvline(0, color='grey', ls='dotted', lw=2)

ax.plot(ngcMBE2922_MAR_wPcf_T-era5_MAR_wPcf_T_with_height_interp, AMIP_1979_MAR_wPcf_height_in_meter, label='Control', color="#009E73", ls='solid', lw=LW)
ax.plot(ngcWIN0000_MAR_wPcf_T-era5_MAR_wPcf_T_with_height_interp, AMIP_1979_MAR_wPcf_height_in_meter, label='OptiFlux', color="#D55E00", ls='solid', lw=LW)
#ax.plot(era5_MAR_wPcf_T, era5_MAR_wPcf_height_in_meter, label='ERA5', color='black', lw=LW)

#ax.plot(ngcMBE2922_3_MAR_wPcf_rho, AMIP_1979_MAR_wPcf_height_in_meter, label='Control', color="black", ls='solid', lw=LW)

ax.set_title(r"West. P.", pad=30)
#ax.set_ylabel(r"Height / m ")
ax.set_xlabel(r"AirT / K")

XTICKS = np.linspace(XLOW, XHIGH, NUM_TICKS)
ax.set_xticks(XTICKS)
ax.set_xlim(XLOW, XHIGH)

YTICKS = np.linspace(YLOW, YHIGH, 3)
ax.set_yticks(YTICKS)
ax.set_ylim(YLOW, YHIGH)

ax.spines[["bottom", "left"]].set_position(("outward", 20))
ax.spines[["right", "top"]].set_visible(False)

ax.text(-0.3, 1.15, '(c)', fontsize=BUCHSTABI, transform=ax.transAxes, color='black')

#ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), fancybox=True, shadow=False, ncol=4)

##################################################################################################################
##################################################################################################################

ax = plt.subplot(G[1, 6:])
ax.axvline(0, color='grey', ls='dotted', lw=2)

ax.plot(ngcMBE2922_MAR_ePcf_T-era5_MAR_ePcf_T_with_height_interp, AMIP_1979_MAR_ePcf_height_in_meter, label=r'Control$-$ERA5', color="#009E73", ls='solid', lw=LW)
ax.plot(ngcWIN0000_MAR_ePcf_T-era5_MAR_ePcf_T_with_height_interp, AMIP_1979_MAR_ePcf_height_in_meter, label=r'OptiFlux$-$ERA5', color="#D55E00", ls='solid', lw=LW)
#ax.plot(era5_MAR_ePcf_T, era5_MAR_wPcf_height_in_meter, label='ERA5', color='black', lw=LW)

#ax.plot(ngcMBE2922_3_MAR_wPcf_rho, AMIP_1979_MAR_wPcf_height_in_meter, label='Control', color="black", ls='solid', lw=LW)

ax.set_title(r"East. P.", pad=30)
#ax.set_ylabel(r"Height / m ")
ax.set_yticklabels([])
ax.set_xlabel(r"AirT / K")

ax.set_xlim(XLOW, XHIGH)
XTICKS = np.linspace(XLOW, XHIGH, NUM_TICKS)
ax.set_xticks(XTICKS)

YTICKS = np.linspace(YLOW, YHIGH, 3)
ax.set_yticks(YTICKS)
ax.set_ylim(YLOW, YHIGH)

ax.spines[["bottom", "left"]].set_position(("outward", 20))
ax.spines[["right", "top", "left"]].set_visible(False)

ax.text(-0.3, 1.15, '(d)', fontsize=BUCHSTABI, transform=ax.transAxes, color='black')

ax.legend(loc='upper center', bbox_to_anchor=(-0, -0.2), fancybox=True, shadow=False, ncol=1)

##################################################################################################################
##################################################################################################################
#                             xx, xx        yy, yy
#fig.add_artist(plt.Line2D((0.302, 0.475), (0.11, 0.11), color="grey", lw=1, ls='solid', zorder=0))  # Line at 0m
#fig.add_artist(plt.Line2D((0.302, 0.475), (0.219, 0.53), color="grey", lw=1, ls='solid', zorder=0))  # Line at 4000m

filename = f"figs/fig_06.pdf"
plt.savefig(filename, facecolor='white', bbox_inches='tight', dpi=800)

plt.show()
fig = plt.figure(figsize=(24,10), facecolor="w", edgecolor="k")
#fig.suptitle(f"{win_exp}: March 1979 \n Moist Static Energy \n"+r"$\mathrm{MSE} = c_{\mathrm{p}} \cdot T + L_{\mathrm{v}} \cdot q + g \cdot z$", y=1.2)

G = gridspec.GridSpec(1, 8, hspace=0.45, wspace=0.4)

#####################################################################################################################
BUCHSTABI = 45
#####################################################################################################################

XLOW  = -10
XHIGH = 10
YLOW  = 0
YHIGH = 15000

##################################################################################################################
LW=5
NUM_TICKS = 8
##################################################################################################################

ax = plt.subplot(G[0, 0:3])
ax.axvline(0, color='grey', ls='dotted', lw=2)

ax.plot((ngcMBE2922_3_MAR_wPcf_rho-ngcMBE2922_3_MAR_ePcf_rho)*1000, AMIP_1979_MAR_wPcf_height_in_meter, label='Control', color="#009E73", ls='solid', lw=LW)
ax.plot((ngcWIN0000_3_MAR_wPcf_rho-ngcWIN0000_3_MAR_ePcf_rho)*1000, AMIP_1979_MAR_wPcf_height_in_meter, label='OptiFlux', color="#D55E00", ls='solid', lw=LW)
ax.plot((era5_MAR_wPcf_rho-era5_MAR_ePcf_rho)*1000, (era5_MAR_ePcf_height_in_meter+era5_MAR_wPcf_height_in_meter)/2, label='ERA5', color='black', lw=LW)

#ax.plot(ngcMBE2922_3_MAR_wPcf_rho, AMIP_1979_MAR_wPcf_height_in_meter, label='Control', color="black", ls='solid', lw=LW)

ax.set_title(r"West. P.$-$East. P.", pad=30)
ax.set_ylabel(r"Height / m ")
ax.set_xlabel(r"$\Delta \rho$ / $\cdot10^{-3}$ kg$\cdot$m$^{-3}$")

ax.set_xlim(XLOW, XHIGH)
ax.set_yticks([0,2000,4000,6000,8000,10000,12000,14000])
ax.set_ylim(YLOW, YHIGH)

# Configure the left y-axis
ax.spines[["bottom", "left"]].set_position(("outward", 20))
ax.spines[["top", "right"]].set_visible(False)  # Hide both top and right spines

# Set up the left y-axis with label and tick labels
ax.set_ylabel("Height / m")  # Set the y-axis label on the left
ax.tick_params(axis="y", which="both", labelleft=True)  # Show tick labels on the left y-axis

# Annotate the subplot
ax.text(-0.3, 1.15, '(a)', fontsize=BUCHSTABI, transform=ax.transAxes, color='black')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), fancybox=True, shadow=False, ncol=1)


##################################################################################################################
##################################################################################################################
XLOW  = -3
XHIGH = 1
NUM_TICKS = 5

YLOW  = 0
YHIGH = 4000
##################################################################################################################
##################################################################################################################

ax = plt.subplot(G[0, 4:6])
ax.axvline(0, color='grey', ls='dotted', lw=2)

ax.plot(ngcMBE2922_MAR_wPcf_T-era5_MAR_wPcf_T_with_height_interp, AMIP_1979_MAR_wPcf_height_in_meter, label='Control', color="#009E73", ls='solid', lw=LW)
ax.plot(ngcWIN0000_MAR_wPcf_T-era5_MAR_wPcf_T_with_height_interp, AMIP_1979_MAR_wPcf_height_in_meter, label='OptiFlux', color="#D55E00", ls='solid', lw=LW)
#ax.plot(era5_MAR_wPcf_T, era5_MAR_wPcf_height_in_meter, label='ERA5', color='black', lw=LW)

#ax.plot(ngcMBE2922_3_MAR_wPcf_rho, AMIP_1979_MAR_wPcf_height_in_meter, label='Control', color="black", ls='solid', lw=LW)

ax.set_title(r"West. P.", pad=30)
#ax.set_ylabel(r"Height / m ")
ax.set_xlabel(r"AirT / K")

XTICKS = np.linspace(XLOW, XHIGH, NUM_TICKS)
ax.set_xticks(XTICKS)
ax.set_xlim(XLOW, XHIGH)

YTICKS = np.linspace(YLOW, YHIGH, 3)
ax.set_yticks(YTICKS)
ax.set_ylim(YLOW, YHIGH)

ax.spines[["bottom", "left"]].set_position(("outward", 20))
ax.spines[["right", "top"]].set_visible(False)

ax.text(-0.3, 1.15, '(b)', fontsize=BUCHSTABI, transform=ax.transAxes, color='black')

#ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), fancybox=True, shadow=False, ncol=4)

##################################################################################################################
##################################################################################################################

ax = plt.subplot(G[0, 6:])
ax.axvline(0, color='grey', ls='dotted', lw=2)

ax.plot(ngcMBE2922_MAR_ePcf_T-era5_MAR_ePcf_T_with_height_interp, AMIP_1979_MAR_ePcf_height_in_meter, label=r'Control$-$ERA5', color="#009E73", ls='solid', lw=LW)
ax.plot(ngcWIN0000_MAR_ePcf_T-era5_MAR_ePcf_T_with_height_interp, AMIP_1979_MAR_ePcf_height_in_meter, label=r'OptiFlux$-$ERA5', color="#D55E00", ls='solid', lw=LW)
#ax.plot(era5_MAR_ePcf_T, era5_MAR_wPcf_height_in_meter, label='ERA5', color='black', lw=LW)

#ax.plot(ngcMBE2922_3_MAR_wPcf_rho, AMIP_1979_MAR_wPcf_height_in_meter, label='Control', color="black", ls='solid', lw=LW)

ax.set_title(r"East. P.", pad=30)
#ax.set_ylabel(r"Height / m ")
ax.set_yticklabels([])
ax.set_xlabel(r"AirT / K")

ax.set_xlim(XLOW, XHIGH)
XTICKS = np.linspace(XLOW, XHIGH, NUM_TICKS)
ax.set_xticks(XTICKS)

YTICKS = np.linspace(YLOW, YHIGH, 3)
ax.set_yticks(YTICKS)
ax.set_ylim(YLOW, YHIGH)

ax.spines[["bottom", "left"]].set_position(("outward", 20))
ax.spines[["right", "top", "left"]].set_visible(False)

ax.text(-0.3, 1.15, '(c)', fontsize=BUCHSTABI, transform=ax.transAxes, color='black')

ax.legend(loc='upper center', bbox_to_anchor=(-0, -0.2), fancybox=True, shadow=False, ncol=1)

##################################################################################################################
##################################################################################################################
#                             xx, xx        yy, yy
#fig.add_artist(plt.Line2D((0.302, 0.475), (0.11, 0.11), color="grey", lw=1, ls='solid', zorder=0))  # Line at 0m
#fig.add_artist(plt.Line2D((0.302, 0.475), (0.219, 0.53), color="grey", lw=1, ls='solid', zorder=0))  # Line at 4000m

filename = f"figs/fig_06_v1.png"
plt.savefig(filename, facecolor='white', bbox_inches='tight', dpi=400)

plt.show()
SIZE = 25
plt.rcParams["axes.labelsize"] = SIZE
plt.rcParams["legend.fontsize"] = SIZE
plt.rcParams["xtick.labelsize"] = SIZE
plt.rcParams["ytick.labelsize"] = SIZE
plt.rcParams["font.size"] = SIZE
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.major.size'] = 12
plt.rcParams['ytick.major.size'] = 12
# Enable LaTeX font
plt.rc('text', usetex=False)
plt.rc('font', family='sans-serif')

fig = plt.figure(figsize=(6,8), facecolor="w", edgecolor="k")
#fig.suptitle(f"{win_exp}: March 1979 \n Moist Static Energy \n"+r"$\mathrm{MSE} = c_{\mathrm{p}} \cdot T + L_{\mathrm{v}} \cdot q + g \cdot z$", y=1.2)

G = gridspec.GridSpec(1, 1, hspace=0.45, wspace=0.4)

#####################################################################################################################
BUCHSTABI = 30
#####################################################################################################################

XLOW  = -10
XHIGH = 10
YLOW  = 0
YHIGH = 15000

##################################################################################################################
LW=5
NUM_TICKS = 8
##################################################################################################################

ax = plt.subplot(G[0, 0:3])
ax.axvline(0, color='grey', ls='dotted', lw=2)

ax.plot((ngcMBE2922_3_MAR_wPcf_rho-ngcMBE2922_3_MAR_ePcf_rho)*1000, AMIP_1979_MAR_wPcf_height_in_meter, label='Control', color="#009E73", ls='solid', lw=LW)
ax.plot((ngcWIN0000_3_MAR_wPcf_rho-ngcWIN0000_3_MAR_ePcf_rho)*1000, AMIP_1979_MAR_wPcf_height_in_meter, label='OptiFlux', color="#D55E00", ls='solid', lw=LW)
ax.plot((era5_MAR_wPcf_rho-era5_MAR_ePcf_rho)*1000, (era5_MAR_ePcf_height_in_meter+era5_MAR_wPcf_height_in_meter)/2, label='ERA5', color='black', lw=LW)

#ax.plot(ngcMBE2922_3_MAR_wPcf_rho, AMIP_1979_MAR_wPcf_height_in_meter, label='Control', color="black", ls='solid', lw=LW)

ax.set_title(r"West. P.$-$East. P.", pad=30)
ax.set_ylabel(r"Height / m ")
ax.set_xlabel(r"$\Delta \rho$ / $\cdot10^{-3}$ kg$\cdot$m$^{-3}$")

ax.set_xlim(XLOW, XHIGH)
ax.set_yticks([0,2000,4000,6000,8000,10000,12000,14000])
ax.set_ylim(YLOW, YHIGH)

# Configure the left y-axis
ax.spines[["bottom", "left"]].set_position(("outward", 20))
ax.spines[["top", "right"]].set_visible(False)  # Hide both top and right spines

# Set up the left y-axis with label and tick labels
ax.set_ylabel("Height / m")  # Set the y-axis label on the left
ax.tick_params(axis="y", which="both", labelleft=True)  # Show tick labels on the left y-axis

# Annotate the subplot
ax.text(-0.3, 1.15, '(a)', fontsize=BUCHSTABI, transform=ax.transAxes, color='black')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), fancybox=True, shadow=False, ncol=3)

filename = f"figs/fig_06_v2.png"
plt.savefig(filename, facecolor='white', bbox_inches='tight', dpi=400)

plt.show()
