#%%
# Preamble
import intake
import xarray as xr
import numpy as np
import sys
import pathlib
import easygems.healpix as egh
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs
from matplotlib.patches import Rectangle

sys.path.append(str(pathlib.Path.cwd() / "../../src"))
import toolbox

#%%
# Data
cat = intake.open_catalog("https://digital-earths-global-hackathon.github.io/catalog/catalog.yaml")["EU"]
ZOOM = 7
t1 = "2020-03-01"
t2 = "2020-03-30"
icon_d3hp003 = (
    cat["icon_d3hp003"](zoom=ZOOM)
    .to_dask()
)
icon_d3hp003 = egh.attach_coords(icon_d3hp003)
icon_ngc4008 = (
    cat["icon_ngc4008"](zoom=ZOOM)
    .to_dask()
)
icon_ngc4008 = egh.attach_coords(icon_ngc4008)
ifs_tco3999_ng5_rcbmf_cf = (
    cat["ifs_tco3999-ng5_rcbmf_cf"](time='PT1H', zoom=ZOOM)
    .to_dask()
)
ifs_tco3999_ng5_rcbmf_cf = egh.attach_coords(ifs_tco3999_ng5_rcbmf_cf)
nicam_gl11 = (
    cat["nicam_gl11"](time='PT6H', zoom = ZOOM)
    .to_dask()
)
nicam_gl11 = egh.attach_coords(nicam_gl11)
um_glm_n2560_RAL3p3 = (
    cat["um_glm_n2560_RAL3p3"](time='PT3H', zoom = ZOOM)
    .to_dask()
)
um_glm_n2560_RAL3p3 = egh.attach_coords(um_glm_n2560_RAL3p3)
ERA5 = (
    cat["ERA5"](zoom = ZOOM)
    .to_dask()
)

#%%
# Process Data
def pacific(ds):
    return (ds.lat > BOTTOM) & (ds.lat < TOP) & (ds.lon > LEFT) & (ds.lon < RIGHT)

def ocean(ds):
    return ds.ocean_fraction_surface == 1
    
def ocean_native(ds):
    return ds.cell_sea_land_mask == -2
TOP = 10
BOTTOM = -10
LEFT = 150
RIGHT = 260

#%%
# Time Filter
### ICON
icon_d3hp003_MAR = icon_d3hp003.sel(pressure=100000).sel(time=slice(t1,t2)).mean(dim='time') #.sel(time=t1)
icon_ngc4008_MAR = icon_ngc4008.sel(level_full=90).sel(time=slice(t1,t2)).mean(dim='time') #.sel(time=t1) #.sel(time=slice(t1,t2)).mean(dim='time')
### IFS
ifs_tco3999_ng5_rcbmf_cf_MAR = ifs_tco3999_ng5_rcbmf_cf.sel(level=1000).sel(time=slice(t1,t2)).mean(dim='time')
### NICAM
nicam_gl11_MAR = nicam_gl11.sel(lev=1000).sel(time=slice(t1,t2)).mean(dim='time') #.sel(time=t1).mean(dim='time')
### Unified Model
um_glm_n2560_RAL3p3_MAR = um_glm_n2560_RAL3p3.sel(pressure=1000).sel(time=slice(t1,t2)).mean(dim='time') #.sel(time=t1).mean(dim='time')
### ERA5
ERA5_MAR = ERA5.sel(level=1000).sel(time=slice(t1,t2)).mean(dim='time') #.sel(time=t1).mean(dim='time')

#%%
# Spatial Filter
### ICON
##### icon_ngc4008
icon_ngc4008_MAR_Pcf = icon_ngc4008_MAR.where(pacific(ocean(icon_ngc4008_MAR)), drop=True)
##### icon_d3hp003
icon_d3hp003_MAR_Pcf = icon_d3hp003_MAR.where(pacific(ocean(icon_ngc4008_MAR)), drop=True)
### IFS
ifs_tco3999_ng5_rcbmf_cf_MAR_Pcf = ifs_tco3999_ng5_rcbmf_cf_MAR.where(pacific(ifs_tco3999_ng5_rcbmf_cf_MAR).compute(), drop=True)
#### NICAM
nicam_gl11_MAR_Pcf = nicam_gl11_MAR.where(pacific(nicam_gl11_MAR).compute(), drop=True)
### Unified Model
um_glm_n2560_RAL3p3_MAR_Pcf = um_glm_n2560_RAL3p3_MAR.where(pacific(um_glm_n2560_RAL3p3_MAR).compute(), drop=True)
### ERA5
ERA5_MAR_Pcf = ERA5_MAR.where(pacific(ERA5_MAR).compute(), drop=True)
# Winds
### Zonal Winds
icon_ngc4008_MAR_Pcf_ua = icon_ngc4008_MAR_Pcf.ua
icon_d3hp003_MAR_Pcf_ua = icon_d3hp003_MAR_Pcf.ua
ifs_tco3999_ng5_rcbmf_cf_MAR_Pcf_ua = ifs_tco3999_ng5_rcbmf_cf_MAR_Pcf.ua
nicam_gl11_MAR_Pcf_ua = nicam_gl11_MAR_Pcf.ua
um_glm_n2560_RAL3p3_MAR_Pcf_ua = um_glm_n2560_RAL3p3_MAR_Pcf.ua
ERA5_MAR_Pcf_ua = ERA5_MAR_Pcf.u
### Meridional Winds
icon_ngc4008_MAR_Pcf_va = icon_ngc4008_MAR_Pcf.va
icon_d3hp003_MAR_Pcf_va = icon_d3hp003_MAR_Pcf.va
ifs_tco3999_ng5_rcbmf_cf_MAR_Pcf_va = ifs_tco3999_ng5_rcbmf_cf_MAR_Pcf.va
nicam_gl11_MAR_Pcf_va = nicam_gl11_MAR_Pcf.va
um_glm_n2560_RAL3p3_MAR_Pcf_va = um_glm_n2560_RAL3p3_MAR_Pcf.va
ERA5_MAR_Pcf_va = ERA5_MAR_Pcf.v
### Wind Speed
icon_ngc4008_MAR_Pcf_ws             = (icon_ngc4008_MAR_Pcf_ua**2 + icon_ngc4008_MAR_Pcf_va**2) ** 0.5
icon_d3hp003_MAR_Pcf_ws             = (icon_d3hp003_MAR_Pcf_ua**2 + icon_d3hp003_MAR_Pcf_va**2) ** 0.5
ifs_tco3999_ng5_rcbmf_cf_MAR_Pcf_ws = (ifs_tco3999_ng5_rcbmf_cf_MAR_Pcf_ua**2 + ifs_tco3999_ng5_rcbmf_cf_MAR_Pcf_va**2) ** 0.5
nicam_gl11_MAR_Pcf_ws               = (nicam_gl11_MAR_Pcf_ua**2 + nicam_gl11_MAR_Pcf_va**2) ** 0.5
um_glm_n2560_RAL3p3_MAR_Pcf_ws      = (um_glm_n2560_RAL3p3_MAR_Pcf_ua**2 + um_glm_n2560_RAL3p3_MAR_Pcf_va**2) ** 0.5
ERA5_MAR_Pcf_ws                     = (ERA5_MAR_Pcf_ua**2 + ERA5_MAR_Pcf_va**2) ** 0.5
print("Done")

#%%
# PDFs
bins = np.arange(0, 15, 0.2)
## Wind Speed
icon_ngc4008_MAR_Pcf_ws_PDF             = np.histogram(icon_ngc4008_MAR_Pcf_ws, bins=bins, density=True)
icon_d3hp003_MAR_Pcf_ws_PDF             = np.histogram(icon_d3hp003_MAR_Pcf_ws, bins=bins, density=True)            
ifs_tco3999_ng5_rcbmf_cf_MAR_Pcf_ws_PDF = np.histogram(ifs_tco3999_ng5_rcbmf_cf_MAR_Pcf_ws, bins=bins, density=True)
nicam_gl11_MAR_Pcf_ws_PDF               = np.histogram(nicam_gl11_MAR_Pcf_ws, bins=bins, density=True)              
um_glm_n2560_RAL3p3_MAR_Pcf_ws_PDF      = np.histogram(um_glm_n2560_RAL3p3_MAR_Pcf_ws, bins=bins, density=True)     
ERA5_MAR_Pcf_ws_PDF                     = np.histogram(ERA5_MAR_Pcf_ws, bins=bins, density=True)                    
# Sea Level Pressure
icon_ngc4008_MAR_Pcf_slp = icon_ngc4008_MAR_Pcf.pres_msl
icon_d3hp003_MAR_Pcf_slp = icon_d3hp003_MAR_Pcf.psl
ifs_tco3999_ng5_rcbmf_cf_MAR_Pcf_slp = ifs_tco3999_ng5_rcbmf_cf_MAR_Pcf.psl
ERA5_MAR_Pcf_slp = ERA5_MAR_Pcf.msl
# Meridional Mean: March
MERID_BINS = int(RIGHT-LEFT)
hist_opts = dict(bins=MERID_BINS, range=(LEFT, RIGHT))
def calc_merid_mean(variable, **kwargs):
    """Compute a zonal-mean (along `lon`) for multi-dimensional input."""
    counts_per_bin, bin_edges = np.histogram(variable.lon, **hist_opts)

    def _compute_varsum(var, **kwargs):
        """Helper function to compute histogram for a single timestep."""
        varsum_per_bin, _ = np.histogram(variable.lon, weights=var, **kwargs)
        return varsum_per_bin

    # For more information see:
    # https://docs.xarray.dev/en/stable/generated/xarray.apply_ufunc.html
    varsum = xr.apply_ufunc(
        _compute_varsum,  # function to map
        variable,  # variables to loop over
        kwargs=hist_opts,  # keyword arguments passed to the function
        input_core_dims=[["cell"]],  # dimensions that should not be kept
        # Description of the output dataset
        dask="parallelized",
        vectorize=True,
        output_core_dims=[("lon",)],
        dask_gufunc_kwargs={
            "output_sizes": {"lon": hist_opts["bins"]},
        },
        output_dtypes=["f8"],
    )

    return varsum / counts_per_bin, bin_edges
icon_ngc4008_MAR_Pcf_slp_MM, lon_bins = calc_merid_mean(icon_ngc4008_MAR_Pcf_slp, **hist_opts)
icon_d3hp003_MAR_Pcf_slp_MM, lon_bins = calc_merid_mean(icon_d3hp003_MAR_Pcf_slp, **hist_opts)
ifs_tco3999_ng5_rcbmf_cf_MAR_Pcf_slp_MM, lon_bins = calc_merid_mean(ifs_tco3999_ng5_rcbmf_cf_MAR_Pcf_slp, **hist_opts)
ERA5_MAR_Pcf_slp_MM, lon_bins = calc_merid_mean(ERA5_MAR_Pcf_slp, **hist_opts)

#%%
# Statistics
icon_ngc4008_MAR_Pcf_ws_mean = icon_ngc4008_MAR_Pcf_ws.mean().compute()
icon_ngc4008_MAR_Pcf_ws_sigma = np.std(icon_ngc4008_MAR_Pcf_ws).compute()
icon_d3hp003_MAR_Pcf_ws_mean = icon_d3hp003_MAR_Pcf_ws.mean().compute()           
icon_d3hp003_MAR_Pcf_ws_sigma = np.std(icon_d3hp003_MAR_Pcf_ws).compute()
ifs_tco3999_ng5_rcbmf_cf_MAR_Pcf_ws_mean = ifs_tco3999_ng5_rcbmf_cf_MAR_Pcf_ws.mean().compute()
ifs_tco3999_ng5_rcbmf_cf_MAR_Pcf_ws_sigma = np.std(ifs_tco3999_ng5_rcbmf_cf_MAR_Pcf_ws).compute()
nicam_gl11_MAR_Pcf_ws_mean = nicam_gl11_MAR_Pcf_ws.mean().compute()
nicam_gl11_MAR_Pcf_ws_sigma = np.std(nicam_gl11_MAR_Pcf_ws).compute()
um_glm_n2560_RAL3p3_MAR_Pcf_ws_mean = um_glm_n2560_RAL3p3_MAR_Pcf_ws.mean().compute()
um_glm_n2560_RAL3p3_MAR_Pcf_ws_sigma = np.std(um_glm_n2560_RAL3p3_MAR_Pcf_ws).compute()
ERA5_MAR_Pcf_ws_mean = ERA5_MAR_Pcf_ws.mean().compute()
ERA5_MAR_Pcf_ws_sigma = np.std(ERA5_MAR_Pcf_ws).compute()

#%%
## Plotting
SIZE = 21
plt.rcParams["axes.labelsize"] = SIZE
plt.rcParams["legend.fontsize"] = SIZE
plt.rcParams["xtick.labelsize"] = SIZE
plt.rcParams["ytick.labelsize"] = SIZE
plt.rcParams["font.size"] = SIZE
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.major.size'] = 6
plt.rcParams['ytick.major.size'] = 6
# Enable LaTeX font
plt.rc('text', usetex=False)
plt.rc('font', family='sans-serif')

#%%
fig = plt.figure(figsize=(13, 18), facecolor="w", edgecolor="k")
#fig.suptitle(f"{win_exp}: Surface Wind Speed \n {-BOTTOM}°S to {TOP}°N and {LEFT}°E to {RIGHT}°E", y=1.05)

G = gridspec.GridSpec(3, 2, width_ratios=[9, 1], height_ratios=[1, 1, 1], wspace=0.45, hspace=0.8)

#####################################################################################################################
BUCHSTABI = SIZE + 5
#####################################################################################################################

ax1 = plt.subplot(G[0, 0], projection=ccrs.PlateCarree(central_longitude=180))
ax1.set_extent([120, 290, -30, 30], crs=ccrs.PlateCarree())

ax1.coastlines()
# Add degree values to x and y ticks
gl = ax1.gridlines(draw_labels=True, xlocs=[150, -100], ylocs=[-10, 10])
gl.xlabel_style = {'size': SIZE-5}  
gl.ylabel_style = {'size': SIZE-5} 
gl.top_labels = False
gl.right_labels = False

ax1.add_patch(Rectangle((-30,-10),110,20,edgecolor='black', facecolor='None', lw=5, alpha=1, zorder=10))

ax1.text(-0.22, 1.2, '(a)', fontsize=BUCHSTABI, transform=ax1.transAxes)
pos = ax1.get_position()
ax1.set_position([pos.x0, pos.y0 - 0.06, pos.width, pos.height])

############################################################################################################
OFFSET = -0.08
VERT_LINE = 0.01
OFFSET2 = -0.075
LW = 3
MS = 8

XLOW = 0
XHIGH = 10
UPPER = 0.7
LOWER = 0
############################################################################################################

ax1 = plt.subplot(G[1,0])

#ax1.plot(icon_ngc4008_MAR_Pcf_ws_PDF[1][1:], icon_ngc4008_MAR_Pcf_ws_PDF[0], label='icon ngc4008', ls='solid', lw=LW, zorder=10, color='C0')
#ax1.scatter(icon_ngc4008_MAR_Pcf_ws_mean, OFFSET+0.06, marker='o', clip_on=False, zorder=10, s=40, lw=3, color='C0')
#ax1.plot([icon_ngc4008_MAR_Pcf_ws_mean-icon_ngc4008_MAR_Pcf_ws_sigma, icon_ngc4008_MAR_Pcf_ws_mean+icon_ngc4008_MAR_Pcf_ws_sigma], [OFFSET+0.06, OFFSET+0.06], ls='-', clip_on=False, color='C0')
#ax1.plot([icon_ngc4008_MAR_Pcf_ws_mean-icon_ngc4008_MAR_Pcf_ws_sigma, icon_ngc4008_MAR_Pcf_ws_mean-icon_ngc4008_MAR_Pcf_ws_sigma], [OFFSET+0.06-VERT_LINE, OFFSET+0.06+VERT_LINE], ls='-', clip_on=False, color='C0')
#ax1.plot([icon_ngc4008_MAR_Pcf_ws_mean+icon_ngc4008_MAR_Pcf_ws_sigma, icon_ngc4008_MAR_Pcf_ws_mean+icon_ngc4008_MAR_Pcf_ws_sigma], [OFFSET+0.06-VERT_LINE, OFFSET+0.06+VERT_LINE], ls='-', clip_on=False, color='C0')

ax1.plot(ERA5_MAR_Pcf_ws_PDF[1][1:], ERA5_MAR_Pcf_ws_PDF[0], label='ERA5', lw=LW, color='black')
ax1.scatter(ERA5_MAR_Pcf_ws_mean, OFFSET-0.09, marker='o', clip_on=False, zorder=10, s=40, lw=3, color='black')
ax1.plot([ERA5_MAR_Pcf_ws_mean-ERA5_MAR_Pcf_ws_sigma, ERA5_MAR_Pcf_ws_mean+ERA5_MAR_Pcf_ws_sigma], [OFFSET-0.09, OFFSET-0.09], ls='-', clip_on=False, color='black')
ax1.plot([ERA5_MAR_Pcf_ws_mean-ERA5_MAR_Pcf_ws_sigma, ERA5_MAR_Pcf_ws_mean-ERA5_MAR_Pcf_ws_sigma], [OFFSET-0.09-VERT_LINE, OFFSET-0.09+VERT_LINE], ls='-', clip_on=False, color='black')
ax1.plot([ERA5_MAR_Pcf_ws_mean+ERA5_MAR_Pcf_ws_sigma, ERA5_MAR_Pcf_ws_mean+ERA5_MAR_Pcf_ws_sigma], [OFFSET-0.09-VERT_LINE, OFFSET-0.09+VERT_LINE], ls='-', clip_on=False, color='black')


ax1.plot(icon_d3hp003_MAR_Pcf_ws_PDF[1][1:], icon_d3hp003_MAR_Pcf_ws_PDF[0], label='icon d3hp003', ls='solid', lw=LW, color='C0')
ax1.scatter(icon_d3hp003_MAR_Pcf_ws_mean, OFFSET, marker='o', clip_on=False, zorder=10, s=40, lw=3, color='C0')
ax1.plot([icon_d3hp003_MAR_Pcf_ws_mean-icon_d3hp003_MAR_Pcf_ws_sigma, icon_d3hp003_MAR_Pcf_ws_mean+icon_d3hp003_MAR_Pcf_ws_sigma], [OFFSET, OFFSET], ls='-', clip_on=False, color='C0')
ax1.plot([icon_d3hp003_MAR_Pcf_ws_mean-icon_d3hp003_MAR_Pcf_ws_sigma, icon_d3hp003_MAR_Pcf_ws_mean-icon_d3hp003_MAR_Pcf_ws_sigma], [OFFSET-VERT_LINE, OFFSET+VERT_LINE], ls='-', clip_on=False, color='C0')
ax1.plot([icon_d3hp003_MAR_Pcf_ws_mean+icon_d3hp003_MAR_Pcf_ws_sigma, icon_d3hp003_MAR_Pcf_ws_mean+icon_d3hp003_MAR_Pcf_ws_sigma], [OFFSET-VERT_LINE, OFFSET+VERT_LINE], ls='-', clip_on=False, color='C0')

ax1.plot(ifs_tco3999_ng5_rcbmf_cf_MAR_Pcf_ws_PDF[1][1:], ifs_tco3999_ng5_rcbmf_cf_MAR_Pcf_ws_PDF[0], label='ifs tco3999 ng5 rcbmf cf', ls='solid', lw=LW, color='C1')
ax1.scatter(ifs_tco3999_ng5_rcbmf_cf_MAR_Pcf_ws_mean, OFFSET+0.03, marker='o', clip_on=False, zorder=10, s=40, lw=3, color='C1')
ax1.plot([ifs_tco3999_ng5_rcbmf_cf_MAR_Pcf_ws_mean-ifs_tco3999_ng5_rcbmf_cf_MAR_Pcf_ws_sigma, ifs_tco3999_ng5_rcbmf_cf_MAR_Pcf_ws_mean+ifs_tco3999_ng5_rcbmf_cf_MAR_Pcf_ws_sigma], [OFFSET+0.03, OFFSET+0.03], ls='-', clip_on=False, color='C1')
ax1.plot([ifs_tco3999_ng5_rcbmf_cf_MAR_Pcf_ws_mean-ifs_tco3999_ng5_rcbmf_cf_MAR_Pcf_ws_sigma, ifs_tco3999_ng5_rcbmf_cf_MAR_Pcf_ws_mean-ifs_tco3999_ng5_rcbmf_cf_MAR_Pcf_ws_sigma], [OFFSET+0.03-VERT_LINE, OFFSET+0.03+VERT_LINE], ls='-', clip_on=False, color='C1')
ax1.plot([ifs_tco3999_ng5_rcbmf_cf_MAR_Pcf_ws_mean+ifs_tco3999_ng5_rcbmf_cf_MAR_Pcf_ws_sigma, ifs_tco3999_ng5_rcbmf_cf_MAR_Pcf_ws_mean+ifs_tco3999_ng5_rcbmf_cf_MAR_Pcf_ws_sigma], [OFFSET+0.03-VERT_LINE, OFFSET+0.03+VERT_LINE], ls='-', clip_on=False, color='C1')

ax1.plot(nicam_gl11_MAR_Pcf_ws_PDF[1][1:], nicam_gl11_MAR_Pcf_ws_PDF[0], label='nicam', lw=LW, color='C2')
ax1.scatter(nicam_gl11_MAR_Pcf_ws_mean, OFFSET-0.03, marker='o', clip_on=False, zorder=10, s=40, lw=3, color='C2')
ax1.plot([nicam_gl11_MAR_Pcf_ws_mean-nicam_gl11_MAR_Pcf_ws_sigma, nicam_gl11_MAR_Pcf_ws_mean+nicam_gl11_MAR_Pcf_ws_sigma], [OFFSET-0.03, OFFSET-0.03], ls='-', clip_on=False, color='C2')
ax1.plot([nicam_gl11_MAR_Pcf_ws_mean-nicam_gl11_MAR_Pcf_ws_sigma, nicam_gl11_MAR_Pcf_ws_mean-nicam_gl11_MAR_Pcf_ws_sigma], [OFFSET-0.03-VERT_LINE, OFFSET-0.03+VERT_LINE], ls='-', clip_on=False, color='C2')
ax1.plot([nicam_gl11_MAR_Pcf_ws_mean+nicam_gl11_MAR_Pcf_ws_sigma, nicam_gl11_MAR_Pcf_ws_mean+nicam_gl11_MAR_Pcf_ws_sigma], [OFFSET-0.03-VERT_LINE, OFFSET-0.03+VERT_LINE], ls='-', clip_on=False, color='C2')

ax1.plot(um_glm_n2560_RAL3p3_MAR_Pcf_ws_PDF[1][1:], um_glm_n2560_RAL3p3_MAR_Pcf_ws_PDF[0], label='um glm n2560 RAL3p3', lw=LW, color='C3')
ax1.scatter(um_glm_n2560_RAL3p3_MAR_Pcf_ws_mean, OFFSET-0.06, marker='o', clip_on=False, zorder=10, s=40, lw=3, color='C3')
ax1.plot([um_glm_n2560_RAL3p3_MAR_Pcf_ws_mean-um_glm_n2560_RAL3p3_MAR_Pcf_ws_sigma, um_glm_n2560_RAL3p3_MAR_Pcf_ws_mean+um_glm_n2560_RAL3p3_MAR_Pcf_ws_sigma], [OFFSET-0.06, OFFSET-0.06], ls='-', clip_on=False, color='C3')
ax1.plot([um_glm_n2560_RAL3p3_MAR_Pcf_ws_mean-um_glm_n2560_RAL3p3_MAR_Pcf_ws_sigma, um_glm_n2560_RAL3p3_MAR_Pcf_ws_mean-um_glm_n2560_RAL3p3_MAR_Pcf_ws_sigma], [OFFSET-0.06-VERT_LINE, OFFSET-0.06+VERT_LINE], ls='-', clip_on=False, color='C3')
ax1.plot([um_glm_n2560_RAL3p3_MAR_Pcf_ws_mean+um_glm_n2560_RAL3p3_MAR_Pcf_ws_sigma, um_glm_n2560_RAL3p3_MAR_Pcf_ws_mean+um_glm_n2560_RAL3p3_MAR_Pcf_ws_sigma], [OFFSET-0.06-VERT_LINE, OFFSET-0.06+VERT_LINE], ls='-', clip_on=False, color='C3')

#ax1.set_title(f"March")
ax1.set_ylabel(r"PDF")
ax1.set_xlabel("Surface Wind Speed / ms$^{-1}$")

ax1.set_xlim(XLOW, XHIGH)
ax1.set_ylim(LOWER, UPPER)
ax1.set_xticks([0,2,4,6,8,10])
#ax1.set_xticklabels([f"{LEFT}",f"{int(LEFT+abs(LEFT-RIGHT)/2)}",f"{RIGHT}",])

ax1.spines[["left"]].set_position(("outward", 20))
ax1.spines[["bottom"]].set_position(("outward", 60))
ax1.spines[["right", "top"]].set_visible(False)

ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -2.3), fancybox=True, shadow=False, ncol=2)
ax1.text(-0.22, 1.2, '(b)', fontsize=BUCHSTABI, transform=ax1.transAxes)

YLOW = 1013
YTOP = 1008

############################################################################################################
############################################################################################################

ax1 = plt.subplot(G[2, 0])

#(icon_ngc4008_MAR_Pcf_slp_MM/100).plot(ax=ax1, label="icon ngc4008", ls='solid', lw=LW, color='C0')
(icon_d3hp003_MAR_Pcf_slp_MM/100).plot(ax=ax1, label="icon d3hp003", ls='solid', lw=LW, color='C0')
(ifs_tco3999_ng5_rcbmf_cf_MAR_Pcf_slp_MM/100).plot(ax=ax1, label="ifs tco3999 ng5 rcbmf cf", ls='solid', lw=LW, color='C1')
(ERA5_MAR_Pcf_slp_MM/100).plot(ax=ax1, label="ERA5", ls='solid', lw=LW, color='black')

#ax1.plot(0, np.abs(icon_ngc4008_MAR_Pcf_slp_MM[0])/100, marker='o', clip_on=False, ms=MS, color='C0')
#ax1.plot(MERID_BINS-0.5, np.abs(icon_ngc4008_MAR_Pcf_slp_MM[-1])/100, marker='o', clip_on=False, ms=MS, color='C0')

ax1.plot(0, np.abs(icon_d3hp003_MAR_Pcf_slp_MM[0])/100, marker='o', clip_on=False, ms=MS, color='C0')
ax1.plot(MERID_BINS-0.5, np.abs(icon_d3hp003_MAR_Pcf_slp_MM[-1])/100,  marker='o', clip_on=False, ms=MS, color='C0')

ax1.plot(0, np.abs(ifs_tco3999_ng5_rcbmf_cf_MAR_Pcf_slp_MM[0])/100, marker='o', clip_on=False, ms=MS, color='C1')
ax1.plot(MERID_BINS-0.5, np.abs(ifs_tco3999_ng5_rcbmf_cf_MAR_Pcf_slp_MM[-1])/100, marker='o', clip_on=False, ms=MS, color='C1')

ax1.plot(0, np.abs(ERA5_MAR_Pcf_slp_MM[0])/100, marker='o', clip_on=False, ms=MS, color='black')
ax1.plot(MERID_BINS-0.5, np.abs(ERA5_MAR_Pcf_slp_MM[-1])/100, marker='o', clip_on=False, ms=MS, color='black')

#ax1.axhline(0, zorder=1, lw=0.5, color="grey")

ax1.set_title("", fontsize=SIZE)
ax1.set_ylabel(r"Surface Pressure / hPa")
ax1.set_xlabel("Longitude / deg")
ax1.set_xlim(0, MERID_BINS)
ax1.set_xticks([0, MERID_BINS])
ax1.set_xticklabels([f"{LEFT}°E", f"100°W"])
ax1.set_ylim(YTOP, YLOW)
ax1.set_yticks([1008, 1009, 1010, 1011, 1012, 1013])

ax1.spines[["left", "bottom"]].set_position(("outward", 20))
ax1.spines[["right", "top"]].set_visible(False)
#ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.3), fancybox=True, shadow=False, ncol=5)
ax1.text(-0.22, 1.2, '(c)', fontsize=BUCHSTABI, transform=ax1.transAxes)

############################################################################################################
############################################################################################################

ax2 = plt.subplot(G[2, 1])

#ax2.plot(0, np.abs(icon_ngc4008_MAR_Pcf_slp_MM[0]-icon_ngc4008_MAR_Pcf_slp_MM[-1])/100, label="icon ngc4008", marker='o', ms=MS, zorder=10, color='C0')
ax2.plot(0, np.abs(icon_d3hp003_MAR_Pcf_slp_MM[0]-icon_d3hp003_MAR_Pcf_slp_MM[-1])/100, label="icon d3hp003", marker='o', ms=MS, color='C0')
ax2.plot(0, np.abs(ifs_tco3999_ng5_rcbmf_cf_MAR_Pcf_slp_MM[0]-ifs_tco3999_ng5_rcbmf_cf_MAR_Pcf_slp_MM[-1])/100, label="ifs tco3999 ng5 rcbmf cf", marker='o', ms=MS, color='C1')
ax2.plot(0, np.abs(ERA5_MAR_Pcf_slp_MM[0]-ERA5_MAR_Pcf_slp_MM[-1])/100, label="ERA5", marker='o', ms=MS, color='black')

#ax2.axhline(0, zorder=1, lw=0.5, color="grey")

ax2.set_title("", fontsize=SIZE)
ax2.set_ylabel(r"$\Delta p_{250^\circ}^{185^\circ}$ / hPa")
ax2.set_xlabel("")
#ax2.set_xlim(0, MERID_BINS)
#ax2.set_xticks([0, MERID_BINS / 2, MERID_BINS])
ax2.set_xticklabels([])
ax2.set_xticks([])
ax2.set_ylim(0, 4)
ax2.set_yticks([0,1,2,3,4])


ax2.spines[["bottom"]].set_position(("outward", 20))
ax2.spines[["right", "top", "bottom"]].set_visible(False)
#ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.32), fancybox=True, shadow=False, ncol=5)

ax2.text(-1.4, 1.2, '(d)', fontsize=BUCHSTABI, transform=ax2.transAxes)
#ax2.text(2, 1.1, '(d)', fontsize=BUCHSTABI, transform=ax2.transAxes, color='white')


filename = f"figs/fig_03.png"
plt.savefig(filename, facecolor='white', bbox_inches='tight', dpi=100)
plt.show()


# %%
