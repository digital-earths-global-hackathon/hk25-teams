#%%
# Preamble
import intake
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
ifs_tco3999_ng5_rcbmf = (
    cat["ifs_tco3999-ng5_rcbmf"](dim='3D', zoom=ZOOM)
    .to_dask()
)
ifs_tco3999_ng5_rcbmf = egh.attach_coords(ifs_tco3999_ng5_rcbmf)
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
ERA5;
print("Gathered all data")

#%%
# Process Data
def east_pacific(ds):
    return (ds.lat > BOTTOM) & (ds.lat < TOP) & (ds.lon > ePcf_LEFT) & (ds.lon < ePcf_RIGHT)
    
def west_pacific(ds):
    return (ds.lat > BOTTOM) & (ds.lat < TOP) & (ds.lon > wPcf_LEFT) & (ds.lon < wPcf_RIGHT)

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

#g_at_equator = 9.78
#R_d = 287.05

#%%
# Time Filter
### ICON
icon_d3hp003_MAR = icon_d3hp003.sel(time=slice(t1,t2)).mean(dim='time') #.sel(time=t1)
icon_ngc4008_MAR = icon_ngc4008.sel(time=slice(t1,t2)).mean(dim='time') #.sel(time=t1) #.sel(time=slice(t1,t2)).mean(dim='time')
### IFS
ifs_tco3999_ng5_rcbmf_cf_MAR = ifs_tco3999_ng5_rcbmf_cf.sel(time=slice(t1,t2)).mean(dim='time') #.sel(time=t1).mean(dim='time')
#ifs_tco3999_ng5_rcbmf_MAR = ifs_tco3999_ng5_rcbmf.sel(time=slice(t1,t2)).mean(dim='time') #.sel(time=t1).mean(dim='time')
### NICAM
nicam_gl11_MAR = nicam_gl11.sel(time=slice(t1,t2)).mean(dim='time') #.sel(time=t1).mean(dim='time')
### Unified Model
um_glm_n2560_RAL3p3_MAR = um_glm_n2560_RAL3p3.sel(time=slice(t1,t2)).mean(dim='time') #.sel(time=t1).mean(dim='time')
### ERA5
ERA5_MAR = ERA5.sel(time=slice(t1,t2)).mean(dim='time') #.sel(time=t1).mean(dim='time')
print("Applied all Time Filter")

#%%
# Spatial Filter
### ICON
##### icon_ngc4008
icon_ngc4008_MAR_wPcf = icon_ngc4008_MAR.where(west_pacific(ocean(icon_ngc4008_MAR)), drop=True)
icon_ngc4008_MAR_ePcf = icon_ngc4008_MAR.where(east_pacific(ocean(icon_ngc4008_MAR)), drop=True)
icon_ngc4008_MAR_wPcf_zg = icon_ngc4008_MAR_wPcf.zg.mean(dim='cell')
##### icon_d3hp003
icon_d3hp003_MAR_wPcf = icon_d3hp003_MAR.where(west_pacific(ocean(icon_ngc4008_MAR)), drop=True)
icon_d3hp003_MAR_ePcf = icon_d3hp003_MAR.where(east_pacific(ocean(icon_ngc4008_MAR)), drop=True)
### IFS
##### rcbmf cf
ifs_tco3999_ng5_rcbmf_cf_MAR_wPcf = ifs_tco3999_ng5_rcbmf_cf_MAR.where(west_pacific(ifs_tco3999_ng5_rcbmf_cf_MAR).compute(), drop=True)
ifs_tco3999_ng5_rcbmf_cf_MAR_ePcf = ifs_tco3999_ng5_rcbmf_cf_MAR.where(east_pacific(ifs_tco3999_ng5_rcbmf_cf_MAR).compute(), drop=True)
##### rcbmf
#ifs_tco3999_ng5_rcbmf_MAR_wPcf = ifs_tco3999_ng5_rcbmf_MAR.where(west_pacific(ifs_tco3999_ng5_rcbmf_MAR).compute(), drop=True)
#ifs_tco3999_ng5_rcbmf_MAR_ePcf = ifs_tco3999_ng5_rcbmf_MAR.where(east_pacific(ifs_tco3999_ng5_rcbmf_MAR).compute(), drop=True)
#### NICAM
nicam_gl11_MAR_wPcf = nicam_gl11_MAR.where(west_pacific(nicam_gl11_MAR).compute(), drop=True)
nicam_gl11_MAR_ePcf = nicam_gl11_MAR.where(east_pacific(nicam_gl11_MAR).compute(), drop=True)
### Unified Model
um_glm_n2560_RAL3p3_MAR_wPcf = um_glm_n2560_RAL3p3_MAR.where(west_pacific(um_glm_n2560_RAL3p3_MAR).compute(), drop=True)
um_glm_n2560_RAL3p3_MAR_ePcf = um_glm_n2560_RAL3p3_MAR.where(east_pacific(um_glm_n2560_RAL3p3_MAR).compute(), drop=True)
### ERA5
ERA5_MAR_wPcf = ERA5_MAR.where(west_pacific(ERA5_MAR).compute(), drop=True)
ERA5_MAR_ePcf = ERA5_MAR.where(east_pacific(ERA5_MAR).compute(), drop=True)
print("Applied all Spatial Filter")

#%%
# Compute Density
R_d = 287.05  # J/(kg·K)
### ICON
##### icon_ngc4008
icon_ngc4008_MAR_wPcf_Tv = icon_ngc4008_MAR_wPcf.ta * (1 + 0.61 * icon_ngc4008_MAR_wPcf.hus)
icon_ngc4008_MAR_ePcf_Tv = icon_ngc4008_MAR_ePcf.ta * (1 + 0.61 * icon_ngc4008_MAR_ePcf.hus)
icon_ngc4008_MAR_wPcf_rho = (icon_ngc4008_MAR_wPcf.pfull / (R_d * icon_ngc4008_MAR_wPcf_Tv)).mean(dim='cell')
icon_ngc4008_MAR_ePcf_rho = (icon_ngc4008_MAR_ePcf.pfull / (R_d * icon_ngc4008_MAR_ePcf_Tv)).mean(dim='cell')
##### icon_d3hp003
icon_d3hp003_MAR_wPcf_Tv = icon_d3hp003_MAR_wPcf.ta * (1 + 0.61 * icon_d3hp003_MAR_wPcf.hus)
icon_d3hp003_MAR_ePcf_Tv = icon_d3hp003_MAR_ePcf.ta * (1 + 0.61 * icon_d3hp003_MAR_ePcf.hus)
icon_d3hp003_MAR_wPcf_rho = (icon_d3hp003_MAR_wPcf.pressure / (R_d * icon_d3hp003_MAR_wPcf_Tv)).mean(dim='cell')
icon_d3hp003_MAR_ePcf_rho = (icon_d3hp003_MAR_ePcf.pressure / (R_d * icon_d3hp003_MAR_ePcf_Tv)).mean(dim='cell')
### IFS
##### rcbmf cf
ifs_tco3999_ng5_rcbmf_cf_MAR_wPcf_Tv = ifs_tco3999_ng5_rcbmf_cf_MAR_wPcf.ta * (1 + 0.61 * ifs_tco3999_ng5_rcbmf_cf_MAR_wPcf.hus)
ifs_tco3999_ng5_rcbmf_cf_MAR_ePcf_Tv = ifs_tco3999_ng5_rcbmf_cf_MAR_ePcf.ta * (1 + 0.61 * ifs_tco3999_ng5_rcbmf_cf_MAR_ePcf.hus)
ifs_tco3999_ng5_rcbmf_cf_MAR_wPcf_rho = ((ifs_tco3999_ng5_rcbmf_cf_MAR_wPcf.level*100) / (R_d * ifs_tco3999_ng5_rcbmf_cf_MAR_wPcf_Tv)).mean(dim='cell')
ifs_tco3999_ng5_rcbmf_cf_MAR_ePcf_rho = ((ifs_tco3999_ng5_rcbmf_cf_MAR_ePcf.level*100) / (R_d * ifs_tco3999_ng5_rcbmf_cf_MAR_ePcf_Tv)).mean(dim='cell')
##### rcbmf
#ifs_tco3999_ng5_rcbmf_MAR_wPcf_Tv = ifs_tco3999_ng5_rcbmf_MAR_wPcf.t * (1 + 0.61 * ifs_tco3999_ng5_rcbmf_MAR_wPcf.q)
#ifs_tco3999_ng5_rcbmf_MAR_ePcf_Tv = ifs_tco3999_ng5_rcbmf_MAR_ePcf.t * (1 + 0.61 * ifs_tco3999_ng5_rcbmf_MAR_ePcf.q)
#ifs_tco3999_ng5_rcbmf_MAR_wPcf_rho = ((ifs_tco3999_ng5_rcbmf_MAR_wPcf.level*100) / (R_d * ifs_tco3999_ng5_rcbmf_MAR_wPcf_Tv)).mean(dim='value')
#ifs_tco3999_ng5_rcbmf_MAR_ePcf_rho = ((ifs_tco3999_ng5_rcbmf_MAR_ePcf.level*100) / (R_d * ifs_tco3999_ng5_rcbmf_MAR_ePcf_Tv)).mean(dim='value')
### NICAM
nicam_gl11_MAR_wPcf_Tv = nicam_gl11_MAR_wPcf.ta * (1 + 0.61 * nicam_gl11_MAR_wPcf.hus)
nicam_gl11_MAR_ePcf_Tv = nicam_gl11_MAR_ePcf.ta * (1 + 0.61 * nicam_gl11_MAR_ePcf.hus)
nicam_gl11_MAR_wPcf_rho = ((nicam_gl11_MAR_wPcf.lev*100) / (R_d * nicam_gl11_MAR_wPcf_Tv)).mean(dim='cell')
nicam_gl11_MAR_ePcf_rho = ((nicam_gl11_MAR_wPcf.lev*100) / (R_d * nicam_gl11_MAR_ePcf_Tv)).mean(dim='cell')
### Unified Model
um_glm_n2560_RAL3p3_MAR_wPcf_Tv = um_glm_n2560_RAL3p3_MAR_wPcf.ta * (1 + 0.61 * um_glm_n2560_RAL3p3_MAR_wPcf.hus)
um_glm_n2560_RAL3p3_MAR_ePcf_Tv = um_glm_n2560_RAL3p3_MAR_ePcf.ta * (1 + 0.61 * um_glm_n2560_RAL3p3_MAR_ePcf.hus)
um_glm_n2560_RAL3p3_MAR_wPcf_rho = ((um_glm_n2560_RAL3p3_MAR_wPcf.pressure*100) / (R_d * um_glm_n2560_RAL3p3_MAR_wPcf_Tv)).mean(dim='cell')
um_glm_n2560_RAL3p3_MAR_ePcf_rho = ((um_glm_n2560_RAL3p3_MAR_ePcf.pressure*100) / (R_d * um_glm_n2560_RAL3p3_MAR_ePcf_Tv)).mean(dim='cell')
### ERA5
ERA5_MAR_wPcf_Tv = ERA5_MAR_wPcf.t * (1 + 0.61 * ERA5_MAR_wPcf.q)
ERA5_MAR_ePcf_Tv = ERA5_MAR_ePcf.t * (1 + 0.61 * ERA5_MAR_ePcf.q)
ERA5_MAR_wPcf_rho = ((ERA5_MAR_wPcf.level*100) / (R_d * ERA5_MAR_wPcf_Tv)).mean(dim='cell')
ERA5_MAR_ePcf_rho = ((ERA5_MAR_ePcf.level*100) / (R_d * ERA5_MAR_ePcf_Tv)).mean(dim='cell')
print("Computed Density")

#%%
# Plotting
SIZE = 55
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
print("Started Plotting right now.")

#%%
#fig = plt.figure(figsize=(20,20), facecolor="w", edgecolor="k")
#G = gridspec.GridSpec(2, 1, hspace=0.45, wspace=0.4, height_ratios=[1, 3])
fig = plt.figure(figsize=(22, 20), facecolor="w")

# Absolute manual control over position [left, bottom, width, height] in figure coordinates (0–1)
top_height = 0.35
bottom_height = 0.7
v_gap = -0.3 # Try increasing this for more space

# Top plot
ax1 = fig.add_axes(
    [0.08, 1 - top_height - v_gap, 0.88, top_height],
    projection=ccrs.PlateCarree(central_longitude=180)
)

# Bottom plot
ax = fig.add_axes(
    [0.08, 0.08, 0.88, bottom_height]
)

BUCHSTABI = 65

#####################################################################################################################
### Top Plot
#####################################################################################################################

#ax1 = plt.subplot(G[0, 0], projection=ccrs.PlateCarree(central_longitude=180))
ax1.set_extent([120, 290, -30, 30], crs=ccrs.PlateCarree())

ax1.coastlines()
# Add degree values to x and y ticks
gl = ax1.gridlines(draw_labels=True, xlocs=[150, 170, -120, -100], ylocs=[-10, 10])
gl.xlabel_style = {'size': SIZE-5}  
gl.ylabel_style = {'size': SIZE-5} 
gl.top_labels = False
gl.right_labels = False

ax1.add_patch(Rectangle((-30,-10),20,20,edgecolor='black', facecolor='None', lw=12, alpha=1, zorder=11))
ax1.add_patch(Rectangle((60,-10),20,20,edgecolor='black', facecolor='None', lw=12, alpha=1, zorder=11))
ax1.add_patch(Rectangle((-30.5,-10.5),111,21,edgecolor='pink', facecolor='None', lw=15, alpha=1, zorder=10))

ax1.text(-20, 30, "Western Pacific", ha="center", va="bottom", fontsize=SIZE+5, color="black", zorder=11)
ax1.text(70, 30, "Eastern Pacific", ha="center", va="bottom", fontsize=SIZE+5, color="black", zorder=11)

ax1.text(-0.15, 1.2, '(a)', fontsize=BUCHSTABI, transform=ax1.transAxes)

#####################################################################################################################
### Bottom Plot
#####################################################################################################################

XLOW  = -25
XHIGH = 25
YLOW  = 1000
YHIGH = 150

LW=6
NUM_TICKS = 8

#ax = plt.subplot(G[1, 0])
ax.axvline(0, color='grey', ls='dotted', lw=2)

ax.plot((ERA5_MAR_wPcf_rho-ERA5_MAR_ePcf_rho)*1000, ERA5_MAR_wPcf.level, label='ERA5', color='black', lw=LW)
ax.plot((icon_d3hp003_MAR_wPcf_rho-icon_d3hp003_MAR_ePcf_rho)*1000, icon_d3hp003_MAR_wPcf_rho.pressure/100, label='icon d3hp003', ls='solid', lw=LW)
ax.plot((ifs_tco3999_ng5_rcbmf_cf_MAR_wPcf_rho-ifs_tco3999_ng5_rcbmf_cf_MAR_ePcf_rho)*1000, ifs_tco3999_ng5_rcbmf_cf_MAR_ePcf_rho.level, label='ifs rcbmf cf', ls='solid', lw=LW)
ax.plot((nicam_gl11_MAR_wPcf_rho-nicam_gl11_MAR_ePcf_rho)*1000, nicam_gl11_MAR_wPcf_rho.lev, label='nicam gl11', ls='solid', lw=LW)
ax.plot((um_glm_n2560_RAL3p3_MAR_wPcf_rho-um_glm_n2560_RAL3p3_MAR_ePcf_rho)*1000, um_glm_n2560_RAL3p3_MAR_wPcf.pressure, label='um glm n2560 RAL3p3', ls='solid', lw=LW)

ax.set_title(r"West. P.$-$East. P.", pad=30)
#ax.set_ylabel(r"Height / m ")
ax.set_xlabel(r"$\Delta \rho$ / $\cdot10^{-3}$ kg$\cdot$m$^{-3}$")

#ax.set_xlim(XLOW, XHIGH)
#ax.set_yticks([0,2000,4000,6000,8000,10000,12000,14000])
ax.set_ylim(YLOW, YHIGH)

# Configure the left y-axis
ax.spines[["bottom", "left"]].set_position(("outward", 20))
ax.spines[["top", "right"]].set_visible(False)  # Hide both top and right spines

# Set up the left y-axis with label and tick labels
ax.set_ylabel("Pressure / hPa")  # Set the y-axis label on the left
ax.tick_params(axis="y", which="both", labelleft=True)  # Show tick labels on the left y-axis

# Annotate the subplot
ax.text(-0.15, 1.15, '(b)', fontsize=BUCHSTABI, transform=ax.transAxes, color='black')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), fancybox=True, shadow=False, ncol=2)


filename = f"figs/fig_02.png"
plt.savefig(filename, facecolor='white', bbox_inches='tight', dpi=100)

plt.show()

# %%
