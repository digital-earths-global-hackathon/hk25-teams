import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import easygems.healpix as egh
import healpy as hp
import seaborn as sns
sns.set_context('talk')
import pandas as pd
import intake
import pathlib
import sys
sys.path.append("../../src/")
import toolbox as toolbox

current_loc = 'EU'
cat = intake.open_catalog("https://digital-earths-global-hackathon.github.io/catalog/catalog.yaml")[current_loc]
ds = cat["icon_d3hp003"](zoom = 7).to_dask()
ds = egh.attach_coords(ds)
nside = egh.get_nside(ds)
ring_index = toolbox.nest2ring_index(ds, nside)

ds['time'] = ds['time'] - pd.Timedelta(days = 1)
ua_aug = ds['ua'].sel(time = slice('2020-08-01', '2020-08-31'))
va_aug = ds['va'].sel(time = slice('2020-08-01', '2020-08-31'))
wa_aug = ds.wa.sel(time = slice('2020-08-01', '2020-08-31'))

ua_aug = ua_aug.compute()
va_aug = va_aug.compute()
wa_aug = wa_aug.compute()

conv_aug = toolbox.compute_conv(ua_aug, va_aug, ring_index, nside)

atl = ((ds.lat <= 20) & (ds.lat >= -20) & (ds.lon >= 330) & (ds.lon <= 340)).values[ring_index] 
epac = ((ds.lat <= 20) & (ds.lat >= -20) & (ds.lon >= 235) & (ds.lon <= 255)).values[ring_index] 

epac_conv_aug = conv_aug.isel(cell = epac)
epac_wa_aug = wa_aug.isel(cell = ring_index).isel(cell = epac)
epac_va_aug = va_aug.isel(cell = ring_index).isel(cell = epac)

atl_conv_aug = conv_aug.isel(cell = atl)
atl_wa_aug = wa_aug.isel(cell = ring_index).isel(cell = atl)
atl_va_aug = va_aug.isel(cell = ring_index).isel(cell = atl)

fig, ax = plt.subplots(figsize = (16, 10), facecolor = 'white', ncols = 2, sharey = True)

quiver_skip = 8

ax[0].set_title('East Pacific')
im1 = ax[0].contourf(
    epac_conv_aug.lat.groupby('lat').mean(),
    epac_conv_aug.pressure/100, 
    epac_conv_aug.mean('time').groupby('lat').mean(),
    levels = np.linspace(-2.5e-5, 2.5e-5, 51),
    cmap = 'bwr'
)

q0 = ax[0].quiver(
    epac_wa_aug.lat.groupby('lat').mean()[::quiver_skip], 
    epac_va_aug.pressure/100, 
    epac_va_aug.mean('time').groupby('lat').mean()[:, ::quiver_skip], 
    epac_wa_aug.mean('time').groupby('lat').mean()[:, ::quiver_skip],
    scale = 2, 
    scale_units = 'xy',
    width = 0.004
)
ax[0].quiverkey(q0, X = 0.8, Y = 0.9715, U = 5, label = '5 m/s', labelpos = 'E')

ax[1].set_title('Atlantic')
im1 = ax[1].contourf(
    atl_conv_aug.lat.groupby('lat').mean(),
    atl_conv_aug.pressure/100, 
    atl_conv_aug.mean('time').groupby('lat').mean(),
    levels = np.linspace(-2.5e-5, 2.5e-5, 51),
    cmap = 'bwr'
)

q0 = ax[1].quiver(
    atl_wa_aug.lat.groupby('lat').mean()[::quiver_skip], 
    atl_va_aug.pressure/100, 
    atl_va_aug.mean('time').groupby('lat').mean()[:, ::quiver_skip], 
    atl_wa_aug.mean('time').groupby('lat').mean()[:, ::quiver_skip],
    scale = 2, 
    scale_units = 'xy',
    width = 0.004
)
ax[1].quiverkey(q0, X = 0.8, Y = 0.9715, U = 5, label = '5 m/s', labelpos = 'E')

ax[0].invert_yaxis()
ax[1].invert_yaxis()
ax[0].set_ylim([100000/100, 20000/100])
ax[1].set_ylim([100000/100, 20000/100])

ax[0].set_ylabel('pressure / hPa')
ax[0].set_xlabel('latitude / \N{DEGREE SIGN}N')
ax[1].set_xlabel('latitude / \N{DEGREE SIGN}N')

cbar = fig.colorbar(im1, ax=ax, orientation='horizontal', fraction=0.15, pad=0.13, aspect = 50, shrink = 0.6)
cbar.set_label(r'$\mathcal{C}$ / s$^{-1}$')
fig.savefig(
    'test.png',
    bbox_inches = 'tight'
)