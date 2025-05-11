#%%
import uxarray as ux
import easygems.healpix as eghp
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cf
import cmocean
# %%
import cartopy.crs as ccrs
# %%
import intake

# Hackathon data catalogs
cat_url = "https://digital-earths-global-hackathon.github.io/catalog/catalog.yaml"
cat = intake.open_catalog(cat_url).online
model_run = cat.icon_ngc4008
# %%
ds_coarsest = model_run().to_dask()
ds_fine = model_run(zoom=7).to_dask()
# %%
# %%
fig, ax = plt.subplots(
    subplot_kw={"projection": ccrs.PlateCarree()},
    figsize=(10, 5),
)

eghp.healpix_show(
    ds_coarsest.ts.isel(time=0),
    ax=ax,
    cmap=cmocean.cm.thermal,
)
ax.set_global()

# eghp.healpix_show(
#     ds_fine.tas.isel(time=0),
#     ax=ax,
#     cmap=cmocean.cm.thermal,
# )
# %%
uxds_coarsest = ux.UxDataset.from_healpix(ds_coarsest)

uxds_fine = ux.UxDataset.from_healpix(ds_fine)
uxds_fine
# %%
boulder_lon = -105.2747
boulder_lat = 40.0190

time_slice = slice("2020-01-02T00:00:00.000000000", "2023-01-01T00:00:00.000000000")


# %%
boulder_face = uxds_fine.uxgrid.get_faces_containing_point(
    point_lonlat=[boulder_lon, boulder_lat]
)
# %%
uxda_fine = uxds_fine.tas
uxda_coarsest = uxds_coarsest.tas


# %%
uxda_fine.isel(n_face=boulder_face).sel(time=time_slice).to_xarray().plot(
    label="Boulder"
)
uxda_coarsest.sel(time=time_slice).mean("n_face").to_xarray().plot(label="Global mean")

plt.legend()
# %%

projection = ccrs.Robinson(central_longitude=-135.5808361)

uxda_fine.isel(time=0).plot(
    projection=projection,
    cmap="inferno",
    features=["borders", "coastline"],
    title="Global temperature",
    width=700,
)
# %%

uxda_fine.isel(time=0).plot(
    backend="matplotlib",
    projection=projection,
    cmap="inferno",
    features=["borders", "coastline"],
    title="Global temperature",
    width=1100,
)
# %%
lon_bounds = (boulder_lon - 20, boulder_lon + 40)
lat_bounds = (boulder_lat - 20, boulder_lat + 12)

# %%
uxda_fine_subset = uxda_fine.isel(time = 0).subset.bounding_box(
    lon_bounds=lon_bounds,
    lat_bounds=lat_bounds,
)
# %%
print(
    "Global temperature average: ", uxda_fine.isel(time=0).mean("n_face").values, " K"
)
print(
    "Regional subset's temperature average: ", uxda_fine_subset.mean("n_face").values, " K"
)
# %%
projection = ccrs.Robinson(central_longitude=boulder_lon)

uxda_fine_subset.plot(
    projection=projection,
    cmap="inferno",
    features=["borders", "coastline"],
    title="Boulder temperature",
    width=1100,
)
# %%
uxda_fine.uxgrid
# %%
uxgrid = ux.Grid.from_healpix(zoom=0, pixels_only=False)
uxgrid
# %%
projection = ccrs.Orthographic(central_latitude=90)
projection.threshold /= 100  # Smoothes out great circle arcs

uxgrid.plot(
    periodic_elements="ignore",  # Allow Cartopy to handle periodic elements
    crs=ccrs.Geodetic(),  # Enables edges to be plotted as GCAs
    project=True,
    projection=projection,
    width=500,
    title="HEALPix (Orthographic Proj), zoom=0",
)
# %%
# plot the grid zoom=7
uxgrid = ux.Grid.from_healpix(zoom=7, pixels_only=False)
uxgrid.plot(
    periodic_elements="ignore",  # Allow Cartopy to handle periodic elements
    crs=ccrs.Geodetic(),  # Enables edges to be plotted as GCAs
    project=True,
    projection=projection,
    width=500,
    title="HEALPix (Orthographic Proj), zoom=7",
)
# %%
projection = ccrs.Mollweide()
projection.threshold /= 100  # Smoothes out great circle arcs

uxgrid.plot(
    periodic_elements="ignore",  # Allow Cartopy to handle periodic elements
    crs=ccrs.Geodetic(),  # Enables edges to be plotted as GCAs
    project=True,
    projection=projection,
    width=500,
    title="HEALPix (Mollweide Proj), zoom=0",
)
# %%
uxda_coarsest.isel(time=0).plot(
    periodic_elements="ignore",  # Allow Cartopy to handle periodic elements
    crs=ccrs.Geodetic(),  # Enables edges to be plotted as GCAs
    project=True,
    projection=projection,
    features=["borders", "coastline"],
    cmap="inferno",
    title="Temperature (Mollweide Proj), zoom=0",
    width=500,
)
# %%
