#%%
import uxarray as ux
import easygems.healpix as eghp
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cf
import cmocean
import cartopy.crs as ccrs
import intake

#%%
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
cat_url = "https://digital-earths-global-hackathon.github.io/catalog/catalog.yaml"
cat = intake.open_catalog(cat_url)
model_run = cat.online.icon_d3hp003
# %%
ds = model_run(zoom=9, time = 'P1D').to_dask()
# %%
ds
# %%
uxds = ux.UxDataset.from_healpix(ds)
uxda = uxds["ts"]
uxda
# %%
print(
    "Global surface temperature average on ", uxda.time[0].values, ": ", uxda.isel(time=0).mean().values, " K"
)
# %%
%%time

projection = ccrs.Robinson()

uxda.isel(time=0).plot(
    projection=projection,
    cmap="inferno",
    features=["borders", "coastline"],
    title="Global surface temperature (Polygon raster)",
    width=700,
)
# %%
%%time
projection = ccrs.Robinson()

# Controls the size of each pixel (smaller value leads to larger pixels)
pixel_ratio = 0.5

uxda.isel(time=0).plot.points(
    projection=projection,
    rasterize=True,
    dynamic=False,
    width=1000,
    height=500,
    pixel_ratio=pixel_ratio,
    cmap="inferno",
    title=f"Global surface temperature (Point raster), pixel_ratio={pixel_ratio}",
)
# %%
cat_url = "https://digital-earths-global-hackathon.github.io/catalog/catalog.yaml"
cat = intake.open_catalog(cat_url)
model_run = cat.online.icon_d3hp003
ds =  model_run(zoom=9, time = 'P1D').to_dask()
# %%
uxds = ux.UxDataset.from_healpix(ds)
uxda = uxds["ts"]
# %%
uxda_coarse = ux.UxDataset.from_healpix(model_run(zoom=4, time = 'P1D').to_dask())["ts"]
# %%
uxda_coarse.uxgrid.face_node_connectivity
# %%
boulder_lat = 40.0190
uxda_lat = uxda_coarse.cross_section.constant_latitude(boulder_lat)
# %%
import geoviews.feature as gf

uxda_lat.isel(time=0).plot(
    rasterize=False,
    projection=projection,
    global_extent=True,
    cmap="inferno",
    clim=(220, 310),
    features=["coastline"],
    title=f"Global surface temperature cross-section at {boulder_lat} degrees latitude",
    width=700,
) * gf.grid(projection=projection)
# %%
uxda_lat_int = uxda_coarse.cross_section.constant_latitude_interval(
    [boulder_lat-20, boulder_lat+20]
)
# %%
uxda_lat_int.isel(time=0).plot(
    rasterize=False,
    projection=projection,
    global_extent=True,
    cmap="inferno",
    clim=(220, 310),
    features=["coastline"],
    title=f"Global surface temperature cross-section at {boulder_lat} degrees latitude",
    width=700,
) * gf.grid(projection=projection)
# %%
zonal_mean_ts = uxda.isel(time=0).zonal_mean(lat=(-90, 90, 5))
# %%
(
    uxda.isel(time=0).plot(
        cmap="inferno",
        # periodic_elements="split",
        height=300,
        width=600,
        colorbar=False,
        ylim=(-90, 90),
    )
    + zonal_mean_ts.plot.line(
        x="ts_zonal_mean",
        y="latitudes",
        height=300,
        width=180,
        ylabel="",
        ylim=(-90, 90),
        xlim=(220, 310),
        # xticks=[220, 250, 280, 310],
        yticks=[-90, -45, 0, 45, 90],
        grid=True,
    )
).opts(title="Temperature and its Zonal means at every 5 degree latitude")
# %%
model_run_older = cat.online.icon_ngc4008

ds_older = model_run_older(zoom=8, time="P1D").to_dask()

uxds_older = ux.UxDataset.from_healpix(ds_older)
uxds_older
# %%
uxds_older["ts"].isel(time=0).plot(
    projection=projection,
    cmap="inferno",
    features=["borders", "coastline"],
    title="Global surface temperature (zoom = 8) - Older Simulation",
    width=700,
)
# %%
%%time
uxda_older_remapped = uxds_older["ts"].isel(time=0).remap.inverse_distance_weighted(
    uxds.uxgrid, k=3, remap_to="face centers", coord_type="cartesian"
)
# %%
(uxda.isel(time=0) - uxda_older_remapped).plot(
    projection=projection,
    cmap="RdBu_r",
    features=["borders", "coastline"],
    title="Global surface temperature difference between older and newer simulations (zoom = 9)",
    clim=(-25,25),
    width=700,
)
# %%
zonal_anomaly = uxda - zonal_mean_ts
# %%
zonal_anomaly
# %%
zonal_anomaly.isel(time= 0).plot(
    cmap="RdBu_r",
    # periodic_elements="split",
    height=300,
    width=600,
    colorbar=False,
) 
# %%
full_data = uxda.isel(time=0)
zonal_mean = full_data.zonal_mean()
# %%
zonal_grid_proxy = full_data.cross_section.constant_longitude(
    0
)
# %%
zonal_mean_remapped = zonal_mean.remap.inverse_distance_weighted(
    zonal_grid_proxy.uxgrid, k=3, remap_to="face centers", coord_type="cartesian"
)
# %%
