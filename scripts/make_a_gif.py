import intake
import easygems.healpix as egh
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import cartopy.feature as cf
import pandas as pd

# Add this to the environment by running "conda install gif" while in the easygems env
import gif

data_source = "um_glm_n1280_GAL9"
zoom_level = 5
date_window = "2020-02" # To subset the data from the catalogue
observable = "rlut" #  toa_outgoing_longwave_flux

gif_pd_timerange = pd.date_range('2020-02-20 00:00:00', '2020-02-25 00:00:00', freq='2h')
gif_filename = "example_rlut_Feb.gif"

cat = intake.open_catalog("https://digital-earths-global-hackathon.github.io/catalog/catalog.yaml")["UK"]
ds = cat[data_source](zoom=zoom_level).to_dask()

fragment_da = ds[observable].sel(time=date_window) # Part of the DataArray used to generate the gif

@gif.frame
def worldmap(in_data, **kwargs):
    projection = ccrs.Robinson(central_longitude=-135.5808361)
    fig, ax = plt.subplots(
        figsize=(8, 4), subplot_kw={"projection": projection}, constrained_layout=True
    )
    ax.set_global()

    egh.healpix_show(in_data, ax=ax, **kwargs)
    ax.add_feature(cf.COASTLINE, linewidth=0.8)
    ax.add_feature(cf.BORDERS, linewidth=0.4)


frames = [worldmap(fragment_da.sel(time=t), cmap="bone_r") for t in gif_pd_timerange] ;
gif.save(frames, gif_filename, duration=50)


