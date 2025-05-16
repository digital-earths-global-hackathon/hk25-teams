import intake
from easygems import healpix as egh
from global_land_mask import globe
import healpix
import cartopy.crs as ccrs
import uxarray as ux
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
from datetime import datetime
import storm_icon_o as storm_icon

def log(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

log("Loading dataset...")

cat = intake.open_catalog("https://digital-earths-global-hackathon.github.io/catalog/catalog.yaml")
experiment = cat.online["icon_d3hp003"]
data = experiment(time="PT6H", zoom=7, time_method="inst").to_dask().chunk({"time": 100})
data = data.pipe(egh.attach_coords)

log("Subsetting region and selecting time range...")

ds_slp_region = (
    data['psl']
    .where(((data["lat"] > 0) & (data["lat"] < 50)) & ((data["lon"] > 330) | (data["lon"] < 40)), drop=True)
)
ds_slp_region = ds_slp_region.sel(time=slice("2020-01-01T06:00:00", "2021-02-28T18:00:00"))

log("Performing grid remapping...")

zoom = 7
grid_spacing = 64 / 2**zoom

lons = np.arange(grid_spacing / 2, 360, grid_spacing)
lats = np.arange(90 - grid_spacing / 2, -90, -grid_spacing)

lon2d, lat2d = np.meshgrid(lons, lats)

pix_values = healpix.ang2pix(
    ds_slp_region.crs.healpix_nside,
    lon2d,
    lat2d,
    nest=ds_slp_region.crs.healpix_order,
    lonlat=True
)

pix = xr.DataArray(
    data=pix_values,
    dims=("lat", "lon"),
    coords={"lat": lats, "lon": lons}
)

slp_gridded = ds_slp_region.drop_vars(["lat", "lon"], errors="ignore").sel(cell=pix, method="nearest")
slp = slp_gridded
time = pd.to_datetime(slp.time.values)
lat = slp['lat'].values
lon = slp['lon'].values
T = slp.shape[0]

log(f"Data loaded: {T} time steps. Starting storm detection...")

# Extract time metadata
year = time.year
month = time.month
day = time.day
hour = time.hour

# Storm detection for a single time step
def process_t(tt):
    if tt % 50 == 0:
        log(f"Processing time step {tt}/{T}")
    slp_t = slp[tt].values
    lon_storms_a, lat_storms_a, amp_storms_a = storm_icon.detect_storms(slp_t, lon, lat, res=2, Npix_min=9, cyc='anticyclonic', globe=True)
    lon_storms_c, lat_storms_c, amp_storms_c = storm_icon.detect_storms(slp_t, lon, lat, res=2, Npix_min=9, cyc='cyclonic', globe=True)
    return {
        "tt": tt,
        "lon_a": lon_storms_a,
        "lat_a": lat_storms_a,
        "amp_a": amp_storms_a,
        "lon_c": lon_storms_c,
        "lat_c": lat_storms_c,
        "amp_c": amp_storms_c,
        "year": year[tt],
        "month": month[tt],
        "day": day[tt],
        "hour": hour[tt],
    }

# Run detection in parallel
results = Parallel(n_jobs=16)(
    delayed(process_t)(tt) for tt in tqdm(range(T))
)

log("Storm detection completed. Consolidating results...")

# Aggregate detection results
lon_storms_a, lat_storms_a, amp_storms_a = [], [], []
lon_storms_c, lat_storms_c, amp_storms_c = [], [], []
year_list, month_list, day_list, hour_list = [], [], [], []

for res in results:
    lon_storms_a.append(res['lon_a'])
    lat_storms_a.append(res['lat_a'])
    amp_storms_a.append(res['amp_a'])
    lon_storms_c.append(res['lon_c'])
    lat_storms_c.append(res['lat_c'])
    amp_storms_c.append(res['amp_c'])
    year_list.append(res['year'])
    month_list.append(res['month'])
    day_list.append(res['day'])
    hour_list.append(res['hour'])

log("Combining all results into storm list...")

storms = storm_icon.storms_list(lon_storms_a, lat_storms_a, amp_storms_a,
                                 lon_storms_c, lat_storms_c, amp_storms_c)

log("Saving results to 'storm_det_slp_Africa_icon.npz'...")

np.savez('storm_det_slp_Africa_icon.npz',
         storms=storms,
         year=np.array(year_list),
         month=np.array(month_list),
         day=np.array(day_list),
         hour=np.array(hour_list))

log("All tasks completed.")
