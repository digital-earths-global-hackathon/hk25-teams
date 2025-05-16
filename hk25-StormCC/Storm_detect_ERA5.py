import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from joblib import Parallel, delayed
from tqdm import tqdm
import pandas as pd
import storm_icon_o as storm_icon  

def log(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

# ===== Define region and time range =====
nc_files = ["slp_2020_Africa.nc", "slp_2021_Africa.nc"]

LAT_MIN, LAT_MAX = 0, 50
LON_MIN, LON_MAX = -30, 40
T_START = "2020-01-01T06:00:00"
T_END   = "2021-02-28T18:00:00"

# ===== Load ERA5 'msl' variable (lazy loading) =====
log("Loading ERA5 'msl' variable (lazy)...")
slp = xr.open_mfdataset(
    nc_files,
    combine="nested",
    concat_dim="time",
    parallel=True,
    engine="netcdf4",
    data_vars="minimal",
    coords="minimal",
    compat="override"
)["msl"]

# ===== Spatial and temporal subsetting =====
log("Filtering by region and time...")
slp = slp.rename({"valid_time": "time"})
slp = slp.sel(time=slice(T_START, T_END))
slp = slp.sel(time=slp.time.dt.hour.isin([0, 6, 12, 18]))
slp = slp.sel(latitude=slice(LAT_MAX, LAT_MIN))  # Latitude must be from high to low
slp = slp.sel(longitude=slice(LON_MIN, LON_MAX)).sortby("longitude")

# ===== Extract dimension information =====
time = pd.to_datetime(slp.time.values)
lat = slp.latitude.values
lon = (slp.longitude.values % 360)
T = slp.sizes["time"]

log(f"Data ready: {T} time steps, longitude {lon.min():.1f}–{lon.max():.1f}, latitude {lat.min():.1f}–{lat.max():.1f}")

year = time.year
month = time.month
day = time.day
hour = time.hour

# ===== Storm detection function for a single time step =====
def process_t(tt):
    if tt % 10 == 0:
        log(f"Processing time step {tt+1}/{T}")
    slp_t = slp.isel(time=tt).values  
    lon_a, lat_a, amp_a = storm_icon.detect_storms(slp_t, lon, lat, res=2, Npix_min=9, cyc='anticyclonic', globe=True)
    lon_c, lat_c, amp_c = storm_icon.detect_storms(slp_t, lon, lat, res=2, Npix_min=9, cyc='cyclonic', globe=True)
    return {
        "tt": tt,
        "lon_a": lon_a, "lat_a": lat_a, "amp_a": amp_a,
        "lon_c": lon_c, "lat_c": lat_c, "amp_c": amp_c,
        "year": year[tt], "month": month[tt], "day": day[tt], "hour": hour[tt],
    }

# ===== Run storm center detection in parallel =====
log("Starting parallel storm center detection...")
results = Parallel(n_jobs=16)(delayed(process_t)(tt) for tt in tqdm(range(T)))

# ===== Organize results into a storm list =====
log("Organizing results into storm list...")
lon_a, lat_a, amp_a = [], [], []
lon_c, lat_c, amp_c = [], [], []
year_list, month_list, day_list, hour_list = [], [], [], []

for r in results:
    lon_a.append(r["lon_a"]); lat_a.append(r["lat_a"]); amp_a.append(r["amp_a"])
    lon_c.append(r["lon_c"]); lat_c.append(r["lat_c"]); amp_c.append(r["amp_c"])
    year_list.append(r["year"]); month_list.append(r["month"])
    day_list.append(r["day"]); hour_list.append(r["hour"])

storms = storm_icon.storms_list(lon_a, lat_a, amp_a, lon_c, lat_c, amp_c)

# ===== Save the results =====
output_file = "storm_det_slp_Africa_ERA5.npz"
log(f"Saving to {output_file} ...")

np.savez(output_file,
         storms=storms,
         year=np.array(year_list),
         month=np.array(month_list),
         day=np.array(day_list),
         hour=np.array(hour_list))

log("Storm detection completed successfully.")


