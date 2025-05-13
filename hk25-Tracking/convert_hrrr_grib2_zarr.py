########################################################
# Author: Ziming Chen, PNNL
# This code is used to
#   - convert HRRR Grib2 files into a Zarr format compatible with StormCast’s HrrrEra5Dataset class:
#   1. Parse all hourly GRIB2 files via cfgrib + xarray
#   2.	Stack them over time with consistent channel dimensions
#   3.	Add metadata: time, channel, lat/lon
#   4.	Write to .zarr using to_zarr(...)
#   5.	Generate and save normalization stats: means.npy, stds.npy
#   6.	(Optional but needed) Create invariants.zarr
# Attention: this script deals with hourly data for each time step at each time
########################################################

# %%
# 
import os
import glob
import numpy as np
import xarray as xr
import cfgrib
import yaml
import gc

# === User Configuration ===
r_lat            = [31, 46]
r_lon            = [250, 275]
i_YrRange        = [2017, 2017]
i_MonRange       = [1, 2, 3]


# %%
def parse_timestamp_from_path(path):
    import re
    import os
    folder_date   = os.path.basename(os.path.dirname(path))
    match = re.search(r't(\d\d)z', os.path.basename(path))
    if match:
        hour = int(match.group(1))
        # base_date = "2017-01-01" # date
        return np.datetime64(f"{folder_date[:4]}-{folder_date[4:6]}-{folder_date[6:]}T{hour:02d}:00")
    else:
        raise ValueError("Cannot parse hour from filename: " + path)

import calendar
def days_in_month(year, month):
    # The monthrange() function returns a tuple (weekday of the first day of the month, number of days in the month)
    return calendar.monthrange(year, month)[1]

def load_variable_metadata(yaml_path):
    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)
    return config["variables"]

def read_variable(f, short_name, type_of_level, level):
    return xr.open_dataset(
        f,
        engine="cfgrib",
        backend_kwargs={
            "filter_by_keys": {
                "typeOfLevel": type_of_level,
                "shortName": short_name,
                "level": level,
                "stepType": "instant"
            }
        }
    )

def read_height_above_ground_variables(f, short_name, type_of_level, level):
    ds = xr.open_dataset(f,
                        engine="cfgrib",
                        backend_kwargs={
                            "filter_by_keys": {
                                "typeOfLevel": type_of_level,
                                "stepType": "instant",
                                "level": level*1.
                            }
                        }
                        )
    # print(ds[short_name])
    ds            = ds[short_name]
    ds            = ds.to_dataset(name=short_name)
    return ds

import xesmf as xe
def RegridHRRR_to_LatLonGrid(da_channels, lat, lon):
    # This function regrids the raw HRRR data onto lat/lon grid (~3km)
    # create input grid dataset
    input_grid = xr.Dataset({
        "lat": (["y", "x"], da_channels.latitude.values),
        "lon": (["y", "x"], da_channels.longitude.values)
    })

    # Define target lat/lon grid with ~3km spacing (~0.03º)
    lat_min, lat_max = min(lat), max(lat)
    lon_min, lon_max = min(lon), max(lon)
    # lat_min, lat_max    = da_channels.latitude.min().values, da_channels.latitude.max().values
    # lon_min, lon_max    = da_channels.longitude.min().values, da_channels.longitude.max().values
    target_lat = np.arange(lat_min, lat_max + 0.03, 0.03)
    target_lon = np.arange(lon_min, lon_max + 0.03, 0.03)
    lon2d, lat2d        = np.meshgrid(target_lon, target_lat)
    output_grid= xr.Dataset({
        "lat": (["y", "x"], lat2d),
        "lon": (["y", "x"], lon2d)
    })

    # Create and apply regridder
    regridder  = xe.Regridder(input_grid, output_grid, method="bilinear", 
                              periodic=False, reuse_weights=False)
    da_channels_= regridder(da_channels)
    #
    da_channels_= da_channels_.assign_coords(
        y=("y", output_grid["lat"][:, 0].data),
        x=("x", output_grid["lon"][0, :].data),
        latitude=(("y", "x"), output_grid["lat"].data),
        longitude=(("y", "x"), output_grid["lon"].data)
    )
    #
    return da_channels_

def convert_hrrr_grib2_to_zarr(input_dir, output_zarr_path, output_stats_dir, lat, lon,
                               Year=2017, Mon=1, day=1):
    """
    This function is used to read and convert the hourly HRRR dataset in grib2 format into zarr format
    input_dir: path with HRRR original data
    output_zarr_path: output path for the zarr data
    lat, lon: range of desired domain 
    Year, start_time="01-01-00", end_time="12-31-23"
    """
    from datetime import datetime
    lat_min, lat_max = min(lat), max(lat)
    lon_min, lon_max = min(lon), max(lon)
    #
    s_year        = [str(i) for i in range(Year, Year+1)]
    s_MM          = [str(i).zfill(2) for i in range(Mon, Mon + 1)]
    start_time, end_time = f"{str(Mon).zfill(2)}-{str(day).zfill(2)}-00", \
                           f"{str(Mon).zfill(2)}-{str(day).zfill(2)}-23"
    #
    grib_files    = []
    for s_yr in s_year:
        for s_mon in s_MM:
            i_yr   = int(s_yr)
            i_month= int(s_mon)
            # i_days = days_in_month(i_yr, i_month)
            #
            # for day in range(1, i_days + 1):
            s_DD_tmp        = str(day).zfill(2)
            grib_files_tmp  = sorted(glob.glob(os.path.join(input_dir, 
                                               f"{s_yr}{s_mon}{s_DD_tmp}/hrrr.t??z.wrfnatf00.grib2"))
                                                )
            start_dt        = datetime.strptime(s_yr + "-" + start_time, "%Y-%m-%d-%H") 
            end_dt          = datetime.strptime(s_yr + "-" + end_time, "%Y-%m-%d-%H")
            grib_files_tmp  = [f for f in grib_files_tmp if start_dt <= 
                                parse_timestamp_from_path(f).astype("M8[s]").astype(datetime) <= end_dt]
            grib_files.extend(grib_files_tmp)
    #
    all_data   = []
    times      = []
    #
    for f in grib_files:
        print(f"Loaded variables from {f}:")
        times.append(parse_timestamp_from_path(f))
        # times.append(parse_timestamp_from_path(f).astype("M8[s]"))
        #
        # Load existing Zarr time steps if file exists
        zarr_year_path      = os.path.join(output_zarr_path, f"{Year}.zarr")
        if os.path.exists(zarr_year_path):
            try:
                ds_existing = xr.open_zarr(zarr_year_path)
                existing_times = set(ds_existing.time.values.astype("datetime64[s]"))
                t_check     = np.datetime64(parse_timestamp_from_path(f).item(), "s")
                if t_check in existing_times:
                    print(f"Time {parse_timestamp_from_path(f)} already exists in {zarr_year_path}. Skipping write.")
                    continue
            except Exception as e:
                print(f"Could not read existing Zarr for time check: {e}")
        #
        datasets   = []
        for var in load_variable_metadata("hrrr_variables.yaml"):
            try:
                ds = read_variable(f, var["short_name"], var["type_of_level"], var["level"])
            except Exception as e:
                print(f"Failed to load {var['name']}: {e}")
                # print(ds.data_var)
            if var["short_name"] == "t2m" or var["short_name"] == "u10" or var["short_name"] == "v10":
                ds = read_height_above_ground_variables(f, var["short_name"], var["type_of_level"], var["level"])
                s_Tmp = var["short_name"]
                print(f"individually read {s_Tmp}")
            ds = ds.rename({var["short_name"]: var["name"]})
            # ds = RegridHRRR_to_LatLonGrid(ds, lat, lon)
            datasets.append(ds)
        #
        del ds
        gc.collect()
        # Convert to a DataArray: shape will be (channel, y, x)
        da_channels= xr.merge(datasets, compat='override').to_array(dim="channel")
        del datasets
        gc.collect()
        #
        # Expand with a time dim (1, channel, y, x)
        da_channels = da_channels.expand_dims("time")
        da_channels = RegridHRRR_to_LatLonGrid(da_channels, lat, lon)
        #
        # Append to full list
        all_data.append(da_channels)
        #
    ## merge all data 
    if 'da_channels' in vars():
        da_channels= xr.concat(all_data, dim="time")
        da_channels= da_channels.assign_coords(time=("time", times))
        # print(da_channels.shape)
        del all_data
        gc.collect()
        #
        path       = os.path.join(output_zarr_path, f"{Year}.zarr")
        #
        os.makedirs(os.path.dirname(path), exist_ok=True)
        ## drop the conflicting var from my dataset
        if "valid_time" in da_channels.coords:
            da_channels = da_channels.drop_vars("valid_time")
        ## -------------- 2025.05.10 ------------------- 
        if os.path.exists(path):
            da_channels.to_dataset(name="HRRR").to_zarr(path, mode="a", append_dim="time", consolidated=True)
        else:
            da_channels.to_dataset(name="HRRR").to_zarr(path, mode="w", consolidated=True)
        ## -------------- 2025.05.10 -------------------
        # print(f"Written data for time {times}: \
        #         lat min {da_channels.latitude.min().item()}, \
        #         lat max {da_channels.latitude.max().item()}; \
        #         lon min {da_channels.longitude.min().item()}, \
        #         lon max {da_channels.longitude.max().item()}")
        # print("")
        #
        # # Append to full list
        # all_data.append(da_channels)
        print("")
        # # print(all_data)
        print(f"{Year}-{start_time} to {Year}-{end_time} done!")
        print("")
        print("")

# %%
# ========================================================================
s_DomainName     = "Zarr" # "Zarr_western_us" #
zarr_path        = "/pscratch/sd/c/chenzm/my_data/HRRR/" + s_DomainName + "/HRRR"

for iyr in i_YrRange:
    for iMon in i_MonRange:
        i_days   = days_in_month(iyr, iMon)
        for iDay in range(1, i_days+1, 1):
            convert_hrrr_grib2_to_zarr(
                input_dir    = "/pscratch/sd/c/chenzm/my_data/HRRR/nat_hybrid_lev",
                output_zarr_path = os.path.join(zarr_path, "train"),
                output_stats_dir=os.path.join(zarr_path, "stats"),
                lat          = r_lat, lon = r_lon,
                Year         = iyr,
                Mon=iMon, day=iDay
            )
