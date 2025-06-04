#!/bin/python
# 
# Cyclone detection and plotting using TempestExtremes and the ClimateDT data.
# This use-case requires TempestExtremes to be installed in the used environment.
# The data request can be added in get_data_as_nc
# TODO: move request to main runscript

import sys
sys.path.insert(1, '.')

from preprocessing import get_data_as_nc

import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import subprocess
import huracanpy
import glob
from pathlib import Path

# output data as netcdf file
ncpath= "./data/"
track_path= "./data/"

# Define start and end dates
start_date = datetime.strptime("2035-06-01", "%Y-%m-%d")
end_date = datetime.strptime("2035-07-31", "%Y-%m-%d")

datelist = [
    (start_date + timedelta(days=i)).strftime("%Y%m%d")
    for i in range((end_date - start_date).days + 1)
]

# For every day get the data from the data-bridge and stores it as netcdf file. 
# Then run the DetectNodes processing to detect the tropical cyclones
for date in datelist:

    ncfile=f"tmp_icon_{date}.nc"
    in_file = f"{ncpath}/{ncfile}"
    out_file = f"{track_path}/nodes/nodes_{date}.txt"

    # Skip date if output already exists
    if Path(out_file).exists():
        print(f"{out_file} exists, skipping...")
        continue

    print(f"Processing {date}")
    
    get_data_as_nc(date,ncout=ncfile, ncpath=ncpath)

    subprocess.run([
        "DetectNodes",
        "--in_data", in_file,
        "--out", out_file,
        "--searchbymin", "msl",
        "--latname", "latitude",
        "--lonname", "longitude",
        "--closedcontourcmd", "msl,200.0,5.5,0;_DIFF(zg250,zg500),-6,6.5,1.0",
        "--mergedist", "6.0",
        "--outputcmd", "msl,min,0;_VECMAG(uas,vas),max,2"
    ])
    
    subprocess.run(["head", out_file])

# now combine all fines 
flist = glob.glob(f'{track_path}/nodes/*.txt')
all_nodes = f'{track_path}/nodes/all_nodes.dat'

with open(all_nodes, 'w') as outfile:
    for fname in flist:
        with open(fname, 'r') as infile:
            outfile.write(infile.read())

in_file = f"{track_path}/nodes/all_nodes.dat"
out_file = f"{track_path}/tracks/icon_{start_date}_{end_date}.csv"

# create tracks by stichting nodes together
subprocess.run([ 
    "StitchNodes",
    "--in", in_file,
    "--out", out_file,
    "--out_file_format", "csv",
    "--in_fmt", "lon,lat,slp,wind",
    "--range", "8.0",
    "--mintime", "54h",
    "--maxgap", "24h",
    "--threshold", "wind,>=,10.0,10;lat,<=,50.0,10;lat,>=,-50.0,10",
])

tracks = huracanpy.load(out_file)
huracanpy.plot.tracks(tracks.lon, tracks.lat, intensity_var=tracks.wind)
plt.title(f"ICON SSP3-7.0 {start_date} to {end_date}")
plt.savefig(f"./plots/icon_projection_tc_tracks_wind_{start_date}_{end_date}.png")
