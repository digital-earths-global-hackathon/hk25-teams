#!/bin/bash
## Snapshot retrieval with TempestExtremes' NodeFileCompose
## By Stella Bourdin

# Script Parameters
run='um_glm_n1280_GAL9' # Code of your simulation (must be the reference in the catalog)
zoom=9

# Folders
scr_dir='/work/scratch-nopw2/sbourdin/' # Change to your own scratch/temporary folder
data_pp_dir=$scr_dir/$run/data_healpix # dir to store preprocessed (pp) files

# Input
tracks_file="../../TC_tracks/$run.csv"

# Connectivity File
CONNECT_FILE=/home/users/sbourdin/WCRP_Hackathon/hk25-teams/hk25-TropCyc/ConnectivityFiles/ConnectivityFiles_for_healpix_zoom_$zoom.txt

for var in uas vas
do

    # Output
    if ! [ -d ../../snapshots/${var}/ ]; then mkdir ../snapshots/${var}/; fi
    snaps_file="../../snapshots/${var}/${run}_${var}.nc"
    
    ## Prepare file list
    flist=`ls $data_pp_dir/*${var}_zoom_${zoom}.nc`
    INPUT=""
    for f in $flist
    do
    INPUT="$INPUT$f;"
    done
    INPUT=${INPUT:0:-1} # Remove last ;
    echo $INPUT > INPUT.txt
    
    conda run -n hackathon NodeFileCompose \
    --in_nodefile "$tracks_file" \
    --in_connect $CONNECT_FILE \
    --in_nodefile_type "SN" \
    --in_fmt "(auto)" \
    --in_data_list "INPUT.txt" \
    --out_data "$snaps_file" \
    --var "$var" \
    --varout "$var" \
    --out_grid "XY" \
    --dx 0.1 \
    --resx 100 \
    --snapshots

done