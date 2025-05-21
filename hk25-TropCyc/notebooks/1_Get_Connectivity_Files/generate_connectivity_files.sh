# Script to run with Generate_healpix_grids.ipynb
# Written by Bryce Harrop
# Adapted by Stella Bourdin

for zoom in {0..1}; do
    
    in_grid=/work/scratch-nopw2/sbourdin/ConnectivityFiles/healpix_grid_zoom_${zoom}_format_exodus.nc
    out_connect=/work/scratch-nopw2/sbourdin/ConnectivityFiles/connect_healpix_zoom_${zoom}.txt

    if [ ! -f "${out_connect}" ]; then
	echo Generating connectivity file for zoom level $zoom
	GenerateConnectivityFile \
	    --in_mesh ${in_grid} \
	    --out_connect ${out_connect}
    fi
done




