# Tropical Cyclones (hk25-TropCyc)

Tropical Cyclones (TCs) are intense organised convective systems that are responsible for nearly half of the worldwide disaster-related costs (CRED & UNDRR, 2020). Historically, they have been difficult to represent in coarse-resolution climate models because of their small size and their sensitivity to convective parametrisations. It has been shown that increasing resolution to 25km allows models to represent the number and distribution of cyclones correctly (Roberts et al. 2020), but the intensity remains largely underestimated, and the structure is not well represented (Bourdin et al. 2024). Baker et al. (2024) showed that increasing resolution up to 5km improves the realism of intensity, intensification rate, lifecycle and structure of TCs in NextGEMS simulations.

In this team, we investigated how TCs are represented in the new sets of simulations in terms of statistics, structure, lifecycle and link with the environment.

**Coordination**: Stella Bourdin (stella.bourdin@physics.ox.ac.uk), Alex Baker (alexander.baker@reading.ac.uk), Arthur Avenas (arthur.avenas@esa.int), Xu Chen (chenx@g.ecc.u-tokyo.ac.jp)

In this folder we leave for legacy scripts to track TCs in the simulations, as well as some of the data (in particular track datasets).

## Useful resources

* [TempestExtremes](https://github.com/ClimateGlobalChange/tempestextremes)

  Some introductory content is provided in `notebooks/TE_intro.ipynb` & `notebooks/TE_tracking_UZ.ipynb`.

* [HuracanPy](https://huracanpy.readthedocs.io/en/latest/)

## Connectivity Files for TempestExtremes

With great help from Bryce Harrop, we are now able to run TempestExtremes on HealPix, using dedicated ConnectivityFiles. 
Files can be generated using instructions in `notebooks/1_Get_Connectivity_Files/`. Corresponding files for zoom 0-2 are provided via GitHub in `ConnectivityFiles` while files for zoom 3-10 can be found on JASMIN in `/home/users/sbourdin/WCRP_Hackathon/hk25-teams/hk25-TropCyc/ConnectivityFiles/` or in [this Google Drive folder](https://drive.google.com/drive/folders/1fNDDQA_G-yy05SP8J8pV2EIFJCs1bOtb?usp=sharing).

## TC tracking

In the scope of this hackathon, TC tracking has been performed with two algorithms, in any case using the TempestExtremes software:

* UZ (Zarzycki et Ullrich, 2017; Ullrich et al., 2021) when geopotential on pressure levels was available;
* UZ-2D (Unpublished, see notebook for script, contact S. Bourdin for info) when it was not.

For each algorithm, the process is in two steps:

1. Download the necessary data on HealPix zoom 8 (~25km is supposedly a good compromise) with the script in `notebooks/2_TC_tracking/UZ_tracking_pre-processing.ipynb` or `2D_tracking_pre-processing.ipynb` (depending on which algorithm you want to run). Data will be chuncked into monthly files which is a good compromise for efficiency/memory use.

2. Run the tracking with the script in `notebooks/2_TC_tracking/UZ_tracking_TempestExtremes.ipynb` or `2D_tracking_TempestExtremes.ipynb`.

At the end of the process, you will obtain the tracks as a CSV file, whose name is the name of the simulation.


*Technical details*:
TCs are to be tracked in all relevant simulations (i.e. simulations covering TC-prone areas) using TempestExtremes and the algorithm described in Ullrich et al. 2021.
Output should include, a minima, track_id, time, lon, lat, maximum wind speed in a 2°GCD radius and minimum SLP.

## TC-centered snapshot retrieval

Instructions are provided to retrieve TC-centered snapshots in `notebooks/3_snapshot_retrieval/`.
As for the tracking, two steps are required:

1. Download the necessary data (at highest zoom level available this time) with the `Pre-processing` script (available as notebook for one variable at a time, or as a bash script for several variables)
2. Run NodeFileCompose as in the `NodeFileCompose` script (available as notebook for one variable at a time, or as a bash script for several variables)

NB: It is recommended to adjust the snapshots' output grid to match the zoom level.

Snapshots files are too heavy to be shared via GitHub. 
They will be available on JASMIN at `/home/users/sbourdin/WCRP_Hackathon/hk25-teams/hk25-TropCyc/snapshot` and
I will try to share them through [this Google Drive folder](https://drive.google.com/drive/folders/1fNDDQA_G-yy05SP8J8pV2EIFJCs1bOtb?usp=sharing)