## Participants

Louise Nuijens (TUD)
Edoardo Foschi (TUD)
Jacotte Monroe (TUD)
Geet George (TUD)
Owen O’Driscoll (TUD)
Chris Chapman (CSIRO/TUD)
Lorenzo Davoli (TUD/UNIMIB)
Alessandro Storer (UNIMIB)
Claudia Pasquero (UNIMIB)
Xuanyu Chen (CSU)
Elizabeth Thompson (NOAA)
Mia Sophie Specht (MPI)
Jan-Hendrik Malles (Uni Bremen)


## Shared objectives

We have two working hypotheses that guide the hacking and aim to identify fingerprints of finer-scale air-sea interactions with a focus on the tropics:

1. Km-scale atmospheric variability (and interaction with the ocean) matters for mean air-sea fluxes and upper ocean mixing. 

   Hacking activities include:
   a. Calculation of air-sea heat fluxes and stress using COARE, decomposing into speed, gust, stability, transfer coefficient, drag contributions.
   b. Perform flux calculation at different grid resolutions (model zoom levels)
   c. Diagnose effective transfer coefficients used across models compared to COARE by using the modeled air-sea flux.
   d. Analyze fluxes, wind stress (curl), upper ocean vorticity/mixing from low wind (non-precipitating) atmospheres to high wind (precipitating) atmospheres

2. Km-scale (2 km - 200 km) SST patterns and ocean heat content imprint on atmospheric winds and clouds 

   Hacking activities include:
   a. Preliminary analysis of fine-scale SST patterns in coupled (IFS-FESOM ifs_tco3999-ng5_rcbmf, zoom=11,time='PT1H') and uncoupled simulations (NICAM nicam_gl11, zoom=9,  time='PT3H').
   b. Correlate wind divergence with SST gradient across grid resolution (model zoom levels) - in collaboration with other teams.
   c. Calculate correlations between SST and i. wind speed, ii. air temperature, iii. specific humidity, iv. cloud fraction, and their dependence on wind speed, large scale forcing, and as a function of scale.
   d. Compare the CF - SST relationships to satellite composites.
   e. Perform backtrajectory analysis of mesoscale divergence patterns to analyze scale-growth.
   f. Analyze the response of (deeper) ocean and atmosphere on excess heat in upper ocean. 
 

## Model Simulations (first focus)

- IFS-FESOM 2.8km simulation (RCBMF): ifs_tco3999-ng5_rcbmf
- ICON nextGEMS Cycle 4 simulation: icon_ngc4008 (coupled)
- ICON 2.5 km simulation: icon_d3hp003 (uncoupled)
- NICAM nicam_gl11, zoom=9,  time='PT3H'
  
