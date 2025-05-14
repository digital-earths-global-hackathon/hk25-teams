# MCS tracking (hk25-MCS)

## Motivation

Convective storms, especially those that develop into mesoscale convective systems (MCSs), play a crucial role in producing rainfall and hazardous weather across the globe. While recent studies have shown that DYAMOND models can capture certain aspects of tropical MCSs, such as their frequency and diurnal cycle, significant challenges remain. In particular, accurately representing the distribution of precipitation and its relationship with the surrounding environment continues to be a major hurdle ([Su et al. 2022](https://doi.org/10.2151/jmsj.2022-033); [Feng et al. 2023](https://doi.org/10.1029/2022GL102603); [Song et al. 2024](https://doi.org/10.1029/2024GL109945); [Feng et al. 2025](https://doi.org/10.1029/2024JD042204)).

Previous DYAMOND phases provided two 40-day simulation periods for summer and winter, limiting the statistical robustness of model evaluations. The Digital Earth Global Hackathon will extend these simulations to a full year using multiple global kilometer-scale models, creating an unprecedented opportunity to assess how well they capture the full spectrum of convective storms and extreme events. As part of this effort, we are organizing an MCS tracking activity to develop and apply new analysis tools, exchange insights, and strengthen collaborations within the broader atmospheric science community.

## Participants

Organizers: 

- [Zhe Feng](mailto:zhe.feng@pnnl.gov)
- [William Jones](mailto:william.jones@physics.ox.ac.uk)
- [Andreas Prein](mailto:andreas.prein@usys.ethz.ch)

| **Participant** | **Tracker** | **Node** |
|----------|----------|----------|
| [Zhe Feng](mailto:zhe.feng@pnnl.gov) | [PyFLEXTRKR](https://github.com/FlexTRKR/PyFLEXTRKR)        | US-West        |
| [Paul Ullrich](mailto:ullrich4@llnl.gov)        | [TempestExtreme](https://github.com/ClimateGlobalChange/tempestextremes)        | US-West        |
| [John Mejia](mailto:John.Mejia@dri.edu)        | [ATRACKCS](https://doi.org/10.5281/zenodo.7025989)        | US-West        |
| [Vanessa Robledo](mailto:vrobledodelgado@uiowa.edu)        | [ATRACKCS](https://doi.org/10.5281/zenodo.7025989)        | US-Central        |
| [Julia Kukulies](mailto:kukulies@ucar.edu)        | [tobac](https://github.com/tobac-project/tobac)        | US-Central        |
| [William Jones](mailto:william.jones@physics.ox.ac.uk)        | [tobac](https://github.com/tobac-project/tobac)        | UK        |
| [Ben Maybee](mailto:B.W.Maybee@leeds.ac.uk) | [simpleTrack](https://github.com/thmstein/simple-track) | UK |
| [Mark Muetzelfeldt](mailto:mark.muetzelfeldt@reading.ac.uk) | [PyFLEXTRKR](https://github.com/FlexTRKR/PyFLEXTRKR) | UK |
| [Torsten Auerswald](mailto:t.auerswald@reading.ac.uk) | [PyFLEXTRKR](https://github.com/FlexTRKR/PyFLEXTRKR) | UK |
| [Chao Li](mailto:chao.li@mpimet.mpg.de) | [PyFLEXTRKR](https://github.com/FlexTRKR/PyFLEXTRKR) | Germany |
| [Andreas Prein](mailto:andreas.prein@usys.ethz.ch)        | [MOAAP](https://github.com/AndreasPrein/MOAAP)        | Germany        |
| [Fengfei Song](mailto:songfengfei@ouc.edu.cn)        | [PyFLEXTRKR](https://github.com/FlexTRKR/PyFLEXTRKR)        | China        |
| [Jinyan Song](mailto:songjinyan@stu.ouc.edu.cn)        | [PyFLEXTRKR](https://github.com/FlexTRKR/PyFLEXTRKR)        | China        |


## Sketch of initial activities

- Standardize MCS tracking output formats (e.g., netCDF, lat/lon grid, [HEALPix grid](https://healpix.sourceforge.io/index.php))
- Compare simulated MCS statistics against satellite observations
- Extract mesoscale environmental variables surrounding MCS tracks
- Compare simulated MCS environments and relationships with MCS characteristics (lifetime, size, precipitation, etc.) across model ensembles

## Data Access

- NERSC: Data can be downloaded from this [Globus link](https://app.globus.org/file-manager?origin_id=41bda5dc-c193-43e8-a922-0fe4f94490e7&origin_path=%2F).
    - Period: 2019 to 2021
- Unified Model: Data can be downloaded from the object store (see URL and how to access below) or accessed directly from a notebook (see the [UM demo notebook](https://github.com/digital-earths-global-hackathon/hk25-teams/blob/main/hk25-MCS/demo_UM_mcs_track_stats_healpix.ipynb) for details).
    - Period: 2020-02-01 to 2021-02-28 
    - Currently there are two tracked simulation with the sim IDs:
    - um_glm_n2560_RAL3p3
      - https<nolink>://hackathon-o.s3-ext.jc.rl.ac.uk/sim-data/analysis/PyFLEXTRKR/um_glm_n2560_RAL3p3  
    - um_glm_n1280_CoMA9_TBv1p2
      - https<nolink>://hackathon-o.s3-ext.jc.rl.ac.uk/sim-data/analysis/PyFLEXTRKR/um_glm_n1280_CoMA9_TBv1p2_catalog_par  
    
    - Those who wish to download the data can do so by using rclone (as described [here](https://github.com/digital-earths-global-hackathon/tools/blob/main/dataset_transfer/UK_s3_rclone.md), replace the URL in the example by the one above)

### Observations

- Two versions of combined Tb+IMERG data from this [Globus link](https://app.globus.org/file-manager?origin_id=41bda5dc-c193-43e8-a922-0fe4f94490e7&origin_path=%2Ftracking%2Fmcs%2Fobs%2F&two_pane=true).
  - IMERG V06B (2019-01-01 to 2021-09-30)
  - IMERG V07B (2019-01-01 to 2021-12-31)

### Simulations

- SCREAM ([Data Description](https://github.com/digital-earths-global-hackathon/hk25/blob/main/content/models/scream.md))
    - Period: 2019-08-01 to 2020-08-31
- Unified Model ([Data Description](https://github.com/digital-earths-global-hackathon/hk25/blob/main/content/models/um.md))
    - Period: 2020-01-20 to 2021-02-28

## MCS Tracking Protocol

- Follows MCSMIP protocol under [Tracking criteria](https://mcsmip.github.io/design/)
- Example notebook reading/remapping HEALPix data (current repository)
- Standardize tracking output formats ([unify MCS mask files](https://github.com/WACCEM/MCSMIP-DYAMOND/blob/main/src/unify_mask_files.py))
- Example MCS mask files (TBD)
