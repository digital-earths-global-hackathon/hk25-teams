# MCS tracking (hk25-MCS)

## Motivation

Convective storms, especially those that develop into mesoscale convective systems (MCSs), play a crucial role in producing rainfall and hazardous weather across the globe. While recent studies have shown that DYAMOND models can capture certain aspects of tropical MCSs, such as their frequency and diurnal cycle, significant challenges remain. In particular, accurately representing the distribution of precipitation and its relationship with the surrounding environment continues to be a major hurdle ([Su et al. 2022](https://doi.org/10.2151/jmsj.2022-033); [Feng et al. 2023](https://doi.org/10.1029/2022GL102603); [Song et al. 2024](https://doi.org/10.1029/2024GL109945); [Feng et al. 2024](https://doi.org/10.22541/essoar.172405876.67413040/v1)).

Previous DYAMOND phases provided two 40-day simulation periods for summer and winter, limiting the statistical robustness of model evaluations. The WCRP Digital Earth Global Hackathon will extend these simulations to a full year using multiple global kilometer-scale models, creating an unprecedented opportunity to assess how well they capture the full spectrum of convective storms and extreme events. As part of this effort, we are organizing an MCS tracking activity to develop and apply new analysis tools, exchange insights, and strengthen collaborations within the broader atmospheric science community.

## Participants

Organizers: 

- [Zhe Feng](mailto:zhe.feng@pnnl.gov)
- [William Jones](mailto:william.jones@physics.ox.ac.uk)
- [Andreas Prein](mailto:andreas.prein@usys.ethz.ch)

| **Participant** | **Tracker** | **Node** |
|----------|----------|----------|
| [Zhe Feng](mailto:zhe.feng@pnnl.gov) | [PyFLEXTRKR](https://github.com/FlexTRKR/PyFLEXTRKR)        | US-West        |
| [Paul Ullrich](mailto:ullrich4@llnl.gov)        | [TempestExtreme](https://github.com/ClimateGlobalChange/tempestextremes)        | US-West        |
| [Julia Kukulies](mailto:kukulies@ucar.edu)        | [tobac](https://github.com/tobac-project/tobac)        | US-Central        |
| [William Jones](mailto:william.jones@physics.ox.ac.uk)        | [tobac](https://github.com/tobac-project/tobac)        | UK        |
| [Mark Muetzelfeldt](mailto:mark.muetzelfeldt@reading.ac.uk) | [PyFLEXTRKR](https://github.com/FlexTRKR/PyFLEXTRKR) | UK |
| [Ben Maybee](mailto:B.W.Maybee@leeds.ac.uk) | [simpleTrack](https://github.com/thmstein/simple-track) | UK |
| [Andreas Prein](mailto:andreas.prein@usys.ethz.ch)        | [MOAAP](https://github.com/AndreasPrein/MOAAP)        | ?        |
| [Vanessa Robledo](mailto:vrobledodelgado@uiowa.edu)        | [ATRACKCS](https://doi.org/10.5281/zenodo.7025989)        | ?        |
| [John Mejia](mailto:John.Mejia@dri.edu)        | [ATRACKCS](https://doi.org/10.5281/zenodo.7025989)        | US-West        |
| [Fengfei Song](mailto:songfengfei@ouc.edu.cn)        | [PyFLEXTRKR](https://github.com/FlexTRKR/PyFLEXTRKR)        | China        |
| [Jinyan Song](mailto:songjinyan@stu.ouc.edu.cn)        | [PyFLEXTRKR](https://github.com/FlexTRKR/PyFLEXTRKR)        | China        |



## Data Access

- Data can be downloaded from this [Globus link](https://app.globus.org/file-manager?origin_id=87909b37-fbcf-4735-a72e-1a406301a053&origin_path=%2F).
- Period: 2019-08-01 to 2020-09-01

### Observations

- Two versions of combined Tb+IMERG data (IMERG V06B & IMERG V07B)

### Simulations

- SCREAM ([HEALPix format](https://healpix.sourceforge.io/index.php))

## MCS Tracking Protocol

- Follows MCSMIP protocol under [Tracking criteria](https://mcsmip.github.io/design/)
- Example notebook reading/remapping HEALPix data (link TBD)
- Standardize tracking output formats ([unify MCS mask files](https://github.com/WACCEM/MCSMIP-DYAMOND/blob/main/src/unify_mask_files.py))
- Example MCS mask files ([Globus link](https://app.globus.org/file-manager?destination_id=87909b37-fbcf-4735-a72e-1a406301a053&destination_path=%2Fsample_mcs_mask%2F&two_pane=true))
- Data sharing (upload instructions TBD)
