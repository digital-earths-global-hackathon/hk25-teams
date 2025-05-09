# Hemispheric Albedo Symmetry (hk25-AlbedoSym)

Observations of Earth show a persistent, remarkable, and unexplained hemispheric symmetry of albedo, or reflected shortwave radiation. This has been documented for decades, but most reliably so since the creation of a long-term CERES record (see e.g. [Voigt et al. 2013](https://doi.org/10.1175/JCLI-D-12-00132.1), [Datseris and Stevens 2021](https://doi.org/10.1029/2021AV000440), [Jonsson and Bender 2022](https://doi.org/10.1175/JCLI-D-20-0970.1)). The observations show that all-sky symmetry is established through a compensation between clouds and clear-sky reflection: where the NH is about 4 W/m2 brighter from the clear-sky atmosphere, 2 W/m2 brighter from the surface, and 6 W/m2 dimmer from clouds.

However, CMIP class models struggle to reproduce the observed degree of symmetry (e.g., [Rugenstein and Hakuba 2023](https://doi.org/10.1029/2022GL101802), [Jonsson and Bender 2023](https://doi.org/10.5194/esd-14-345-2023), [Crueger et al. 2023](https://doi.org/10.1175/JCLI-D-22-0923.1)). Models show biases away from symmetry in all possible configurations: (1) each hemisphere is too bright, (2) each hemisphere is too dark, (3) NH is too bright, but SH is too dark, or (4) NH is too dark, but SH is too bright. Model biases are strongly linked to cloud biases, so one may speculate that a km-scale model which resolves deep convection may better capture the observed hemispheric symmetry in reflection.

The observed all-sky symmetry exists in the climatology, and has persisted despite significant dimming trends in both hemispheres, but the hemsipheres do deviate from perfect symmetry on shorter monthly timescales. Unfortunately, this makes assessment of a 12-month simulation from a km-scale model very difficult. For this project, we propose to follow the approach of [Voigt et al. 2013](https://doi.org/10.1175/JCLI-D-12-00132.1) and assess (a) the spatial decorrelation structure of the albedo, and (b) the triviality of a hemispheric albedo symmetry in the km-scale simulation.

**Coordination**: Clare Singer (cesinger23@gmail.com)

#### Sketch of initial activities
* Calculate Δλ and Δφ via autocorrelation of zonal- or meridional-mean reflected SW, as described in [Voigt et al. 2013](https://doi.org/10.1175/JCLI-D-12-00132.1).
* How does Δλ and Δφ from the km-scale model compare to that from CERES observations Δλ=36° and Δφ=10° ([Voigt et al. 2013](https://doi.org/10.1175/JCLI-D-12-00132.1))?
* Divide surface into boxes of size (Δλ, Δφ) and randomly assign to halves to compute reflection from random "hemispheres."
* How trivial is a result of hemispheric symmetry (at various levels, 0.1 W/m2 or 1 W/m2) in the km-scale model? Reproduce Fig 4 from [Voigt et al. 2013](https://doi.org/10.1175/JCLI-D-12-00132.1).
* How does the asymmetry and spatial standard deviation from the km-scale model compare with the CMIP3 models plotted in Fig 6? (CMIP5/6 models could also be added to this comparison.)

As a complemetary task to the all-sky analysis:
* Perform the [Crueger et al. 2023](https://doi.org/10.1175/JCLI-D-22-0923.1) decomposition and assess contributions from clear-sky atmosphere, surface, and clouds to symmetry in the km-scale model.
* Compute the reference asymmetry from the km-scale model (eq. 9).
* What is the role of cloud masking in the km-scale model? Reproduce Fig 4 from [Crueger et al. 2023](https://doi.org/10.1175/JCLI-D-22-0923.1).
