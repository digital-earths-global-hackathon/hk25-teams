# Evaluation of Meso-scale Degree of Organization of Convection (hk25-DOCmeso)

### Project Details

Observational studies show, the degree to which convection is in a more or less organized state on spatial scales of 100-200 km (mesoscale) is closely coupled with tropical-mean radiative fluxes (Bony et al., 2020). Therefore, mesoscale DOC is likely an important convective feature for a realistic representation of the radiation budget. In this workshop division, we calculate mesoscale DOC from model output and satellite observations, highlight model differences in the representation of DOC, and compare model DOC with observed DOC. The first two days will be spent calculating and contrasting different measures of DOC. On the third day, we collaborate with the [large-scale environmental conditions group](/hk25-LargeScaleP/README.md) to compare meso-scale DOC with environmental conditions, and on the final day, we summarize the key findings and tidy up the github repository for future work.

#### Project Lead: Philip Blackberg (philip.blackberg@monash.edu)

#### Project members:

#### Number of open slots for students: 4-6 students

#### Expertise needed:

No particular expertise needed. It may be helpful with a surface-level familiarity with dask/xarray, the qsub scheduler, and submitting python scripts (rather than notebooks). 

### Project Description

#### Background:

Deep convection organize across spatial- and temporal scales and include a wide spectra of convective events ranging from isolated and transient mesoscale convective events (Houze, 1977\) up to long-lived super-clusters and planetary envelopes (Madden & Julian, 1971; Mapes & Houze, 1993). Given the great diversity in scales and convective events included, we may consider several ways to quantify the degree to which convection is in a more or less organized state (Degree of Organization of Convection \- DOC) (Mandorli & Stubenrauch, 2023). Mesoscale DOC (100-200 km), quantified as the deviation of the spatial distribution of convective cores from that expected from a random distribution (Tompkins, 2017), has been found to be associated with a drying of the lower free troposphere and a cooling from tropical-mean radiative fluxes. In particular, about 40% of the variance in the radiation budget is explained by mesoscale DOC on monthly timescales (Bony et al., 2020). Given the storm-resolving capabilities of the new generation of high-resolution models (\<5 km horizontal grid spacing), the evaluation of mesoscale DOC may be an important first step in a process-based analysis of the representation of organized convection and the associated radiative fluxes.

#### Primary research question:

Can high-resolution models capture the observed mesoscale DOC?

#### Secondary research questions:

Can high-resolution models capture the observed connection between mesoscale DOC and radiative fluxes?

#### Primary output:

Comparison of mesoscale DOC across models, and between models and observations. This may be done at specific timescales; daily and/or hourly, and at different spatial scales; Maritime Continent, East Pacific, tropical, and/or global.

#### Secondary outputs:

Ideally, the investigation will serve as the base for a more comprehensive evaluation of model DOC. Further, tools/metrics for evaluating DOC from high-resolution data at different spatial and temporal scales may be shared through the workshop division github repository. 

### Methodology

#### Datasets: 

Model output (2D, global):  
Precipitation  
LW fluxes (all sky and clear sky)  
SW fluxes (all sky and clear sky)

Observations (2D, global):  
IMERG precipitation data

#### Methods:

Scripts for timestep calculation of mesoscale DOC will be provided. The metric calculation of DOC will be based on a latlon grid close to the provided model resolution, which means the HEALPix grid needs to be converted to latlon. Quantification of radiative fluxes may be calculated on the native grid. All the necessary packages for calculating mesoscale DOC are available on the standard Gadi conda environment.

#### References:

Bony, S., Semie, A., Kramer, R. J., Soden, B., Tompkins, A. M., & Emanuel, K. A. (2020). Observed modulation of the tropical radiation budget by deep convective organization and lower-tropospheric stability \[e2019AV000155 10.1029/2019AV000155\]. AGU Advances, 1 (3), e2019AV000155. https ://doi.org/https://doi.org/10.1029/2019AV000155

Houze, R. A. (1977). Structure and dynamics of a tropical squall–line system. Monthly Weather Review, 105 (12), 1540–1567. https://doi.org/10.1175/1520-0493(1977)105⟨1540:SADOAT⟩2.0.CO;2

Madden, R. A., & Julian, P. R. (1971). Detection of a 40–50 day oscillation in the zonal wind in the tropical pacific. Journal of Atmospheric Sciences, 28 (5), 702–708. https://doi.org/10.1175/1520-0469(1971)028⟨0702:DOADOI⟩2.0.CO;2

Mandorli, G., & Stubenrauch, C. J. (2023). Assessment of object-based indices to identify convective organization. EGUsphere, 2023, 1–29. https://doi.org/10.5194/egusphere-2023-1985

Mapes, B. E., & Houze, R. A. (1993). Cloud clusters and superclusters over the oceanic warm pool. Monthly Weather Review, 121, 1398–1416. https://doi.org/10.1175/1520-0493(1993)121⟨1398:CCASOT⟩2.0.CO;2

Tompkins, A. M., & Semie, A. G. (2017). Organization of tropical convection in low vertical wind shears:Role of updraft entrainment. Journal of Advances in Modeling Earth Systems, 9 (2), 1046–1068. https://doi.org/https://doi.org/10.1002/2016MS000802