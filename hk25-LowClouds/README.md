# Towards enhancing climate projections through improved understanding and simulations of marine low clouds (hk25-LowClouds)

### Project Details

#### Project Lead: Qinggang Gao (qinggang.gao@unimelb.edu.au)

#### Project members:

#### Number of open slots for students: 4-6

#### Expertise needed:

Python. Experience with shallow convection is highly appreciated but not required.

### Project Description

#### Background:

The response of low clouds to warming is the greatest source of uncertainty in climate projections1. Clouds have a net radiative cooling effect (around \-20 W m\-2) on Earth’s radiation budget due to compensating shortwave (-47 W m\-2) and longwave (28 W m\-2) radiation effects2. Changes in cloud coverage, optical depth, and albedo can either amplify or moderate Earth’s climate sensitivity3–5. This sensitivity is typically quantified as the rise in equilibrium global mean temperature resulting from a doubling of atmospheric CO2 concentration relative to a preindustrial reference state2. The Intergovernmental Panel on Climate Change Sixth Assessment Report (IPCC AR6) assesses the equilibrium climate sensitivity to be very likely within the range of 2-5°C based on multiple lines of evidence, while excluding direct estimates from Coupled Model Intercomparison Project Phase 6 (CMIP6) model simulations. This is partly because, despite decades of development, low cloud feedback leads to larger model spread in CMIP6 than CMIP56. More accurate simulations of low cloud feedback are essential for improving model estimates of climate sensitivity7.  
   
Current climate models struggle to accurately represent clouds8, particularly low clouds driven by shallow convection9. Model resolution is a fundamental limitation, as the key processes governing marine low clouds occur at scales of tens of meters or smaller, which are too small to be resolved by global climate models used for future projections, where grid spacings are on the order of tens of kilometers. Moist convection thus needs to be parameterized in these models, which is hardly constrained by observations or theories and leads to large uncertainties in model simulations. Advances in high-performance computing have made it possible to conduct kilometer-resolution simulations at regional scales over multiple years and at global scales for several months. While shallow convection can be explicitly resolved rather than parameterized at kilometer-resolution, other relevant processes for low clouds such as boundary layer mixing remain to be parameterized. Although global kilometer-resolution simulations may become feasible for climate projections by 2060s1, higher resolution alone does not guarantee more accurate predictions10. Indeed, our preliminary analysis indicates that a high-resolution reanalysis focused on Australia from Bureau of Meteorology exhibits even larger bias in Earth’s reflected solar radiation than a global low-resolution reanalysis and the CMIP6 *historical* model ensemble, when compared to a satellite product (Fig. 1). This is attributed to an overestimation of low cloud cover in the high-resolution reanalysis, which reflects too much solar radiation. Understanding the mechanisms driving the model bias is crucial to leverage the benefits of increased model resolution.  
![](lowclouds1.png)

**Figure 1**. 2001-2014 annual mean top of atmosphere (TOA) upward shortwave (SW) radiation in (a) Clouds and the Earth’s Radiant Energy System (CERES) Energy Balanced and Filled TOA satellite radiation product11, and its difference with (b) the European Centre for Medium-Range Weather Forecasts Reanalysis v5 (ERA5)12, (c) the ensemble mean of 49 CMIP6 *historical* model simulations, and (d) Bureau of Meteorology Atmospheric high-resolution Regional Reanalysis for Australia \- Convective-Scale Version 2 (BARRA-C2). The reanalysis products combine numerical models with observations to provide a best estimate of the atmospheric states. The horizontal grid spacings are around 110 km for CERES, 30 km for ERA5, 150-200 km for CMIP6 *historical*, and 4.4 km for BARRA-C2.

#### Primary research question:

How do the km-scale climate models represent TOA upward shortwave radiation and low clouds compared to satellite products (CERES and Himawari)?

#### Secondary research questions:

What is the impact on surface energy balance?  
Is there land-sea contrast? If so, why?  
Are there improvements compared to CMIP6 model simulations?

#### Primary output:

A detailed understanding of model-data discrepancies and the benefit of high-resolution simulations.

#### Secondary outputs:

If we can build a solid understanding of the model behaviours and limitations, we can consider to draft a short manuscript.

### Methodology

#### Datasets: 

2D: hourly global surface and TOA radiative dluxes, total/high/middle/low cloud cover  
3D: 6-hourly global temperature, humidity, wind, cloud liquid/ice water.  
Other datasets: CERES, Himawari, CloudSat/CALIPSO, CMIP6 simulations (I can easily share if required)

#### Methods:

Climatological analyses can be combined with case studies using different datasets. These tasks can be splitted into groups based on interests.  
 

#### References

1\.       	Schneider, T. *et al.* Climate goals and computing the future of clouds. *Nat Clim Chang* 7, 3–5 (2017).

2\.       	IPCC. *Climate Change 2021: The Physical Science Basis. Contribution of Working Group I to the Sixth Assessment Report of the Intergovernmental Panel on Climate Change*. vol. In Press (Cambridge University Press, Cambridge, United Kingdom and New York, NY, USA, 2021).

3\.       	Schneider, T., Kaul, C. M. & Pressel, K. G. Possible climate transitions from breakup of stratocumulus decks under greenhouse warming. *Nat Geosci* 12, 163–167 (2019).

4\.       	Vogel, R. *et al.* Strong cloud–circulation coupling explains weak trade cumulus feedback. *Nature* 612, 696–700 (2022).

5\.       	Sherwood, S. C., Bony, S. & Dufresne, J. L. Spread in model climate sensitivity traced to atmospheric convective mixing. *Nature* 505, 37–42 (2014).

6\.       	Zelinka, M. D. *et al.* Causes of Higher Climate Sensitivity in CMIP6 Models. *Geophys Res Lett* 47, e2019GL085782 (2020).

7\.       	Schneider, T., Ruby Leung, L. & Wills, R. C. J. Opinion: Optimizing climate models with process knowledge, resolution, and artificial intelligence. *Atmos Chem Phys* 24, 7041–7062 (2024).

8\.       	Hyder, P. *et al.* Critical Southern Ocean climate model biases traced to atmospheric model cloud errors. *Nat Commun* 9, 1–17 (2018).

9\.       	Bony, S. *et al.* Clouds, circulation and climate sensitivity. *Nat Geosci* 8, 261–268 (2015).

10\.    	Heim, C., Hentgen, L., Ban, N. & Schär, C. Inter-model Variability in Convection-Resolving Simulations of Subtropical Marine Low Clouds. *Journal of the Meteorological Society of Japan. Ser. II* 99, 1271–1295 (2021).

11\.    	Loeb, N. G. *et al.* Clouds and the Earth’s Radiant Energy System (CERES) Energy Balanced and Filled (EBAF) Top-of-Atmosphere (TOA) Edition-4.0 Data Product. *J Clim* 31, 895–918 (2018).

12\.    	Hersbach, H. *et al.* The ERA5 global reanalysis. *Quarterly Journal of the Royal Meteorological Society* 146, 1999–2049 (2020).