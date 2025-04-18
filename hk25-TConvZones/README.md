# Project Title: Unravelling the representation of Tropical Convergence Zones in *km-scale* model runs (hk25-TConvZones)

### Project Details

#### Project Lead: Aditya Sengupta (aditya.sengupta@student.unimelb.edu.au) & co-lead \[open for Research Fellows/Students\]

#### Project members: 

#### Number of open slots for students: 6–8

#### Expertise needed:

Expertise in handling daily and sub-daily rainfall data; tropical rainfall dynamics; ITCZ and SPCZ theory and knowledge on coupled/atmosphere-only model ITCZ/SPCZ biases ***is appreciated but not required***.  

### Project Description

#### Background:

The biases in the representation of tropical rainfall patterns, particularly arising from systematic biases in tropical convergence zones, have been a long-standing aspect across generations of climate models (Fiedler et al., 2020\). 

*Double ITCZ:*

One such bias is the double Inter-tropical Convergence Zone (ITCZ) bias in coupled models and heavy rainfall biases in atmosphere-only models (Kim et al., 2021; Zhou et al., 2022; see Figure 1\). Using a set of CMIP and AMIP models, Zhou et al., (2022) highlighted that the large-scale double ITCZ bias in CMIP models is linked to the local scale rainfall drizzling bias in AMIP models. Other studies have highlighted that this bias is linked with SST biases in the South-east Pacific Ocean (E. Dong et al., 2025\); mean vertical circulation (Respati et al., 2024\) and frequency of deep/strong convection (Ma et al., 2023\).

While high resolution models show an improvement in simulating tropical rainfall, they do not eliminate the double ITCZ bias (Y. Dong et al., 2022; Ma et al., 2023; Zhou et al., 2022\), with virtually no improvement seen in atmosphere only high-res models (Moreno-Chamarro et al., 2022\), owing primarily to similar drizzling biases and biases in convective regimes. While these studies are very comprehensive, they have only used mesoscale-resolving models and in cases where they have used storm-resolving km-scale models, they only use short one-month long runs. In this Hackathon, we have a year-long km-scale resolution model output. This will be used to compare the improvements or lack thereof in the bespoke run with observations and low resolution model runs in order to gain a process-oriented understanding of the improvements.

   
![](tconvzones1.png)
**Fig. 1** a) Spatial patterns of annual-mean tropical precipitation bias in CMIP6 simulations (CMIP6 \- GPCP) and b) decomposition into the part that exists in AMIP simulations without the SST bias (AMIP6 \- GPCP) and c) the part that arises from the SST biases (CMIP6 \- AMIP6). The black contours indicate the annual-mean precipitation climatology (drawn at 5, 8, and 11 mm day21). (d) Annual-mean zonal-mean precipitation in GPCP, AMIP6 and CMIP6. (e) Annual-mean zonal-mean precipitation bias that exists without the SST bias (orange) and that arises after coupling to ocean (blue). Source: (Zhou et al., 2022\)

*Seasonal cycle:*

![](tconvzones2.png)
**Fig. 2** Climatological location of the ITCZ during 1983-2005, during different seasons, based on observations and CMIP6 models. Some well-known distinct ITCZ features are highlighted in the results from the observations, while the double-ITCZ biases in the eastern Pacific and Atlantic basins are apparent in the CMIP6 results (season Nov-Apr). The areas over which the double-ITCZ biases are quantified are shown as red boxes in panel (d). Source: Mamalakis et al., (2021)

Improvements in regional tropical rainfall patterns in high-resolution models are mainly linked with improvements in orography and also due to the improvements in the seasonal cycle of the ITCZ (Doi et al., 2012\). These improvements in the meridional migration of the ITCZ in high-resolution coupled models has been linked to improvements in the seasonal changes in SST patterns, but in high-resolution atmosphere-only models, these improvements are largely associated with improvements in the seasonal wind-reversal patterns in models (Song & Zhang, 2020\). Given this relationship between wind and rainfall biases, we can use the km-scale model runs to compare seasonal cycle of rainfall and seasonal wind-reversal patterns.

#### Primary research question:

- How well do high resolution “*km-scale*” storm-resolving models simulate the Tropical Convergence Zones and its seasonal cycle over different ocean basins, and what are the reasons for these improvements?

#### Secondary research questions:

- If the double ITCZ bias is reduced in the “*km-scale*” model runs, then are the improvements related to the improvements in drizzling bias in these models?  
    
- Are the improvements in the seasonal cycle of the ITCZ related to improved representation of winds and seasonality of wind-reversal associated with major monsoon systems in the tropics?  
    
- Are other major convergence zones, such as the SPCZ, SACZ, etc. in the tropics (shown in Figure 2), better represented in the “*km-scale*” model runs?

#### Primary output:

A report with detailed comparison of the “*km-scale*” model run with observations and potentially some low resolution CMIP models and high-resolution HighResMIP models.

#### Secondary outputs:

Ideally we could potentially spin this into a paper depending on our findings but at minimum, it would be great to have a detailed document which could be potentially used as a briefing note for the Centre in the future or potentially a repository of code for analysing mean state features in storm-resolving models.

### Methodology

#### Datasets: 

Variables from the *km-scale* atmosphere only model run: 

- Precipitation rate/accumulated rainfall totals,   
- Low-level cloud fraction (LLC; if available),   
- ToA radiative fluxes;  
- Wind at different pressure levels;  
-  Surface specific humidity.   
- Total column water vapour (TCWV)

Same variables required from CMIP6/AMIP6 models and possibly at least one HighResMIP run, but we can access them through NCI.

In addition, we will require access to observation datasets for 

- precipitation (GPM-IMERG or GPCP or TRMM);   
- CERES dataset for ToA radiative fluxes;  
- cloud fraction data from CALIPSO;  
- Wind speed data from reanalysis datasets (20CR or ERA5)  
- Observation or reanalysis TCWV

#### Methods:

*Initial Analysis:*

Initial analysis will involve familiarising ourselves with the new HEALPix grid in which the data will be provided, and then finalising a work plan for how to approach our research questions or potentially come up with alternative research questions. A rough methodological approach is provided below, but there is room to change

*For Double ITCZ analysis:*

We will use established techniques and indices to calculate the double ITCZ bias in models and compare with observations. Following on from this, we will analyse the sources of these biases using methods outline in the paper by ***Zhou et al., (2022)***, where they analyse the surface radiative flux biases and the drizzling biases in different convective regimes in AMIP models. Too frequent no-rain/light-rain events and infrequent heavy-rain events induces hemispherically symmetric rainfall biases in AMIP models. This drizzling bias also induces a hemispherically asymmetric cloud radiative effect bias in the southern tropics which in-turn induces an SST bias in the coupled models leading to the double ITCZ bias. We will analyse if these biases are present in our *km-scale* model run. 

*For ITCZ seasonal cycle:*

This analysis could be done in a few different ways \-

1. First Analysing the seasonal average ITCZ patterns and comparing it with observations and low resolution model runs;  
     
2. Second, by comparing the seasonal meridional movement of the ITCZ peak rainfall region/peak wind region using a centroid detection analysis to check if the extent of the observed meridional movements are captured by the high resolution model.

We can compare the wind-reversals during monsoon season to compare how well regional rainfall patterns are captured during tropical monsoon season in different regions. There is room for more ideas in this part of the project, and lots of scope to learn new things\!\!

#### References

Doi, T., Vecchi, G. A., Rosati, A. J., & Delworth, T. L. (2012). *Biases in the Atlantic ITCZ in Seasonal–Interannual Variations for a Coarse- and a High-Resolution Coupled Climate Model*. https://doi.org/10.1175/JCLI-D-11-00360.1

Dong, E., Song, F., Wu, L., Dong, L., Wang, S., Liu, F., & Wang, H. (2025). The Process-Oriented Understanding on the Reduced Double-ITCZ Bias in the High-Resolution CESM1. *Geophysical Research Letters*, *52*(1), e2024GL112087. https://doi.org/10.1029/2024GL112087

Dong, Y., Armour, K. C., Battisti, D. S., & Blanchard-Wrigglesworth, E. (2022). *Two-Way Teleconnections between the Southern Ocean and the Tropical Pacific via a Dynamic Feedback*. https://doi.org/10.1175/JCLI-D-22-0080.1

Fiedler, S., Crueger, T., D’Agostino, R., Peters, K., Becker, T., Leutwyler, D., Paccini, L., Burdanowitz, J., Buehler, S. A., Cortes, A. U., Dauhut, T., Dommenget, D., Fraedrich, K., Jungandreas, L., Maher, N., Naumann, A. K., Rugenstein, M., Sakradzija, M., Schmidt, H., … Stevens, B. (2020). *Simulated Tropical Precipitation Assessed across Three Major Phases of the Coupled Model Intercomparison Project (CMIP)*. https://doi.org/10.1175/MWR-D-19-0404.1

Kim, H., Kang, S. M., Takahashi, K., Donohoe, A., & Pendergrass, A. G. (2021). Mechanisms of tropical precipitation biases in climate models. *Climate Dynamics*, *56*(1), 17–27. https://doi.org/10.1007/s00382-020-05325-z

Ma, X., Zhao, S., Zhang, H., & Wang, W. (2023). The double‐ITCZ problem in CMIP6 and the influences of deep convection and model resolution. *International Journal of Climatology*, *43*. https://doi.org/10.1002/joc.7980

Mamalakis, A., Randerson, J. T., Yu, J.-Y., Pritchard, M. S., Magnusdottir, G., Smyth, P., Levine, P. A., Yu, S., & Foufoula-Georgiou, E. (2021). Zonally contrasting shifts of the tropical rain belt in response to climate change. *Nature Climate Change*, *11*(2), 143–151. https://doi.org/10.1038/s41558-020-00963-x

Moreno-Chamarro, E., Caron, L.-P., Loosveldt Tomas, S., Vegas-Regidor, J., Gutjahr, O., Moine, M.-P., Putrasahan, D., Roberts, C. D., Roberts, M. J., Senan, R., Terray, L., Tourigny, E., & Vidale, P. L. (2022). Impact of increased resolution on long-standing biases in HighResMIP-PRIMAVERA climate models. *Geoscientific Model Development*, *15*(1), 269–289. https://doi.org/10.5194/gmd-15-269-2022

Respati, M. R., Dommenget, D., Segura, H., & Stassen, C. (2024). Diagnosing drivers of tropical precipitation biases in coupled climate model simulations. *Climate Dynamics*, *62*(9), 8691–8709. https://doi.org/10.1007/s00382-024-07355-3

Song, F., & Zhang, G. J. (2020). *The Impacts of Horizontal Resolution on the Seasonally Dependent Biases of the Northeastern Pacific ITCZ in Coupled Climate Models*. https://doi.org/10.1175/JCLI-D-19-0399.1

Zhou, W., Leung, L. R., & Lu, J. (2022). *Linking Large-Scale Double-ITCZ Bias to Local-Scale Drizzling Bias in Climate Models*. https://doi.org/10.1175/JCLI-D-22-0336.1