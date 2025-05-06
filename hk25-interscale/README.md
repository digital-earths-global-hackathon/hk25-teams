# Science, Tutorials and Cross-cutting Activities

This repository collects the work of different science teams  and cross-cutting activities during the global hackathon. It serves as a collaboration platform and provides information about for participants.

In the [./hk25-tutorials](https://github.com/digital-earths-global-hackathon/hk25-teams/blob/main/hk25-tutorials) you will find some simple **tutorials** to get familiar with the data.

## Contribute with your own idea

Create you own science team or cross cutting activity! Open a pull request and create a directory with a  **unique identifier (uid)** following the pattern `hk25-uid`[^1]. Add the description of your science team or cross-cutting activity in a README file. You can follow the scheme provided by the list of [existing teams](https://digital-earths-global-hackathon.github.io/hk25/scienceteams/) or [activities](https://digital-earths-global-hackathon.github.io/hk25/crosscutting/) . Once you have your science team, or cross-cutting activity use this repository to upload scripts, tutorials and more.

[^1]: Ideally a short descriptive names.  See the existing teams and activities for some examples.

# Inter-Scale Energy Transfers in the Atmosphere (hk25-InterScale)

Many atmospheric phenomena, especially in the tropics, are scale-coupled. Small-scale convection (scales much less than 100 km) tends to aggregate into meso-scales (100-500 km), and both of these smaller-scale processes feed upscale onto much larger phenomena - e.g. MCSs, Tropical Cyclones, Equatorial Waves and the MJO. The reverse is also true through downscale energy transer, and is more well convered in the literature. This team aims to explore the inter-scale energetics of the atmosphere using the [LoSSETT](https://github.com/ElliotMG/LoSSETT) tool to extract the $\mathcal{D}_\ell$ term (see LoSSETT README), and make progress toward the following with similarly interested collaborators:

#### Outline of planned activities
* Run LoSSETT on the global year-long simulations to extract a climatology of $\mathcal{D}_\ell$
* Explore the inter-scale energetic structure of select atmospheric phenomena
* Understand the transfer of energy near the grid-scale of these km-scale simulations
* Examine values of the same $\mathcal{D}_\ell$ at different HEALPix levels to ensure energy conservation away from filter scale

**Coordination**: Dan Shipley (daniel.shipley@reading.ac.uk), Elliot McKinnon-Gray (e.b.mckinnon-gray@pgr.reading.ac.uk)
