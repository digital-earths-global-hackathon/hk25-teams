# Energetics of Tropical Rainbelts  (hk25-RainBelt)

Global km-scale grids permits an explicit representation of convective storms and the processes they entails. Over the tropical ocean, precipitation occurs in a variety of environments. Precipitating regions could be related to strong sea surface temperature gradients and a bottom-heavy circulation (e.g., Eastern Pacific) or a top-heavy circulation, weak SST gradients, and light winds (e.g., Western Pacific). Due to the diversity of pathways in which precipitation occurs, we will analyze how convection is represented across the tropical oceans in the different participating km-scale models using an energetic-moisture framework.

**Coordination**: Hans Segura Cajachagua (hans.segura@mpimet.mpg.de)

## The Framework
To give a framework to our analysis, the following equation is used [Raymond et al., 2009](https://doi.org/10.3894/JAMES.2009.1.9): 

$$ P - E = \frac{Q_\mathrm{rad}+Q_\mathrm{sfc}}{\Gamma}$$

where $P$ is precipitation, $E$ is evaporation, $\Gamma$ denotes the normalized gross moist stability, $Q_{\mathrm{rad}}$ is radiative heating in the total atmospheric column, $Q_{\mathrm{sfc}}$ is heating from surface fluxes. In this definition, $\Gamma$ is ratio between the export of energy from the column and the import of moisture into the column: 

$$\Gamma =  \frac{\nabla.\langle \overrightarrow{V}h \rangle}{\nabla.\langle \overrightarrow{V}q \rangle}$$

where $\langle \rangle$ indicates vertical integration, $\overrightarrow{V}$ is the wind vector, $h$ is the moist static energy, $q$ is specific humidity, and $\nabla$ is the gradient. The vertical component of $\Gamma$ ($\Gamma_\mathrm{v}$) gives the ratio between the vertical advection of energy in the column and the import of moisture into the column:

$$\Gamma_\mathrm{v} =  \frac{\langle \omega \frac{\mathrm{d}h}{\mathrm{d}p} \rangle}{\nabla.\langle \overrightarrow{V}q \rangle}$$

where $\omega$ is upward motion and $\frac{\mathrm{d}h}{\mathrm{d}p}$ is the vertical gradient of moist static energy. 

This framework will guide the analyses of km-scale climate simulations by taking into account the following questions:
* 1) How different is the representation of the tropical rainbelt and warm pool precipitation in the Western Pacific?
* 2) Are the differences in the Western Pacific related to $Q_\mathrm{rad}$, $Q_\mathrm{sfc}$, or $\Gamma$?
* 3) If differences in $Q_\mathrm{rad}$ are big, are they related to ice concentration? 
* 4) Are the differences in $Q_\mathrm{sfc}$ related to surface winds? and how is this related to the energetics in the boundary layer?
* 5) Can we relate the differences in $\Gamma$ to the effiency of the model to export vertically energy and import moisture into the column $\Gamma_\mathrm{v}$?
* 6) How is representation of convectively coupled equatorial waves?

Please note that these questions could be changed throughout the hackathon. 

## The Group's dynamic
### Group effort
These questions will be devided in 5 groups. Group 1 will be in charge of questions 1 and 2, Group 2 for question 3, Group 3 for question 4, Group 4 for question 5, and Group 5 for question 6. 

#### Sketch of initial activities
* identify the tropical rainbelt
* calculate the entropy forcing and the net precipitation flux at the surface
* compute the type of circulation (top- or bottom-heavy) in the tropical rainbelt
* extract the spectrum of convective coupled equatorial waves 

Some initals example of using data are in : hk25-RainBelt/scripts/scripts_hansS/
### Communication
The global hackathon is a great opportunity to work with our peers across the world. To enhance this synergy one or two zoom meeting with a duration of 1 hour are planned. The time for the meeting(s) will be decided in a way to involve all nodes working in this subject. 
Moreover, we recommend people assisting the same node to find a common place to hack.

## Recomendations
Please take into account these steps: 
* Follow  [the how to hack](https://digital-earths-global-hackathon.github.io/hamburg-node/howtotech/) and the [how2hack](https://github.com/digital-earths-global-hackathon/hk25/blob/main/content/how2hack.md)
* The hk25-RainBelt will coordinate the activities using github issues. An issue could be: "Representation of Tropical Rainbelt". So, this is issue will be only dedicated to the analysis of the representation of the tropical rainbelt. This means that figures and discussions about the tropical rainbelt will be posted in this issue.
* It is highly recommended that after finishing the analysis, the final scripts and the description of the analysis should be merged into the repository. To do this, follow the following steps:
1) Clone the repository (git clone ...)
2) Create a branch (git branch "Name")
3) Add an folder in scripts/ with the following convention: scripts_[XXX], where XXX is any kind of abreviation (name, surname, etc) that will facilitate the identification of the owner of the folder. Please do not modify the scripts in other folders.
4) Add a small resume of the analysis of the group in the docs/ folder. The file of the description could be .md or .rst
5) Add and commit the changes (git add | git commit).
6) Push the changes (git push) and open a request associated with the issue. 
