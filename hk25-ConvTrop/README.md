# Project Title: Convectively-Coupled Systems in the Tropics as Simulated in Global Storm Resolving Models (hk25-ConvTrop)

### Project Details

#### Project Lead: Martin Singh / Reyhan Respati (reyhan.respati@monash.edu)

#### Project members:

#### Number of open slots for students: \~3

#### Expertise needed:

Someone familiar with HEALPix grid and km-scale model output processing

### Project Description

#### Background:

Convectively-coupled large-scale systems in the tropics can be categorised into: 1\) slow moving moisture modes, 2\) fast moving inertio-gravity waves, and 3\) mixed systems in between. This is the ***moisture mode-to-gravity wave spectrum*** framework (see [Adames et al 2019](https://doi.org/10.1175/JAS-D-19-0121.1) and [Adames 2022](https://doi.org/10.1175/JAS-D-21-0215.1)). Moisture mode controls the convection through moistening the tropospheric column, while inertio-gravity wave alters the low-level buoyancy to trigger convection. This project is aimed to find out whether this distinction in how large-scale tropical systems govern the convection is well simulated by the km-scale models.

#### Primary research question:

What are the simulated characteristics of slow and fast moving large-scale convectively-coupled tropical systems in the high-resolution models?

#### Secondary research questions:

How are they compared to the observation?

#### Primary output:

Thermodynamic characteristics (e.g., vertical profile of q’ and T’) and dynamic characteristics (e.g., low-level horizontal wind structure) of slow moving moisture modes and fast moving inertio-gravity waves simulated by the models

#### Secondary outputs:

Rainfall characteristics associated with slow and fast moving tropical systems

### Methodology

#### Datasets: 

Simulation outputs: rlut, pr, ua, va, wa, zg, ta, hus  
Observation datasets: ERA5 (same variables as above), GPM-IMERG precipitation  
Domain: global tropics (30S-30N)

#### Methods:

* Separating slow vs fast moving systems based on their phase speed using 2D wavenumber-frequency filtering  
* Object identification and tracking using a tracking algorithm (e.g., TOBAC)  
* Composite averaging and statistical significance testing

Each bullet point can be done by a project member, but unfortunately this does not allow for parallel working (i.e., filtering \-\> tracking \-\> compositing in order)