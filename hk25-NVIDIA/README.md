# AI Representations with NVIDIA (hk25-NVIDIA)

NVIDIA has been exploring the use of AI to improve the workflows associated with km-scale models.  For the Hackathon they have developed a prototype emulator allows massive compression of km-scale data, which then can provide a platofrm for accelerated analysis.

In this cross-cutting activity we will evaluate the capabilities and limitations of the emulator, and its ability to learn differences among km-scale models. 

**Note** Participants are asked to register for the team as early as possible so the varied skill levels can be evalauted and approrpiate activities defined. 

**Coordination**: Karthik Kashinath (kkashinath@nvidia.com), Noah Brenowitz (nbrenowitz@nvidia.com)

#### Sketch of initial activities:
* 45-minute technical talk on generative diffusion model-based emulation of km-scale climate data (day 1)
* 60-minute hands-on tutorial session on training and inferencing diffusion models (day 1)
* AI inference of model trained on an earlier version of ICON and ERA5 to study limits of climate faithfulness
* Extend AI training to additional models hosted on NERSC mode
* Scale out the AI traning to other hackathon nodes and models
  
* Project Description

Title: Climate in a Bottle: A generative foundation model for the kilometer-scale atmosphere

Key Words: Climate modeling, generative AI, diffusion models, high-resolution climate data, HEALPix grid, super-resolution, climate emulation

Summary: The Climate in a Bottle (cBottle) project aims to emulate the entire globe at 5km-resolution (~12M pixels) using a diffusion-based foundation model without autoregression. This model generates km-scale images for dozens of observables of the Earth's atmosphere by applying diffusion on the sphere discretized with the equal-area HEALPix grid. The framework consists of a globally-trained coarse-resolution image generator followed by a locally-trained super-resolution stage. The coarse model incorporates higher noise levels and spatio-temporal embedding to produce a reasonable seasonal and diurnal cycle, while the super-resolution models of the raw 5km data are made affordable by training on subdomains and using an overlapping multidiffusion for global inferences. We subject cBottle to a battery of climate diagnostics and showcase novel use-cases enabled by our diffusion modeling approach, such as zero-shot bias correction and global climate down-scaling.

The main implications for the climate community are:
1.	O(1000)x compression - changes paradigm of km-scale data access and informatics. 
2.	Domain adaptation/transfer, inpainting, channel filling - opens door to multimodal km-scale foundation model of climate.
3.	Climate modeling centers can train their own emulators instead of delivering large data sets. Sample efficiency, patch-based training, and performance optimization makes model training feasible in O(5k) GPU-hours, inference O(secs). 

Key Results:

•	Super-resolution model represents 256x compression of petabytes of years-long km-scale simulations trained on as little as 4 weeks of simulation output. The end-to-end synthesis pipeline is a 3000x compression. Compression is quoted as the input size / output size.

•	High-quality end-to-end generation of 5km global fields (12.5 M pixels)
 ![image](https://github.com/user-attachments/assets/6208a342-7218-419d-93dc-d946fbdad2a5)

•	Climate faithfulness in its ability to generate realistic climatology, climate variability across timescales, and produce realistic extreme weather statistics. Potential applications to insurance and infrastructure planning.
 ![image](https://github.com/user-attachments/assets/b3b3104c-8e7f-4694-9f8f-67a67146a6cd)

•	Domain adaptation between ERA5 and ICON (bias correction, channel in-filling, and global downscaling). Down-scaling is achieved by applying the ICON super-resolution model in a zero-shot manner with ERA5 inputs. 
![image](https://github.com/user-attachments/assets/d2d0db10-4b13-4112-941e-4dd16a4d4231)

This schematic summarizes the overall framework:
![image](https://github.com/user-attachments/assets/f59c83a0-ba51-4278-a6c7-a826f05a9556)


Limitations:

•	Non-stationary trends such as how heat waves vary under climate change are not optimally represented in the current version of cBottle, and the model varies too much from the diurnal cycle. We expect these can be addressed by training on more data and further tuning of the noise schedule.
•	Lack of temporal coherence. The framework cannot diagnose the duration of events or commonly used metrics like the return time of an event. We are working on extending a next generation version of the macroscale generator that uses video diffusion to the full suite of diagnostics.

