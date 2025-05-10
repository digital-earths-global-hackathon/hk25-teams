# AI Representations with NVIDIA (hk25-NVIDIA)

NVIDIA has been exploring the use of AI to improve the workflows associated with km-scale models.  For the Hackathon they have developed a prototype emulator allows massive compression of km-scale data, which then can provide a platofrm for accelerated analysis.

In this cross-cutting activity we will evaluate the capabilities and limitations of the emulator, and its ability to learn differences among km-scale models. 

**Note** Participants are asked to register for the team as early as possible so the varied skill levels can be evaluated and appropriate activities defined.

**Coordination**: Karthik Kashinath (kkashinath@nvidia.com), Noah Brenowitz (nbrenowitz@nvidia.com)

#### Sketch of initial activities:

* 45-minute technical talk on generative diffusion model-based emulation of km-scale climate data (day 1)
* 60-minute hands-on tutorial session on training and inferencing diffusion models (day 1)
* AI inference of model trained on an earlier version of ICON and ERA5 to study limits of climate faithfulness + visualization
* Extend AI training to additional models hosted on NERSC mode
* Scale out the AI training to other hackathon nodes and models

Some potential science questions to explore with the AI-generated km-scale fields:
- Is cloud scale organization modulated by synoptics in the right way?
- Do the Southern ocean "cloud streets" look realistic?
- Does the diurnal cycle look right over the amazon?


#### Potential ideas for hacking

- Inferences for other science questions (e.g., hk25-ConvOrg, hk25-MCS, hk25-ShallowCirc, hk25-StCu)
- Train on other GSRMs/unified training on multiple datasets
- Reducing global super-res inference latency on a single GPU from minutes to seconds

#### Project Description

Climate in a Bottle: A generative foundation model for the kilometer-scale atmosphere

Key Words: Climate modeling, generative AI, diffusion models, high-resolution climate data, HEALPix grid, super-resolution, climate emulation

Summary: The Climate in a Bottle (cBottle) project aims to emulate the entire globe at 5km-resolution (~12M pixels) using a diffusion-based foundation model without autoregression. This model generates km-scale images for dozens of observables of the Earth's atmosphere by applying diffusion on the sphere discretized with the equal-area HEALPix grid. The framework consists of a globally-trained coarse-resolution image generator followed by a locally-trained super-resolution stage. The coarse model incorporates higher noise levels and spatio-temporal embedding to produce a reasonable seasonal and diurnal cycle, while the super-resolution models of the raw 5km data are made affordable by training on subdomains and using an overlapping multidiffusion for global inferences. We subject cBottle to a battery of climate diagnostics and showcase novel use-cases enabled by our diffusion modeling approach, such as zero-shot bias correction and global climate down-scaling.

The main implications for the climate community are:
1.	O(1000)x compression - changes paradigm of km-scale data access and informatics. 
2.	Domain adaptation/transfer, inpainting, channel filling (Downscaling is achieved by applying the ICON-trained super-resolution model in a zero-shot manner with ERA5 inputs) - opens door to multimodal km-scale foundation model of climate. 
3.	Climate modeling centers can train their own emulators instead of delivering large data sets. Sample efficiency, patch-based training, and performance optimization makes model training feasible in O(5k) GPU-hours, inference O(secs).
4.	Climate faithfulness in its ability to generate realistic climatology, climate variability across timescales, large-scale modes of variability, and produce realistic extreme weather statistics. Potential applications to insurance and infrastructure planning.

This schematic summarizes the overall framework:

![image](https://github.com/user-attachments/assets/9006f23b-8092-4927-b6b0-e93a9f495539)

High-quality end-to-end generation of 5km global fields (12.5 M pixels)

![image](https://github.com/user-attachments/assets/6755c4ad-0c1c-4499-a845-60507e2cca8b)

Tropical cyclone occurrence probability derived from ERA5 (1980–2017) and cBottle (1940–2021)

![image](https://github.com/user-attachments/assets/86b5fb4f-5991-453c-aa38-82638b27289f)


Some (known) limitations of cBottle:

•	Non-stationary trends such as how heat waves vary under climate change are not optimally represented in the current version of cBottle, and the model varies too much from the diurnal cycle. We expect these can be addressed by training on more data and further tuning of the noise schedule.
•	Lack of temporal coherence. The framework cannot diagnose the duration of events or commonly used metrics like the return time of an event. We are working on extending a next generation version of the macroscale generator that uses video diffusion to the full suite of diagnostics.

#### cBottle Project GitHub
https://github.com/NVlabs/cBottle
