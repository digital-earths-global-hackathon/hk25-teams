# Generic DGGS Format Specification (hk25-EOPF_DGGS)

We aim to make the **HEALPix coordinate system** more accessible and user-friendly, enabling scientists and engineers to seamlessly work with diverse geospatial datasets (in-situ, Satelite, model output). Our efforts focus on simplifying manipulation and analysis workflows for **Discrete Global Grid Systems (DGGS)**, particularly HEALPix-based data, with a strong emphasis on interoperability and scalability.

We have been developing a suite of tools, including:

- [xdggs](https://xdggs.readthedocs.io/en/latest/)
- [healpix-convolution](https://healpix-convolution.readthedocs.io/en/latest/)
- [xhealpixify](https://xhealpixify.readthedocs.io/en/latest/)
- [foscat]()
- Jupyter Books for initiating HEALPix usage workflows.  

These tools are built within the **PANGEO ecosystem** (Xarray, Zarr, Dask, Jupyter, fsspec, ..). Through the [EOPF DGGS project](https://github.com/EOPF-DGGS/), we are now evolving them into a **generic DGGS format**, enabling datasets such as **Sentinel-2** and **Sentinel-3** to take full advantage of **HEALPix-based Zarr storage**.

**Coordination:** Tina Odaka (tina.odaka@ifremer.fr)

---

## Cloud Access

For those who cannot join on-site hackathon nodes, cloud infrastructure will be made available via the [PANGEO-EOSC JupyterHub](https://pangeo-data.github.io/pangeo-eosc/), thanks to our collaboration with **EGI**.

🔹 **Request Access:**  
If you wish to use this infrastructure, please follow the instructions on the [PANGEO-EOSC getting started guide](https://pangeo-data.github.io/pangeo-eosc/users/users-getting-started.html) and indicate your participation in this hackathon.

---

## Initial Activities

Our primary objective is to define and prepare the foundational structure of a DGGS format tailored for **Sentinel** and **DestinE** datasets. This initiative builds upon the work we carried out during the [**Hack4RioMar** hackathon](https://fair2adapt.github.io/Hack4RiOMAR/).  

### 1. Zarr v3 on HEALPix

- Implement and test **sharded Zarr v3** with optimal chunking strategies for HEALPix datasets.
- Integrate with `icechunk`, `zarr3-python`, and `xarray`.

### 2. Multiresolution Data Access

- Use **Sentinel-2 L2A** data as a case study for pyramidal representation across resolutions.
- Leverage **cloud-hosted datasets** via the **[EOPF Zarr Sample Service](https://github.com/EOPF-Sample-Service/eopf-sample-notebooks/tree/main)** and [it's STAC API](https://stac.browser.user.eopf.eodc.eu)
- Align temporal selection with *Takasuka et al. (2024)*.

### 3. Interoperability

- Advance support for **CF Conventions**, **GeoZarr**, and **WGS84 compatibility** in HEALPix/Zarr structures.
- Collaborate with the **ICON/HEALPix-Zarr format initiative** and the **EOPF DGGS team** to ensure broad interoperability and community-driven standardization.
