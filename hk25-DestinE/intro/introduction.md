# Destination Earth with the Climate DT Consortium

### Helpful links
- [Insula Code](https://platform.destine.eu/services/service/insula-code/)
- [MARS requests catalogue](https://climate-catalogue.lumi.apps.dte.destination-earth.eu/?root=root)
- [Climate DT overview](https://destine.ecmwf.int/climate-change-adaptation-digital-twin-climate-dt/)
- [Earthkit docs](https://earthkit.readthedocs.io/en/latest/)

## Accessing DestinE data
### Starting with DESP

The following steps assumes one has been granted upgraded access to [DESP](https://platform.destine.eu).

To explore the ClimateDT data we can use the Insula - Code service. To use it one first must go to [Insula Code](https://platform.destine.eu/services/service/insula-code/) and then sign in (upper left corner). Once one is logged in click "Go to service". Then a server will be started that will launch a jupyter lab. There are multiple folders to begin with. Select `polytope-lab` -> `climate-dt`. There are multiple example jupyter notebooks in this folder which can be used as basis for any analysis

### Starting outside of DESP
A python environment to start working outside of DESP can be created using the `requirements.txt` provided in this folder.
There are different ways how one can enable access to the data-bridge where the DestineE data is stored. To read the data outside of DESP one still requires a token. This token can be generated the following way:

1. Copy the `desp-authentication.py` file from the jupyterlab Insula Code service in `polytope-lab` to your laptop.
2. Same as in the example jupyter notebooks run the following lines of code in your python notebook/script:
```python
%%capture cap
%run path/to/desp-authentication.py
output_1 = cap.stdout.split('}\n')
access_token = output_1[-1][0:-1]
```
3. Create a request (see below) and run `earthkit.data.from_source(<args>)` (see Insula Code examples) to get the data.

### Requesting data
The data from Climate DT is hosted remotely. Therefore, the data must be requested so that it can be used. The data requests follow the syntax of the [MARS](https://confluence.ecmwf.int/display/UDOC/MARS+command+and+request+syntax) archive. This could look for instance like the following example from climate-dt-earthkit-example-domain.ipynb with pointers to what one may want to change:

```
request = {
    'activity': 'ScenarioMIP',   <- What type of run
    'class': 'd1',                  
    'dataset': 'climate-dt',        
    'date': '20200102',          <- What date is requested
    'experiment': 'SSP3-7.0',
    'expver': '0001',
    'generation': '1',
    'levtype': 'sfc',            <- What type of level
    'model': 'IFS-NEMO',         <- Which model
    'param': '134/165/166',      <- Which variables using their grib IDs
    'realization': '1',
    'resolution': 'standard',
    'stream': 'clte',
    'time': '0100',              <- What time should be selected
    'type': 'fc'
}
```
The data can then be requested using earthkit and can be converted to an xarray.Dataset if wanted. Other request examples can be created in [MARS requests catalogue](https://climate-catalogue.lumi.apps.dte.destination-earth.eu/?root=root).