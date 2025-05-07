# Destination Earth with the Climate DT Consortium

### Frequently asked questions and direct links to the answer
- [Where can I run my interactive analysis? -> Insula Code](https://platform.destine.eu/services/service/insula-code/)
- [Which simulations are available? -> Climate DT simulation overview](https://destine.ecmwf.int/climate-change-adaptation-digital-twin-climate-dt/)
- [Which variables were saved? -> Data catalogue for ClimateDT](https://confluence.ecmwf.int/display/DDCZ/Climate+DT+Phase+1+data+catalogue#ClimateDTPhase1datacatalogue-Outputparameters)
- [Are there example notebooks for my interactive analysis? -> Polytope examples](https://github.com/destination-earth-digital-twins/polytope-examples/tree/main/climate-dt)
- [How do I adjust my data request? -> STAC catalogue](https://climate-catalogue.lumi.apps.dte.destination-earth.eu/?root=root)
- [Is there documentation on Earthkit? -> Earthkit docs](https://earthkit.readthedocs.io/en/latest/)
- [Can I see the status of the different data bridges somewhere? -> status.data.destination-earth.eu](https://status.data.destination-earth.eu/LUMI)

## Accessing DestinE data
### Option 1: Interactive data analysis on the Destination Earth Service Platform (DESP)

The following steps can be followed by users who have been granted upgraded access to [DESP](https://platform.destine.eu).

To explore the ClimateDT data via the DESP one can use the Insula - Code service to run Jupyter Notebooks interactively. First go to [Insula Code](https://platform.destine.eu/services/service/insula-code/), sign in (upper left corner), then click "Go to service". Then a server will be started that will launch a Jupyter lab. There are multiple folders to begin with. Select polytope-lab -> climate-dt. There are multiple example jupyter notebooks in this folder which can be used as basis for any analysis.

### Option 2: local data analysis

To explore the Climate DT data locally (e.g. on a HPC/laptop), a python environment can be created using the requirements.txt provided in this folder. You can run any of the [example notebooks](https://github.com/destination-earth-digital-twins/polytope-examples/tree/main/climate-dt), which will require you to authenticate with your personal DESP username and password.


### Notes on how to select data from available simulations:

An overview of the available Climate DT simulations is provided [here](https://destine.ecmwf.int/climate-change-adaptation-digital-twin-climate-dt/#simulations).
The data selection is done in each notebook in the polytope request, for example in [climate-dt-earthkit-example-domain.ipynb](https://github.com/destination-earth-digital-twins/polytope-examples/blob/main/climate-dt/climate-dt-earthkit-example-domain.ipynb) with the following code block:

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

This polytope request can be modified to request data from any of the available simulations. The prototype [STAC catalogue](https://climate-catalogue.lumi.apps.dte.destination-earth.eu/?root=root) offers an interactive interface to create your custom data request.
