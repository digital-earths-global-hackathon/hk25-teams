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

As an example, you can select the DestinE storylines notebook, e.g. `climate-dt-earthkit-fe-story-nudging.ipynb`.

Example output should be:


<img width="475" alt="image" src="https://github.com/user-attachments/assets/53576b86-6907-43bd-9c6f-0b26027e2387" />

Success!

**Updating an existing environment in Insula**  
 Most notebooks listed in polytope-lab -> climate-dt will require you to update the default Python environment on Insula by executing early in one of the first cells:

```bash
pip install --upgrade --user earthkit
pip install --user kaleido
```

**Creating a new environment in Insula**   
If you are using Insula on the DESP to access data, the following instructions listed in Option 2 will generate a working python kernel that will be visible in jupyter.

### Option 2: local data analysis
#### 2.1 Using pip

To explore the Climate DT data locally (e.g. on a HPC/laptop), a python environment can be created using the requirements.txt provided in this folder. You can run any of the [example notebooks](https://github.com/destination-earth-digital-twins/polytope-examples/tree/main/climate-dt), which will require you to authenticate with your personal DESP username and password.

Generate a python virtual environment:

`python -m venv /home/jovyan/my_env`

Activate your environment:

`source /home/jovyan/my_env/bin/activate`

Install the required libraries from the following file https://github.com/destination-earth-digital-twins/polytope-examples/blob/main/requirements.txt:

`pip install requirements.txt`

Install ipykernel to make the kernel visible in your notebooks:

`pip install ipykernel`

`python -m venv /home/jovyan/my_env`

`ipython kernel install --user --name=my_env`

You should now be able to select this kernel and access DT data.

If you have previously created a python venv you may need to update the versions of some packages. You can do this manually or by reinstalling from the requirements.txt

#### 2.2 Using conda
When using conda the approach is very similar to what is described in the [how to tech](https://github.com/digital-earths-global-hackathon/hamburg-node/blob/main/content/howtotech.md). Here we use a different environment.yml file which you can find in the [polytopes-examples repo](https://github.com/destination-earth-digital-twins/polytope-examples/blob/main/environment.yml). The rest is the same so please check those guidelines for more detailed advise.

These are the main steps: 
1. Create a directory, e.g. `destine_env` in the `work` of the project.
2. Change into the `destine_env` directory and copy/move the [environment.yml](https://github.com/destination-earth-digital-twins/polytope-examples/blob/main/environment.yml) there `wget https://raw.githubusercontent.com/destination-earth-digital-twins/polytope-examples/refs/heads/main/environment.yml`.
3. Create the environment (this may take a while) by running `mamba env create -f environment.yml -p <path-to-your-environment-folder>`
4. Activate the environment `mamba activate <path-to-your-environment-folder`
5. Create a jupyter kernel if you want to use jupyter notebooks by running `python3 -m ipykernel install --name global-hackathon-destine --user`

   
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
