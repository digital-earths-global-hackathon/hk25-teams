#!/bin/python
#
# Get the ClimateDT data either as file or as dataset. 

from polytope.api import Client
import earthkit.data

def get_data_as_file(request:dict, output_path:str) -> list:
    """Get the data based on a request

    Parameters:
    -----------
    request: dictionary
            Request for the data from the FDB
    output_path: string
            path where the grib file should be stored
    Returns:
    --------
    files: list of strings
            List of the output paths of the files requested
    """
    # using existing 
    client = Client(
        address="polytope.lumi.apps.dte.destination-earth.eu",
    )

    # Optionally revoke previous requests
    client.revoke("all")

    # The data will be saved in the current working directory
    files = client.retrieve("destination-earth", request, output_path)
    return files

def get_data_polytope(request:dict) -> object:
    """Get the data based on a request

    Parameters:
    -----------
    request: dictionary
            Request for the data from the FDB
    Returns:
    --------
    data: object
            data from the request
    """
    data = earthkit.data.from_source("polytope", "destination-earth", request, stream=False, \
                               address='polytope.lumi.apps.dte.destination-earth.eu')
    return data

def test_get_data(as_file:bool=True, polytope:bool=True) -> None:
    """Test the get_data function using a dummy request.

    Parameters:
    ----------
    as_file: try get_data_as_file
    streaming: try get_data_streaming
    """
    request = {
        "activity": "cmip6",
        "class": "d1",
        "dataset": "climate-dt",
        "experiment": "hist",
        "generation": "1",
        "levtype": "sfc",
        "date": "20180911", #/to/20180915",
        "model": "icon",
        "expver": "0001",
        "param": "166",
        "realization": "1",
        "resolution": "high",
        "stream": "clte",
        "time": "0000",
        "type": "fc",
        "feature": {
        "type": "boundingbox",
        "points" : [[25, -62 ], [37.0, -77]],
        },
    }

    if as_file:
        files = get_data_as_file(request=request, \
                     output_path="test.grib")
    
        print(files)

    if polytope:
        data = get_data_polytope(request)
        print(data._json())

    return

#test_get_data(as_file=False)