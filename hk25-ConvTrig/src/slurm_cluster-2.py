from pathlib import Path
import xarray as xr
import numpy as np

from importlib import reload
from dask_jobqueue import SLURMCluster
from dask.distributed import Client
import dask.config

from tempfile import NamedTemporaryFile, TemporaryDirectory # Creating temporary Files/Dirs
from getpass import getuser # Libaray to copy things


def init_dask_slurm_cluster(scale = 8, processes = 2, walltime="00:30:00", memory="128GB", dash_address=8787):
    dask.config.set(
        {
            "distributed.worker.data-directory": "/scratch/m/m300883/dask_temp",
            "distributed.worker.memory.target": 0.75,
            "distributed.worker.memory.spill": 0.85,
            "distributed.worker.memory.pause": 0.95,
            "distributed.worker.memory.terminate": 0.98,
        }
    )

    scluster = SLURMCluster(
        queue           = "compute",
        walltime        = walltime,
        memory          = memory,
        cores           = processes,
        processes       = processes,
        account         = "mh0033",
        name            = "m300883-dask-cluster",
        interface       = "ib0",
        asynchronous    = False,
        scheduler_options = {'dashboard_address': f':{dash_address}'},
        log_directory   = "/scratch/m/m300883/dask_logs/",
        local_directory = "/scratch/m/m300883/dask_temp/",
    )
    
    client = Client(scluster)
    scluster.scale(jobs=scale)
    print(scluster.job_script())
    nworkers = scale * processes
    client.wait_for_workers(nworkers)              # waits for all workers to be ready, can be submitted now

    return client, scluster