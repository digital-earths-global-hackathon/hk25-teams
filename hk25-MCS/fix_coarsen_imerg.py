"""
This script is designed to coarsen the IMERG HEALPix dataset to lower zoom level and write it to Zarr format.
It also subsets the dataset to include only timestamps that are at a specific minute (e.g., 0 or 30).

Author:
    Zhe Feng <zhe.feng@pnnl.gov>

Created:
    2025-05-15
"""
import numpy as np
import xarray as xr
import logging
import os
import gc
from dask.distributed import Client, LocalCluster

#-------------------------------------------------------------------
def zoom_level_from_nside(nside):
    """
    Calculate the zoom level from the NSIDE value.

    Args:
        nside (int): NSIDE value, must be a power of 2.
    
    Returns:
        int: Zoom level corresponding to the NSIDE value.
    """
    zoom = int(np.log2(nside))
    if 2**zoom != nside:
        raise ValueError("NSIDE must be a power of 2.")
    return zoom

def setup_logging():
    """
    Set the logging message level

    Args:
        None.

    Returns:
        None.
    """
    logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)

#-------------------------------------------------------------------
def setup_dask_client(parallel, n_workers, threads_per_worker, logger=None):
    """
    Set up a Dask client for parallel processing
    
    Args:
        parallel: bool
            Whether to use parallel processing
        n_workers: int
            Number of workers for the Dask cluster
        threads_per_worker: int
            Number of threads per worker
        logger: logging.Logger, optional
            Logger for status messages
            
    Returns:
        dask.distributed.Client or None: Dask client if parallel is True, None otherwise
    """
    if logger is None:
        logger = logging.getLogger(__name__)
        
    if not parallel:
        logger.info("Running in sequential mode (parallel=False)")
        return None
    
    logger.info(f"Setting up Dask cluster with {n_workers} workers, {threads_per_worker} threads per worker")
    cluster = LocalCluster(
        n_workers=n_workers,
        threads_per_worker=threads_per_worker,
        memory_limit='auto',
    )
    client = Client(cluster)
    logger.info(f"Dask dashboard: {client.dashboard_link}")
    
    return client

#-------------------------------------------------------------------
def write_zarr(ds, out_zarr, client=None, logger=None):
    """
    Write dataset to Zarr with optimized chunking for HEALPix grid.
    
    Args:
        ds: xarray.Dataset
            Dataset to write
        out_zarr: str
            Output Zarr store path
        client: dask.distributed.Client, optional
            Dask client for distributed computation
        logger: logging.Logger, optional
            Logger for status messages
            
    Returns:
        None
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    # Optimize cell chunking for HEALPix grid
    zoom_level = zoom_level_from_nside(ds.crs.attrs['healpix_nside'])
    chunksize_time = 24
    chunksize_cell = 12 * 4**zoom_level
    
    # Make time chunks more even if needed
    if isinstance(chunksize_time, (int, float)) and chunksize_time != 'auto':
        total_times = ds.sizes['time']
        chunks = total_times // chunksize_time
        if chunks * chunksize_time < total_times:
            # We have a remainder - try to make chunks more even
            if total_times % chunks == 0:
                chunksize_time = total_times // chunks
            elif total_times % (chunks + 1) == 0:
                chunksize_time = total_times // (chunks + 1)
    
    # Set proper chunking for HEALPix output
    chunked_hp = ds.chunk({
        "time": chunksize_time, 
        "cell": chunksize_cell, 
    })
    # Report dataset size and chunking info
    logger.info(f"Output dataset dimensions: {dict(chunked_hp.sizes)}")
    logger.info(f"Output chunking scheme: time={chunksize_time}, cell={chunksize_cell}")

    # ---------- WRITE HEALPIX ZARR OUTPUT ----------
    logger.info(f"Starting Zarr write to: {out_zarr}")
    
    # Create a delayed task for Zarr writing
    write_task = chunked_hp.to_zarr(
        out_zarr,
        mode="w",
        consolidated=True,  # Enable for better performance when reading
        compute=False      # Create a delayed task
    )
    
    # Compute the task, with progress reporting
    if client:
        from dask.distributed import progress
        import psutil

        # Temporarily suppress distributed.shuffle logs during progress display
        shuffle_logger = logging.getLogger('distributed.shuffle')
        original_level = shuffle_logger.level
        shuffle_logger.setLevel(logging.ERROR)  # Only show errors, not warnings

        # Get cluster state information before processing
        memory_usage = client.run(lambda: psutil.Process().memory_info().rss / 1e9)
        logger.info(f"Current memory usage across workers (GB): {memory_usage}")
               
        try:
            # Compute with progress tracking
            future = client.compute(write_task)
            logger.info("Writing Zarr (this may take a while)...")
            progress(future)  # Shows a progress bar in notebooks or detailed progress in terminals

            result = future.result()
            logger.info("Zarr write completed successfully")
        except Exception as e:
            logger.error(f"Zarr write failed: {str(e)}")
            raise
        finally:
            # Restore original log level
            shuffle_logger.setLevel(original_level)
    else:
        # Compute locally if no client
        write_task.compute()

    logger.info(f"Zarr file complete: {out_zarr}")


if __name__ == "__main__":

    # Set up logging
    setup_logging()
    logger = logging.getLogger(__name__)

    # Specify parameters
    # version = 'V06B'
    version = 'V07B'
    zoom_level = 9
    write_filter_time_zarr = False

    parallel = True
    n_workers = 32
    threads_per_worker = 4

    # Minute filter - set to 0 for on-the-hour, 30 for half-hour
    minute_filter = 0

    # Input Zarr file
    in_zarr = "/global/cfs/cdirs/m4581/gsharing/hackathon/OBS/IMERG_V07B_hp9.zarr"
    out_dir = "/pscratch/sd/w/wcmca1/GPM/IR_IMERG_Combined_V07B/"
    out_basename = f"IMERG_{version}_"
    out_zarr_filt = os.path.join(out_dir, f"{out_basename}hp{zoom_level}.zarr")

    # Setup Dask client
    client = setup_dask_client(parallel, n_workers, threads_per_worker, logger)

    # Write time-filtered dataset to Zarr
    if write_filter_time_zarr:
        # Open the first file to get the shape
        ds = xr.open_zarr(in_zarr, consolidated=True)

        # Extract minutes from each timestamp
        minutes = ds.time.dt.minute.values
        
        # Define tolerance (in minutes)
        tolerance = 1  # 1 minute tolerance
        
        # Create mask for timestamps near the specified minute
        minute_mask = np.abs(minutes - minute_filter) < tolerance
        
        # Apply mask and update dataset
        logger.info(f"Filtering timestamps to include only those at minute {minute_filter} (±{tolerance})")
        ds = ds.sel(time=ds.time[minute_mask])
        logger.info(f"After filtering: {len(ds.time)} timestamps remaining")

        # Write to Zarr
        write_zarr(ds, out_zarr_filt, client=client, logger=logger)
    else:
        # Read the time-filtered Zarr
        ds = xr.open_zarr(out_zarr_filt, consolidated=True)
        logger.info(f"Opened initial Zarr file: {out_zarr_filt}")

    # Loop through zoom levels and write Zarr files
    dn = ds
    for x in range(zoom_level-1,-1,-1):
        s = str(x)
        # Output Zarr name
        out_fn = f"{out_dir}{out_basename}hp{s}.zarr"
        print(f"Coarsening to zoom level {s}...")

        # Coarsen the dataset
        dx = dn.coarsen(cell=4).mean()

        # Update HEALPix level metadata
        dx['crs'].attrs['healpix_nside'] = 2**int(x)

        # Write to Zarr
        write_zarr(dx, out_fn, client=client, logger=logger)
        print(f"✓ Wrote to: {out_fn}")

        # Update dataset with the new coarsened data
        dn = dx
        del dx
        gc.collect()

    # Close the dataset
    ds.close()
    
    # Clean up Dask client if it exists
    if client is not None:
        logger.info("Shutting down Dask client")
        client.close()
        logger.info("Dask client shut down successfully")