import xarray as xr
import numpy as np
from tqdm import tqdm
from dask.distributed import Client, LocalCluster

from tcpyPI import pi

import intake
from easygems import healpix as egh

import warnings
warnings.filterwarnings("ignore", category=FutureWarning) # don't warn us about future package conflicts

import healpy as hp

#WARNING! some models have 3D only when called with different keys!
sim_name = "um_glm_n1280_GAL9" 
out_dir = "/work/bb1153/b383007/hk25-hamburg/out_data/" #directory to save PI fields
zoom_level = 5
time_res=6
print(f"Simulation name: {sim_name}\n Zoom level: {zoom_level}\n Time resolution: {time_res}H")

current_location = "online"
cat = intake.open_catalog("https://digital-earths-global-hackathon.github.io/catalog/catalog.yaml")[current_location]
#list(cat.keys())

def main():
    n_workers=16
    n_cpu=16
    memory_limit="50GB"
    processes=True
    
    cluster = LocalCluster(n_workers=n_workers, 
                           threads_per_worker=n_cpu // n_workers,
                           memory_limit=memory_limit,
                           processes=processes,
                           dashboard_address=46861)
    client = Client(cluster)

    '''create datasets'''
    if "um_" in sim_name:
        ds2d = cat[sim_name](zoom=zoom_level).to_dask()
    
        vnames = {"sst":"ts", "psl":"psl", "t":"ta", "q":"hus"}
    
        ds3d = cat[sim_name](zoom=zoom_level, time="PT3H").to_dask()
        
        dim="pressure"
        
    if "icon_" in sim_name:
        ds2d = cat[sim_name](zoom=zoom_level).to_dask()
        
        vnames = {"sst":"ts", "psl":"psl", "t":"tas", "q":"huss"}
    
        ds3d = cat[sim_name](zoom=zoom_level).to_dask()
    
        dim="pressure"
        ds3d = ds3d.assign_coords(pressure=ds3d[dim]/100)
    if "ifs_" in sim_name:
        ds2d = cat[sim_name](zoom=zoom_level, dim="2D").to_dask()
        
        vnames = {"sst":"sst", "psl":"msl", "t":"t", "q":"q"}
    
        ds3d = cat[sim_name](zoom=zoom_level, dim="3D").to_dask()
    
        dim="level"
    
    
    ds3d = ds3d.sortby(dim, ascending=False)
    time= ds2d.time.where(ds2d.time.dt.hour % time_res == 0, drop=True)
    
    ts = ds2d[vnames["sst"]]
    psl = ds2d[vnames["psl"]]
    t = ds3d[vnames["t"]]
    q = ds3d[vnames["q"]]

    #adjust spatial resolution for regular grid conversion
#    if ("icon_" in sim_name) or ("um_" in sim_name):
    ddeg = 0.2
    nside = nside = ds3d.crs.healpix_nside
    lon = np.arange(0, 360, ddeg)
    #lat = np.arange(90, -90+ddeg, -ddeg)
    lat = np.arange(60, -60+ddeg, -ddeg)
#    else:
#        lat = np.sort(ds3d.lat)
#        lat = lat[(lat<=60) & (lat>=60)]
#        lon = np.sort(ds3d.lon)
    
    pix = xr.DataArray(
        hp.ang2pix(nside, *np.meshgrid(lon, lat), nest=True, lonlat=True),
        coords=(("lat", lat), ("lon", lon)),
    )

    '''compute PI fields per timestep and save each snapshot to disk'''
    p=ds3d[dim]
    CKCD=0.9
    
    def get_pi(ts, psl, p, t, q):
        result = xr.apply_ufunc(
            pi,
            ts, psl, p, t, q,
            kwargs=dict(CKCD=CKCD, ascent_flag=0, diss_flag=1, ptop=50, 
                        miss_handle=1),
            input_core_dims=[
                [], [], [dim, ], [dim, ], [dim, ],
            ],
            output_core_dims=[
                [], [], [], [], []
            ],
            vectorize=True
        )
        vmax, pmin, ifl, t0, otl = result
        return vmax, pmin, t0, otl
    
    def process_chunk(ts, psl, p, t, q):
        return get_pi(ts, psl, p, t, q)

    if "ifs_" in sim_name:
        ts = ts.drop_vars(["lat", "lon"])
        psl = psl.drop_vars(["lat", "lon"])
        t = t.drop_vars(["lat", "lon"])
        q = q.drop_vars(["lat", "lon"])
    
    chunk_size = 25  # Adjust based on available memory or number of workers
    lat_chunks = [slice(i, i + chunk_size) for i in range(0, len(lat), chunk_size)]
    for ti in tqdm(range(len(time))):
        ts_ti = ts.sel(time=time[ti]).load()[pix] - 273.15
        psl_ti = psl.sel(time=time[ti]).load()[pix] / 100
        t_ti = t.sel(time=time[ti]).load()[:, pix] - 273.15
        q_ti = q.sel(time=time[ti]).load()[:, pix] * 1000
    #    if "icon_" in sim_name:
    #        psl_i = psl_i/100
        
        futures = []
        for chunk in lat_chunks:
            ts_cti = ts_ti.isel(lat=chunk)
            psl_cti = psl_ti.isel(lat=chunk)
            t_cti = t_ti.isel(lat=chunk)
            q_cti = q_ti.isel(lat=chunk)
    
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', 'Sending large graph of size', UserWarning)
                futures.append(client.submit(process_chunk, ts_cti, psl_cti, p, t_cti, q_cti))
            
        results = np.array(client.gather(futures), dtype=object)
    
        # Initialize with empty (NaN) arrays for each variable
        shape = (1, len(lat), len(lon))
        empty_arr = np.full(shape, np.nan, dtype=np.float32)
        dims = ["time", "lat", "lon"]
        
        out_ds = xr.Dataset({
                'vmax': (dims, empty_arr.copy()), 
                'pmin': (dims, empty_arr.copy()),
           #     'ifl': (dims, empty_arr),
                't0': (dims, empty_arr.copy()),
                'otl': (dims, empty_arr.copy()),
           #     'sst': (dims, empty_arr),
           #     't': (dims, empty_arr),
           #     'q': (dims, empty_arr),
           #     'msl': (dims, empty_arr),
                },
                coords={
                "time": [time[ti].values],
                "lat": lat,
                "lon": lon}
        )
        
        # add names and units to the structure
        out_ds.vmax.attrs['standard_name'],out_ds.vmax.attrs['units']='Maximum Potential Intensity','m/s'
        out_ds.pmin.attrs['standard_name'],out_ds.pmin.attrs['units']='Minimum Central Pressure','hPa'
        #out_ds.ifl.attrs['standard_name']='pyPI Flag'
        out_ds.t0.attrs['standard_name'],out_ds.t0.attrs['units']='Outflow Temperature','K'
        out_ds.otl.attrs['standard_name'],out_ds.otl.attrs['units']='Outflow Temperature Level','hPa'
    
        out_ds["vmax"].loc[dict(time=time[ti])] = xr.concat(results[:,0], dim="lat").values
        out_ds["pmin"][0, :, :] = xr.concat(results[:,1], dim="lat").values
        out_ds["t0"]  [0, :, :] = xr.concat(results[:,2], dim="lat").values
        out_ds["otl"] [0, :, :] = xr.concat(results[:,3], dim="lat").values
        #results = [xr.concat(results[i], dim="lat") for i in range(len(out_ds))]
        
        out_ds.to_netcdf(out_dir+f'{sim_name}_pi_{time[ti].values}'+'.nc')
    client.close()

if __name__ == "__main__":
    main()


    