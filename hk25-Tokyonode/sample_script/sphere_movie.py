#
# provided by Tobias Kölling
#
import intake
import cartopy.crs as ccrs
import cartopy.feature as cf
import matplotlib.pylab as plt
import easygems.healpix as egh
import cmocean
import tqdm
from pathlib import Path


def worldmap(var, time=0, **kwargs):
    plt.rcParams['figure.facecolor'] = 'black'
    projection = ccrs.NearsidePerspective(central_longitude=time + .01, central_latitude=23.5)
    fig, ax = plt.subplots(
        figsize=(20, 20), subplot_kw={"projection": projection}, constrained_layout=True, dpi=100
    )
    ax.set_global()

    egh.healpix_show(var, ax=ax, **kwargs)
    return fig
    #ax.add_feature(cf.COASTLINE, linewidth=0.8)
    #ax.add_feature(cf.BORDERS, linewidth=0.4)

settings = {
    "sfcwind": dict(cmap="cmo.ice", vmin=0, vmax=25),
}

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("runid", default="nicam_gl11")
    parser.add_argument("varname", default="sfcwind")
    parser.add_argument("size", type=int, default=40)
    parser.add_argument("i", type=int, default=0)
    args = parser.parse_args()

    
    cat = intake.open_catalog("https://digital-earths-global-hackathon.github.io/catalog/catalog.yaml")["JAPAN"]
    ds = cat[args.runid](time="PT1H", zoom=9, chunks={"cell": "auto"}).to_dask()
    if args.runid == "nicam_gl11":
        ds = ds.isel(time=slice(1, None, 2))
    if not "sfcwind" in ds:
        ds = ds.assign(sfcwind=lambda ds: (ds.uas**2 + ds.vas**2)**.5)

        
    
    var = ds[args.varname]
    kwargs = settings[args.varname]
    speed = -1
    
    data = None
    current_chunk = None
    chunksize = var.chunks[0][0]  # HACK: chunks technically don't have to be uniform in dask (but they are here)
    for i in tqdm.tqdm(range(args.i, args.i + args.size)):
        out = Path(f"/work/tobias/plots/out/{args.runid}_{var.name}_{"-".join(map(lambda s: ":".join(map(str, s)), kwargs.items()))}_{i:05d}.png")
        out.parent.mkdir(parents=True, exist_ok=True)
        chunk = i // chunksize
        if chunk != current_chunk:
            del data
            data = var.isel(time=slice(chunk * chunksize, (chunk + 1) * chunksize)).compute()
            current_chunk = chunk
        d = data.isel(time=i % chunksize)
        fig = worldmap(d, i * speed, **kwargs)
        fig.savefig(out)
        plt.close("all")

if __name__ == "__main__":
    main()
