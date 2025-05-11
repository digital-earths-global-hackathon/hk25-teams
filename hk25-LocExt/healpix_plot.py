#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
(c) 2025 under a MIT License (https://mit-license.org)

Authors:
- Lukas Brunner || lukas.brunner@uni-hamburg.de

"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import easygems.healpix as egh
import cartopy.feature as cfeature
from matplotlib import patches


def get_listed_colormap(levels, cmap='viridis', extend='neither', white=None, return_colors=False):
    """

    Parameters
    ----------
    levels : list
        List of levels giving the bounds of the levels, i.e., the number of colors is len(levels) - 1
    """
    
    if isinstance(levels, (int, np.int64)):
        nr = levels
    else:
        nr = len(levels) - 1

    if white is not None:
        nr -= 1

    if extend == 'both':
        nr += 2
    elif extend == 'max' or extend == 'min':
        nr += 1

    colors = mpl.colormaps[cmap](np.linspace(0, 1, nr))
    if white is not None:
        if white == 'first':
            colors = np.concatenate([[[1, 1, 1, 1]], colors])
        if white == 'last':
            colors = np.concatenate([colors, [[1, 1, 1, 1]]])

    if return_colors:
        return colors

    if extend == 'neither':
        cmap = mpl.colors.ListedColormap(colors)
    elif extend == 'both':
        cmap = mpl.colors.ListedColormap(colors[1:-1])
        cmap.set_under(colors[0])
        cmap.set_over(colors[-1])
    elif extend == 'min':
        cmap = mpl.colors.ListedColormap(colors[1:])
        cmap.set_under(colors[0])
    elif extend == 'max':
        cmap = mpl.colors.ListedColormap(colors[:-1])
        cmap.set_over(colors[-1])
    else:
        raise ValueError

    return cmap
    

def get_diverging_colormap(levels=12, cmap_neg='Blues', cmap_pos='Reds', middle_white=True, extend='both', return_colors=False):
    if isinstance(levels, int):
        nr = levels
    else:
        nr = len(levels) - 1

    white = None
    if nr % 2 == 0:
        nr //= 2
        if middle_white:
            nr -= 2
            white = [[1, 1, 1, 1], [1, 1, 1, 1]] 
    else:
        if middle_white:
            nr = (nr - 1) // 2
            white = [[1, 1, 1, 1]] 
        else:
            raise ValueError

    if extend == 'both':
        nr += 2
    elif extend != 'neither':
        raise ValueError(f'extend has to be one of "both", "neither" not {extend}')
            
    neg = get_listed_colormap(nr, cmap_neg, return_colors=True)[::-1]
    pos = get_listed_colormap(nr, cmap_pos, return_colors=True)

    if white is None:
        colors = np.concatenate([neg, pos]) 
    else:
        colors = np.concatenate([neg, white, pos]) 
    
    
    if return_colors:
        return colors
    if extend == 'neither':
        cmap = mpl.colors.ListedColormap(colors)
    else:
        cmap = mpl.colors.ListedColormap(colors[1:-1])
        cmap.set_under(colors[0])
        cmap.set_over(colors[-1])
    return cmap


def hp_plot(
    data, 
    cmap='viridis', 
    ax='Robinson', 
    add_coastlines=False, 
    add_rivers_lakes=False,
    topography=None,
    fill_lakes=True,
    add_colorbar=True, 
    levels=None,
    extend='neither',
    add_gridlines=False,
    dpi=72, 
    proj_kwargs={},
    cbar_kwargs={}, 
    rivers_lakes_kwargs={},
    topography_kwargs={},
    coastline_kwargs={},
    grid_kwargs={},
    **kwargs
):
    """

    Parameters
    ----------
    data : np.ndarray, shape (N,)
        Needs to be on a healpix grid, i.e., N needs to be divisibel by 12 * (2**zoom)**2
    cmap : string, optional, by default 'viridis'
    ax : string or cartopy.ccrs, optional, by default 'Mollweide'
        Possible string values:
        - 'Mollweide'
        - 'PlateCarree'
        - 'Orthographic'
    add_coaslines : bool, optional, by default False
    add_rivers_lakes : bool, optional, by default False
    fill_lakes  bool, optional, by default True
        Only relevent if `add_rivers_lakes=True`. Whether to plot shading within lakes
        this makes them better visible but might hinder seeing the variable shading
    topography : np.ndarray, shape (N,), optional, by default None
        Plot elevation contourlines based on the data passed.
    add_colorbar : bool, optional, by default True
    levels : np.ndarray, optional, by default None
        Can be used to set manual (non equidistant) color levels
    extend : string, optional, one of {'neither', 'min', 'max', 'both'}, by default 'neither'
    add_gridlines : bool, optional, by default False
    dpi : int, optional, by default 150
        Plot resolution. NOTE: sometimes artifacts apear around the zero meridian, changing
        the resoltion might solve this. 
    proj_kwargs : dict, optional
        Keyword arguments passed on to ccrs.<Projection>. Only relevent if ax is a string
        specifying a projection. The allowed values depend on the projection:
        - 'Mollweide': 'central_longitude'
        - 'PlateCarree': 'central_longitude'
        - 'Orthographic': 'central_longitude', 'central_latitude'
    cbar_kwargs : dict, optional
        Keyword arguments passed on to `plt.colorbar`
    coastline_kwargs : dict, optional
        Keyword arguments passed on to `ax.coastlines`
    topography_kwargs : dict, optional
        Keyword arguments passed on to `ax.contour`
    grid_kwargs : dict, optional
        Keyword arguments apssed on to `ax.gridlines`
    **kwargs : optional
        Keyword arguments passed on to `ax.imshow`

    Returns
    -------
    fig, ax, map_: tuple

    Additional information
    ----------------------
    List of cartopy projections: https://scitools.org.uk/cartopy/docs/v0.15/crs/projections.html
    """
    if isinstance(ax, str):
        proj = getattr(ccrs, ax)(**proj_kwargs)
        # increase transform resolution via
        # https://stackoverflow.com/questions/59020032/how-to-plot-a-filled-polygon-on-a-map-in-cartopy
        proj._threshold /= 1000.
        fig, ax = plt.subplots(
            figsize=(20, 10), 
            dpi=dpi, 
            subplot_kw={'projection': proj},
        )
        ax.set_global()
    else:
        fig = plt.gcf()
        
    if add_coastlines:
        defaults = {'color': 'k', 'lw': .5}
        defaults.update(coastline_kwargs)
        ax.coastlines(**defaults)

    if add_rivers_lakes:
        defaults = dict(
            alpha=1,
            facecolor='none',                
            edgecolor='cornflowerblue', 
            # color='cornflowerblue', 
            linewidth=.5,
        )
        defaults.update(rivers_lakes_kwargs)
        if fill_lakes:
            ax.add_feature(
                cfeature.LAKES.with_scale('50m'), 
                alpha=.2,
                facecolor='cornflowerblue',                
                edgecolor='none', 
            )
        ax.add_feature(
            cfeature.LAKES.with_scale('50m'), 
            # alpha=1,
            # facecolor='none',                
            # edgecolor='cornflowerblue', 
            **defaults
        )
        ax.add_feature(
            cfeature.RIVERS,
            # color='cornflowerblue',
            # alpha=1,
            **defaults
        )

    if topography is not None:
        defaults = dict(
            colors='gray',
            levels=range(1000, 10_000, 1000),
            linewidths=.5,
        )
        defaults.update(topography_kwargs)
        
        _, _, nx, ny = np.array(ax.bbox.bounds, dtype=int)
        xlims = ax.get_xlim()
        ylims = ax.get_ylim() 
        im = egh.healpix_resample(
            topography, 
            xlims, ylims, 
            nx, ny, 
            ax.projection, 
            method='linear', 
            nest=True)
        
        map_ = ax.contour(
            im, 
            extent=xlims + ylims, 
            origin="lower",
            **defaults
        )

    if levels is not None:
        if isinstance(cmap, str):
            cmap = get_listed_colormap(levels, cmap, extend)
            kwargs.update({
                'vmin': levels[0], 
                'vmax': levels[-1],
        })
        else: 
            if isinstance(cmap, mpl.colors.ListedColormap):  # convert back to color list 
                if extend == 'max':
                    cmap = np.concatenate([cmap.colors, [cmap.get_over()]])
                elif extend == 'min':
                    cmap = np.concatenate([[cmap.get_under()], cmap.colors])
                elif extend == 'both':
                    cmap = np.concatenate([[cmap.get_under()], cmap.colors, [cmap.get_over()]])
                else:
                    cmap = cmap.colors
                    
            cmap, norm = mpl.colors.from_levels_and_colors(levels, cmap, extend=extend)
            kwargs.update({
            'norm': norm, 
            })
        if 'ticks' not in cbar_kwargs:
            cbar_kwargs.update({
                'ticks': levels,
            })
    
    _, _, nx, ny = np.array(ax.bbox.bounds, dtype=int)
    xlims = ax.get_xlim()
    ylims = ax.get_ylim()

    im = egh.healpix_resample(
        data, 
        xlims, ylims, 
        nx, ny, 
        ax.projection, 
        method='nearest', 
        nest=True,
    )
   
    map_ = ax.imshow(
        im, 
        extent=xlims + ylims, 
        origin="lower", 
        cmap=cmap, 
        interpolation='none',
        **kwargs,
    )

    if add_gridlines:
        ax.gridlines(**grid_kwargs)

    if add_colorbar:
        plt.colorbar(map_, ax=ax,shrink=.8, fraction=.03, extend=extend, **cbar_kwargs)
        
    return fig, ax, map_


def plot_polygon(ax, corners, closed=True, **kwargs):
    """
    Plot a user-defined polygon on the map.

    Parameters
    ----------
    ax : plt.axes
    corners : list of tuple (lon, lat)
    closed : bool, optional, by default True
        Connet the last and first corner as well
    kwargs : dict, optional
        Keyword arguments passed on to `mpatches.Polygon`
    """
    tmp = dict(
        edgecolor = 'k',
        facecolor = 'none',
        lw = 1,
        zorder = 10
    )
    tmp.update(kwargs)
    poly = patches.Polygon(
        corners, 
        closed=closed, 
        transform=ccrs.PlateCarree(),
        **tmp,
    )
    ax.add_patch(poly)