import xarray as xr
import numpy as np
import healpy as hp
import easygems.healpix as egh
from typing import Tuple, Optional

# ------------------------------------------------------------------------------
# Functions to determine triggering area of MCSs
# ------------------------------------------------------------------------------
def add_circular_trigger_areas(
        mcs_trigger_locs: xr.DataArray,
        RADII: np.ndarray,
        hp_grid: str,
        ) -> xr.DataArray:
    """
    Add circular trigger areas to the MCS trigger locations.

    This function calculates the circular trigger areas around the given 
    trigger locations based on the specified radii and Healpix grid 
    configuration. The trigger areas are represented as indices of the 
    Healpix cells within the specified radius around each trigger location.

    Parameters
    ----------
    mcs_trigger_locs : xr.DataArray
        DataArray containing the MCS trigger locations. It must include 
        the 'trigger_idx' coordinate, which specifies the Healpix 
        cell indices of the trigger locations.
    RADII : np.ndarray
        Array of radii (in the same units as the Healpix grid) for which 
        the trigger areas are to be calculated.
    hp_grid : str
        String specifying the Healpix grid configuration. This is used to 
        determine the nside and nesting scheme of the grid.

    Returns
    -------
    xr.DataArray
        Updated DataArray with an additional coordinate 'trigger_area_idxs', 
        which contains the indices of the Healpix cells within the trigger 
        areas for each radius in RADII.

    Notes
    -----
    - The Healpix grid attributes (nside and nest) are determined using 
        helper functions from the `egh` module.
    - The trigger areas are initialized and stored in a new coordinate 
        'trigger_area_idxs', which is a 3D array with dimensions corresponding 
        to the number of tracks, the maximum number of cells in any trigger 
        area, and the number of radii in RADII.
    """
    # Determine attributes of the healpix grid
    nside = egh.get_nside(hp_grid)
    nest = True if egh.get_nest(hp_grid) else False

    for i, radius in enumerate(RADII):
        # Get all pixels within radius around triggering location, i.e. get the
        # 'triggering area'
        trigger_area_idxs = [
            _get_trigger_area_idxs(nside, nest, cell_idx, radius)
            for cell_idx in mcs_trigger_locs['trigger_idx'].values
            ]

        if i == 0:
            # Initialize data_array to save cell indices of the trigger area
            mcs_trigger_locs['trigger_area_idxs'] = \
                _init_trigger_area_idxs_array(
                    mcs_trigger_locs['tracks'].values, trigger_area_idxs,
                    RADII
                    )

        for j, idxs in enumerate(trigger_area_idxs):
            mcs_trigger_locs['trigger_area_idxs'][j, :idxs.shape[0], i] = idxs

    return mcs_trigger_locs


def _get_trigger_area_idxs(
        nside: int,
        nest: bool,
        cell_idx: int,
        radius_deg: float,
        ) -> np.ndarray:
    """
    Get the indices of the trigger area for a given cell index and radius.
    The trigger area is defined as the area within a certain radius from the
    healpix cell that marks the MCS center at its triggering time.
    Parameters
    ----------
    nside : int
        The nside parameter for the HEALPix map.
    nest : bool
        If True, use nested indexing. If False, use ring indexing.
    cell_idx : int
        The index of the cell for which to get the trigger area indices.
    radius_deg : float
        The radius in degrees for the trigger area.
    Returns
    -------
    np.ndarray
        The indices of the trigger area.
    """
    return hp.query_disc(
        nside, hp.pix2vec(nside, cell_idx, nest=nest), np.radians(radius_deg),
        inclusive=False, nest=nest,
    )

def _init_trigger_area_idxs_array(
        tracks: np.ndarray,
        trigger_area_idxs: list,
        radii: np.ndarray,
        ) -> xr.DataArray:
    """
    Initialize a DataArray to store trigger area indices.
    This function creates an xarray DataArray with dimensions corresponding to 
    tracks, cells, and radius degrees. The array is initialized with NaN values 
    and is intended to store indices of trigger areas for each track and radius.
    Parameters
    ----------
    tracks : np.ndarray
        An array representing the tracks for which the trigger area indices 
        are being initialized.
    trigger_area_idxs_list : list
        A list of arrays, where each array contains the indices of trigger areas 
        for a specific track.
    radii_degree : np.ndarray
        An array of radius values (in degrees) for which the trigger area
        indices are calculated.
    Returns
    -------
    xr.DataArray
        A DataArray with dimensions ['tracks', 'cell', 'radius'],
        initialized with NaN values, and coordinates corresponding to the input
        parameters.
    """    
    cells = np.arange(
        0, np.array([x.shape[0] for x in trigger_area_idxs]).max()
        )

    trigger_area_idxs_array = xr.DataArray(
        data=np.full(
            shape=(tracks.size, cells.size, radii.size),
            fill_value=np.nan,
        ),
        dims=['tracks', 'cell', 'radius'],
        coords={'tracks': tracks, 'cell': cells, 'radius': radii},
        )
    trigger_area_idxs_array['radius'].attrs['units'] = 'degree'
    return trigger_area_idxs_array


# ------------------------------------------------------------------------------
# Functions to subsample MCSs
# ------------------------------------------------------------------------------
def remove_land_triggers(
        mcs_trigger_locs: xr.DataArray,
        ocean_mask: xr.DataArray,
        ) -> xr.DataArray:
    """
    Removes land-based triggers from the given MCS (Mesoscale Convective
    Systems) trigger locations based on an ocean mask and returns only the
    all ocean-based triggers.

    Parameters:
    -----------
    mcs_trigger_locs : xr.DataArray
        An xarray DataArray containing the trigger locations for MCS. It is
        expected to have a dimension 'tracks' and a variable
        'trigger_idx' representing the Healpix cell index of the
        triggers.
    ocean_mask : xr.DataArray
        An xarray DataArray representing the ocean mask. It is used to determine
        whether the trigger areas are entirely over the ocean.

    Returns:
    --------
    xr.DataArray
        A filtered xarray DataArray containing only the MCS trigger locations
        that are entirely over the ocean. The returned DataArray has the same
        structure as the input but excludes land-based triggers.
    """
    mcs_trigger_locs['is_trigger_area_all_ocean'] = (
        'tracks', _is_max_trigger_area_all_ocean(
            mcs_trigger_locs, ocean_mask
            )
        )
    mcs_trigger_locs_ocean = mcs_trigger_locs.where(
        mcs_trigger_locs['is_trigger_area_all_ocean'], drop=True
        )
    mcs_trigger_locs_ocean = mcs_trigger_locs_ocean.drop(
        'is_trigger_area_all_ocean'
        )
    mcs_trigger_locs_ocean['trigger_idx'] = \
        mcs_trigger_locs_ocean['trigger_idx'].astype(int)
    
    return mcs_trigger_locs_ocean


def _is_max_trigger_area_all_ocean(
        mcs_trigger_locs: xr.DataArray,
        ocean_mask: xr.DataArray
        ) -> xr.DataArray:
    is_max_trigger_area_all_ocean = []
    for track in mcs_trigger_locs['tracks']:
        # Get the trigger area cell indices for current track without filling
        # NaNs
        trigger_area_idxs = mcs_trigger_locs['trigger_area_idxs'].sel(
            radius=mcs_trigger_locs['radius'].max().values,
            tracks=track,
            )
        trigger_area_idxs = trigger_area_idxs[~np.isnan(trigger_area_idxs)]
        
        # Check if all cells in the trigger area are ocean
        is_max_trigger_area_all_ocean.append(
            all(~np.isnan(ocean_mask.sel(cell=trigger_area_idxs)))
            )
    return is_max_trigger_area_all_ocean


# ------------------------------------------------------------------------------
# Functions to subsample variables in MCS trigger area
# ------------------------------------------------------------------------------
def get_var_in_trigger_area(
        *vars,
        times_before_trigger: Optional[np.timedelta64] = None,
        analysis_time: Optional[Tuple[np.datetime64]] = None,
        ) -> xr.DataArray:
    """
    Retrieve variable values in the triggering area of MCSs.

    This function retrieves the values of a given variable within the 
    triggering area of Mesoscale Convective Systems (MCSs) for each track 
    and radius. If a time range before the triggering is specified, it 
    retrieves the variable values for multiple time steps.

    Parameters
    ----------
    mcs_trigger_locs : xr.DataArray
        DataArray containing the MCS trigger locations. It must include 
        the 'trigger_area_idxs' coordinate, which specifies the Healpix 
        cell indices of the trigger areas.
    data_field : xr.DataArray
        DataArray containing the simulation data. It must include a 
        'cell' dimension corresponding to the Healpix grid.
    times_before_trigger : np.timedelta64, optional
        Time range before the triggering to retrieve variable values. If 
        None, only the values at the triggering time are retrieved.

    Returns
    -------
    xr.DataArray
        DataArray containing the variable values in the triggering area 
        for each track, radius, and optionally time. The dimensions are 
        ['tracks', 'cell', 'radius'] if no time range is specified, or 
        ['tracks', 'cell', 'radius', 'time'] if a time range is provided.
    """
    if times_before_trigger is None:
        return _get_var_in_trigger_area(*vars)
    else:
        return _get_var_in_trigger_area_multiple(
            *vars, times_before_trigger, analysis_time
            )


def _get_var_in_trigger_area(
        mcs_trigger_locs: xr.DataArray,
        data_field: xr.DataArray,
        ) -> xr.DataArray:
    """
    Extracts variable values within the trigger area for each MCS track and
    radius. This function processes the provided MCS trigger locations and a
    data field to extract the variable values within the defined trigger area
    for each track and radius. The trigger area is determined based on the
    indices corresponding to the MCS trigger locations and the specified radius.

    Parameters
    ----------
    mcs_trigger_locs : xr.DataArray
        A DataArray containing the MCS trigger locations with dimensions
        ['tracks', 'radius', ...]. It includes metadata such as 'start_basetime'
        for each track.
    data_field : xr.DataArray
        A DataArray containing the data field to be analyzed. It must have a
        'time' dimension and a 'cell' dimension that corresponds to spatial
        indices.
    
    Returns
    -------
    xr.DataArray
        A DataArray containing the variable values within the trigger area for
        each track and radius. The dimensions are ['tracks', 'cells', 'radius'],
        where 'cells' corresponds to the number of spatial indices in the
        trigger area for each radius.
    """
    var_in_trigger_area = _init_var_in_trigger_area(mcs_trigger_locs)
    
    for j, track in enumerate(mcs_trigger_locs['tracks']):
        # Subsample the data field for analysis
        mcs_start_basetime = mcs_trigger_locs.sel(tracks=track)\
            ['start_basetime'].values
        var_before_trigger = data_field.sel(
            time=mcs_start_basetime, method='pad'
            )

        for i, radius in enumerate(mcs_trigger_locs['radius']):
            trigger_area_idxs = _select_trigger_area_idxs(
                mcs_trigger_locs, track, radius
                )
            var_in_trigger_area[j, :trigger_area_idxs.shape[0], i] = \
                var_before_trigger.sel(cell=trigger_area_idxs).data
    
    return var_in_trigger_area


def _get_var_in_trigger_area_multiple(
        mcs_trigger_locs: xr.DataArray,
        data_field: xr.DataArray,
        times_before_trigger: np.timedelta64,
        analysis_time: np.datetime64,
        ) -> xr.DataArray:
    """
    Extracts and aggregates data from a specified field within the trigger area 
    of multiple MCS (Mesoscale Convective Systems) tracks over a given time 
    period before the trigger event.

    Parameters
    ----------
    mcs_trigger_locs : xr.DataArray
        An xarray DataArray containing the MCS trigger locations, including 
        track and radius information.
    data_field : xr.DataArray
        An xarray DataArray representing the data field to be analyzed (e.g., 
        temperature, humidity) with dimensions including 'time' and 'cell'.
    times_before_trigger : np.timedelta64
        The time duration before the MCS trigger event to consider for analysis.
    analysis_time : np.datetime64
        The start time of the analysis period. Data before this time will be 
        excluded.

    Returns
    -------
    xr.DataArray
        A DataArray containing the extracted data field values within the 
        trigger area for each track and radius over the specified time period.
    """
    _check_time_before_trigger_validity(data_field, times_before_trigger)
    var_in_trigger_area = _init_var_in_trigger_area_multiple(
        mcs_trigger_locs, data_field, times_before_trigger
        )
    
    for j, track in enumerate(mcs_trigger_locs['tracks']):
        # Subsample the data fileld for analysis
        mcs_start_basetime = mcs_trigger_locs.sel(tracks=track)\
            ['start_basetime'].values
        pre_mcs_start_basetime = mcs_start_basetime - times_before_trigger
        if pre_mcs_start_basetime < analysis_time[0]: 
            continue
        var_before_trigger = data_field.sel(
            time=slice(pre_mcs_start_basetime, mcs_start_basetime),
        )

        for i, radius in enumerate(mcs_trigger_locs['radius']):
            trigger_area_idxs = _select_trigger_area_idxs(
                mcs_trigger_locs, track, radius
                )
            var_in_trigger_area[j, :trigger_area_idxs.shape[0], i, :] = \
                var_before_trigger.sel(cell=trigger_area_idxs).data.transpose()
    
    return var_in_trigger_area


def _init_var_in_trigger_area(
         mcs_trigger_locs: xr.DataArray,
        ) -> xr.DataArray:
    """
    Initialize a DataArray that stores variables within the trigger area.

    This function creates an xarray DataArray with dimensions corresponding
    to tracks, cells, and radii, and initializes it with NaN values. The
    coordinates for the DataArray are derived from the input `mcs_trigger_locs`.

    Parameters
    ----------
    mcs_trigger_locs : xr.DataArray
        Input DataArray containing the coordinates `tracks`, `cell`, and
        `radius` that define the dimensions of the trigger area.

    Returns
    -------
    xr.DataArray
        A DataArray initialized with NaN values, having dimensions
        ['tracks', 'cell', 'radius'] and corresponding coordinates.
    """
    tracks = mcs_trigger_locs['tracks']
    cells = mcs_trigger_locs['cell']
    radii = mcs_trigger_locs['radius']

    return xr.DataArray(
        data=np.full(
            shape=(tracks.size, cells.size, radii.size), fill_value=np.nan,
            ),
        dims=['tracks', 'cell', 'radius'],
        coords={'tracks': tracks, 'cell': cells, 'radius': radii},
        )


def _init_var_in_trigger_area_multiple(
        mcs_trigger_locs: xr.DataArray,
        *vars
        ) -> xr.DataArray:
    """
    Initialize a multi-dimensional xarray.DataArray that stores the variables in
    the trigger area.

    This function creates an xarray.DataArray with dimensions corresponding to 
    tracks, cells, radii, and time steps before triggering. The array is filled 
    with NaN values and is intended to store data for multiple variables in the 
    trigger area of MCS (Mesoscale Convective Systems).

    Parameters
    ----------
    mcs_trigger_locs : xr.DataArray
        An xarray.DataArray containing the MCS trigger locations with dimensions 
        'tracks', 'cell', and 'radius'.
    *vars : tuple
        A variable-length argument list of xarray.DataArray objects, which are 
        used to determine the number of time steps before triggering.

    Returns
    -------
    xr.DataArray
        An xarray.DataArray initialized with NaN values, with dimensions 
        ['tracks', 'cell', 'radius', 'time'] and corresponding coordinates.
    """
    tracks = mcs_trigger_locs['tracks']
    cells = mcs_trigger_locs['cell']
    radii = mcs_trigger_locs['radius']

    # Get the number of time steps before triggering
    i_time_before_trigger = _get_i_time_before_trigger(*vars)
    time = np.arange(-i_time_before_trigger, 0, 1)

    return xr.DataArray(
        data=np.full(
            shape=(tracks.size, cells.size, radii.size, time.size),
            fill_value=np.nan,
            ),
        dims=['tracks', 'cell', 'radius', 'time'],
        coords={'tracks': tracks, 'cell': cells, 'radius': radii, 'time': time},
        )


def _get_i_time_before_trigger(
        data_field: xr.DataArray,
        times_before_trigger: np.timedelta64,
        ) -> int:
    """
    Calculate the number of time steps in a data field that fall within a 
    specified time range before the last time step.

    Parameters
    ----------
    data_field : xr.DataArray
        The data array containing a 'time' coordinate to evaluate.
    times_before_trigger : np.timedelta64
        The time duration before the last time step to consider. If None, 
        defaults to 1 time step.

    Returns
    -------
    int
        The number of time steps within the specified time range.
    """
    if times_before_trigger is not None:
        last_time = data_field['time'].isel(time=-1)
        start_time = last_time - times_before_trigger
        data_field = data_field.where(
            (data_field['time'] >= start_time) &
            (data_field['time'] < last_time), drop=True
            )
        return data_field['time'].size
    else:
        return 1
    

def _get_sample_frequency(data_field: xr.DataArray) -> bool:
    """
    Determine the sample frequency of a time-series data field.

    This function calculates the differences between consecutive time points
    in the `time` dimension of the provided xarray DataArray. It ensures that
    the time dimension is uniformly sampled. If the time differences are not
    uniform, a ValueError is raised.

    Parameters
    ----------
    data_field : xr.DataArray
        The input data field containing a `time` dimension.

    Returns
    -------
    sample_frequency : np.ndarray
        The unique time difference representing the sample frequency.

    Raises
    ------
    ValueError
        If the time dimension of the data field is not uniformly sampled.
    """
    sample_frequency = np.unique(data_field.time.diff('time'))
    if sample_frequency.size != 1:
        raise ValueError(
            "The time dimension of the data field is not uniformly sampled."
            )
    else:
        return sample_frequency
    

def _check_time_before_trigger_validity(
        data_field: xr.DataArray,
        times_before_trigger: np.timedelta64,
        ):
    """
    Validates the `times_before_trigger` parameter against the sampling
    frequency of the provided data field.

    Parameters
    ----------
    data_field : xr.DataArray
        The data array containing the time series data for which the
        validation is performed. The sampling frequency is derived from
        this data.
    times_before_trigger : np.timedelta64
        The time duration before the trigger event that needs to be
        validated. Must be a multiple of the sampling frequency.

    Raises
    ------
    ValueError
        If `times_before_trigger` is not a multiple of the sampling
        frequency of the `data_field`.

    Notes
    -----
    The sampling frequency is calculated based on the time intervals
    in the `data_field`. Ensure that the `data_field` has a consistent
    time step for accurate validation.
    """
    sample_frequency = _get_sample_frequency(data_field)
    if (times_before_trigger % sample_frequency) != 0:
        raise ValueError(
            "Please provide a 'times_before_trigger' that is a multiple of " +
            "the sampling frequency of the data field. The sampling " +
            f"frequency is {(sample_frequency / np.timedelta64(1, 'h'))[0]} " +
            "hours.")
    

def _select_trigger_area_idxs(
        mcs_trigger_locs: xr.DataArray,
        track: int,
        radius: int,
        ) -> xr.DataArray:
    """
    Selects the indices of the trigger area for a specific track and radius 
    from the given MCS trigger locations.

    Parameters
    ----------
    mcs_trigger_locs : xr.DataArray
        DataArray containing the MCS trigger locations, including the 
        'trigger_area_idxs' variable.
    track : int
        The track identifier for which the trigger area indices are to be 
        selected.
    radius : int
        The radius value specifying the spatial extent of the trigger area.

    Returns
    -------
    xr.DataArray
        A DataArray containing the indices of the trigger area for the 
        specified track and radius, with NaN values removed.
    """
    trigger_area_idxs = mcs_trigger_locs['trigger_area_idxs']\
        .sel(tracks=track, radius=radius)
    return trigger_area_idxs[~np.isnan(trigger_area_idxs)]