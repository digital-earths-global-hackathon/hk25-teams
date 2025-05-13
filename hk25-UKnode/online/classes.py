'''
Created on May 10, 2022

@author: cshort
'''

import numpy as np

MISVAL = -999


class StormS:
    storm = []
    area = []
    centroidx = []
    centroidy = []
    centroidlon = []  # WK - added so that can plot lat/lon tracks
    centroidlat = []  # WK - added so that can plot lat/lon tracks
    boxleft = []
    boxup = []
    boxwidth = []
    boxheight = []
    was = []
    life = []
    track_xpos = []
    track_ypos = []
    u = [MISVAL]
    v = [MISVAL]
    maxrain = []
    meanrain = []
    parent = []
    child = []
    overlap_area_with_chosen_advected_storm = []
    accreted = []
    cell = []


class Track(object):
    def __init__(self, ID, storms):
        self.ID = ID
        self.active = True
        if not isinstance(storms, list):
            storms = [storms]
        self.storms = storms

    def add_storm(self, storm):
        self.storms.append(storm)

    def get_storm(self, time):
        storms_at_time = []
        for storm in self.storms:
            if storm.time == time:
                storms_at_time.append(storm)
        if len(storms_at_time) > 1:
            raise ValueError("Can't have more than one storm in track at previous time")
        elif len(storms_at_time) == 1:
            return storms_at_time[0]
        else:
            return None

    def get_storms(self, start_time=None, end_time=None):
        if start_time is None and end_time is None:
            storms = self.storms
        elif start_time is None:
            storm_at_end_time = self.get_storm(end_time)
            if storm_at_end_time is None:
                storms = []
            else:
                storms = [storm for storm in self.storms if storm.time <= end_time]
        elif end_time is None:
            storm_at_start_time = self.get_storm(start_time)
            if storm_at_start_time is None:
                storms = []
            else:
                storms = [storm for storm in self.storms if storm.time >= start_time]
        else:
            storm_at_start_time = self.get_storm(start_time)
            storm_at_end_time = self.get_storm(end_time)
            if storm_at_start_time is None or storm_at_end_time is None:
                storms = []
            else:
                storms = [storm for storm in self.storms if start_time <= storm.time <= end_time]

        return storms

    def get_times(self):
        return [storm.time for storm in self.storms]

    def get_times_hhmm(self):
        return [storm.time.strftime('%H%M') for storm in self.storms]

    def get_start_time(self):
        first_storm = self.storms[0]
        if self.get_lifetime() > 1 and first_storm.status not in ["I", "SI"]:
            raise ValueError("Storm initial status is wrong")
        return first_storm.time

    def get_end_time(self):
        last_storm = self.storms[-1]
        if last_storm.status not in ["T", "MT"]:
            print("Track is still active")
            print(self.active)
            return None
        else:
            return last_storm.time

    def get_statuses(self):
        return [storm.status for storm in self.storms]

    def get_deviation_angles(self, remove_nans=False):
        angles = [storm.deviation_angle for storm in self.storms]
        if remove_nans:
            angles = [angle for angle in angles if not np.isnan(angle)]
        return angles

    def get_changes_in_direction(self):
        angles = [storm.change_in_direction for storm in self.storms]
        return angles

    def get_lifetime(self):
        return len(self.storms)

    def is_primary_tracked(self):
        # When a new storm is identified, primary_tracked is set to F by default.
        # This could bias the counts of primary vs secondary tracked. Therefore
        # return None at the initial time instead
        # TODO This might need adapting to deal with the ["T"] track case
        return [storm.primary_tracked if storm.status != "I" else None for storm in self.storms]

    def get_max_area(self):
        return np.max([storm.area for storm in self.storms])

    def get_mean_precip_rate(self):
        return np.mean([storm.meanrain for storm in self.storms])

    def get_mean_precip_rates(self):
        return [storm.meanrain for storm in self.storms]

    def get_mean_Tbs(self):
        return [storm.meanTb for storm in self.storms]

    def get_max_precip_rate(self):
        return np.max([storm.meanrain for storm in self.storms])

    def get_max_precip_rates(self):
        return [storm.meanrain for storm in self.storms]

    def get_total_precip(self, time_res_mins):
        # TODO: Need to double check units here
        # PBA19 has total precip in units of m3
        return np.sum([storm.meanrain * (time_res_mins / 60.0) for storm in self.storms])

    def get_total_precip_mass(self, grid_length_m, time_res_mins):
        # storm.meanrain is in units of kg m-2 h-1
        return np.sum(
            [storm.meanrain * (storm.area * grid_length_m * grid_length_m) * (time_res_mins / 60.0) for storm in
             self.storms])

    def is_in_region(self, region):
        """
        :param region: rgion dictionary, like  reg_SA = dict(lons=(13,35), lats=(-35,-22) )
        :return: Tracks that have a storm initiating in this region
        """
        in_region = [(storm.centroidlon  > region['lons'][0]) & (storm.centroidlon <=region['lons'][1]) & (storm.centroidlat  > region['lats'][0]) & (storm.centroidlat <=region['lats'][1]) for storm in self.storms]
        # print("360 being subtracted from lon")
        if all(in_region):
            return True
        else:
            return False

class Config(object):
    def __init__(self, wind_type, max_dist_type, centroid_type, grid_length,
                 time_res_mins, smoothing_pixels, thresholds, padding_type,
                 padding_pixels):
        self.wind_type = wind_type
        self.max_dist_type = max_dist_type
        self.centroid_type = centroid_type
        self.grid_length = grid_length
        self.time_res_mins = time_res_mins
        self.smoothing_pixels = smoothing_pixels
        self.thresholds = thresholds
        self.padding_type = padding_type
        self.padding_pixels = padding_pixels

    def get_name(self):
        return "config_{:s}_{:s}_{:s}_{:.1f}km_{:d}min_{:d}sm_{:s}_{:s}{:d}px".format(self.wind_type,
                                                                                      self.max_dist_type,
                                                                                      self.centroid_type,
                                                                                      self.grid_length,
                                                                                      self.time_res_mins,
                                                                                      self.smoothing_pixels,
                                                                                      self.thresholds,
                                                                                      self.padding_type,
                                                                                      self.padding_pixels)