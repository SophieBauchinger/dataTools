# -*- coding: utf-8 -*-
"""
@Author: Sophie Bauchimger, IAU
@Date: Fri Apr 28 09:51:49 2023

Auxiliary functions:
    monthly_mean(df), daily_mean(df), ds_to_gdf(ds)
    rename_columns(columns) - for caribic data extraction

"""
import numpy as np
import datetime as dt
import pandas as pd
import geopandas
from shapely.geometry import Point

import toolpac.calc.binprocessor as bp

from dictionaries import coord_dict, get_col_name

#%% Data extraction
def monthly_mean(df, first_of_month=True):
    """
    Returns dataframe with monthly averages of all values

    df: Pandas DataFrame with datetime index
    first_of_month: bool, if True sets monthly mean timestamp to first of that month
    """
    # group by month then calculate mean
    df_MM = df.groupby(pd.PeriodIndex(df.index, freq="M")).mean(numeric_only=True)

    if first_of_month: # reset index to first of month
        df_MM['Date_Time'] = [dt.datetime(y, m, 1) for y, m in
                              zip(df_MM.index.year, df_MM.index.month)]
        df_MM.set_index('Date_Time', inplace=True)
    return df_MM

def daily_mean(df):
    """
    Returns dataframe with monthly averages of all values
    df: Pandas DataFrame with datetime index
    """
    # group by day then calculate mean
    df_D = df.groupby(pd.PeriodIndex(df.index, freq="D")
                      ).mean(numeric_only=True)
    df_D['Date_Time'] = [dt.datetime(y, m, d) for y, m, d in
                         zip(df_D.index.year, df_D.index.month, df_D.index.day)]
    df_D.set_index('Date_Time', inplace=True)
    return df_D

def ds_to_gdf(ds):
    """ Convert xarray Dataset to GeoPandas GeoDataFrame """
    df = ds.to_dataframe()
    geodata = [Point(lat, lon) for lon, lat in zip(
        df.index.to_frame()['longitude'], df.index.to_frame()['latitude'])]

    # create geodataframe using lat and lon data from indices
    df.reset_index(inplace=True)
    df.drop(['longitude', 'latitude', 'scalar', 'P0'], axis=1, inplace=True)
    gdf = geopandas.GeoDataFrame(df, geometry=geodata)
    index_time = [dt.datetime(y, 1, 1) for y in gdf.time]
    gdf['time'] = index_time
    gdf.set_index('time', inplace=True)
    return gdf

def rename_columns(columns):
    """ Get new column names and col_name_dict for AMES data structure.
    Get only short name + unit; Save description in dict
    Standardise names via case changes
    """
    val_names = [x.split(";")[0] for x in columns if len(x.split(";")) >= 3]
    units = [x.split(";")[-1][:-1] for x in columns if len(x.split(";")) >= 3]

    for i, x in enumerate(val_names): # Standardise column name
        if x.startswith('d_'): val_names[i] = x[:-3] + x[-3:].upper()
        elif x.startswith('int'): continue
        elif not x=='p': val_names[i] = x.upper()

    # coloumn names with corrected case, in correct order
    new_names = [v+u for v,u in zip(val_names, units)]
    # eg. 'CH4 [ppb]' : 'CH4; CH4 mixing ratio; [ppb]\n',
    #     'd_CH4; absolute precision of CH4; [ppb]\n'
    dictionary = dict(zip(new_names, [x for x in columns
                                      if len(x.split(";")) >= 3]))
    dictionary_reversed = dict(zip([x for x in columns
                                    if len(x.split(";")) >= 3], new_names))

    return new_names, dictionary, dictionary_reversed

#%%  Data Handling
def get_lin_fit(df, substance='N2OcatsMLOm', degree=2): # previously get_mlo_fit
    """ Given one year of reference data, find the fit parameters for
    the substance (col name) """
    df.dropna(how='any', subset=substance, inplace=True)
    year, month = df.index.year, df.index.month
    t_ref = year + (month - 0.5) / 12 # obtain frac year for middle of the month
    mxr_ref = df[substance].values
    fit = np.poly1d(np.polyfit(t_ref, mxr_ref, degree))
    print(f'Fit parameters obtained: {fit}')
    return fit

def make_season(month):
    """ If given array of months, return integer representation of seasons
    1 - spring, 2 - summer, 3 - autumn, 4 - winter """
    season = len(month)*[None]
    for i, m in enumerate(month):
        if m in   [3, 4, 5]:    season[i] = 1 # spring
        elif m in [6, 7, 8]:    season[i] = 2 # summer
        elif m in [9, 10, 11]:  season[i] = 3 # autumn
        elif m in [12, 1, 2]:   season[i] = 4 # winter
    return season

#%% Caribic combine GHG measurements with INT and INT2 coordinates
def coord_combo(c_obj, save=True):
    """ Create dataframe with all possible coordinates but
    no measurement / substance values """
    # merge lists of coordinates for all pfxs in the object
    coords = list(set([i for i in [coord_dict()[pfx] for pfx in c_obj.pfxs]
             for i in i])) + ['geometry', 'Flight number']
    df = c_obj.data['GHG'].copy() # copy bc don't want to overwrite data
    for pfx in [pfx for pfx in c_obj.pfxs if pfx!='GHG']:
        df = df.combine_first(c_obj.data[pfx].copy())
    df.drop([col for col in df.columns if col not in coords],
            axis=1, inplace=True) # rmv measurement data

    if save: c_obj.data['coord_combo'] = df
    return df

def subs_merge(c_obj, subs, save=True, detr=True):
    """ Insert GHG data into full coordinate df from coord_merge() """
    # create reference df if it doesn't exist
    if not 'coord_combo' in c_obj.data.keys():
        coord_combo(c_obj)
    substance = get_col_name(subs, c_obj.source, 'GHG')
    ghg_data = pd.DataFrame(c_obj.data['GHG'][substance],
                            index = c_obj.data['GHG'].index)

    if detr:
        ghg_data = ghg_data.join(c_obj.data[f'detr_GHG_{subs}'],
                                 rsuffix='_detrend') # add detrended data to
        ghg_data.drop([x for x in ghg_data.columns if x.endswith('_detrend')],
                      axis=1, inplace=True) # duplicate columns

    merged = c_obj.data['coord_combo'].join(ghg_data, rsuffix='_ghg')
    merged.drop([x for x in merged.columns if x.endswith('_ghg')],
                axis=1, inplace=True) # duplicate columns

    subs_cols = [c for c in ghg_data.columns if c in merged.columns]
    # reorder columns
    merged = merged[subs_cols +
                    [c for c in merged.columns if c not in subs_cols]]

    if save: c_obj.data[f'{subs}_data'] = merged
    return merged

#%% Binning of global data sets
def bin_1d(glob_ob, subs, c_pfx=None, single_yr=None, **kwargs):
    """
    Returns 1D binned objects for each year as lists (lat / lon)
    Parameters:
        substance (str): e.g. 'sf6'
        single_yr (int): if specified, use only data for that year
    """
    print('Changes in tools affect the initialised caribic instance')

    substance = get_col_name(subs, glob_ob.source, c_pfx)

    out_x_list, out_y_list = [], []
    if single_yr is not None: years = [int(single_yr)]
    else: years = glob_ob.years

     # for Caribic, need to choose the df
    if glob_ob.source == 'Caribic': df = glob_ob.data[c_pfx]
    else: df = glob_ob.df

    for yr in years: # loop through available years if possible
        df_yr = df[df.index.year == yr]

        x = np.array([df_yr.geometry[i].x for i in range(len(df_yr.index))]) # lat
        y = np.array([df_yr.geometry[i].y for i in range(len(df_yr.index))]) # lon

        xbmin, xbmax = min(x), max(x)
        ybmin, ybmax = min(y), max(y)

        # average over lon / lat
        # out_x = bin_1d_2d.bin_1d(df_yr[substance], x,
        #                          xbmin, xbmax, self.grid_size)
        # out_y = bin_1d_2d.bin_1d(df_yr[substance], y,
        #                          ybmin, ybmax, self.grid_size)

        out_x = bp.Simple_bin_1d(df_yr[substance], x,
                                 bp.Bin_equi1d(xbmin, xbmax, glob_ob.grid_size))
        out_x.__dict__.update(bp.Bin_equi1d(xbmin, xbmax, glob_ob.grid_size).__dict__)

        out_y = bp.Simple_bin_1d(df_yr[substance], y,
                                 bp.Bin_equi1d(ybmin, ybmax, glob_ob.grid_size))
        out_y.__dict__.update(bp.Bin_equi1d(ybmin, ybmax, glob_ob.grid_size).__dict__)

        out_x_list.append(out_x); out_y_list.append(out_y)

    return out_x_list, out_y_list

def bin_2d(glob_ob, subs, c_pfx=None, single_yr=None, **kwargs):
    """
    Returns 2D binned object for each year as a list
    Parameters:
        substance (str): if None, uses default substance for the object
        single_yr (int): if specified, uses only data for that year
    """
    substance = get_col_name(subs, glob_ob.source, c_pfx)

    out_list = []
    if single_yr is not None: years = [int(single_yr)]
    else: years = glob_ob.years

    # for Caribic, need to choose the df
    if glob_ob.source == 'Caribic': df = glob_ob.data[c_pfx]
    else: df = glob_ob.df

    for yr in years: # loop through available years if possible
        df_yr = df[df.index.year == yr]

        x = np.array([df_yr.geometry[i].x for i in range(len(df_yr.index))]) # lat
        y = np.array([df_yr.geometry[i].y for i in range(len(df_yr.index))]) # lon

        xbmin, xbmax, xbsize = min(x), max(x), glob_ob.grid_size
        ybmin, ybmax, ybsize = min(y), max(y), glob_ob.grid_size

        # out = bin_1d_2d.bin_2d(np.array(df_yr[substance]), x, y,
        #                        xbmin, xbmax, xbsize, ybmin, ybmax, ybsize)

        out = bp.Simple_bin_2d(np.array(df_yr[substance]), x, y,
                               bp.Bin_equi1d(xbmin, xbmax, xbsize,
                                             ybmin, ybmax, ybsize))
        out.__dict__.update(bp.Bin_equi1d(xbmin, xbmax, xbsize,
                                          ybmin, ybmax, ybsize).__dict__)

        out_list.append(out)
    return out_list