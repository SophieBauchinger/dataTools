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
        df_MM['Date_Time'] = [dt.datetime(y, m, 1) for y, m in zip(df_MM.index.year, df_MM.index.month)]
        df_MM.set_index('Date_Time', inplace=True)
    return df_MM

def daily_mean(df):
    """
    Returns dataframe with monthly averages of all values
    df: Pandas DataFrame with datetime index
    """
    # group by day then calculate mean
    df_D = df.groupby(pd.PeriodIndex(df.index, freq="D")).mean(numeric_only=True)
    df_D['Date_Time'] = [dt.datetime(y, m, d) for y, m, d in zip(df_D.index.year, df_D.index.month, df_D.index.day)]
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
    """ Get new column names and col_name_dict for AMES data structure """
    # get val_name from start of the line, then unit from last part
    val_names = [x.split(";")[0] for x in columns if len(x.split(";")) >= 3] # columns with short; long; unit
    units = [x.split(";")[-1][:-1] for x in columns if len(x.split(";")) >= 3] # get last part [unit] with [-1], then remove \n with [:-1]

    # Need to make column names of the same substance the same even though some are upper / lower case
    for i, x in enumerate(val_names): 
        if x.startswith('d_'): val_names[i] = x[:-3] + x[-3:].upper() # changing the letter case, nothing else
        elif x.startswith('int'): continue # no change in case for int_ variable names
        elif not x=='p': val_names[i] = x.upper() # not making p uppercase, only subst names
    
    new_names = [v+u for v,u in zip(val_names, units)] # coloumn names with corrected case, in correct order
    dictionary = dict(zip(new_names, [x for x in columns if len(x.split(";")) >= 3])) # eg. 'CH4 [ppb]' : 'CH4; CH4 mixing ratio; [ppb]\n', 'd_CH4; absolute precision of CH4; [ppb]\n'
    dictionary_reversed = dict(zip([x for x in columns if len(x.split(";")) >= 3], new_names))

    return new_names, dictionary, dictionary_reversed

# def same_merge(x): return ','.join(x[x.notnull()].astype(str))
# def same_col_merge(df):
#     """ Merge all columns with the same name when given a dataframe """
#     return df.groupby(level=0, axis=1).apply(lambda x: x.apply(same_merge, axis=1))

#%%  Data Handling
def get_lin_fit(df, substance='N2OcatsMLOm', degree=2): # previously get_mlo_fit
    """ Given one year of reference data, find the fit parameters for the substance (col name) """
    df.dropna(how='any', subset=substance, inplace=True)
    year, month = df.index.year, df.index.month
    t_ref = year + (month - 0.5) / 12 # obtain fractional year for middle of the month
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
    """ Create dataframe with all possible coordinates but no measurement / substance values """
    coords = list(set([i for i in [coord_dict()[pfx] for pfx in c_obj.pfxs] for i in i])) + ['geometry', 'Flight number'] # merge lists of coordinates for all pfxs in the object
    df = c_obj.data['GHG'].copy() # copy bc don't want to overwrite data 
    for pfx in [pfx for pfx in c_obj.pfxs if pfx!='GHG']:
        df = df.combine_first(c_obj.data[pfx].copy())
    df.drop([col for col in df.columns if col not in coords], axis=1, inplace=True) # rmv measurement data

    if save: c_obj.data['coord_combo'] = df
    return df

def subs_merge(c_obj, subs, save=True, detr=True):
    """ Insert GHG data into full coordinate dataframe obtained from coord_merge() """
    if not 'coord_combo' in c_obj.data.keys(): # create reference df if it doesn't exist
        coord_combo(c_obj)
    substance = get_col_name(subs, c_obj.source, 'GHG')
    ghg_data = pd.DataFrame(c_obj.data['GHG'][substance], index = c_obj.data['GHG'].index)

    if detr: 
        ghg_data = ghg_data.join(c_obj.data[f'detr_GHG_{subs}'], rsuffix='_detrend') # add detrended data to 
        ghg_data.drop([x for x in ghg_data.columns if x.endswith('_detrend')], axis=1, inplace=True) # duplicate columns

    merged = c_obj.data['coord_combo'].join(ghg_data, rsuffix='_ghg')
    merged.drop([x for x in merged.columns if x.endswith('_ghg')], axis=1, inplace=True) # duplicate columns

    subs_cols = [c for c in ghg_data.columns if c in merged.columns]
    merged = merged[subs_cols + [c for c in merged.columns if c not in subs_cols]] # reorder columns 

    if save: c_obj.data[f'{subs}_data'] = merged
    return merged