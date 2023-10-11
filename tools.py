# -*- coding: utf-8 -*-
""" Auxiliary functions for data extraction and handling.

@Author: Sophie Bauchinger, IAU
@Date: Fri Apr 28 09:51:49 2023

"""
import numpy as np
import datetime as dt
import pandas as pd
import geopandas
from shapely.geometry import Point
import copy
from metpy.units import units

import toolpac.calc.binprocessor as bp
from toolpac.conv.times import datetime_to_fractionalyear as dt_to_fy

import dictionaries as dcts

#%% Data extraction
def yearly_mean(df):
    """ Return the yearly mean and min/max values of all columns in a dataframe """
    cols = [c for c in df.columns if c in [s.col_name for s in dcts.get_substances()]]
    df = df[cols]

    df_mean = df.groupby(pd.PeriodIndex(df.index, freq="Y")).mean(numeric_only=True)
    df_mean = df_mean.rename(columns = {c : 'mean_'+c for c in df_mean.columns})

    df_min = df.groupby(pd.PeriodIndex(df.index, freq="Y")).min(numeric_only=True)
    df_min = df_min.rename(columns = {c : 'min_'+c for c in df_min.columns})

    df_max = df.groupby(pd.PeriodIndex(df.index, freq="Y")).max(numeric_only=True)
    df_max = df_max.rename(columns = {c : 'max_'+c for c in df_max.columns})
    
    df = pd.concat([df_mean, df_min, df_max], axis=1)
    return df

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

def ds_to_gdf(ds, source='Mozart'):
    """ Convert xarray Dataset to GeoPandas GeoDataFrame """
    df = ds.to_dataframe()

    if 'longitude' in df.columns and 'latitude' in df.columns:
        # drop rows without geodata
        df.dropna(subset=['longitude', 'latitude'], how='any', inplace=True)
        geodata = [Point(lat, lon) for lat, lon in zip(
            df['latitude'], df['longitude'])]
    else:
        geodata = [Point(lat, lon) for lon, lat in zip(
        df.index.to_frame()['longitude'], df.index.to_frame()['latitude'])]

    # create geodataframe using lat and lon data from indices
    df.reset_index(inplace=True)
    for drop_col in ['longitude', 'latitude', 'scalar', 'P0']: # drop as unnecessary
        if drop_col in df.columns: df.drop([drop_col], axis=1, inplace=True)
    gdf = geopandas.GeoDataFrame(df, geometry=geodata)

    if not gdf.time.dtype == '<M8[ns]': # mzt, check if time is not in datetime format
        index_time = [dt.datetime(y, 1, 1) for y in gdf.time]
        gdf['time'] = index_time
    gdf.set_index('time', inplace=True)
    gdf.index = gdf.index.floor('S') # remove micro/nanoseconds

    return gdf

def rename_columns(columns):
    """ Create dictionary relating column name with AMES_variable object

    Relate dataframe column name with all information in

    Get new column names and col_name_dict for AMES data structure.
    Get only short name + unit; Save description in dict
    Standardise names via case changes
    """
    col_dict = {}
    rename_dict = {}
    for x in columns:
        if len(x.split(';')) == 3:
            col_name, long_name, unit = [i.strip() for i in x.split(';')]
        else:
            col_name = x.split(";")[0].strip()

        rename_dict.update({x : col_name})
        if col_name in [i.col_name for i in dcts.get_coordinates()]:
            coord = dcts.get_coord(col_name=col_name) # store info
            col_dict.update({col_name : coord})

    return col_dict, rename_dict

def process_emac_s4d(ds, incl_model=True, incl_tropop=True, incl_subs=True):
    """ Choose which variables to keep when importing EMAC data .

    Parameters:
        ds: currrent xarray dataset
        inlc_subs (bool): keep tracer substances
        incl_model (bool): keep modelled meteorological data
        incl_tropop (bool): keep tropopause-relevant variabels

    Variable description:
        time - datetime [ns]
        tlon - track longitude [degrees_east]
        tlat - track latitude [degrees_north]
        tpress - track pressure [hPa]
        tps - track surface pressure [Pa]
        tracer_* - modelled substances [mol/mol]
        tropop_* - tropopause relevant variables
        ECHAM5_* - modelled met. data
        e5vdiff_tpot* - potential temperature [K]
    """
    variables = ['time', 'tlon', 'lev', 'tlat', 'tpress', 'tps']
    if incl_model:
        variables.extend([v for v in ds.variables
                          if v.startswith(('ECHAM5_', 'e5vdiff_tpot'))
                          and not v.endswith(('m1', 'aclc'))])
    if incl_tropop:
        variables.extend([v for v in ds.variables
                          if v.startswith('tropop_') and not v.endswith('_f')
                          and not any([x in v for x in ['_clim', 'pblh']])])
    if incl_subs:
        tracers = dcts.get_substances(**{'ID':'EMAC'})
        tracers_at_fl = [t+'_at_fl' for t in tracers]
        variables.extend([v for v in ds.variables if
                          (v in tracers or v in tracers_at_fl) ])
    # only keep specified variables
    ds = ds[variables]
    for var in ds.variables: # streamline units
        if hasattr(ds[var], 'units'):
            if ds[var].units == 'Pa': ds[var] = ds[var].metpy.convert_units(units.hPa)
            elif ds[var].units == 'm': ds[var] = ds[var].metpy.convert_units(units.km)
            ds[var] = ds[var].metpy.dequantify() # makes units an attribute again
    # if either lon or lat are nan, drop that timestamp
    ds = ds.dropna(subset=['tlon', 'tlat'], how='any', dim='time')
    ds = ds.rename({'tlon':'longitude', 'tlat':'latitude'})
    ds['time'] = ds.time.dt.round('S') # rmvs floating pt errors
    return ds

def process_emac_s4d_s(ds, incl_model=True, incl_tropop=True, incl_subs=True):
    """ Keep only variables that depend only on time and are available in
    subsampled data """
    ds = process_emac_s4d(ds, incl_model, incl_tropop, incl_subs)
    variables = [v for v in ds.variables if ds[v].dims == ('time',)]
    return ds[variables]

#%% Data selection
def data_selection(glob_obj, flights=None, years=None, latitudes=None,
                   tropo=False, strato=False, extr_events=False, **kwargs):
    """ Return new Caribic instance with selection of parameters
        flights (int / list(int))
        years (int / list(int))
        latitudes (tuple): lat_min, lat_max
        tropo, strato, extr_events (bool)
        kwargs: e.g. tp_def, ... - for strat / trop filtering
    """
    out = copy.deepcopy(glob_obj)
    if flights is not None: out = out.sel_flight(flights)
    if years is not None: out = out.sel_year(years)
    if latitudes is not None:
        out = out.sel_latitude(*latitudes)
        out.status.update({'latitudes' : latitudes})
    if strato:
        out = out.sel_strato(**kwargs)
        out.status.update({'strato' : True})
    if tropo:
        out = out.sel_tropo(**kwargs)
        out.status.update({'tropo' : True})
    if extr_events:
        out = out.filter_extreme_events(**kwargs)
        out.status.update({'no_ee' : True})
    return out

#%% Data Handling
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

def assign_t_s(df, TS, coordinate, tp_val=0):
    """ Returns the bool series of t / s after applying appropriate comparison for a chosen vcoord.

    Parameters:
        df (DataFrame): reference data - e.g. track pressure / TP p distance to track p
        TS (str): 't' / 's';  indicates troposphere / stratosphere
        coord (str): dp, pt, z

    optional:
        tp_val (float): value of tropopause in chosen coordinates. For non-relative coords
    """
    if ((coordinate in ['p', 'dp'] and TS == 't')
        or (coordinate in ['pt', 'z'] and TS == 's')):
        return df.gt(tp_val)

    elif ((coordinate in ['p', 'dp'] and TS == 's')
          or (coordinate in ['pt', 'z'] and TS == 't')):
        return df.lt(tp_val)

    else: raise KeyError(f'Strat/Trop assignment undefined for {coordinate}')

def get_lin_fit(series, degree=2): # previously get_mlo_fit
    """ Given one year of reference data, find the fit parameters for
    the substance (col name) """
    year, month = series.index.year, series.index.month
    t_ref = year + (month - 0.5) / 12 # obtain frac year for middle of the month
    mxr_ref = series.values
    fit = np.poly1d(np.polyfit(t_ref, mxr_ref, degree))
    print(f'Fit parameters obtained: {fit}')
    return fit

def pre_flag(data_arr, ref_arr, crit='n2o', limit = 0.97, **kwargs):
    """ Sort data into strato / tropo based on difference to ground obs.

    Returns dataframe containing index and strato/tropo/pre_flag columns

    Parameters:
        data_arr (pd.Series) : msmt data to be sorted into stratr / trop air
        ref_arr (pd.Series) : reference data to use for filtering (background)
        crit (str) : substance to use for flagging
        limit (float) : tracer mxr fraction below which air is classified
                        as stratospheric
    """
    data_arr.sort_index(inplace=True)
    df_flag = pd.DataFrame({f'strato_{data_arr.name}':np.nan,
                            f'tropo_{data_arr.name}':np.nan},
                           index=data_arr.index)

    fit = get_lin_fit(ref_arr)
    t_obs_tot = np.array(dt_to_fy(df_flag.index, method='exact'))
    df_flag.loc[data_arr < limit * fit(t_obs_tot),
           (f'strato_{data_arr.name}', f'tropo_{data_arr.name}')] = (True, False)

    df_flag[f'flag_{crit}'] = 0
    df_flag.loc[df_flag[f'strato_{data_arr.name}'] == True, f'flag_{crit}'] = 1

    if kwargs.get('verbose'):
        print('Result of pre-flagging: \n',
              df_flag[f'flag_{crit}'].value_counts())
    return df_flag

def conv_molarity_PartsPer(x, unit):
    """ Convert molarity (mol/mol) to given unit (eg. ppb). """
    factor = {'ppm' : 1e6, # per million
               'ppb' : 1e9, # per billion
               'ppt' : 1e12, # per trillion
               'ppq' : 1e15, # per quadrillion
               }
    # e.g. n2o: 300 ppb, 3e-7 mol/mol
    return x*factor[unit]

def conv_PartsPer_molarity(x, unit):
    """ Convert x from [unit] to molarity (mol/mol) """
    factor = {'ppm' : 1e-6, # per million
               'ppb' : 1e-9, # per billion
               'ppt' : 1e-12, # per trillion
               'ppq' : 1e-15, # per quadrillion
               }
    return x*factor[unit]

#%% Caribic combine GHG measurements with INT and INT2 coordinates
def coord_merge_substance(c_obj, subs, save=True, detr=True):
    """ Insert msmt data into full coordinate df from coord_merge() """
    # create reference df if it doesn't exist
    if not 'met_data' in dir(c_obj): df = c_obj.coord_combo()
    else: df = c_obj.met_data.copy()

    if detr:
        try: c_obj.detrend(subs) # add detrended data to all dataframes
        except: print(f'Detrending unsuccessful for {subs.upper()}, proceeding without. ')

    subs_cols = dcts.get_substances(short_name=subs, source=c_obj.source)
    subs_cols.update(dcts.get_substances(short_name='d_'+subs, source=c_obj.source))

    for pfx in c_obj.pfxs:
        data = c_obj.data[pfx].sort_index()
        cols = [k for k,v in subs_cols.items() if v.ID == pfx and v.col_name in data.columns]
        df = df.join(data[cols])

    # Reorder columns to match initial dataframes & put substance to front
    df = df[list(['Flight number', 'p [mbar]'] + cols
                 + [c for c in df.columns if c not in
                    list(['Flight number', 'p [mbar]', 'geometry']+cols)]
                 + ['geometry'])]

    df.dropna(subset = subs_cols, how='all', inplace=True) # drop rows without any subs data
    return df

#%% Binning of global data sets
# def bin_prep(glob_obj, subs, **kwargs):
#     c_pfx = kwargs.get('c_pfx') # only for caribic data; otherwise None
#     substance = dcts.get_col_name(subs, glob_obj.source, c_pfx)

#     if kwargs.get('single_yr') is not None:
#         years = [int(kwargs.get('single_yr'))]
#     else: years = glob_obj.years

#     # for Caribic, need to choose the df
#     if glob_obj.source == 'Caribic': df = glob_obj.data[c_pfx]
#     else: df = glob_obj.df

#     if kwargs.get('detr') is True:
#         if not 'detr_'+substance in df.columns:
#             glob_obj.detrend(subs)
#         substance = 'detr_' + substance

#     print(substance)

#     return substance, years, df

def bin_1d(glob_obj, subs, **kwargs):
    """ Returns 1D binned objects for each year as lists (lat / lon)

    Parameters:
        subs (str): e.g. 'sf6'.
    Optional parameters:
        c_pfx (str): caribic file pfx, required for caribic data
        single_yr (int): if specified, use only data for that specific year

    Returns:
        out_x_list, out_y_list: lists of Bin1D objects binned along x / y
    """
    # df = glob_obj.df
    substance = subs.col_name

    if kwargs.get('xbinlimits') is not None: # not equidistant binning
        x_binclassinstance = bp.Bin_notequi1d(kwargs.get('xbinlimits'))
    if kwargs.get('ybinlimits') is not None:
        y_binclassinstance = bp.Bin_notequi1d(kwargs.get('ybinlimits'))

    out_x_list, out_y_list = [], []
    for yr in glob_obj.years: # loop through available years
        df_yr = glob_obj.df[glob_obj.df.index.year == yr]

        x = np.array([df_yr.geometry[i].x for i in range(len(df_yr.index))]) # lat
        if kwargs.get('xbinlimits') is None: # equidistant binning
            xbmin, xbmax = min(x), max(x)
            x_binclassinstance = bp.Bin_equi1d(xbmin, xbmax, glob_obj.grid_size)
        out_x = bp.Simple_bin_1d(df_yr[substance], x, x_binclassinstance)
        out_x.__dict__.update(x_binclassinstance.__dict__)

        y = np.array([df_yr.geometry[i].y for i in range(len(df_yr.index))]) # lon
        if kwargs.get('ybinlimits') is None:
            ybmin, ybmax = min(y), max(y)
            y_binclassinstance = bp.Bin_equi1d(ybmin, ybmax, glob_obj.grid_size)
        out_y = bp.Simple_bin_1d(df_yr[substance], y, y_binclassinstance)
        out_y.__dict__.update(y_binclassinstance.__dict__)

        if not all(np.isnan(out_x.vmean)) or all(np.isnan(out_y.vmean)):
            out_x_list.append(out_x); out_y_list.append(out_y)
        else: print(f'everything is nan for {yr}')

    return out_x_list, out_y_list

def bin_2d(glob_obj, subs, **kwargs):
    """
    Returns 2D binned object for each year as a list
    Parameters:
        substance (str): if None, uses default substance for the object
        single_yr (int): if specified, uses only data for that year
    """
    substance = subs.col_name

    out_list = []
    for yr in glob_obj.years: # loop through available years if possible
        df_yr = glob_obj.df[glob_obj.df.index.year == yr]

        xbinlimits = kwargs.get('xbinlimits')
        ybinlimits = kwargs.get('ybinlimits')

        x = np.array([df_yr.geometry[i].x for i in range(len(df_yr.index))]) # lat
        if xbinlimits is None:
            # use equidistant binning if not specified else
            xbmin, xbmax = min(x), max(x)
            xbinlimits = list(bp.Bin_equi1d(xbmin, xbmax, glob_obj.grid_size).xbinlimits)

        y = np.array([df_yr.geometry[i].y for i in range(len(df_yr.index))]) # lon
        if ybinlimits is None:
            ybmin, ybmax = min(y), max(y)
            ybinlimits = list(bp.Bin_equi1d(ybmin, ybmax, glob_obj.grid_size).xbinlimits)

        # create binclassinstance that's valid for both equi and nonequi
        binclassinstance = bp.Bin_notequi2d(xbinlimits, ybinlimits)
        out = bp.Simple_bin_2d(np.array(df_yr[substance]), x, y, binclassinstance)
        out.__dict__.update(binclassinstance.__dict__)

        out_list.append(out)
    return out_list