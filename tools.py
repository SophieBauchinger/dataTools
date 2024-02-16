# -*- coding: utf-8 -*-
""" Auxiliary functions for data extraction and handling.

@Author: Sophie Bauchinger, IAU
@Date: Fri Apr 28 09:51:49 2023

"""
import numpy as np
import datetime as dt
import os
import pandas as pd
import geopandas
from shapely.geometry import Point
from metpy.units import units

import toolpac.calc.binprocessor as bp
from toolpac.conv.times import datetime_to_fractionalyear as dt_to_fy
from toolpac.conv.times import secofday_to_datetime

import dictionaries as dcts


# %% Data extraction

def time_mean(df, f, first_of_month=True, minmax=False) -> pd.DataFrame:
    """ Group values by time and return the respective averages.
    Parameters: 
        df (pd.DataFrame): Input data to be averaged
        f (str): Frequency. One of 'D', 'M', 'Y'
        first_of_month (bool): Set day column to 1
        minmax (bool): Return dataframe containing mean, max and min of the grouped data
    """
    # Only average substance columns
    cols = [c for c in df.columns if c in [s.col_name for s in dcts.get_substances()]]
    df = df[cols]

    df_mean = df.groupby(pd.PeriodIndex(df.index, freq=f)).mean(numeric_only=True)
    df_mean['Date_Time'] = np.nan
    for i, (y, m, d) in enumerate(zip(
            df_mean.index.year,
            df_mean.index.month if f != 'Y' else None,
            df_mean.index.day if f == 'D' else (
                [1] * len(df_mean.index) if first_of_month else None))):
        df_mean['Date_Time'][i] = dt.datetime(y, m, d)

    df_mean.set_index('Date_Time', inplace=True)
    if minmax:
        df_min = df.groupby(pd.PeriodIndex(df.index, freq=f)).min(numeric_only=True)
        df_max = df.groupby(pd.PeriodIndex(df.index, freq=f)).max(numeric_only=True)

        df_mean = df_mean.rename(columns={c: 'mean_' + c for c in df_mean.columns})
        df_min = df_min.rename(columns={c: 'min_' + c for c in df_min.columns})
        df_max = df_max.rename(columns={c: 'max_' + c for c in df_max.columns})

        return pd.concat([df_mean, df_min, df_max], axis=1)

    return df_mean

def ds_to_gdf(ds) -> pd.DataFrame:
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
    for drop_col in ['longitude', 'latitude', 'scalar', 'P0']:  # drop as unnecessary
        if drop_col in df.columns: df.drop([drop_col], axis=1, inplace=True)
    gdf = geopandas.GeoDataFrame(df, geometry=geodata)

    if not gdf.time.dtype == '<M8[ns]':  # mzt, check if time is not in datetime format
        index_time = [dt.datetime(y, 1, 1) for y in gdf.time]
        gdf['time'] = index_time
    gdf.set_index('time', inplace=True)
    gdf.index = gdf.index.floor('S')  # remove micro/nanoseconds

    return gdf

def rename_columns(columns) -> (dict, dict):
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

        rename_dict.update({x: col_name})
        if col_name in [i.col_name for i in dcts.get_coordinates()]:
            coord = dcts.get_coord(col_name=col_name)  # store info
            col_dict.update({col_name: coord})

    return col_dict, rename_dict

def process_emac_s4d(ds, incl_model=True, incl_tropop=True, incl_subs=True):
    """ Choose which variables to keep when importing EMAC data .

    Parameters:
        ds: currrent xarray dataset
        incl_subs (bool): keep tracer substances
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
        tracers = dcts.get_substances(**{'ID': 'EMAC'})
        tracers_at_fl = [t + '_at_fl' for t in tracers]
        variables.extend([v for v in ds.variables if
                          (v in tracers or v in tracers_at_fl)])
    # only keep specified variables
    ds = ds[variables]
    for var in ds.variables:  # streamline units
        if hasattr(ds[var], 'units'):
            if ds[var].units == 'Pa':
                ds[var] = ds[var].metpy.convert_units(units.hPa)
            elif ds[var].units == 'm':
                ds[var] = ds[var].metpy.convert_units(units.km)
            ds[var] = ds[var].metpy.dequantify()  # makes units an attribute again
    # if either lon or lat are nan, drop that timestamp
    ds = ds.dropna(subset=['tlon', 'tlat'], how='any', dim='time')
    ds = ds.rename({'tlon': 'longitude', 'tlat': 'latitude'})
    ds['time'] = ds.time.dt.round('S')  # rmvs floating pt errors
    return ds


def process_emac_s4d_s(ds, incl_model=True, incl_tropop=True, incl_subs=True):
    """ Keep only variables that depend only on time and are available in
    subsampled data """
    ds = process_emac_s4d(ds, incl_model, incl_tropop, incl_subs)
    variables = [v for v in ds.variables if ds[v].dims == ('time',)]
    return ds[variables]

def process_caribic(ds): 
    # ds = ds.drop_dims([d for d in ds.dims if 'header_lines' in d])
    variables = [v for v in ds.variables if ds[v].dims == ('time',)]
    return ds[variables]


def process_clams(ds): 
    variables = ['ERA5_TEMP',
                 'ERA5_PRESS',
                 'ERA5_THETA',
                 'ERA5_GPH',
                 'ERA5_PV',
                 'ERA5_EQLAT',

                 'ERA5_TROP1_THETA',
                 'ERA5_TROP1_PRESS',
                 'ERA5_TROP1_Z',
                 'ERA5_PRESS_2_0_Main',
                 'ERA5_PRESS_3_5_Main',
                 'ERA5_THETA_2_0_Main',
                 'ERA5_THETA_3_5_Main',
                 'ERA5_GPH_2_0_Main',
                 'ERA5_GPH_3_5_Main',]

    return ds[variables]

def process_atom_clams(ds):
    """ Additional time values for ATom as otherwise the function breaks """    
    variables = ['ATom_UTC_Start', 
                 'ERA5_TEMP', 'ERA5_PRESS', 'ERA5_THETA',
                 'ERA5_GPH', 'ERA5_PV', 'ERA5_EQLAT',
                 'ERA5_TROP1_THETA',
                 'ERA5_TROP1_PRESS',
                 'ERA5_TROP1_Z',
                 'ERA5_PRESS_2_0_Main',
                 'ERA5_PRESS_3_5_Main',
                 'ERA5_THETA_2_0_Main',
                 'ERA5_THETA_3_5_Main',
                 'ERA5_GPH_2_0_Main',
                 'ERA5_GPH_3_5_Main',]
    ds = ds[variables]
    
    # find flight date from file name
    filepath = ds['ATom_UTC_Start'].encoding['source']
    fname = os.path.basename(filepath)
    date_string = fname.split('_')[1]
    date = dt.datetime(year = int(date_string[:4]), 
                        month = int(date_string[4:6]), 
                        day = int(date_string[-2:]))
    
    # generate datetimes for each timestamp
    datetimes = [secofday_to_datetime(date, secofday + 5) for secofday in ds['ATom_UTC_Start'].values]
    ds = ds.assign(Time = datetimes) 
    
    ds = ds.drop_vars('ATom_UTC_Start')
    
    return ds


# %% Data selection
def minimise_tps(tps) -> list:
    """ Returns a reduced list of tropopause coordinates.
    
    1. Remove vcoords != pt if tp exists with pt
    2. remove all cpt, combo tp
    3. remove all ECMWF tps 
    4. Remove modelled N2O tp
    5. Remove duplicates of O3 tp 
    6. Remove 1.5 PVU ERA5 dyn tp
    7. Remove non-relative tps if relative exists in tps
    """
    # check if coord exists with pt, remove if it does 
    tp_to_remove = []
    for tp in tps:  # 1
        try:
            dcts.get_coord(vcoord='pt', model=tp.model, source=tp.source,
                           tp_def=tp.tp_def,
                           pvu=tp.pvu, crit=tp.crit, rel_to_tp=tp.rel_to_tp)
        except KeyError:
            continue
        if not tp.vcoord == 'pt': tp_to_remove.append(tp)
    [tp_to_remove.append(tp) for tp in tps if tp.tp_def in ['cpt', 'combo']]  # 2
    [tp_to_remove.append(tp) for tp in tps if tp.model in ['ECMWF', 'EMAC']]  # 3
    [tp_to_remove.append(tp) for tp in tps
     if tp.col_name in [tp.col_name for tp in dcts.get_coordinates(tp_def='chem', crit='n2o', model='not_MSMT')]]  # 4
    [tp_to_remove.append(tp) for tp in tps
     if tp.col_name in ['int_CARIBIC2_H_rel_TP', 'int_O3']]
    [tp_to_remove.append(tp) for tp in tps if tp.pvu == 1.5]  # 6
    for tp in tps:  # 7
        try:
            rel_tp = dcts.get_coord(rel_to_tp=True, model=tp.model, source=tp.source,
                                    tp_def=tp.tp_def,
                                    pvu=tp.pvu, crit=tp.crit, vcoord=tp.vcoord)
        except KeyError:
            continue
        if (any(tp.col_name == rel_tp.col_name for tp in tps)
                and tp.rel_to_tp is False):
            tp_to_remove.append(tp)
    tps = tps.copy()
    for tp in set(tp_to_remove): tps.remove(tp)
    tps.sort(key=lambda x: x.tp_def)
    return tps


# %% Data Handling
def make_season(month) -> np.array:
    """ If given array of months, return integer representation of seasons
    1 - spring, 2 - summer, 3 - autumn, 4 - winter """
    season = len(month) * [None]
    for i, m in enumerate(month):
        if m in [3, 4, 5]:
            season[i] = 1  # spring
        elif m in [6, 7, 8]:
            season[i] = 2  # summer
        elif m in [9, 10, 11]:
            season[i] = 3  # autumn
        elif m in [12, 1, 2]:
            season[i] = 4  # winter
    return season


def assign_t_s(df, TS, coordinate, tp_val=0) -> pd.Series:
    """ Returns the bool series of t / s after applying appropriate comparison for a chosen vcoord.

    Parameters:
        df (DataFrame): reference data - e.g. track pressure / TP p distance to track p
        TS (str): 't' / 's';  indicates troposphere / stratosphere
        coordinate (str): dp, pt, z

        tp_val (float): value of tropopause in chosen coordinates. For non-relative coords
    """
    if ((coordinate in ['p', 'dp'] and TS == 't')
            or (coordinate in ['pt', 'z'] and TS == 's')):
        return df.gt(tp_val)

    elif ((coordinate in ['p', 'dp'] and TS == 's')
          or (coordinate in ['pt', 'z'] and TS == 't')):
        return df.lt(tp_val)

    else:
        raise KeyError(f'Strat/Trop assignment undefined for {coordinate}')


def get_lin_fit(series, degree=2) -> np.array:  # previously get_mlo_fit
    """ Given one year of reference data, find the fit parameters for
    the substance (col name) """
    year, month = series.index.year, series.index.month
    t_ref = year + (month - 0.5) / 12  # obtain frac year for middle of the month
    mxr_ref = series.values
    fit = np.poly1d(np.polyfit(t_ref, mxr_ref, degree))
    print(f'Fit parameters obtained: {fit}')
    return fit


def pre_flag(data_arr, ref_arr, crit='n2o', limit=0.97, **kwargs) -> pd.DataFrame:
    """ Sort data into strato / tropo based on difference to ground obs.

    Parameters:
        data_arr (pd.Series) : msmt data to be sorted into stratr / trop air
        ref_arr (pd.Series) : reference data to use for filtering (background)
        crit (str) : substance to use for flagging
        limit (float) : tracer mxr fraction below which air is classified
                        as stratospheric

    Returns: dataframe containing index and strato/tropo/pre_flag columns
    """
    data_arr.sort_index(inplace=True)
    df_flag = pd.DataFrame({f'strato_{data_arr.name}': np.nan,
                            f'tropo_{data_arr.name}': np.nan},
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
    factor = {'ppm': 1e6,  # per million
              'ppb': 1e9,  # per billion
              'ppt': 1e12,  # per trillion
              'ppq': 1e15,  # per quadrillion
              }
    # e.g. n2o: 300 ppb, 3e-7 mol/mol
    return x * factor[unit]


def conv_PartsPer_molarity(x, unit):
    """ Convert x from [unit] to molarity (mol/mol) """
    factor = {'ppm': 1e-6,  # per million
              'ppb': 1e-9,  # per billion
              'ppt': 1e-12,  # per trillion
              'ppq': 1e-15,  # per quadrillion
              }
    return x * factor[unit]


# %% Plotting tools
def add_zero_line(ax, axis='y'):
    """ Highlight the gridline at 0 for the chosen axis on the given Axes object. 

    Call when everything else has been plotted already, otherwise the limits will be messy. 
    """
    zero_lines = np.delete(ax.get_ygridlines(), ax.get_yticks() != 0)
    for l in zero_lines:
        l.set_color('k')
        l.set_linestyle('-.')

    if len(zero_lines) == 0:
        xlims = ax.get_xlim()
        ax.hlines(0, *xlims)
        ax.set_xlim(*xlims)

# %% Binning of global data sets

def bin_1d(glob_obj, subs, **kwargs) -> (list, list):
    """ 
    Returns 1D binned objects for each year as lists (lat / lon)

    Parameters:
        subs (dictionaries.Substance) 

    Optional parameters:
        c_pfx (str): caribic file pfx, required for caribic data
        single_yr (int): if specified, use only data for that specific year

    Returns:
        out_x_list, out_y_list: lists of Bin1D objects binned along x / y
    """
    substance = subs.col_name
    if kwargs.get('detr') and 'detr_' + substance in glob_obj.df.columns:
        substance = 'detr_' + substance

    out_lat_list, out_lon_list = [], []
    for yr in glob_obj.years:  # loop through available years
        df_yr = glob_obj.df[glob_obj.df.index.year == yr]

        lat = np.array([df_yr.geometry[i].y for i in range(len(df_yr.index))])  # lat
        if kwargs.get('lat_binlimits'):
            lat_binclassinstance = bp.Bin_notequi1d(kwargs.get('lat_binlimits'))
        else:
            lat_bmin, lat_bmax = np.nanmin(lat), np.nanmax(lat)
            lat_binclassinstance = bp.Bin_equi1d(lat_bmin, lat_bmax, glob_obj.grid_size)
        out_lat = bp.Simple_bin_1d(df_yr[substance], lat, lat_binclassinstance)
        out_lat.__dict__.update(lat_binclassinstance.__dict__)

        lon = np.array([df_yr.geometry[i].x for i in range(len(df_yr.index))])  # lon
        if kwargs.get('lon_binlimits'):
            lon_binclassinstance = bp.Bin_notequi1d(kwargs.get('lon_binlimits'))
        else:
            lon_bmin, lon_bmax = np.nanmin(lon), np.nanmax(lon)
            lon_binclassinstance = bp.Bin_equi1d(lon_bmin, lon_bmax, glob_obj.grid_size)
        out_lon = bp.Simple_bin_1d(df_yr[substance], lon, lon_binclassinstance)
        out_lon.__dict__.update(lon_binclassinstance.__dict__)

        if not all(np.isnan(out_lat.vmean)) or all(np.isnan(out_lon.vmean)):
            out_lat_list.append(out_lat)
            out_lon_list.append(out_lon)
        else:
            print(f'everything is nan for {yr}')

    return out_lat_list, out_lon_list


def bin_2d(glob_obj, subs, **kwargs) -> list:
    """
    Returns 2D binned object for each year as a list

    Parameters:
        substance (str): if None, uses default substance for the object
        single_yr (int): if specified, uses only data for that year
    """
    substance = subs.col_name
    if kwargs.get('detr'):
        if 'detr_' + substance in glob_obj.df.columns:
            substance = 'detr_' + substance
        else:
            print(f'detr_{substance} not found in {glob_obj.source} dataframe.')

    out_list = []
    for yr in glob_obj.years:  # loop through available years if possible
        df_yr = glob_obj.df[glob_obj.df.index.year == yr]

        lat = np.array([df_yr.geometry[i].y for i in range(len(df_yr.index))])  # lat
        lat_binlimits = kwargs.get('lat_binlimits')
        if lat_binlimits is None:
            # use equidistant binning if not specified else
            lat_bmin, lat_bmax = np.nanmin(lat), np.nanmax(lat)
            lat_binlimits = list(bp.Bin_equi1d(lat_bmin, lat_bmax, glob_obj.grid_size).xbinlimits)

        lon = np.array([df_yr.geometry[i].x for i in range(len(df_yr.index))])  # lon
        lon_binlimits = kwargs.get('lon_binlimits')
        if lon_binlimits is None:
            lon_bmin, lon_bmax = np.nanmin(lon), np.nanmax(lon)
            lon_binlimits = list(bp.Bin_equi1d(lon_bmin, lon_bmax, glob_obj.grid_size).xbinlimits)

        # create binclassinstance that's valid for both equi and nonequi
        binclassinstance = bp.Bin_notequi2d(lat_binlimits, lon_binlimits)
        out = bp.Simple_bin_2d(np.array(df_yr[substance]), lat, lon, binclassinstance)
        out.__dict__.update(binclassinstance.__dict__)

        out_list.append(out)
    return out_list
