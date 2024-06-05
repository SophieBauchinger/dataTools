# -*- coding: utf-8 -*-

""" Auxiliary functions for data extraction and handling.

@Author: Sophie Bauchinger, IAU
@Date: Fri Apr 28 09:51:49 2023

Functions: 
    # --- Data extraction --- 
    time_mean(df, f, first_of_month, minmax)
    ds_to_gdf(ds)
    rename_columns(columns)
    
    # --- EMAC data handling --- 
    process_emac_s4d(ds, incl_model, incl_tropop, incl_subs)
    process_emac_s4d_s(ds, incl_model, incl_tropop, incl_subs)
    
    # --- TPChange ERA5 / CLaMS reanalysis ---
    ERA5_variables()
    process_TPC_V02(ds)
    process_TPC(ds)
    interpolate_onto_timestamps(dataframe, times, prefix)

    # --- Data selection ---
    minimise_tps(tps, vcoord)
    
    # --- Data handling ---
    make_season(month)
    assign_t_s(df, TS, coordinate, tp_val)
    get_lin_fit(series, degree)
    pre_flag(data_arr, ref_arr, crit, limit, **kwargs)
    conv_molarity_PartsPer(x, unit)
    conv_PartsPer_molarity(x, unit)
    
    # --- Plotting helpers ---
    add_zero_line(ax, axis)
    
    # --- Binning of geodataframes ---
    bin_1d(glob_obj, subs, **kwargs)
    bin_2d(glob_obj, subs, **kwargs)
    
    # --- Create animations ---
    make_gif(pdir, fnames)
      
"""

import datetime as dt
import dill
import geopandas
import glob
from metpy.units import units
import numpy as np
import pandas as pd
from PIL import Image
import os
from shapely.geometry import Point
from scipy.ndimage import zoom, gaussian_filter

import toolpac.calc.binprocessor as bp # type: ignore
from toolpac.conv.times import datetime_to_fractionalyear as dt_to_fy # type: ignore
from toolpac.conv.times import secofday_to_datetime, datetime_to_secofday # type: ignore

import dataTools.dictionaries as dcts

# %% 
def get_path():
    return dcts.get_path()

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
    df_mean.reset_index(inplace=True)
    
    if f == 'D': 
        df_mean['Date_Time'] = df_mean['Date_Time'].apply(lambda x: dt.datetime(x.year, x.month, x.day))
    elif f == 'M' and first_of_month: 
        df_mean['Date_Time'] = df_mean['Date_Time'].apply(lambda x: dt.datetime(x.year, x.month, 1))
    else: 
        df_mean['Date_Time'] = df_mean['Date_Time'].apply(lambda x: dt.datetime(x.year, x.month, 15))
        
    df_mean.set_index('Date_Time', inplace=True)
    
    # df_mean['Date_Time', i] = dt.datetime(y, m, d)
    
    # df_mean['Date_Time'] = np.nan
    # for i, (y, m, d) in enumerate(zip(
    #         df_mean.index.year,
    #         df_mean.index.month if f != 'Y' else None,
    #         df_mean.index.day if f == 'D' else (
    #             [1] * len(df_mean.index) if first_of_month else None))):
    #     df_mean['Date_Time', i] = dt.datetime(y, m, d)

    # df_mean.set_index('Date_Time', inplace=True)
    
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

def rename_columns(columns) -> dict:
    """ Create dictionary relating column name with AMES_variable object

    Relate dataframe column name with all information in

    Get new column names and col_name_dict for AMES data structure.
    Get only short name + unit; description found in coordinate instance for specific col_name
    Standardise names via case changes
    """
    col_name_dict = {}
    for x in columns:
        if len(x.split(';')) == 3:
            col_name, long_name, unit = [i.strip() for i in x.split(';')]
        else:
            col_name = x.split(";")[0].strip()
        col_name_dict.update({x: col_name})
    return col_name_dict

# EMAC data handling
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
        tracers = [s.col_name for s in dcts.get_substances(**{'ID': 'EMAC'})]
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
    """ Keep only variables that depend only on time and are available in subsampled data """
    ds = process_emac_s4d(ds, incl_model, incl_tropop, incl_subs)
    variables = [v for v in ds.variables if ds[v].dims == ('time',)]
    return ds[variables]

# TPChange ERA5 / CLaMS reanalysis interpolated onto flight tracks
def ERA5_variables(): 
    """ All variables for TPChange ERA5 reanalysis datasets. """
    met_vars = [
        'ERA5_PV',
        'ERA5_EQLAT',
        'ERA5_TEMP',
        'ERA5_PRESS',
        'ERA5_THETA',
        'ERA5_PHI', 
        'ERA5_GPH',
        ]

    dyn_tps = [f'ERA5_dynTP_{vcoord}_{pvu}_Main' 
               for pvu in ['1_5', '2_0', '3_5'] 
               for vcoord in ['PHI', 'THETA', 'PRESS', 'GPH']]

    therm_tps = [f'ERA5_thermTP_{vcoord}_Main' 
               for vcoord in ['Z', 'THETA', 'PRESS']]
    
    therm_tps_V02 = [f'ERA5_TROP1_{vcoord}' 
            for vcoord in ['Z', 'THETA', 'PRESS']]
    
    other_vars = ['ERA5_O3']
    
    return set(met_vars + dyn_tps + therm_tps + therm_tps_V02 + other_vars)

def process_TPC_V02(ds): # up to V02
    """ Preprocess datasets for ERA5 / CLaMS renalayis data up to version 2. """
    return ds[[v for v in ERA5_variables() if v in ds.variables]]

def process_TPC(ds): # from V04
    """ Preprocess datasets for ERA5 / CLaMS renalayis data from version .04 onwards. 
    
    NB CARIBIC: drop_variables = ['CARIBIC2_LocalTime']
    NB ATom:    drop_variables = ['ATom_UTC_Start', 'ATom_UTC_Stop', 'ATom_End_LAS']

    """
    def flatten_TPdims(ds):
        """ Deals with Tropopause variables having additional dimensions indicating Main / Second / ... 
        Used for ERA5 / CLaMS reanalysis datasets from version .03
        """
        TP_vars = [v for v in ds.variables if any(d.endswith('TP') for d in ds[v].dims)]
        TP_qualifier_dict = {0 : '_Main', 
                            1 : '_Second', 
                            2 : '_Third'}
        
        for variable in TP_vars: 
            # get secondary dimension for the current multi-dimensional variable
            [TP_dim] = [d for d in ds[variable].dims if d.endswith('TP')] # should only be a single one!
            
            for TP_value in ds[variable][TP_dim].values: 
                ds[variable + TP_qualifier_dict[TP_value]] = ds[variable].isel({TP_dim : TP_value})
            
            ds = ds.drop_vars(variable)
        
        return ds
    
    # Flatten variables that have multiple tropoause dimensions (thermTP, dynTP)
    ds = flatten_TPdims(ds)
    return ds[[v for v in ERA5_variables() if v in ds.variables]]

# Interpolation
def interpolate_onto_timestamps(dataframe, times, prefix='') -> pd.DataFrame:
    """ Interpolate met data onto given measurement timestamps. 
    
    Parameters: 
        dataframe (pd.DataFrame): data to be interpolated
        times (array, list): Timestamps to be used for interpolating onto
    """
    if isinstance(dataframe, geopandas.GeoDataFrame):
        dataframe = pd.DataFrame(dataframe[[c for c in dataframe.columns if c != 'geometry']])

    # add measurement timestamps to met_data
    new_indices = [i for i in times if i not in dataframe.index]

    expanded_df = pd.concat([dataframe, pd.Series(index=new_indices, dtype='object')])
    expanded_df.drop(columns=[0], inplace=True)
    expanded_df.sort_index(inplace=True)

    try:
        expanded_df.interpolate(method='time', inplace=True, limit=2)  # , limit=500)
    except TypeError:
        print(f'Check if type {type(dataframe)} is suitable for time-wise interpolation!')

    regridded_data = expanded_df.loc[times]  # return only measurement timestamps
    
    # Rename columns using prefix
    regridded_data.rename(columns = {col:prefix+col for col in regridded_data.columns}, 
                          inplace=True)
    
    return regridded_data

# %% Data selection
def minimise_tps(tps, vcoord=None) -> list:
    """ Returns a reduced list of tropopause coordinates.

    1. Remove tps with other vertical coords if vcoord is specified
    2. remove all cpt, combo tp
    3. remove all ECMWF tps
    4. Remove modelled N2O tp
    5. Remove duplicates of O3 tp
    6. Remove 1.5 PVU ERA5 dyn tp
    7. Remove non-relative tps if relative exists in tps
    """
    # 1. Remove all non-tropopause related coordinates
    while len([tp for tp in tps if str(tp.tp_def)=='nan']) > 0: # repeat until none left 
        print('hi')
        [tps.remove(tp) for tp in tps if str(tp.tp_def) == 'nan']
    
    # 2. Remove all coordinates with vcoords other than the specified one
    if vcoord: [tps.remove(tp) for tp in tps if tp.vcoord != vcoord]
    
    [tps.remove(tp) for tp in tps if (
        tp.model in ['EMAC', 'ECMWF'] or
        tp.tp_def in ['cpt', 'combo'] or
        tp.pvu == 1.5
        )]
    
    # check if coord exists with pt, remove if it does
    tp_to_remove = []
    for tp in tps:  # 1
        try:
            dcts.get_coord(vcoord='pt', model=tp.model, source=tp.source,
                           tp_def=tp.tp_def, ID=tp.ID, 
                           pvu=tp.pvu, crit=tp.crit, rel_to_tp=tp.rel_to_tp)
        except KeyError:
            continue
        except ValueError: 
            print('ValueError', dcts.get_coordinates(vcoord='pt', model=tp.model, source=tp.source,
                           tp_def=tp.tp_def,
                           pvu=tp.pvu, crit=tp.crit, rel_to_tp=tp.rel_to_tp)); continue
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
                                    tp_def=tp.tp_def, ID=tp.ID,
                                    pvu=tp.pvu, crit=tp.crit, vcoord=tp.vcoord)
        except KeyError:
            continue
        except ValueError: 
            print('ValueError2'); continue
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

def get_lin_fit(series, degree=2, verbose=False) -> np.array:  # previously get_mlo_fit
    """ Given one year of reference data, find the fit parameters for
    the substance (col name) """
    year, month = series.index.year, series.index.month
    t_ref = year + (month - 0.5) / 12  # obtain frac year for middle of the month
    mxr_ref = series.values
    fit = np.poly1d(np.polyfit(t_ref, mxr_ref, degree))
    if verbose: 
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
                           index=data_arr.index, dtype=object)

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

def add_world(axs, fname = \
    r'c:\Users\sophie_bauchinger\Documents\GitHub\110m_cultural_511\ne_110m_admin_0_map_units.shp'): 
    """ Adds country outlines to the given axis with zorder 0 as thin grey lines. """
    world = geopandas.read_file(fname)
    if isinstance(axs, list): 
        for ax in axs: 
            world.boundary.plot(ax=ax, color='grey', linewidth=0.3, zorder=0)
    else: 
        world.boundary.plot(ax=axs, color='grey', linewidth=0.3, zorder=0)

def nan_zoom(input, factor, order=1, mode='grid-constant', grid_mode=False, **kwargs):
    """ The array is zoomed using spline interpolation of the requested order and may contain NaN values.
    Args: 
        input (array_like): The input array
        zoom (float or sequence): The zoom factor along the axes. 
    Based on scipy.ndimage.zoom: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.zoom.html
    """
    array_copy = np.copy(input)
    nan_mask = np.isnan(array_copy)
    array_copy[nan_mask] = 0

    # Create a weight array with 0 at NaN positions and 1 elsewhere
    weights = np.ones_like(input)
    weights[nan_mask] = 0

    # Apply zoom to both arrays
    zoomed_array = zoom(array_copy, factor, order=order, mode=mode, grid_mode=grid_mode, **kwargs)
    zoomed_weights = zoom(weights, factor, order=order, mode=mode, grid_mode=grid_mode, **kwargs)

    # Restore NaN values
    if not all(i == 1 for i in zoomed_weights.flatten()):
        zoomed_array[zoomed_weights == 0] = np.nan

    return zoomed_array

def nan_gaussian_filter(input, sigma, **kwargs):
    """ Multidimensional Gaussian filter for arrays containing NaN values.
    
    Args: 
        input (array_like): The input array
        sigma (scalar / sequence of scalars): Standard deviation for Gaussian kernel. 
    
    Based on scipy.ndimage.gaussian_filter: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_filter.html
    """
    array_copy = np.copy(input)
    nan_mask = np.isnan(array_copy)
    array_copy[nan_mask] = 0

    # Create a weight array with 0 at NaN positions and 1 elsewhere
    weights = np.ones_like(input)
    weights[nan_mask] = 0

    # Apply Gaussian filter to both arrays
    filtered_array = gaussian_filter(array_copy, sigma=sigma, **kwargs)
    filtered_weights = gaussian_filter(weights, sigma=sigma, **kwargs)

    if all(weight == 1 for weight in weights.flatten()): 
        return filtered_array

    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        filtered_array /= filtered_weights
        filtered_array[filtered_weights == 0] = np.nan  # Restore NaNs where weights are 0

    return filtered_array

# %% Binning of global data sets
def bin_1d(glob_obj, subs, **kwargs) -> tuple[list, list]:
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

        lat = np.array([df_yr.geometry.iloc[i].y for i in range(len(df_yr.index))])  # lat
        if kwargs.get('lat_binlimits'):
            lat_binclassinstance = bp.Bin_notequi1d(kwargs.get('lat_binlimits'))
        else:
            lat_bmin, lat_bmax = np.nanmin(lat), np.nanmax(lat)
            lat_binclassinstance = bp.Bin_equi1d(lat_bmin, lat_bmax, glob_obj.grid_size)
        out_lat = bp.Simple_bin_1d(df_yr[substance], lat, lat_binclassinstance)
        out_lat.__dict__.update(lat_binclassinstance.__dict__)

        lon = np.array([df_yr.geometry.iloc[i].x for i in range(len(df_yr.index))])  # lon
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

        lat = np.array([df_yr.geometry.iloc[i].y for i in range(len(df_yr.index))])  # lat
        lat_binlimits = kwargs.get('lat_binlimits')
        
        if lat_binlimits is None:
            # use equidistant binning if not specified else
            lat_bmin, lat_bmax = np.nanmin(lat), np.nanmax(lat)
            lat_binlimits = list(bp.Bin_equi1d(lat_bmin, lat_bmax, glob_obj.grid_size).xbinlimits)

        lon = np.array([df_yr.geometry.iloc[i].x for i in range(len(df_yr.index))])  # lon
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

#%% Miscellaneous
def make_gif(pdir=None, fnames=None): # Animate changes over years
    if not pdir: 
        pdir = r'C:\Users\sophie_bauchinger\sophie_bauchinger\Figures\tp_scatter_2d'
    for vc in ['p', 'pt', 'z']:
        tps = dcts.get_coordinates(vcoord=vc, tp_def='not_nan', rel_to_tp=False)
        for tp in tps:
            # fn = pdir+.format(, '_'+str(year) if year else ''))
            frames = [Image.open(image) for image in glob.glob(f'{pdir}/{tp.col_name}*_*.png')]
            if len(frames)==0: frames = [Image.open(image) for image in glob.glob(f'{pdir}/{tp.col_name[:-1]}*_*.png')]

            # frames = [Image.open(image) for image in glob.glob(f"{pdir}/*.JPG")]
            frame_one = frames[0]
            frame_one.save(f'C:/Users/sophie_bauchinger/sophie_bauchinger/Figures/tp_scatter_2d_GIFs/{tp.col_name}.gif',
                           format="GIF", append_images=frames,
                           save_all=True, duration=200, loop=0)

def gif_from_images(images, output_path='test_output.gif', duration=500):
    """ Saves given images to the output path as a gif. """
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=0
    )

class InitialisationError(Exception): 
    """ Raised when initialisation of a class is not intended. """
    pass
