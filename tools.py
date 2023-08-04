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
import copy

import toolpac.calc.binprocessor as bp

from dictionaries import coord_dict, get_col_name, get_v_coord, get_h_coord, substance_list

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

def EMAC_vars_filter(x):
    """ 
        dt - delta_time [s]
        nstep - current time step
        tlon - track longitude [degrees_east]
        tlat - track latitude [degrees_north]
        tpress - track pressure [hPa]
        tps - track surface pressure [Pa]
    """
    vars_to_keep = ['dt', 'nstep', 'tlon', 'tlat', 'tpress', 'tps', 'time', 
                    'YYYYMMDD', 'lev', 'hyam', 'hybm', 'ilev', 'hyai', 'hybi']
    vars_to_drop = [v for v in x.variables if not (v.startswith('tropop_') or
                                                   # v.startswith('ECHAM5_') or
                                                   v in vars_to_keep)]
    return x.drop_vars(vars_to_drop)

#%% Data selection
def data_selection(c_obj, flights=None, years=None, latitudes=None, 
                   tropo=False, strato=False, extr_events=False, **kwargs):
    """ Return new Caribic instance with selection of parameters 
        flights (int / list(int))
        years (int / list(int))
        latitudes (tuple): lat_min, lat_max
        tropo, strato, extr_events (bool)
        kwargs: e.g. tp_def, ... - for strat / trop filtering 
    """
    out = copy.deepcopy(c_obj)
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

def assign_t_s(df, TS, coordinate, tp_val=0):
    """ Returns the appropriate comparison for a chosen coord to sort into trop / strat
    Parameters: 
        df (DataFrame): reference data 
        TS (str): 't' / 's';  troposphere / stratosphere
        coord (str): dp, pt, z
        tp_val (float): value of tropopause in chosen coordinates
    """
    if ((coordinate in ['dp'] and TS == 't') 
        or (coordinate in ['pt', 'z'] and TS == 's')):
        return df.gt(tp_val)

    elif ((coordinate in ['dp'] and TS == 's') 
          or (coordinate in ['pt', 'z'] and TS == 't')):
        return df.lt(tp_val)

    else: raise KeyError(f'STrat/Trop assignment undefined for {coordinate}')

def coordinate_tools(tp_def, y_pfx, ycoord, pvu=3.5, xcoord=None, x_pfx='INT2'):
    """ Get appropriate coordinates and labels. """

    y_coord = get_v_coord(y_pfx, ycoord, tp_def, pvu)

    pv = '%s' % (f', {pvu}' if tp_def=='dyn' else '')
    model = '%s' % (' - ECMWF' if (y_pfx=='INT' and tp_def in ['dyn', 'therm']) else '')
    model += '%s' % (' - ERA5' if (y_pfx=='INT2' and tp_def in ['dyn', 'therm']) else '')
    y_labels = {'z' : f'$\Delta$z ({tp_def+pv+model}) [km]',
                'pt' : f'$\Delta\Theta$ ({tp_def+pv+model}) [K]',
                'dp' : f'$\Delta$p ({tp_def+pv+model}) [hPa]'}
    y_label = y_labels[ycoord]

    # x coordinate = equivalent latitude
    if xcoord is not None: 
        x_coord = get_h_coord(x_pfx, xcoord)
        x_label = 'Eq. latitude (%s) ' % ('ECMWF' if x_pfx=='INT' else 'ERA5') + '[°N]'
        return y_coord, y_label, x_coord, x_label

    return y_coord, y_label

#%% Caribic combine GHG measurements with INT and INT2 coordinates
def coord_combo(c_obj, save=True):
    """ Create dataframe with all possible coordinates but
    no measurement / substance values """
    # merge lists of coordinates for all pfxs in the object
    coords = list(set([i for i in [coord_dict()[pfx] for pfx in c_obj.pfxs]
             for i in i])) + ['geometry', 'Flight number']
    if 'GHG' in c_obj.pfxs: df = c_obj.data['GHG'].copy() # copy bc don't want to overwrite data
    else: df = pd.DataFrame()
    for pfx in [pfx for pfx in c_obj.pfxs if pfx!='GHG']:
        df = df.combine_first(c_obj.data[pfx].copy())
    df.drop([col for col in df.columns if col not in coords],
            axis=1, inplace=True) # remove non-met / non-coord data

    if 'INT2' in c_obj.pfxs: # create thermal TP relative coordinates
        df['int_dp_strop_hpa_ERA5 [hPa]'] = df['int_ERA5_PRESS [hPa]'] - df['int_ERA5_TROP1_PRESS [hPa]']
        df['int_pt_rel_sTP_K_ERA5 [K]'] = df['int_Theta [K]'] - df['int_ERA5_TROP1_THETA [K]']

    # reorder columns
    df = df[list(['Flight number', 'p [mbar]'] 
                      + [col for col in df.columns 
                         if col not in ['Flight number', 'p [mbar]', 'geometry']] 
                      + ['geometry'])]
    return df

def coord_merge_substance(c_obj, subs, save=True, detr=True):
    """ Insert msmt data into full coordinate df from coord_merge() """
    # create reference df if it doesn't exist
    if not 'met_data' in dir(c_obj): df = coord_combo(c_obj)
    else: df = c_obj.met_data.copy()

    subs_cols = {}
    for pfx in c_obj.pfxs:
        if subs in substance_list(pfx):
            print(subs, pfx)
            subs_cols.update({pfx : get_col_name(subs, 'Caribic', pfx)})
    
    # subs_cols = {pfx:get_col_name(subs, 'Caribic', pfx) for pfx in c_obj.pfxs
    #              if get_col_name(subs, 'Caribic', pfx) is not None}

    if detr: 
        try: c_obj.detrend(subs) # add detrended data to all dataframes
        except: print(f'Detrending unsuccessful for {subs.upper()}, proceeding without. ')

    for pfx, substance in subs_cols.items():
        data = c_obj.data[pfx].sort_index()
        # print(data['geometry'])
        cols = [col for col in data if substance in col] # include detrended etc
        # print(pfx, data[cols].columns)
        df = df.join(data[cols])

        # Reorder columns to match initial dataframes & put substance to front
        df = df[list(['Flight number', 'p [mbar]'] + cols 
                          + [c for c in df.columns if c not in 
                              list(['Flight number', 'p [mbar]', 'geometry']+cols)]
                          + ['geometry'])]
    df.dropna(subset = subs_cols.values(), how='all', inplace=True) # drop rows without any subs data
    return df

#%% Binning of global data sets
def bin_prep(glob_obj, subs, **kwargs):
    c_pfx = kwargs.get('c_pfx') # only for caribic data; otherwise None
    substance = get_col_name(subs, glob_obj.source, c_pfx)

    if kwargs.get('single_yr') is not None: 
        years = [int(kwargs.get('single_yr'))]
    else: years = glob_obj.years

    # for Caribic, need to choose the df
    if glob_obj.source == 'Caribic': df = glob_obj.data[c_pfx]
    else: df = glob_obj.df

    if kwargs.get('detr') is True: 
        if not 'detr_'+substance in df.columns:
            glob_obj.detrend(subs)
        substance = 'detr_' + substance

    print(substance)

    return substance, years, df

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
    substance, years, df = bin_prep(glob_obj, subs, **kwargs)

    if kwargs.get('xbinlimits') is not None: # not equidistant binning 
        x_binclassinstance = bp.Bin_notequi1d(kwargs.get('xbinlimits'))
    if kwargs.get('ybinlimits') is not None:
        y_binclassinstance = bp.Bin_notequi1d(kwargs.get('ybinlimits'))

    out_x_list, out_y_list = [], []
    for yr in years: # loop through available years
        df_yr = df[df.index.year == yr]

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
    substance, years, df = bin_prep(glob_obj, subs, **kwargs)

    out_list = []
    for yr in years: # loop through available years if possible
        df_yr = df[df.index.year == yr]

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