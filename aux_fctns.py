# -*- coding: utf-8 -*-
"""
@Author: Sophie Bauchimger, IAU
@Date: Fri Apr 28 09:51:49 2023

Auxiliary functions:
    get_fct_substance(substance)
    get_col_name(substance, source)
    get_lin_fit(df, substance='N2OcatsMLOm', degree=2)
"""
import numpy as np
import datetime as dt
import pandas as pd
import geopandas
from shapely.geometry import Point

from toolpac.outliers import ol_fit_functions as fct

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

#%% Working with data
def get_fct_substance(substance):
    """ Returns appropriate fct from toolpac.outliers.ol_fit_functions to a substance """
    df_func_dict = {'co2': fct.higher,
                    'ch4': fct.higher,
                    'n2o': fct.simple, 
                    'sf6': fct.quadratic, 
                    'trop_sf6_lag': fct.quadratic, 
                    'sulfuryl_fluoride': fct.simple, 
                    'hfc_125': fct.simple, 
                    'hfc_134a': fct.simple, 
                    'halon_1211': fct.simple, 
                    'cfc_12': fct.simple, 
                    'hcfc_22': fct.simple, 
                    'int_co': fct.quadratic}
    return df_func_dict[substance.lower()]

def get_col_name(substance, source):
    """ 
    Returns column name for substance as saved in dataframe 
        source (str) 'Caribic', 'Mauna_Loa', 'Mace_Head', 'Mozart' 
        substance (str): sf6, n2o, co2, ch4
    """
    cname=None
    if source=='Caribic': # caribic / ames
        col_names = {
            'sf6': 'SF6 [ppt]',
            'n2o': 'N2O [ppb]',
            'co2': 'CO2 [ppm]',
            'ch4': 'CH4 [ppb]'}

    elif source=='Mauna_Loa': # mauna loa. monthly or daily median
        col_names = {
            'sf6': 'SF6catsMLOm',
            'n2o': 'N2OcatsMLOm',
            'co2': 'CO2catsMLOm',
            'ch4': 'CH4catsMLOm'}

    elif source=='Mace_Head': # mace head
        col_names={'sf6': 'SF6 [ppt]',
                   'ch2cl2': 'CH2Cl2 [ppt]'}
        
    elif source=='Mozart': # mozart
        col_names = {'sf6': 'SF6'}

    try: cname = col_names[substance.lower()]
    except: print(f'Column name not found for {substance} in {source}')
    return cname

def get_vlims(substance):
    """ Get default limits for colormaps per substance """
    v_limits = {
        'sf6': (6,9),
        'n2o': (0,10),
        'co2': (0,10),
        'ch4': (0,10)}
    return v_limits[substance.lower()]

def get_lin_fit(df, substance='N2OcatsMLOm', degree=2): # previously get_mlo_fit
    """ Given one year of reference data, find the fit parameters for the substance (col name) """
    df.dropna(how='any', subset=substance, inplace=True)
    year, month = df.index.year, df.index.month
    t_ref = year + (month - 0.5) / 12 # obtain fractional year for middle of the month
    mxr_ref = df[substance].values
    fit = np.poly1d(np.polyfit(t_ref, mxr_ref, degree))
    print(f'Fit parameters obtained: {fit}')
    return fit

def get_default_unit(substance):
    unit = {
        'sf6': 'ppt',
        'n2o': 'ppb',
        'co2': 'ppm',
        'ch4': 'ppb'}
    return unit[substance.lower()]
