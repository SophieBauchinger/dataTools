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
    df: Pandas DataFrame with datetime index
    first_of_month: bool, if True sets monthly mean timestamp to first of that month

    Returns dataframe with monthly averages of all values
    """
    # group by month then calculate mean
    df_MM = df.groupby(pd.PeriodIndex(df.index, freq="M")).mean(numeric_only=True)

    if first_of_month: # reset index to first of month
        df_MM['Date_Time'] = [dt.datetime(y, m, 1) for y, m in zip(df_MM.index.year, df_MM.index.month)]
        df_MM.set_index('Date_Time', inplace=True)
    return df_MM

def ds_to_gdf(ds):
    """ Convert xarray Dataset to GeoPandas GeoDataFrame """ 
    df = ds.to_dataframe()
    geodata = [Point(lat, lon) for lon, lat in zip(
        df.index.to_frame()['longitude'], df.index.to_frame()['latitude'])]

    # create geodataframe using lat and lon data from indices
    gdf = geopandas.GeoDataFrame(
        df.reset_index().drop(['longitude', 'latitude', 'scalar', 'P0'], axis=1),
        geometry=geodata)
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
    """ Returns column name for substance as saved in dataframe 
        source: str, can be 'car', 'mlo', 'mhd', 'moz' """
    cname=None
    if source=='car': # caribic / ames
        col_names = {
            'sf6': 'SF6 [ppt]',
            'n2o': 'N2O [ppb]',
            'co2': 'CO2 [ppm]',
            'ch4': 'CH4 [ppb]'}

    elif source=='mlo': # mauna loa
        col_names = {
            'sf6': 'SF6catsMLOm',
            'n2o': 'N2OcatsMLOm',
            'co2': 'CO2catsMLOm',
            'ch4': 'CH4catsMLOm'}

    elif source=='mhd': # mace head
        col_names={'sf6': 'SF6[ppt]'}
        
    elif source=='moz': # mozart
        col_names = {'sf6': 'SF6'}
    try: cname = col_names[substance.lower()]
    except: print('Corresponding column name not found')
    return cname

def get_lin_fit(df, substance='N2OcatsMLOm', degree=2): # previously get_mlo_fit
    """ Given one year of reference data, find the fit parameters for the substance (col name) """
    df.dropna(how='any', subset=substance, inplace=True)
    year, month = df.index.year, df.index.month
    t_ref = year + (month - 0.5) / 12 # obtain fractional year for middle of the month
    mxr_ref = df[substance].values
    fit = np.poly1d(np.polyfit(t_ref, mxr_ref, degree))
    print(f'Fit parameters obtained: {fit}')
    return fit
