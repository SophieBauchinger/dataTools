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
    """ Get new column names and col_name_dict for AMES data structure 
    #!!! int_CARIBIC2_CO; CO; Carbon monoxide mixing ratio; [ppbv]; [ppbv]
    """
    # get val_name from start of the line, then unit from last part
    val_names = [x.split(";")[0] for x in columns if len(x.split(";")) >= 3] # columns with short; long; unit
    units = [x.split(";")[-1][:-1] for x in columns if len(x.split(";")) >= 3] # get last part [unit] with [-1], then remove \n with [:-1]

    # Need to make column names of the same substance the same even though some are upper / lower case
    for i, x in enumerate(val_names): 
        if x.startswith('d_'): val_names[i] = x[:-3] + x[-3:].upper() # changing the letter case, nothing else
        elif x.startswith('int'): continue # no change in case for int_ variable names
        elif not x=='p': val_names[i] = x.upper() # not making p uppercase, only subst names
    
    new_names = [v+u for v,u in zip(val_names, units)] # coloumn names with corrected case, in correct order
    dictionary = dict(zip([x for x in columns if len(x.split(";")) >= 3], new_names))

    return new_names, dictionary
    

def same_merge(x): return ','.join(x[x.notnull()].astype(str))
def same_col_merge(df):
    """ Merge all columns with the same name when given a dataframe """
    return df.groupby(level=0, axis=1).apply(lambda x: x.apply(same_merge, axis=1))

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

# #%% Dictionaries for finding fctnbs, col names, v lims, default unit
# def get_fct_substance(substance):
#     """ Returns appropriate fct from toolpac.outliers.ol_fit_functions to a substance """
#     df_func_dict = {'co2': fct.higher,
#                     'ch4': fct.higher,
#                     'n2o': fct.simple, 
#                     'sf6': fct.quadratic, 
#                     'trop_sf6_lag': fct.quadratic, 
#                     'sulfuryl_fluoride': fct.simple, 
#                     'hfc_125': fct.simple, 
#                     'hfc_134a': fct.simple, 
#                     'halon_1211': fct.simple, 
#                     'cfc_12': fct.simple, 
#                     'hcfc_22': fct.simple, 
#                     'int_co': fct.quadratic}
#     return df_func_dict[substance.lower()]

# def get_col_name(substance, source, c_pfx='GHG', CLaMS=True):
#     """ 
#     Returns column name for substance as saved in dataframe 
#         source (str) 'Caribic', 'Mauna_Loa', 'Mace_Head', 'Mozart' 
#         substance (str): sf6, n2o, co2, ch4
#     """
#     cname=None
#     if source=='Caribic' and c_pfx=='GHG': # caribic / ames
#         col_names = {
#             'sf6': 'SF6 [ppt]',
#             'n2o': 'N2O [ppb]',
#             'co2': 'CO2 [ppm]',
#             'ch4': 'CH4 [ppb]'}

#     elif source=='Caribic' and c_pfx=='INT2': # 
#         col_names = {
#             'noy': 'int_NOy [ppb]',
#             'no' : 'int_NO [ppb]',
#             'ch4': 'int_CLaMS_CH4 [ppb]',
#             'co' : 'int_CLaMS_CO [ppb]',
#             'co2': 'int_CLaMS_CO2 [ppm]',
#             'h2o': 'int_CLaMS_H2O [ppm]',
#             'n2o': 'int_CLaMS_N2O [ppb]',
#             'o3' : 'int_CLaMS_O3 [ppb]'
#             }
#     elif source=='Caribic' and c_pfx=='INT': # 
#         col_names = {
#             'co' : 'int_CO [ppb]',
#             'o3' : 'int_O3 [ppb]',
#             'h2o': 'int_H2O_gas [ppm]',
#             'no' : 'int_NO [ppb]',
#             'noy': 'int_NOy [ppb]',
#             'co2': 'int_CO2 [ppm]',
#             'ch4': 'int_CH4 [ppb]'
#             }

#     elif source=='Mauna_Loa': # mauna loa. monthly or daily median
#         col_names = {
#             'sf6': 'SF6catsMLOm',
#             'n2o': 'N2OcatsMLOm',
#             'co2': 'CO2catsMLOm',
#             'ch4': 'CH4catsMLOm'}

#     elif source=='Mace_Head': # mace head
#         col_names={'sf6': 'SF6 [ppt]',
#                    'ch2cl2': 'CH2Cl2 [ppt]'}
        
#     elif source=='Mozart': # mozart
#         col_names = {'sf6': 'SF6'}

#     try: cname = col_names[substance.lower()]
#     except: print(f'Column name not found for {substance} in {source}')
#     return cname

# def get_vlims(substance):
#     """ Get default limits for colormaps per substance """
#     v_limits = {
#         'sf6': (6,9),
#         'n2o': (0,10),
#         'co2': (0,10),
#         'ch4': (0,10)}
#     return v_limits[substance.lower()]

# def get_default_unit(substance):
#     unit = {
#         'sf6': 'ppt',
#         'n2o': 'ppb',
#         'co2': 'ppm',
#         'ch4': 'ppb'}
#     return unit[substance.lower()]
