# -*- coding: utf-8 -*-
"""
@Author: Sophie Bauchinger, IAU
@Date Thu Oct  5 11:32:36 2023

1. Caribic substance and coordinate names (incl dictionaries)

2. EMAC dataframes and Caribic (and TropopauseData) LAT and LON umdrehen

"""

# CARIBIC NETCDF IS CORRECT !!!
import dill
from data import Caribic, EMAC
from shapely.geometry import Point
import dictionaries as dcts

# caribic = Caribic()
# emac = EMAC()

if not 'og_data_dict' in locals():
    with open('misc_data/yx_caribic_data_dict.pkl', 'rb') as f:
        og_data_dict = dill.load(f)

"""
WRONG: 
    min(caribic.met_data.geometry.x)
    -33.87
    
    min(caribic.met_data.geometry.y)
    -123.17

    min(emac.df.geometry.x)
    -34.3650016784668
    
    min(emac.df.geometry.y)
    -123.76499938964844
    
"""

c_dfs = ['GHG', 'INT', 'INT2'] # , 'sf6', 'n2o', 'co2', 'ch4', 'met_data']
# df, df_sorted
e_dfs = ['df']

def remove_unit(df):
    for col in df.columns: 
        if '[' in col: 
            print(col.split())
            name = col.split()[0]
            df.rename(columns={col : name}, inplace=True)
    return df

def create_new_dict(df):
    dictionary = {}
    
    subses = [s.col_name for s in dcts.get_substances()]
    coords = [c.col_name for c in dcts.get_coordinates()]
    
    for col in df.columns:
        if col in subses: dictionary.update({col : dcts.get_subs(col_name=col)})
        elif col in coords: dictionary.update({col : dcts.get_coord(col_name=col)})
        else: print(f'rogue: {col}')
    
    return dictionary

def change_caribic(caribic):
    for k in caribic.data: 
        try: 
            remove_unit(caribic.data[k])
            swap_x_y(caribic.data[k])
        except: caribic.data[k] = create_new_dict(caribic.data[k[:-5]])
    return caribic
        


def swap_x_y(gdf):
    gdf.sort_index()
    longitudes = gdf.geometry.y
    latitudes = gdf.geometry.x
    
    geodata = [Point(lon,lat) for lon,lat in zip(longitudes, latitudes)]
    
    gdf.geometry = geodata
    
    return gdf