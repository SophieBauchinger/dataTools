# -*- coding: utf-8 -*-
"""
Created on Tue May  9 09:11:52 2023

@author: sophie_bauchinger
""" 
import datetime as dt
import geopandas
import numpy as np
from os.path import exists
import pandas as pd
from shapely.geometry import Point
from calendar import monthrange
import xarray as xr

import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable as sm
from matplotlib.colors import ListedColormap as lcm

from toolpac.calc import bin_1d_2d
from toolpac.readwrite import find
from toolpac.readwrite.FFI1001_reader import FFI1001DataReader
from toolpac.conv.times import fractionalyear_to_datetime

from aux_fctns import monthly_mean, daily_mean, ds_to_gdf, get_col_name, get_vlims, get_default_unit, same_col_merge, rename_columns

# supress a gui backend userwarning, not really advisible
import warnings; warnings.filterwarnings("ignore", category=UserWarning, module='matplotlib')

gdf = geopandas.GeoDataFrame() # initialise GeoDataFrame
years = range(2005, 2021)
verbose = False
c_pfxs = ['GHG']

df_tot = pd.DataFrame()

parent_dir = r'E:\CARIBIC\Caribic2data'
for yr in years:
    if not any(find.find_dir("*_{}*".format(yr), parent_dir)):
        years = np.delete(years, np.where(years==yr)) # removes current year if there's no data
        if verbose: print(f'No data found for {yr} in Caribic. Removing {yr} from list of years')
        continue
    df = pd.DataFrame()
    print(f'Reading in Caribic data for {yr}')

    # First collect data from individual flights
    for current_dir in find.find_dir("Flight*_{}*".format(yr), parent_dir)[1:]:
        flight_nr = int(str(current_dir)[-12:-9])
        for pfx in c_pfxs: # can include different prefixes here too
            f = find.find_file(f'{pfx}_*', current_dir)
            if not f or len(f)==0: # show error msg and go directly to next loop
                if verbose: print(f'No {pfx} File found for Flight {flight_nr} in {yr}')
                continue
            elif len(f) > 1: f.sort() # sort list of files, then take latest

            f_data = FFI1001DataReader(f[0], df=True, xtype = 'secofday')
            df_temp = f_data.df # index = Datetime
            df_temp.insert(0, 'Flight number',
                           [flight_nr for i in range(df_temp.shape[0])])
            if len(c_pfxs)>1: df_temp.insert(1, 'Prefix', [pfx for i in range(df_temp.shape[0])]) # add pfx column if more than one prefix given 
            df = pd.concat([df, df_temp])
    df_tot = pd.concat([df_tot, df])

#%% 

diff = [[fn, day_index - day_col] for fn, day_index, day_col in zip(df_tot['Flight number'], df_tot.index.day, df_tot['day; date of sampling: day; [dd]\n'])]

wrong_day = []
for i in range(0, len(diff)):
    if diff[i][1] != 0.0: 
        wrong_day.append(diff[i][0])

output = []
for x in wrong_day:
    if x not in output:
        output.append(x)
print(output)

df_wrong = pd.DataFrame()
for fn in output:
     df_wrong = pd.concat([df_wrong, df_tot[df_tot["Flight number"] == fn]])
     
