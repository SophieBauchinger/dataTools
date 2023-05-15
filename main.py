# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 15:56:59 2023

@author: sophie_bauchinger
"""
import numpy as np
import pandas as pd

from toolpac.outliers.outliers import get_no_nan

from data_classes import Caribic, Mozart, Mauna_Loa, Mace_Head
from time_lag import calc_time_lags, plot_time_lags

from gradients import plot_gradient_by_season
from aux_fctns import get_lin_fit
from filter_outliers import pre_flag, filter_strat_trop, filter_trop_outliers
from dictionaries import get_fct_substance, get_col_name
from detrend import detrend_substance
from plot import plot_scatter_global, plot_global_binned_1d, plot_global_binned_2d, plot_1d_LonLat, plot_local

#%% Get data
year_range = range(1980, 2021)

mlo_sf6 = Mauna_Loa(year_range)
mlo_n2o = Mauna_Loa(year_range, substance='n2o')

caribic = Caribic(year_range, pfxs = ['GHG', 'INT', 'INT2']) # 2005-2020

mhd = Mace_Head() # only 2012 data available

mzt = Mozart(year_range) # only available up to 2008

#%% Plot data
plot_scatter_global(caribic, subs='sf6')
plot_global_binned_1d(caribic, subs='sf6', c_pfx='GHG')
plot_global_binned_2d(caribic, subs='sf6', c_pfx='GHG')

plot_scatter_global(mzt, subs='sf6')
plot_global_binned_1d(mzt, 'sf6')
plot_global_binned_2d(mzt, 'sf6')
plot_1d_LonLat(mzt, 'sf6')

plot_local(mlo_sf6, 'sf6')
plot_local(mlo_n2o, 'n2o')
plot_local(mhd, 'sf6')

#%% Time lags
# Get and prep reference data 
ref_min, ref_max = 2003, 2020
mlo_MM = Mauna_Loa(range(ref_min, ref_max)).df
mlo_MM.resample('1M') # add rows for missing months, filled with NaN 
mlo_MM.interpolate(inplace=True) # linearly interpolate missing data

# loop through years of caribic data
for c_year in range(2005, 2022):
    c_data = caribic.data['GHG'].select_year(c_year)
    if len(c_data[c_data['SF6 [ppt]'].notna()]) < 1: 
        continue
    else:
        lags = calc_time_lags(c_data, mlo_MM)
        if all(np.isnan(np.array(lags))): 
            print(f'no lags calculated for {c_year}'); continue
        plot_time_lags(c_data, lags, ref_min, ref_max)

#%% Get stratosphere / troposphere flags based on n2o mixing ratio
# loop through years of caribic data
data_filtered = pd.DataFrame() # initialise full dataframe
for c_year in range(2005, 2022): 
    print(f'{c_year}')
    c_data = caribic.data['GHG'].select_year(c_year)
    # print('cols:', c_data.columns)

    crit = 'n2o'; n2o_filtered = pd.DataFrame()
    if len(get_no_nan(c_data.index, c_data['N2O [ppb]'], c_data['d_N2O [ppb]'])[0]) < 1: # check for valid data
        print('! no n2o data')
    else:
        n2o_filtered =  filter_strat_trop(c_data, crit)
        data_filtered = pd.concat([data_filtered, n2o_filtered])

    crit = 'sf6'; sf6_filtered = pd.DataFrame()
    if crit=='sf6' and len(get_no_nan(c_data.index, c_data['SF6 [ppt]'], c_data['d_SF6 [ppt]'])[0]) < 1: # check for valid data
            print('! no sf6 data')
    else: 
        sf6_filtered =  filter_strat_trop(c_data, crit)
        data_filtered = pd.concat([data_filtered, sf6_filtered])

data_stratosphere = data_filtered.loc[data_filtered['strato'] == True]
# print(data_stratosphere.value_counts)
data_troposphere = data_filtered.loc[data_filtered['tropo'] == True]

data_trop_outlier = filter_trop_outliers(data_filtered, ['n2o'], source='Caribic')


#%% Detrend
# for now only have mlo data for n2o and sf6, so can only detrend those 
sf6_detr = detrend_substance(caribic, 'sf6', mlo_sf6.df)
n2o_detr = detrend_substance(caribic, 'n2o', mlo_n2o.df)

#%% Plot gradients 
plot_gradient_by_season(caribic, 'SF6 [ppt]')
