# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 15:56:59 2023

@author: sophie_bauchinger
"""
import pandas as pd

from toolpac.outliers.outliers import get_no_nan

from data_classes import Caribic, Mozart, Mauna_Loa, Mace_Head
from time_lag import calc_time_lags
from dictionaries import substance_list

from gradients import plot_gradient_by_season
from filter_outliers import filter_strat_trop, filter_trop_outliers
from detrend import detrend_substance
from plot import plot_scatter_global, plot_global_binned_1d, plot_global_binned_2d, plot_1d_LonLat, plot_local

#%% Get data
year_range = range(1980, 2021)

mlo_sf6 = Mauna_Loa(year_range, data_Day=True)
mlo_n2o = Mauna_Loa(year_range, substance='n2o')
mlo_co2 = Mauna_Loa(year_range, substance='co2')
mlo_ch4 = Mauna_Loa(year_range, substance='ch4')
mlo_co  = Mauna_Loa(year_range, substance='co')

mlo_data = {'sf6' : mlo_sf6, 'n2o' : mlo_n2o, 
            'co2' : mlo_co2, 'ch4' : mlo_ch4, 
            'co' : mlo_co }

caribic = Caribic(year_range, pfxs = ['GHG', 'INT', 'INT2']) # 2005-2020
# available substance in caribic data:
# 'GHG':    ['ch4', 'co2', 'n2o', 'sf6']
# 'INT':    ['co', 'o3', 'h2o', 'no', 'noy', 'co2', 'ch4', 'f11', 'f12', 'n2o']
# 'INT2':   ['noy', 'no', 'ch4', 'co', 'co2', 'h2o', 'n2o', 'o3']

mhd = Mace_Head() # only 2012 data available

mzt = Mozart(year_range) # only available up to 2008

#%% Plot data
plot_scatter_global(caribic, subs='sf6')
plot_global_binned_1d(caribic, subs='sf6', c_pfx='GHG')
plot_global_binned_2d(caribic, subs='sf6', c_pfx='GHG')

mzt_yr = 2008
plot_scatter_global(mzt, subs='sf6', single_yr=mzt_yr)
plot_global_binned_1d(mzt, 'sf6', single_yr=mzt_yr)
plot_global_binned_2d(mzt, 'sf6', single_yr=mzt_yr)
plot_1d_LonLat(mzt, 'sf6', single_yr=mzt_yr)

for subs, mlo_obj in mlo_data.items():
    plot_local(mlo_obj, subs)

plot_local(mhd, 'sf6')

#%% Time lags - lags are added to dictionary caribic.lags as lags_pfx : dataframe
calc_time_lags(caribic, mlo_sf6, range(2005, 2020), substance = 'sf6', plot=False)
calc_time_lags(caribic, mlo_n2o, range(2005, 2020), substance = 'n2o', pfx='INT2', plot=False)

calc_time_lags(caribic, mlo_n2o, range(2005, 2020), substance = 'n2o', plot=False)
calc_time_lags(caribic, mlo_co2, range(2005, 2020), substance = 'co2', pfx='INT2', plot=False)

# to delete the newly created attributes use
# del caribic.__dict__['lags']

#%% Detrend
# for now only have mlo data for n2o and sf6, so can only detrend those 
sf6_detr = detrend_substance(caribic, 'sf6', mlo_sf6)
n2o_detr = detrend_substance(caribic, 'n2o', mlo_n2o)

#%% Plot gradients 
for pfx in caribic.pfxs:
    for subs in substance_list(pfx):
        plot_gradient_by_season(caribic, subs, pfx)

#%% Filter tropospheric / stratospheric data points based on tracer mixing ratio wrt background data
substances = ['co2', 'n2o', 'sf6', 'ch4']

ref_dfs = {'sf6' : mlo_sf6,
           'n2o' : mlo_n2o,
           'co2' : mlo_co2}

for subs in substances: 
    for pfx in caribic.pfxs:
        filter_strat_trop(caribic, mlo_co2, subs,  'GHG')
        filter_trop_outliers(caribic, subs, 'INT2', crit='n2o')
        

filter_strat_trop(caribic, mlo_co2, 'co2', 'GHG')
filter_strat_trop(caribic, mlo_n2o, 'n2o', 'INT2')
filter_strat_trop(caribic, mlo_co2, 'co2', 'INT2')

for subs in ['ch4', 'co2', 'co']:
    filter_trop_outliers(caribic, subs, 'INT2', crit='n2o')


# # loop through years of caribic data
# data_filtered = pd.DataFrame() # initialise new dataframe
# for c_year in range(2005, 2022): 
#     print(f'{c_year}')
#     c_data = caribic.data['GHG'][caribic.data['GHG'].index.year == c_year]

#     crit = 'n2o'
#     n2o_filtered = pd.DataFrame()
#     if len(get_no_nan(c_data.index, c_data['N2O [ppb]'], c_data['d_N2O [ppb]'])[0]) < 1: # check for valid data
#         print('! no n2o data')
#     else:
#         n2o_filtered =  filter_strat_trop(caribic, mlo_n2o, crit)
#         data_filtered = pd.concat([data_filtered, n2o_filtered])

#     crit = 'sf6'; sf6_filtered = pd.DataFrame()
#     if crit=='sf6' and len(get_no_nan(c_data.index, c_data['SF6 [ppt]'], c_data['d_SF6 [ppt]'])[0]) < 1: # check for valid data
#             print('! no sf6 data')
#     else: 
#         sf6_filtered =  filter_strat_trop(c_data, crit)
#         data_filtered = pd.concat([data_filtered, sf6_filtered])

# data_stratosphere = data_filtered.loc[data_filtered['strato'] == True]
# # print(data_stratosphere.value_counts)
# data_troposphere = data_filtered.loc[data_filtered['tropo'] == True]

# data_trop_outlier = filter_trop_outliers(data_filtered, ['n2o'], source='Caribic')



