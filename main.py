# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 15:56:59 2023

@author: sophie_bauchinger
"""
import numpy as np
import pandas as pd

from toolpac.outliers.outliers import get_no_nan

from local_data import Mauna_Loa, Mace_Head
from global_data import Caribic, Mozart
from time_lag import calc_time_lags, plot_time_lags

from strat_filter_on_caribic_data import get_fct_substance, get_lin_fit, pre_flag, filter_strat_trop, filter_trop_outliers, detrend_substance, plot_gradient_by_season

#%% Get data
mlo_sf6 = Mauna_Loa(range(2008, 2020))
mlo_n2o = Mauna_Loa(range(2008, 2020), substance='n2o')

mlo_sf6_df = mlo_sf6.df
mlo_n2o_df = mlo_n2o.df

caribic = Caribic(range(2005, 2020))
c_df = caribic.df

#%% Plot data
mlo_sf6.plot()
mlo_n2o.plot()

caribic.plot_scatter()
caribic.plot_1d()
caribic.plot_2d()

#%% Time lags
# Get and prep reference data 
ref_min, ref_max = 2003, 2020
mlo_MM = Mauna_Loa(range(ref_min, ref_max)).df
mlo_MM.resample('1M') # add rows for missing months, filled with NaN 
mlo_MM.interpolate(inplace=True) # linearly interpolate missing data

# loop through years of caribic data
for c_year in range(2005, 2022):
    c_data = caribic.select_year(c_year)
    if len(c_data[c_data['SF6 [ppt]'].notna()]) < 1: 
        continue
    else:
        lags = calc_time_lags(c_data, mlo_MM)
        if all(np.isnan(np.array(lags))): 
            print(f'no lags calculated for {c_year}'); continue
        plot_time_lags(c_data, lags, ref_min, ref_max)

#%% Filter tropospheric and stratospheric data
# loop through years of caribic data
data_filtered = pd.DataFrame() # initialise full dataframe
for c_year in range(2005, 2022): 
    print(f'{c_year}')
    c_data = caribic.select_year(c_year)
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

data_trop_outlier = filter_trop_outliers(data_filtered, ['n2o'], source='car')


#%% Detrend
mlo_detrend_ref = Mauna_Loa(range(2006, 2020)).df
data_detr = detrend_substance(c_df, 'SF6 [ppt]', mlo_detrend_ref, 'SF6catsMLOm')
