# -*- coding: utf-8 -*-
"""
@Author: Sophie Bauchimger, IAU
@Date: Thu Apr 27 15:56:59 2023

Main script for Caribic measurement analysis routine.

Substances in Caribic data:
'GHG':    ['ch4', 'co2',        'n2o', 'sf6']
'INT':    ['ch4', 'co2', 'co' ,             'o3' , 'noy', 'no' , 'h2o']
'INT2':   ['ch4', 'co2', 'co' , 'n2o',      'o3' , 'noy', 'no' , 'h2o', 'f11', 'f12']
"""
from data_classes import Caribic, Mozart, Mauna_Loa, Mace_Head
from time_lag import calc_time_lags
from dictionaries import substance_list

from plot.gradients import plot_gradient_by_season
from filter_outliers import filter_strat_trop, filter_trop_outliers
from detrend import detrend_substance
from plot.data import plot_scatter_global, plot_global_binned_1d, plot_global_binned_2d, plot_1d_LonLat, plot_local
from plot.eqlat import plot_eqlat_deltheta

#%% Get data
year_range = range(1980, 2021)

mlo_data = {subs : Mauna_Loa(year_range, substance=subs) for subs in substance_list('MLO')}
caribic = Caribic(year_range, pfxs = ['GHG', 'INT', 'INT2']) # 2005-2020
mhd = Mace_Head() # only 2012 data available
mzt = Mozart(year_range) # only available up to 2008

# examples for creating new objects with only certain year / flight number / latitude :
# c_2008 = caribic.sel_year(2008)
# c_fl340 = caribic.sel_flight(340)
c_gt30N = caribic.sel_latitude(30, 90)

# c_yr08_to_12 = caribic.sel_year(*range(2008, 2012))
# c_fl340_to_360 = caribic.sel_flight(*range(340, 360))

#%% Plot data
import matplotlib.pyplot as plt

for pfx in caribic.pfxs: # scatter plots of all caribic data
    substs = [x for x in substance_list(pfx) if x not in ['f11', 'f12', 'no', 'noy', 'o3', 'h2o']]
    f, axs = plt.subplots(int(len(substs)/2), 2, figsize=(10,len(substs)*1.5), dpi=200)
    plt.suptitle(f'Caribic {(pfx)}')
    for subs, ax in zip(substs, axs.flatten()): 
        plot_scatter_global(caribic, subs, c_pfx=pfx, as_subplot=True, ax=ax)
    f.autofmt_xdate()
    plt.tight_layout()
    plt.show()

for pfx in caribic.pfxs: # lon/lat plots of all caribic data 
    substs = [x for x in substance_list(pfx) if x not in ['f11', 'f12', 'no', 'noy', 'o3', 'h2o']]
    for subs in substs: 
        plot_global_binned_1d(caribic, subs=subs, c_pfx=pfx, single_graph=True)

for pfx in caribic.pfxs: # maps of all caribic data
    substs = [x for x in substance_list(pfx) if x not in ['f11', 'f12', 'no', 'noy', 'o3', 'h2o']]
    for subs in substs: 
        plot_global_binned_2d(caribic, subs=subs, c_pfx=pfx)

plot_global_binned_1d(mzt, 'sf6', single_graph=True)
yr_ranges = [mzt.years[i:i + 9] for i in range(0, len(mzt.years), 9)] # creates year ranges of 6 items for mozart data
for yr_range in yr_ranges:
    plot_global_binned_2d(mzt, 'sf6', years=yr_range)

plot_1d_LonLat(mzt, 'sf6', single_yr = 2005)

for subs, mlo_obj in mlo_data.items():
    plot_local(mlo_obj, subs)

plot_local(mhd, 'sf6')

#%% Time lags - lags are added to dictionary caribic.lags as lags_pfx : dataframe
lag_plot= True

calc_time_lags(caribic, mlo_data['sf6'], range(2005, 2020), substance = 'sf6', plot_all=lag_plot)
calc_time_lags(caribic, mlo_data['n2o'], range(2005, 2020), substance = 'n2o', pfx='INT2', plot_all=lag_plot)

calc_time_lags(caribic, mlo_data['n2o'], range(2005, 2020), substance = 'n2o', plot_all=lag_plot)
calc_time_lags(caribic, mlo_data['co2'], range(2005, 2020), substance = 'co2', pfx='INT2', plot_all=lag_plot)

# to delete the newly created attributes may use
# del caribic.data['lag_GHG'] etc

#%% Detrend
for subs in ['sf6', 'n2o', 'co2', 'ch4']:
    detrend_substance(caribic, subs, mlo_data[subs])

#%% Plot gradients 
for pfx in ['INT2']:# caribic.pfxs:
    for subs in substance_list(pfx):
        plot_gradient_by_season(caribic, subs, 'INT2', tp='pvu', pvu = 2.0)

#%% Filter tropospheric / stratospheric data points based on n2o mixing ratio wrt Mauna Loa data 
for pfx in caribic.pfxs: 
    filter_strat_trop(caribic, mlo_data['n2o'], 'n2o', pfx)

for pfx in caribic.pfxs:
    for subs in substance_list(pfx):
        if subs in ['sf6', 'n2o', 'co2', 'ch4']:
            filter_trop_outliers(caribic, subs, pfx, crit=subs)

#%% Eq. lat vs potential temperature wrt tropopause
plot_eqlat_deltheta(caribic, c_pfx='INT2', subs='n2o', tp='pvu')
