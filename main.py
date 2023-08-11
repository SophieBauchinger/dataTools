# -*- coding: utf-8 -*-
"""
@Author: Sophie Bauchimger, IAU
@Date: Thu Apr 27 15:56:59 2023

Main script for Caribic measurement analysis routine.

Substances in Caribic data:
'GHG':    ['ch4', 'co2',        'n2o', 'sf6']
'INT':    ['ch4', 'co2', 'co' ,             'o3' , 'noy', 'no' , 'h2o']
'INT2':   ['ch4', 'co2', 'co' , 'n2o',      'o3' , 'noy', 'no' , 'h2o', 'f11', 'f12']

Met / Reanalysis data for Caribic-2:
    'Flight number',
     'p [mbar]',
     'geometry'
     'int_z_km [km]',
     'int_eqlat [deg]',
     'int_ERA5_EQLAT [deg N]',
     'int_ERA5_PRESS [hPa]',
     'int_ERA5_PV [PVU]',
     'int_ERA5_TEMP [K]',

"""
import dill
from os.path import exists
from os import remove
import matplotlib.pyplot as plt

from data import Caribic, Mozart
from groundbased import Mauna_Loa, Mace_Head
from lags import calc_time_lags
from dictionaries import substance_list, get_tp_params

# from data import detrend_substance
import plot.data
from plot.gradients import plot_gradient_by_season
# from plot.eqlat import plot_eqlat_deltheta
# from baseline import baseline_filter
# from tools import data_selection

#%% Get Data
year_range = range(1980, 2021)

mlo_data = {subs : Mauna_Loa(year_range, subs=subs) for subs
            in substance_list('MLO')}
mhd = Mace_Head() # only 2012 data available
mzt = Mozart(year_range) # only available up to 2008

# only calculate caribic data if necessary
def load_caribic(fname = 'caribic.pkl'):
    """ 'caribic_180723.pkl' ;  """
    if exists(fname): # Avoid long file loading times
        with open(fname, 'rb') as f:
            caribic = dill.load(f)
        del f
    else: caribic = Caribic(year_range, pfxs = ['GHG', 'INT', 'INT2'])
    return caribic

def save_caribic(fname = 'caribic_dill.pkl'):
    """ Drop rows where pressure values don't exist, then save
    caribic object to dill file """
    with open(fname, 'wb') as f:
        dill.dump(caribic, f)

def del_caribic_file(fname = 'caribic_dill.pkl'): remove(fname)

emac = load_caribic('emac.pkl')
emac_subsam = load_caribic('emac_subsam.pkl')
caribic = load_caribic()
# c_data = load_caribic('caribic_data.pkl')

# caribic_filtered = caribic.filter_extreme_events('chem')

# save_caribic(fname= 'carbic_dill_mod.pkl')

# examples for creating new objects with only certain year / flight nr / lat :
# kwargs = {'tp_def' : 'chem'}
# new_caribic = data_selection(caribic, flights=None, years=None, latitudes=None,
#                               tropo=False, strato=False, extr_events=False, **kwargs)


#%% Tropopause definitions
tp_param_dicts = get_tp_params() # ^ for tropopause / stratosphere sorting
for tp_params in tp_param_dicts:
    caribic.filter_extreme_events(**tp_params, plot=True)

#%% Plot data
for pfx in caribic.pfxs: # scatter plots of all caribic data
    substs = [x for x in substance_list(pfx)
              if x not in ['f11', 'f12', 'no', 'noy', 'o3', 'h2o']]
    f, axs = plt.subplots(int(len(substs)/2), 2,
                          figsize=(10,len(substs)*1.5), dpi=200)
    plt.suptitle(f'Caribic {(pfx)}')
    for subs, ax in zip(substs, axs.flatten()):
        plot.data.scatter_global(caribic, subs, c_pfx=pfx, as_subplot=True, ax=ax)
    f.autofmt_xdate()
    plt.tight_layout()
    plt.show()

for pfx in caribic.pfxs: # lon/lat plots of all caribic data
    substs = [x for x in substance_list(pfx)
              if x not in ['f11', 'f12', 'no', 'noy', 'o3', 'h2o']]
    for subs in substs:
        plot.data.plot_binned_1d(caribic, subs=subs, c_pfx=pfx, single_graph=True)

for pfx in caribic.pfxs: # maps of all caribic data
    substs = [x for x in substance_list(pfx)
              if x not in ['f11', 'f12', 'no', 'noy', 'o3', 'h2o']]
    for subs in substs:
        plot.data.plot_binned_2d(caribic, subs=subs, c_pfx=pfx)

plot.data.plot_binned_1d(mzt, 'sf6', single_graph=True)
# creates year ranges of 6 items for mozart data
yr_ranges = [mzt.years[i:i + 9] for i in range(0, len(mzt.years), 9)]
for yr_range in yr_ranges:
    plot.data.plot_binned_2d(mzt, 'sf6', years=yr_range)

plot.data.lonlat_1d(mzt, 'sf6', single_yr = 2005)

for subs, mlo_obj in mlo_data.items():
    plot.data.local(mlo_obj, subs)

plot.data.local(mhd, 'sf6')

#%% Time lags - lags are added to dictionary caribic.lags as lags_pfx : dataframe
lag_plot= False

calc_time_lags(caribic, mlo_data['sf6'], range(2005, 2020),
               substance = 'sf6', plot_all=lag_plot)
calc_time_lags(caribic, mlo_data['n2o'], range(2005, 2020),
               substance = 'n2o', pfx='INT2', plot_all=lag_plot)

calc_time_lags(caribic, mlo_data['n2o'], range(2005, 2020),
               substance = 'n2o', plot_all=lag_plot)
calc_time_lags(caribic, mlo_data['co2'], range(2005, 2020),
               substance = 'co2', pfx='INT2', plot_all=lag_plot)

# to delete the newly created attributes may use
# del caribic.data['lag_GHG'] etc

#%% Detrend
for subs in ['sf6', 'n2o', 'co2', 'ch4']:
    caribic.detrend(subs, mlo_data[subs])

#%% Plot gradients
for pfx in ['INT2']:# caribic.pfxs:
    for subs in substance_list(pfx):
        plot_gradient_by_season(caribic, subs, 'INT2', tp='pvu', pvu = 2.0)

#%% Filter tropospheric / stratospheric data points based on
# chem = {'GHG' : ['n2o'],
#         'INT' : ['o3'],
#         'INT2' : ['n2o', 'o3']}
# therm = {'INT' : ['dp', 'pt'],
#          'INT2' : ['dp', 'pt', 'z']}
# dyn_3_5 = {'INT' : ['dp', 'pt', 'z'],
#            'INT2' : ['pt']}
# dyn_1_5 = dyn_2_0 = {'INT2' : ['pt']}


# Baseline Filter on tropospheric datasets
for pfx in ['GHG', 'INT', 'INT2']:
    for subs in substance_list(pfx):
        f, axs = plt.subplots(1, 3, figsize=(10, 4), dpi=200)
        for tp_def, ax in zip(['chem', 'therm', 'dyn'], axs.flatten()):
            pass # baseline_filter(caribic, subs=subs, c_pfx=pfx, tp_def=tp_def, ax=ax)
        f.autofmt_xdate()
        plt.tight_layout()
        plt.show()

# n2o mixing ratio wrt Mauna Loa data

# ref_obj = Mauna_Loa(year_range, 'n2o')

# for pfx in caribic.pfxs:
#     chemical(caribic, mlo_data['n2o'], 'n2o', pfx)

# for pfx in caribic.pfxs:
#     for subs in substance_list(pfx):
#         if subs in ['sf6', 'n2o', 'co2', 'ch4']:
#             filter_trop_outliers(caribic, subs, pfx, crit=subs)

#%% Eq. lat vs potential temperature wrt tropopause
# plot_eqlat_deltheta(caribic, c_pfx='INT2', subs='n2o', tp='pvu')
