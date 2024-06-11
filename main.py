# -*- coding: utf-8 -*-
"""
@Author: Sophie Bauchinger, IAU
@Date: Thu Apr 27 15:56:59 2023

Main script for Caribic measurement analysis routine. Theoretically.
"""
import matplotlib.pyplot as plt

from  dataTools.data.Caribic import Caribic
from  dataTools.data.CampaignData import CampaignData
from  dataTools.data.EMAC import EMAC
from  dataTools.data.Mozart import Mozart
from  dataTools.data._local import MaunaLoa, MaceHead

import dataTools.dictionaries as dcts
from dataTools import tools

# from data import detrend_substance
# import plot.data
# from plot.gradients import plot_gradient_by_season
# from plot.eqlat import plot_eqlat_deltheta
# from baseline import baseline_filter
# from tools import data_selection

#%% Get Data
# mlo = MaunaLoa()
# mhd = MaceHead() # only 2012 data available
# mzt = Mozart() # only available up to 2008
caribic = Caribic()
# emac = EMAC()
# tpause = TropopauseData()+
shtr = CampaignData('SHTR')
wise = CampaignData('WISE')
pgs = CampaignData('PGS')
# tacts = CampaignData('TACTS')
atom = CampaignData('ATOM')
# hippo = CampaignData('HIPPO') #!!! would have to implement in toolpac

# sql_objs = [shtr, wise, atom] # pgs, tacts

all_data = caribic + shtr + wise + pgs

#%% Get main coordinates and substances
caribic_tps = tools.minimise_tps(dcts.get_coordinates(tp_def='not_nan', source='Caribic'))
pt = dcts.get_coord(vcoord='pt', model='MSMT')

lat = dcts.get_coord(hcoord='lat', ID='GHG')
lon = dcts.get_coord(hcoord='lon', ID='GHG')
eql = dcts.get_coord(hcoord='eql', model='ERA5')

sf6 = dcts.get_subs(short_name='detr_sf6', ID='GHG')
n2o = dcts.get_subs(short_name='detr_n2o', ID='GHG')
co2 = dcts.get_subs(short_name='detr_co2', ID='GHG')
ch4 = dcts.get_subs(short_name='detr_ch4', ID='GHG')

#%% Tropopause definition comparison


#%% Plot data
# for pfx in caribic.pfxs: # scatter plots of all caribic data
#     substs = [x for x in dcts.substance_list(pfx)
#               if x not in ['f11', 'f12', 'no', 'noy', 'o3', 'h2o']]
#     f, axs = plt.subplots(int(len(substs)/2), 2,
#                           figsize=(10,len(substs)*1.5), dpi=200)
#     plt.suptitle(f'Caribic {(pfx)}')
#     for subs, ax in zip(substs, axs.flatten()):
#         plot.data.scatter_global(caribic, subs, c_pfx=pfx, as_subplot=True, ax=ax)
#     f.autofmt_xdate()
#     plt.tight_layout()
#     plt.show()

# for pfx in caribic.pfxs: # lon/lat plots of all caribic data
#     substs = [x for x in dcts.substance_list(pfx)
#               if x not in ['f11', 'f12', 'no', 'noy', 'o3', 'h2o']]
#     for subs in substs:
#         plot.data.plot_binned_1d(caribic, subs=subs, c_pfx=pfx, single_graph=True)

# for pfx in caribic.pfxs: # maps of all caribic data
#     substs = [x for x in dcts.substance_list(pfx)
#               if x not in ['f11', 'f12', 'no', 'noy', 'o3', 'h2o']]
#     for subs in substs:
#         plot.data.plot_binned_2d(caribic, subs=subs, c_pfx=pfx)

# plot.data.plot_binned_1d(mzt, 'sf6', single_graph=True)
# # creates year ranges of 6 items for mozart data
# yr_ranges = [mzt.years[i:i + 9] for i in range(0, len(mzt.years), 9)]
# for yr_range in yr_ranges:
#     plot.data.plot_binned_2d(mzt, 'sf6', years=yr_range)

# plot.data.lonlat_1d(mzt, 'sf6', single_yr = 2005)

# # for subs, mlo_obj in mlo_data.items():
# #     plot.data.local(mlo_obj, subs)

# plot.data.local(mhd, 'sf6')

#%% Time lags - lags are added to dictionary caribic.lags as lags_pfx : dataframe
# lag_plot= False

# calc_time_lags(caribic, mlo_data['sf6'], range(2005, 2020),
#                substance = 'sf6', plot_all=lag_plot)
# calc_time_lags(caribic, mlo_data['n2o'], range(2005, 2020),
#                substance = 'n2o', pfx='INT2', plot_all=lag_plot)

# calc_time_lags(caribic, mlo_data['n2o'], range(2005, 2020),
#                substance = 'n2o', plot_all=lag_plot)
# calc_time_lags(caribic, mlo_data['co2'], range(2005, 2020),
#                substance = 'co2', pfx='INT2', plot_all=lag_plot)

# to delete the newly created attributes may use
# del caribic.data['lag_GHG'] etc

#%% Detrend
# for subs in ['sf6', 'n2o', 'co2', 'ch4']:
#     caribic.detrend(subs, mlo_data[subs])

#%% Plot gradients
# for pfx in ['INT2']:# caribic.pfxs:
#     for subs in dcts.substance_list(pfx):
#         plot_gradient_by_season(caribic, subs, 'INT2', tp='pvu', pvu = 2.0)

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
# for pfx in ['GHG', 'INT', 'INT2']:
#     for subs in dcts.substance_list(pfx):
#         f, axs = plt.subplots(1, 3, figsize=(10, 4), dpi=200)
#         for tp_def, ax in zip(['chem', 'therm', 'dyn'], axs.flatten()):
#             pass # baseline_filter(caribic, subs=subs, c_pfx=pfx, tp_def=tp_def, ax=ax)
#         f.autofmt_xdate()
#         plt.tight_layout()
#         plt.show()

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
