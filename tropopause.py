# -*- coding: utf-8 -*-
"""
@Author: Sophie Bauchinger, IAU
@Date Mon Aug 14 14:06:26 2023

"""

import matplotlib.pyplot as plt
import dill
import numpy as np
import pandas as pd

from toolpac.calc.binprocessor import Bin_equi1d, Simple_bin_1d

from dictionaries import get_coord, get_tp_params, get_coordinates

#%% Import data
if not 'emac' in locals(): 
    with open('misc_data\emac_complete.pkl', 'rb') as f:
        emac = dill.load(f)

if not 'caribic' in locals():
    with open('misc_data\caribic.pkl', 'rb') as f: 
        caribic = dill.load(f)

# Generate needed values from EMAC if not available
if not 'tp' in emac.data: 
    emac.create_tp()

def get_obj(source):
    if source == 'Caribic': return caribic
    if source =='EMAC': return emac

#%% Tropopause height vs latitude

# def tph_vs_lat(glob_obj, **tp_params):
#     # if ID == 'ECMWF':
#     #     data = glob_obj.met_data
#     #     latitude = np.array(data.geometry.x)

#     #     dp_col =  'int_dp_strop_hpa [hPa]'
#     #     p_col = 'p [mbar]'
#     #     tp_p = data[p_col] - data[dp_col]

#     # elif ID == 'ERA5':
#     #     data = glob_obj.met_data
#     #     latitude = np.array(data.geometry.x)
#     #     tp_p = data['int_ERA5_TROP1_PRESS [hPa]']

#     # elif ID == 'EMAC':
#     #     latitude = np.array(glob_obj.df.geometry.x)
#     #     tp_p = glob_obj.df['tropop_tp_WMO'] * 1e-2

#     return latitude, tp



#%% Scatter plot
def plot_tp_height(ax, obj, plot_params, **tp_params):
    """ """
    if obj.source == 'Caribic': 
        data = obj.met_data
    elif obj.source == 'EMAC':
        data = obj.df

    x = np.array(data.geometry.x)
    # no dynamical tropopause in the tropics
    if tp_params['tp_def'] == 'dyn':
        data = data.where(pd.Series([(i>30 or i<-30) for i in x ]) )
        x = data.geometry.x
    # no cold point tropopause outside the tropics
    if tp_params['tp_def'] == 'cpt':
        data = data.where(pd.Series([(i<30 or i>-30) for i in x ]) )
        x = data.geometry.x

    v = data[tp_params['col_name']]

    if tp_params['unit'] == 'Pa': 
        print(tp_params['col_name'], tp_params['unit'])
        v = v*1e-2 #!!! NONSENSE but want to make it into hPa

    # latitudes, tropopause_heights = tph_vs_lat(obj, ID, **tp_params)
    ax.scatter(x, v, label = '{}_{}'.format(tp_params['ID'], tp_params['tp_def']))
    ax.set_xlabel('Latitude [°N]')
    ax.set_ylabel('Pressure of the thermal tropopause [hPa]')
    # ax.set_ylim(plot_params['ylim'])
    # ax.set_xlim(plot_params['xlim'])
    return

def plot_av_tp_height(ax, obj, plot_params, **tp_params):
    """ """
    if obj.source == 'Caribic': data = obj.met_data
    elif obj.source == 'EMAC': data = obj.df
    if not tp_params['col_name'] in data.columns: return
    
    x = np.array(data.geometry.x)
    v = data[tp_params['col_name']]

    if tp_params['tp_def'] == 'dyn':
        data = data.where(pd.Series([(i>30 or i<-30) for i in x ]) )
        x = data.geometry.x
    # no cold point tropopause outside the tropics
    if tp_params['tp_def'] == 'cpt':
        data = data.where(pd.Series([(i<30 or i>-30) for i in x ]) )
        x = data.geometry.x

    if tp_params['unit'] == 'Pa': 
        v = v*1e-2 #!!! NONSENSE but want to make it into hPa

    xbmin, xbmax, xbsize = -90, 90, 10
    bci = Bin_equi1d(xbmin, xbmax, xbsize)
    bin1d = Simple_bin_1d(v,x,bci)
    
    # ax.plot(bin1d.xmean, bin1d.vmean, label = ID)
    #colors = {'clim':'grey', 'cpt':'blue', 'dyn':'green', 'therm':'red', 'combo':'grey'}
    ax.scatter(bin1d.xmean, bin1d.vmean, #c=colors[tp_params['tp_def']],
               label = '{}_{}'.format(tp_params['ID'], tp_params['tp_def']))
    # ax.scatter(bin1d.xmean, bin1d.vmean, label = tp_params['col_name'])
    ax.errorbar(bin1d.xmean, bin1d.vmean, bin1d.vstdv, capsize=2)

    ax.set_xlabel('Latitude [°N]')
    ax.set_ylabel('{}{} [{}]'.format('$\Delta$' if tp_params['rel_to_tp'] else '', 
                                     tp_params['vcoord'], tp_params['unit']))
    # ax.set_ylim(plot_params['ylim'])
    # ax.set_xlim(plot_params['xlim'])
    return

def plot_abs(vcoord):
    """ Plot all absolute tropopause heights on a pressure plot """
    tps = get_coordinates(vcoord=vcoord, tp_def='not_nan', rel_to_tp=False)
    fig, axs = plt.subplots(1, dpi=150)
    plt.title(f'absolute {vcoord}')
    for tp in tps:
        # plot_tp_height(axs, get_obj(tp.source), 
        #                 plot_params = {},#'ylim':(50, 500), 'xlim':(-40, 90)},
        #                 **tp.__dict__)
        plot_av_tp_height(axs, get_obj(tp.source), 
                        plot_params = {},#'ylim':(50, 500), 'xlim':(-40, 90)},
                        **tp.__dict__)
        plt.legend(title=tp.tp_def)
    axs.invert_yaxis()
    plt.show()

def plot_rel(vcoord):
    """ Plot all relative tropopause heights on a pressure plot """
    tps = get_coordinates(vcoord=vcoord, tp_def = 'not_nan', rel_to_tp=True)
    fig, axs = plt.subplots(1, dpi=150)
    plt.title(f'relative {vcoord}')
    for tp in tps:
        # plot_tp_height(axs, get_obj(tp.source), 
        #                 plot_params = {},#'ylim':(50, 500), 'xlim':(-40, 90)},
        #                 **tp.__dict__)
        plot_av_tp_height(axs, get_obj(tp.source), 
                        plot_params = {},#'ylim':(50, 500), 'xlim':(-40, 90)},
                        **tp.__dict__)
        plt.legend(title=tp.tp_def)
    axs.invert_yaxis()
    plt.show()

for vc in ['p', 'pt', 'z']:
    plot_abs(vcoord=vc)
    plot_rel(vcoord=vc)

# fig, axs = plt.subplots(1, dpi=250)
# for ax, obj, ID in zip([axs, axs, axs], 
#                        [caribic, caribic, emac], 
#                        ['ECMWF', 'ERA5', 'EMAC']): 

#     tps = get_coordinates(**{'vcoord':'pt', 'source':'Caribic'})
#     for tp in tps.values(): 
#         tp_params= tp.__dict__
#         plot_tp_height(ax, caribic, 
#                        plot_params = {'ylim':(50, 500), 'xlim':(-40, 90)},
#                        **tp.__dict__)



#%% Binned
# def plot_av_tp_height(ax, obj, ID, plot_params, **tp_params):
#     latitudes, tropopause_heights = tph_vs_lat(obj, ID, **tp_params)


    
#     v = tropopause_heights
#     x = latitudes
    
#     bin1d = Simple_bin_1d(v,x,bci)
    
#     # ax.plot(bin1d.xmean, bin1d.vmean, label = ID)
#     ax.scatter(bin1d.xmean, bin1d.vmean, label = ID)
#     ax.errorbar(bin1d.xmean, bin1d.vmean, bin1d.vstdv, capsize=2)
#     ax.set_xlabel('Latitude [°N]')
#     ax.set_ylabel('Pressure of the thermal tropopause [hPa]')
#     ax.set_ylim(plot_params['ylim'])
#     ax.set_xlim(plot_params['xlim'])
#     ax.invert_yaxis()
#     return

# fig, axs = plt.subplots(1, dpi=250)
# for ax, obj, ID in zip([axs, axs, axs], 
#                        [caribic, caribic, emac], 
#                        ['ECMWF', 'ERA5', 'EMAC']): 
#     plot_av_tp_height(ax, obj, ID, plot_params = {'ylim':(50, 500),
#                                                   'xlim':(-40, 90)})
# plt.legend()
# plt.show()