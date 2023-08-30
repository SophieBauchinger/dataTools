# -*- coding: utf-8 -*-
"""
@Author: Sophie Bauchinger, IAU
@Date Mon Aug 14 14:06:26 2023

Plotting Tropopause heights for different tropopauses different vertical coordinates
"""
import matplotlib.pyplot as plt
import dill
import numpy as np

from toolpac.calc.binprocessor import Bin_equi1d, Simple_bin_1d

from tools import make_season
from dictionaries import get_coordinates, dict_season

#%% Import data
if not 'emac' in locals(): 
    from data import EMACData
    emac = EMACData()

if not 'caribic' in locals():
    with open('misc_data\Caribic.pkl', 'rb') as f: 
        caribic = dill.load(f)

# Generate needed values from EMAC if not available
if not 'tp' in emac.data: 
    emac.create_tp()

def get_obj(source):
    if source == 'Caribic': return caribic
    if source =='EMAC': return emac

#%% Scatter plot
def tp_vs_latitude(ax, obj, plot_params, **tp_params):
    """ """
    if obj.source == 'Caribic': data = obj.met_data.copy()
    elif obj.source == 'EMAC': data = obj.df.copy()
    if not tp_params['col_name'] in data.columns: return

    if tp_params['tp_def'] == 'dyn': # dynamic TP only outside the tropics
        data = data[np.array([(i>30 or i<-30) for i in np.array(data.geometry.x) ])]
    if tp_params['tp_def'] == 'cpt': # cold point TP only in the tropics 
        data = data[np.array([(i<30 or i>-30) for i in np.array(data.geometry.x) ])]

    x = np.array(data.geometry.x)
    v = data[tp_params['col_name']]

    ax.scatter(x, v,
               label = '{}_{}'.format(tp_params['source'], tp_params['tp_def']))

    # if tp_params['var1'] in data.columns and not tp_params['rel_to_tp']:
    #     ax.scatter(x, data[tp_params['var1']], c='k')

    ax.set_xlabel('Latitude [°N]')
    ax.set_ylabel('{}{} [{}]'.format('$\Delta$' if tp_params['rel_to_tp'] else '', 
                                     tp_params['vcoord'], tp_params['unit']))
    # ax.set_ylim(plot_params['ylim'])
    # ax.set_xlim(plot_params['xlim'])
    return

def av_tp_vs_latitude(ax, obj, plot_params, **tp_params):
    """ """
    if obj.source == 'Caribic': data = obj.met_data.copy()
    elif obj.source == 'EMAC': data = obj.df.copy()
    if not tp_params['col_name'] in data.columns: return

    if tp_params['tp_def'] == 'dyn': # dynamic TP only outside the tropics
        data = data[np.array([(i>30 or i<-30) for i in np.array(data.geometry.x) ])]
    if tp_params['tp_def'] == 'cpt': # cold point TP only in the tropics 
        data = data[np.array([(i<30 and i>-30) for i in np.array(data.geometry.x) ])]

    x = np.array(data.geometry.x)
    v = data[tp_params['col_name']]

    xbmin, xbmax, xbsize = -90, 90, 10
    bci = Bin_equi1d(xbmin, xbmax, xbsize)
    bin1d = Simple_bin_1d(v,x,bci)
    
    # ax.plot(bin1d.xmean, bin1d.vmean, label = ID)
    #colors = {'clim':'grey', 'cpt':'blue', 'dyn':'green', 'therm':'red', 'combo':'grey'}
    ax.scatter(bin1d.xmean, bin1d.vmean, #c=colors[tp_params['tp_def']],
               label = '{}_{}'.format(tp_params['model'], tp_params['tp_def']))
    # ax.scatter(bin1d.xmean, bin1d.vmean, label = tp_params['col_name'])   
    ax.errorbar(bin1d.xmean, bin1d.vmean, bin1d.vstdv, capsize=2)

    ax.set_xlabel('Latitude [°N]')
    ax.set_ylabel('{}{} [{}]'.format('$\Delta$' if tp_params['rel_to_tp'] else '', 
                                     tp_params['vcoord'], tp_params['unit']))
    # ax.set_ylim(plot_params['ylim'])
    # ax.set_xlim(plot_params['xlim'])
    return

def tp_scatter(vcoord, rel, av=True):
    tps = get_coordinates(vcoord=vcoord, tp_def='not_nan', rel_to_tp=rel)
    fig, axs = plt.subplots(1, dpi=150, figsize=(7,5))
    plt.title('{} {}'.format('TP in' if not rel else 'TP wrt flight in', vcoord))
    for tp in tps:
        if av: 
            av_tp_vs_latitude(axs, get_obj(tp.source), 
                            plot_params = {},#'ylim':(50, 500), 'xlim':(-40, 90)},
                            **tp.__dict__)
        else: 
            tp_vs_latitude(axs, get_obj(tp.source), 
                            plot_params = {},#'ylim':(50, 500), 'xlim':(-40, 90)},
                            **tp.__dict__)
        plt.legend()
    if vcoord == 'p': axs.invert_yaxis()
    plt.show()

for vc in ['p', 'pt', 'z']:
    tp_scatter(vcoord=vc, rel=False, av=True)
    tp_scatter(vcoord=vc, rel=False, av=False)
    tp_scatter(vcoord=vc, rel=True, av=True)

#%% Per season
def seasonal_av_tp_vs_latitude(axs, obj, plot_params, **tp_params):
    """ """
    if obj.source == 'Caribic': data = obj.met_data.copy()
    elif obj.source == 'EMAC': data = obj.df.copy()
    if not tp_params['col_name'] in data.columns: return

    data['season'] = make_season(data.index.month) # 1 = spring etc

    if tp_params['tp_def'] == 'dyn': # dynamic TP only outside the tropics
        data = data[np.array([(i>30 or i<-30) for i in np.array(data.geometry.x) ])]
    if tp_params['tp_def'] == 'cpt': # cold point TP only in the tropics 
        data = data[np.array([(i<30 or i>-30) for i in np.array(data.geometry.x) ])]

    for s,ax in zip(set(data['season'].tolist()), axs.flatten()):
        df = data.loc[data['season'] == s]
        x = np.array(df.geometry.x)
        v = df[tp_params['col_name']]
    
        xbmin, xbmax, xbsize = -90, 90, 10
        bci = Bin_equi1d(xbmin, xbmax, xbsize)
        bin1d = Simple_bin_1d(v,x,bci)

        ax.scatter(bin1d.xmean, bin1d.vmean, # c=colors[tp_params['tp_def']],
                   label = '{}_{}'.format(tp_params['model'], tp_params['tp_def']))
                   # c = dict_season()[f'color_{s}'],
                   # label = dict_season()[f'name_{s}'])
        ax.errorbar(bin1d.xmean, bin1d.vmean, bin1d.vstdv, capsize=2)

        ax.set_xlabel('Latitude [°N]')
        ax.set_ylabel('{}{} [{}]'.format('$\Delta$' if tp_params['rel_to_tp'] else '', 
                                         tp_params['vcoord'], tp_params['unit']))
        ax.set_title(dict_season()[f'name_{s}'])
        # ax.set_ylim(plot_params['ylim'])
        # ax.set_xlim(plot_params['xlim'])
    return

def seasonal_tp_scatter(vcoord, rel):
    tps = get_coordinates(vcoord=vcoord, tp_def='not_nan', rel_to_tp=rel)
    fig, axs = plt.subplots(2,2, dpi=150, figsize=(9,5))
    plt.suptitle(f'TP in {vcoord}')
    for tp in tps:
        seasonal_av_tp_vs_latitude(axs, get_obj(tp.source), 
                        plot_params = {},#'ylim':(50, 500), 'xlim':(-40, 90)},
                        **tp.__dict__)

    if vcoord == 'p':
        for ax in axs.flatten(): ax.invert_yaxis()
    fig.tight_layout()
    
    lines, labels = axs.flatten()[0].get_legend_handles_labels()
    # lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    fig.legend(lines, labels, loc='center right')
    plt.subplots_adjust(right=0.8)
    
    plt.show()
    return 

for vc in ['p', 'pt', 'z']:
    seasonal_tp_scatter(vcoord=vc, rel=False)
