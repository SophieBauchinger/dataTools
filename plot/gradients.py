# -*- coding: utf-8 -*-
"""
@Author: Sophie Bauchimger, IAU
@Date: Fri Apr 28 14:13:28 2023

Plotting of gradients - wants detrended data, sorted into atmos. layers

"""
import numpy as np
import matplotlib.pyplot as plt

import toolpac.calc.binprocessor as bp

from detrend import detrend_substance
from dictionaries import get_col_name, dict_season, choose_column
from tools import make_season, coordinate_tools
from data import Mauna_Loa

#%% Plotting Gradient by season
# Fct definition in C_plot needed these:
# select_var=['fl_ch4','fl_sf6', 'fl_n2o'] # flagged data
# select_value=[0,0,0]
# select_cf=['GT','GT', 'GT'] # operators

# ptsmin (int): minimum number of pts for a bin to be considered #!!! implement

def plot_gradient_by_season(c_obj, subs, c_pfx = None, tp_def='therm', pvu = 3.5, errorbars=False,
                            detr=False, note=None, ycoord='pt', y_bin=None):
    """
    Plotting gradient by season using 1D binned data. Detrended data used by default
    (Inspired by C_plot.pl_gradient_by_season)

    Parameters:
        c_obj (Caribic)
        subs (str): substance e.g. 'sf6'
        c_pfx (str): 'GHG', 'INT', 'INT2'
        tp_def (str): tropopause definition
        pvu (float): potential vorticity for dyn. tp definition. 1.5, 2.0 or 3.5
        errorbars (bool)
        y_bin (int): bin size for 1D binning (depends on coordinate)
        detr (bool)
        note (str): shown as text box on the plot
    """
    if c_pfx is not None and c_pfx in c_obj.pfxs: data = c_obj.data[c_pfx]
    elif subs in c_obj.data.keys(): data = c_obj.data[subs]
    else: print(f'No data found for {subs} / {c_pfx}'); return

    try: substance = get_col_name(subs, c_obj.source, c_pfx)
    except: substance = get_col_name(subs, 'Caribic', 'GHG')
    if detr: 
        if not f'detr_{substance}' in data.columns:
            try: 
                detrend_substance(c_obj, subs, Mauna_Loa(c_obj.years, subs), save=True)
                data = c_obj.data[c_pfx]
                substance = f'detr_{substance}'
            except: print('Detrending not successful, proceeding with original data.')
        else: substance = f'detr_{substance}'
    
    try: y_coord, y_label = coordinate_tools(tp_def=tp_def, c_pfx=c_pfx, ycoord=ycoord, pvu=pvu)
    except: 
        y_coord = choose_column(data, var='y-coordinate')
        y_label = input('Please input the y-label\n')
    y_bins = {'z' : 0.5, 'pt' : 10, 'p' : 40}
    if not y_bin: y_bin = y_bins[ycoord]
    min_y, max_y = np.nanmin(data[y_coord].values), np.nanmax(data[y_coord].values)
    nbins = (max_y - min_y) / y_bin
    y_array = min_y + np.arange(nbins) * y_bin + y_bin * 0.5

    data['season'] = make_season(data.index.month) # 1 = spring etc
    out_dict = {}
    fig, ax = plt.subplots(dpi=200)
    for s in set(data['season'].tolist()):
        df = data.loc[data['season'] == s]
        y_values = df[y_coord].values # df[eq_lat_col].values #
        x_values = df[substance].values

        out_dict[f'bin1d_{s}'] = bp.Simple_bin_1d(x_values, y_values,
                                                  bp.Bin_equi1d(min_y, max_y, y_bin))
        vmean = (out_dict[f'bin1d_{s}']).vmean
        vcount = (out_dict[f'bin1d_{s}']).vcount
        vmean = np.array([vmean[i] if vcount[i] >= 5
                          else np.nan for i in range(len(vmean))])

        plt.plot(vmean, y_array, '-', marker='o', c=dict_season()[f'color_{s}'],
                 label=dict_season()[f'name_{s}'])

        if errorbars: # add error bars
            vstdv = (out_dict[f'bin1d_{s}']).vstdv
            plt.errorbar(vmean, y_array, None, vstdv,
                         c=dict_season()[f'color_{s}'], elinewidth=0.5)

    plt.tick_params(direction='in', top=True, right=True)
    if note: plt.annotate(note, xy=(0.025, 0.925), xycoords='axes fraction',
                          bbox=dict(boxstyle="round", fc="w"))

    plt.ylim([min_y, max_y])
    plt.ylabel(y_label)
    plt.xlabel(f'{substance}') # [4:]
    if detr: # remove the 'delta_' and replace with symbol
        plt.xlabel('$\Delta $' + substance.split("_")[-1])

    plt.legend()
    plt.show()

#%% Fct calls - gradients
# if __name__=='__main__':
    # only calculate caribic if necessary
    # calc_c = False
    # if calc_c:
    #     if exists('caribic_dill.pkl'): # Avoid long file loading times
    #         with open('caribic_dill.pkl', 'rb') as f:
    #             caribic = dill.load(f)
    #         del f
    #     else: caribic = Caribic(range(1980, 2021), pfxs = ['GHG', 'INT', 'INT2'])

    # for subs in ['ch4', 'co2', 'sf6', 'n2o']:
    #     plot_gradient_by_season(caribic, subs,  tp='pvu', pvu = 2.0)

    # for subs in ['ch4', 'co2', 'sf6', 'n2o']:
    #     plot_gradient_by_season(caribic.sel_latitude(30, 90), subs, tp='z', pvu = 2.0, note='lat>30°N')

    # for subs in ['ch4', 'co2', 'sf6', 'n2o']:
    #     for tp in ['therm', 'dyn', 'pvu']:
    #         plot_gradient_by_season(caribic, subs,  c_pfx='INT2', tp=tp, pvu = 2.0)
