# -*- coding: utf-8 -*-
"""
@Author: Sophie Bauchimger, IAU
@Date: Fri Apr 28 14:13:28 2023

Plotting of gradients - wants detrended data, sorted into atmos. layers

"""
import numpy as np
import matplotlib.pyplot as plt

import toolpac.calc.binprocessor as bp

from dictionaries import get_col_name, dict_season, choose_column
from tools import make_season, coordinate_tools

#%% Plotting Gradient by season
# Fct definition in C_plot needed these:
# select_var=['fl_ch4','fl_sf6', 'fl_n2o'] # flagged data
# select_value=[0,0,0]
# select_cf=['GT','GT', 'GT'] # operators

# ptsmin (int): minimum number of pts for a bin to be considered #!!! implement

# def plot_gradient_by_season(c_obj, subs, c_pfx = None, tp_def='therm', pvu = 3.5, errorbars=False,
#                             detr=True, note=None, ycoord='pt', y_bin=None):
    # """
    # Plotting gradient by season using 1D binned data. Detrended data used by default
    # (Inspired by C_plot.pl_gradient_by_season)

    # Parameters:
    #     c_obj (Caribic)
    #     subs (str): substance e.g. 'sf6'
    #     c_pfx (str): 'GHG', 'INT', 'INT2'
    #     tp_def (str): tropopause definition
    #     pvu (float): potential vorticity for dyn. tp definition. 1.5, 2.0 or 3.5
    #     errorbars (bool)
    #     y_bin (int): bin size for 1D binning (depends on coordinate)
    #     detr (bool)
    #     note (str): shown as text box on the plot
    # """

def plot_gradient_by_season(c_obj, subs, x_params = {}, y_params = {},
                            detr=True, note=None, errorbars=False):
    """
    Plotting gradient by season using 1D binned data. Detrended data used by default
    (Inspired by C_plot.pl_gradient_by_season)

    x_params (dict):
        keys: x_pfx
    y_params (dict):
        keys: ycoord, tp_def, y_pfx, pvu
    """
    if (not all(i in x_params.keys() for i in  ['x_pfx'])
                or not all(i in y_params.keys() for i in ['ycoord', 'y_pfx', 'tp_def'])):
        raise KeyError('Please supply all necessary parameters: x_pfx, (xcoord) / ycoord, tp_def, y_pfx, (pvu)')
    if not subs in c_obj.data.keys(): c_obj.create_substance_df(subs)
    data = c_obj.data[subs]

    # x-axis
    substance = get_col_name(subs, c_obj.source, x_params['x_pfx'])
    if detr: # detrended data for x-axis
        substance = f'detr_{substance}'
        if not substance in data.columns:
            raise ValueError(f'Detrended data not available for {subs.upper()}')

    # y-axis
    y_coord, y_label = coordinate_tools(**y_params)
    y_bins = {'z' : 0.5, 'pt' : 10, 'p' : 40}
    if not 'y_bin' in y_params.keys():
        y_bin = y_bins[y_params['ycoord']]
    else: y_bin = y_params['y_bin']

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
    plt.xlabel(f'{substance}')
    if detr: # remove the 'delta_' and replace with symbol
        plt.xlabel('$\Delta $' + substance.split("_")[-1] + ' wrt. 2005')

    plt.legend()
    plt.show()

#%% Fct calls - gradients

if __name__ == '__main__':
    if False: caribic = True # BS to avoid error

    yp1 = {'tp_def' : 'chem',
           'y_pfx' : 'INT2',
           'ycoord' : 'z'}

    yp2 = {'tp_def' : 'therm',
           'y_pfx' : 'INT',
           'ycoord' : 'pt'}

    yp3 = {'tp_def' : 'therm',
           'y_pfx' : 'INT2',
           'ycoord' : 'pt'}

    yp4 = {'tp_def' : 'dyn',
           'y_pfx' : 'INT',
           'ycoord' : 'pt',
           'pvu' : 3.5}

    for subs in ['sf6']: #:, 'n2o', 'co2', 'ch4']:
        for yp in [yp1, yp2, yp3, yp4]:
            x_params = {'x_pfx' : 'GHG'}
            y_params = yp
            plot_gradient_by_season(caribic.sel_latitude(30, 90), subs, x_params, y_params, note='lat>30°N')
