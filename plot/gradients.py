# -*- coding: utf-8 -*-
"""
@Author: Sophie Bauchimger, IAU
@Date: Fri Apr 28 14:13:28 2023

Plotting of gradients - wants detrended data, sorted into atmos. layers

"""
import dill
from os.path import exists
import numpy as np
import matplotlib.pyplot as plt

from toolpac.calc.binprocessor import Simple_bin_1d, Bin_equi1d #!!!

from detrend import detrend_substance
from dictionaries import get_col_name
from tools import subs_merge, make_season

#%% Plotting Gradient by season
# Fct definition in C_plot needed these:
# select_var=['fl_ch4','fl_sf6', 'fl_n2o'] # flagged data
# select_value=[0,0,0]
# select_cf=['GT','GT', 'GT'] # operators

# ptsmin (int): minimum number of pts for a bin to be considered #!!! implement
from data import Mauna_Loa
def plot_gradient_by_season(c_obj, subs, tp='therm', pvu = 2.0, errorbars=False,
                            bsize=None, use_detr=True, note=None):
    """
    Plotting gradient by season using 1D binned data. Detrended data used by default
    (Inspired by C_plot.pl_gradient_by_season)

    Parameters:
        c_obj (Caribic)
        subs (str): substance e.g. 'sf6'
        tp (str): tropopause definition
        pvu (float): potential vorticity for dyn. tp definition. 1.5, 2.0 or 3.5
        errorbars (bool)
        bsize (int): bin size for 1D binning (depends on coordinate)
        use_detr (bool)
        note (str): shown as text box on the plot
    """

    if not f'{subs}_data' in c_obj.data.keys():
        if use_detr: detrend_substance(c_obj, subs, Mauna_Loa(c_obj.years, subs))
        subs_merge(c_obj, subs, save=True, detr=True) # creates data[f'{subs}_data']

    data = c_obj.data[f'{subs}_data']
    substance = get_col_name(subs, c_obj.source, 'GHG')
    # if merged df exists, but without detrended data
    if use_detr and 'delta_'+substance not in data.columns:
        # create delta_{substance} column
        detrend_substance(c_obj, subs, Mauna_Loa(c_obj.years, subs))
        # re-create data[f'{subs}_data']
        subs_merge(c_obj, subs, save=True, detr=True)
    if use_detr: substance = 'delta_'+substance

    # Get column name for y axis depending on function parameter
    if tp == 'z':
        # height relative to the tropopause in km: H_rel_TP
        y_coord = 'int_CARIBIC2_H_rel_TP [km]'
        y_label = '$\Delta$z [km]'

    elif tp == 'pvu':
        # pot temp difference to potential vorticity surface
        y_coord = 'int_ERA5_D_{}_{}PVU_BOT [K]'.format(str(pvu)[0], str(pvu)[2])
        y_label = f'$\Delta\Theta$ ({pvu} PVU - ERA5) [K]'
    elif tp =='therm':
        #  potential T. difference relative to thermal tropopause from ECMWF
        y_coord = 'int_pt_rel_sTP_K [K]'
        y_label = f'$\Delta\Theta$ ({tp} - ECMWF) [K]'
    elif tp == 'dyn':
        #  potential T difference rel. to dynamical (PV=3.5PVU) tp from ECMWF
        y_coord = 'int_pt_rel_dTP_K [K]'
        y_label = f'$\Delta\Theta$ ({tp} - ECMWF) [K]'
    else:
        y_coord = 'p [mbar]'
        y_label = 'Pressure [mbar]'

    bsize_dict = {'z' : 0.25, 'pvu': 5, 'therm': 5, 'dyn': 5, None:40} # {y_coord : bsize}
    if not bsize: bsize = bsize_dict[tp]

    min_y, max_y = np.nanmin(data[y_coord].values), np.nanmax(data[y_coord].values)

    nbins = (max_y - min_y) / bsize
    y_array = min_y + np.arange(nbins) * bsize + bsize * 0.5

    data['season'] = make_season(data.index.month) # 1 = spring etc
    dict_season = {'name_1': 'MAM, spring', 'name_2': 'JJA, summer',
                   'name_3': 'SON, autumn', 'name_4': 'DJF, winter',
                   'color_1': 'blue', 'color_2': 'orange',
                   'color_3': 'green', 'color_4': 'red'}

    fig, ax = plt.subplots(dpi=200)

    for s in set(data['season'].tolist()):
        df = data.loc[data['season'] == s]
        y_values = df[y_coord].values # df[eq_lat_col].values #
        x_values = df[substance].values

        dict_season[f'bin1d_{s}'] = Simple_bin_1d(x_values, y_values,
                                                  Bin_equi1d(min_y, max_y, bsize))
        # bin_1d_2d.bin_1d(x_values, y_values, min_y, max_y, bsize)

        vmean = (dict_season[f'bin1d_{s}']).vmean
        vcount = (dict_season[f'bin1d_{s}']).vcount
        vmean = np.array([vmean[i] if vcount[i] >= 5
                          else np.nan for i in range(len(vmean))])

        plt.plot(vmean, y_array, '-', marker='o', c=dict_season[f'color_{s}'],
                 label=dict_season[f'name_{s}'])

        if errorbars: # add error bars
            vstdv = (dict_season[f'bin1d_{s}']).vstdv
            plt.errorbar(vmean, y_array, None, vstdv,
                         c=dict_season[f'color_{s}'], elinewidth=0.5)

    plt.tick_params(direction='in', top=True, right=True)
    if note: plt.annotate(note, xy=(0.025, 0.925), xycoords='axes fraction',
                          bbox=dict(boxstyle="round", fc="w"))

    plt.ylim([min_y, max_y])
    plt.ylabel(y_label)
    plt.xlabel(f'{substance}') # [4:]
    if use_detr: # remove the delta_
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
