# -*- coding: utf-8 -*-
"""
@Author: Sophie Bauchimger, IAU
@Date: Fri Apr 28 14:13:28 2023

Plotting of gradients - wants detrended data, sorted into atmos. layers

"""
import numpy as np
import matplotlib.pyplot as plt

from toolpac.calc import bin_1d_2d

from data_classes import Caribic, Mauna_Loa
from detrend import detrend_substance

import C_tools

#%% Plotting Gradient by season
""" What data needs to be put in here? """
# select_var=['fl_ch4','fl_sf6', 'fl_n2o'] # flagged data
# select_value=[0,0,0]
# select_cf=['GT','GT', 'GT'] # operators 

def plot_gradient_by_season(data, substance, tropopause='therm', errorbars=False, 
                          min_y=-50, max_y=80, bsize=10, ptsmin=5):
    """ 
    Plotting gradient by season using 1D binned data 
    Parameters:
        data: pandas (geo)dataframe, detrended
        substance: str, eg. 'SF6 [ppt]'
        tropopause: str, which tropopause definition to use 
        min_y, max_y: int, defines longitude range to plot
        bsize: int, bin size for 1D binning
        ptsmin: int, minimum number of pts for a bin to be considered 
    
    Re-implementation of C_plot.pl_gradient_by_season
    """
    # c_obs = data[substance].values
    # t_obs =  np.array(datetime_to_fractionalyear(data.index, method='exact'))

    nbins = (max_y - min_y) / bsize
    y_array = min_y + np.arange(nbins) * bsize + bsize * 0.5

    data['season'] = C_tools.make_season(data.index.month) # 1 = spring etc
    dict_season = {'name_1': 'spring-MAM', 'name_2': 'summer-JJA', 'name_3': 'autumn-SON', 'name_4': 'winter-DJF',
                   'color_1': 'blue', 'color_2': 'orange', 'color_3': 'green', 'color_4': 'red'}

    for s in set(data['season'].tolist()):
        df = data.loc[data['season'] == s]

        y_values = df.geometry.y # df[f'int_pt_rel_thermtp_k'].values # equivalent latitude
        x_values = df[f'detr_{substance}'].values
        dict_season[f'bin1d_{s}'] = bin_1d_2d.bin_1d(x_values, y_values, min_y, max_y, bsize)

    plt.figure(dpi=200)

    x_min = np.nan
    x_max = np.nan
    for s in set(data['season'].tolist()): # using the set to order the seasons 
        vmean = (dict_season[f'bin1d_{s}']).vmean
        vcount = (dict_season[f'bin1d_{s}']).vcount
        vmean = np.array([vmean[i] if vcount[i] >= 5 else np.nan for i in range(len(vmean))])
        
        # find value range for axis limits
        all_vmin = np.nanmin((dict_season[f'bin1d_{s}']).vmin)
        all_vmax = np.nanmax((dict_season[f'bin1d_{s}']).vmax)
        x_min = np.nanmin((x_min, all_vmin))
        x_max = np.nanmax((x_min, all_vmax))

        plt.plot(vmean, y_array, '-',
                 marker='o', c=dict_season[f'color_{s}'], label=dict_season[f'name_{s}'])

        # add error bars
        if errorbars:
            vstdv = (dict_season[f'bin1d_{s}']).vstdv
            plt.errorbar(vmean, y_array, None, vstdv, c=dict_season[f'color_{s}'], elinewidth=0.5)

    plt.tick_params(direction='in', top=True, right=True)

    plt.ylim([min_y, max_y])

    x_min = np.floor(x_min)
    x_max = np.ceil(x_max)
    plt.xlim([x_min, x_max])
    plt.legend()
    plt.show()

#%% Plot gradient by Season
if __name__=='__main__':
    c_df = Caribic(range(2005, 2021)).df
    ref_data = Mauna_Loa(range(2005, 2020)).df
    plot_gradient_by_season(c_df, 'SF6 [ppt]')

    # same result for detrend bc we're looking at the gradient
    c_df_detr = detrend_substance(c_df, 'SF6 [ppt]', ref_data, 'SF6catsMLOm')
    plot_gradient_by_season(c_df_detr, 'SF6 [ppt]')