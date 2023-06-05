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
from dictionaries import get_col_name, choose_column, get_coord_name

import C_tools

#%% Plotting Gradient by season
""" Original data needed these: """
# select_var=['fl_ch4','fl_sf6', 'fl_n2o'] # flagged data
# select_value=[0,0,0]
# select_cf=['GT','GT', 'GT'] # operators 

def plot_gradient_by_season(c_obj, subs, c_pfx = 'INT2', tp='therm', give_choice=False, 
                            errorbars=False, ycoord='pt', bsize=0.5, ptsmin=5):
    """ 
    Plotting gradient by season using 1D binned data 
    Parameters:
        c_obj: caribic object
        subs (str): substance e.g. 'sf6'
        tp (str): tropopause definition 
        bsize: int, bin size for 1D binning
        ptsmin: int, minimum number of pts for a bin to be considered 
    
    Re-implementation of C_plot.pl_gradient_by_season
    """
    data = c_obj.data[c_pfx]
    substance = get_col_name(subs, 'Caribic', c_pfx)
    if not get_col_name(subs, 'Caribic', c_pfx) or substance not in data.columns:
        if give_choice: substance = choose_column(data, subs)
        else: return
        if not substance: return

    detr = False # indicator on whether detrended data is used 
    if f'detr_{c_pfx}_{subs}' in c_obj.data.keys():
        detr_data = c_obj.data[f'detr_{c_pfx}_{subs}']
        print(substance)

        if 'detr_'+ substance  in detr_data.columns: 
            substance = 'detr_' + substance
            print(substance); detr = True

    # y-coordinate
    if c_pfx == 'INT2': # co, co2, ch4, n2o
        y_coord = 'int_CARIBIC2_H_rel_TP [km]' # height relative to the tropopause in km: H_rel_TP; replacement for H_rel_TP
        y_label = '$\Delta$z [km]'

    elif c_pfx =='INT': # co, co2, ch4
        if tp =='therm': y_coord = 'int_pt_rel_sTP_K [K]' #  potential temperature difference relative to thermal tropopause from ECMWF
        elif tp == 'dyn': y_coord = 'int_pt_rel_dTP_K [K]' #  potential temperature difference relative to  dynamical (PV=3.5PVU) tropopause from ECMWF
        y_label = f'$\Delta$T [K] ({tp})'

    else: 
        y_coord = get_coord_name('p', 'Caribic', c_pfx)
        y_label = y_coord

    min_y, max_y = np.nanmin(data[y_coord].values), np.nanmax(data[y_coord].values)

    nbins = (max_y - min_y) / bsize
    y_array = min_y + np.arange(nbins) * bsize + bsize * 0.5

    data['season'] = C_tools.make_season(data.index.month) # 1 = spring etc
    dict_season = {'name_1': 'MAM, spring', 'name_2': 'JJA, summer', 'name_3': 'SON, autumn', 'name_4': 'DJF, winter',
                   'color_1': 'blue', 'color_2': 'orange', 'color_3': 'green', 'color_4': 'red'}

    fig, ax = plt.subplots(dpi=200)

    x_min = np.nan
    x_max = np.nan

    for s in set(data['season'].tolist()):
        df = data.loc[data['season'] == s]
        y_values = df[y_coord].values # df[eq_lat_col].values # 
        if detr: df = detr_data.loc[data['season'] == s]

        x_values = df[substance].values

        if detr: x_values -= detr_data[substance].iloc[0] # subtract value. not the best imprementation ngl

        dict_season[f'bin1d_{s}'] = bin_1d_2d.bin_1d(x_values, y_values, min_y, max_y, bsize)


    # for s in set(data['season'].tolist()): # using the set to order the seasons 
        vmean = (dict_season[f'bin1d_{s}']).vmean
        vcount = (dict_season[f'bin1d_{s}']).vcount
        vmean = np.array([vmean[i] if vcount[i] >= 5 else np.nan for i in range(len(vmean))])
        
        # find value range for axis limits
        # all_vmin = np.nanmin((dict_season[f'bin1d_{s}']).vmin)
        # all_vmax = np.nanmax((dict_season[f'bin1d_{s}']).vmax)
        # x_min = np.nanmin((x_min, all_vmin))
        # x_max = np.nanmax((x_min, all_vmax))

        if c_pfx == 'GHG' or c_pfx == 'INT': 
            plt.scatter(vmean, y_array, marker='o', c=dict_season[f'color_{s}'], label=dict_season[f'name_{s}'])
        else:
            plt.plot(vmean, y_array, '-',
                     marker='o', c=dict_season[f'color_{s}'], label=dict_season[f'name_{s}'])

        # add error bars
        if errorbars:
            vstdv = (dict_season[f'bin1d_{s}']).vstdv
            plt.errorbar(vmean, y_array, None, vstdv, c=dict_season[f'color_{s}'], elinewidth=0.5)

    plt.tick_params(direction='in', top=True, right=True)

    plt.ylim([min_y, max_y])
    plt.ylabel(y_label)
    plt.xlabel(f'{substance}') # [4:]
    if detr: plt.xlabel('$\Delta$' + substance.split("_")[-1])

    x_min, x_max = np.floor(x_min), np.ceil(x_max)
    plt.xlim([x_min, x_max])
    # ax.hlines(0, x_min, x_max, color='gray', ls='dashed', label = 'Thermal tropopause', zorder=2, lw=0.5) 
    plt.legend()
    plt.show()

#%% Fct calls 
if __name__=='__main__':
    calc_caribic = False
    if calc_caribic: 
        caribic = Caribic(range(2005, 2021), pfxs=['INT2'], subst = 'n2o')

    c_n2o_col = get_col_name('n2o', source='Caribic', c_pfx = 'INT2')

    ref_data = Mauna_Loa(range(2005, 2020), 'n2o')
    c_df_detr = detrend_substance(caribic, 'n2o', ref_data)

    for subs in ['o3', 'noy', 'co2', 'ch4', 'no', 'co', 'n2o']:
        plot_gradient_by_season(caribic, subs)


#%% Old definition
# #%% Plotting Gradient by season
# """ What data needs to be put in here? """
# # select_var=['fl_ch4','fl_sf6', 'fl_n2o'] # flagged data
# # select_value=[0,0,0]
# # select_cf=['GT','GT', 'GT'] # operators 

# def plot_gradient_by_season(data, substance, c_pfx = 'INT2',
#                             tp='therm', errorbars=False, ycoord='pt', 
#                             bsize=0.5, ptsmin=5):
#     """ 
#     Plotting gradient by season using 1D binned data 
#     Parameters:
#         data: pandas (geo)dataframe, detrended
#         substance: str, eg. 'SF6 [ppt]'
#         tropopause: str, which tropopause definition to use 
#         min_y, max_y: int, defines y range to plot
#         bsize: int, bin size for 1D binning
#         ptsmin: int, minimum number of pts for a bin to be considered 
    
#     Re-implementation of C_plot.pl_gradient_by_season
#     """
#     substance = get_col_name(substance,'Caribic', c_pfx)
#     if not get_col_name(substance, 'Caribic', c_pfx) or substance not in data.columns:
#         substance = choose_column(data, substance)

#     detr = False
#     if 'detr_'+ substance  in data.columns: 
#         substance = 'detr_' + substance
#         print(substance)
#         detr = True

#     # height relative to tropopauses
#     if 'int_pt_rel_dTP_K [K]' in data.columns: H_rel_TP = 'int_pt_rel_dTP_K [K]'
#     elif 'int_CARIBIC2_H_rel_TP [km]' in data.columns: H_rel_TP = 'int_CARIBIC2_H_rel_TP [km]'
#     else: 
#         try: H_rel_TP = get_coord_name('h_rel_tp', 'Caribic', c_pfx)
#         except: H_rel_TP = choose_column(data, 'h_rel_tp')

#     min_y, max_y = np.nanmin(data[H_rel_TP].values), np.nanmax(data[H_rel_TP].values)

#     nbins = (max_y - min_y) / bsize
#     y_array = min_y + np.arange(nbins) * bsize + bsize * 0.5

#     data['season'] = C_tools.make_season(data.index.month) # 1 = spring etc
#     dict_season = {'name_1': 'MAM, spring', 'name_2': 'JJA, summer', 'name_3': 'SON, autumn', 'name_4': 'DJF, winter',
#                    'color_1': 'blue', 'color_2': 'orange', 'color_3': 'green', 'color_4': 'red'}

#     for s in set(data['season'].tolist()):
#         df = data.loc[data['season'] == s]
#         y_values = df[H_rel_TP].values # df[eq_lat_col].values # 
#         x_values = df[substance].values
#         dict_season[f'bin1d_{s}'] = bin_1d_2d.bin_1d(x_values, y_values, min_y, max_y, bsize)

#     fig, ax = plt.subplots(dpi=200)

#     x_min = np.nan
#     x_max = np.nan
#     for s in set(data['season'].tolist()): # using the set to order the seasons 
#         vmean = (dict_season[f'bin1d_{s}']).vmean
#         vcount = (dict_season[f'bin1d_{s}']).vcount
#         vmean = np.array([vmean[i] if vcount[i] >= 5 else np.nan for i in range(len(vmean))])
        
#         # find value range for axis limits
#         all_vmin = np.nanmin((dict_season[f'bin1d_{s}']).vmin)
#         all_vmax = np.nanmax((dict_season[f'bin1d_{s}']).vmax)
#         x_min = np.nanmin((x_min, all_vmin))
#         x_max = np.nanmax((x_min, all_vmax))

#         plt.plot(vmean, y_array, '-',
#                  marker='o', c=dict_season[f'color_{s}'], label=dict_season[f'name_{s}'])

#         # add error bars
#         if errorbars:
#             vstdv = (dict_season[f'bin1d_{s}']).vstdv
#             plt.errorbar(vmean, y_array, None, vstdv, c=dict_season[f'color_{s}'], elinewidth=0.5)

#     plt.tick_params(direction='in', top=True, right=True)

#     plt.ylim([min_y, max_y])
#     plt.ylabel('$\Delta \Theta$) [K]')
#     plt.xlabel(f'{substance[4:]}')
#     if detr: plt.xlabel('$\Delta$' + substance.split("_")[-1])

#     x_min, x_max = np.floor(x_min), np.ceil(x_max)
#     plt.xlim([x_min, x_max])
#     # ax.hlines(0, x_min, x_max, color='gray', ls='dashed', label = 'Thermal tropopause', zorder=2, lw=0.5) 
#     plt.legend()
#     plt.show()

# #%% Fct calls 
# if __name__=='__main__':
#     caribic_int2 = Caribic(range(2005, 2021), pfxs=['INT2'], subst = 'n2o')
#     c_n2o_col = get_col_name('n2o', source='Caribic', c_pfx = 'INT2')

#     ref_data = Mauna_Loa(range(2005, 2020), 'n2o').df
#     c_df_detr = detrend_substance(caribic_int2.df, c_n2o_col, ref_data, 'N2OcatsMLOm')

#     for subs in ['o3', 'noy', 'co2', 'ch4', 'no', 'co', 'n2o']:
#         plot_gradient_by_season(c_df_detr, subs)
