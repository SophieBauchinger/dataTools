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
from dictionaries import get_col_name
from aux_fctns import subs_merge, make_season

#%% Plotting Gradient by season
""" Fct definition in C_plot needed these: """
# select_var=['fl_ch4','fl_sf6', 'fl_n2o'] # flagged data
# select_value=[0,0,0]
# select_cf=['GT','GT', 'GT'] # operators 

def plot_gradient_by_season(c_obj, subs, c_pfx = 'INT2', tp='therm', pvu = 2.0, give_choice=False, 
                            errorbars=False, bsize=None, ptsmin=5, use_detr=True):
    """ 
    Plotting gradient by season using 1D binned data. Detrended data used by default
    Parameters:
        c_obj (Caribic): obj containing data
        subs (str): substance e.g. 'sf6'
        tp (str): tropopause definition 
        bsize (int) bin size for 1D binning
        ptsmin (int): minimum number of pts for a bin to be considered 

    Re-implementation of C_plot.pl_gradient_by_season
    """
    if not f'{subs}_data' in c_obj.data.keys():
        detrend_substance(caribic, subs, Mauna_Loa(c_obj.years, subs))
        subs_merge(c_obj, subs, save=True, detr=True)

    data = c_obj.data[f'{subs}_data']
    substance = get_col_name(subs, c_obj.source, 'GHG')
    if use_detr and 'delta_'+substance not in data.columns: 
        detrend_substance(c_obj, subs, Mauna_Loa(c_obj.years, subs))
        subs_merge(c_obj, subs, save=True, detr=True)
    if use_detr: 
        substance = 'delta_'+substance
        detr = True
    # except: 
    #     if c_obj.source == 'Caribic': 
    #         data = c_obj.data[c_pfx]
    #         substance = get_col_name(subs, c_obj.source, c_pfx)
    #     elif c_obj.source == 'Mozart': data = c_obj.df
    #     else: raise('Could not find the specified data set. Check your input')

    # Get column name for y axis depending on function parameter     
    if tp == 'z':
        y_coord = 'int_CARIBIC2_H_rel_TP [km]' # height relative to the tropopause in km: H_rel_TP; replacement for H_rel_TP
        y_label = '$\Delta$z [km]'
    elif tp == 'pvu':
        y_coord = 'int_ERA5_D_{}_{}PVU_BOT [K]'.format(str(pvu)[0], str(pvu)[2]) # pot temp difference to potential vorticity surface
        y_label = f'$\Delta\Theta$ ({pvu} PVU - ERA5) [K]'
    elif tp =='therm': 
        y_coord = 'int_pt_rel_sTP_K [K]' #  potential temperature difference relative to thermal tropopause from ECMWF
        y_label = f'$\Delta\Theta$ ({tp} - ECMWF) [K]'
    elif tp == 'dyn': 
        y_coord = 'int_pt_rel_dTP_K [K]' #  potential temperature difference relative to  dynamical (PV=3.5PVU) tropopause from ECMWF
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
    dict_season = {'name_1': 'MAM, spring', 'name_2': 'JJA, summer', 'name_3': 'SON, autumn', 'name_4': 'DJF, winter',
                   'color_1': 'blue', 'color_2': 'orange', 'color_3': 'green', 'color_4': 'red'}

    fig, ax = plt.subplots(dpi=200)

    for s in set(data['season'].tolist()):
        df = data.loc[data['season'] == s]
        y_values = df[y_coord].values # df[eq_lat_col].values # 
        x_values = df[substance].values

        dict_season[f'bin1d_{s}'] = bin_1d_2d.bin_1d(x_values, y_values, min_y, max_y, bsize)

        vmean = (dict_season[f'bin1d_{s}']).vmean
        vcount = (dict_season[f'bin1d_{s}']).vcount
        vmean = np.array([vmean[i] if vcount[i] >= 5 else np.nan for i in range(len(vmean))])
        
        plt.plot(vmean, y_array, '-', marker='o', c=dict_season[f'color_{s}'], label=dict_season[f'name_{s}'])

        if errorbars: # add error bars
            vstdv = (dict_season[f'bin1d_{s}']).vstdv
            plt.errorbar(vmean, y_array, None, vstdv, c=dict_season[f'color_{s}'], elinewidth=0.5)

    plt.tick_params(direction='in', top=True, right=True)

    plt.ylim([min_y, max_y])
    plt.ylabel(y_label)
    plt.xlabel(f'{substance}') # [4:]
    if detr: plt.xlabel('$\Delta $' + substance.split("_")[-1]) # removing the delta_

    plt.legend()
    plt.show()

#%% Fct calls 
if __name__=='__main__':    
    calc_caribic = False
    if calc_caribic: 
        caribic = Caribic(range(2005, 2021), pfxs=['INT2'], subst = 'n2o')

    for subs in ['ch4', 'co2', 'sf6', 'n2o']:
        for tp in ['therm', 'dyn', 'pvu']:
            plot_gradient_by_season(caribic, subs,  c_pfx='INT2', tp=tp, pvu = 2.0)


#%% Müll

    # coord_data = c_obj.data[c_pfx]
    # data = c_obj.data[c_pfx]

    # substance = get_col_name(subs, 'Caribic', c_pfx)
    # if not get_col_name(subs, 'Caribic', c_pfx) or substance not in data.columns:
    #     if give_choice: substance = choose_column(data, subs)
    #     else: return
    #     if not substance: return

    # detr = False # indicator on whether detrended data is used 
    # if use_detr: 
    #     if f'detr_{c_pfx}_{subs}' not in c_obj.data.keys(): 
    #         detrend_substance(c_obj, subs, Mauna_Loa(c_obj.years, subs))
    #     if f'detr_{c_pfx}_{subs}' in c_obj.data.keys():
    #         detr_data = c_obj.data[f'detr_{c_pfx}_{subs}']
    #         if 'detr_'+ substance  in detr_data.columns: 
    #             substance = 'detr_' + substance
    #             detr=True
    #     else: print('Could not generate detrended data, Please check your input')
    
    # # y-coordinate
    # if c_pfx == 'INT2': # co, co2, ch4, n2o
    #     if tp == 'height':
    #         y_coord = 'int_CARIBIC2_H_rel_TP [km]' # height relative to the tropopause in km: H_rel_TP; replacement for H_rel_TP
    #         y_label = '$\Delta$z [km]'
    #     if tp == 'pvu':
    #         y_coord = 'int_ERA5_D_{}_{}PVU_BOT [K]'.format(str(pvu)[0], str(pvu)[2]) # pot temp difference to potential vorticity surface
    #         y_label = f'$\Delta$T wrt. {pvu} PV surface [K]'

    # elif c_pfx =='INT': # co, co2, ch4
    #     if tp =='therm': y_coord = 'int_pt_rel_sTP_K [K]' #  potential temperature difference relative to thermal tropopause from ECMWF
    #     elif tp == 'dyn': y_coord = 'int_pt_rel_dTP_K [K]' #  potential temperature difference relative to  dynamical (PV=3.5PVU) tropopause from ECMWF
    #     y_label = f'$\Delta$T [K] ({tp})'

    # else: 
    #     y_coord = get_coord_name('p', 'Caribic', c_pfx)
    #     y_label = y_coord
    
        # find value range for axis limits
        # all_vmin = np.nanmin((dict_season[f'bin1d_{s}']).vmin)
        # all_vmax = np.nanmax((dict_season[f'bin1d_{s}']).vmax)
        # x_min = np.nanmin((x_min, all_vmin))
        # x_max = np.nanmax((x_min, all_vmax))
        
        # plt.scatter(vmean, y_array, marker='o', c=dict_season[f'color_{s}'], label=dict_season[f'name_{s}'])

        # if c_pfx == 'GHG' or c_pfx == 'INT': 
        #     plt.scatter(vmean, y_array, marker='o', c=dict_season[f'color_{s}'], label=dict_season[f'name_{s}'])
        # else:
            # plt.plot(vmean, y_array, '-',
            #           marker='o', c=dict_season[f'color_{s}'], label=dict_season[f'name_{s}'])