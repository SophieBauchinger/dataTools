# -*- coding: utf-8 -*-
"""
@Author: Sophie Bauchimger, IAU
@Date: 

using fct from 'outliers' created on Tue Apr 11 09:28:22 2023

"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# default='warn' - otherwise df[j] = val gives a warning (outliers.outliers)
pd.options.mode.chained_assignment = None
# supress a gui backend userwarning
import warnings; warnings.filterwarnings("ignore", category=UserWarning,
                                         module='matplotlib')

from toolpac.outliers import outliers
from toolpac.conv.times import datetime_to_fractionalyear

from dictionaries import get_col_name, get_fct_substance

#%% Baseline Filter
def baseline_filter(glob_obj, subs='co2', c_pfx='GHG', tp_def='chem', crit='n2o', 
                    coord='pt', pvu=3.5, direction='p', ref_obj=None, save=True, 
                    plot=True, plot_strat=True, ol_limit=0.1, **kwargs):
    """
    After sorting data into stratospheric and tropospheric, now sort the
    tropospheric data into outliers and non-outliers (in both directions)
    Parameters:
        glob_obj (GlobalData): Caribic
        subs (str): substances to flag, eg. 'n2o'
        tp_def (str): tropopause definition: 'chem', 'dyn', 'therm'
        c_pfx (str): caribic data prefix, eg. 'GHG'
        crit (str): 'n2o', 'o3'
        coord (str): 'dp', 'pt', 'z'
        pvu (float): 1.5, 2.0, 3.5
        direction (str): 'p', 'n', 'pn'
        ref_obj (LocalData): reference data for strat / trop sorting if needed
        save (bool): adds filtered data to glob_obj
    """
    state = f'filter_trop_outliers: subs={subs}, c_pfx={c_pfx}, crit={crit}\n'
    if not glob_obj.source == 'Caribic': 
        raise Exception(state+'Need a Caribic instance.')

    # Tropospheric data
    tropo_obj = glob_obj.sel_tropo(tp_def, **kwargs)
    strato_obj = glob_obj.sel_strato(tp_def, **kwargs)

    if c_pfx not in tropo_obj.pfxs: 
        raise KeyError('Filtering did not result in desired cols being created.')

    data = tropo_obj.data[c_pfx] 
    data.sort_index(inplace=True)

    # Stratospheric data
    strato = strato_obj.data[c_pfx]
    strato.sort_index(inplace=True)

    if isinstance(subs, str): substances = [subs]
    elif isinstance(subs, list): substances = subs

    for subs in substances:
        substance = get_col_name(subs, glob_obj.source, c_pfx)
        if substance not in data.columns:
            print(state+f'{substance} not found in {c_pfx}'); continue
        
        data_flag = pd.DataFrame(data, columns=['Flight number', f'{substance}'])
        time = np.array(datetime_to_fractionalyear(data.index, method='exact'))
        mxr = data[substance].tolist()
        if f'd_{substance}' in data.columns:
            d_mxr = data[f'd_{substance}'].tolist()
        else: d_mxr = None # integrated values of high resolution data
        
        func = get_fct_substance(subs)
        tmp = outliers.find_ol(func, time, mxr, d_mxr, flag=None, # here
                               direction=direction, verbose=False,
                               plot=not(plot), limit=ol_limit, ctrl_plots=not(plot))
        
        data_flag[f'fl_{subs}'] = tmp[0]  # flag
        data_flag[f'ol_{subs}'] = tmp[1]  # residual
        # data_flag[f'ol_rel_TMP_{subs}'] = tmp[2] # warning
        fit_result = [func(t, *tmp[3]) for t in time] # popt1
        data_flag[f'ol_rel_{subs}'] = data_flag[f'ol_{subs}'] / fit_result # residuals
        
        if plot: 
            plot_BLfilter_time(subs, tp_def, crit, coord, pvu, c_pfx, data_flag, 
                                   substance, strato, plot_strat, fit_result, **kwargs)
        
        # no residual value for non-outliers
        data_flag.loc[data_flag[f'fl_{subs}'] == 0, f'ol_{subs}'] = np.nan

    return data_flag

def plot_BLfilter_time(subs, tp_def, crit, coord, pvu, c_pfx, data_flag, 
                       substance, strato, plot_strat, fit_result, **kwargs):
    description = f'{tp_def.upper()}'
    if tp_def == 'chem': description += f' ({crit})'
    elif tp_def == 'dyn': description += f' ({pvu}, {coord})'
    else: description += f' ({coord})'
    
    if 'ax' not in kwargs.keys(): 
        fig, ax = plt.subplots(dpi=200)
        plt.title(f'Baseline filter of {subs.upper()} in {c_pfx} - ' + description)
    else: 
        ax = kwargs['ax']
        ax.set_title(description)

    ax.scatter(data_flag[data_flag[f'fl_{subs}'] != 0].index, 
               data_flag[data_flag[f'fl_{subs}'] != 0][substance],
                c='xkcd:orange',  marker='.', zorder=1, label='extr. events')
    ax.scatter(data_flag[data_flag[f'fl_{subs}'] == 0].index, 
               data_flag[data_flag[f'fl_{subs}'] == 0][substance],
                c='xkcd:blue',  marker='.', zorder=1, label='base')
    if plot_strat:
        ax.scatter(strato.index, strato[substance],
                    c='xkcd:grey', label='strato', 
                    zorder=0,  marker='.')
    ax.plot(data_flag.index, fit_result, 
            label='baseline', c='k', zorder=2)
    ax.set_ylabel(substance)
    ax.set_xlabel('Time delta')
    if 'ax' not in kwargs.keys(): 
        fig.autofmt_xdate()
        plt.legend(); plt.show()