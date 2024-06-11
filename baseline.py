# -*- coding: utf-8 -*-
"""
@Author: Sophie Bauchinger, IAU
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

from toolpac.outliers import outliers # type: ignore
from toolpac.conv.times import datetime_to_fractionalyear # type: ignore

from dataTools.dictionaries import get_col_name, get_fct_substance
from dataTools.data._local import MaunaLoa

mlo_data = MaunaLoa().data

tp_kwargs = {
    'tp_def' : 'chem', 
    'crit' : 'n2o', # 
    'ref_obj' : mlo_data['n2o'],
    'c_pfx' : 'INT2',
    'detr' : True, 
    'coord' : 'pt',
    'pvu' : 3.5,
    }


#%% Baseline Filter
def baseline_filter(glob_obj, subs, subs_pfx='GHG',
                    direction='p', ol_limit=0.1,
                    save=True, plot=True, plot_strat=True, **tp_kwargs):
    """
    After sorting data into stratospheric and tropospheric, now sort the
    tropospheric data into outliers and non-outliers (in both directions)
    Parameters:
        
        glob_obj (GlobalData): Caribic
        subs (str): substances to flag, eg. 'n2o'
        subs_pfx (str): substance source data prefix, eg. 'GHG'
        ref_obj (LocalData): reference data for strat / trop sorting if needed
        direction (str): 'p', 'n', 'pn'
        save (bool): adds filtered data to glob_obj

        tropFilter_kwargs: 
            tp_def (str): 'chem', 'therm' or 'dyn'
            c_pfx (str): 'GHG', 
            crit (str): 'n2o', 'o3'
            coord (str): 'dp', 'pt', 'z'
            pvu (float): 1.5, 2.0, 3.5

    """
    crit = tp_kwargs.get('crit') # ['crit']
    state = f'filter_trop_outliers: subs={subs}, c_pfx={subs_pfx}, crit={crit}\n'
    if not glob_obj.source == 'Caribic': 
        raise Exception(state+'Need a Caribic instance.')

    # Tropospheric data
    tropo_obj = glob_obj.sel_tropo(**tp_kwargs)
    strato_obj = glob_obj.sel_strato(**tp_kwargs)

    if isinstance(subs, str): substances = [subs]
    elif isinstance(subs, list): substances = subs

    for subs in substances:
        substance = get_col_name(subs, glob_obj.source, subs_pfx)
        print('substance', substance)
        if substance is None:
            print(state+f'{subs.upper()} not found in {subs_pfx}'); continue

        #!!! need to make it so that substance df is only cut, not yeeted entirely
        tropo_obj.create_substance_df(subs)
        strato_obj.create_substance_df(subs)
    
        if subs not in tropo_obj.data.keys(): 
            raise KeyError('Filtering did not result in desired DataFrame being created.')
    
        data = tropo_obj.data[subs]
        data.sort_index(inplace=True)
    
        # Stratospheric data
        strato = strato_obj.data[subs]
        strato.sort_index(inplace=True)
    
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
            plot_BLfilter_time(subs, subs_pfx, data_flag, substance, strato, 
                               plot_strat, fit_result, **tp_kwargs)
        
        # no residual value for non-outliers
        data_flag.loc[data_flag[f'fl_{subs}'] == 0, f'ol_{subs}'] = np.nan

    return data_flag

def plot_BLfilter_time(subs, subs_pfx, data_flag, substance, strato, plot_strat, fit_result, ax=None, **tp_kwargs):
    tp_def = tp_kwargs['tp_def']; crit = tp_kwargs['crit']; pvu = tp_kwargs['pvu']; coord = tp_kwargs['coord']
    description = f'{tp_def.upper()}'
    if tp_def == 'chem': description += f' ({crit})'
    elif tp_def == 'dyn': description += f' ({pvu}, {coord})'
    else: description += f' ({coord})'
    
    if 'ax' == None or 'fig' not in locals():
        fig, ax = plt.subplots(dpi=200)
        plt.title(f'Baseline filter of {subs.upper()} in {subs_pfx} - ' + description)
    else: 
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
    if 'fig' in locals(): 
        fig.autofmt_xdate()
        plt.legend(); plt.show()