# -*- coding: utf-8 -*-
"""
@Author: Sophie Bauchimger, IAU
@Date: Tue Apr 11 09:28:22 2023

Calculating statistics for 

Calculate tropospheric non-baseline statistic 

"""
import numpy as np
import matplotlib.pyplot as plt
import dill
from os.path import exists
import pandas as pd
# default='warn' - otherwise df[j] = val gives a warning (outliers.outliers)
pd.options.mode.chained_assignment = None
# supress a gui backend userwarning
import warnings; warnings.filterwarnings("ignore", category=UserWarning,
                                         module='matplotlib')

from inspect import currentframe, getframeinfo

from toolpac.outliers import outliers
from toolpac.outliers import ol_fit_functions as fct
from toolpac.outliers.outliers import get_no_nan, fit_data
from toolpac.conv.times import datetime_to_fractionalyear

from tools import get_lin_fit, pre_flag
from dictionaries import get_fct_substance, get_col_name, substance_list

#!!! TO-DO:
    # implement different trop / strat filter to be used to extract trop data only

def baseline_filter(glob_obj, subs, c_pfx='GHG', crit='n2o', direction='p',
                         ref_obj=None, save=True, plot=True, plot_strat=False,
                         ax=None):
    """
    After sorting data into stratospheric and tropospheric, now sort the
    tropospheric data into outliers and non-outliers (in both directions)
    Parameters:
        glob_obj (GlobalData): e.g. caribic
        subs (str): substances to flag, eg. 'n2o'
        c_pfx (str): caribic data prefix, eg. 'GHG'
        crit (str): criterion for strat / trop sorting (eg. 'n2o')
        ref_obj (LocalData): reference data for strat / trop sorting if needed
        save (bool): adds filtered data to glob_obj
    """
    state = f'filter_trop_outliers: subs={subs}, c_pfx={c_pfx}, crit={crit}\n'
    # check availability of tropospheric filtered datasets
    if not glob_obj.source == 'Caribic': 
        raise Exception(state+'Please supply an instance of CARIBIC data.')
    else:
        trop_crits = [x for x in glob_obj.data[c_pfx].columns if x.startswith('tropo')]
        if len(trop_crits) > 0 and f'tropo_{crit}' in glob_obj.data[c_pfx].columns:
            data = glob_obj.data[c_pfx]
        elif len(trop_crits) > 0 and f'tropo_{crit}' not in glob_obj.data[c_pfx].columns:
            try: 
                data = filter_strat_trop(glob_obj, ref_obj=ref_obj, crit=crit, 
                                         c_pfx=c_pfx, plot=False)
            except: 
                data = glob_obj.data[c_pfx]
                crit = trop_crits[0][5:]
                print(state+f'Cannot use chosen crit, so using {crit} instead')
        else: 
            data = filter_strat_trop(glob_obj, ref_obj=ref_obj, crit=crit, 
                                     c_pfx=c_pfx, plot=False)
        # now take only data that has been flagged as tropospheric 
        strato = data[data[f'strato_{crit}'] == True]
        data = data[data[f'tropo_{crit}'] == True]
    
    if isinstance(subs, str): subs = [subs]
    for subs in subs:
        substance = get_col_name(subs, glob_obj.source, c_pfx)
        if substance not in data.columns:
            raise Exception(state+f'{substance} not found in {glob_obj.source} {c_pfx}')
    
        # check for valid data for that substance
        if len(get_no_nan(data.index, data[substance], data[substance])[0]) < 1:
            raise Exception(state+f'No non-nan {subs} data')
    
        # Get outlier fit function
        try: func = get_fct_substance(subs)
        except:
            print(f'No fit function found for {subs}. Using 2nd order poly with simple harm. {c_pfx}')
            func = fct.simple
    
        # create output dataframe, initiate values
        data_flag = pd.DataFrame(data, columns=['Flight number', f'{substance}',
                                                f'strato_{crit}', f'tropo_{crit}'])
        data_flag[f'ol_{subs}'] = np.nan # outlier
        data_flag[f'ol_rel_{subs}'] = np.nan # residual
        # data_flag[f'fl_{subs}'] = 0 # flag
        # flag = data_flag[f'fl_{subs}'].tolist()
    
        time = np.array(datetime_to_fractionalyear(data.index, method='exact'))
        mxr = data[substance].tolist()
        if f'd_{substance}' in data.columns:
            d_mxr = data[f'd_{substance}'].tolist()
        else: d_mxr = None # integrated values of high resolution data
    
        tmp = outliers.find_ol(func, time, mxr, d_mxr, flag=None, 
                               direction=direction, verbose=False,
                               plot=not(plot), limit=0.1, ctrl_plots=not(plot))

        data_flag[f'fl_{subs}'] = tmp[0]  # flag
        data_flag[f'ol_{subs}'] = tmp[1]  # residual
        data_flag[f'ol_rel_TMP_{subs}'] = tmp[2] # warning
        fit_result = [func(t, *tmp[3]) for t in time] # popt1
        data_flag[f'ol_rel_{subs}'] = data_flag[f'ol_{subs}'] / fit_result # residuals
    
        if plot: 
            if not ax: 
                fig, ax = plt.subplots(dpi=200)
                plt.title(f'{state}')
            ax.scatter(data_flag[data_flag[f'fl_{subs}'] != 0].index, 
                       data_flag[data_flag[f'fl_{subs}'] != 0][substance],
                        c='xkcd:orange',  marker='.', zorder=1, label='pollution')
            ax.scatter(data_flag[data_flag[f'fl_{subs}'] == 0].index, 
                       data_flag[data_flag[f'fl_{subs}'] == 0][substance],
                        c='xkcd:grey',  marker='.', zorder=1, label='background')
            ax.plot(data_flag.index, fit_result, 
                    label='background fit', c='k')
            if plot_strat:
                ax.scatter(strato.index, strato[substance],
                            c='xkcd:baby blue', lw=0.5, label=f'stratospheric ({crit})', 
                            zorder=0,  marker='+')
            ax.set_ylabel(substance)
            ax.set_xlabel('Time delta')
            if not ax: plt.legend(); plt.show()
    
        # no residual value for non-outliers
        # data_flag.loc[data_flag[f'fl_{subs}'] == 0, f'ol_{subs}'] = np.nan

    return data_flag

#%% Fctn calls - outliers
# if __name__=='__main__':
#     year_range = np.arange(2000, 2020)

#     calc_c = False
#     if calc_c: # only calculate caribic if necessary
#         if exists('caribic_dill.pkl'): # Avoid long file loading times
#             with open('caribic_dill.pkl', 'rb') as f:
#                 caribic = dill.load(f)
#             del f
#         else: caribic = Caribic(year_range, c_pfxs = ['GHG', 'INT', 'INT2'])

#     for c_pfx in ['GHG', 'INT2']: # caribic.pfxs:
#         filter_strat_trop(caribic)
#         for subs in substance_list(c_pfx):
#             filter_trop_outliers(caribic, subs, c_pfx)
#     del c_pfx, subs, calc_c

#     filter_strat_trop(caribic)
#     for c_pfx in ['GHG', 'INT2']: # caribic.pfxs:
#         for subs in substance_list(c_pfx):
#             f, axs = plt.subplots(2, 1, dpi=200)
#             for sel_lat, ax in zip([True, False], axs.flatten()):
#                 if not sel_lat: 
#                     filter_trop_outliers(caribic, subs, c_pfx, ax=ax)
#                     ax.set_title('All latitudes')
#                 if sel_lat: 
#                     filter_trop_outliers(caribic.sel_latitude(30, 90), subs, c_pfx, ax=ax)
#                     ax.set_title('Latitudes > 30°N')
#             f.autofmt_xdate()
#             plt.tight_layout()
#             plt.show()
            
#     del c_pfx, subs, calc_c


    # Test outlier identification
    # ol_data = {}
    # df = caribic.data['GHG']
    # for yr in range(2008, 2019):
    #     df[df.index.year == yr]
    #     for dir_val in ['np', 'p', 'n']:
    #         sf6_mxr = df[get_col_name('sf6', 'Caribic')]
    #         ol = outliers.find_ol(fct.simple, df.index, sf6_mxr, None, None,
    #                               plot=True, limit=0.1, direction = dir_val)
    #         ol_data.update({f'{yr}_{dir_val}' : ol})
