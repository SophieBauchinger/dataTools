# -*- coding: utf-8 -*-
"""
@Author: Sophie Bauchimger, IAU
@Date: Tue Apr 11 09:28:22 2023

Filtering data into tropospheric and stratospheric air
Getting outlier statistics for the tropospheric part
Removing linear trends from measurements using ground-based reference data

"""
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn' - otherwise df[j] = val gives a warning (outliers.outliers)
import matplotlib.pyplot as plt

# supress a gui backend userwarning, not really advisible
import warnings; warnings.filterwarnings("ignore", category=UserWarning, module='matplotlib')

from aux_fctns import get_lin_fit
from data_classes import Caribic, Mauna_Loa# , Mozart, Mace_Head
from time_lag import calc_time_lags, plot_time_lags
from dictionaries import get_fct_substance, get_col_name

from toolpac.outliers import outliers
from toolpac.outliers import ol_fit_functions as fct
from toolpac.outliers.outliers import get_no_nan, fit_data
from toolpac.conv.times import datetime_to_fractionalyear #, fractionalyear_to_datetime

#%% filter data into stratosphere and troposphere (using n2o as a tracer)

def pre_flag(glob_obj, ref_obj, crit = 'n2o', limit = 0.97, pfx = 'GHG', verbose=False):
    """ 
    Returns df with new strato / tropo columns, and dataframe pre_flagged with results of pre-flagging 

    Parameters:
        glob_obj (GlobalData) : measurement data to be sorted into stratospheric or tropospheric air 
        ref_obj (LocalData) : reference data to use for filtering (background)
        crit (str) : substance to use for flagging
        limit (float) : tracer mxr fraction below which air is classified as stratospheric 
        pfx (str) : e.g. 'GHG', specifc the caribic datasource
    """
    if glob_obj.source=='Caribic': df = glob_obj.data[pfx]
    else: df = glob_obj.df

    df.assign(strato = np.nan, inplace = True)
    df.assign(tropo = np.nan, inplace = True)
    
    fit = get_lin_fit(ref_obj.df, get_col_name(crit, ref_obj.source))
    t_obs_tot = np.array(datetime_to_fractionalyear(df.index, method='exact'))
    
    col_name = get_col_name(crit, glob_obj.source, pfx)
    if not col_name: return pd.DataFrame()
    df.loc[df[col_name] < limit * fit(t_obs_tot), ('strato', 'tropo')] = (True, False)

    df[f'{crit}_pre_flag'] = 0 
    df.loc[df['strato'] == True, f'{crit}_pre_flag'] = 1
    if verbose: print('Result of pre-flagging: \n', df[f'{crit}_pre_flag'].value_counts())
    
    return df.sort_index() # , preflag_df

def filter_strat_trop(glob_obj, ref_obj, crit, pfx='GHG', save=True, verbose = False, plot=False):
    """ 
    Returns dataset with new bool columns 'strato' and 'tropo' 
    Reconstruction of filter_strat_trop from C_tools (T. Schuck)

    Sort data into stratosphere or troposphere based on outlier statistics 
    with respect to measurements eg. at Mauna Loa Observatory

    Parameters: 
        glob_obj (GlobalData) : measurement data to be sorted into stratospheric or tropospheric air 
        ref_obj (LocalData) : reference data to use for filtering (background)
        crit (str): substance to be used for filtering, eg. n2o 
        save (bool): whether to save the strat / trop filtered data in glob_obj
        verbose (bool)
    """
    data = pre_flag(glob_obj, ref_obj, crit, pfx=pfx) # pre-flagging based on crit  # , preflag_df
    if data.empty: return
    col_name = get_col_name(crit, glob_obj.source, pfx) # get column name
    t_obs_tot = np.array(datetime_to_fractionalyear(data.index, method='exact'))  # find total observation time as fractional year for fctn calls below
    mxr = data[col_name] # measured mixing ratios
    if f'd_{col_name}' in data.columns: 
        d_mxr = data[f'd_{col_name}']
    else: 
        d_mxr = None
        if verbose: print('No abs. errors found for {col_name}')

    ol = outliers.find_ol(fct.simple, t_obs_tot, mxr, d_mxr, 
                          flag = data[f'{crit}_pre_flag'], 
                          plot=True, limit=0.1, direction = 'n')
    
    if plot: 
        fig, ax = plt.subplots()
        ax.scatter(t_obs_tot, mxr, c='grey', lw=1, label='data')
    
        no_nan_time, no_nan_mxr, no_nan_d_mxr = get_no_nan(t_obs_tot, mxr, d_mxr)
        popt0 = fit_data(fct.simple, no_nan_time, no_nan_mxr, no_nan_d_mxr)
        ax.plot(np.array(no_nan_time), fct.simple(np.array(no_nan_time), *popt0), 
                c='r', lw=1, label='initial')
        ax.plot(t_obs_tot, fct.simple(t_obs_tot, *ol[3]), 
                c='k', lw=1, label='filtered')
        fig.legend()
        fig.show()

    data.drop(f'{crit}_pre_flag', axis=1, inplace=True)

    # ^ 4er tuple, 1st ist liste von OL == 1 / 2 / 3, wenn not outlier dann == 0
    data.loc[(ol[0] != 0), ('strato', 'tropo')] = (True, False)
    data.loc[(ol[0] == 0), ('strato', 'tropo')] = (False, True)

    # separate the tropospheric from the stratospheric data, remove trop / strat columns
    df_tropo = data[data['tropo'] == True]
    df_tropo.drop(['tropo', 'strato'], axis=1, inplace=True)

    df_strato = data[data['strato'] == True]
    df_strato.drop(['tropo', 'strato'], axis=1, inplace=True)

    if save: 
        glob_obj.data[f'tropo_{pfx}_{crit}'] = df_tropo
        glob_obj.data[f'strato_{pfx}_{crit}'] = df_strato

    return data, ol

def filter_trop_outliers(glob_obj, subs, pfx, crit=None, ref_obj=None, save=True):
    """ 
    After sorting data into stratospheric and tropospheric, now sort the 
    tropospheric data into outliers and non-outliers (in both directions)
    Parameters:
        glob_obj (GlobalData): e.g. caribic
        subs (str): substances to flag, eg. 'n2o'
        pfx (str): caribic data prefix, eg. 'GHG'
        crit (str): criterion for strat / trop sorting 
        ref_obj (LocalData): reference data for strat / trop sorting if not yet available
    """
    # take only tropospheric data, select according to crit and availability
    trop_crits = [x for x in glob_obj.data.keys() if x.startswith('tropo')]

    if len(trop_crits) == 0: # no tropospheric data
        if not crit: crit = input('Please input a substance to sort strat/trop data with (eg. n2o).\n') 
        filter_strat_trop(glob_obj, ref_obj, crit, pfx=pfx, save=True) # creates tropo data within global_obj
        data = glob_obj.data[f'tropo_{pfx}_{crit}']
    
    if len(trop_crits) == 1 and (crit==None or f'tropo_{pfx}_{crit}' in trop_crits): # one trop dataset exists
        data = glob_obj.data[trop_crits[0]] 

    elif len(trop_crits) > 1: 
        if f'tropo_{pfx}_{crit}' in trop_crits: # ideal case
            data = glob_obj.data[f'tropo_{pfx}_{crit}']

        else: # crit not yet 
            crit = input(f'Please choose one of the following criteria (select by writing e.g. n2o): \n{trop_crits}\n')
            data = glob_obj.data['{}'.format([x for x in trop_crits if x.endswith(crit)][0])]

    else: print('Something went wrong while choosing the data to use'); return

    substance = get_col_name(subs, glob_obj.source, pfx) # get column name for substance to be outlier flagged
    # print(substance)

    if len(get_no_nan(data.index, data[substance], data[substance])[0]) < 1: # check for valid data
        print(f'no {subs} data'); return None

    # Get outlier fit function
    try: func = get_fct_substance(subs)
    except: 
        print(f'No fit function found for {subs}. Using 2nd order poly with simple harm')
        func = fct.simple

    # create output dataframe
    data_flag = pd.DataFrame(data, columns=['Flight number', 'strato', 'tropo', f'{substance}'])
    
    data_flag[f'ol_{subs}'] = np.nan # outlier 
    data_flag[f'ol_rel_{subs}'] = np.nan # residual
    data_flag[f'fl_{subs}'] = 0 # flag

    time = np.array(datetime_to_fractionalyear(data.index, method='exact'))
    mxr = data[substance].tolist()
    if f'd_{substance}' in data.columns:
        d_mxr = data[f'd_{substance}'].tolist()
    else: # this is the case for integrated values of high resolution data
        d_mxr = None

    flag = data_flag[f'fl_{subs}'].tolist()
    tmp = outliers.find_ol(func, time, mxr, d_mxr, flag, direction='pn',
                            plot=True, limit=0.1)

    data_flag[f'fl_{subs}'] = tmp[0]  # flag
    data_flag[f'ol_{subs}'] = tmp[1]  # outlier 
    fit_result = [func(t, *tmp[3]) for t in time]
    data_flag[f'ol_rel_{subs}'] = data_flag[f'ol_{subs}'] / fit_result # residuals

    # data_flag.loc[data_flag[f'fl_{subs}'] == 0, f'ol_{subs}'] = np.nan # no residual value for non-outliers

    return data_flag

#%% Get data
if __name__=='__main__':
    calc_caribic = False
    if calc_caribic: 
        caribic = Caribic(range(2005, 2021), pfxs = ['GHG', 'INT', 'INT2'])

    mlo_sf6 = Mauna_Loa(range(2008, 2020))
    mlo_n2o = Mauna_Loa(range(2008, 2020), substance = 'n2o')
    mlo_co2 = Mauna_Loa(range(2008, 2020), substance = 'co2')

    data_filtered_n2o, n2o_ol = filter_strat_trop(caribic, mlo_n2o, 'n2o', 'INT2')
    data_filtered_co2, co2_ol = filter_strat_trop(caribic, mlo_co2, 'co2', 'INT2')
    
    for subs in ['ch4', 'co2', 'co']:
        filter_trop_outliers(caribic, subs, 'INT2', crit='n2o')

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

#%% Filter tropospheric and stratospheric data

# # test pre_flag with sf6 data
# if __name__=='__main__': 
#     c_pref = Caribic([2008]).df
#     ref_pref = Mauna_Loa(range(2008, 2020), substance = 'sf6').df
#     pref = pre_flag(c_pref, 'SF6 [ppt]', 
#                     np.array(datetime_to_fractionalyear(c_pref.index, method='exact')), 
#                     get_lin_fit(ref_pref, get_col_name('sf6', 'Mauna_Loa')))

# if __name__=='__main__':
#     # loop through years of caribic data
#     data_filtered = pd.DataFrame() # initialise full dataframe
#     for c_year in range(2006, 2009): 
#         print(f'{c_year}')
#         c_data = caribic_data.select_year(c_year)
#         # print('cols:', c_data.columns)

#         crit = 'n2o'; n2o_filtered = pd.DataFrame()
#         if len(get_no_nan(c_data.index, c_data['N2O [ppb]'], c_data['d_N2O [ppb]'])[0]) < 1: # check for valid data
#             print('! no n2o data')
#         else:
#             n2o_filtered =  filter_strat_trop(c_data, crit)
#             data_filtered = pd.concat([data_filtered, n2o_filtered])

#         crit = 'sf6'; sf6_filtered = pd.DataFrame()
#         if crit=='sf6' and len(get_no_nan(c_data.index, c_data['SF6 [ppt]'], c_data['d_SF6 [ppt]'])[0]) < 1: # check for valid data
#                 print('! no sf6 data')
#         else: 
#             sf6_filtered =  filter_strat_trop(c_data, crit)
#             data_filtered = pd.concat([data_filtered, sf6_filtered])

#     data_stratosphere = data_filtered.loc[data_filtered['strato'] == True]
#     data_troposphere = data_filtered.loc[data_filtered['tropo'] == True]

#     data_trop_outlier = filter_trop_outliers(data_filtered, ['sf6', 'n2o'])
