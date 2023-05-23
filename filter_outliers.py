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
from toolpac.outliers.outliers import get_no_nan# , fit_data
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
    df.loc[df[col_name] < limit * fit(t_obs_tot), ('strato', 'tropo')] = (True, False)

    df[f'{crit}_pre_flag'] = 0 
    df.loc[df['strato'] == True, f'{crit}_pre_flag'] = 1
    if verbose: print('Result of pre-flagging: \n', df[f'{crit}_pre_flag'].value_counts())
    
    return df.sort_index() # , preflag_df

from toolpac.outliers.outliers import fit_data, get_no_nan

def filter_strat_trop(glob_obj, ref_obj, crit, pfx='GHG', save=True, verbose = False):
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
    # plot the results
    fig, ax = plt.subplots()
    ax.scatter(t_obs_tot, mxr, c='grey', lw=1, label='data')

    no_nan_time, no_nan_mxr, no_nan_d_mxr = get_no_nan(t_obs_tot, mxr, d_mxr)
    popt0 = fit_data(fct.simple, no_nan_time, no_nan_mxr, no_nan_d_mxr)
    baseline0 = fct.simple(np.array(no_nan_time), *popt0)
    ax.plot(np.array(no_nan_time), baseline0, c='r', lw=1, label='initial')
    
    baseline1 = fct.simple(t_obs_tot, *ol[3])
    ax.plot(t_obs_tot, baseline1, c='k', lw=1, label='filtered')
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
        if not hasattr(glob_obj, 'ol_filtered'): # initialise dict with trop / strat dataframes
            glob_obj.ol_filtered = {f'tropo_{crit}_{pfx}' : df_tropo, f'strato_{crit}_{pfx}' : df_strato} 
        else: # overwrite dataframes if they exist, otherwise create them
            glob_obj.ol_filtered[f'tropo_{crit}_{pfx}'] = df_tropo 
            glob_obj.ol_filtered[f'strato_{crit}_{pfx}'] = df_strato

    return data

def filter_trop_outliers(glob_obj, substance_list, pfx):
    """ 
    After sorting data into stratospheric and tropospheric, now sort the 
    tropospheric data into outliers and non-outliers 
    Parameters:
        data: pandas (geo)dataframe
        substance_list: list of strings, substances to receive flags (e.g. 'n2o')
        source (str): source of msmt data, needed to get column name from substances
    """
    # take only tropospheric data 
    data = glob_obj.data[pfx]

    for subs in substance_list:
        subs = get_col_name(subs, glob_obj.source, pfx) # get column name
        if len(get_no_nan(data.index, data[subs], data[subs])[0]) < 1: # check for valid data
            print(f'no {subs} data'); continue

        try: func = get_fct_substance(subs)
        except: 
            print('No function found. Using 2nd order poly with simple harm')
            func = fct.simple

        data_flag = pd.DataFrame(data, columns=['flight', 'timecref', 'year', 'month', 'day', 'strato', 'tropo'])
        data_flag.columns = [f'fl_{x}' if x in substance_list else x for x in data_flag.columns]

        data_flag[f'ol_{subs}'] = np.nan # outlier ? 
        data_flag[f'ol_rel_{subs}'] = np.nan # 
        data_flag[f'fl_{subs}'] = 0 # flag

        # set all strato flags to a value != 0 to exclude them
        data_flag.loc[data['strato'] == True, f'fl_{subs}'] = -20

        time = np.array(datetime_to_fractionalyear(data.index, method='exact'))
        mxr = data[subs].tolist()
        if f'd_{subs}' in data.columns:
            d_mxr = data[f'd_{subs}'].tolist()
        else:    # case for integrated values of high resolution data
            d_mxr = None
        flag = data_flag[f'fl_{subs}'].tolist()
        tmp = outliers.find_ol(func, time, mxr, d_mxr, flag, direction='pn',
                               plot=True, ctrl_plot=True, limit=0.1)

        data_flag[f'fl_{subs}'] = tmp[0]  # flag
        data_flag[f'ol_{subs}'] = tmp[1]  # residual

        data_flag.loc[data_flag['strato'] == True, f'fl_{subs}'] = np.nan
        data_flag.loc[data_flag['strato'] == True, f'ol_{subs}'] = np.nan

        # no residual value for non-outliers
        data_flag.loc[data_flag[f'fl_{subs}'] == 0, f'ol_{subs}'] = np.nan

        fit_result = [func(t, *tmp[3]) for t in time]
        
        data_flag[f'ol_rel_{subs}'] = data_flag[f'ol_{subs}'] / fit_result

    return data_flag

#%% Get data
if __name__=='__main__':
    calc_caribic = False
    if calc_caribic: 
        caribic = Caribic(range(2005, 2021), pfxs = ['GHG', 'INT', 'INT2'])

    mlo_sf6 = Mauna_Loa(range(2008, 2020))
    mlo_n2o = Mauna_Loa(range(2008, 2020), substance = 'n2o')
    mlo_co2 = Mauna_Loa(range(2008, 2020), substance = 'co2')

    data_filtered_n2o = filter_strat_trop(caribic, mlo_n2o, 'n2o', 'INT2')
    data_filtered_co2 = filter_strat_trop(caribic, mlo_co2, 'co2', 'INT2')

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
