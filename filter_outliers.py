# -*- coding: utf-8 -*-
"""
@Author: Sophie Bauchimger, IAU
@Date: Tue Apr 11 09:28:22 2023

Filtering data into tropospheric and stratospheric air
Getting outlier statistics for the tropospheric part
Removing linear trends from measurements using ground-based reference data

"""
import numpy as np
import sys
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn' - otherwise df[j] = val gives a warning (outliers.outliers)

import matplotlib.pyplot as plt

# supress a gui backend userwarning, not really advisible
import warnings; warnings.filterwarnings("ignore", category=UserWarning, module='matplotlib')

from data_classes import Caribic, Mauna_Loa# , Mozart, Mace_Head
from time_lag import calc_time_lags, plot_time_lags
from aux_fctns import get_fct_substance, get_col_name, get_lin_fit
from detrend import detrend_substance

from toolpac.calc import bin_1d_2d
from toolpac.outliers import outliers
from toolpac.outliers import ol_fit_functions as fct
from toolpac.outliers.outliers import get_no_nan# , fit_data
from toolpac.conv.times import datetime_to_fractionalyear #, fractionalyear_to_datetime

sys.path.insert(0, r'C:\Users\sophie_bauchinger\sophie_bauchinger\Caribic_data_handling')
# import C_SF6_age
import C_tools

#%% Outliers
if __name__=='__main__':
    ol_data = {}
    years = range(2008, 2010)
    c_data = Caribic(years)
    for y in years: # Caribic
        for dir_val in ['np', 'p', 'n']:
            data = c_data.select_year(y)
            sf6_mxr = data['SF6 [ppt]']
            ol = outliers.find_ol(fct.simple, data.index, sf6_mxr, None, None, 
                                  plot=True, limit=0.1, direction = dir_val)
            ol_data.update({f'{y}_{dir_val}' : ol})

#%% filter data into stratosphere and troposphere (using n2o as a tracer)

def pre_flag(data, data_col, t_obs_tot, ref_fit, limit = 0.97, crit='n2o', verbose=False):
    """ 
    Flags everything with lower n2o than limir * mlo_fit(frac_year) 
    as 'strato' in an initial filtering step (default: 3% cut-off)
    Returns dataframe with new strato / tropo columns as well as new dataframe
    with flagging results
    """ 
    # initialise columns to hold strat and trop flags (needs to be done on two lines)
    data = data.assign(strato = np.nan)
    data = data.assign(tropo = np.nan)

    data.loc[data[data_col] < limit * ref_fit(t_obs_tot), ('strato', 'tropo')] = (True, False)

    # create new dataframe to hold preflagging data
    pre_flagged = pd.DataFrame(data, columns=['Flight number', 'strato', 'tropo'])
    pre_flagged[f'{crit}_pre_flag'] = 0 # initialise flag with zeros
    pre_flagged.loc[data['strato'] == True, f'{crit}_pre_flag'] = 1 # set flag indicator for pre-flagged measurements
    if verbose: print('Result of pre-flagging: \n', pre_flagged.value_counts()) # show results of preflagging
    return data, pre_flagged

def filter_strat_trop(data, crit, source='Caribic'):
    """ 
    Returns dataset with new bool columns 'strato' and 'tropo' 
    Reconstruction of filter_strat_trop from C_tools (T. Schuck)

    Sort data into stratosphere or troposphere based on outlier statistics 
    with respect to measurements eg. at Mauna Loa Observatory

    Parameters: 
        data: DataFrame of data to be sorted (e.g. Caribic)
        crit: substance to be used for filtering, eg. n2o or sf6 
    """
    df = Mauna_Loa(range(2008, 2020), substance = crit).df
    mlo_fit = get_lin_fit(df, get_col_name(crit, 'Mauna_Loa'))

    col_nm = get_col_name(crit, source) # get column name
    print(col_nm)
    data = data.dropna(how='any', subset=[col_nm]) # choose only rows where n2o data exists
    t_obs_tot = np.array(datetime_to_fractionalyear(data.index, method='exact'))  # find total observation time as fractional year for fctn calls below

    data, pre_flagged = pre_flag(data, col_nm, t_obs_tot, mlo_fit) # pre-flagging

    mxr = data[col_nm] # measured n2o mixing ratios
    d_mxr = data[f'd_{col_nm}']

    ol = outliers.find_ol(fct.simple, t_obs_tot, mxr, d_mxr, 
                              flag = pre_flagged.n2o_pre_flag, 
                              plot=True, limit=0.1, direction = 'n')

    # ^ 4er tuple, 1st ist liste von OL == 1 / 2 / 3, wenn not outlier dann == 0
    data.loc[(ol[0] != 0), ('strato', 'tropo')] = (True, False)
    data.loc[(ol[0] == 0), ('strato', 'tropo')] = (False, True)

    return data

def filter_trop_outliers(data, substance_list, source='Caribic'):
    """ 
    After sorting data into stratospheric and tropospheric, now sort the 
    tropospheric data into outliers and non-outliers 
    Parameters:
        data: pandas (geo)dataframe
        substance_list: list of strings, substances to receive flags
    """
    # take only tropospheric data 
    for subs in substance_list:
        subs = get_col_name(subs, source)
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
                               plot=True, limit=0.1)

        data_flag[f'fl_{subs}'] = tmp[0]  # flag
        data_flag[f'ol_{subs}'] = tmp[1]  # residual

        data_flag.loc[data_flag['strato'] == True, f'fl_{subs}'] = np.nan
        data_flag.loc[data_flag['strato'] == True, f'ol_{subs}'] = np.nan

        # no residual value for non-outliers
        data_flag.loc[data_flag[f'fl_{subs}'] == 0, f'ol_{subs}'] = np.nan

        fit_result = [func(t, *tmp[3]) for t in time]
        # print(len(fit_result), len(data_flag))
        data_flag[f'ol_rel_{subs}'] = data_flag[f'ol_{subs}'] / fit_result

    return data_flag

#%% Get data
if __name__=='__main__':
    mlo_df = Mauna_Loa(range(2008, 2020)).df
    n2o_df = Mauna_Loa(range(2008, 2020), substance = 'n2o').df
    
    caribic_data = Caribic(range(2005, 2020))
    c_df = caribic_data.df

#%% Time lags
if __name__=='__main__':
    # Get and prep reference data 
    ref_min, ref_max = 2003, 2020
    mlo_MM = Mauna_Loa(range(ref_min, ref_max)).df
    mlo_MM.resample('1M') # add rows for missing months, filled with NaN 
    mlo_MM.interpolate(inplace=True) # linearly interpolate missing data

    # loop through years of caribic data
    for c_year in range(2005, 2022):
        c_data = caribic_data.select_year(c_year)
        if len(c_data[c_data['SF6 [ppt]'].notna()]) < 1: 
            continue
        else:
            lags = calc_time_lags(c_data, mlo_MM)
            if all(np.isnan(np.array(lags))): 
                print(f'no lags calculated for {c_year}'); continue
            plot_time_lags(c_data, lags, ref_min, ref_max)

#%% Filter tropospheric and stratospheric data

# test pre_flag with sf6 data
# if __name__=='__main__': 
#     c_pref = Caribic([2008]).df
#     ref_pref = Mauna_Loa(range(2008, 2020), substance = 'sf6').df
#     pref = pre_flag(c_pref, 'SF6 [ppt]', 
#                     np.array(datetime_to_fractionalyear(c_pref.index, method='exact')), 
#                     get_lin_fit(ref_pref, get_col_name('sf6', 'Mauna_Loa')))

if __name__=='__main__':
    # loop through years of caribic data
    data_filtered = pd.DataFrame() # initialise full dataframe
    for c_year in range(2006, 2009): 
        print(f'{c_year}')
        c_data = caribic_data.select_year(c_year)
        # print('cols:', c_data.columns)

        crit = 'n2o'; n2o_filtered = pd.DataFrame()
        if len(get_no_nan(c_data.index, c_data['N2O [ppb]'], c_data['d_N2O [ppb]'])[0]) < 1: # check for valid data
            print('! no n2o data')
        else:
            n2o_filtered =  filter_strat_trop(c_data, crit)
            data_filtered = pd.concat([data_filtered, n2o_filtered])

        crit = 'sf6'; sf6_filtered = pd.DataFrame()
        if crit=='sf6' and len(get_no_nan(c_data.index, c_data['SF6 [ppt]'], c_data['d_SF6 [ppt]'])[0]) < 1: # check for valid data
                print('! no sf6 data')
        else: 
            sf6_filtered =  filter_strat_trop(c_data, crit)
            data_filtered = pd.concat([data_filtered, sf6_filtered])

    data_stratosphere = data_filtered.loc[data_filtered['strato'] == True]
    data_troposphere = data_filtered.loc[data_filtered['tropo'] == True]

    data_trop_outlier = filter_trop_outliers(data_filtered, ['sf6', 'n2o'])
