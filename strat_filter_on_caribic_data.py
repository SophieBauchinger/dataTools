# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 09:28:22 2023

@author: sophie_bauchinger
"""
#%% Imports
import numpy as np
import sys
import pandas as pd
from pathlib import Path
import datetime as dt

import matplotlib.pyplot as plt

sys.path.insert(0, r'C:\Users\sophie_bauchinger\sophie_bauchinger\toolpac_tutorial')
from toolpac_tutorial import Mauna_Loa, Mace_Head, Caribic, Mozart

from toolpac.calc import bin_1d_2d
from toolpac.outliers import outliers
from toolpac.outliers import ol_fit_functions as fct
from toolpac.outliers.outliers import get_no_nan, fit_data
from toolpac.age import calculate_lag as cl
from toolpac.conv.times import datetime_to_fractionalyear, fractionalyear_to_datetime

sys.path.insert(0, r'C:\Users\sophie_bauchinger\sophie_bauchinger\Caribic_data_handling')
from C_filter import filter_strat_trop, filter_outliers
import C_read
import C_SF6_age
import C_tools

#%% Get data
# sf6_path = r'C:\Users\sophie_bauchinger\sophie_bauchinger\toolpac_tutorial\mlo_SF6_MM.dat'
sf6_df = Mauna_Loa(range(2008, 2020), substance = 'sf6').df

n2o_path = r'C:\Users\sophie_bauchinger\sophie_bauchinger\misc_data'
n2o_fname = 'mlo_N2O_MM.dat'

n2o_df = Mauna_Loa(range(2008, 2020), substance = 'n2o').df

caribic_data = Caribic(range(2016, 2020))
c_df = caribic_data.df

#%% Time Lag calculations

def cal_time_lags(c_data, ref_data, ref_subs = 'SF6catsMLOm'):
    """ Calculate and plot time lag for caribic data wrt mauna loa msmts"""
    t_ref = np.array(datetime_to_fractionalyear(ref_data.index, method='exact'))
    c_ref = np.array(ref_data[ref_subs])
    
    c_obs_tot = np.array(c_data[caribic_data.substance])
    t_obs_tot = np.array(datetime_to_fractionalyear(c_data.index, method='exact'))

    print(f'Calculating lags for {c_data.index.year[0]}')

    lags = []
    for t_obs, c_obs in zip(t_obs_tot, c_obs_tot):
        lag = cl.calculate_lag(t_ref, c_ref, t_obs, c_obs, plot=True)
        lags.append((lag))
    # print('length of lags and mean for ', c_year, ': ', len(lags),  np.nanmean(np.array(lags)))
    return lags

def plot_time_lags(c_data, lags, ref_lims):
    """ Plot calculated time lags of a single year of caribic data """
    print(f'Plotting lags for {c_year}')
    fig, ax = plt.subplots(dpi=300)
    plt.scatter(c_data.index, lags, marker='+')
    ax.hlines(np.nanmean(np.array(lags)), 
              dt.datetime(c_data.index.year[0], 1, 1), 
              dt.datetime(c_data.index.year[0], 12, 31), 'r', ls='dashed')
    plt.title('CARIBIC {} time lag {} wrt. MLO {} - {}'.format(
        caribic_data.substance_short, c_data.index.year[0], *ref_lims))
    plt.ylabel('Time lag [yr]')
    plt.xlabel('CARIBIC Measurement time')
    fig.autofmt_xdate()
    return True

if __name__=='__main__':
    # Prep reference data 
    mlo_time_lims = (2000, 2020)
    mlo_MM = Mauna_Loa(years = np.arange(*mlo_time_lims)).df_MM
    mlo_MM.resample('1M') # add rows for missing months, filled with NaN 
    mlo_MM.interpolate(inplace=True) # linearly interpolate missing data
    
    # loop through years of caribic data
    for c_year in range(2016, 2020):
        c_data = caribic_data.select_year(c_year)
        lags = cal_time_lags(c_data, mlo_MM)
        if all(np.isnan(np.array(lags))): 
            print(f'no lags calculated for {c_year}'); continue
        plot_time_lags(c_data, lags, mlo_time_lims)

#%% n2o strat trop filter
def get_mlo_fit(mlo_df, substance='N2OcatsMLOm'):
    """ Given one year of reference data, find the fit parameters for n2o """
    df = mlo_df.dropna(how='any', subset=substance)
    year, month = df.index.year, df.index.month
    mlo_t_ref = year + (month - 0.5) / 12 # obtain fractional year for middle of the month
    mlo_mxr_ref = df[substance].values
    mlo_fit = np.poly1d(np.polyfit(mlo_t_ref, mlo_mxr_ref, 2))
    print(f'MLO fit parameters obtained: {mlo_fit}')
    return mlo_fit


def pre_flag(data, n2o_col, t_obs_tot, mlo_fit):
    """ 
    everything with lower n2o than mlo_lim*mlo_fit(frac_year) is flagged 
    as 'strato' in an initial filtering step 
    """ 
    mlo_lim = 0.97

    data = data.assign(strato = np.nan)
    data = data.assign(tropo = np.nan)

    data.loc[data[n2o_col] < mlo_lim * mlo_fit(t_obs_tot), ('strato', 'tropo')] = (True, False)

    # create new dataframe to hold preflagging data
    pre_flagged = pd.DataFrame(data, columns=['Flight number', 'strato', 'tropo'])
    pre_flagged['n2o_pre_flag'] = 0 # initialise flag with zeros
    pre_flagged.loc[data['strato'] == True, 'n2o_pre_flag'] = 1 # set flag indicator for pre-flagged measurements
    print('Result of pre-flagging: \n', pre_flagged.value_counts()) # show results of preflagging
    return data, pre_flagged

def filter_strat_trop(data, ref_data, crit, mlo_fit):
    """ 
    Sort data into stratosphere or troposphere based on reference data
    
    Parameters: 
        data: DataFrame of data to be sorted 
        ref_data: DataFrame of reference data
        crit: substance to be used for categorisation, eg. n2o or sf6 
    """
    # find column names for caribic data
    n2o_col = caribic_data.get_col_name('n2o')
    sf6_col = caribic_data.get_col_name('sf6')

    # choose only rows where sf6 and n2o data exists
    data = data.dropna(how='any', subset=[n2o_col])
    data = data.dropna(how='any', subset=[sf6_col])

    # find total observation time as fractional year
    t_obs_tot = np.array(datetime_to_fractionalyear(data.index, method='exact'))

    # initialise columns to hold strat and trop flags (needs to be done on two lines)


    data, pre_flagged = pre_flag(data, n2o_col, t_obs_tot, mlo_fit) # pre-flagging

    # OUTLIER 
    if crit == 'n2o':
        dir_val = 'n'
        n2o_mxr = data[n2o_col] # measured n2o mixing ratios
        n2o_d_mxr = data['d_N2O [ppb]']
        # print(data.index, data[n2o_col],  pre_flagged.n2o_flag)

        ol_n2o = outliers.find_ol(fct.simple, t_obs_tot, n2o_mxr, n2o_d_mxr, flag = pre_flagged.n2o_pre_flag, 
                              plot=True, limit=0.1, direction = dir_val)
        # ^ 4er tuple, 1st ist liste von OL=1, !OL=0
        data.loc[(ol_n2o[0] != 0), ('strato', 'tropo')] = (True, False)
        data.loc[(ol_n2o[0] == 0), ('strato', 'tropo')] = (False, True)

    if crit == 'sf6':
        dir_val = 'n'
        sf6_mxr = data[sf6_col] # measured n2o mixing ratios
        sf6_d_mxr = data['d_SF6 [ppt]']

        ol_sf6 = outliers.find_ol(fct.simple, t_obs_tot, sf6_mxr, sf6_d_mxr, flag = pre_flagged.n2o_pre_flag, 
                              plot=True, limit=0.1, direction = dir_val)
        data.loc[(ol_sf6[0] != 0), ('strato', 'tropo')] = (True, False)
        data.loc[(ol_sf6[0] == 0), ('strato', 'tropo')] = (False, True)

    return data

def filtered_caribic(c_year):
    print(f'{c_year}')
    c_data = caribic_data.select_year(c_year)
    
    for crit in ['n2o', 'sf6']:
        if crit=='n2o' and len(get_no_nan(c_data.index, c_data['N2O [ppb]'], c_data['d_N2O [ppb]'])[0]) < 1: # check for valid data
            print('! no n2o data'); continue
        try: print(c_data['N2O [ppb]'], '\n cols:', c_data.columns)
        except: pass

        if crit=='sf6' and len(get_no_nan(c_data.index, c_data['SF6 [ppt]'], c_data['d_SF6 [ppt]'])[0]) < 1: # check for valid data
                print('! no sf6 data'); continue
        try: print(c_data['SF6 [ppt]'], '\n cols:', c_data.columns)
        except: pass

        return filter_strat_trop(c_data, ref_data, crit, mlo_fit)

if __name__=='__main__':
    mlo_fit = get_mlo_fit(n2o_df)
    ref_data = n2o_df
    pv_lim=2.

    # loop through years of caribic data
    data_filtered = pd.DataFrame()
    for c_year in range(2017, 2022): 
        single_year = filtered_caribic(c_year)
        data_filtered = pd.concat([data_filtered, single_year])

    data_stratosphere = data_filtered.loc[data_filtered['strato'] == True]
    data_troposphere = data_filtered.loc[data_filtered['tropo'] == True]
