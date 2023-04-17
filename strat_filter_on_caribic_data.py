# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 09:28:22 2023

@author: sophie_bauchinger
"""

import numpy as np
import sys
import pandas as pd
from pathlib import Path
import datetime as dt

import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable as sm

sys.path.insert(0, r'C:\Users\sophie_bauchinger\sophie_bauchinger\toolpac_tutorial')
from toolpac_tutorial import Mauna_Loa, Mace_Head, Caribic, Mozart

from toolpac.calc import bin_1d_2d
from toolpac.outliers import outliers, ol_fit_functions
from toolpac.age import calculate_lag as cl
from toolpac.conv.times import datetime_to_fractionalyear, fractionalyear_to_datetime

sys.path.insert(0, r'C:\Users\sophie_bauchinger\sophie_bauchinger\Caribic_data_handling')
from C_filter import filter_strat_trop, filter_outliers
import C_read
import C_SF6_age
import C_tools

# sf6_path = r'C:\Users\sophie_bauchinger\sophie_bauchinger\toolpac_tutorial\mlo_SF6_MM.dat'
sf6_df = Mauna_Loa(range(2008, 2020), substance = 'sf6').df

n2o_path = r'C:\Users\sophie_bauchinger\sophie_bauchinger\misc_data'
n2o_fname = 'mlo_N2O_MM.dat'

n2o_df = Mauna_Loa(range(2008, 2020), substance = 'n2o').df

caribic_data = Caribic(range(2016, 2020))
c_df = caribic_data.df

#%% Time Lag calculations
mlo_time_lims = (2000, 2020)
mlo_MM = Mauna_Loa(years = np.arange(*mlo_time_lims)).df_MM
mlo_MM.resample('1M') # add rows for missing months, filled with NaN 
mlo_MM.interpolate(inplace=True) # linearly interpolate missing data

t_ref = np.array(datetime_to_fractionalyear(mlo_MM.index, method='exact'))
c_ref = np.array(mlo_MM['SF6catsMLOm'])

for c_year in range(2016, 2020):
    c_data = caribic_data.select_year(c_year)
    t_obs_tot = np.array(datetime_to_fractionalyear(c_data.index, method='exact'))
    c_obs_tot = np.array(c_data[caribic_data.substance])

    lags = []
    for t_obs, c_obs in zip(t_obs_tot, c_obs_tot):
        lag = cl.calculate_lag(t_ref, c_ref, t_obs, c_obs, plot=True)
        lags.append((lag))
    print(c_year, ': ', len(lags),  np.nanmean(np.array(lags)))

    if all(np.isnan(np.array(lags))): continue

    fig, ax = plt.subplots(dpi=300)
    plt.scatter(c_data.index, lags, marker='+')
    ax.hlines(np.nanmean(np.array(lags)), dt.datetime(c_year, 1, 1), dt.datetime(c_year, 12, 31), 'r', ls='dashed')
    plt.title('CARIBIC {} time lag {} wrt. MLO {} - {}'.format(caribic_data.substance_short, c_year, *mlo_time_lims))
    plt.ylabel('Time lag [yr]')
    plt.xlabel('CARIBIC Measurement time')
    fig.autofmt_xdate()

#%% strat trop filter
from C_filter import filter_strat_trop, filter_outliers
from toolpac.outliers import ol_fit_functions as fct

# for y in range(2008, 2010):
# data_ref = Caribic([2008, 2009, 2010]).df

def get_mlo_fit(mlo_df, substance='N2OcatsMLOm'):
    """ Given one year of reference data, find the fit parameters for n2o """
    df = mlo_df.dropna(how='any', subset=substance)
    year = df.index.year
    month = df.index.month
    mlo_t_ref = year + (month - 0.5) / 12 # obtain fractional year for middle of the month
    mlo_mxr_ref = df[substance].values
    return np.poly1d(np.polyfit(mlo_t_ref, mlo_mxr_ref, 2))

mlo_fit = get_mlo_fit(n2o_df)

ref_data = n2o_df
pv_lim=2.
mlo_lim = 0.97
# obtain MLO fit parameters


print(f'MLO fit parameters obtained: {mlo_fit}')


def filter_strat_trop(data, ref_data, crit):
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

    # find total observation time as fractional year
    t_obs_tot = np.array(datetime_to_fractionalyear(data.index, method='exact'))

    # initialise columns to hold strat and trop flags
    data = data.assign(strato = np.nan)
    data = data.assign(tropo = np.nan)
    


    # PRE FLAGGING
    # alles das kleinere n20 daten hat als mlo_lim * mlo_fit(frac_year) wird als stratospheric eingestuft
    data.loc[data[n2o_col] < mlo_lim * mlo_fit(t_obs_tot), ('strato', 'tropo')] = (True, False)

    # create new dataframe to hold preflagging data
    pre_flagged = pd.DataFrame(data, columns=['Flight number', 'strato', 'tropo'])

    pre_flagged.loc[pre_flagged['strato'] == True, 'n2o_flag'] = -1 # set flag indicator for pre-flagged measurements
    print('Result of pre-flagging: \n', pre_flagged.value_counts()) # show results of preflagging


    # OUTLIER 
    if crit == 'n2o':
        dir_val = 'n'
        # n2o_mxr = data[n2o_col] # measured n2o mixing ratios

        data.loc['d_n2o [ppb]'] = 0

        print(data.index, data[n2o_col],  pre_flagged.n2o_flag)

        ol_n2o = outliers.find_ol(ol_fit_functions.simple, t_obs_tot, data[n2o_col], data['d_n2o [ppb]'], flag = pre_flagged.n2o_flag, 
                              plot=True, limit=0.1, direction = dir_val)
        return ol_n2o
        

    # Caribic SF6
    # sf6_mxr = data[sf6_col]
    # ol_sf6 = outliers.find_ol(ol_fit_functions.simple, data.index, sf6_mxr, None, None, 
    #                       plot=True, limit=0.1, direction = dir_val)
    # ^ 4er tuple, 1st ist liste von OL=1, !OL=0

    # data_filtered = data[ol_sf6]

    # Caribic N2O


#%% 
from C_filter import filter_strat_trop, filter_outliers
from toolpac.outliers import ol_fit_functions as fct
from toolpac.outliers.outliers import get_no_nan, fit_data

if __name__=='__main__':
    crit = 'n2o'
    for year in [2015]:
        c_data = caribic_data.select_year(year)
        # filtered_data = filter_strat_trop(c_data, ref_data, crit)

    data = c_data

    n2o_col = caribic_data.get_col_name('n2o')
    sf6_col = caribic_data.get_col_name('sf6')

    # choose only rows where sf6 and n2o data exists
    data = data.dropna(how='any', subset=[n2o_col])

    # find total observation time as fractional year
    t_obs_tot = np.array(datetime_to_fractionalyear(data.index, method='exact'))

    # initialise columns to hold strat and trop flags
    data = data.assign(strato = np.nan)
    data = data.assign(tropo = np.nan)
    


    # PRE FLAGGING
    # alles das kleinere n20 daten hat als mlo_lim * mlo_fit(frac_year) wird als stratospheric eingestuft
    data.loc[data[n2o_col] < mlo_lim * mlo_fit(t_obs_tot), ('strato', 'tropo')] = (True, False)

    # create new dataframe to hold preflagging data
    pre_flagged = pd.DataFrame(data, columns=['Flight number', 'strato', 'tropo'])

    pre_flagged['n2o_flag'] = 0
    pre_flagged.loc[pre_flagged['strato'] == True, 'n2o_flag'] = -1 # set flag indicator for pre-flagged measurements
    print('Result of pre-flagging: \n', pre_flagged.value_counts()) # show results of preflagging


#!! The preflagging is not done in the way the function below wants it to be apparently. Bish

    # OUTLIER 
    if crit == 'n2o':
        dir_val = 'n'
        # n2o_mxr = data[n2o_col] # measured n2o mixing ratios
        # data.loc['d_n2o [ppb]'] = 0
        
        flag = pre_flagged.n2o_flag.copy()

        no_nan_time, no_nan_mxr, no_nan_d_mxr = get_no_nan(t_obs_tot, data[n2o_col], data[n2o_col], flag, flagged=True)#, data['d_n2o [ppb]'])#, pre_flagged.n2o_flag)

        fit_data(ol_fit_functions.simple, no_nan_time, no_nan_mxr, no_nan_d_mxr)        

        
        time = t_obs_tot
        mxr = data[n2o_col]
        d_mxr = data[n2o_col]
        flag_in = flag.copy()
        func = ol_fit_functions.simple

        tmp_time, tmp_mxr, tmp_d_mxr = get_no_nan(time, mxr, d_mxr, flag=flag_in, flagged=False) # =False: all flagged data is removed, only unflagged data is returned

        popt1 = fit_data(func, tmp_time, tmp_mxr, tmp_d_mxr)


        ol_n2o = outliers.find_ol(ol_fit_functions.simple, t_obs_tot, data[n2o_col], d_mxr=data[n2o_col], flag = flag, plot=True, limit=0.1, direction = dir_val, verbose=True)
        
        

# not meant to call this stuff in new code (legacy function)
# =============================================================================
#     # ref_year = 2005
# 
#     # data['year_delta'] = datetime_to_fractionalyear(data.index)
#     # data['year_delta'] = data['year_delta'] - ref_year
# 
# 
#     # outliers.ol_iteration_for_subst(subs, data, data_flag,
#     #                                 func=fct.simple, direction='n', limit=0.1, plot=True)
#     # print(f'year {year}', np.array(data.strato), np.array(data.tropo))
# =============================================================================


# def pre_flag(data, ref_data, mlo_lim=0.97):
    

# return mlo_fit

# print(pre_flagging(c_data, n2o_df))
    
# n2o_filtered = filter_strat_trop(c_data, n2o_df, crit='n2o', plot=True)

#%% Outliers
# Caribic
for y in range(2008, 2010):
    data = Caribic([y]).df
    # Caribic SF6
    for dir_val in ['n']:
        sf6_mxr = data['SF6; SF6 mixing ratio; [ppt]\n']
        ol_sf6 = outliers.find_ol(ol_fit_functions.simple, data.index, sf6_mxr, None, None, 
                              plot=True, limit=0.1, direction = dir_val)
        # ^ 4er tuple, 1st ist liste von OL=1, !OL=0

    data_filtered = data[ol_sf6()]
    # Caribic N2O
    for dir_val in ['n']:
        n20_mxr = data['N2O; N2O mixing ratio; [ppb]\n']
        ol_n20 = outliers.find_ol(ol_fit_functions.simple, data.index, n20_mxr, None, None, 
                              plot=True, limit=0.1, direction = dir_val)

# Mace Head
for dir_val in ['np', 'p', 'n']: 
    data = Mace_Head().df
    sf6_mxr = data['SF6[ppt]']
    ol = outliers.find_ol(ol_fit_functions.simple, data.index, sf6_mxr, None, None, 
                          plot=True, limit=0.1, direction = dir_val)

# Mauna Loa
for y in range(2008, 2010): 
    for dir_val in ['np', 'p', 'n']: 
        data = Mauna_Loa([y]).df
        sf6_mxr = data['SF6catsMLOm']
        ol = outliers.find_ol(ol_fit_functions.simple, data.index, sf6_mxr, None, None, 
                              plot=True, limit=0.1, direction = dir_val)

