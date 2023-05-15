# -*- coding: utf-8 -*-
"""
@Author: Sophie Bauchimger, IAU
@Date: Thu Apr 27 15:59:11 2023

Time Lag calculations
"""

import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import pandas as pd

from toolpac.age import calculate_lag as cl
from toolpac.conv.times import datetime_to_fractionalyear

from dictionaries import get_col_name

def calc_time_lags(c_obj, ref_obj, yr, substance='sf6', pfx='GHG', 
                   ref_min=2003, ref_max=2020, verbose=False, plot=True):
    """ Calculate and plot time lag for caribic data wrt mauna loa msmts"""
    ref_df = ref_obj.df
    ref_df.resample('1M') # add rows for missing months, filled with NaN 
    ref_df.interpolate(inplace=True) # linearly interpolate missing data

    df = c_obj.data[pfx]
    df = df[df.index.year == yr]

    ref_subs = get_col_name(substance, ref_obj.source)
    c_ref = np.array(ref_df[ref_subs])
    t_ref = np.array(datetime_to_fractionalyear(ref_df.index, method='exact'))
    
    subs = get_col_name(substance, c_obj.source)
    c_obs_tot = np.array(df[subs])
    t_obs_tot = np.array(datetime_to_fractionalyear(df.index, method='exact'))
    
    # break if there is no data
    if df[~pd.isna(df[subs])].empty or len(c_obs_tot[~np.isnan(c_obs_tot)])==0: 
        print(f'No valid non-nan data found in {yr} for {subs}') ; return None

    print(f'Calculating lags for {yr}')

    lags = []
    for t_obs, c_obs in zip(t_obs_tot, c_obs_tot):
        lag = cl.calculate_lag(t_ref, c_ref, t_obs, c_obs, plot=True)
        lags.append((lag))

    if verbose: print('length of lags and mean for {yr} : {len(lags)}  {np.nanmean(np.array(lags))}')
    if plot: plot_time_lags(df, lags, ref_min, ref_max, subs)

    return lags


def plot_time_lags(df, lags, ref_min, ref_max, subs = 'sf6'):
    """ Plot calculated time lags of a single year of caribic data """
    
    fig, ax = plt.subplots(dpi=300)
    plt.scatter(df.index, lags, marker='+')
    ax.hlines(np.nanmean(np.array(lags)), 
              dt.datetime(df.index.year[0], 1, 1), 
              dt.datetime(df.index.year[0], 12, 31), 
              'r', ls='dashed', label = 'Mean')
    plt.title('CARIBIC {} time lag {} wrt. MLO {} - {}'.format(
        subs, df.index.year[0], ref_min, ref_max))
    plt.ylabel('Time lag [yr]')
    plt.xlabel('CARIBIC Measurement time')
    plt.legend()
    fig.autofmt_xdate()
    return True

#%% Fct calls
if __name__=='__main__':
    from data_classes import Caribic, Mauna_Loa

    year_range = (2000, 2020)

    mlo_sf6 = Mauna_Loa(year_range)
    mlo_n2o = Mauna_Loa(year_range, substance='n2o')
    
    caribic = Caribic(year_range, pfxs = ['GHG', 'INT', 'INT2'])

    for yr in range(2005, 2020): 
        calc_time_lags(caribic, mlo_sf6, yr, substance = 'sf6')

    for yr in range(2005, 2020): 
        calc_time_lags(caribic, mlo_n2o, yr, substance = 'n2o')
