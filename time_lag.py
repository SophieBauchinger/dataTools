# -*- coding: utf-8 -*-
"""
@Author: Sophie Bauchimger, IAU
@Date: Thu Apr 27 15:59:11 2023

Time Lag calculations
"""

import numpy as np
import datetime as dt
import matplotlib.pyplot as plt

from toolpac.age import calculate_lag as cl
from toolpac.conv.times import datetime_to_fractionalyear

def calc_time_lags(c_data, ref_data, subs='SF6 [ppt]', ref_subs = 'SF6catsMLOm'):
    """ Calculate and plot time lag for caribic data wrt mauna loa msmts"""
    t_ref = np.array(datetime_to_fractionalyear(ref_data.index, method='exact'))
    c_ref = np.array(ref_data[ref_subs])
    
    c_obs_tot = np.array(c_data[subs])
    t_obs_tot = np.array(datetime_to_fractionalyear(c_data.index, method='exact'))

    print(f'Calculating lags for {c_data.index.year[0]}')

    lags = []
    for t_obs, c_obs in zip(t_obs_tot, c_obs_tot):
        lag = cl.calculate_lag(t_ref, c_ref, t_obs, c_obs, plot=True)
        lags.append((lag))
    # print('length of lags and mean for ', c_year, ': ', len(lags),  np.nanmean(np.array(lags)))
    return lags

def plot_time_lags(c_data, lags, ref_min, ref_max, ref_subs = 'SF6catsMLOm'):
    """ Plot calculated time lags of a single year of caribic data """
    fig, ax = plt.subplots(dpi=300)
    plt.scatter(c_data.index, lags, marker='+')
    ax.hlines(np.nanmean(np.array(lags)), 
              dt.datetime(c_data.index.year[0], 1, 1), 
              dt.datetime(c_data.index.year[0], 12, 31), 'r', ls='dashed')
    plt.title('CARIBIC {} time lag {} wrt. MLO {} - {}'.format(
        ref_subs, c_data.index.year[0], ref_min, ref_max))
    plt.ylabel('Time lag [yr]')
    plt.xlabel('CARIBIC Measurement time')
    fig.autofmt_xdate()
    return True

#%% Fct calls 
if __name__=='__main__':
    from local_data import Mauna_Loa
    from data_classes import Caribic

    mlo_time_lims = (2000, 2020)
    mlo_MM = Mauna_Loa(years = np.arange(*mlo_time_lims)).df #.df_monthly_mean
    mlo_MM.resample('1M') # add rows for missing months, filled with NaN 
    mlo_MM.interpolate(inplace=True) # linearly interpolate missing data

    t_ref = np.array(datetime_to_fractionalyear(mlo_MM.index, method='exact'))
    c_ref = np.array(mlo_MM['SF6catsMLOm'])

    for c_year in range(2012, 2014):
        c_data = Caribic([c_year]).df
        t_obs_tot = np.array(datetime_to_fractionalyear(c_data.index, method='exact'))
        c_obs_tot = np.array(c_data['SF6 [ppt]'])
    
        lags = []
        for t_obs, c_obs in zip(t_obs_tot, c_obs_tot):
            lag = cl.calculate_lag(t_ref, c_ref, t_obs, c_obs, plot=True)
            lags.append((lag))
    
        fig, ax = plt.subplots(dpi=300)
        plt.hlines(0, np.nanmin(c_data.index), np.nanmax(c_data.index), color='k', ls='dashed')
        plt.scatter(c_data.index, lags, marker='+')
        plt.title('CARIBIC SF$_6$ time lag {} wrt. MLO {} - {}'.format(c_year, *mlo_time_lims))
        plt.ylabel('Time lag [yr]')
        plt.xlabel('CARIBIC Measurement time')
        fig.autofmt_xdate()
