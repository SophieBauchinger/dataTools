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

def calc_time_lags(c_obj, ref_obj, years, substance='sf6', pfx='GHG', 
                   ref_min=2003, ref_max=2020, plot_yr=False, plot_all=True, save=True, verbose=False):
    """ Calculate and plot time lag for caribic data wrt mauna loa msmts for specified years 
    c_obj (Caribic)
    ref_obj (LocalData)
    """
    subs = get_col_name(substance, c_obj.source, pfx)
    if subs == None: return 
    df_tot = c_obj.data[pfx]
    df_tot= df_tot[~pd.isna(df_tot[subs])] # getting rid of nan stuff

    lag_index = df_tot[(df_tot.index.year >= min(years)) & 
                                (df_tot.index.year <= max(years))]

    all_lags = []
    for yr in years:
        if yr not in df_tot.index.year: # break if there is no data
            print(f'No valid non-nan data found in {yr} for {subs}'); continue
        else: print(f'Calculating lags for {pfx}, {substance} in {yr}')

        ref_df = ref_obj.df
        ref_df.resample('1M') # add rows for missing months, filled with NaN 
        ref_df.interpolate(inplace=True) # linearly interpolate missing data

        df = df_tot[df_tot.index.year == yr]

        ref_subs = get_col_name(substance, ref_obj.source)
        c_ref = np.array(ref_df[ref_subs])
        t_ref = np.array(datetime_to_fractionalyear(ref_df.index, method='exact'))

        
        c_obs_tot = np.array(df[subs])
        t_obs_tot = np.array(datetime_to_fractionalyear(df.index, method='exact'))

        lags = []
        for t_obs, c_obs in zip(t_obs_tot, c_obs_tot):
            lag = cl.calculate_lag(t_ref, c_ref, t_obs, c_obs, plot=True)
            lags.append((lag))

        if verbose: print('length of lags and mean for {yr} : {len(lags)} {np.nanmean(np.array(lags))}')
        if plot_yr: plot_time_lags(df, lags, [yr], ref_min, ref_max, subs)

        all_lags.extend(lags)

    lag_index = df_tot[(df_tot.index.year >= min(years)) & 
                       (df_tot.index.year <= max(years))].index

    col_name = f'lag_{substance} [yr]'
    df_lags = pd.DataFrame(all_lags, index = lag_index, columns = [col_name])
    if plot_all: plot_time_lags(df_lags, all_lags, years)

    if save: # save in data dictionary 
        if f'lag_{pfx}' not in c_obj.data.keys(): # new lag df for this pfx, create the dict entry  
            c_obj.data[f'lag_{pfx}'] = df_lags 
        elif col_name not in c_obj.data[f'lag_{pfx}'].columns: # new substance
            combined_df = c_obj.data[f'lag_{pfx}'].join(df_lags) 
            c_obj.data[f'lag_{pfx}'] = combined_df
        else: # overwrite column for current substance # [f'lag_{substance} [yr]'] = all_lags
            c_obj.data[f'lag_{pfx}'].merge(df_lags) 
    return df_lags


def plot_time_lags(df, lags, years, ref_min=2003, ref_max=2020, subs = 'sf6'):
    """ Plot calculated time lags of a single year of caribic data """
    fig, ax = plt.subplots(dpi=300)
    plt.scatter(df.index, lags, marker='+')
    ax.hlines(np.nanmean(np.array(lags)), 
              dt.datetime(min(years), 1, 1), 
              dt.datetime(max(years), 12, 31), 
              'r', ls='dashed', label = 'Mean')
    plt.title('{} CARIBIC time lag wrt. MLO {} - {}'.format(
        subs.upper(), ref_min, ref_max))
    plt.ylabel('Time lag [yr]')
    plt.xlabel('CARIBIC Measurement time')
    plt.legend()
    fig.autofmt_xdate()
    plt.show()
    return 

#%% Fct calls
if __name__=='__main__':
    from dictionaries import substance_list
    calc_caribic = False
    if calc_caribic: 
        from data import Caribic, Mauna_Loa, Mace_Head, Mozart
        year_range = range(2000, 2018)
        mlo_data = {subs : Mauna_Loa(year_range, substance=subs) for subs in substance_list('MLO')}
        caribic = Caribic(year_range, pfxs = ['GHG', 'INT', 'INT2']) # 2005-2020
        mhd = Mace_Head() # only 2012 data available
        mzt = Mozart(year_range) # only available up to 2008

    lag_plot= True

    calc_time_lags(caribic, mlo_data['sf6'], range(2005, 2020), substance = 'sf6', plot_all=lag_plot)
    calc_time_lags(caribic, mlo_data['n2o'], range(2005, 2020), substance = 'n2o', pfx='INT2', plot_all=lag_plot)

    calc_time_lags(caribic, mlo_data['n2o'], range(2005, 2020), substance = 'n2o', plot_all=lag_plot)
    calc_time_lags(caribic, mlo_data['co2'], range(2005, 2020), substance = 'co2', pfx='INT2', plot_all=lag_plot)
