# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 09:28:22 2023

@author: sophie_bauchinger
"""
#%% Imports
import numpy as np
import sys
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn' - otherwise df[j] = val gives a warning (outliers.outliers)
from pathlib import Path
import datetime as dt

import matplotlib.pyplot as plt

import warnings # using this to suppress a gui backend userwarning, not really advisible
warnings.filterwarnings("ignore", category=UserWarning, module='matplotlib')

sys.path.insert(0, r'C:\Users\sophie_bauchinger\sophie_bauchinger\toolpac_tutorial')
from toolpac_tutorial import Mauna_Loa, Mace_Head, Caribic, Mozart

from toolpac.calc import bin_1d_2d
from toolpac.outliers import outliers
from toolpac.outliers import ol_fit_functions as fct
from toolpac.outliers.outliers import get_no_nan, fit_data
from toolpac.age import calculate_lag as cl
from toolpac.conv.times import datetime_to_fractionalyear, fractionalyear_to_datetime

sys.path.insert(0, r'C:\Users\sophie_bauchinger\sophie_bauchinger\Caribic_data_handling')
from C_filter import filter_outliers
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
    for c_year in range(2005, 2022):
        c_data = caribic_data.select_year(c_year)
        if len(c_data[c_data['SF6 [ppt]'].notna()]) < 1: 
            continue
        else:
            lags = cal_time_lags(c_data, mlo_MM)
            if all(np.isnan(np.array(lags))): 
                print(f'no lags calculated for {c_year}'); continue
            plot_time_lags(c_data, lags, mlo_time_lims)

#%% filter data into stratosphere and troposphere (using n2o as a tracer)
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
    everything with lower n2o than mlo_lim*mlo_fit(frac_year) is flagged (3% cut-off)
    as 'strato' in an initial filtering step 
    """ 
    mlo_lim = 0.97

    # initialise columns to hold strat and trop flags (needs to be done on two lines)
    data = data.assign(strato = np.nan)
    data = data.assign(tropo = np.nan)

    data.loc[data[n2o_col] < mlo_lim * mlo_fit(t_obs_tot), ('strato', 'tropo')] = (True, False)

    # create new dataframe to hold preflagging data
    pre_flagged = pd.DataFrame(data, columns=['Flight number', 'strato', 'tropo'])
    pre_flagged['n2o_pre_flag'] = 0 # initialise flag with zeros
    pre_flagged.loc[data['strato'] == True, 'n2o_pre_flag'] = 1 # set flag indicator for pre-flagged measurements
    print('Result of pre-flagging: \n', pre_flagged.value_counts()) # show results of preflagging
    return data, pre_flagged

def filter_strat_trop(data, crit):
    """ 
    Reconstruction of filter_strat_trop from C_tools (T. Schuck)

    Sort data into stratosphere or troposphere based on outlier statistics 
    with respect to measurements eg. at Mauna Loa Observatory
    
    Returns dataset with new bool columns 'strato' and 'tropo' 
    
    Parameters: 
        data: DataFrame of data to be sorted 
        crit: substance to be used for filtering, eg. n2o or sf6 
    """
    # OUTLIER: Trop / Strat identification using outlier statistics
    if crit == 'n2o':
        mlo_fit = get_mlo_fit(n2o_df)
        ref_data = n2o_df

        n2o_col = caribic_data.get_col_name('n2o') # get column name
        data = data.dropna(how='any', subset=[n2o_col]) # choose only rows where n2o data exists
        t_obs_tot = np.array(datetime_to_fractionalyear(data.index, method='exact'))  # find total observation time as fractional year for fctn calls below

        data, pre_flagged = pre_flag(data, n2o_col, t_obs_tot, mlo_fit) # pre-flagging

        n2o_mxr = data[n2o_col] # measured n2o mixing ratios
        n2o_d_mxr = data['d_N2O [ppb]']
        # print(data.index, data[n2o_col],  pre_flagged.n2o_flag)

        ol_n2o = outliers.find_ol(fct.simple, t_obs_tot, n2o_mxr, n2o_d_mxr, flag = pre_flagged.n2o_pre_flag, 
                              plot=True, limit=0.1, direction = 'n')
        print('\n OL N2O\n', ol_n2o[0].values)
        # ^ 4er tuple, 1st ist liste von OL == 1 / 2 / 3, wenn not outlier dann == 0
        data.loc[(ol_n2o[0] != 0), ('strato', 'tropo')] = (True, False)
        data.loc[(ol_n2o[0] == 0), ('strato', 'tropo')] = (False, True)

    if crit == 'sf6':
        mlo_fit = get_mlo_fit(sf6_df, substance='SF6catsMLOm')
        ref_data = sf6_df

        sf6_col = caribic_data.get_col_name('sf6')
        data = data.dropna(how='any', subset=[sf6_col]) # choose only rows where sf6 data exists
        t_obs_tot = np.array(datetime_to_fractionalyear(data.index, method='exact'))
        data, pre_flagged = pre_flag(data, sf6_col, t_obs_tot, mlo_fit) # pre-flagging
        sf6_mxr = data[sf6_col] # measured n2o mixing ratios
        sf6_d_mxr = data['d_SF6 [ppt]']

        ol_sf6 = outliers.find_ol(fct.simple, t_obs_tot, sf6_mxr, sf6_d_mxr, flag = pre_flagged.n2o_pre_flag, 
                              plot=True, limit=0.1, direction = 'n')
        data.loc[(ol_sf6[0] != 0), ('strato', 'tropo_ol')] = (True, False)
        data.loc[(ol_sf6[0] == 0), ('strato', 'tropo_ol')] = (False, True)

    return data

def get_fct_substance(substance):
    df_func_dict = {
        'sf6': fct.simple,
        'n2o' : fct.simple
        }
    return df_func_dict[substance]

def filter_trop_outliers(data, substance_list):
    """ 
    After sorting data into stratospheric and tropospheric, now sort the 
    tropospheric data into outliers and non-outliers 
    Parameters:
        data: pandas (geo)dataframe
        substance_list: list of strings, substances to be receive a flag 
    """
    # take only tropospheric data 
    for subs in substance_list:
        subs = caribic_data.get_col_name(subs)
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
        # data_flag.loc[data_flag[f'fl_{subst}'] == 0, f'ol_{subst}'] = np.nan

        fit_result = [func(t, *tmp[3]) for t in time]
        # print(len(fit_result), len(data_flag))
        data_flag[f'ol_rel_{subs}'] = data_flag[f'ol_{subs}'] / fit_result

    return data_flag


if __name__=='__main__':
    # loop through years of caribic data
    data_filtered = pd.DataFrame() # initialise full dataframe
    for c_year in range(2005, 2022): 
        print(f'{c_year}')
        c_data = caribic_data.select_year(c_year)
        print('cols:', c_data.columns)

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
    print(data_stratosphere.value_counts)
    data_troposphere = data_filtered.loc[data_filtered['tropo'] == True]

    data_trop_outlier = filter_trop_outliers(data_filtered, ['n2o'])

#%% Detrend data for a specific substance wrt free troposphere ? reference data 
def detrend_substance(data, substance, ref_data, ref_subs, degree=2):
    """ (redefined from C_tools.detrend_subs)
    Remove linear trend of viable substances eg. SF6 using reference data 
    Parameters:
        data: pandas (geo)dataframe of observations to detrend, index=datetime
        substance: str, column name of data (e.g. 'SF6 [ppt]')
        ref_data: pandas (geo)dataframe of reference data to detrend on, index=datetime
        ref_subs: str, column name of reference data
    """
    # ignore reference data earlier and later than two years before/after msmts
    two_yrs = dt.timedelta(356*2)
    ref_data_sliced = ref_data[min(data.index)-two_yrs : max(data.index)+two_yrs]

    c_obs = data[substance].values()
    t_obs =  np.array(datetime_to_fractionalyear(data.index, method='exact'))

    c_ref = ref_data_sliced[ref_subs].values()
    t_ref = np.array(datetime_to_fractionalyear(ref_data_sliced.index, method='exact'))

    ts_fit = np.polyfit(t_ref, c_ref, degree)
    c_fit = np.poly1d(ts_fit)

    detrend_correction = c_fit(t_obs) - c_fit(min(t_obs))
    c_obs_detr = c_obs - detrend_correction

    return c_obs_detr

#%% Plotting Gradient by season


def pl_gradient_by_season(data, substance, tropopause='therm', 
                          min_y=-50, max_y=80, bsize=10, ptsmin=5):
    """ 
    Plotting gradient by season using 1D binned data 
    Parameters:
        data: pandas (geo)dataframe
        substance: str, eg. 'SF6 [ppt]'
        tropopause: str, which tropopause definition to use 
        min_y, max_y: int, defines longitude range to plot
        bsize: int, bin size for 1D binning
        ptsmin: int, minimum number of pts for a bin to be considered 
    """
    var = 1 # select_var_list
    value = 1
    cf = C_tools.get_op('lt')

    c_obs = data[substance].values()
    t_obs =  np.array(datetime_to_fractionalyear(data.index, method='exact'))

    nbins = (max_y - min_y) / bsize
    y_array = min_y + np.arange(nbins) * bsize + bsize * 0.5

    dict_season = {'name_1': 'spring', 'name_2': 'summer', 'name_3': 'autumn', 'name_4': 'winter',
                   'color_1': 'blue', 'color_2': 'orange', 'color_3': 'green', 'color_4': 'red'}

    for s in set(data['season'].tolist()):
        df_sub = data.loc[data['season'] == s]
        y_values = df_sub[yvar].values
        x_values = df_sub[f'detr_{substance}'].values
        dict_season[f'bin1d_{s}'] = bin_1d_2d.bin_1d(x_values, y_values, min_y, max_y, bsize)



#%% 
mlo_fit = get_mlo_fit(n2o_df)
ref_data = n2o_df
data = caribic_data.select_year(2019)

crit = 'n2o'
n2o_col = caribic_data.get_col_name('n2o') # get column name
data = data.dropna(how='any', subset=[n2o_col]) # choose only rows where n2o data exists
t_obs_tot = np.array(datetime_to_fractionalyear(data.index, method='exact'))  # find total observation time as fractional year for fctn calls below
data, pre_flagged = pre_flag(data, n2o_col, t_obs_tot, mlo_fit) # pre-flagging

n2o_mxr = data[n2o_col] # measured n2o mixing ratios
n2o_d_mxr = data['d_N2O [ppb]']
# print(data.index, data[n2o_col],  pre_flagged.n2o_flag)

ol_n2o = outliers.find_ol(fct.simple, t_obs_tot, n2o_mxr, n2o_d_mxr, flag = pre_flagged.n2o_pre_flag, 
                      plot=True, limit=0.1, direction = 'n')
# ^ 4er tuple, 1st ist liste von OL=1, !OL=0

data.loc[(ol_n2o[0] != 0), ('strato', 'tropo')] = (True, False)
data.loc[(ol_n2o[0] == 0), ('strato', 'tropo')] = (False, True)