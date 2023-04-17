# -*- coding: utf-8 -*-
"""
@Author: Sophie Bauchimger, IAU
@Date: Wed Feb 22 14:13:17 2023

Outlier Filtering of Caribic data
Applying timelag function to Caribic data
"""
import numpy as np
import matplotlib.pyplot as plt

from toolpac.outliers import outliers, ol_fit_functions
from toolpac.age import calculate_lag as cl
from toolpac.conv.times import datetime_to_fractionalyear

from toolpac_tutorial import Mauna_Loa, Caribic

#%% Outlier flag
for y in range(2008, 2010):
    for dir_val in ['np', 'p', 'n']:
        data = Caribic([y]).df
        sf6_mxr = data['SF6; SF6 mixing ratio; [ppt]\n']
        ol = outliers.find_ol(ol_fit_functions.simple, data.index, sf6_mxr, None, None, 
                              plot=True, limit=0.1, direction = dir_val)

#%% Time Lag calculations
# mlo_tot = pd.concat([monthly_mean(mlo_data(mlo_file_MM, year = y), first_of_month=True)
#                      for y in range(2000, 2014)])
# mlo_tot = monthly_mean(mlo_data(year = np.arange(2004, 2014)))
# pd.concat([mlo_data(year = y) for y in range(2000, 2014)])
mlo_time_lims = (2000, 2020)
mlo_MM = Mauna_Loa(years = np.arange(*mlo_time_lims)).df #.df_monthly_mean
mlo_MM.resample('1M') # add rows for missing months, filled with NaN 
mlo_MM.interpolate(inplace=True) # linearly interpolate missing data

t_ref = np.array(datetime_to_fractionalyear(mlo_MM.index, method='exact'))
c_ref = np.array(mlo_MM['SF6catsMLOm'])

for c_year in range(2012, 2014):
    c_data = Caribic([c_year]).df
    t_obs_tot = np.array(datetime_to_fractionalyear(c_data.index, method='exact'))
    c_obs_tot = np.array(c_data['SF6; SF6 mixing ratio; [ppt]\n'])

    lags = []
    for t_obs, c_obs in zip(t_obs_tot, c_obs_tot):
        lag = cl.calculate_lag(t_ref, c_ref, t_obs, c_obs, plot=True)
        lags.append((lag))

    fig, ax = plt.subplots(dpi=300)
    plt.scatter(c_data.index, lags, marker='+')
    plt.title('CARIBIC SF$_6$ time lag {} wrt. MLO {} - {}'.format(c_year, *mlo_time_lims))
    plt.ylabel('Time lag [yr]')
    plt.xlabel('CARIBIC Measurement time')
    fig.autofmt_xdate()

#%%% ´Redraw outlier plots
# import matplotlib.pyplot as plt
# import numpy as np
# from toolpac.outliers.outliers import get_no_nan, fit_data

# mxr = np.array(sf6_mxr)
# flag = ol[0]

# from toolpac.conv.times import datetime_to_fractionalyear
# time = data.index
# time_datetime = time
# time = datetime_to_fractionalyear(time_datetime, method='exact')

# d_mxr = None
# func = ol_fit_functions.quadratic

# cf_fit = plt.figure()
# cf_ax1 = cf_fit.add_subplot(111)
# # cf_ax1.set_title(subst)
# plt.xlabel('time_delta')
# plt.ylabel('mxr')
# #cf_ax1.set_xlim(np.nanmin(time) - 0.1, np.nanmax(time) + 0.1)
# #cf_ax1.set_ylim(np.nanmin(mxr) * 0.9, np.nanmax(mxr) * 1.1)
# cf_ax1.scatter(time, mxr, color='lightgray')

# fl_mxr = [mxr[i] if flag[i] != 0 else None for i in range(len(mxr))]
# # needed for points with flag 0 not to be overplotted
# cf_ax1.scatter(time, fl_mxr, c=flag)
# tmp_time, tmp_mxr, tmp_d_mxr = get_no_nan(time, mxr, d_mxr, flagged=True)
# popt1 = fit_data(func, tmp_time, tmp_mxr, tmp_d_mxr)
# yfit = func(np.array(time), *popt1)
# cf_ax1.plot(time, yfit, color='red', label='no flagging', linewidth=2)
# tmp_time, tmp_mxr, tmp_d_mxr = get_no_nan(time, mxr, d_mxr, flag=flag, flagged=False)
# popt1 = fit_data(func, tmp_time, tmp_mxr, tmp_d_mxr)
# yfit = func(np.array(time), *popt1)
# cf_ax1.plot(time, yfit, color='black', label='baseline final', linewidth=2)
# cf_fit.legend()
# cf_fit.show()