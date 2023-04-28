# -*- coding: utf-8 -*-
"""
@Author: Sophie Bauchimger, IAU
@Date: Fri Apr 28 09:58:15 2023

Defines function to remove trend from measurements 

"""

import numpy as np
import matplotlib.pyplot as plt
import datetime as dt

from toolpac.conv.times import datetime_to_fractionalyear


def detrend_substance(data, substance, ref_data, ref_subs, degree=2, plot=True):
    """ (redefined from C_tools.detrend_subs)
    Remove trend of in measurements of substances such as SF6 using reference data 
    Parameters:
        data: pandas (geo)dataframe of observations to detrend, index=datetime
        substance: str, column name of data (e.g. 'SF6 [ppt]')
        ref_data: pandas (geo)dataframe of reference data to detrend on, index=datetime
        ref_subs: str, column name of reference data
    """
    c_obs = data[substance].values
    t_obs =  np.array(datetime_to_fractionalyear(data.index, method='exact'))

    ref_data.dropna(how='any', subset=ref_subs) 

    # ignore reference data earlier and later than two years before/after msmts
    two_yrs = dt.timedelta(356*2)
    ref_data = ref_data[min(data.index)-two_yrs : max(data.index)+two_yrs]
    ref_data.dropna(how='any', subset=ref_subs, inplace=True) # remove NaN rows
    c_ref = ref_data[ref_subs].values
    t_ref = np.array(datetime_to_fractionalyear(ref_data.index, method='exact'))

    c_fit = np.poly1d(np.polyfit(t_ref, c_ref, 2)) # get popt, then make into fct

    detrend_correction = c_fit(t_obs) - c_fit(min(t_obs))
    c_obs_detr = c_obs - detrend_correction

    if plot:
        plt.figure(dpi=200)
        plt.scatter(t_obs, c_obs, color='orange', label='Flight data')
        plt.scatter(t_ref, c_ref, color='gray', label='MLO data')
        plt.scatter(t_obs, c_obs_detr, color='green', label='detrended')
        plt.plot(t_obs, np.nanmin(c_obs) + detrend_correction, color='black', ls='dashed', label='trendline')
        plt.legend()
        plt.show()

    data[f'detr_{substance}'] = c_obs_detr
    return data
