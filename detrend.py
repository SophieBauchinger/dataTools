# -*- coding: utf-8 -*-
"""
@Author: Sophie Bauchimger, IAU
@Date: Fri Apr 28 09:58:15 2023

Defines function to remove trend from measurements 

"""

import numpy as np
import geopandas
import matplotlib.pyplot as plt
import datetime as dt

from toolpac.conv.times import datetime_to_fractionalyear
from dictionaries import get_col_name

def detrend_substance(caribic_obj, substance, ref_data, degree=2, plot=True):
    """ (redefined from C_tools.detrend_subs)
    Remove trend of in measurements of substances such as SF6 using reference data 
    Parameters:
        global_obj: GlobalData object, Caribic
            global_obj.data is dictionary where data[pfx] are GeoDataFrames
            with datetime index 
        substance: str, column name of data (e.g. 'sf6')
        ref_data: pandas (geo)dataframe of reference data to detrend on, index=datetime
        ref_subs: str, column name of reference data
    """
    for pfx in caribic_obj.pfxs: 
        df = caribic_obj.data[pfx]

        car_subs = get_col_name(substance, 'Caribic', pfx)
        ref_subs = get_col_name(substance, 'Mauna_Loa')

        if not car_subs in df.columns or not ref_subs in ref_data.columns: 
            print('Data not found'); continue

        c_obs = df[car_subs].values
        t_obs =  np.array(datetime_to_fractionalyear(df.index, method='exact'))
    
        # ignore reference data earlier and later than two years before/after msmts
        ref_data = ref_data[min(df.index)-dt.timedelta(356*2) : max(df.index)+dt.timedelta(356*2)]
        ref_data.dropna(how='any', subset=ref_subs, inplace=True) # remove NaN rows
        c_ref = ref_data[ref_subs].values
        t_ref = np.array(datetime_to_fractionalyear(ref_data.index, method='exact'))

        popt = np.polyfit(t_ref, c_ref, degree)
        c_fit = np.poly1d(popt) # get popt, then make into fct
        print(c_fit)
    
        detrend_correction = c_fit(t_obs) - c_fit(min(t_obs))
        c_obs_detr = c_obs - detrend_correction
        c_obs_delta = c_obs_detr - c_fit(min(t_obs)) # get variance (?) by substracting offset from 0

        data = {f'detr_{car_subs}' : c_obs_detr,
                f'delta_{car_subs}' : c_obs_delta}
    
        if plot:
            plt.figure(dpi=200)
            plt.title(f'{substance.upper()} {pfx}')
            plt.scatter(t_obs, c_obs, color='orange', label='Flight data')
            plt.scatter(t_ref, c_ref, color='gray', label='MLO data')
            plt.scatter(t_obs, c_obs_detr, color='green', label='detrended')
            plt.plot(t_obs, c_fit(t_obs), color='black', ls='dashed', label='trendline')
            plt.legend()
            plt.show()

        # c_obs_delta = 
        gdf_detr = geopandas.GeoDataFrame(data, index = df.index, geometry=df.geometry)
        
        caribic_obj.data[f'{pfx}_{car_subs}_detr'] = gdf_detr

        # df[f'detr_{car_subs}'] = c_obs_detr
        # data[f'delta_detr_substance']
    return gdf_detr

