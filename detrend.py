# -*- coding: utf-8 -*-
"""
@Author: Sophie Bauchimger, IAU
@Date: Fri Apr 28 09:58:15 2023

Defines function to remove trend from measurements

"""

import numpy as np
import geopandas
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt

from toolpac.conv.times import datetime_to_fractionalyear
from dictionaries import get_col_name

def detrend_substance(c_obj, subs, loc_obj, degree=2, plot=True):
    """ (redefined from C_tools.detrend_subs)
    Remove trend of in measurements of substances using Mauna Loa as a reference
    Parameters:
        c_obj: GlobalData object, Caribic
            c_obj.data is dictionary where data[pfx] are GeoDataFrames
        substance: str, column name of data (e.g. 'sf6')
        loc_obj: (geo)dataframe of reference data to detrend on, index=datetime
    """
    detr_data = {}
    for c_pfx in c_obj.pfxs:
        df = c_obj.data[c_pfx]
        flight_df = pd.DataFrame(df['Flight number'], index = df.index)
        ref_df = loc_obj.df

        car_subs = get_col_name(subs, 'Caribic', c_pfx)
        ref_subs = get_col_name(subs, loc_obj.source)

        if not car_subs in df.columns or not ref_subs in loc_obj.df.columns:
            print('Data not found'); continue

        c_obs = df[car_subs].values
        t_obs =  np.array(datetime_to_fractionalyear(df.index, method='exact'))

        # ignore reference data earlier and later than two years before/after msmts
        ref_df = ref_df[min(df.index)-dt.timedelta(356*2)
                        : max(df.index)+dt.timedelta(356*2)]
        ref_df.dropna(how='any', subset=ref_subs, inplace=True) # remove NaN rows
        c_ref = ref_df[ref_subs].values
        t_ref = np.array(datetime_to_fractionalyear(ref_df.index, method='exact'))

        popt = np.polyfit(t_ref, c_ref, degree)
        c_fit = np.poly1d(popt) # get popt, then make into fct
        print(c_fit)

        detrend_correction = c_fit(t_obs) - c_fit(min(t_obs))
        c_obs_detr = c_obs - detrend_correction
        # get variance (?) by substracting offset from 0
        c_obs_delta = c_obs_detr - c_fit(min(t_obs))

        data = {f'detr_{car_subs}' : c_obs_detr,
                f'delta_{car_subs}' : c_obs_delta}

        if plot:
            plt.figure(dpi=200)
            plt.title(f'{subs.upper()} {c_pfx}')
            plt.scatter(t_obs, c_obs, color='orange', label='Flight data')
            plt.scatter(t_ref, c_ref, color='gray', label='MLO data')
            plt.scatter(t_obs, c_obs_detr, color='green', label='trend removed')
            plt.plot(t_obs, c_fit(t_obs), color='black', ls='dashed',
                     label='trendline')
            plt.legend()
            plt.show()

        gdf_detr = geopandas.GeoDataFrame(data, index = df.index,
                                          geometry=df.geometry)
        gdf_detr = gdf_detr.join(flight_df)

        c_obj.data[f'detr_{c_pfx}_{subs}'] = gdf_detr
        detr_data[f'detr_{c_pfx}_{subs}'] = gdf_detr

    return detr_data

#%% Fctn calls
if __name__=='__main__':
    from data import Caribic, Mauna_Loa
    from dictionaries import substance_list
    year_range = (2000, 2020)

    calc_caribic = False
    if calc_caribic:
        caribic = Caribic(year_range, pfxs = ['GHG', 'INT', 'INT2'])

    calc_mlo = False
    if calc_mlo:
        year_range = range(1980, 2021)
        mlo_data = {subs : Mauna_Loa(year_range, substance=subs) for subs
                    in substance_list('MLO')}

    co2_detr = detrend_substance(caribic, 'co2', mlo_data['co2'])
    n2o_detr = detrend_substance(caribic, 'n2o', mlo_data['n2o'])

