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

from plot.data import scatter_global

def detrend_substance(c_obj, subs, loc_obj, degree=2, save=True, plot=False,
                      as_subplot=False, ax=None, c_pfx=None):
    """ (redefined from C_tools.detrend_subs)
    Remove trend of in measurements of substances using Mauna Loa as reference
    Parameters:
        c_obj (GlobalData/Caribic): measurement dataset
        subs (str): substance e.g. 'sf6'
        loc_obj (LocalData): reference data to detrend on, index=datetime
    """
    detr_data = {}

    if c_pfx: pfxs = [c_pfx]
    else: pfxs = c_obj.pfxs

    if not as_subplot:
        fig, axs = plt.subplots(len(pfxs), dpi=250, figsize=(6,10))
        plt.title(f'{c_obj.source} {subs.upper()}')
    elif ax is None:
        ax = plt.gca()

    for c_pfx, i in zip(pfxs, range(len(pfxs))):
        if not as_subplot: 
            ax = axs[i]
        df = c_obj.data[c_pfx]
        # flight_df = pd.DataFrame(df['Flight number'], index = df.index)
        ref_df = loc_obj.df

        substance = get_col_name(subs, 'Caribic', c_pfx)
        ref_subs = get_col_name(subs, loc_obj.source)

        if not substance in df.columns or not ref_subs in loc_obj.df.columns:
            print('Data not found'); continue

        c_obs = df[substance].values
        t_obs =  np.array(datetime_to_fractionalyear(df.index, method='exact'))

        # ignore reference data earlier and later than two years before/after msmts
        ref_df = ref_df[min(df.index)-dt.timedelta(356*2)
                        : max(df.index)+dt.timedelta(356*2)]
        ref_df.dropna(how='any', subset=ref_subs, inplace=True) # remove NaN rows
        c_ref = ref_df[ref_subs].values
        t_ref = np.array(datetime_to_fractionalyear(ref_df.index, method='exact'))

        popt = np.polyfit(t_ref, c_ref, degree)
        c_fit = np.poly1d(popt) # get popt, then make into fct
        # print(c_fit)

        detrend_correction = c_fit(t_obs) - c_fit(min(t_obs))
        c_obs_detr = c_obs - detrend_correction
        # get variance (?) by substracting offset from 0
        c_obs_delta = c_obs_detr - c_fit(min(t_obs))

        df_detr = pd.DataFrame({f'detr_{substance}' : c_obs_detr,
                                 f'delta_{substance}' : c_obs_delta,
                                 f'fit_{substance}' : c_fit(t_obs)}, 
                                index = df.index)
        # maintain relationship between detr and fit columns
        df_detr[f'fit_{substance}'] = df_detr[f'fit_{substance}'].where(
            ~df_detr[f'detr_{substance}'].isnull(), np.nan)

        detr_data[f'detr_{c_pfx}_{subs}'] = df_detr
        detr_data[f'popt_{c_pfx}_{subs}'] = popt

        if save:
            if not f'detr_{c_pfx}' in c_obj.data.keys():
                c_obj.data[f'detr_{c_pfx}'] = geopandas.GeoDataFrame(
                    c_obj.data[c_pfx]['Flight number'],
                    index = df.index, geometry=df.geometry)
            c_obj.data[f'detr_{c_pfx}'] = c_obj.data[f'detr_{c_pfx}'].join(
                df_detr, lsuffix='DROP').filter(regex="^(?!.*DROP)")

        ax.scatter(t_obs, c_obs, color='orange', label='Flight data')
        ax.scatter(t_ref, c_ref, color='gray', label='MLO data')
        ax.scatter(t_obs, c_obs_detr, color='green', label='trend removed')
        ax.plot(t_obs, c_fit(t_obs), color='black', ls='dashed',
                 label='trendline')

    if plot and not as_subplot:
        plt.legend()
        plt.show()

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

    for c_pfx in caribic.pfxs:
        substs = [x for x in substance_list(c_pfx)
                  if x in ['ch4', 'co2', 'n2o', 'sf6', 'co']]
        f, axs = plt.subplots(int(len(substs)/2), 2,
                              figsize=(10,len(substs)*1.5), dpi=200)
        plt.suptitle(f'Caribic {c_pfx} detrended wrt Mauna Loa')
        for subs, ax in zip(substs, axs.flatten()):
            output = detrend_substance(caribic, subs, mlo_data[subs], save=True,
                              as_subplot=True, ax=ax, c_pfx=c_pfx)
        f.autofmt_xdate()
        plt.tight_layout()
        plt.show()


#%% reset detr datasets 
if __name__=='__main__':
    del caribic.data['detr_GHG']
    del caribic.data['detr_INT']
    del caribic.data['detr_INT2']
