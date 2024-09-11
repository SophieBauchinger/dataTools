# -*- coding: utf-8 -*-
"""
@Author: Sophie Bauchinger, IAU
@Date Mon Feb 26 14:25:18 2024

"""

import pandas as pd
import numpy as np

from toolpac.conv.times import datetime_to_fractionalyear as dt_to_fy
from toolpac.outliers import outliers
from toolpac.outliers import ol_fit_functions as fct


def get_lin_fit(series, degree=2) -> np.array:  # previously get_mlo_fit
    """ Given one year of reference data, find the fit parameters for
    the substance (col name) """
    year, month = series.index.year, series.index.month
    t_ref = year + (month - 0.5) / 12  # obtain frac year for middle of the month
    mxr_ref = series.values
    fit = np.poly1d(np.polyfit(t_ref, mxr_ref, degree))
    print(f'Fit parameters obtained: {fit}')
    return fit

def pre_flag(data_arr, ref_arr, crit='n2o', limit=0.97, **kwargs) -> pd.DataFrame:
    """ Sort data into strato / tropo based on difference to ground obs.

    Parameters:
        data_arr (pd.Series) : msmt data to be sorted into stratr / trop air
        ref_arr (pd.Series) : reference data to use for filtering (background)
        crit (str) : substance to use for flagging
        limit (float) : tracer mxr fraction below which air is classified
                        as stratospheric

    Returns: dataframe containing index and strato/tropo/pre_flag columns
    """
    data_arr.sort_index(inplace=True)
    df_flag = pd.DataFrame({f'strato_{data_arr.name}': np.nan,
                            f'tropo_{data_arr.name}': np.nan},
                           index=data_arr.index)

    fit = get_lin_fit(ref_arr)
    t_obs_tot = np.array(dt_to_fy(df_flag.index, method='exact'))
    df_flag.loc[data_arr < limit * fit(t_obs_tot),
    (f'strato_{data_arr.name}', f'tropo_{data_arr.name}')] = (True, False)

    df_flag[f'flag_{crit}'] = 0
    df_flag.loc[df_flag[f'strato_{data_arr.name}'] == True, f'flag_{crit}'] = 1

    if kwargs.get('verbose'):
        print('Result of pre-flagging: \n',
              df_flag[f'flag_{crit}'].value_counts())
    return df_flag


def n2o_filter(times, campaign_n2o_data, reference_n2o_data) -> pd.DataFrame:
    """ Filter strato / tropo data based on specific column of N2O mixing ratios. 
    
    alles als pandas series
    
    """

    df_sorted = pd.DataFrame(index=times)
    df_sorted.sort_index(inplace=True)
    
    mxr = campaign_n2o_data  # measured mixing ratios
    d_mxr = None
    t_obs_tot = np.array(dt_to_fy(times, method='exact'))

    # Calculate simple pre-flag
    ref_mxr = reference_n2o_data
    df_flag = pre_flag(mxr, ref_mxr, 'n2o')

    flag = df_flag['flag_n2o'] if 'flag_n2o' in df_flag.columns else None

    strato = 'strato_N2O'
    tropo = 'tropo_N2O'

    fit_function = fct.simple

    ol = outliers.find_ol(fit_function, t_obs_tot, mxr, d_mxr,
                          flag=flag, verbose=False, plot=True, ctrl_plots=True, 
                          limit=0.1, direction='n')
    # ^ 4er tuple, 1st is list of OL == 1/2/3 - if not outlier then OL==0
    df_sorted.loc[(flag != 0 for flag in ol[0]), (tropo, strato)] = (False, True)
    df_sorted.loc[(flag == 0 for flag in ol[0]), (tropo, strato)] = (True, False)

    df_sorted.drop(columns=[s for s in df_sorted.columns
                            if not s.startswith(('Flight', 'tropo', 'strato'))],
                   inplace=True)
    df_sorted = df_sorted.convert_dtypes()

    return df_sorted