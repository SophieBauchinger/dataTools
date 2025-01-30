# -*- coding: utf-8 -*-
""" Tropopause definition implementation (and statistics?)

@Author: Sophie Bauchinger, IAU
@Date: Fri Dec 20 17:43:00 2024
"""
import numpy as np
import pandas as pd

from toolpac.conv.times import datetime_to_fractionalyear as dt_to_fy 
from toolpac.outliers import outliers 

import dataTools.dictionaries as dcts
from dataTools import tools
from dataTools.data.local import MaunaLoa

# What do I need for the N2O baseline filter? 
# 1. the right data

#%% N2O statistical filter
def n2o_baseline_filter(df, n2o_coord, **kwargs) -> tuple[pd.DataFrame]:
    """ Statistically filter strato / tropo data based on specific column of N2O mixing ratios. """
    # Get reference dataset
    ref_years = np.arange(min(df.index.year) - 2, max(df.index.year) + 3)
    loc_obj = MaunaLoa(ref_years) if not kwargs.get('loc_obj') else kwargs.get('loc_obj')
    ref_subs = dcts.get_subs(substance='n2o', ID=loc_obj.ID)  # dcts.get_col_name(subs, loc_obj.source)

    if kwargs.get('verbose'):
        print(f'N2O sorting: {n2o_coord} ')

    n2o_column = n2o_coord.col_name

    df_sorted = pd.DataFrame(index=df.index)
    if 'Flight number' in df.columns: df_sorted['Flight number'] = df['Flight number']
    df_sorted[n2o_column] = df[n2o_column]

    if f'd_{n2o_column}' in df.columns:
        df_sorted[f'd_{n2o_column}'] = df[f'd_{n2o_column}']
    if f'detr_{n2o_column}' in df.columns:
        df_sorted[f'detr_{n2o_column}'] = df[f'detr_{n2o_column}']

    df_sorted.sort_index(inplace=True)
    df_sorted.dropna(subset=[n2o_column], inplace=True)

    mxr = df_sorted[n2o_column]  # measured mixing ratios
    d_mxr = None if f'd_{n2o_column}' not in df_sorted.columns else df_sorted[f'd_{n2o_column}']
    t_obs_tot = np.array(dt_to_fy(df_sorted.index, method='exact'))

    # Check if units of data and reference data match, if not change data
    if str(n2o_coord.unit) != str(ref_subs.unit):
        if kwargs.get('verbose'): print(f'Note units do not match: {n2o_coord.unit} vs {ref_subs.unit}')

        if n2o_coord.unit == 'mol mol-1':
            mxr = tools.conv_molarity_PartsPer(mxr, ref_subs.unit)
            if d_mxr is not None: d_mxr = tools.conv_molarity_PartsPer(d_mxr, ref_subs.unit)
        elif n2o_coord.unit == 'pmol mol-1' and ref_subs.unit == 'ppt':
            pass
        else:
            raise NotImplementedError(f'No conversion between {n2o_coord.unit} and {ref_subs.unit}')

    # Calculate simple pre-flag
    ref_mxr = loc_obj.df.dropna(subset=[ref_subs.col_name])[ref_subs.col_name]
    df_flag = tools.pre_flag(mxr, ref_mxr, 'n2o', **kwargs)
    flag = df_flag['flag_n2o'].values if 'flag_n2o' in df_flag.columns else None

    strato = f'strato_{n2o_column}'
    tropo = f'tropo_{n2o_column}'

    fit_function = dcts.lookup_fit_function('n2o')

    ol = outliers.find_ol(fit_function, t_obs_tot, mxr, d_mxr,
                            flag=flag, 
                            verbose=kwargs.get('verbose', False), 
                            plot=kwargs.get('plot', False), 
                            ctrl_plots=False,
                            limit=kwargs.get('ol_limit', 0.1), 
                            direction='n')
    # ^tuple, 1st is list of OL == 1/2/3 - if not outlier then OL==0
    # flag, residual, warning, popt1, baseline
    df_sorted.loc[(flag != 0 for flag in ol[0]), (tropo, strato)] = (False, True)
    df_sorted.loc[(flag == 0 for flag in ol[0]), (tropo, strato)] = (True, False)

    n2o_df = pd.DataFrame({
        f'{n2o_column}' : mxr, 
        f'{n2o_column}_flag' : ol[0], 
        f'{n2o_column}_residual' : ol[1],
        f'{n2o_column}_baseline' : ol[4]})
    if d_mxr is not None: 
        n2o_df[f'd_{n2o_column}'] = d_mxr
    
    df_sorted.drop(columns=[s for s in df_sorted.columns
                            if not s.startswith(('Flight', 'tropo', 'strato'))],
                    inplace=True)
    df_sorted = df_sorted.convert_dtypes()

    return df_sorted, n2o_df

def total_n2o_df(df, n2o_coordinates, **kwargs):
    """ Combine df_sorted and baseline data from multiple N2O data sources. """
    
    n2o_sorted = pd.DataFrame(df['Flight number'] if 'Flight number' in df.columns else None,
                             index=df.index)
    n2o_df = pd.DataFrame() 
    
    # N2O filter
    for tp in n2o_coordinates:
        n2o_tp_sorted, n2o_tp_df = n2o_baseline_filter(n2o_coord=tp, **kwargs)
        if 'Flight number' in n2o_tp_sorted.columns:
            n2o_tp_sorted.drop(columns=['Flight number'], inplace=True)  # del duplicate col

        # Combine data
        n2o_sorted = pd.concat([n2o_sorted, n2o_tp_sorted], axis=1)
        n2o_df = n2o_df.combine_first(n2o_tp_df)

    n2o_sorted = n2o_sorted.convert_dtypes()
    
    #!!! For futher use - add all columns in n2o_df with '_residual' and '_baseline' to self.df
    return n2o_sorted, n2o_df

#%% Ozone
def o3_filter_lt60(df, o3_subs) -> pd.DataFrame:
    """ Flag ozone mixing ratios below 60 ppb as tropospheric. """
    o3_sorted = pd.DataFrame(index=df.index)
    o3_sorted.loc[df[o3_subs.col_name].lt(60),
    (f'strato_{o3_subs.col_name}', f'tropo_{o3_subs.col_name}')] = (False, True)
    return o3_sorted

# TODO: implement o3_baseline_filter
def o3_baseline_filter(self, **kwargs) -> pd.DataFrame:
    """ Use climatology of Ozone from somewhere (?) - seasonality? - and use as TP filter. """
    raise NotImplementedError('O3 Baseline filter has not yet been implemented')
