# -*- coding: utf-8 -*-
""" Tropopause definition implementation (and statistics?)

@Author: Sophie Bauchinger, IAU
@Date: Fri Dec 20 17:43:00 2024
"""
import datetime as dt
import numpy as np
from pathlib import Path
import pandas as pd
from scipy.interpolate import interp1d

from toolpac.conv.times import datetime_to_fractionalyear as dt_to_fy 
from toolpac.outliers import outliers 

import dataTools.dictionaries as dcts
from dataTools import tools
from dataTools.data.local import MaunaLoa


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
def o3_baseline_filter(df, o3_coord, **kwargs) -> pd.DataFrame:
    """ Use climatology of Ozone from somewhere (?) - seasonality? - and use as TP filter. """
    raise NotImplementedError('O3 Baseline filter has not yet been implemented')

# --- KIT Ozone tropopause calculation
# ------------------------------------------------------------------------------

def load_tp_hght_data(path: Path, sep=";", v_scal=1) -> dict:
    """
    Load rel. TP height data from csv file.

    Parameters
    ----------
    path : str or Path
        DESCRIPTION.
    sep : str, optional
        DESCRIPTION. The default is ";".
    v_scal : int, optional
        DESCRIPTION. The default is 1.

    Returns
    -------
    dict
        height relative to tropopause data.
    """
    with open(path, "r") as file_obj:
        tp_data = file_obj.readlines()

    for i, line in enumerate(tp_data):
        tp_data[i] = line.strip().rsplit(sep)

    tp_hght = np.array(tp_data[0][1:], dtype=float)
    time = []
    ozone = []

    for line in tp_data[1:-1]:
        time.append(int(line[0]))
        ozone.append([float(o) * v_scal for o in line[1:]])

    time = np.array(time)
    ozone = np.array(ozone)

    result = {}
    result["tp_hght"] = tp_hght
    result["montly_avg"] = {}
    result["montly_avg"]["month"] = time
    result["montly_avg"]["ozone"] = ozone

    # convert time given as monthly mean to days of year
    doy = time * 365 / 12 - (365 / 12 / 2)  # ignore leap years and so on...
    time = np.array(list(range(1, 366)))
    tmp_o3 = np.zeros([len(time), len(tp_hght)])
    for i in range(len(tp_hght)):  # interpolate O3 for each TP height
        f_ip = interp1d(
            doy,
            ozone[:, i],
            kind="linear",
            bounds_error=False,
            fill_value="extrapolate",
        )
        tmp_o3[:, i] = f_ip(time)
    ozone = tmp_o3

    result["DOY_interp"] = {}
    result["DOY_interp"]["DOY"] = time
    result["DOY_interp"]["ozone"] = ozone

    return result

def calc_o3tp_relhght(
    tpdata: dict,
    v_ozone: float,
    sel_month: int,
    sel_day_of_month=False,
    vmiss=9999,
    v_lessthanmin=-9999,
    v_ozone_min=60.0,
    hreltp_min=-1.5,
) -> float:
    """
    Find relative tropopause height for a given ozone value, v_ozone.
    result is based on linear interpolation of rel. TP heights that correspond
    to the ozone values closest to v_ozone.

    Parameters
    ----------
    tpdata : dict
        DESCRIPTION.
    v_ozone : float
        DESCRIPTION.
    sel_month : int
        DESCRIPTION.
    sel_day : TYPE, optional
        if specified, monthly tpdata is interpolated linearly to days of
            year. The default is False.
    vmiss : TYPE, optional
        DESCRIPTION. The default is 9999.
    v_ozone_min : TYPE, optional
        DESCRIPTION. The default is 60.0.
    hreltp_min : TYPE, optional
        DESCRIPTION. The default is -1.5.

    Returns
    -------
    float
        height relative to tropopause or VMISS.
    """
    if v_ozone <= v_ozone_min:
        return v_lessthanmin  # v_ozone too low, return vmiss

    # first line: TP heights
    tp_hght = tpdata["tp_hght"]

    if sel_day_of_month:
        sel_time = (
            dt.datetime(2010, sel_month, sel_day_of_month) - dt.datetime(2010, 1, 1)  # doy from time-
        ).days + 1  # delta object
        time = tpdata["DOY_interp"]["DOY"]
        ozone = tpdata["DOY_interp"]["ozone"]
    else:
        sel_time = sel_month
        time = tpdata["montly_avg"]["month"]
        ozone = tpdata["montly_avg"]["ozone"]

    # find corresponding time index
    ix_t = np.arange(len(time))[np.where(time == sel_time)]

    # search corresponding ozone array for adjacent values
    sel_ozone = ozone[ix_t]
    sel_ozone = sel_ozone.reshape(len(sel_ozone[0]))  # flatten...
    ix_close = np.array([np.argmin(np.abs(sel_ozone - v_ozone))], dtype=int)

    if ix_close in (0, len(sel_ozone) - 1):
        # no bracketing value! interpolation not possible,
        return vmiss

    # there is a bracketing value...
    # find the next closest / = bracketing o3 value
    o3_close = sel_ozone[ix_close]
    sel_ozone[ix_close] = vmiss
    ix_brack = np.array([np.argmin(np.abs(sel_ozone - v_ozone))], dtype=int)

    # interpolate H_rel_TP based on rel. distance to given ozone value
    rel_dist_o3 = np.abs(o3_close - v_ozone) / np.abs(o3_close - sel_ozone[ix_brack])
    result = (tp_hght[ix_close] + (tp_hght[ix_brack] - tp_hght[ix_close]) * rel_dist_o3)[0]

    if result <= hreltp_min:
        result = v_lessthanmin  # h_rel_TP too low, return vmiss

    return result

def calc_HrelTP(df, o3_coord, dropna=False):
    """ Calculate `o3tp_relheight` for the given ozone data.
    
    Parameters:
        df (pd.DataFrame)
        o3_coord (dcts.Substance | dcst.Coordinate)
    """
    tpdata = load_tp_hght_data(path = dcts.get_path() + r"misc_data\reference_data\O3_climatology_Hohenpeissenberg_for_H_rel_TP.csv")
    v_ozone = df[o3_coord.col_name]
    months = df.index.month
    o3tp_relheight = [calc_o3tp_relhght(tpdata, v, m) for v,m in zip(v_ozone.values, months)]

    hreltp = pd.Series(o3tp_relheight, 
                       name = 'H_rel_TP', 
                       index = df.index)
    
    # Replace value of vmiss with NaN (NB: setting vmiss=nan breaks the function)
    hreltp.mask(hreltp == 9999., np.nan, inplace = True)
    
    tropo = 'tropo_H_rel_TP'
    strato = 'strato_H_rel_TP'
    
    df_sorted = pd.DataFrame({strato: pd.Series(np.nan, dtype=object),
                              tropo: pd.Series(np.nan, dtype=object)},
                             index=df.index)
    
    df_sorted.loc[hreltp.lt(0), (strato, tropo)] = (False, True)
    df_sorted.loc[hreltp.gt(0), (strato, tropo)] = (True, False)
    
    hreltp.mask(hreltp == -9999., np.nan, inplace = True)

    return hreltp, df_sorted, v_ozone

if __name__=='__main__':
    from dataTools.data.Caribic import Caribic
    caribic = Caribic()
    [o3_subs] = caribic.get_substs(col_name = 'int_O3')
    hreltp, df_sorted, v_ozone = calc_HrelTP(caribic.df, o3_subs)

#%% Various tropopause-related tools  
def tropo_strato_ratios(df_sorted, tps, **kwargs) -> tuple[pd.DataFrame]: 
    """ Calculates the ratio of tropospheric / stratospheric datapoints for the given tropopause definitions.
    
    Args: 
        df_sorted (pd.DataFrame): Dataframe with tropo/strato bool values. 
        tps (list[dcts.Coordinate]): Tropopause definitions to calculate ratios for
    
    Returns a dataframe with tropospheric (True) and stratospheric (False) flags per TP definition. 
    """
    # Select data 
    tropo_cols = ['tropo_' + tp.col_name for tp in tps
                  if 'tropo_' + tp.col_name in df_sorted]

    df = df_sorted[tropo_cols]
    shared_indices = tools.get_shared_indices(df, tps)
    df = df[df.index.isin(shared_indices)]

    # Get counts 
    tropo_counts = df[df == True].count(axis=0)
    strato_counts = df[df == False].count(axis=0)

    count_df = pd.DataFrame({True: tropo_counts, False: strato_counts}).transpose()
    count_df.dropna(axis=1, inplace=True)
    count_df.rename(columns={c: c[6:] for c in count_df.columns}, inplace=True)

    # Calculate ratios 
    ratio_df = pd.DataFrame(columns=count_df.columns, index=['ratios'])
    ratios = [count_df[c][True] / count_df[c][False] for c in count_df.columns]
    ratio_df.loc['ratios'] = ratios  # set col

    return count_df, ratio_df

def seasonal_tropospheric_average(GlobalObject, subs, tps) -> dict[dict]:
    """ Returns seasonal average values of `subs`
    for the given tropopause definitions as a nested dictionary. """
    out_dict = {}
    for tp in tps:
        out_dict[tp.col_name] = {}
        for s in set(GlobalObject.df.season.values):
            df = GlobalObject.sel_season(s).sel_tropo(tp).df
            mean_tp_s = df[subs.col_name].mean()
            out_dict[tp.col_name][s] = mean_tp_s
    return out_dict