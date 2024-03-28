# -*- coding: utf-8 -*-
"""
@Author: Sophie Bauchinger, IAU
@Date: Wed Jul  5 14:41:45 2023

Filtering of data in tropospheric / stratospheric origin depending on TP definition

Functions return df_sorted which contains boolean strato* / tropo* columns
which are named according to TP definition and parameters used to create them
"""
import numpy as np
import pandas as pd
# default='warn' - otherwise df[j] = val gives a warning (outliers.outliers)
pd.options.mode.chained_assignment = None
# suppress a gui backend userwarning
import matplotlib.pyplot as plt
import warnings; warnings.filterwarnings("ignore", category=UserWarning,
                                         module='matplotlib')

from toolpac.outliers import outliers
from toolpac.outliers.outliers import fit_data
from toolpac.conv.times import datetime_to_fractionalyear as dt_to_fy

from dataTools import tools
import dataTools.dictionaries as dcts

# from tools import get_lin_fit, assign_t_s

filter_types = {
    'chem' : ['n2o', 'o3'], # 'crit'
    'therm' : ['therm'],
    'dyn' : ['1.5pvu', '2pvu', '3.5pvu'], # 'pvu'
    }

coordinates = {
    'pt' : 'Potential temperature [K]',
    'dp' : 'Pressure difference [hPa]',
    'z' : 'Geopotential height [km]',
    }

#%% Troposphere / Stratosphere sorting - function definitions

# Sort trop / strat using mixing ratios relative to trop. background value
def pre_flag(glob_obj, ref_obj, crit='n2o', limit = 0.97, ID = 'GHG',
             save=True, verbose=False, subs_col=None):
    """ Sort data into strato / tropo based on difference to ground obs.

    Returns dataframe containing index and strato/tropo/pre_flag columns

    Parameters:
        glob_obj (GlobalData) : msmt data to be sorted into stratr / trop air
        ref_obj (LocalData) : reference data to use for filtering (background)
        crit (str) : substance to use for flagging
        limit (float) : tracer mxr fraction below which air is classified
                        as stratospheric
        ID (str) : e.g. 'GHG', specify the caribic datasource
        save (bool): add result to glob_obj
    """
    state = f'pre_flag: crit={crit}, ID={ID}\n'
    if glob_obj.source=='Caribic':
        df = glob_obj.data[ID].copy()
    else: df = glob_obj.df.copy()
    df.sort_index(inplace=True)

    if subs_col is not None: substance = subs_col
    else: substance = dcts.get_col_name(crit, glob_obj.source, ID)
    if not substance: raise ValueError(state+'No {crit} data in {ID}')

    fit = tools.get_lin_fit(ref_obj.df, dcts.get_col_name(crit, ref_obj.source))

    df_flag = pd.DataFrame({f'strato_{crit}':np.nan,
                            f'tropo_{crit}':np.nan},
                           index=df.index)

    t_obs_tot = np.array(dt_to_fy(df_flag.index, method='exact'))
    df_flag.loc[df[substance] < limit * fit(t_obs_tot),
           (f'strato_chem_{crit}', f'tropo_chem_{crit}')] = (True, False)

    df_flag[f'flag_{crit}'] = 0
    df_flag.loc[df_flag[f'strato_chem_{crit}'] == True, f'flag_{crit}'] = 1

    if verbose: print('Result of pre-flagging: \n',
                      df_flag[f'flag_{crit}'].value_counts())
    if save and glob_obj.source == 'Caribic':
        glob_obj.data[ID][f'flag_{crit}'] = df_flag[f'flag_{crit}']

    return df_flag

# Sort trop / strat using tracer mixing ratio
def chemical(glob_obj, crit='n2o', ID='GHG', ref_obj=None, detr=True,
             verbose = False, plot=False, limit=0.97, subs=None, **kwargs):
    """ Returns DataFrame with bool columns 'strato' and 'tropo'.

    Reconstruction of filter_strat_trop from C_filter (T. Schuck)
    Sort data into stratosphere or troposphere based on outlier statistics
    with respect to measurements eg. at Mauna Loa Observatory
    Resulting dataframe only contains rows where filtering was possible.

    Parameters:
        glob_obj (GlobalData) : measurement data to be sorted into
                                stratospheric or tropospheric air
        ref_obj (LocalData) : reference data to use for filtering (background)
        crit (str): substance to be used for filtering, eg. n2o
        save (bool): whether to save the strat / trop filtered data in glob_obj
        verbose (bool)

    Data availability:
        GHG: 'n2o'
        INT: 'o3'
        INT2: 'o3', 'n2o'
    """
    if crit not in set([i.short_name for i in dcts.get_substances(ID=ID)]):
        default_crit = {'GHG' : 'n2o', 'INT' : 'o3', 'INT2' : 'n2o'}
        print(f'{crit} not available in {ID}, using {default_crit[ID]}')
        crit = default_crit[ID]

    state = f'filter_strat_trop: crit={crit}, ID={ID}\n'
    tropo = f'tropo_chem_{crit}'
    strato = f'strato_chem_{crit}'

    data = glob_obj.data[ID].copy()
    df_sorted = pd.DataFrame({strato:pd.Series(np.nan, dtype='float'),
                              tropo:pd.Series(np.nan, dtype='float')},
                             index=data.index)

    if crit == 'o3': # INT and INT2 have coordinates relative to O3 TP (Zahl 2003)
        cols = {'INT' : 'int_h_rel_TP [km]',
                'INT2' : 'int_CARIBIC2_H_rel_TP [km]'}
        col, coord = cols[ID], 'z'

        df_sorted.loc[tools.assign_t_s(data[col], 't', coord),
                    (strato, tropo)] = (False, True)
        df_sorted.loc[tools.assign_t_s(data[col], 's', coord),
                    (strato, tropo)] = (True, False)

        df_sorted.dropna(subset=tropo, inplace=True) # remove rows without data
        df_sorted.sort_index(inplace=True)

        if plot:
            plot_sorted(glob_obj, df_sorted, crit, ID, subs=subs, detr=detr, **kwargs)

    if crit == 'n2o' and ID in ['GHG', 'INT2']:
        substance = dcts.get_col_name(crit, glob_obj.source, ID) # get column name
        if detr:
            if not 'detr_'+substance in data.columns:
                glob_obj.detrend(subs, save=True)
            substance = 'detr_'+substance
        df_sorted[substance] = data[substance]
        if f'd_{substance}' in data.columns:
            df_sorted[f'd_{substance}'] = data[f'd_{substance}']

        # Calculate simple pre-flag if not in data
        if not f'flag_{crit}' in data.columns:
            if ref_obj is None: raise ValueError('Need to supply a ref_obj.')
            if detr: pre_flag(glob_obj, ref_obj=ref_obj, crit=crit, ID=ID, verbose=verbose, subs_col = substance)
            else: pre_flag(glob_obj, ref_obj=ref_obj, crit=crit, ID=ID, verbose=verbose)
        if f'flag_{crit}' in data.columns:
            # make part of df_sorted so that substance=nan rows get dropped
            try: df_sorted[f'flag_{crit}'] = glob_obj.data[ID][f'flag_{crit}']
            except: print('Pre-flagging unsuccessful, proceeding without')

        df_sorted.dropna(subset=substance, inplace=True) # remove rows without data
        df_sorted.sort_index(inplace=True)

        mxr = df_sorted[substance] # measured mixing ratios
        d_mxr = None
        if f'd_{substance}' in df_sorted.columns: d_mxr = df_sorted[f'd_{substance}']
        elif verbose: print(state+f'No abs. error for {crit}')
        t_obs_tot = np.array(dt_to_fy(df_sorted.index, method='exact'))
        try: flag = df_sorted[f'flag_{crit}']
        except: flag = None

        func = dcts.get_subs(col_name=substance).function
        ol = outliers.find_ol(func, t_obs_tot, mxr, d_mxr,
                              flag = flag, verbose=False, plot=False,
                              limit=0.1, direction = 'n')
        # ^ 4er tuple, 1st is list of OL == 1/2/3 - if not outlier then OL==0
        df_sorted.loc[(flag != 0 for flag in ol[0]), (strato, tropo)] = (True, False)
        df_sorted.loc[(flag == 0 for flag in ol[0]), (strato, tropo)] = (False, True)

        if plot:
            popt0 = fit_data(func, t_obs_tot, mxr, d_mxr)
            plot_sorted(glob_obj, df_sorted, crit, ID, popt0, ol[3], subs=subs, detr=detr, **kwargs)

    return df_sorted

# Sort trop / strat using temperature gradient
def thermal(glob_obj, coord='dp', ID='INT', verbose=False, plot=False,
            subs='co2', **kwargs):
    """ Sort into strat/trop depending on temperature lapse rate / gradient

    Parameters:
        coord (str): dp, pt, z - coordinate (rel. to tropopause)
        ID (str): INT, INT2

    Data availability:
        INT: dp, pt
        INT2: dp, pt, z
        """
    if (ID == 'INT2' and coord == 'z'):
        print(f'{coord} not available in {ID}, using pt'); coord = 'pt'
    elif (coord not in ['dp', 'pt', 'z'] or ID not in ['INT', 'INT2']):
        raise ValueError(f'Cannot sort {ID} with thermal TP with coord={coord}')

    data = glob_obj.data[ID].copy()

    tropo = f'tropo_therm_{coord}'
    strato = f'strato_therm_{coord}'

    df_sorted = pd.DataFrame({strato:pd.Series(np.nan, dtype='float'),
                              tropo:pd.Series(np.nan, dtype='float')},
                             index=data.index)

    if ID == 'INT2':
        cols  = {
            'dp' : 'int_ERA5_PRESS [hPa]',
            'dp_tp' : 'int_ERA5_TROP1_PRESS [hPa]',
            'pt' : 'int_Theta [K]',
            'pt_tp' : 'int_ERA5_TROP1_THETA [K]'}
        col, TP = cols[coord], cols[f'{coord}_tp']

        df_sorted.loc[tools.assign_t_s(data[col], 't', coordinate = coord, tp_val=data[TP]),
                    (strato, tropo)] = (False, True)
        df_sorted.loc[tools.assign_t_s(data[col], 's', coordinate = coord, tp_val=data[TP]),
                    (strato, tropo)] = (True, False)

    elif ID == 'INT':
        coords = {
            'dp' : 'int_dp_strop_hpa [hPa]',
            'pt' : 'int_pt_rel_sTP_K [K]',
            'z' : 'int_z_rel_sTP_km [km]'}
        col = coords[coord]
        df_sorted.loc[tools.assign_t_s(data[col], 't', coord),
                    (strato, tropo)] = (False, True)
        df_sorted.loc[tools.assign_t_s(data[col], 's', coord),
                    (strato, tropo)] = (True, False)

    if verbose: print(df_sorted[strato].value_counts())
    if plot: plot_sorted(glob_obj, df_sorted, coord, ID, subs=subs, **kwargs)

    df_sorted.dropna(subset=tropo, inplace=True) # remove rows without data
    df_sorted.sort_index(inplace=True)
    return df_sorted

# Sort trop / strat using potential vorticity gradient
def dynamical(glob_obj, pvu=3.5, coord = 'pt', ID='INT', verbose=False,
              plot=False, subs='co2', **kwargs):
    """ Sort into strat/trop depending on potential vorticity gradient / values
    Parameters:
        coord (str): dp, pt, z - coordinate (rel. to tropopause). Required for INT
        pvu (float): 1.5, 2.0, 3.5 - value of potential vorticity surface for TP. Required for INT2
        ID (str): INT, INT2

    Data availability:
        INT: coord: 'dp', 'pt', 'z' / pvu: 3.5
        INT2: coord: 'pt' / pvu: 1.5, 2.0, 3.5
    """
    if not ID in ['INT', 'INT2']:
        raise ValueError(f'Cannot sort {ID} using dynamical tropopause')
    if ID=='INT2' and pvu not in [1.5, 2.0, 3.5]:
        raise ValueError(f'No {ID} data for {pvu} pvu')
    elif ID == 'INT' and pvu != 3.5:
        print(f'{pvu} PVU not available for {ID}, using 3.5 PVU'); pvu = 3.5

    data = glob_obj.data[ID].copy()

    tropo = 'tropo_dyn_{}_{}'.format(coord, str(pvu).replace('.', '_'))
    strato = 'strato_dyn_{}_{}'.format(coord, str(pvu).replace('.', '_'))

    df_sorted = pd.DataFrame({strato:pd.Series(np.nan, dtype='float'),
                            tropo:pd.Series(np.nan, dtype='float')},
                           index=data.index)

    if ID == 'INT2':
        col = 'int_ERA5_D_{}PVU_BOT [K]'.format(str(pvu).replace('.', '_'))
    if ID == 'INT':
        coords = {
            'dp' : 'int_dp_dtrop_hpa [hPa]',
            'pt' : 'int_pt_rel_dTP_K [K]',
            'z' : 'int_z_rel_dTP_km [km]'}
        col = coords[coord]

    df_sorted.loc[tools.assign_t_s(data[col], 't', coord),
                (strato, tropo)] = (False, True)
    df_sorted.loc[tools.assign_t_s(data[col], 's', coord),
                (strato, tropo)] = (True, False)

    if verbose: print(df_sorted[strato].value_counts())
    if plot: plot_sorted(glob_obj, df_sorted, coord, ID, subs=subs, **kwargs)

    df_sorted.dropna(subset=tropo, inplace=True) # remove rows without data
    df_sorted.sort_index(inplace=True)
    return df_sorted

# Plotting sorted data
def plot_sorted(glob_obj, df_sorted, crit, ID, popt0=None, popt1=None,
                subs=None, subs_col=None, detr=True, **kwargs):
    """ Plot strat / trop sorted data """
    # only take data with index that is available in df_sorted
    if subs in glob_obj.data.keys(): df = glob_obj.data[subs]
    elif glob_obj.source=='Caribic': df = glob_obj.data[ID]
    else: df = glob_obj.df
    data = df[df.index.isin(df_sorted.index)]

    # data = glob_obj.data[ID][glob_obj.data[ID].index.isin(df_sorted.index)]
    data.sort_index(inplace=True)

    # separate trop/strat data for any criterion
    tropo_col = [col for col in df_sorted.columns if col.startswith('tropo')][0]
    strato_col = [col for col in df_sorted.columns if col.startswith('strato')][0]

    # take 'data' here because substances may not be available in df_sorted
    df_tropo = data[df_sorted[tropo_col] == True]
    df_strato = data[df_sorted[strato_col] == True]

    if crit in ['o3', 'n2o'] and not subs: subs = crit

    if 'subs_pfx' in kwargs.keys():
        subs_pfx = kwargs['subs_pfx']
        substance = dcts.get_col_name(subs, glob_obj.source, kwargs['subs_pfx'])
    else:
        if subs_col is None and subs is not None:
            for subs_pfx in (ID, 'GHG', 'INT', 'INT2'):
                try: substance = dcts.get_col_name(subs, glob_obj.source, subs_pfx); break
                except: substance = None; continue
        else: substance = subs_col; subs_pfx = ID
    if substance is None:
        print(f'Cannot plot {subs}, not available in {ID}.'); return
    if 'detr_'+substance in data.columns: substance = 'detr_'+substance

    fig, ax = plt.subplots(dpi=200)
    plt.title(f'{crit} filter on {ID} data')
    ax.scatter(df_strato.index, df_strato[substance],
                c='grey',  marker='.', zorder=0, label='strato')
    ax.scatter(df_tropo.index, df_tropo[substance],
                c='xkcd:kelly green',  marker='.', zorder=1, label='tropo')

    if popt0 is not None and popt1 is not None and (subs==crit or subs is None):
        # only plot baseline for chemical tropopause def and where crit is being plotted
        t_obs_tot = np.array(dt_to_fy(df_sorted.index, method='exact'))
        ls = 'solid'
        if not subs_pfx == ID: ls = 'dashed'
        func = dcts.get_subs(col_name=substance).function
        ax.plot(df_sorted.index, func(t_obs_tot-2005, *popt0),
                c='r', lw=1, ls=ls, label='initial')
        ax.plot(df_sorted.index, func(t_obs_tot-2005, *popt1),
                c='k', lw=1, ls=ls, label='filtered')

    # plt.ylim(220, 340)

    plt.ylabel(substance)
    plt.legend()
    plt.show()
