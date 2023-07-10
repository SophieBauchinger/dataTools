# -*- coding: utf-8 -*-
"""
@Author: Sophie Bauchimger, IAU
@Date: Wed Jul  5 14:41:45 2023

Filtering of data in tropospheric / stratospheric origin

chemical:
    GHG: 'n2o'
    INT: 'o3'
    INT2: 'o3', 'n2o'

thermal: 
    INT: 'dp', 'pt'
    INT2: 'dp', 'pt', 'z'

dynamical:
    INT: 'dp', 'pt', 'z' / 3.5
    INT2: 'pt' / 1.5, 2.0, 3.5


"""
import numpy as np
import pandas as pd
# default='warn' - otherwise df[j] = val gives a warning (outliers.outliers)
pd.options.mode.chained_assignment = None
# supress a gui backend userwarning
import matplotlib.pyplot as plt
import warnings; warnings.filterwarnings("ignore", category=UserWarning,
                                         module='matplotlib')

from toolpac.outliers import outliers
from toolpac.outliers.outliers import fit_data
from toolpac.conv.times import datetime_to_fractionalyear

import data #!!! kinda cyclic, but need Mauna_Loa for pre_flag
from tools import get_lin_fit, assign_t_s
from dictionaries import get_fct_substance, get_col_name

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

#%% Baseline filtering 

# Sort trop / strat using mixing ratios relative to trop. background value
def pre_flag(glob_obj, ref_obj=None, crit='n2o', limit = 0.97, c_pfx = 'GHG', 
             save=True, verbose=False):
    """ Sort data into strato / tropo based on difference to ground obs.

    Returns dataframe containing index and strato/tropo/pre_flag columns 

    Parameters:
        glob_obj (GlobalData) : msmt data to be sorted into stratr / trop air
        ref_obj (LocalData) : reference data to use for filtering (background)
        crit (str) : substance to use for flagging
        limit (float) : tracer mxr fraction below which air is classified
                        as stratospheric
        c_pfx (str) : e.g. 'GHG', specify the caribic datasource
        save (bool): add result to glob_obj
    """
    state = f'pre_flag: crit={crit}, c_pfx={c_pfx}\n'
    if glob_obj.source=='Caribic': 
        df = glob_obj.data[c_pfx].copy()
    else: df = glob_obj.df.copy()
    df.sort_index(inplace=True)
    
    substance = get_col_name(crit, glob_obj.source, c_pfx)
    if not substance: raise ValueError(state+'No {crit} data in {c_pfx}')

    if not ref_obj: 
        if verbose: print(state+f'No reference data supplied. Using Mauna Loa {crit} data')
        ref_obj = data.Mauna_Loa(glob_obj.years, crit)

    fit = get_lin_fit(ref_obj.df, get_col_name(crit, ref_obj.source))

    df_flag = pd.DataFrame({f'strato_{crit}':np.nan, 
                            f'tropo_{crit}':np.nan}, 
                           index=df.index)

    t_obs_tot = np.array(datetime_to_fractionalyear(df_flag.index, method='exact'))
    df_flag.loc[df[substance] < limit * fit(t_obs_tot),
           (f'strato_chem_{crit}', f'tropo_chem_{crit}')] = (True, False)

    df_flag[f'flag_{crit}'] = 0
    df_flag.loc[df_flag[f'strato_chem_{crit}'] == True, f'flag_{crit}'] = 1

    if verbose: print('Result of pre-flagging: \n',
                      df_flag[f'flag_{crit}'].value_counts())
    if save and glob_obj.source == 'Caribic':
        glob_obj.data[c_pfx][f'flag_{crit}'] = df_flag[f'flag_{crit}']
    
    return df_flag

def chemical(glob_obj, crit='n2o', c_pfx='GHG', ref_obj=None,
             verbose = False, plot=True, limit=0.97, subs=None):
    """ Returns data set with new bool columns 'strato' and 'tropo'
    Reconstruction of filter_strat_trop from C_tools (T. Schuck)

    Sort data into stratosphere or troposphere based on outlier statistics
    with respect to measurements eg. at Mauna Loa Observatory

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
    if c_pfx == 'GHG' and crit != 'n2o':
        print(f'{crit} not available in {c_pfx}, using n2o')
        crit = 'n2o'
    elif c_pfx == 'INT' and crit != 'o3':
        print(f'{crit} not available in {c_pfx}, using o3')
        crit = 'o3'
    elif c_pfx == 'INT' and crit not in ['n2o', 'o3']:
        print(f'{crit} not available in {c_pfx}, using n2o')
        crit = 'n2o'

    state = f'filter_strat_trop: crit={crit}, c_pfx={c_pfx}\n'
    tropo = f'tropo_chem_{crit}'
    strato = f'strato_chem_{crit}'

    data = glob_obj.data[c_pfx].copy()
    df_sorted = pd.DataFrame({strato:pd.Series(np.nan, dtype='float'), 
                              tropo:pd.Series(np.nan, dtype='float')},
                             index=data.index)

    if crit == 'o3': # INT and INT2 have coordinates relative to O3 TP (Zahl 2003)
        cols = {'INT' : 'int_h_rel_TP [km]',
                'INT2' : 'int_CARIBIC2_H_rel_TP [km]'}
        col, coord = cols[c_pfx], 'z'

        df_sorted.loc[assign_t_s(data[col], 't', coord), 
                    (strato, tropo)] = (False, True)
        df_sorted.loc[assign_t_s(data[col], 's', coord), 
                    (strato, tropo)] = (True, False)
        if plot: plot_sorted(glob_obj, df_sorted, crit, c_pfx, subs=subs)

    if crit == 'n2o' and c_pfx in ['GHG', 'INT2']:
        substance = get_col_name(crit, glob_obj.source, c_pfx) # get column name
        df_sorted[substance] = data[substance]
        if  f'd_{substance}' in data.columns: df_sorted[f'd_{substance}'] = data[f'd_{substance}']

        # Calculate simple pre-flag if not in data 
        if not f'flag_{crit}' in data.columns: 
            flag = None
            pre_flag(glob_obj, ref_obj=ref_obj, crit=crit, c_pfx=c_pfx, verbose=verbose)
        if f'flag_{crit}' in data.columns: 
            try: df_sorted[f'flag_{crit}'] = glob_obj.data[c_pfx][f'flag_{crit}']
            except: print('Pre-flagging unsuccessful, proceeding without')

        df_sorted.dropna(subset=substance, inplace=True) # remove rows without data
        df_sorted.sort_index(inplace=True)

        mxr = df_sorted[substance] # measured mixing ratios
        if f'd_{substance}' in df_sorted.columns: d_mxr = df_sorted[f'd_{substance}']
        else: d_mxr = None; print(state+f'No abs. error for {crit}')
        t_obs_tot = np.array(datetime_to_fractionalyear(df_sorted.index, method='exact'))
        try: flag = df_sorted[f'flag_{crit}']
        except: pass

        func = get_fct_substance(crit)
        ol = outliers.find_ol(func, t_obs_tot, mxr, d_mxr,
                              flag = flag, verbose=False, plot=not(plot),
                              limit=0.1, direction = 'n')

        # ^ 4er tuple, 1st is list of OL == 1/2/3 - if not outlier then OL==0
        df_sorted.loc[(ol[0] != 0), (strato, tropo)] = (True, False)
        df_sorted.loc[(ol[0] == 0), (strato, tropo)] = (False, True)
        df_sorted.drop(columns=substance, inplace=True)

        if plot: 
            popt0 = fit_data(func, t_obs_tot, mxr, d_mxr)
            plot_sorted(glob_obj, df_sorted, crit, c_pfx, popt0, ol[3], subs=subs)
    return df_sorted

# Sort trop / strat using temperature gradient
def thermal(glob_obj, coord='dp', c_pfx='INT', verbose=False, plot=False, subs='co2'):
    """ Sort into strat/trop depending on temperature lapse rate / gradient
    
    Parameters: 
        coord (str): dp, pt, z - coordinate (rel. to tropopause)
        c_pfx (str): INT, INT2
        
    Data availability: 
        INT: dp, pt
        INT2: dp, pt, z
        """

    if (coord not in ['dp', 'pt', 'z'] or c_pfx not in ['INT', 'INT2'] 
        or (c_pfx == 'INT2' and coord == 'z')):
        print(f'Thermal TP sorting not yet implemented for {glob_obj.source} {c_pfx} with coord = {coord}')

    data = glob_obj.data[c_pfx].copy()

    tropo = f'tropo_therm_{coord}'
    strato = f'strato_therm_{coord}'

    df_sorted = pd.DataFrame({strato:pd.Series(np.nan, dtype='float'), 
                              tropo:pd.Series(np.nan, dtype='float')}, 
                             index=data.index)

    if c_pfx == 'INT2':
        cols  = {
            'dp' : 'int_ERA5_PRESS [hPa]',
            'dp_tp' : 'int_ERA5_TROP1_PRESS [hPa]',
            'pt' : 'int_Theta [K]',
            'pt_tp' : 'int_ERA5_TROP1_THETA [K]'}
        col, TP = cols[coord], cols[f'{coord}_tp']

        df_sorted.loc[assign_t_s(data[col], 't', coordinate = coord, tp_val=data[TP]), 
                    (strato, tropo)] = (False, True)
        df_sorted.loc[assign_t_s(data[col], 's', coordinate = coord, tp_val=data[TP]), 
                    (strato, tropo)] = (True, False)

    elif c_pfx == 'INT':
        coords = {
            'dp' : 'int_dp_strop_hpa [hPa]',
            'pt' : 'int_pt_rel_sTP_K [K]',
            'z' : 'int_z_rel_sTP_km [km]'}
        col = coords[coord]
        df_sorted.loc[assign_t_s(data[col], 't', coord), 
                    (strato, tropo)] = (False, True)
        df_sorted.loc[assign_t_s(data[col], 's', coord), 
                    (strato, tropo)] = (True, False)

    if verbose: print(df_sorted[strato].value_counts())
    if plot: plot_sorted(glob_obj, df_sorted, coord, c_pfx, subs=subs)

    return df_sorted

# Sort trop / strat using potential vorticity gradient
def dynamical(glob_obj, pvu=3.5, coord = 'pt', c_pfx='INT', verbose=False, plot=False, subs='co2'):
    """ Sort into strat/trop depending on potential vorticity gradient / values 
    Parameters: 
        coord (str): dp, pt, z - coordinate (rel. to tropopause). Required for INT
        pvu (float): 1.5, 2.0, 3.5 - value of potential vorticity surface for TP. Required for INT2
        c_pfx (str): INT, INT2
    
    Data availability: 
        INT: coord: 'dp', 'pt', 'z' / pvu: 3.5 
        INT2: coord: 'pt' / pvu: 1.5, 2.0, 3.5 
    """
    if not c_pfx in ['INT', 'INT2']: raise ValueError(f'Cannot dyn sort {c_pfx}')
    if c_pfx=='INT2' and pvu not in [1.5, 2.0, 3.5]: 
        raise ValueError(f'No {c_pfx} data for {pvu} pvu')
    elif c_pfx == 'INT' and pvu != 3.5: 
        print(f'{pvu} PVU not available for {c_pfx}, setting it to 3.5 PVU')
        pvu = 3.5

    data = glob_obj.data[c_pfx].copy()

    tropo = 'tropo_dyn_{}_{}'.format(coord, str(pvu).replace('.', '_'))
    strato = 'strato_dyn_{}_{}'.format(coord, str(pvu).replace('.', '_'))

    df_sorted = pd.DataFrame({strato:pd.Series(np.nan, dtype='float'), 
                            tropo:pd.Series(np.nan, dtype='float')}, 
                           index=data.index)

    if c_pfx == 'INT2': 
        col = 'int_ERA5_D_{}PVU_BOT [K]'.format(str(pvu).replace('.', '_'))
    if c_pfx == 'INT':
        coords = {
            'dp' : 'int_dp_dtrop_hpa [hPa]',
            'pt' : 'int_pt_rel_dTP_K [K]',
            'z' : 'int_z_rel_dTP_km [km]'}
        col = coords[coord]

    df_sorted.loc[assign_t_s(data[col], 't', coord), 
                (strato, tropo)] = (False, True)
    df_sorted.loc[assign_t_s(data[col], 's', coord), 
                (strato, tropo)] = (True, False)
    
    if verbose: print(df_sorted[strato].value_counts())
    if plot: plot_sorted(glob_obj, df_sorted, coord, c_pfx, subs=subs)
    
    return df_sorted

# Plotting sorted data
def plot_sorted(glob_obj, df_sorted, crit, c_pfx, popt0=None, popt1=None, subs=None):
    """ Plot strat / trop sorted data """ 
    # only take data with index that is available in df_sorted 
    data = glob_obj.data[c_pfx][glob_obj.data[c_pfx].index.isin(df_sorted.index)]
    data.sort_index(inplace=True)

    # separate trop/strat data for any criterion
    tropo_col = [col for col in df_sorted.columns if col.startswith('tropo')][0]
    strato_col = [col for col in df_sorted.columns if col.startswith('strato')][0]

    # take 'data' here because substances may not be available in df_sorted
    df_tropo = data[df_sorted[tropo_col] == True]
    df_strato = data[df_sorted[strato_col] == True]

    if crit in ['o3', 'n2o'] and not subs: subs = crit

    substance = get_col_name(subs, glob_obj.source, c_pfx)
    if substance is None: print('No {subs} in {c_pfx}'); return
    
    fig, ax = plt.subplots(dpi=200)
    plt.title('{} ({}) filtered using {}-{}'.format(
        subs.upper(), c_pfx, *tropo_col.split('_')[1:]))
    ax.scatter(df_strato.index, df_strato[substance],
                c='xkcd:kelly green',  marker='.', zorder=1, label='strato')
    ax.scatter(df_tropo.index, df_tropo[substance],
                c='grey',  marker='.', zorder=0, label='tropo')

    if popt0 is not None and popt1 is not None and (subs==crit or subs is None):
        t_obs_tot = np.array(datetime_to_fractionalyear(
            df_sorted.index, method='exact'))
        ax.plot(df_sorted.index, get_fct_substance(crit)(t_obs_tot-2005, *popt0), 
                c='r', lw=1, label='initial')
        ax.plot(df_sorted.index, get_fct_substance(crit)(t_obs_tot-2005, *popt1),
                c='k', lw=1, label='filtered')

    plt.ylabel(substance)
    plt.xlabel('Time delta')
    plt.legend()
    plt.show()
