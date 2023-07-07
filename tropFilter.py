# -*- coding: utf-8 -*-
"""
@Author: Sophie Bauchimger, IAU
@Date: Wed Jul  5 14:41:45 2023

Filtering of data in tropospheric / stratospheric origin
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
from toolpac.outliers import ol_fit_functions as fct
from toolpac.outliers.outliers import get_no_nan, fit_data
from toolpac.conv.times import datetime_to_fractionalyear

from data import Mauna_Loa
from tools import get_lin_fit
from dictionaries import get_fct_substance, get_col_name, substance_list

filter_types = {
    'chem' : ['n2o', 'o3'], # 'crit'
    'therm' : ['therm'], # 'lapse_rate'
    'dyn' : ['1.5pvu', '2pvu', '3.5pvu'], # 'pvu'
    }

#%% Baseline filtering 
#Filter tropospheric / stratospheric air using n2o mixing ratio
def pre_flag(glob_obj, ref_obj=None, crit='n2o', limit = 0.97, c_pfx = 'GHG', 
             save=True, verbose=False):
    """ Sort data into strato / tropo based on difference to ground obs.

    Returns dataframe containing index and strato/tropo/pre_flag columns and 

    Parameters:
        glob_obj (GlobalData) : msmt data to be sorted into stratr / trop air
        ref_obj (LocalData) : reference data to use for filtering (background)
        crit (str) : substance to use for flagging
        limit (float) : tracer mxr fraction below which air is classified
                        as stratospheric
        c_pfx (str) : e.g. 'GHG', specifc the caribic datasource
        save (bool): add result to glob_obj
    """
    state = f'pre_flag: crit={crit}, c_pfx={c_pfx}\n'
    if glob_obj.source=='Caribic': 
        df = glob_obj.data[c_pfx]
    else: df = glob_obj.df
    df.sort_index(inplace=True)

    df_flag = pd.DataFrame({f'strato_{crit}':np.nan, f'tropo_{crit}':np.nan}, 
                           index=df.index)

    if not ref_obj: 
        if verbose: print(state+f'No reference data supplied. Using Mauna Loa {crit} data')
        ref_obj = Mauna_Loa(glob_obj.years, crit)

    fit = get_lin_fit(ref_obj.df, get_col_name(crit, ref_obj.source))
    t_obs_tot = np.array(datetime_to_fractionalyear(df.index, method='exact'))

    substance = get_col_name(crit, glob_obj.source, c_pfx)
    if not substance: raise Exception(state+'No {crit} data in {c_pfx}')

    df_flag.loc[df[substance] < limit * fit(t_obs_tot),
           (f'strato_{crit}', f'tropo_{crit}')] = (True, False)

    df_flag[f'flag_{crit}'] = 0
    df_flag.loc[df_flag[f'strato_{crit}'] == True, f'flag_{crit}'] = 1
    if verbose: print('Result of pre-flagging: \n',
                      df_flag[f'flag_{crit}'].value_counts())
    
    if save and glob_obj.source == 'Caribic':
        glob_obj.data[c_pfx] = pd.concat([glob_obj.data[c_pfx], df_flag])
    
    return df_flag

def chemical(glob_obj, ref_obj=None, crit='n2o', c_pfx='GHG', 
                      verbose = False, plot=True, limit=0.97):
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
    
    Available substances: 
        stratospheric tracer o3
        tropospheric tracer co
        
        GHG: n2o
        INT: co, o3, h2o
        INT2: co, o3, h2o
    
    """

    if ID == 'GHG':    return ['ch4', 'co2', 'n2o', 'sf6']
    if ID == 'INT':    return ['co', 'o3', 'h2o', 'no', 'noy', 'co2', 'ch4']
    if ID == 'INT2':   return ['co', 'o3', 'h2o', 'no', 'noy', 'co2', 'ch4',
                               'n2o', 'f11', 'f12']

    'int_CARIBIC2_H_rel_TP [km]'


    state = f'filter_strat_trop: crit={crit}, c_pfx={c_pfx}\n'
    try: flag =  glob_obj.data[c_pfx][f'flag_{crit}']
    except: # f'flag_{crit}' not in glob_obj.data[c_pfx].columns:
        if verbose: print(state + 'No pre-flagged data found, calculating now')
        try: 
            pre_flag(glob_obj, ref_obj=ref_obj, crit=crit, c_pfx=c_pfx, verbose=verbose)
            flag =  glob_obj.data[c_pfx][f'flag_{crit}']
        except: 
            if verbose: print('Pre-flagging unsuccessful, proceeding without')
            flag = None
    data = glob_obj.data[c_pfx]

    substance = get_col_name(crit, glob_obj.source, c_pfx) # get column name
    substance = 'int_Tpot [K]'
    mxr = data[substance] # measured mixing ratios
    if f'd_{substance}' in data.columns: d_mxr = data[f'd_{substance}']
    else: d_mxr = None; print(state+f'No abs. error for {crit}')
    t_obs_tot = np.array(datetime_to_fractionalyear(data.index, method='exact'))

    func = get_fct_substance(crit)
    ol = outliers.find_ol(func, t_obs_tot, mxr, d_mxr,
                          flag = flag, verbose=False, 
                          plot=not(plot), limit=0.1, direction = 'n')

    # ^ 4er tuple, 1st is list of OL == 1/2/3 - if not outlier then OL==0
    data.loc[(ol[0] != 0), (f'strato_{crit}', f'tropo_{crit}')] = (True, False)
    data.loc[(ol[0] == 0), (f'strato_{crit}', f'tropo_{crit}')] = (False, True)

    # separate trop/strat data
    df_tropo = data[data[f'tropo_{crit}'] == True]
    df_strato = data[data[f'strato_{crit}'] == True]

    if plot:
        t_strato = np.array(datetime_to_fractionalyear(
            df_strato.index, method='exact'))
        t_tropo = np.array(datetime_to_fractionalyear(
            df_tropo.index, method='exact'))
        fig, ax = plt.subplots(dpi=200)
        plt.title(f'{state}')
        # ax.scatter(t_obs_tot-2005, mxr,
        #            c='silver', lw=1, label='data', zorder=0,  marker='+')
        ax.scatter(t_strato-2005, df_strato[substance],
                    c='xkcd:kelly green',  marker='.', zorder=1, label='strato')
        ax.scatter(t_tropo-2005, df_tropo[substance],
                    c='grey',  marker='.', zorder=0, label='tropo')
        no_nan_time, no_nan_mxr, no_nan_d_mxr = get_no_nan(t_obs_tot, mxr, d_mxr)
        popt0 = fit_data(func, no_nan_time, no_nan_mxr, no_nan_d_mxr)
        ax.plot(np.array(no_nan_time), func(np.array(no_nan_time), *popt0),
                c='r', lw=1, label='initial')
        ax.plot(t_obs_tot, func(t_obs_tot, *ol[3]),
                c='k', lw=1, label='filtered')

        plt.ylabel(substance)
        plt.xlabel('Time delta')
        plt.legend()
        plt.show()

    return data

def thermal(glob_obj, crit='dp', c_pfx='INT', verbose=False, plot=True):
    """ Sort into strat/trop depending on temperature lapse rate / gradient"""
    data = glob_obj.data[c_pfx].copy()

    tropo = f'tropo_therm_{crit}'
    strato = f'strato_therm_{crit}'

    # pd.DataFrame({'strato':pd.Series(np.nan, dtype='float')}, index=data.index)

    df_flag = pd.DataFrame({strato:pd.Series(np.nan, dtype='float'), 
                            tropo:pd.Series(np.nan, dtype='float')}, 
                           index=data.index)

    if c_pfx == 'INT2':
        data['int_dp_strop_hpa [hPa]'] = (data['int_ERA5_PRESS [hPa]'] 
                                          - data['int_ERA5_TROP1_PRESS [hPa]'])
        data['int_pt_rel_sTP_K [K]'] = (data['int_Theta [K]']
                                        - data['int_ERA5_TROP1_THETA [K]'])

    if crit == 'dp': # pressure lower above TP
        coord = 'int_dp_strop_hpa [hPa]' # pressure difference relative to thermal tropopause
        df_flag.loc[(data[coord] > 0), (strato, tropo)] = (False, True)
        df_flag.loc[(data[coord] < 0), (strato, tropo)] = (True, False)

    elif crit == 'pt': # potential temperature higher above TP
        coord = 'int_pt_rel_sTP_K [K]' # potential temperature difference relative to thermal tropopause
        df_flag.loc[(data[coord] < 0), (strato, tropo)] = (False, True)
        df_flag.loc[(data[coord] > 0), (strato, tropo)] = (True, False)

    elif crit == 'z': # geopotential height higher above TP
        coord = 'int_z_rel_sTP_km [km]' # geopotential height relative to thermal tropopause
        df_flag.loc[(data[coord] < 0), (strato, tropo)] = (False, True)
        df_flag.loc[(data[coord] > 0), (strato, tropo)] = (True, False)

    else: raise Exception(f'Thermal TP sorting not yet implemented for {glob_obj.source} {c_pfx} with crit = {crit}')
    if verbose: print(df_flag[strato].value_counts())

    return df_flag

def dynamical(glob_obj, pvu=2.0, c_pfx=None, verbose=False, plot=True):
    """ Sort into strat/trop depending on potential vorticity gradient / values """

    # 'tp_theta_1_5pvu'   : 'int_ERA5_D_1_5PVU_BOT [K]',                  # THETA-Distance to local 1.5 PVU surface (ERA5)
    # 'tp_theta_2_0pvu'   : 'int_ERA5_D_2_0PVU_BOT [K]',                  # -"- 2.0 PVU
    # 'tp_theta_3_5pvu'   : 'int_ERA5_D_3_5PVU_BOT [K]',                  # -"- 3.5 PVU
    # 'h_rel_tp'          : 'int_CARIBIC2_H_rel_TP [km]',                 # H_rel_TP; replacement for H_rel_TP
    
    # (INT2)
    # 'int_ERA5_D_1_5PVU_BOT [K]',                  # THETA-Distance to local 1.5 PVU surface (ERA5)
    # 'int_ERA5_D_2_0PVU_BOT [K]',                  # -"- 2.0 PVU
    # 'int_ERA5_D_3_5PVU_BOT [K]',                  # -"- 3.5 PVU

    # (INT):
    # 'int_z_rel_dTP_km [km]',                          # geopotential height relative to dynamical (PV=3.5PVU) tropopause from ECMWF
    # 'int_dp_dtrop_hpa [hPa]',                         # pressure difference relative to dynamical (PV=3.5PVU) tropopause from ECMWF
    # 'int_pt_rel_dTP_K [K]',                           # potential temperature difference relative to  dynamical (PV=3.5PVU) tropopause from ECMWF
    pass

#%% 
# c_pfx=None; source=None
# if source=='Caribic' and c_pfx=='GHG': # caribic / int
#     col_names = {
#         'p' : 'p [mbar]'}

# if source=='Caribic' and c_pfx=='INT': # caribic / int
#     col_names = {
#         'p'             : 'p [mbar]',
#         'h_rel_tp'      : 'int_h_rel_TP [km]',
#         'pv'            : 'int_PV [PVU]',
#         'to_air_tmp'    : 'int_ToAirTmp [degC]',                            # Total Air Temperature
#         'tpot'          : 'int_Tpot [K]',                                   # potential temperature derived from measured pressure and temperature
#         'z'             : 'int_z_km [km]',                                  # geopotential height of sample from ECMWF
#         'dp_tp_therm'   : 'int_dp_strop_hpa [hPa]',                         # pressure difference relative to thermal tropopause from ECMWF
#         'dp_tp_dym'     : 'int_dp_dtrop_hpa [hPa]',                         # pressure difference relative to dynamical (PV=3.5PVU) tropopause from ECMWF
#         'pt_rel_therm'  : 'int_pt_rel_sTP_K [K]',                           # potential temperature difference relative to thermal tropopause from ECMWF
#         'pt_rel_dyn'    : 'int_pt_rel_dTP_K [K]',                           # potential temperature difference relative to  dynamical (PV=3.5PVU) tropopause from ECMWF
#         'z_rel_therm'   : 'int_z_rel_sTP_km [km]',                          # geopotential height relative to thermal tropopause from ECMWF
#         'z_rel_dyn'     : 'int_z_rel_dTP_km [km]',                          # geopotential height relative to dynamical (PV=3.5PVU) tropopause from ECMWF
#         'eq_lat'        : 'int_eqlat [deg]',                                # equivalent latitude in degrees north from ECMWF
#         }

# elif source=='Caribic' and c_pfx=='INT2': # caribic / int2
#     col_names = {
#         'p'                 : 'p [mbar]',                                   # pressure (mean value)
#         'h_rel_tp'          : 'int_CARIBIC2_H_rel_TP [km]',                 # H_rel_TP; replacement for H_rel_TP
#         'pv'                : 'int_ERA5_PV [PVU]',                          # Potential vorticity (ERA5)
#         'theta'             : 'int_Theta [K]',                              # Potential temperature
#         'p_era5'            : 'int_ERA5_PRESS [hPa]',                       # Pressure (ERA5)
#         't'                 : 'int_ERA5_TEMP [K]',                          # Temperature (ERA5)
#         'eq_lat'            : 'int_ERA5_EQLAT [deg N]',                     # Equivalent latitude (ERA5)
#         'tp_p'              : 'int_ERA5_TROP1_PRESS [hPa]',                 # Pressure of local lapse rate tropopause (ERA5)
#         'tp_theta'          : 'int_ERA5_TROP1_THETA [K]',                   # Pot. temperature of local lapse rate tropopause (ERA5)
#         'mean_age'          : 'int_AgeSpec_AGE [year]',                     # Mean age from age-spectrum (10 yr)
#         'modal_age'         : 'int_AgeSpec_MODE [year]',                    # Modal age from age-spectrum (10 yr)
#         'median_age'        : 'int_AgeSpec_MEDIAN_AGE [year]',              # Median age from age-spectrum
#         'tp_theta_1_5pvu'   : 'int_ERA5_D_1_5PVU_BOT [K]',                  # THETA-Distance to local 1.5 PVU surface (ERA5)
#         'tp_theta_2_0pvu'   : 'int_ERA5_D_2_0PVU_BOT [K]',                  # -"- 2.0 PVU
#         'tp_theta_3_5pvu'   : 'int_ERA5_D_3_5PVU_BOT [K]',                  # -"- 3.5 PVU
#         }