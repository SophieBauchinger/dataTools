# -*- coding: utf-8 -*-
""" Mixin for implementing data analysis functionality

@Author: Sophie Bauchinger, IAU
@Date: Wed Jun 12 13:16:00 2024
"""
import copy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from toolpac.conv.times import datetime_to_fractionalyear as dt_to_fy # type: ignore
from toolpac.outliers import outliers # type: ignore

import dataTools.dictionaries as dcts
from dataTools import tools
from dataTools.data._local import MaunaLoa

class AnalysisMixin: 
    """ Mixin for GlobalData classes to allow further data manipulation, e.g. detrending and homogenisation. 
    
    Methods: 
        get_shared_indices(tps, df)
            Find timestamps that all tps coordinates have valid data for
        remove_non_shared_indices(inplace, **kwargs)
            Remove data points where not all tps coordinates have data
        detrend_substance(substance, ...)
            Remove trend wrt. 2005 Mauna Loa from substance, then add to data
        detrend_all
            Call detrend_substances on all available substances
        filter_extreme_events(**kwargs)
            Filter for tropospheric data, then remove extreme events
    """
    # --- Helper for comparing datasets properly ---
    def get_shared_indices(self, tps=None, df=False):
        """ Make reference for shared indices of chosen tropopause definitions. """
        if not 'df_sorted' in self.data:
            self.create_df_sorted()
            
        data = self.df_sorted if not df else self.df
        prefix = 'tropo_' if not df else ''
        
        tps = self.tps if tps is None else tps
        
        if self.source != 'MULTI': 
            tropo_cols = [prefix + tp.col_name for tp in tps if prefix + tp.col_name in data]
            indices = data.dropna(subset=tropo_cols, how='any').index

        else: 
            # Cannot do this without mashing together all the n2o / o3 tropopauses!
            tps_non_chem = [tp for tp in tps if not tp.tp_def == 'chem']
            tropo_cols_non_chem = [prefix + tp.col_name for tp in tps_non_chem if prefix + tp.col_name in data]
            indices_non_chem = data.dropna(subset=tropo_cols_non_chem, 
                                           how='any').index
            # Combine N2O tropopauses. (ignore Caribic O3 tropopause bc only one source)
            tps_n2o = [tp for tp in tps if tp.crit == 'n2o']
            tropo_cols_n2o = [prefix + tp.col_name for tp in tps_n2o if prefix + tp.col_name in data]
            n2o_indices = data.dropna(subset=tropo_cols_n2o,
                                      how='all').index
            
            print('Getting shared indices using\nN2O measurements: {} and dropping O3 TPs: {}'.format(
                [str(tp)+'\n' for tp in tps_non_chem], 
                [tp for tp in tps if tp not in tps_n2o + tps_non_chem]))
            
            indices = indices_non_chem[[i in n2o_indices for i in indices_non_chem]]
            
            # indices = [i for i in indices_non_chem if i in n2o_indices]
        
        return indices

    def remove_non_shared_indices(self, inplace = True, **kwargs): # filter_non_shared_indices
        """ Returns a class instances with all non-shared indices of the given tps filtered out. """
        tps = (self.tps if 'tps' not in kwargs else kwargs.get('tps'))
        shared_indices = self.get_shared_indices(tps)

        out = type(self).__new__(self.__class__)  # new class instance
        for attribute_key in self.__dict__:  # copy attributes
            out.__dict__[attribute_key] = copy.deepcopy(self.__dict__[attribute_key])

        out.data = {}
        df_list = [k for k in self.data
                   if isinstance(self.data[k], pd.DataFrame)]  # includes geodataframes
        for k in df_list:  # only take data from chosen years
            out.data[k] = self.data[k][self.data[k].index.isin(shared_indices)]

        out.status['shared_i_coords'] = tps
        
        if inplace: 
            self.data = out.data
        
        return out

    def detrend_substance(self, subs, loc_obj=None, save=True, plot=False, note='', 
                          **kwargs) -> tuple[pd.DataFrame, np.ndarray]:
        """
        Remove multi-year linear trend from substance wrt. free troposphere measurements from main dataframe.

        Re-implementation of C_tools.detrend_subs. 

        Parameters:
            subs (Substance): Substance to detrend

            loc_obj (LocalData): free troposphere data, defaults to Mauna_Loa. Optional
            save (bool): adds detrended values to main dataframe. Optional
            plot (bool): show original, detrended and reference data. Optional
            note (str): add note to plot. Optional
            
            Returns the polyfit trend parameters as array. 
        """
        # Prepare reference data
        if loc_obj is None:
            loc_obj = MaunaLoa(substances=[subs.short_name],
                               years=range(2005, max(self.years) + 2))

        ref_df = loc_obj.df
        ref_subs = dcts.get_subs(substance=subs.short_name, ID=loc_obj.ID)
        ref_df.dropna(how='any', subset=ref_subs.col_name, inplace=True)  # remove NaN rows

        c_ref = ref_df[ref_subs.col_name].values
        t_ref = np.array(dt_to_fy(ref_df.index, method='exact'))

        popt = np.polyfit(t_ref, c_ref, 2)
        c_fit = np.poly1d(popt)  # get popt, then make into fct

        # Prepare data to be detrended
        df = self.df.copy()

        df.dropna(axis=0, subset=[subs.col_name], inplace=True)
        df.sort_index()
        c_obs = df[subs.col_name].values
        t_obs = np.array(dt_to_fy(df.index, method='exact'))

        # convert data units to reference data units if they don't match
        if str(subs.unit) != str(ref_subs.unit):
            if kwargs.get('verbose'): 
                print(f'Units do not match : {subs.unit} vs {ref_subs.unit}')

            if subs.unit == 'mol mol-1':
                c_obs = tools.conv_molarity_PartsPer(c_obs, ref_subs.unit)
            elif subs.unit == 'pmol mol-1' and ref_subs.unit == 'ppt':
                pass
            else: 
                raise NotImplementedError(f'Units do not match: \
                                          {subs.unit} vs {ref_subs.unit} \n\
                                              Solution not yet available. ')

        detrend_correction = c_fit(t_obs) - c_fit(min(t_obs)) 
        c_obs_detr = c_obs - detrend_correction

        # get variance (?) by subtracting offset from 0
        c_obs_delta = c_obs_detr - c_fit(min(t_obs))

        df_detr = pd.DataFrame({f'DATA_{subs.col_name}'   : c_obs,
                                f'DETRtmin_{subs.col_name}'   : c_obs_detr,
                                f'delta_{subs.col_name}'  : c_obs_delta,
                                f'detrFit_{subs.col_name}': c_fit(t_obs), 
                                f'detr_{subs.col_name}' : c_obs / c_fit(t_obs)},
                               index=df.index)

        # maintain relationship between detr and fit columns
        df_detr[f'detrFit_{subs.col_name}'] = df_detr[f'detrFit_{subs.col_name}'].where(
            ~df_detr[f'detr_{subs.col_name}'].isnull(), np.nan)
            

        if save:
            self.df[f'detr_{subs.col_name}'] = df_detr[f'detr_{subs.col_name}']
            
        if plot:
            fig, ax = plt.subplots(dpi=150, figsize=(6, 4))

            ax.scatter(df_detr.index, df_detr['DATA_' + subs.col_name], label='Flight data',
                       color='orange', marker='.')
            ax.scatter(df_detr.index, df_detr['detr_' + subs.col_name], label='trend removed',
                       color='green', marker='.')
            ax.scatter(ref_df.index, c_ref, label=f'Reference {loc_obj.source}',
                       color='gray', alpha=0.4, marker='.')

            df_detr.sort_index()
            # t_obs = np.array(datetime_to_fractionalyear(df_detr.index, method='exact'))
            ax.plot(df_detr.index, c_fit(t_obs), label='trendline',
                    color='black', ls='dashed')

            ax.set_ylabel(subs.label())  # ; ax.set_xlabel('Time')
            # if not self.source=='Caribic': ax.set_ylabel(f'{subs.col_name} [{ref_subs.unit}]')
            if note != '':
                leg = ax.legend(title=note)
                leg._legend_box.align = "left"
            else:
                ax.legend()

        return df_detr, popt

    def detrend_all(self, verbose=False):
        """ Add detrended data wherever possible for all available substances. """
        ref_substances = set(s.short_name for s in dcts.get_substances(ID='MLO') 
                                   if not s.short_name.startswith('d_'))
        substances = [s for s in self.substances if s.short_name in ref_substances]
        for subs in substances:
            if verbose: print(f'Detrending {subs}. ')
            self.detrend_substance(subs, save=True)

# --- Calculate standard deviation statistics ---
    def strato_tropo_stdv(self, subs, tps=None, **kwargs) -> pd.DataFrame:
        """ Calculate overall variability of stratospheric and tropospheric air (not binned). 
        
        Parameters: 
            subs (dcts.Substance)
            tps (list[dcts.Coordinate])
            
            key seasonal (bool): additionally calculate seasonal variables 
        
        Returns a dataframe with 
            columns:  tropo/strato + stdv/mean/rstv 
            index: tropopause definitions (+ _season if seasonal)
        """
        tps = self.tps if not tps else tps
        shared_indices = self.get_shared_indices(tps)
        shared_df = self.df_sorted[self.df_sorted.index.isin(shared_indices)]
        
        stdv_df = pd.DataFrame(columns=['tropo_stdv', 'strato_stdv',
                                        'tropo_mean', 'strato_mean',
                                        'rel_tropo_stdv', 'rel_strato_stdv'],
                               index=[tp.col_name for tp in tps])

        subs_data = self.df[self.df.index.isin(shared_df.index)][subs.col_name]

        for tp in tps:
            t_stdv = subs_data[shared_df['tropo_' + tp.col_name]].std(skipna=True)
            s_stdv = subs_data[shared_df['strato_' + tp.col_name]].std(skipna=True)

            t_mean = subs_data[shared_df['tropo_' + tp.col_name]].mean(skipna=True)
            s_mean = subs_data[shared_df['tropo_' + tp.col_name]].mean(skipna=True)

            stdv_df.loc[tp.col_name, 'tropo_stdv'] = t_stdv
            stdv_df.loc[tp.col_name, 'strato_stdv'] = s_stdv

            stdv_df.loc[tp.col_name, 'tropo_mean'] = t_mean
            stdv_df.loc[tp.col_name, 'strato_mean'] = s_mean

            stdv_df.loc[tp.col_name, 'rel_tropo_stdv'] = t_stdv / t_mean * 100
            stdv_df.loc[tp.col_name, 'rel_strato_stdv'] = s_stdv / s_mean * 100
        
        if kwargs.get('seasonal'):
            for s in set(self.df.season): 
                data = subs_data[subs_data.season == s]
                tp_df = shared_df[shared_df.index.isin(subs_data.index)]
                for tp in tps:
                    t_stdv = data[shared_df['tropo_' + tp.col_name + f'_{s}']].std(skipna=True)
                    s_stdv = data[shared_df['strato_' + tp.col_name + f'_{s}']].std(skipna=True)
    
                    t_mean = data[shared_df['tropo_' + tp.col_name + f'_{s}']].mean(skipna=True)
                    s_mean = data[shared_df['tropo_' + tp.col_name + f'_{s}']].mean(skipna=True)
    
                    stdv_df.loc[tp.col_name + f'_{s}', 'tropo_stdv'] = t_stdv
                    stdv_df.loc[tp.col_name + f'_{s}', 'strato_stdv'] = s_stdv
    
                    stdv_df.loc[tp.col_name + f'_{s}', 'tropo_mean'] = t_mean
                    stdv_df.loc[tp.col_name + f'_{s}', 'strato_mean'] = s_mean
    
                    stdv_df.loc[tp.col_name + f'_{s}', 'rel_tropo_stdv'] = t_stdv / t_mean * 100
                    stdv_df.loc[tp.col_name + f'_{s}', 'rel_strato_stdv'] = s_stdv / s_mean * 100

        return stdv_df

    def rms_seasonal_vstdv(self, subs, coord, **kwargs) -> pd.DataFrame: # binned and seasonal 
        """ Root mean squared of seasonal variability in 1D bins for given substance and tp. 
        Args: 
            subs (dcts.Substance)
            coord (dcts.Coordinate)
        
            key bci_1d (bp.Bin_equi1d, bp.Bin_notequi1d): 1D-binning structure
            key xbsize (float): Bin-size for coordinate
            
        Returns dataframe with rms_vstdv, rms_rvstd, + seasonal vstdv, rvstd, vcount 
        """
        data_dict = self.bin_1d_seasonal(subs, coord, **kwargs)
        seasons = list(data_dict.keys())

        df = pd.DataFrame(index = data_dict[seasons[0]].xintm)
        df['rms_vstdv'] = np.nan
        df['rms_rvstd'] = np.nan
        
        for s in data_dict: 
            df[f'vstdv_{s}'] = data_dict[s].vstdv
            df[f'rvstd_{s}'] = data_dict[s].rvstd
            df[f'vcount_{s}'] = data_dict[s].vcount

        s_cols_vstd = [c for c in df.columns if c.startswith('vstdv')]
        s_cols_rvstd = [c for c in df.columns if c.startswith('rvstd')]
        n_cols = [c for c in df.columns if c.startswith('vcount')]
        
        # for each bin, calculate the root mean square of the season's standard deviations
        for i in df.index: 
            n = df.loc[i][n_cols].values
            nom = sum(n) - len([i for i in n if i])
            
            s_std = df.loc[i][s_cols_vstd].values
            denom_std = np.nansum([( n[j]-1 ) * s_std[j]**2 for j in range(len(seasons))])
            df.loc[i, 'rms_vstdv'] = np.sqrt(denom_std / nom) if not nom==0 else np.nan
            
            s_rstd = df.loc[i][s_cols_rvstd].values
            denom_rstd = np.nansum([( n[j]-1 ) * s_rstd[j]**2 for j in range(len(seasons))])
            df.loc[i, 'rms_rvstd'] = np.sqrt(denom_rstd / nom) if not nom==0 else np.nan
        
        return df

# --- Filter for extreme events in tropospheric air ---
    def filter_extreme_events(self, plot_ol=False, **tp_kwargs):
        """ Returns only tropospheric background data (filter out tropospheric extreme events)
        1. tp_kwargs are used to select tropospheric data only (if object is not already purely tropospheric)
        2. For each substance in the dataset, extreme 
         
        Filter out all tropospheric extreme events.

        Returns new Caribic object where tropospheric extreme events have been removed.
        Result depends on tropopause definition for tropo / strato sorting.

        Parameters:
            key tp_def (str): 'chem', 'therm' or 'dyn'
            key crit (str): 'n2o', 'o3'
            key vcoord (str): 'pt', 'dp', 'z'
            key pvu (float): 1.5, 2.0, 3.5
            key limit (float): pre-flag limit for chem. TP sorting

            key ID (str): 'GHG', 'INT', 'INT2'
            key verbose (bool)
            key plot (bool)
            key subs (str): substance for plotting
        """
        if self.status.get('tropo') is not None:
            out = copy.deepcopy(self)
        elif self.status.get('strato'):
            raise Warning('Cannot filter extreme events in purely stratospheric dataset')
        else:
            out = self.sel_tropo(**tp_kwargs)
        out.data = {k: v for k, v in self.data.items() if k not in ['sf6', 'n2o', 'ch4', 'co2']}

        for k in out.data:
            if not isinstance(out.data[k], pd.DataFrame) or k == 'met_data': continue
            data = out.data[k].sort_index()

            for column in data.columns:
                # coordinates
                if column in [c.col_name for c in dcts.get_coordinates(vcoord='not_mxr')] + ['Flight number']: continue
                if column in [c.col_name + '_at_fl' for c in dcts.get_coordinates(vcoord='not_mxr')]: continue
                if column.startswith('d_'):
                    continue
                # substances
                if column in [s.col_name for s in dcts.get_substances()]:
                    substance = column
                    time = np.array(dt_to_fy(data.index, method='exact'))
                    mxr = data[substance].tolist()
                    if f'd_{substance}' in data.columns:
                        d_mxr = data[f'd_{substance}'].tolist()
                    else:
                        d_mxr = None  # integrated values of high resolution data

                    # Find extreme events
                    tmp = outliers.find_ol(dcts.get_subs(col_name=substance).function,
                                           time, mxr, d_mxr, flag=None,  # here
                                           direction='p', verbose=False,
                                           plot=plot_ol, limit=0.1, ctrl_plots=False)

                    # Set rows that were flagged as extreme events to 9999, then nan
                    for c in [c for c in data.columns if substance in c]:  # all related columns
                        data.loc[(flag != 0 for flag in tmp[0]), c] = 9999
                    out.data[k].update(data)  # essential to update before setting to nan
                    out.data[k].replace(9999, np.nan, inplace=True)

                else:
                    print(f'Cannot filter {column}, removing it from the dataframe')
                    out.data[k].drop(columns=[column], inplace=True)

        out.status.update({'EE_filter': True})
        return out
