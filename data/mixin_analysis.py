# -*- coding: utf-8 -*-
""" Mixin Classes for analysing global data: Analysis, Binning, TropopauseSorter

@Author: Sophie Bauchinger, IAU
@Date: Wed Jun 12 13:16:00 2024

"""
import copy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from toolpac.conv.times import datetime_to_fractionalyear as dt_to_fy  # type: ignore
from toolpac.outliers import outliers  # type: ignore

import dataTools.dictionaries as dcts
from dataTools import tools
from dataTools.data.local import MaunaLoa

import dataTools.data.tropopause as tp_tools


# %% Mixin for implementing data analysis
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
    def detrend_substance(self, subs, **kwargs) -> tuple[pd.DataFrame, np.ndarray]:
        """
        Remove multi-year linear trend from substance wrt. free troposphere measurements from main dataframe.

        Re-implementation of C_tools.detrend_subs. 

        Parameters:
            subs (Substance): Substance to detrend

            key loc_obj (LocalData): free troposphere data, defaults to Mauna_Loa. Optional
            key save (bool): adds detrended values to main dataframe. Optional, default True
            key plot (bool): show original, detrended and reference data. Optional
            key note (str): add note to plot. Optional
            
            Returns the polyfit trend parameters as array. 
        """
        # Prepare reference data
        loc_obj = kwargs.get('loc_obj', MaunaLoa(substances=[subs.short_name],
                                                 years=range(2005, max(self.years) + 2)))
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
                raise NotImplementedError(
                    f'Units do not match for detrending {subs.col_name}: \n subs: {subs.unit} vs. ref: {ref_subs.unit}')

        detrend_correction = c_fit(t_obs) - c_fit(min(t_obs))
        c_obs_detr = c_obs - detrend_correction

        # get variance (?) by subtracting offset from 0
        c_obs_delta = c_obs_detr - c_fit(min(t_obs))

        df_detr = pd.DataFrame({f'DATA_{subs.col_name}'    : c_obs,
                                f'DETRtmin_{subs.col_name}': c_obs_detr,
                                f'delta_{subs.col_name}'   : c_obs_delta,
                                f'detrFit_{subs.col_name}' : c_fit(t_obs),
                                f'detr_{subs.col_name}'    : c_obs / c_fit(t_obs)},
                               index=df.index)

        # maintain relationship between detr and fit columns
        df_detr[f'detrFit_{subs.col_name}'] = df_detr[f'detrFit_{subs.col_name}'].where(
            ~df_detr[f'detr_{subs.col_name}'].isnull(), np.nan)
        if kwargs.get('save', True):
            self.df[f'detr_{subs.col_name}'] = df_detr[f'detr_{subs.col_name}']

        if kwargs.get('plot'):
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
            if 'note' in kwargs:
                leg = ax.legend(title=kwargs.get('note'))
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
            try: 
                self.detrend_substance(subs, save=True)
            except NotImplementedError as err:
                if verbose: print(err)

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


# %% Mixin for tropopause-related sorting and data manipulation
class TropopauseSorterMixin:
    """ Filters for stratosphere / troposphere 
    
    Methods: 
        n2o_filter(**kwargs)
            Use N2O data to create strato/tropo reference for data.
        o3_filter_lt60
            Assign tropospheric flag to all Ozone values below 60 ppb. 
        
        create_df_sorted(**kwargs)
            Use all chosen tropopause definitions to create strato/tropo reference.

        calc_ratios(group_vc=False)
            Calculate ratio of tropo/strato datapoints.

        filter_extreme_events(**kwargs)
            Filter for tropospheric data, then remove extreme events
        detrend_substance(substance, ...)
            Remove trend wrt. 2005 Mauna Loa from substance, then add to data
    """

    def calculate_average_distange_to_tropopause(self, tps): # TODO: implement
        """ Calculate the average distance of measurements around the tropopause. """

    def create_df_sorted(self, save=True, **kwargs) -> pd.DataFrame:
        """ Create basis for strato / tropo sorting with any TP definitions fitting the criteria.
        If no kwargs are specified, df_sorted is calculated for all possible definitions
        df_sorted: index(datetime), strato_{col_name}, tropo_{col_name} for all tp_defs
        
        Parameters: 
            key verbose (bool): Make the function more talkative        
        """
        if self.source not in ['Caribic', 'EMAC', 'TP', 'HALO', 'ATOM', 'HIAPER', 'MULTI']:
            raise NotImplementedError(f'Cannot create df_sorted for {self.source} data.')
        
        data = self.df.copy()

        # create df_sorted with flight number if available
        df_sorted = pd.DataFrame(data['Flight number'] if 'Flight number' in data.columns else None,
                                 index=data.index)
        rel_tps = self.get_tps(rel_to_tp = True)

        # --- N2O filter ---
        for tp in [tp for tp in self.get_tps(crit = 'n2o')
                   if tp.col_name not in ['N2O_baseline', 'N2O_residual']]:
            n2o_sorted, n2o_df = tp_tools.n2o_baseline_filter(
                data, n2o_coord = tp, **kwargs)
            if 'Flight number' in n2o_sorted.columns:
                n2o_sorted.drop(columns=['Flight number'], inplace=True)  # del duplicate col
            df_sorted = pd.concat([df_sorted, n2o_sorted], axis=1)

        # --- Dyn / Therm / CPT / N2O_residual / ... tropopauses ---
        for tp in [tp for tp in rel_tps]:
            if tp.col_name not in data.columns:
                print(f'Note: {tp.col_name} not found, continuing.')
                continue

            inverse_zero = False
            if (tp.vcoord == 'p' or tp.crit == 'n2o'):
                inverse_zero = True

            if kwargs.get('verbose'): print(f'Sorting {tp}')

            tp_df = data.dropna(axis=0, subset=[tp.col_name])
            tp_df = copy.deepcopy(tp_df)

            if tp.tp_def == 'dyn':  # dynamic TP only outside the tropics - latitude filter
                tp_df = tp_df[np.array([(i > 30 or i < -30) for i in np.array(tp_df.geometry.y)])]
            if tp.tp_def == 'cpt':  # cold point TP only in the tropics
                tp_df = tp_df[np.array([(30 > i > -30) for i in np.array(tp_df.geometry.y)])]

            # define new column names
            tropo = 'tropo_' + tp.col_name
            strato = 'strato_' + tp.col_name

            tp_sorted = pd.DataFrame({strato: pd.Series(np.nan, dtype=object),
                                      tropo: pd.Series(np.nan, dtype=object)},
                                     index=tp_df.index)
            
            if not inverse_zero: # Tropo < 0, Strato > 0
                tp_sorted.loc[tp_df[tp.col_name].lt(0), (strato, tropo)] = (False, True)
                tp_sorted.loc[tp_df[tp.col_name].gt(0), (strato, tropo)] = (True, False)
            else: # Tropo < 0, Strato > 0 - applies for rel. Pressure and N2O_residual
                tp_sorted.loc[tp_df[tp.col_name].lt(0), (strato, tropo)] = (True, False)
                tp_sorted.loc[tp_df[tp.col_name].gt(0), (strato, tropo)] = (False, True)

            # # add data for current tp def to df_sorted
            tp_sorted = tp_sorted.convert_dtypes()

            df_sorted[tropo] = tp_sorted[tropo]
            df_sorted[strato] = tp_sorted[strato]

        # --- Ozone: Flag O3 < 60 ppb as tropospheric ---
        if any(tp.crit == 'o3' for tp in rel_tps) and not self.source == 'MULTI':
            # Choose o3_subs for filtering
            o3_substs = self.get_substs(short_name='o3')
            if len(o3_substs) == 1:
                [o3_subs] = o3_substs
            elif self.source == 'Caribic':
                if any(s.ID == 'INT' for s in o3_substs):
                    [o3_subs] = [s for s in o3_substs if s.ID == 'INT']
                elif any(s.ID == 'MS' for s in o3_substs):
                    [o3_subs] = [s for s in o3_substs if s.ID == 'MS']
                else:
                    [o3_subs] = o3_substs[0]
                    print(f'Using {o3_subs} to filter for <60 ppb as defaults not available.')
            else:
                raise KeyError('Need to be more specific which Ozone values should be used for <60 ppb sorting. ')
            o3_sorted = tp_tools.o3_filter_lt60(self.df, o3_subs)
            
            # rename O3_sorted columns to the corresponding O3 tropopause coord to update
            for tp in [tp for tp in rel_tps if tp.crit == 'o3']:
                o3_sorted[f'tropo_{tp.col_name}'] = o3_sorted[f'tropo_{o3_subs.col_name}']
                o3_sorted[f'strato_{tp.col_name}'] = o3_sorted[f'strato_{o3_subs.col_name}']
                df_sorted.update(o3_sorted, overwrite=False)

        df_sorted = df_sorted.convert_dtypes()
        if save:
            self.data['df_sorted'] = df_sorted

        return df_sorted

    @property
    def df_sorted(self) -> pd.DataFrame:
        """ Bool dataframe indicating Troposphere / Stratosphere sorting of various coords"""
        if 'df_sorted' not in self.data:
            self.create_df_sorted(save=True)
        return self.data['df_sorted']

    def unambiguously_sorted_indices(self, tps) -> tuple[pd.Index, pd.Index]:
        """ Get indices of datapoints that are identified consistently as tropospheric / stratospheric. """
        shared_tropo = shared_strato = self.get_shared_indices(tps)

        for tp in tps:  # iteratively remove non-shared indices
            shared_tropo = shared_tropo[self.df_sorted.loc[shared_tropo, 'tropo_' + tp.col_name]]
            shared_strato = shared_strato[self.df_sorted.loc[shared_strato, 'strato_' + tp.col_name]]

        return shared_tropo, shared_strato

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

    def rms_seasonal_vstdv(self, subs, coord, **kwargs) -> pd.DataFrame:  # binned and seasonal
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

        df = pd.DataFrame(index=data_dict[seasons[0]].xintm)
        df['rms_vstdv'] = np.nan
        df['rms_rvstd'] = np.nan

        for s in data_dict:
            df[f'vstdv_{s}'] = data_dict[s].vstdv
            df[f'rvstd_{s}'] = data_dict[s].rvstd
            df[f'vcount_{s}'] = data_dict[s].vcount

        s_cols_vstd = [c for c in df.columns if c.startswith('vstdv')]
        s_cols_rvstd = [c for c in df.columns if c.startswith('rvstd')]
        n_cols = [c for c in df.columns if c.startswith('vcount')]

        # for each bin, calculate the root-mean-square of the season's standard deviations
        for i in df.index:
            n = df.loc[i][n_cols].values
            nom = sum(n) - len([i for i in n if i])

            s_std = df.loc[i][s_cols_vstd].values
            denom_std = np.nansum([(n[j] - 1) * s_std[j] ** 2 for j in range(len(seasons))])
            df.loc[i, 'rms_vstdv'] = np.sqrt(denom_std / nom) if not nom == 0 else np.nan

            s_rstd = df.loc[i][s_cols_rvstd].values
            denom_rstd = np.nansum([(n[j] - 1) * s_rstd[j] ** 2 for j in range(len(seasons))])
            df.loc[i, 'rms_rvstd'] = np.sqrt(denom_rstd / nom) if not nom == 0 else np.nan

        return df
