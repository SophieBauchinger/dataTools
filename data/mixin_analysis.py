# -*- coding: utf-8 -*-
""" Mixin Classes for analysing global data: Analysis, Binning, TropopauseSorter

@Author: Sophie Bauchinger, IAU
@Date: Wed Jun 12 13:16:00 2024

TODO: Implement .identify_bins_relative_to_tropopause()
    identifying the lowest stratospheric bins (according to bin size ? )

"""
import copy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import toolpac.calc.binprocessor as bp  # type: ignore
from toolpac.conv.times import datetime_to_fractionalyear as dt_to_fy  # type: ignore
from toolpac.outliers import outliers  # type: ignore

import dataTools.dictionaries as dcts
from dataTools import tools
from dataTools.data.local import MaunaLoa


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
    def get_shared_indices(self, tps=None, df=False):
        """ Make reference for shared indices of chosen tropopause definitions. """
        if 'df_sorted' not in self.data:
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
                [str(tp) + '\n' for tp in tps_non_chem],
                [tp for tp in tps if tp not in tps_n2o + tps_non_chem]))

            indices = indices_non_chem[[i in n2o_indices for i in indices_non_chem]]

            # indices = [i for i in indices_non_chem if i in n2o_indices]

        return indices

    def remove_non_shared_indices(self, inplace=True, **kwargs):
        """ Returns a class instances with all non-shared indices of the given tps filtered out. """
        tps = kwargs.get('tps', self.tps)
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
                raise NotImplementedError(f'Units do not match: \
                                          {subs.unit} vs {ref_subs.unit} \n\
                                              Solution not yet available. ')

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
            self.detrend_substance(subs, save=True)

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

    def n2o_baseline_filter(self, **kwargs) -> pd.DataFrame:
        """ Filter strato / tropo data based on specific column of N2O mixing ratios. 
        Args: 
            save_n2o_baseline (bool): Create self.data['n2o_baseline']. Default False
        """
        data = self.df

        # Choose N2O data to use (Substance object)
        if 'coord' in kwargs:
            n2o_coord = kwargs.get('coord')

        elif len([c for c in self.coordinates if c.crit == 'n2o']) == 1:
            [n2o_coord] = [c for c in self.coordinates if c.crit == 'n2o']

        else:
            default_n2o_IDs = dict(Caribic='GHG', ATOM='GCECD', HALO='UMAQS', HIAPER='NWAS', EMAC='EMAC', TP='INT')
            if self.source not in default_n2o_IDs.keys():
                raise NotImplementedError(f'N2O sorting not available for {self.source}')

            n2o_coord = dcts.get_coord(crit='n2o', ID=default_n2o_IDs[self.source])

            if n2o_coord.col_name not in data.columns:
                raise Warning(f'Could not find {n2o_coord.col_name} in {self.ID} data.')

        # Get reference dataset
        ref_years = np.arange(min(self.years) - 2, max(self.years) + 3)
        loc_obj = MaunaLoa(ref_years) if not kwargs.get('loc_obj') else kwargs.get('loc_obj')
        ref_subs = dcts.get_subs(substance='n2o', ID=loc_obj.ID)  # dcts.get_col_name(subs, loc_obj.source)

        if kwargs.get('verbose'):
            print(f'N2O sorting: {n2o_coord} ')

        n2o_column = n2o_coord.col_name

        df_sorted = pd.DataFrame(index=data.index)
        if 'Flight number' in data.columns: df_sorted['Flight number'] = data['Flight number']
        df_sorted[n2o_column] = data[n2o_column]

        if f'd_{n2o_column}' in data.columns:
            df_sorted[f'd_{n2o_column}'] = data[f'd_{n2o_column}']
        if f'detr_{n2o_column}' in data.columns:
            df_sorted[f'detr_{n2o_column}'] = data[f'detr_{n2o_column}']

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

        if kwargs.get('save_n2o_baseline'):
            # Add baseline stats to data dictionary 
            n2o_df = pd.DataFrame({
                f'{n2o_column}' : mxr, 
                f'{n2o_column}_flag' : ol[0], 
                f'{n2o_column}_residual' : ol[1],
                f'{n2o_column}_baseline' : ol[4]})
            if d_mxr is not None: 
                n2o_df[f'd_{n2o_column}'] = d_mxr
            if 'n2o_baseline' not in self.data: 
                self.data['n2o_baseline'] = pd.DataFrame()
            self.data['n2o_baseline'] = self.data['n2o_baseline'].join(n2o_df, how = 'outer')

        df_sorted.drop(columns=[s for s in df_sorted.columns
                                if not s.startswith(('Flight', 'tropo', 'strato'))],
                       inplace=True)
        df_sorted = df_sorted.convert_dtypes()
        return df_sorted

    # TODO: implement o3_baseline_filter
    def o3_baseline_filter(self, **kwargs) -> pd.DataFrame:
        """ Use climatology of Ozone from somewhere (?) - seasonality? - and use as TP filter. """
        raise NotImplementedError('O3 Baseline filter has not yet been implemented')

    def o3_filter_lt60(self) -> pd.DataFrame:
        """ Flag ozone mixing ratios below 60 ppb as tropospheric. """
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
            raise KeyError('Need to be more specific in which Ozone values should be used for sorting. ')

        o3_sorted = pd.DataFrame(index=self.df.index)
        o3_sorted.loc[self.df[o3_subs.col_name].lt(60),
        (f'strato_{o3_subs.col_name}', f'tropo_{o3_subs.col_name}')] = (False, True)
        return o3_sorted, o3_subs

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

        # Get tropopause coordinates
        tps = self.get_tps()

        # N2O filter
        for tp in [tp for tp in tps if tp.crit == 'n2o']:
            # if self.source == 'MULTI': break
            n2o_sorted = self.n2o_baseline_filter(coord=tp, **kwargs)
            if 'Flight number' in n2o_sorted.columns:
                n2o_sorted.drop(columns=['Flight number'], inplace=True)  # del duplicate col
            df_sorted = pd.concat([df_sorted, n2o_sorted], axis=1)

        # Dyn / Therm / CPT / Combo tropopauses
        for tp in [tp for tp in tps if not tp.vcoord == 'mxr']:
            if tp.col_name not in data.columns:
                print(f'Note: {tp.col_name} not found, continuing.')
                continue

            if kwargs.get('verbose'): print(f'Sorting {tp}')

            tp_df = data.dropna(axis=0, subset=[tp.col_name])

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

            # tropo: high p (gt 0), low everything else (lt 0)
            tp_sorted.loc[tp_df[tp.col_name].gt(0) if tp.vcoord == 'p' else tp_df[tp.col_name].lt(0),
                (strato, tropo)] = (False, True)

            # strato: low p (lt 0), high everything else (gt 0)
            tp_sorted.loc[tp_df[tp.col_name].lt(0) if tp.vcoord == 'p' else tp_df[tp.col_name].gt(0),
                (strato, tropo)] = (True, False)

            # # add data for current tp def to df_sorted
            tp_sorted = tp_sorted.convert_dtypes()

            df_sorted[tropo] = tp_sorted[tropo]
            df_sorted[strato] = tp_sorted[strato]

        # Ozone: Flag O3 < 60 ppb as tropospheric
        if any(tp.crit == 'o3' for tp in tps) and not self.source == 'MULTI':
            o3_sorted, o3_subs = self.o3_filter_lt60()
            # rename O3_sorted columns to the corresponding O3 tropopause coord to update
            for tp in [tp for tp in tps if tp.crit == 'o3']:
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

    def tropo_strato_ratios(self, filter='only_shared', **kwargs) -> tuple[pd.DataFrame]: 
        """ Calculates the ratio of tropospheric / stratospheric datapoints for the given tropopause definitions.
        
        Args: 
            tps (list[dcts.Coordinate]): Tropopause definitions to calculate ratios for
            filter (str): only_shared | only_non_shared | None
        
        Returns a dataframe with tropospheric (True) and stratospheric (False) flags per TP definition. 
        """
        # Select data 
        tps = kwargs.get('tps', self.tps)
        tropo_cols = ['tropo_' + tp.col_name for tp in tps
                      if 'tropo_' + tp.col_name in self.df_sorted]

        shared_indices = self.get_shared_indices(tps)
        df = self.df_sorted[tropo_cols]
        if filter == 'only_shared': 
            df = df[df.index.isin(shared_indices)]
        elif filter == 'only_non_shared': 
            df = df[~df.index.isin(shared_indices)]

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


# %% Mixin for adding binning methods to GlobalData objects

class BinningMixin:
    """ Holds methods for binning global data in 1D/2D/3D in selected coordinates. 
    
    Methods:
        bin_1d(subs, coord, bci_1d, xbsize, df)

        bin_2d(subs, xcoord, ycoord, bci_2d, xbsize, ybsize)
        
        bin_3d(subs, zcoord, bci_3d, xbsize, ybsize, zbsize, eql)
        
        bin_LMS(subs, tp, df, bci_3d, zbsize, nr_of_bins)
        
        bin_1d_seasonal(subs, coord, bci_1d, xbsize, df)
        
        bin_2d_seasonal(subs, xcoord, ycoord, bci_2d, xbsize, ybsize, df)
            Bin substance data onto a 2D-grid of the given coordinates for each season. 
        
    """

    def make_bci(self, xcoord, ycoord=None, zcoord=None, **kwargs): 
        """ Create n-dimensional binclassinstance using standard coordinate limits / bin sizes. 
        
        Args: 
            *coord (dcts.Coordinate)
            
            key *bsize (float): Size of the bin
            key *bmin, *bmax (float): Outer bounds for bins. Optional 
            
        Returns Bin_equi*d binning structure for all given dimensions.  
        """
        dims = sum([dim is not None for dim in [xcoord, ycoord, zcoord]])

        if dims not in [1,2,3]:
            raise ValueError('Something went wrong when evaluating dimension numbers. ') 

        xbsize = kwargs.get('xbsize', xcoord.get_bsize())
        def_xbmin, def_xbmax = self.get_var_lims(xcoord, bsize = xbsize, **kwargs)
        xbmin = kwargs.get('xbmin', def_xbmin)
        xbmax = kwargs.get('xbmax', def_xbmax)
        
        if dims == 1: 
            return bp.Bin_equi1d(xbmin, xbmax, xbsize)

        ybsize = kwargs.get('ybsize', ycoord.get_bsize())
        def_ybmin, def_ybmax = self.get_var_lims(ycoord, bsize = ybsize, **kwargs)
        ybmin = kwargs.get('ybmin', def_ybmin)
        ybmax = kwargs.get('ybmax', def_ybmax)
        
        if dims == 2: 
            return bp.Bin_equi2d(xbmin, xbmax, xbsize, 
                                 ybmin, ybmax, ybsize)

        zbsize = kwargs.get('zbsize', zcoord.get_bsize())
        def_zbmin, def_zbmax = self.get_var_lims(zcoord, bsize = zbsize, **kwargs)
        zbmin = kwargs.get('zbmin', def_zbmin)
        zbmax = kwargs.get('zbmax', def_zbmax)
        
        return bp.Bin_equi3d(xbmin, xbmax, xbsize,
                             ybmin, ybmax, ybsize,
                             zbmin, zbmax, zbsize)

    def bin_1d(self, var, xcoord, **kwargs) -> bp.Simple_bin_1d:
        """ Bin substance data in self.df onto 1D-bins of the given coordinate. 
        
        Args: 
            var (dcts.Substance, dcts.Coordinate)
            
            xcoord (dcts.Substance, dcts.Coordinate) - 1st bin dimension 
            ycoord (dcts.Substance, dcts.Coordinate) - 2nd bin dimension 
            
            key bci_3d (bp.Bin_equi3d, bp.Bin_notequi3d): 3D-Binning structure
            key xbsize (float): 1st dim bin size
            key ybsize (float): 2nd dim bin size
        
        Returns bp.Simple_bin_1d object
        """
        df = kwargs.get('df', self.df)
        x = self.get_var_data(xcoord, df=df)
        bci_1d = kwargs.get('bci_1d', self.make_bci(xcoord, **kwargs))

        out = bp.Simple_bin_1d(np.array(df[var.col_name]), x,
                               bci_1d, count_limit=self.count_limit)

        return out

    def bin_2d(self, var, xcoord, ycoord, **kwargs) -> bp.Simple_bin_2d:
        """ Bin substance data in self.df onto an x-y grid spanned by the given coordinates. 

        Args: 
            var (dcts.Substance, dcts.Coordinate)
            
            xcoord (dcts.Substance, dcts.Coordinate) - 1st bin dimension 
            ycoord (dcts.Substance, dcts.Coordinate) - 2nd bin dimension 
            
            key bci_2d (bp.Bin_equi2d, bp.Bin_notequi2d): 2D-Binning structure
            key xbsize (float): 1st dim bin size
            key ybsize (float): 2nd dim bin size
            key lognorm_fit (bool): Toggle fitting a lognorm distr. to the binned data. Default True
        
        Returns bp.Simple_bin_2d object or tools.Bin2DFitted object
        """
        x = self.get_var_data(xcoord)
        y = self.get_var_data(ycoord)
        bci_2d = kwargs.get('bci_2d', self.make_bci(xcoord, ycoord, **kwargs))

        if kwargs.get('lognorm_fit', True): 
            out = tools.Bin2DFitted(np.array(self.df[var.col_name]),
                                    x, y, bci_2d,
                                    count_limit=self.count_limit)
        else: 
            out = bp.Simple_bin_2d(np.array(self.df[var.col_name]), 
                                   x, y, bci_2d, 
                                   count_limit=self.count_limit)

        return out

    def bin_3d(self, var, xcoord, ycoord, zcoord, **kwargs) -> bp.Simple_bin_3d:
        """ Bin variable data onto a 3D-grid given by z-coordinate / (equivalent) latitude / longitude. 

        Args: 
            var (dcts.Substance, dcts.Coordinate)
            
            xcoord (dcts.Substance, dcts.Coordinate) - 1st bin dimension 
            ycoord (dcts.Substance, dcts.Coordinate) - 2nd bin dimension 
            zcoord (dcts.Substance, dcts.Coordinate) - 3rd bin dimension 
            
            key bci_3d (bp.Bin_equi3d, bp.Bin_notequi3d): 3D-Binning structure
            key xbsize (float): 1st dim bin size
            key ybsize (float): 2nd dim bin size
            key zbsize (float): 3rd dim bin size
            key lognorm_fit (bool): Toggle fitting a lognorm distr. to the binned data. Default True
        
        Returns a Simple_bin_2d object or tools.Bin3DFitted object including LogNorm distribution fits.
        """

        x = self.get_var_data(xcoord)
        y = self.get_var_data(ycoord)
        z = self.get_var_data(zcoord)
        bci_3d = kwargs.get('bci_3d', self.make_bci(xcoord, ycoord, zcoord, **kwargs))
        
        if kwargs.get('lognorm_fit', True): 
            out = tools.Bin3DFitted(np.array(self.df[var.col_name]),
                                    x, y, z, bci_3d,
                                    count_limit=self.count_limit)
        else: 
            out = bp.Simple_bin_3d(np.array(self.df[var.col_name]), 
                                   x, y, z, bci_3d, 
                                   count_limit=self.count_limit)
        return out

    def bin_LMS(self, subs, tp, df=None, nr_of_bins=3, **kwargs) -> bp.Simple_bin_3d:
        """ Bin data onto lon-lat-tp grid, then return only the lowermost stratospheric bins. 
        
        Args: 
            subs (dcts.Substance): Substance data to bin
            tp (dcts.Coordinate): Tropopause Definition used to select LMS data

            df (pd.DataFrame): Stratospheric dataset (filtered using TP). Optional
            nr_of_bins (int): Max nr. of bins over the tropopause that should be returned

            key bci_3d(bp.Bin_equi3d, bp.Bin_notequi3d): Binned data
            key zbsize (float): Size of vertical bins

        Returns bp.Simple_bin_3d object
        """

        if not tp.rel_to_tp:
            raise Exception('tp has to be relative to tropopause')

        xbsize = ybsize = self.grid_size
        zbsize = kwargs.get('zbsize', tp.get_bsize())

        if not isinstance(df, pd.DataFrame):
            df = self.sel_strato(**tp.__dict__).df

        x = df.geometry.x
        y = df.geometry.y
        xbmin, xbmax = -180, 180
        ybmin, ybmax = -90, 90

        z = df[tp.col_name]

        # nr_of_bins = min(out.nz, nr_of_bins)
        zbmax = ((np.nanmax(z) // zbsize) + 1) * zbsize
        zbmax = min(zbsize * nr_of_bins, zbmax)
        zbmin = (np.nanmin(z) // zbsize) * zbsize

        bci_3d = kwargs.get('bci_3d', bp.Bin_equi3d(xbmin, xbmax, xbsize,
                                                    ybmin, ybmax, ybsize,
                                                    zbmin, zbmax, zbsize))
        out = bp.Simple_bin_3d(np.array(df[subs.col_name]),
                               x, y, z, bci_3d,
                               count_limit=self.count_limit)
        return out

    def bin_1d_seasonal(self, var, xcoord, **kwargs) -> dict[bp.Simple_bin_1d]:
        """ Bin substance data onto the given coordinate for each season. 
        Args: 
            subs (dcts.Substance)
            coord (dcts.Coordinate)
            
            key bci_1d (bp.Bin_equi1d, bp.Bin_notequi1d): 1D-binning structure
            key xbsize (float)
        
        Returns dictionary of bp.Simple_bin_1d objects for each season. 
        """

        df = kwargs.get('df', self.df)
        if 'season' not in df.columns:
            df['season'] = tools.make_season(df.index.month)

        bci_1d = self.make_bci(xcoord, **kwargs)

        out_dict = {}
        for s in set(self.df['season']): 
            out_dict[s] = self.sel_season(s).bin_1d(var, xcoord, 
                                                    bci_1d = bci_1d, 
                                                    **kwargs)
        return out_dict

    def bin_2d_seasonal(self, var, xcoord, ycoord, **kwargs) -> dict[bp.Simple_bin_2d]: 
        """ Seasonal binning of var along xyz coordinates. """
        if 'season' not in self.df.columns:
            self.df['season'] = tools.make_season(self.df.index.month)
            
        bci_2d = self.make_bci(xcoord, ycoord, **kwargs)

        out_dict = {}
        for s in set(self.df['season']): 
            out_dict[s] = self.sel_season(s).bin_2d(var, xcoord, ycoord, 
                                                    bci_2d = bci_2d, 
                                                    **kwargs)
        return out_dict

    def bin_3d_seasonal(self, var, xcoord, ycoord, zcoord, **kwargs) -> dict[bp.Simple_bin_3d]: 
        """ Seasonal binning of var along xyz coordinates. """
        if 'season' not in self.df.columns:
            self.df['season'] = tools.make_season(self.df.index.month)
            
        bci_3d = self.make_bci(xcoord, ycoord, zcoord, **kwargs)
        
        out_dict = {}
        for s in set(self.df['season']): 
            out_dict[s] = self.sel_season(s).bin_3d(var, xcoord, ycoord, zcoord, 
                                                    bci_3d = bci_3d, 
                                                    **kwargs)
        return out_dict

    # def reorder_seasonal_dicts(self, dictionary): 
    #     """ From var - season get to season - var for nested dictionaries. """
    #     seasonal_dict = {}
    #     for s in set(self.df['season']):
    #         seasonal_dict[s] = {k:v[s] for k,v in dictionary.items()}
    #     return seasonal_dict