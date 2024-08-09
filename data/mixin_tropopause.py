# -*- coding: utf-8 -*-
""" Mixin for tropopause-related sorting and data manipulation

@Author: Sophie Bauchinger, IAU
@Date: Wed Jun 12 13:16:00 2024

TODO: Implement .identify_bins_relative_to_tropopause()
    identifying the lowest stratospheric bins (according to bin size ? )
"""
import numpy as np
import pandas as pd

from toolpac.conv.times import datetime_to_fractionalyear as dt_to_fy # type: ignore
from toolpac.outliers import outliers # type: ignore

import dataTools.dictionaries as dcts
from dataTools import tools
from dataTools.data._local import MaunaLoa

class TropopauseSorterMixin:
    """Filters for stratosphere / troposphere --- TropopauseSorterMixin 
    
    Methods: 
        n2o_filter(**kwargs)
            Use N2O data to create strato/tropo reference for data
        o3_filter_lt60
        
        
        create_df_sorted(**kwargs)
            Use all chosen tropopause definitions to create strato/tropo reference
        calc_ratios(group_vc=False)
            Calculate ratio of tropo/strato datapoints

        filter_extreme_events(**kwargs)
            Filter for tropospheric data, then remove extreme events
        detrend_substance(substance, ...)
            Remove trend wrt. 2005 Mauna Loa from substance, then add to data
    """

    def n2o_filter(self, **kwargs) -> pd.DataFrame:
        """ Filter strato / tropo data based on specific column of N2O mixing ratios. """
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
        ref_years = np.arange(min(self.years)-2, max(self.years)+3)
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
                raise NotImplementedError('No conversion between {subs.unit} and {ref_subs.unit}')

        # Calculate simple pre-flag
        ref_mxr = loc_obj.df.dropna(subset=[ref_subs.col_name])[ref_subs.col_name]
        df_flag = tools.pre_flag(mxr, ref_mxr, 'n2o', **kwargs)
        flag = df_flag['flag_n2o'].values if 'flag_n2o' in df_flag.columns else None

        strato = f'strato_{n2o_column}'
        tropo = f'tropo_{n2o_column}'

        fit_function = dcts.lookup_fit_function('n2o')

        ol = outliers.find_ol(fit_function, t_obs_tot, mxr, d_mxr,
                              flag=flag, verbose=False, plot=False, ctrl_plots=False, 
                              limit=0.1, direction='n')
        # ^ 4er tuple, 1st is list of OL == 1/2/3 - if not outlier then OL==0
        df_sorted.loc[(flag != 0 for flag in ol[0]), (tropo, strato)] = (False, True)
        df_sorted.loc[(flag == 0 for flag in ol[0]), (tropo, strato)] = (True, False)

        df_sorted.drop(columns=[s for s in df_sorted.columns
                                if not s.startswith(('Flight', 'tropo', 'strato'))],
                       inplace=True)
        df_sorted = df_sorted.convert_dtypes()
        return df_sorted

    def o3_filter_lt60(self) -> pd.DataFrame:
        """ Flag ozone mixing ratios below 60 ppb as tropospheric. """
        o3_substs = self.get_substs(short_name = 'o3') 
        
        if len(o3_substs) == 1: 
            [o3_subs] = o3_substs

        elif self.source == 'Caribic':
            if any(s.ID == 'INT' for s in o3_substs): 
                [o3_subs] = [s for s in o3_substs if s.ID=='INT']
            elif any(s.ID == 'MS' for s in o3_substs): 
                [o3_subs] = [s for s in o3_substs if s.ID=='MS']
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
            key relative_only (bool): Skip non-relative coords if rel. is available
        
        """
        if self.source in ['Caribic', 'EMAC', 'TP', 'HALO', 'ATOM', 'HIAPER', 'MULTI']:
            data = self.df.copy()
        else:
            raise NotImplementedError(f'Cannot create df_sorted for {self.source} data.')

        # create df_sorted with flight number if available
        df_sorted = pd.DataFrame(data['Flight number'] if 'Flight number' in data.columns else None,
                                 index=data.index)

        # Get tropopause coordinates
        tps = self.get_tps()
        
        # N2O filter
        for tp in [tp for tp in tps if tp.crit == 'n2o']:
            # if self.source == 'MULTI': break
            n2o_sorted = self.n2o_filter(coord=tp, **kwargs)
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
                                      tropo : pd.Series(np.nan, dtype=object)},
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

    def calc_tropo_strato_ratios(self, tps = None, ratios = True, shared = True) -> pd.DataFrame:
        """ Calculate ratio of tropospheric / stratospheric datapoints for given tropopause definitions.
        
        Parameters: 
            tps (List[Coordinate]): Tropopause definitions to calculate ratios for
            ratios (bool): Include ratio of tropo/strato counts in output 
            shared (True, False, No): Use only shared or non-shared datapoints 
        
        Returns a dataframe with counts (and ratios) for True / False values for all available tps
        """
        
        if not tps: 
            tps = self.tps
        
        tr_cols = [c for c in self.df_sorted.columns if c.startswith('tropo_')]
        # df = self.df_sorted[tr_cols]

        # TODO choose here the 'shared' value
        
        if shared: 
            tropo_cols = ['tropo_'+tp.col_name for tp in tps 
                        if 'tropo_'+tp.col_name in self.df_sorted]
            df = self.df_sorted.dropna(subset=tropo_cols, how='any')
            
        elif not shared: 
            tropo_cols = [c for c in self.df_sorted.columns if c.startswith('tropo_')]
            shared_df = self.df_sorted.dropna(subset=tropo_cols, how='any')
            df = self.df_sorted[~ self.df_sorted.index.isin(shared_df.index)]

        elif shared == 'No': 
            tropo_cols = ['tropo_' + tp.col_name for tp in tps 
                          if 'tropo_' + tp.col_name in self.df_sorted]
            shared_df = self.df_sorted.dropna(subset=tropo_cols, how='any')
            df = self.df_sorted[~ self.df_sorted.index.isin(shared_df.index)]

        else: 
            raise KeyError(f'Invalid value for shared: {shared}')

        tropo_counts = df[df==True].count(axis=0)
        strato_counts = df[df==False].count(axis=0)
        
        count_df = pd.DataFrame({True : tropo_counts, False : strato_counts}).transpose()
        count_df.dropna(axis=1, inplace=True)
        count_df.rename(columns={c: c[6:] for c in count_df.columns}, inplace=True)
        
        if not ratios: 
            return count_df

        ratio_df = pd.DataFrame(columns=count_df.columns, index=['ratios'])
        ratios = [count_df[c][True] / count_df[c][False] for c in count_df.columns]
        ratio_df.loc['ratios'] = ratios  # set col

        return pd.concat([count_df, ratio_df])

    def calc_ratios(self, tps=None, ratios=True) -> pd.DataFrame:
        """ Calculate ratio of tropospheric / stratospheric datapoints for all TP definitions. """
        return self.calc_tropo_strato_ratios(tps=tps, ratios=ratios, shared=False)

    def calc_shared_ratios(self, tps=None, ratios=True) -> pd.DataFrame:
        """ Calculate ratios of tropo / strato data for given tps on shared datapoints. """
        return self.calc_tropo_strato_ratios(tps=tps, ratios=ratios, shared=True)

    def calc_non_shared_ratios(self, tps=None, ratios=True) -> pd.DataFrame:
        """ 
        Calculate ratios of tropo / strato data for given tps only for non-shared datapoints.
        Can be useful to check results of only using shared vs. all datapoints. 
        """
        return self.calc_tropo_strato_ratios(tps=tps, ratios=ratios, shared='No')

    def shared_tropo_strato_indices(self, tps) -> tuple[pd.Index, pd.Index]:
        """ Get indices of datapoints that are identified consistently as tropospheric / stratospheric. """
        shared_tropo = shared_strato = self.get_shared_indices(tps)

        for tp in tps: # iteratively remove non-shared indices 
            shared_tropo = shared_tropo[self.df_sorted.loc[shared_tropo, 'tropo_'+tp.col_name]] 
            shared_strato = shared_strato[self.df_sorted.loc[shared_strato, 'strato_'+tp.col_name]] 

        return shared_tropo, shared_strato

    def identify_bins_relative_to_tropopause(self, subs, tp, **kwargs) -> pd.DataFrame:
        """ Flag each datapoint according to its distance to specific tropopause definitions. """
        # TODO implement this
