# -*- coding: utf-8 -*-
""" Class definitions for data import and analysis from various sources.

@Author: Sophie Bauchinger, IAU
@Date: Fri Apr 28 14:13:28 2023

Classes:
    # GlobalData
    Caribic
    EMAC
    TropopauseData
    Mozart

    # LocalData
    Mauna_Loa
    Mace_Head
"""
import datetime as dt
import geopandas
import numpy as np
import pandas as pd
from shapely.geometry import Point
import xarray as xr
import os
from functools import partial
import matplotlib.pyplot as plt
from metpy import calc
from metpy.units import units
import dill
import copy

# from toolpac.calc import bin_1d_2d
from toolpac.readwrite import find
from toolpac.readwrite.FFI1001_reader import FFI1001DataReader
from toolpac.conv.times import fractionalyear_to_datetime
from toolpac.outliers import outliers
from toolpac.conv.times import datetime_to_fractionalyear

import dictionaries as dcts
import tools

# TODO: fix the underlying problem in toolpac rather than just suppressing stuff
import warnings
from pandas.errors import SettingWithCopyWarning
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

# %% Global data
class GlobalData():
    """ Contains global datasets with longitude/latitude for each datapoint.

    Attributes:
        years(List[int]) : years included in the stored data
        source (str) : source of the input data, e.g. 'Caribic'
        grid_size (int) : default grid size for binning
        status (dict) : stores information on operations that change the stored data

    Methods:
        binned_1d(subs, **kwargs)
            Bin substance data over latitude
        binned_2d(subs, **kwargs)
            Bin substance data on a longitude/latitude grid
        detrend_substance(substance, ...)
            Remove trend wrt. 2005 Mauna Loa from substance, then add to data

        sel_year(*years)
            Remove all data not in the chosen years
        sel_latitude(lat_min, lat_max)
            Remove all data not in the chosen latitude range
        sel_eqlat(eql_min, eql_max)
            Remove all data not in the chosen equivalent latitude range
        sel_season(season)
            Remove all data not in the chosen season
        sel_flight(flights)
            Remove all data that is not from the chosen flight numbers

        n2o_filter(**kwargs)
            Use N2O data to create strato/tropo reference for data
        create_df_sorted(**kwargs)
            Use all chosen tropopause definitions to create strato/tropo reference
        calc_ratios(group_vc=False)
            Calculate ratio of tropo/strato datapoints
        sel_atm_layer(atm_layer, **kwargs)
            Remove all data not in the chosen atmospheric layer (tropo/strato)
        sel_tropo()
            Remove all stratospheric datapoints
        sel_strato()
            Remove all tropospheric datapoints
        filter_extreme_events(**kwargs)
            Filter for tropospheric data, then remove extreme events
    """

    def __init__(self, years, grid_size=5, count_limit=5):
        """
        years: array or list of integers
        grid_size: int
        v_limits: tuple
        """
        self.years = years
        self.grid_size = grid_size
        self.count_limit = count_limit
        self.status = {}  # use this dict to keep track of changes made to data
        self.source = self.ID = None
        self.data = {}

    def binned_1d(self, subs, **kwargs) -> (list, list):
        """
        Returns 1D binned objects for each year as lists (lat / lon)
        Parameters:
            subs (Substance): dcts.Substance instance
        """
        return tools.bin_1d(self, subs, **kwargs)  # out_x_list, out_y_list

    def binned_2d(self, subs, **kwargs) -> list:
        """
        Returns 2D binned object for each year as a list
        Parameters:
            subs (Substance): dcts.Substance instance
        """
        return tools.bin_2d(self, subs, **kwargs)  # out_list

    def get_shared_indices(self, tps=None, df=True): 
        """ Make reference for shared indices of chosen tropopause definitions. """
        if not 'df_sorted' in self.data: 
            self.create_df_sorted()
        if not tps: 
            tps = tools.minimise_tps(dcts.get_coordinates(tp_def='not_nan'))
        
        data = self.df_sorted if not df else self.df
        prefix = 'tropo_' if not df else ''
            
        tropo_cols = [prefix+tp.col_name for tp in tps if prefix+tp.col_name in data]
        indices = data.dropna(subset=tropo_cols, how='any').index
        
        return indices

    def identify_bins_relative_to_tropopause(self, subs, tp, **kwargs) -> pd.DataFrame: 
        """ Flag each datapoint according to its distance to specific tropopause definitions. """
        #TODO implement this 
        #!!! implement this 

    def detrend_substance(self, substance, loc_obj=None, ID=None,
                          save=True, plot=False, note='') -> np.ndarray:
        """
        Remove multi-year linear trend from substance wrt. free troposphere measurements from all dataframes.
        Returns the polyfit trend parameters as array.
        (redefined from C_tools.detrend_subs)

        Parameters:
            substance (str): substance to detrend e.g. 'sf6'
            loc_obj (LocalData): free troposphere data, defaults to Mauna_Loa
            ID (str): choose specific dataframe to detrend, e.g. 'GHG'
            save (bool): add detrended values to stored data
            plot (bool): show original, detrended and reference data
            note (str): add note to plot
        """
        # Prepare reference data
        if loc_obj is None:
            loc_obj = MaunaLoa(substances=[substance],
                               years=range(2005, max(self.years) + 2))

        ref_df = loc_obj.df
        ref_subs = dcts.get_subs(substance=substance, ID=loc_obj.ID)
        if ref_subs is None: raise ValueError(f'No reference data found for {substance}')
        # ignore reference data earlier and later than two years before/after msmts
        # ref_df = ref_df[min(df.index)-dt.timedelta(356*2)
        #                 : max(df.index)+dt.timedelta(356*2)]
        ref_df.dropna(how='any', subset=ref_subs.col_name, inplace=True)  # remove NaN rows
        c_ref = ref_df[ref_subs.col_name].values
        t_ref = np.array(datetime_to_fractionalyear(ref_df.index, method='exact'))

        popt = np.polyfit(t_ref, c_ref, 2)
        c_fit = np.poly1d(popt)  # get popt, then make into fct

        # Prepare data to be detrended
        substances = dcts.get_substances(short_name=substance)
        if ID is not None:
            dataframes = [ID]
        else:
            dataframes = [k for k in self.data if isinstance(self.data[k], pd.DataFrame) and
                          any(s for s in substances if s.col_name in self.data[k].columns)]
        subs_cols = set(c for k in dataframes for c in self.data[k].columns
                         if c in [s.col_name for s in substances])

        df_detr = pd.DataFrame()

        for i, k in enumerate(dataframes):  # go through each dataframe and detrend
            df = self.data[k].copy()
            for subs in [s for s in substances if s.col_name in df.columns]:
                df.dropna(axis=0, subset=[subs.col_name], inplace=True)
                df.sort_index()
                c_obs = df[subs.col_name].values
                t_obs = np.array(datetime_to_fractionalyear(df.index, method='exact'))

                # convert glob obj data to loc obs unit if units don't match
                if str(subs.unit) != str(ref_subs.unit):
                    # print(f'Units do not match : {subs.unit} vs {ref_subs.unit}')
                    if subs.unit == 'mol mol-1':
                        c_obs = tools.conv_molarity_PartsPer(c_obs, ref_subs.unit)
                    elif subs.unit == 'pmol mol-1' and ref_subs.unit == 'ppt':
                        pass

                detrend_correction = c_fit(t_obs) - c_fit(min(t_obs))
                c_obs_detr = c_obs - detrend_correction
                # get variance (?) by subtracting offset from 0
                c_obs_delta = c_obs_detr - c_fit(min(t_obs))

                df_detr_subs = pd.DataFrame({f'init_{subs.col_name}': c_obs,
                                             f'detr_{subs.col_name}': c_obs_detr,
                                             f'delta_{subs.col_name}': c_obs_delta,
                                             f'detrFit_{subs.col_name}': c_fit(t_obs)},
                                            index=df.index)
                # maintain relationship between detr and fit columns
                df_detr_subs[f'detrFit_{subs.col_name}'] = df_detr_subs[f'detrFit_{subs.col_name}'].where(
                    ~df_detr_subs[f'detr_{subs.col_name}'].isnull(), np.nan)

                if save:
                    columns = [f'detr_{subs.col_name}',
                               f'delta_{subs.col_name}',
                               f'detrFit_{subs.col_name}']

                    # add to data, then move geometry column to the end again
                    self.data[k][columns] = df_detr_subs[columns]
                    self.data[k]['geometry'] = self.data[k].pop('geometry')

                df_detr = pd.concat([df_detr, df_detr_subs])

        if plot:
            substances = [dcts.get_subs(col_name=s) for s in subs_cols
                          if 'detr_' + s in df_detr.columns]
            fig, axs = plt.subplots(len(substances), dpi=150, figsize=(6, 4 * len(substances)), sharex=True)
            for subs, ax in zip(substances, axs.flatten()):
                ax.scatter(df_detr.index, df_detr['init_' + subs.col_name], label='Flight data',
                           color='orange', marker='.')
                ax.scatter(df_detr.index, df_detr['detr_' + subs.col_name], label='trend removed',
                           color='green', marker='.')
                ax.scatter(ref_df.index, c_ref, label='MLO data',
                           color='gray', alpha=0.4, marker='.')

                df_detr.sort_index()
                t_obs = np.array(datetime_to_fractionalyear(df_detr.index, method='exact'))
                ax.plot(df_detr.index, c_fit(t_obs), label='trendline',
                        color='black', ls='dashed')

                ax.set_ylabel(dcts.make_subs_label(subs))  # ; ax.set_xlabel('Time')
                # if not self.source=='Caribic': ax.set_ylabel(f'{subs.col_name} [{ref_subs.unit}]')
                if note != '':
                    leg = ax.legend(title=note)
                    leg._legend_box.align = "left"
                else:
                    ax.legend()

        return df_detr, popt

    def detrend_all(self, verbose=False): 
        """ Add detrended data wherever possible for all available substances. """
        substances = [dcts.get_subs(col_name = col) for col in self.df.columns 
                      if col in [s.col_name for s in dcts.get_substances()
                                 if not s.short_name.startswith(('detr_', 'd_'))]]
        detr_subs = set([s.short_name for s in substances 
                         if any(subs.ID in ['MLO'] for subs 
                                in dcts.get_substances(short_name=s.short_name))])
        for subs in detr_subs: 
            if verbose: print(f'Detrending all {subs} data. ')
            self.detrend_substance(subs, loc_obj=None, ID=None, save=True, plot=False)

    def sel_subset(self, **kwargs):
        """ Allows making multiple selections at once. 

        Parameters: 
            key year (list or int)
            key latitude (Tuple[float, float])
            key eqlat (Tuple[float, float])
            key flight (list or int)
            key tropo / strato (dict)
        """
        # check function input
        if any([f'sel_{k}' not in self.__dir__() for k in kwargs]): 
            raise KeyError('Subset selection is not possible with the following parameters:\
                            {}'.format([k for k in kwargs if f'sel_{k}' not in self.__dir__() ]))
        
        out = type(self).__new__(self.__class__)  # new class instance
        for attribute_key in self.__dict__:  # copy attributes
            out.__dict__[attribute_key] = copy.deepcopy(self.__dict__[attribute_key])
        out.data = self.data.copy()  # stops self.data being overwritten
        
        def get_fctn(class_inst, selection):
            if f'sel_{selection}' in class_inst.__dir__():
                return getattr(class_inst, f'sel_{selection}')
            else: raise KeyError(f'sel_{selection}')

        for selection in kwargs: 
            if isinstance(kwargs.get(selection), (int, str)): 
                out = get_fctn(out, selection)(kwargs.get(selection))
            if isinstance(kwargs.get(selection), (set, list, tuple)):
                out = get_fctn(out, selection)(*kwargs.get(selection))
            if isinstance(kwargs.get(selection), dict): 
                out = get_fctn(out, selection)(**kwargs.get(selection))
        
        return out 
        
    def sel_year(self, *years: int):
        """ Returns GlobalData object containing only data for selected years. """

        # input validation, choose only years that are actually available
        yr_list = [yr for yr in years if yr in self.years]
        if len(yr_list) == 0:
            raise KeyError(f'No valid data for any of the given years: {years}')
        if len(yr_list) != len(years):
            print(f'Note: No data available for {[yr for yr in years if yr not in self.years]}')

        out = type(self).__new__(self.__class__)  # new class instance
        for attribute_key in self.__dict__:  # copy attributes
            out.__dict__[attribute_key] = copy.deepcopy(self.__dict__[attribute_key])
        out.data = self.data.copy()  # stops self.data being overwritten

        if self.source in ['Caribic', 'EMAC', 'TP']:
            # Dataframes
            df_list = [k for k in self.data
                       if isinstance(self.data[k], pd.DataFrame)]  # or Geo-dataframe
            for k in df_list:  # only take data from chosen years
                out.data[k] = out.data[k][out.data[k].index.year.isin(yr_list)]
                out.data[k].sort_index(inplace=True)

            if hasattr(out, 'flights'):
                out.flights = list(set(out.data[df_list[-1]]['Flight number']))
                out.flights.sort()

            # Datasets
            ds_list = [k for k in self.data
                       if isinstance(self.data[k], xr.Dataset)]
            for k in ds_list:
                out.data[k] = out.data[k].sel(time=out.data[k].time.dt.year.isin(yr_list))

        elif self.source == 'Mozart':  # yearly data
            for k in [k for k in self.data if isinstance(self.data[k], pd.DataFrame)]:
                out.data[k] = out.data[k][out.data[k].index.year.isin(yr_list)].sort_index()
            for k in [k for k in self.data if isinstance(self.data[k], xr.Dataset)]:
                out.data[k] = out.data[k].sel(time=yr_list)

        yr_list.sort()
        out.years = yr_list
        return out

    def sel_latitude(self, lat_min: float, lat_max: float):
        """ Returns GlobalData object containing only data for selected latitudes """
        # copy everything over without changing the original class instance
        out = type(self).__new__(self.__class__)
        for attribute_key in self.__dict__:
            out.__dict__[attribute_key] = copy.deepcopy(self.__dict__[attribute_key])
        out.data = self.data.copy()

        if self.source in ['Caribic', 'EMAC', 'TP']:
            df_list = [k for k in self.data
                       if isinstance(self.data[k], geopandas.GeoDataFrame)]  # needed for latitude selection
            for k in df_list:  # delete everything that isn't the chosen lat range
                out.data[k] = out.data[k].cx[-180:180, lat_min:lat_max]
                out.data[k].sort_index(inplace=True)
            for k in [k for k in self.data if k not in df_list and isinstance(self.data[k], pd.DataFrame)]: # non-geodataframes
                indices = [index for df_indices in [out.data[k].index for k in df_list] for index in
                           df_indices]  # all indices in the Geo-dataframes
                out.data[k] = out.data[k].loc[out.data[k].index.isin(indices)]

            # update available years, flights
            if len(df_list) != 0:
                out.years = list(set(out.data[df_list[-1]].index.year))
                out.years.sort()

                if hasattr(out, 'flights'):
                    out.flights = list(set(out.data[df_list[-1]]['Flight number']))
                    out.flights.sort()

            # Datasets
            ds_list = [k for k in self.data
                       if isinstance(self.data[k], xr.Dataset)]

            for k in ds_list:
                out.data[k] = out.data[k].where(out.data[k]['latitude'] > lat_min)
                out.data[k] = out.data[k].where(out.data[k]['latitude'] < lat_max)

            # update years if it hasn't happened with the dataframe already
            if 'df' not in self.data and self.source == 'EMAC':  # only dataset exists
                self.years = list(set(pd.to_datetime(self.data['ds']['time'].values).year))

            # update object status
            if 'latitude' in out.status:
                out.status['latitude'] = (max([out.status['latitude'][0], lat_min]),
                                          min([out.status['latitude'][1], lat_max]))
            else:
                out.status['latitude'] = (lat_min, lat_max)

        elif self.source == 'Mozart':
            out.df = out.df.query(f'latitude > {lat_min}')
            out.df = out.df.query(f'latitude < {lat_max}')
            out.years = list(set(out.df.index.year))

            if hasattr(out, 'ds'):
                out.ds = out.ds.sel(latitude=slice(lat_min, lat_max))
            if hasattr(out, 'SF6'):
                out.SF6 = out.df['SF6']

        else:
            raise Warning(f'Not implemented for {self.source}')

        return out

    def sel_eqlat(self, eql_min: float, eql_max: float, model='ERA5'):
        """ Returns GlobalData object containing only data for selected equivalent latitudes """
        # copy everything over without changing the original class instance
        out = type(self).__new__(self.__class__)
        for attribute_key in self.__dict__:
            out.__dict__[attribute_key] = copy.deepcopy(self.__dict__[attribute_key])
        out.data = self.data.copy()

        if self.source not in ['Caribic', 'TP']:
            raise NotImplementedError('Action not yet supported for non-Caribic data')
        eql_col = dcts.get_coord(model=model, hcoord='eql').col_name
        df = self.data['df'].copy()
        df = df[df[eql_col] > eql_min]
        df = df[df[eql_col] < eql_max]
        out.data['df'] = df

        df_list = [k for k in self.data
                   if isinstance(self.data[k], pd.DataFrame)]  # all dataframes

        for k in df_list:  # delete everything outside eql range
            out.data[k] = out.data[k][out.data[k].index.isin(df.index)]

        # update object status
        if 'eq_lat' in out.status:
            out.status['eq_lat'] = (max([out.status['eq_lat'][0], eql_min]),
                                    min([out.status['eq_lat'][1], eql_max]))
        else:
            out.status['eq_lat'] = (eql_min, eql_max)

        return out

    def sel_season(self, *seasons):
        """ Return GlobalData object containing only pd.DataFrames for the chosen season
        
        Parameters: 
            season (List[int]): list of multiple of 1,2,3,4
                1 - spring, 2 - summer, 3 - autumn, 4 - winter """
        if 'season' in self.status:
            if any(s == dcts.dict_season()[f'name_{s}'] for s in self.status['season']):
                return self
            raise Warning('Cannot select {} as already filtered for {}'.format(
                [dcts.dict_season()[f'name_{s}'] for s in seasons], self.status['season']))
        out = type(self).__new__(self.__class__)  # new class instance
        for attribute_key in self.__dict__:  # copy attributes
            out.__dict__[attribute_key] = copy.deepcopy(self.__dict__[attribute_key])

        out.data = {}
        # Dataframes
        df_list = [k for k in self.data
                   if isinstance(self.data[k], pd.DataFrame)]  # or Geodf
        for k in df_list:  # only take data from chosen years
            out.data[k] = self.data[k].copy()
            out.data[k]['season'] = tools.make_season(out.data[k].index.month)

            mask = [i in seasons for i in out.data[k]['season']]
            out.data[k] = out.data[k].loc[mask]
            out.data[k] = out.data[k].drop(columns=['season'])
            out.data[k].sort_index(inplace=True)

        if hasattr(out, 'flights'):
            out.flights = list(set(out.data[df_list[-1]]['Flight number']))
            out.flights.sort()

        out.status['season'] = [dcts.dict_season()[f'name_{s}'] for s in seasons]
        return out

    def sel_flight(self, flights, verbose=False):
        """ Returns Caribic object containing only data for selected flights
            flight_list (int / list) """
        if not hasattr(self, 'flights'):
            raise NotImplementedError(f'Flight selection not available for {self.source}.')
        if isinstance(flights, int): flights = [flights]
        # elif isinstance(flights, range): flights = list(flights)
        invalid = [f for f in flights if f not in self.flights]
        if len(invalid) > 0 and verbose:
            print(f'No data found for flights {invalid}. Proceeding without.')
        flights = [f for f in flights if f in self.flights]

        out = type(self).__new__(self.__class__)  # create new class instance
        for attribute_key in self.__dict__:  # copy stuff like pfxs
            out.__dict__[attribute_key] = copy.deepcopy(self.__dict__[attribute_key])
        # very important so that self.data doesn't get overwritten
        out.data = self.data.copy()

        df_list = [k for k in self.data
                   if isinstance(self.data[k], pd.DataFrame)]  # list of all datasets to cut
        for k in df_list:  # delete everything but selected flights
            out.data[k] = out.data[k][
                out.data[k]['Flight number'].isin(flights)]
            out.data[k].sort_index(inplace=True)

        out.flights = flights  # update to chosen & available flights
        out.years = list(set(out.data['df'].index.year))
        out.years.sort()
        out.flights.sort()

        return out

    def n2o_filter(self, **kwargs) -> pd.DataFrame:
        """ Filter strato / tropo data based on specific column of N2O mixing ratios. """
        if self.source not in ['Caribic', 'TP', 'EMAC']:
            raise NotImplementedError(f'N2O sorting not available for {self.source}')

        loc_obj = MaunaLoa(self.years) if not kwargs.get('loc_obj') else kwargs.get('loc_obj')

        if 'ID' in kwargs:
            ID = kwargs.pop('ID')
        elif 'GHG' in self.data or dcts.get_subs('n2o', ID='GHG').col_name in self.df.columns:
            ID = 'GHG'
        elif self.source == 'TP' and not dcts.get_subs('n2o', ID='GHG').col_name in self.df.columns:
            raise Warning('Please specify an ID to narrow down which data to use for the N2O filter.')
        else:
            ID = self.source if not hasattr(self, 'ID') else self.ID

        if 'crit' in kwargs: del kwargs['crit']  # needs to be n2o anyway & duplicated otherwise

        subs = dcts.get_subs('n2o', ID=ID)
        ref_subs = dcts.get_subs(substance='n2o', ID=loc_obj.ID)  # dcts.get_col_name(subs, loc_obj.source)

        if kwargs.get('detr'): self.detrend_substance(subs.short_name, save=True)

        dataframes = [k for k in self.data if isinstance(self.data[k], pd.DataFrame)]
        if subs.col_name not in [c for k in dataframes for c in self.data[k].columns]:
            print('Cannot find {subs.col_name} anywhere in the data.')

        # find the dataframe that has the column in it
        if self.source == 'Caribic':
            data = self.data[ID]
        else:
            data = self.df

        if subs.col_name not in data.columns:
            raise Warning(f'Could not find {subs.col_name} in {ID} data.')

        if kwargs.get('verbose'):
            print(f'N2O sorting: {subs} ')

        df_sorted = pd.DataFrame(index=data.index)
        if 'Flight number' in data.columns: df_sorted['Flight number'] = data['Flight number']
        df_sorted[subs.col_name] = data[subs.col_name]

        if f'd_{subs.col_name}' in data.columns:
            df_sorted[f'd_{subs.col_name}'] = data[f'd_{subs.col_name}']
        if f'detr_{subs.col_name}' in data.columns:
            df_sorted[f'detr_{subs.col_name}'] = data[f'detr_{subs.col_name}']

        df_sorted.sort_index(inplace=True)
        df_sorted.dropna(subset=[subs.col_name], inplace=True)

        mxr = df_sorted[subs.col_name]  # measured mixing ratios
        d_mxr = None if not f'd_{subs.col_name}' in df_sorted.columns else df_sorted[f'd_{subs.col_name}']
        t_obs_tot = np.array(datetime_to_fractionalyear(df_sorted.index, method='exact'))

        # Check if units of data and reference data match, if not change data
        if str(subs.unit) != str(ref_subs.unit):
            if kwargs.get('verbose'): print(f'Note units do not match: {subs.unit} vs {ref_subs.unit}')
            if subs.unit == 'mol mol-1':
                mxr = tools.conv_molarity_PartsPer(mxr, ref_subs.unit)
                if d_mxr is not None: d_mxr = tools.conv_molarity_PartsPer(d_mxr, ref_subs.unit)
            elif subs.unit == 'pmol mol-1' and ref_subs.unit == 'ppt':
                pass
            else:
                raise NotImplementedError('No conversion between {subs.unit} and {ref_subs.unit}')

        # Calculate simple pre-flag
        ref_mxr = loc_obj.df.dropna(subset=[ref_subs.col_name])[ref_subs.col_name]
        df_flag = tools.pre_flag(mxr, ref_mxr, 'n2o', **kwargs)
        flag = df_flag['flag_n2o'] if 'flag_n2o' in df_flag.columns else None

        strato = f'strato_{subs.col_name}'
        tropo = f'tropo_{subs.col_name}'

        func = dcts.get_fct_substance('n2o')

        ol = outliers.find_ol(func, t_obs_tot, mxr, d_mxr,
                              flag=flag, verbose=False, plot=False,
                              limit=0.1, direction='n')
        # ^ 4er tuple, 1st is list of OL == 1/2/3 - if not outlier then OL==0
        df_sorted.loc[(flag != 0 for flag in ol[0]), (tropo, strato)] = (False, True)
        df_sorted.loc[(flag == 0 for flag in ol[0]), (tropo, strato)] = (True, False)

        df_sorted.drop(columns=[s for s in df_sorted.columns
                                if not s.startswith(('Flight', 'tropo', 'strato'))],
                       inplace=True)
        df_sorted = df_sorted.convert_dtypes()
        return df_sorted

    def o3_filter(self, **kwargs) -> pd.DataFrame: 
        """ Flag ozone mixing ratios below 60 ppb as tropospheric. """
        subs = dcts.get_subs('o3', ID='INT')
        if not subs.col_name in self.df.columns: 
            raise KeyError(f'Could not find Ozone measurements ({subs.col_name}) in the dataset.')
        df_sorted = pd.DataFrame(index = self.df.index)
        df_sorted.loc[self.df[subs.col_name].lt(60), 
                      (f'strato_{subs.col_name}', f'tropo_{subs.col_name}')] = (False, True)
        return df_sorted

    def create_df_sorted(self, save=True, **kwargs) -> pd.DataFrame:
        """ Create basis for strato / tropo sorting with any TP definitions fitting the criteria.
        If no kwargs are specified, df_sorted is calculated for all possible definitions
        df_sorted: index(datetime), strato_{col_name}, tropo_{col_name} for all tp_defs
        """
        if self.source in ['Caribic', 'EMAC', 'TP']:
            data = self.df.copy()
        else:
            raise NotImplementedError(f'Cannot create df_sorted for {self.source} data.')
        # create df_sorted with flight number if available
        df_sorted = pd.DataFrame(data['Flight number'] if 'Flight number' in data.columns else None,
                                 index=data.index)

        # apply necessary changes to kwargs to only get appropriate tps
        if not 'source' in kwargs and self.source in ['Caribic', 'EMAC']:
            kwargs.update({'source': self.source})
        if not 'tp_def' in kwargs: kwargs.update({'tp_def': 'not_nan'})  # rmv irrelevant stuff
        if not 'vcoord' in kwargs: kwargs.update({'vcoord': 'not_nan'})  # rmv var='frac'

        tps = dcts.get_coordinates(**kwargs)
        tps = [tp for tp in tps
               if (not tp.tp_def in ['combo', 'cpt']
                   and not tp.vcoord == 'lev')]

        # adding bool columns for each tp coordinate to df_sorted
        for tp in [tp for tp in tps if tp.crit == 'n2o']:  # N2O filter
            n2o_sorted = self.n2o_filter(**tp.__dict__)
            if 'Flight number' in n2o_sorted.columns:
                n2o_sorted.drop(columns=['Flight number'], inplace=True)  # del duplicate col
            df_sorted = pd.concat([df_sorted, n2o_sorted], axis=1)

        for tp in [tp for tp in tps if not tp.vcoord == 'mxr']:
            # All other tropopause definitions
            if not tp.col_name in data.columns:
                print(f'Note: {tp.col_name} not found, continuing.')
                continue

            # if rel_to_tp coordinate exists, take that one and remove the other one
            if not tp.rel_to_tp:  
                coord_dct = {k: v for k, v in tp.__dict__.items()
                             if k in ['tp_def', 'model', 'vcoord', 'crit', 'pvu']}
                try:
                    rel_coord = dcts.get_coord(**coord_dct, rel_to_tp=True)
                    if rel_coord.col_name in data.columns:
                        tps.remove(tp)
                        continue  # skip current tp if it exists as relative too
                finally:
                    if kwargs.get('verbose'):
                        print('Using non-relative TP: ', tp)

            if kwargs.get('verbose'): print('TP sorting: ', tp)
            tp_df = data.dropna(axis=0, subset=[tp.col_name])

            if tp.tp_def == 'dyn':  # dynamic TP only outside the tropics - latitude filter
                pass# tp_df = tp_df[np.array([(i > 30 or i < -30) for i in np.array(tp_df.geometry.y)])]
            if tp.tp_def == 'cpt':  # cold point TP only in the tropics
                tp_df = tp_df[np.array([(30 > i > -30) for i in np.array(tp_df.geometry.y)])]

            # define new column names
            tropo = 'tropo_' + tp.col_name
            strato = 'strato_' + tp.col_name

            tp_sorted = pd.DataFrame({strato: pd.Series(np.nan),
                                      tropo: pd.Series(np.nan)},
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

        # flag tropospheric ozone below 60 ppb as tropospheric
        try: 
            o3_sorted = self.o3_filter()
            o3_tropo = o3_sorted[[c for c in o3_sorted if c.startswith('tropo')]]
            o3_strato = o3_sorted[[c for c in o3_sorted if c.startswith('strato')]]
            for tp in [tp for tp in tps if tp.crit=='o3' and tp.vcoord=='z']:
                o3_sorted[f'tropo_{tp.col_name}'] = o3_tropo
                o3_sorted[f'strato_{tp.col_name}'] = o3_strato
                df_sorted.update(o3_sorted, overwrite=False)
        except KeyError: 
            print('Ozone data not found. ')

        df_sorted = df_sorted.convert_dtypes()
        if save: self.data['df_sorted'] = df_sorted
        return df_sorted

    def calc_ratios(self, group_vc=False) -> pd.DataFrame:
        """ Calculate ratio of tropospheric / stratospheric datapoints.
        Returns a dataframe with counts and ratios for True / False values
        NOTE!! No True / False rows if group==True
        """
        tr_cols = [c for c in self.df_sorted.columns if c.startswith('tropo_')]
        tropo_counts = self.df_sorted[tr_cols].apply(pd.value_counts)
        tropo_counts.dropna(axis=1, inplace=True)
        tropo_counts.rename(columns={c: c[6:] for c in tropo_counts.columns}, inplace=True)

        ratio_df = pd.DataFrame(columns=tropo_counts.columns, index=['ratios'])
        ratios = [tropo_counts[c][True] / tropo_counts[c][False] for c in tropo_counts.columns]
        ratio_df.loc['ratios'] = ratios  # set col

        tropo_counts = pd.concat([tropo_counts, ratio_df])

        # if group_vc:
        #     grouped_ratios = pd.DataFrame(index=['ratios'])
        #     # make coordinates so that grouping by model is possible
        #     tps = [dcts.get_coord(col_name=c) for c in tropo_counts.columns
        #            if c in [c.col_name for c in dcts.get_coordinates() if c.tp_def not in ['combo', 'cpt']]]

        #     # create pseudo coordinate for n2o filter
        #     subses = [dcts.get_subs(col_name=c) for c in tropo_counts.columns if
        #               c in [s.col_name for s in dcts.get_substances()]]
        #     subs_tps = [dcts.Coordinate(**subs.__dict__, tp_def='chem', crit='n2o', vcoord='mxr', rel_to_tp='False') for
        #                 subs in subses]
        #     tps = tps + subs_tps

        #     # group by model and average the ratios
        #     for tp_def in set(tp.tp_def for tp in tps):
        #         for model in set(tp.model for tp in tps if tp.tp_def == tp_def):
        #             tps_to_group = [tp for tp in tps if (tp.model == model and tp.tp_def == tp_def)]

        #             crits = set(tp.crit for tp in tps_to_group)

        #             if len(crits) > 1:
        #                 for crit in crits:
        #                     cols = [tp.col_name for tp in tps_to_group if tp.crit == crit]
        #                     label = f'{model}_{tp_def}_{crit}'
        #                     print(label)
        #                     grouped_ratios[label] = np.nanmean(ratios[cols])

        #             else:
        #                 cols = [tp.col_name for tp in tps_to_group][0]
        #                 crit_label = '_' + crits if tp_def == 'chem' else ''
        #                 label = f'{model}_{tp_def}' + crit_label
        #                 print(label)
        #                 grouped_ratios[label] = np.nanmean(ratio_df[cols])

        #             return grouped_ratios

        return tropo_counts

    def calc_shared_ratios(self, tps=None): 
        """ Calculate ratios of tropo / strato data for given tps on shared datapoints. """
        if not tps: 
            tps = tools.minimise_tps(dcts.get_coordinates(tp_def='not_nan'))
        tropo_cols = ['tropo_'+tp.col_name for tp in tps if 'tropo_'+tp.col_name in self.df_sorted]

        shared_df = self.df_sorted.dropna(subset=tropo_cols, how='any')
        # shared_df = shared_df[shared_df != 0].dropna(subset=tropo_cols)

        tropo_counts = shared_df[tropo_cols].apply(pd.value_counts)
        tropo_counts.dropna(axis=1, inplace=True)
        tropo_counts.rename(columns={c: c[6:] for c in tropo_counts.columns}, inplace=True)

        ratio_df = pd.DataFrame(columns=tropo_counts.columns, index=['ratios'])
        ratios = [tropo_counts[c][True] / tropo_counts[c][False] for c in tropo_counts.columns]
        ratio_df.loc['ratios'] = ratios  # set col

        tropo_counts = pd.concat([tropo_counts, ratio_df])
        
        return tropo_counts

    def calc_non_shared_ratios(self, tps=None): 
        """ Calculate ratios of tropo / strato data for given tps on shared datapoints. """
        tps = tools.minimise_tps(dcts.get_coordinates(tp_def='not_nan'))
        tropo_cols = ['tropo_'+tp.col_name for tp in tps if 'tropo_'+tp.col_name in self.df_sorted]

        shared_df = self.df_sorted.dropna(subset=tropo_cols, how='any')
        non_shared_df = self.df_sorted[ ~ self.df_sorted.index.isin(shared_df.index)]

        tropo_counts = non_shared_df[tropo_cols].apply(pd.value_counts)
        tropo_counts.dropna(axis=1, inplace=True)
        tropo_counts.rename(columns={c: c[6:] for c in tropo_counts.columns}, inplace=True)

        ratio_df = pd.DataFrame(columns=tropo_counts.columns, index=['ratios'])
        ratios = [tropo_counts[c][True] / tropo_counts[c][False] for c in tropo_counts.columns]
        ratio_df.loc['ratios'] = ratios  # set col

        tropo_counts = pd.concat([tropo_counts, ratio_df])

        return tropo_counts

    def strato_tropo_stdv(self, subs, tps=None, **kwargs) -> pd.DataFrame:
        """ Calculate overall variability of stratospheric and tropospheric air. """
        if not tps: 
            tps = tools.minimise_tps(dcts.get_coordinates(tp_def='not_nan'))
        tropo_cols = ['tropo_'+tp.col_name for tp in tps if 'tropo_'+tp.col_name in self.df_sorted]

        shared_df = self.df_sorted.dropna(subset=tropo_cols, how='any')
        stdv_df = pd.DataFrame(columns= ['tropo_stdv', 'strato_stdv',
                                         'tropo_mean', 'strato_mean',
                                         'rel_tropo_stdv', 'rel_strato_stdv'],
                               index = [tp.col_name for tp in tps])

        subs_data = self.df[self.df.index.isin(shared_df.index)][subs.col_name]

        for tp in tps: 
            t_stdv = subs_data[shared_df['tropo_'+tp.col_name]].std(skipna=True)
            s_stdv = subs_data[shared_df['strato_'+tp.col_name]].std(skipna=True)
            
            t_mean = subs_data[shared_df['tropo_'+tp.col_name]].mean(skipna=True)
            s_mean = subs_data[shared_df['tropo_'+tp.col_name]].mean(skipna=True)

            stdv_df['tropo_stdv'][tp.col_name] = t_stdv
            stdv_df['strato_stdv'][tp.col_name] = s_stdv
            
            stdv_df['tropo_mean'][tp.col_name] = t_mean
            stdv_df['strato_mean'][tp.col_name] = s_mean
            
            stdv_df['rel_tropo_stdv'][tp.col_name] = t_stdv/t_mean * 100 
            stdv_df['rel_strato_stdv'][tp.col_name] = s_stdv/s_mean * 100
        
        return stdv_df

    @property
    def df_sorted(self) -> pd.DataFrame:
        """ Bool dataframe indicating Troposphere / Stratosphere sorting of various coords"""
        if 'df_sorted' not in self.data:
            self.create_df_sorted(save=True)
        return self.data['df_sorted']

    def sel_atm_layer(self, atm_layer: str, **kwargs):
        """ Create GlobalData object with strato / tropo sorting.

        Parameters:
            atm_layer (str): atmospheric layer: 'tropo' or 'strato'

            key tp_def (str): 'chem', 'therm' or 'dyn'
            key crit (str): 'n2o', 'o3'
            key coord (str): 'pt', 'dp', 'z'
            key pvu (float): 1.5, 2.0, 3.5
            key limit (float): pre-flag limit for chem. TP sorting
        """
        out = type(self).__new__(self.__class__)  # create new class instance
        for attribute_key in self.__dict__:  # copy stuff like pfxs
            out.__dict__[attribute_key] = copy.deepcopy(self.__dict__[attribute_key])

        if 'source' not in kwargs and self.source in ['Caribic', 'EMAC']:
            kwargs.update({'source': self.source})

        if 'rel_to_tp' not in kwargs and any(
                c.rel_to_tp for c in dcts.get_coordinates(**kwargs)):
            kwargs.update({'rel_to_tp': True})

        tp = dcts.get_coord(**kwargs)

        if 'df_sorted' not in self.data:
            df_sorted = self.create_df_sorted(save=False, **kwargs)
        else:
            df_sorted = self.df_sorted

        if f'{atm_layer}_{tp.col_name}' not in df_sorted:
            raise KeyError(f'Could not find {tp.col_name} in df_sorted.')

        atm_layer_col = f'{atm_layer}_{tp.col_name}'

        # Filter all dataframes to only include indices in df_sorted
        df_list = [k for k in self.data
                   if isinstance(self.data[k], pd.DataFrame)]  # list of all datasets to cut
        for k in df_list:
            out.data[k] = out.data[k][df_sorted[atm_layer_col]]

        if self.source == 'Caribic':
            out.pfxs = [k for k in out.data if k in self.pfxs]

        # update object status
        out.status.update({atm_layer: True})
        if 'TP filter' in out.status:
            out.status['TP filter'] = (out.status['TP filter'], atm_layer_col)
        else:
            out.status['TP filter'] = atm_layer_col

        return out

    def sel_tropo(self, **kwargs):
        """ Returns Caribic object containing only tropospheric data points. """
        return self.sel_atm_layer('tropo', **kwargs)

    def sel_strato(self, **kwargs):
        """ Returns Caribic object containing only tropospheric data points. """
        return self.sel_atm_layer('strato', **kwargs)

    def filter_extreme_events(self, **kwargs):
        """ Filter out all tropospheric extreme events.

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
            out = self.sel_tropo(**kwargs)
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
                    time = np.array(datetime_to_fractionalyear(data.index, method='exact'))
                    mxr = data[substance].tolist()
                    if f'd_{substance}' in data.columns:
                        d_mxr = data[f'd_{substance}'].tolist()
                    else:
                        d_mxr = None  # integrated values of high resolution data

                    func = dcts.get_fct_substance(dcts.get_substances(col_name=substance)[0].short_name)
                    # Find extreme events
                    plot = False if 'plot' not in kwargs else kwargs.get('plot')
                    tmp = outliers.find_ol(func, time, mxr, d_mxr, flag=None,  # here
                                           direction='p', verbose=False,
                                           plot=plot, limit=0.1, ctrl_plots=False)

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


# Caribic
class Caribic(GlobalData):
    """ Stores all available information from Caribic datafiles and allows further analysis.

    Attributes:
        pfxs (List[str]) : Prefixes of stored Caribic data files

    Methods:
        coord_combo()
            Create met_data from available meteorological data
        create_tp_coordinates()
            Calculate tropopause height etc. from available met data
        create_substance_df(detr=False):
            Combine met_data with all substance info, optionally incl. detrended
    """

    def __init__(self, years=range(2005, 2021), pfxs=('GHG', 'INT', 'INT2'),
                 grid_size=5, verbose=False, recalculate=False, detrend_all=True):
        """ Constructs attributes for Caribic object and creates data dictionary.

        Parameters:
            years (List[int]) : import data only for selected years
            pfxs (List[str]) : prefixes of Caribic files to import
            grid_size (int) : grid size in degrees to use for binning
            verbose (bool) : print additional debugging information
            recalculate (bool) : get data from pre-combined file or parent directory
        """
        # no caribic data before 2005, takes too long to check so cheesing it
        super().__init__([yr for yr in years if yr > 2004], grid_size)

        self.source = 'Caribic'
        self.pfxs = pfxs
        self.flights = ()
        self.get_data(verbose=verbose, recalculate=recalculate)  # creates self.data dictionary
        if 'met_data' not in self.data:
            self.data['met_data'] = self.coord_combo()  # reference for met data for all msmts
            self.create_tp_coords()

        for subs in ['sf6', 'n2o', 'co2', 'ch4']:
            if subs not in self.data:
                self.create_substance_df(subs)

        if detrend_all: 
            self.detrend_all()

    def __repr__(self):
        return f"""{self.__class__}
    data: {self.pfxs}
    years: {self.years}
    status: {self.status}"""

    def get_year_data(self, pfx: str, yr: int, parent_dir: str, verbose: bool) -> pd.DataFrame:
        """ Data import for a single year """
        if not any(find.find_dir("*_{}*".format(yr), parent_dir)):
            # removes current year from class attribute if there's no data
            self.years = np.delete(self.years, np.where(self.years == yr))
            if verbose: print(f'No data found for {yr} in {self.source}. \
                              Removing {yr} from list of years')
            return pd.DataFrame()

        print(f'Reading Caribic - {pfx} - {yr}')
        # Collect data from individual flights for current year
        df_yr = pd.DataFrame()
        for current_dir in find.find_dir("Flight*_{}*".format(yr), parent_dir)[1:]:
            flight_nr = int(str(current_dir)[-12:-9])

            f = find.find_file(f'{pfx}_*', current_dir)
            if not f or len(f) == 0:  # no files found
                if verbose: print(f'No {pfx} File found for \
                                  Flight {flight_nr} in {yr}')
                continue
            if len(f) > 1:
                f.sort()  # sort to get most recent v

            f_data = FFI1001DataReader(f[-1], df=True, xtype='secofday',
                                       sep_variables=';')
            df_flight = f_data.df  # index = Datetime
            df_flight.insert(0, 'Flight number',
                             [flight_nr] * df_flight.shape[0])

            col_dict, rename_dict = tools.rename_columns(f_data.VNAME)
            # set names to their short version
            df_flight.rename(columns=rename_dict, inplace=True)
            df_yr = pd.concat([df_yr, df_flight])

        # Convert longitude and latitude into geometry objects
        geodata = [Point(lon, lat) for lon, lat in zip(
            df_yr['lon'], df_yr['lat'])]
        # geodata = [Point(lat, lon) for lon, lat in zip(
        #     df_yr['lon'],
        #     df_yr['lat'])]
        gdf_yr = geopandas.GeoDataFrame(df_yr, geometry=geodata)

        # Drop cols which are saved within datetime, geometry
        if not gdf_yr['geometry'].empty:
            filter_cols = ['TimeCRef', 'year', 'month', 'day',
                           'hour', 'min', 'sec', 'lon', 'lat', 'type']
            del_column_names = [gdf_yr.filter(
                regex='^' + c).columns[0] for c in filter_cols]
            gdf_yr.drop(del_column_names, axis=1, inplace=True)

        return gdf_yr, col_dict

    def get_pfx_data(self, pfx, parent_dir, verbose) -> pd.DataFrame:
        """ Data import for chosen prefix. """
        gdf_pfx = geopandas.GeoDataFrame()
        for yr in self.years:
            gdf_yr, col_dict = self.get_year_data(pfx, yr, parent_dir, verbose)
            
            gdf_pfx = pd.concat([gdf_pfx, gdf_yr])
            if pfx == 'GHG':  # rmv case-sensitive distinction in cols
                cols = ['SF6', 'CH4', 'CO2', 'N2O']
                for col in cols + ['d_' + c for c in cols]:
                    if col.lower() in gdf_pfx.columns:
                        gdf_pfx[col] = gdf_pfx[col].combine_first(gdf_pfx[col.lower()])
                        gdf_pfx.drop(columns=col.lower(), inplace=True)
            if pfx == 'INT':
                gdf_pfx.drop(columns=['int_acetone',
                                      'int_acetonitrile'], inplace=True)
            if pfx == 'INT2':
                gdf_pfx.drop(columns=['int_CARIBIC2_Ac',
                                      'int_CARIBIC2_AN'], inplace=True)
        return gdf_pfx, col_dict

    def get_data(self, verbose=False, recalculate=False) -> dict:
        """ Imports Caribic data in the form of geopandas dataframes.

        Returns data dictionary containing dataframes for each file source and
        dictionaries relating column names with Coordinate / Substance instances.
        """
        self.data = {}  # easiest way of keeping info which file the data comes from
        parent_dir = r'E:\CARIBIC\Caribic2data'

        if not recalculate and os.path.exists(r'misc_data\caribic_data_dict.pkl'):
            with open(r'misc_data\caribic_data_dict.pkl', 'rb') as f:
                self.data = dill.load(f)
            self.data = self.sel_year(*self.years).data

        else:
            print('Importing Caribic Data from remote files.')
            for pfx in self.pfxs:  # can include different prefixes here too
                gdf_pfx, col_dict = self.get_pfx_data(pfx, parent_dir, verbose)

                if gdf_pfx.empty: print("Data extraction unsuccessful. \
                                        Please check your input data"); return

                # Remove dropped columns from dictionary
                pop_cols = [i for i in col_dict if i not in gdf_pfx.columns]
                for key in pop_cols: col_dict.pop(key)

                self.data[pfx] = gdf_pfx
                self.data[f'{pfx}_dict'] = col_dict

        self.flights = list(set(pd.concat(
            [self.data[pfx]['Flight number'] for pfx in self.pfxs])))

        return self.data

    def coord_combo(self) -> pd.DataFrame:
        """ Create dataframe with all possible coordinates but
        no measurement / substance values """
        # merge lists of coordinates for all pfxs in the object
        coords = [y for pfx in self.pfxs for y in dcts.coord_dict(pfx)] + [
            'p', 'geometry', 'Flight number']
        if 'GHG' in self.pfxs:
            # copy bc don't want to overwrite data
            df = self.data['GHG'].copy()
        else:
            df = pd.DataFrame()
        for pfx in [pfx for pfx in self.pfxs if pfx != 'GHG']:
            df = df.combine_first(self.data[pfx].copy())
        df.drop([col for col in df.columns if col not in coords],
                axis=1, inplace=True)  # remove non-met / non-coord data

        # reorder columns
        self.data['met_data'] = df[list(['Flight number', 'p']
                                        + [col for col in df.columns
                                           if col not in ['Flight number', 'p', 'geometry']]
                                        + ['geometry'])]
        return self.data['met_data']

    def create_tp_coords(self) -> pd.DataFrame:
        """ Add calculated relative / absolute tropopause values to .met_data """
        df = self.met_data.copy()
        new_coords = dcts.get_coordinates(**{'ID': 'calc', 'source': 'Caribic'})

        for coord in new_coords:
            # met = tp + rel -> MET - MINUS for either one
            met_col = coord.var1
            minus_col = coord.var2

            if met_col in df.columns and minus_col in df.columns:
                df[coord.col_name] = df[met_col] - df[minus_col]
            else:
                print(f'Could not generate {coord.col_name} as precursors are not available')

        self.data['met_data'] = df
        return df

    def create_df(self) -> pd.DataFrame:
        df = self.met_data.copy()
        for pfx in self.pfxs:
            # df = df.sjoin(self.data[pfx])
            df = pd.merge(self.data[pfx], df, how='outer', sort=True,
                          left_index=True, right_index=True, suffixes=['', '_' + pfx])
            for c in df.columns:
                if f'{c}_{pfx}' in df.columns:
                    df[c] = df[c].combine_first(df[f'{c}_{pfx}'])
                    df = df.drop(columns=f'{c}_{pfx}')
        self.data['df'] = df
        return df

    def create_substance_df(self, subs, detr=True):
        """ Create dataframe containing all met.+ msmt. data for a substance """
        if detr: 
            self.detrend_substance(subs)
        subs_cols = [c for c in self.df
                     if any(i in dcts.get_subs_columns(short_name=subs) 
                            for i in [c, c[5:], c[6:], c[8:]])]

        df = self.df[list(self.met_data.columns) + subs_cols]
        df.dropna(subset = subs_cols, how='all', inplace=True)

        try: # reordering the columns
            df = df[['Flight number', 'p'] 
                    + [c for c in df.columns 
                       if c not in ['Flight number', 'p', 'geometry']] 
                    + ['geometry']]

        except KeyError: pass
        self.data[f'{subs}'] = df
        return self

    @property
    def GHG(self) -> pd.DataFrame:
        if 'GHG' in self.data:
            return self.data['GHG']
        raise Warning('No GHG data available')

    @property
    def INT(self) -> pd.DataFrame:
        if 'INT' in self.data:
            return self.data['INT']
        raise Warning('No INT data available')

    @property
    def INT2(self) -> pd.DataFrame:
        if 'INT2' in self.data:
            return self.data['INT2']
        raise Warning('No INT2 data available')

    @property
    def met_data(self) -> pd.DataFrame:
        if 'met_data' in self.data:
            return self.data['met_data']
        return self.coord_combo()

    @property
    def df(self) -> pd.DataFrame:
        if 'df' in self.data:
            return self.data['df']
        return self.create_df()


# EMAC
class EMAC(GlobalData):
    """ Data class holding information on Caribic-specific EMAC Model output.

    Methods:
        create_tp()
            Create dataset with tropopause relevant parameters
        create_df()
            Create pandas dataframe from time-dependent data
    """

    def __init__(self, years=range(2005, 2020), s4d=True, s4d_s=True, tp=True, df=True, pdir=None):
        if isinstance(years, int): years = [years]
        super().__init__([yr for yr in years if 2000 <= yr <= 2019])
        self.source = 'EMAC'
        self.ID = 'EMAC'
        self.pdir = '{}'.format(r'E:/MODELL/EMAC/TPChange/' if pdir is None else pdir)
        self.get_data(years, s4d, s4d_s, tp, df)

    def __repr__(self):
        self.years.sort()
        return f'EMACData object\n\
            years: {self.years}\n\
            status: {self.status}'

    def get_data(self, years, s4d: bool, s4d_s: bool, tp: bool, df: bool, recalculate=False) -> dict:
        """ Preprocess EMAC model output and create datasets """
        if not recalculate:
            if s4d:
                with xr.open_dataset(r'misc_data\emac_ds.nc', mmap=False) as ds:
                    self.data['s4d'] = ds
            if s4d_s:
                with xr.open_dataset(r'misc_data\emac_ds_s.nc', mmap=False) as ds_s:
                    self.data['s4d_s'] = ds_s
            if tp:
                if os.path.exists(r'misc_data\emac_tp.nc'):
                    with xr.open_dataset(r'misc_data\emac_tp.nc', mmap=False) as tp:
                        self.data['tp'] = tp
                else:
                    self.create_tp()
            if df:
                if os.path.exists(r'misc_data\emac_df.pkl'):
                    with open(r'misc_data\emac_df.pkl', 'rb') as f:
                        self.data['df'] = dill.load(f)
                elif tp:
                    self.create_df()

        else:
            # print('No ready-made files found. Calculating it anew')
            if s4d:  # preprocess: process_s4d
                fnames = self.pdir + "s4d_CARIBIC/*bCARIB2.nc"
                # extract data, each file goes through preprocess first to filter variables & convert units
                with xr.open_mfdataset(fnames, preprocess=partial(tools.process_emac_s4d), mmap=False) as ds:
                    self.data['s4d'] = ds
            if s4d_s:  # preprocess: process_s4d_s
                fnames_s = self.pdir + "s4d_subsam_CARIBIC/*bCARIB2_s.nc"
                # extract data, each file goes through preprocess first to filter variables
                with xr.open_mfdataset(fnames_s, preprocess=partial(tools.process_emac_s4d_s), mmap=False) as ds:
                    self.data['s4d_s'] = ds
            if tp: self.create_tp()
            if tp and df: self.create_df()

        # update years according to available data
        self.years = list(set(pd.to_datetime(self.data['{}'.format('s4d' if s4d else 's4d_s')]['time'].values).year))

        self.data = self.sel_year(*years).data  # needed if data is loaded from file
        return self.data

    def create_tp(self) -> xr.Dataset:
        """ Create dataset with tropopause relevant parameters from s4d and s4d_s"""
        ds = self.data['s4d'].copy()
        ds_s = self.data['s4d_s'].copy()  # subsampled flight level values

        # remove floating point errors for datasets that mess up joining them up
        ds['time'] = ds.time.dt.round('S')
        ds_s['time'] = ds_s.time.dt.round('S')

        for dataset in [ds, ds_s]:
            for var in dataset.variables:
                if hasattr(dataset[var], 'units'):
                    if dataset[var].units == 'Pa':
                        dataset[var] = dataset[var].metpy.convert_units(units.hPa)
                    elif dataset[var].units == 'm':
                        dataset[var] = dataset[var].metpy.convert_units(units.km)
                    dataset[var] = dataset[var].metpy.dequantify()  # makes units an attribute again

        # for var in ds.variables:  # streamline units
        #     if hasattr(ds[var], 'units'):
        #         if ds[var].units == 'Pa':
        #             ds[var] = ds[var].metpy.convert_units(units.hPa)
        #         elif ds[var].units == 'm':
        #             ds[var] = ds[var].metpy.convert_units(units.km)
        #         ds[var] = ds[var].metpy.dequantify()  # makes units an attribute again
        #
        # for var in ds_s.variables:  # streamline units
        #     if hasattr(ds_s[var], 'units'):
        #         if ds_s[var].units == 'Pa':
        #             ds_s[var] = ds_s[var].metpy.convert_units(units.hPa)
        #         elif ds_s[var].units == 'm':
        #             ds_s[var] = ds_s[var].metpy.convert_units(units.km)
        #         ds_s[var] = ds_s[var].metpy.dequantify()  # makes units an attribute again

        # get geopotential height from geopotential (s4d & s4d_s for _at_fl)
        # ^ metpy.dequantify allows putting the units back into being attributes
        print('Calculating ECHAM5_height')
        ds = ds.assign(ECHAM5_height=calc.geopotential_to_height(ds['ECHAM5_geopot'])).metpy.dequantify()
        ds['ECHAM5_height'] = ds['ECHAM5_height'].metpy.convert_units(units.km)
        ds['ECHAM5_height'] = ds['ECHAM5_height'].metpy.dequantify()

        print('Calculating ECHAM5_height_at_fl')
        ds_s = ds_s.assign(
            ECHAM5_height_at_fl=calc.geopotential_to_height(ds_s['ECHAM5_geopot_at_fl'])).metpy.dequantify()
        ds_s['ECHAM5_height_at_fl'] = ds_s['ECHAM5_height_at_fl'].metpy.convert_units(units.km)
        ds_s['ECHAM5_height_at_fl'] = ds_s['ECHAM5_height_at_fl'].metpy.dequantify()

        new_coords = dcts.get_coordinates(**{'ID': 'calc', 'source': 'EMAC', 'var1': 'not_tpress', 'var2': 'not_nan'})
        abs_coords = [c for c in new_coords if c.var2.endswith('_i')]  # get eg. value of pt at tp
        rel_coords = list(dcts.get_coordinates(**{'ID': 'calc', 'source': 'EMAC', 'var1': 'tpress', 'var2': 'not_nan'})
                          + [c for c in new_coords if c not in abs_coords])  # eg. pt distance to tp

        # copy relevant data into new dataframe
        vars_at_fl = ['longitude', 'latitude', 'tpress',
                      'tropop_PV_at_fl', 'e5vdiff_tpot_at_fl',
                      'ECHAM5_tm1_at_fl', 'ECHAM5_tpoteq_at_fl',
                      'ECHAM5_press_at_fl', 'ECHAM5_height_at_fl'] + [
                         v for v in ds_s.variables if v.startswith('tracer_')]
        tp_ds = ds_s[vars_at_fl].copy()

        tp_vars = [v.col_name for v in dcts.get_coordinates(**{'ID': 'EMAC', 'tp_def': 'not_nan'})
                   if not v.col_name.endswith(('_i', '_f')) and v.col_name in ds.variables]
        for var in tp_vars:
            tp_ds[var] = ds[var].copy()

        for coord in abs_coords:
            # e.g. potential temperature at the tropopause (get from index)
            print(f'Calculating {coord.col_name}')
            met = ds[coord.var1]
            met_at_tp = met.sel(lev=ds[coord.var2], method='nearest')
            # remove unnecessary level dimension when adding to tp_ds
            tp_ds[coord.col_name] = met_at_tp.drop_vars('lev')

        for coord in rel_coords:
            # mostly depend on abs_coords so order matters (eg. dp wrt. tp_dyn)
            print(f'Calculating {coord.col_name}')
            met = tp_ds[coord.var1]
            if coord.var2 in ds.variables:
                tp = ds[coord.var2]
            elif coord.var2 in ds_s.variables:
                tp = ds_s[coord.var2]
            elif coord.var2 in tp_ds.variables:
                tp = tp_ds[coord.var2]
            else:
                raise ValueError(f'Could not find {coord.var2} in the data')
            # units aren't propagated when subtracting so add them back
            rel = (met - tp) * units(met.units)
            tp_ds[coord.col_name] = rel.metpy.dequantify()  # makes attr 'units'

        self.data['tp'] = tp_ds
        return self.data['tp']

    def create_df(self, tp=True) -> pd.DataFrame:
        """ Create dataframe from time-dependent variables in dataset """
        if tp:
            if 'tp' not in self.data:
                print('Tropopause dataset not found, generating it now.')
                self.create_tp()
            dataset = self.data['tp']
        else:
            dataset = self.ds_s

        df = dataset.to_dataframe()
        # drop rows without geodata
        df.dropna(subset=['longitude', 'latitude'], how='any', inplace=True)
        # geodata = [Point(lat, lon) for lat, lon in zip(
        #     df['latitude'], df['longitude'])]
        geodata = [Point(lon, lat) for lon, lat in zip(
            df['longitude'], df['latitude'])]
        df.drop(['longitude', 'latitude'], axis=1, inplace=True)
        df.index = df.index.round('S')
        self.data['df'] = geopandas.GeoDataFrame(df, geometry=geodata)
        return self.data['df']

    @property
    def ds(self) -> xr.Dataset:
        """ Allow accessing dataset as class attribute """
        if 's4d' in self.data:
            return self.data['s4d']
        raise Warning('No s4d dataset found')

    @property
    def ds_s(self) -> xr.Dataset:
        """ Allow accessing dataset as class attribute """
        if 's4d_s' in self.data:
            return self.data['s4d_s']
        raise Warning('No s4d_s found')

    @property
    def tp(self) -> xr.Dataset:
        """ Returns dataset with tropopause relevant parameters  """
        if 'tp' not in self.data:
            self.create_tp()
        return self.data['tp']  # tropopause relevant parameters, only time-dependent

    @property
    def df(self) -> pd.DataFrame:
        """ Return dataframe based on subsampled EMAC Data """
        if 'df' not in self.data:
            choice = input('No dataframe found. Generate it now? [Y/N]\n')
            if choice.upper() == 'Y':
                return self.create_df()
            return pd.DataFrame()
        return self.data['df']

    def save_to_dir(self):
        """ Save emac data etc to files """
        dt.datetime.now().strftime("%Y_%m_%d-%p%I_%M_%S")
        pdir = f'misc_data/Emac-{dt.datetime.now().strftime("%Y_%m_%d-%I_%M_%S")}'
        os.mkdir(pdir)

        with open(pdir + '/Emac_inst.pkl', 'wb') as f:
            dill.dump(self, f)
        if 's4d' in self.data: self.ds.to_netcdf(pdir + '/ds.nc')
        if 's4d_s' in self.data: self.ds_s.to_netcdf(pdir + '/ds_s.nc')
        if 'tp' in self.data: self.tp.to_netcdf(pdir + '/tp.nc')
        if 'df' in self.data:
            with open(pdir + '/df.pkl', 'wb') as f:
                dill.dump(self.df, f)
        return self


# Combine Caribic and EMAC
class TropopauseData(GlobalData):
    """ Holds Caribic data and Caribic-specific EMAC Model output """

    def __init__(self, years=range(2005, 2020), interp=True, method='n', df_sorted=True, detrend_all=True):
        if isinstance(years, int): years = [years]
        super().__init__([yr for yr in years if 2000 <= yr <= 2019])
        self.source = 'TP'
        self.get_data()
        # NB select year on this object afterwards bc otherwise interpolation is missing surrounding values
        if interp: self.interpolate_emac(method)
        self.data = self.sel_year(*years).data

        if df_sorted:
            try:
                with open('misc_data/tpdata_df_sorted.pkl', 'rb') as f:
                    self.data['df_sorted'] = dill.load(f)
            finally:
                self.create_df_sorted(save=True)

        if detrend_all or input('Detrend all? [Y/N] ').upper()=='Y': 
            self.detrend_all()

    def __repr__(self):
        self.years.sort()
        return f'{self.__class__}\n\
    years: {self.years}\n\
    status: {self.status}'

    def get_data(self) -> pd.DataFrame:
        """ Return merged dataframe with interpolated EMAC / Caribic data """
        caribic = Caribic(detrend_all=False)
        emac = EMAC()
        df_caribic = caribic.df
        df_emac = emac.df
        df = pd.merge(df_caribic, df_emac, how='outer', sort=True,
                      left_index=True, right_index=True)
        df.geometry = df_caribic.geometry.combine_first(df_emac.geometry)

        df = df.drop(columns=['geometry_x', 'geometry_y'])
        df['Flight number'].interpolate(method='nearest', inplace=True)
        df['Flight number'].interpolate(inplace=True, limit_direction='both')  # fill in first two timestamps too
        df['Flight number'] = df['Flight number'].astype(int)
        self.data['df'] = df
        self.flights = list(set(df['Flight number']))
        return df

    def interpolate_emac(self, method: str, verbose=False) -> dict:
        """ Add interpolated EMAC data to joint df to match caribic timestamps.

        Parameters:
            method (str): interpolation method. Limit is set to 2 consecutive NaN values
                'n' - nearest neighbour, 'b' - bi-linear
            verbose (bool): makes the function more talkative

        Note: Residual NaN values in nearest because EMAC only goes to 2019.
        Explanation on methods see at
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html
        """
        data = self.df.copy()
        tps_emac = [i.col_name for i in dcts.get_coordinates(source='EMAC') if i.col_name in self.df.columns] + [
            i for i in ['ECHAM5_tm1_at_fl', 'ECHAM5_tpoteq_at_fl', 'ECHAM5_press_at_fl'] if i in self.df.columns]
        subs_emac = [i.col_name for i in dcts.get_substances(source='EMAC') if i.col_name in self.df.columns]

        nan_count_i = data[tps_emac[0]].isna().value_counts().loc[True]
        for c in tps_emac + subs_emac:
            if method == 'b':
                data[c].interpolate(method='linear', inplace=True, limit=2)
            elif method == 'n':
                data[c].interpolate(method='nearest', inplace=True, limit=2)
            else:
                raise KeyError('Please choose either b-linear or n-nearest neighbour interpolation.')
            data[c] = data[c].astype(float)
        nan_count_f = data[tps_emac[0]].isna().value_counts().loc[True]

        if verbose: print('{} NaNs in EMAC data filled using {} interpolation'.format(
            nan_count_i - nan_count_f, 'nearest neighbour' if method == 'n' else 'linear'))

        self.data['df'] = data
        self.status['interp_emac'] = True
        return data

    def sort_tropo_strato(self) -> pd.DataFrame:
        """ Returns dataframe with bool strato / tropo columns for various TP definitions. """
        return self.df_sorted

    @property
    def df(self) -> pd.DataFrame:
        """ Combined dataframe for Caribic and EMAC (interpolated) data. """
        return self.data['df']

    @property
    def df_sorted(self) -> pd.DataFrame:
        """ Bool dataset sorted into strato/tropo for various tropopauses. """
        if 'df_sorted' in self.data:
            return self.data['df_sorted']
        return self.create_df_sorted(save=True)


# Mozart
class Mozart(GlobalData):
    """ Stores relevant Mozart data

    Class attributes:
        years: arr
        source: str
        substance: str
        ds: xarray DataFrame
        df: Pandas GeoDataFrame
        x: arr, latitude
        y: arr, longitude (remapped to +-180 deg)
    """

    def __init__(self, years=range(1980, 2009), grid_size=5, v_limits=None):
        """ Initialise Mozart object """
        super().__init__(years, grid_size)
        self.years = years
        self.source = 'Mozart'
        self.ID = 'MZT'
        self.substance = 'SF6'
        self.v_limits = v_limits  # colorbar normalisation limits
        self.data = {}
        self.get_data()

    def __repr__(self):
        return f'Mozart data, subs = {self.substance}'

    def get_data(self, remap_lon=True, verbose=False,
                 fname=r'C:\Users\sophie_bauchinger\Documents\Github\iau-caribic\misc_data\RIGBY_2010_SF6_MOLE_FRACTION_1970_2008.nc'):
        """ Create dataset from given file

        Parameters:
            remap_lon (bool): convert longitude from 0-360 to ±180 degrees
            verbose (bool): make the function more talkative
        """
        with xr.open_dataset(fname) as ds:
            ds = ds.isel(level=27)
        try:
            ds = ds.sel(time=self.years)
        finally:  # keep only data for specified years
            ds = xr.concat([ds.sel(time=y) for y in self.years
                            if y in ds.time], dim='time')
            if verbose: print(f'No data found for \
                              {[y for y in self.years if y not in ds.time]} \
                              in {self.source}')
            self.years = list(ds.time.values)  # only include actually available years

        if remap_lon:  # set longitudes between 180 and 360 to start at -180 towards 0
            new_lon = (((ds.longitude.data + 180) % 360) - 180)
            ds = ds.assign_coords({'longitude': ('longitude', new_lon,
                                                 ds.longitude.attrs)})
            ds = ds.sortby(ds.longitude)  # reorganise values

        self.data['ds'] = ds
        df = tools.ds_to_gdf(self.ds)
        df.rename(columns={'SF6': 'SF6_MZT'}, inplace=True)
        self.data['df'] = df
        try:
            self.data['SF6'] = self.data['df']['SF6_MZT']
        finally:
            pass

        return ds  # xr.concat(datasets, dim = 'time')

    @property
    def ds(self) -> xr.Dataset:
        return self.data['ds']

    @property
    def df(self) -> pd.DataFrame:
        return self.data['df']

    @property
    def SF6(self) -> xr.Dataset:
        return self.data['SF6']


# %% Local data
class LocalData():
    """ Contains time-series data from ground-based stations.

    Attributes:
        years (List[int]) : years included in the stored data
        source (str) : data source, e.g. 'Mauna_Loa'
        ID (str) : short identifier of the data source
        substances (List[str]) : compounds included in the data
        data (Dict[str:pd.DataFrame]) : dictionary of dataframes, keys of form 'co2'
        df (pd.DataFrame) : combined information from all dataframes

    Methods:
        get_data(path):
            import time-series data from files in given path

    """

    def __init__(self, years, substances):
        self.years = years
        if not hasattr(self, 'ID'): self.ID = None
        if not hasattr(self, 'source'): self.source = None
        if substances == 'all':
            self.substances = [s.short_name for s in
                               dcts.get_substances(source=self.source)
                               if not s.short_name.startswith('d_')]
        elif isinstance(substances, (list, set)):
            self.substances = substances
        else:
            raise TypeError(f'<substances> needs to be one of \'all\', \'list\', \'set\', not {type(substances)}')
        self.data = {}

    def create_df(self) -> pd.DataFrame:
        """ Combine all available dataframes in data into one. """
        df = pd.DataFrame()
        for data_value in self.data.values():
            if isinstance(data_value, pd.DataFrame):
                df = pd.concat([df, data_value], axis=1)
        return df

    @property
    def df(self) -> pd.DataFrame:
        if 'df' in self.data:
            return self.data['df']  #
        return self.create_df()


class MaunaLoa(LocalData):
    """ Stores data for all substances measured at Mauna Loa. """

    def __init__(self, years=range(1999, 2021), substances='all', data_D=False,
                 path_dir=r'C:\Users\sophie_bauchinger\Documents\GitHub\iau-caribic\misc_data'):
        """ Initialise Mauna Loa object with all available substance data. """
        self.source = 'Mauna_Loa'
        self.ID = 'MLO'
        super().__init__(years=years, substances=substances)
        if data_D: self.data_D = {}
        self.get_data(path_dir, data_D)

    def __repr__(self):
        return f'Mauna Loa - {self.substances}'

    def get_data(self, path_dir: str, data_D=False):
        """ Add data for all substances in the given directory to data dictionary. """
        for subs in self.substances:
            self.data[subs] = self.get_subs_data(path_dir, subs, freq='M')
            if data_D:
                try:
                    self.data_D[subs] = self.get_subs_data(path_dir, subs, freq='D')
                finally:
                    print(f'No daily MLO data found for {subs}')
                    continue
        return self

    def get_subs_data(self, path_dir: str, subs: dcts.Substance, freq='M') -> pd.DataFrame:
        """ Import data from Mauna Loa files.

        Parameters:
            path_dir (str): Parent directory containing data files
            subs (str): short name of substance data to import
            freq (str): data frequency, 'M' / 'D'
        """
        if subs not in ['sf6', 'n2o', 'ch4', 'co2', 'co']:
            raise NotImplementedError(f'Data format and filepaths not defined for {subs}')


        if freq == 'D' and subs != 'sf6':
            raise Warning('Daily data from Mauna Loa only available for sf6.')

        # get correct path for the chosen substance
        fnames = {
            'sf6': r'/mlo_SF6_{}.dat'.format('Day' if freq == 'D' else 'MM'),
            'n2o': '/mlo_N2O_MM.dat',
            'co':  '/co_mlo_surface-flask_1_ccgg_month.txt',
            'co2': '/co2_mlo_surface-insitu_1_ccgg_MonthlyData.txt',
            'ch4': '/ch4_mlo_surface-insitu_1_ccgg_MonthlyData.txt',
            }
        
        path = path_dir + fnames[subs]

        # 'ch4', 'co', 'co2' : 1st line has header_lines
        if subs in ['co2', 'ch4', 'co']:
            data_format = 'ccgg'
            with open(path, encoding='utf-8') as f:
                header_lines = int(f.readline().split(' ')[-1].strip())
                title = f.readlines()[header_lines - 2].split()
                if title[0].startswith('#'): title = title[2:]  # CO data

        elif subs in ['sf6', 'n2o']:
            data_format = 'CATS'
            header_lines = 0
            with open(path, encoding='utf-8') as f:
                for line in f:
                    if line.startswith('#'):
                        header_lines += 1
                    else:
                        title = line.split()
                        break

        mlo_data = np.genfromtxt(path, skip_header=header_lines)
        df = pd.DataFrame(mlo_data, columns=title, dtype=float)

        # get names of year and month column (depends on substance)
        if data_format == 'CATS':
            yr_col = [x for x in df.columns if 'catsMLOyr' in x][0]
            mon_col = [x for x in df.columns if 'catsMLOmon' in x][0]
        elif data_format == 'ccgg':
            yr_col = 'year'
            mon_col = 'month'

        # keep only specified years
        df = df.loc[df[yr_col] > min(self.years) - 1].loc[df[yr_col] < max(self.years) + 1].reset_index()

        if any('catsMLOday' in s for s in df.columns):  # check if data has day column
            day_col = [x for x in df.columns if 'catsMLOday' in x][0]
            time = [dt.datetime(int(y), int(m), int(d)) for y, m, d in zip(df[yr_col], df[mon_col], df[day_col])]
            df = df.drop(day_col, axis=1)  # get rid of day column
        else:
            time = [dt.datetime(int(y), int(m), 15) for y, m in
                    zip(df[yr_col], df[mon_col])]  # choose middle of month for monthly data

        if data_format == 'CATS':
            df = df.drop(df.iloc[:, :3], axis=1)  # get rid of now unnecessary time data

        elif data_format == 'ccgg':
            filter_cols = [c for c in df.columns if c not in ['value', 'value_std_dev']]
            df.drop(filter_cols, axis=1, inplace=True)
            df.dropna(how='any', subset='value', inplace=True)
            df.rename(columns={'value': f'{subs}_{self.ID}', 'value_std_dev': f'{subs}_std_dev_{self.ID}'},
                      inplace=True)

        df.astype(float)
        df['Date_Time'] = time
        df.set_index('Date_Time', inplace=True)  # make the datetime object the new index
        if data_format == 'CATS':
            df.dropna(how='any', subset=str(subs.upper() + 'catsMLOm'), inplace=True)
        if data_format == 'ccgg' and subs != 'co':
            df.replace([-999.999, -999.99, -99.99, -9], np.nan, inplace=True)
            df.dropna(how='any', subset=f'{subs}_{self.ID}', inplace=True)

        if 'dict' not in self.data:
            self.data['dict'] = {}
        self.data['dict'].update({k: dcts.get_subs(col_name=k) for k in df.columns
                                  if k in [s.col_name for s in dcts.get_substances()]})
        return df

class MaceHead(LocalData):
    """ Stores data for SF6 measurements at Mace Head. """

    def __init__(self, years=(2012), substances='all', 
                 path=r'C:\Users\sophie_bauchinger\Documents\Github\iau-caribic\misc_data\MHD-medusa_2012.dat'):
        """ Initialise Mace Head with (daily and) monthly data in dataframes """

        self.source = 'Mace_Head'
        self.ID = 'MHD'
        super().__init__(years, substances=substances)
        self.path = path
        self.get_data()

    def __repr__(self):
        return f'Mace Head - {self.substances}'

    def get_data(self) -> pd.DataFrame:
        """ Import data from path definitely in init """
        if not hasattr(self, 'data'): self.data = {}
        header_lines = 0
        with open(self.path, encoding='utf-8') as f:
            for i, line in enumerate(f):
                if line.split()[0] == 'unit:':
                    title = list(f)[0].split()  # takes row below units, which is chill
                    header_lines = i + 2
                    break

        # take only relevant substances and feed column info into dictionary
        column_dict = {f'{name}_{self.ID}':
                           dcts.get_subs(short_name=name.lower(), source=self.source)
                       for name in title if name.lower() in self.substances}

        if 'dict' not in self.data:
            self.data['dict'] = column_dict

        mhd_data = np.genfromtxt(self.path, skip_header=header_lines)
        df = pd.DataFrame(mhd_data, dtype=float,
                          columns=[f'{name}_{self.ID}' for name in title])
        df = df.replace(0, np.nan)  # replace 0 with nan for statistics
        df = df.drop(df.iloc[:, :7], axis=1)  # drop unnecessary time columns
        df = df.astype(float)

        df['Date_Time'] = fractionalyear_to_datetime(mhd_data[:, 0])
        df.set_index('Date_Time', inplace=True)  # new index is datetime
        cols = list(column_dict)
        df = df[cols]

        self.data_Hour = {'df': df}
        self.data_Day = {'df': tools.daily_mean(df)}
        self.data['df'] = tools.monthly_mean(df)

        return df

# #%% Function calls
# if __name__=='__main__':
#     caribic = Caribic()
#     emac = EMAC()
#     tpause = TropopauseData()
#     mzt = Mozart()
#     mlo = MaunaLoa()
#     mhd = MaceHead()
