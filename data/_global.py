# -*- coding: utf-8 -*-
""" Class definitions for data import and analysis from various sources.

@Author: Sophie Bauchinger, IAU
@Date: Fri Apr 28 14:13:28 2023

class GlobalData
class Caribic(GlobalData)
class EMAC(GlobalData)
class TropopauseData(GlobalData)
class Mozart(GlobalData)

class LocalData
class Mauna_Loa(LocalData)
class Mace_Head(LocalData)
"""

# TODO ATOM flight & latitude selection
from abc import abstractmethod
import copy
import dill
import matplotlib.pyplot as plt
from metpy import calc
from metpy.units import units
import numpy as np
import os
import pandas as pd
import warnings
import xarray as xr

from toolpac.conv.times import datetime_to_fractionalyear as dt_to_fy # type: ignore
from toolpac.outliers import outliers # type: ignore

import dataTools.dictionaries as dcts
from dataTools import tools

from dataTools.data.mixin_selection import SelectionMixin
from dataTools.data.mixin_binning import BinningMixin
from dataTools.data._local import MaunaLoa

#!! TODO: fix the underlying problem in toolpac rather than just suppressing stuff
from pandas.errors import SettingWithCopyWarning

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

# %% Global data
class GlobalData(SelectionMixin, BinningMixin):
    """ Contains global datasets with longitude/latitude for each datapoint.

    Attributes:
        years(List[int]) : years included in the stored data
        source (str) : source of the input data, e.g. 'Caribic'
        grid_size (int) : default grid size for binning
        status (dict) : stores information on operations that change the stored data

    Methods:
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
        detrend_substance(substance, ...)
            Remove trend wrt. 2005 Mauna Loa from substance, then add to data
    
    --- BinningMixin
        binned_1d(subs, **kwargs)
            Bin substance data over latitude
        binned_2d(subs, **kwargs)
            Bin substance data on a longitude/latitude grid

    --- SelectionMixin
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

    """

    def __init__(self, years, grid_size=5, count_limit=5, **kwargs):
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
        self.tps = ()

    def pickle_data(self, fname: str, pdir=None):
        """ Save data dictionary using dill. """
        if len(fname.split('.')) < 2:
            fname = fname + '.pkl'
        
        if not pdir: 
            pdir = tools.get_path() + '\\misc_data\\'
        
        with open(pdir + fname, 'wb') as f:
            dill.dump(self.data, f)
            print(f'{self.ID} Data dictionary saved to {pdir}\{fname}')

# --- Instance variables (substances / coordinates) ---
    def get_variables(self, category):
        """ Returns list of variables from chosen category with column in self.df """
        if 'df' not in self.data:
            raise KeyError(f'self.data.df not found, cannot return {category} variables.')
        variables = []
        for column in [c for c in self.data['df'] if not c == 'geometry']:
            try:
                if category == 'subs':
                    var = dcts.get_subs(col_name=column)
                elif category == 'coords':
                    var = dcts.get_coord(col_name=column)
                else:
                    continue
                variables.append(var)
            except KeyError:
                continue
        if 'geometry' in self.data['df'].columns and category == 'coords': 
            variables.append(dcts.get_coord(col_name = 'geometry.y'))
            variables.append(dcts.get_coord(col_name = 'geometry.x'))
        return variables

    @property
    def substances(self) -> list:
        """ Returns list of substances in self.df """
        return self.get_variables('subs')

    @property
    def coordinates(self) -> list:
        """ Returns list of non-substance variables in self.df """
        return self.get_variables('coords')

    def get_coords(self, **coord_kwargs) -> list: 
        """ Returns all coordinates that fit the specified parameters and exist in self.df """
        try: 
            coords = [tp for tp in self.coordinates 
                  if tp.col_name in [c.col_name for 
                                     c in dcts.get_coordinates(**coord_kwargs)]]
        except KeyError: 
            coords = []
            warnings.warn('Warning. No coordinates found in data using the given specifications.')
        return coords

    def get_substs(self, **subs_kwargs) -> list: 
        """ Returns all substances that fit the specified parameters and exist in self.df """
        try: 
            substs = [subs for subs in self.substances
                      if subs.col_name in [s.col_name for
                                           s in dcts.get_substances(**subs_kwargs)]]
        except KeyError: 
            substs = []
            warnings.warn('Warning. No substances found in data using the given specifications.')
        return substs

    def get_tps(self, **tp_kwargs) -> list: 
        """ Returns a list of vertical dynamic coordinates that fulfill conditions in tp_kwargs. """
        # 1. filter coordinates for tropopause-relative coordinates only
        tps = [c for c in self.coordinates if (
            str(c.tp_def) != 'nan' and 
            c.var != 'geopot' and 
            (c.vcoord =='mxr' or c.rel_to_tp) ) ]

        # 2. reduce list further using given keyword arguments
        try: 
            filtered_coord_columns = [c.col_name for c in dcts.get_coordinates(**tp_kwargs)]
            tps = [tp for tp in tps if tp.col_name in filtered_coord_columns]

        except KeyError: 
            tps = []
            warnings.warn('Warning. No TP coordinates found in data using the given specifications.')
        return tps
    
    def set_tps(self, **tp_kwargs): 
        """ Set .tps (shorthand for tropopause coordinates) in accordance with tp_kwargs. """
        self.tps = self.get_tps(**tp_kwargs)

# --- Define additional attributes ---
    @property
    def flights(self):
        """ Returns list of flights (as names or numbers) in main dataframe. """
        if 'df' not in self.data:
            raise KeyError('Cannot return available flights without main dataframe.')
        flight_columns = [c for c in self.df.columns if 'flight' in c.lower()]
        if len(flight_columns) < 1:
            raise KeyError('Flight information not available in dataframe.')
        flights = set(self.df[flight_columns[0]])
        return list(flights)

    @property
    @abstractmethod
    def df(self) -> pd.DataFrame:
        if 'df' in self.data:
            return self.data['df']
        return self.create_df()

    @abstractmethod
    def create_df(self):
        """ Require existance of dataframe creation method for child classes. """
        raise NotImplementedError('Child classes need to implement .create_gf()')

    @property
    @abstractmethod
    def met_data(self):
        if 'met_data' in self.data:
            return self.data['met_data']
        return self.get_met_data()

    def __add__(self, glob_obj):
        """ Combine two GlobalData objects into one. Keep only main dataframes. """
        print('Combining objects: \n', self, '\n', glob_obj)

        out = type(self).__new__(GlobalData)  # new class instance
        out.__init__(years=list(set(self.years + glob_obj.years)))
        setattr(out, 'source', 'MULTI')
        setattr(out, 'ID', 'MULTI')

        if 'df_combined' in self.data: 
            if 'df_combined' not in glob_obj.data:
                # add ID as index / column
                new_df = pd.concat([glob_obj.df], 
                                   keys=[glob_obj.ID], 
                                   names=['ID', 'DATETIME'])
            else: 
                new_df = glob_obj.data['df_combined']
            combined_df = pd.concat([self.data['df_combined'], new_df])

        else:
            new_df = glob_obj.data['df']
            combined_df = pd.concat([self.data['df'], new_df], 
                                    keys=[self.ID, glob_obj.ID], 
                                    names=['ID', 'DATETIME'])

        out.data['df_combined'] = combined_df
        
        df = combined_df.reset_index().set_index('DATETIME')
        
        if any(df.index.duplicated()):
            dropped_rows = df[df.index.duplicated()]
            print(f'Dropping {sum(df.index.duplicated())} duplicated timestamps.', dropped_rows)
            df = df[~ df.index.duplicated()]

        out.data['df'] = df
        out.data['ID_per_timestamp'] = combined_df.reset_index().set_index('DATETIME')['ID']
        return out

# --- Import / Calculate new dataframes ---
    @abstractmethod
    def get_met_data(self):
        """ Require existance of dataframe creation method for child classes. """
        if self.ID in ['CAR', 'ATOM', 'HIPPO', 'SHTR', 'PGS', 'WISE']: 
            return self.get_clams_data()
        else: 
            raise NotImplementedError('Subclasses of GlobalData need to implement .get_met_data()')

    def get_clams_data(self, met_pdir=None, save_ds=False, recalculate=False) -> pd.DataFrame:
        """ Creates dataframe for CLaMS data from netcdf files. """
        if self.ID not in ['CAR', 'SHTR', 'WISE', 'ATOM', 'HIPPO', 'PGS']:
            raise KeyError(f'Cannot import CLaMS data for ID {self.ID}')

        alldata_fname = {
            'CAR' : 'caribic_clams_V03.nc'
            }
        if (self.ID in alldata_fname \
            and os.path.exists(tools.get_path() + 'misc_data/' + alldata_fname.get(self.ID)) \
            and not recalculate):
                with xr.open_dataset(tools.get_path() + 'misc_data/' + alldata_fname.get(self.ID)) as ds: 
                    ds = ds
        else: 
            print('Importing CLAMS data')
            campaign_dir_version_dict = { # campaign_pdir, version
                'CAR'  : ('CaribicTPChange',    4),
                'SHTR' : ('SouthtracTPChange',  2),
                'WISE' : ('WiseTPChange',       2), #!!! 4
                'ATOM' : ('AtomTPChange',       4),
                'HIPPO': ('HippoTPChange',      2),
                'PGS'  : ('PolstraccTPChange',  2),
                'PHL' : ('PhileasTPChange',     4),
                }
            campaign_pdir, version = campaign_dir_version_dict[self.ID]
            met_pdir = r'E:/TPChange/' + campaign_pdir
            
            fnames = met_pdir + "/*.nc"
            if self.ID == 'CAR': 
                fnames = met_pdir + "/2*/*.nc"
                
            drop_variables = {'CAR' : ['CARIBIC2_LocalTime'], 
                                'ATOM' : ['ATom_UTC_Start', 'ATom_UTC_Stop', 'ATom_End_LAS']}
                
            # extract data, each file goes through preprocess first to filter variables & convert units
            with xr.open_mfdataset(fnames, 
                                    preprocess = tools.process_TPC if not version==2 else tools.process_TPC_V02,
                                    drop_variables = drop_variables.get(self.ID),
                                    ) as ds:
                ds = ds

        if save_ds: 
            self.data['met_ds'] = ds

        met_df = ds.to_dataframe()

        if self.ID=='CAR': 
            self.data['CLAMS'] = met_df
            self.pfxs = self.pfxs.append('CLAMS')
        else: 
            self.data['met_data'] = met_df

        return met_df

    def calc_coordinates(self, **kwargs): # Calculates mostly tropopause coordinates
        """ Calculate coordinates as specified in through .var1 and .var2. """
        data = self.df
        
        if kwargs.get('recalculate'): 
            data.drop(columns = [c.col_name for c in self.coordinates if c.ID=='calc'], 
                      inplace=True)
        
        all_calc_coords = dcts.get_coordinates(ID='calc') \
                        + dcts.get_coordinates(ID='CLAMS_calc') \
                        + dcts.get_coordinates(ID='MS_calc') \
                        + dcts.get_coordinates(ID='EMAC_calc')

        # Firstly calculate geopotential height from geopotential
        geopot_coords = [c for c in all_calc_coords if (
            c.var1 in data.columns and str(c.var2) == 'nan' )]
        
        for coord in geopot_coords: 
            
            met_data = data[coord.var1].values * units(dcts.get_coord(coord.var1).unit)
            
            # geopot_values = data[coord.var1].values
            # geopot = units.Quantity(geopot_values, 'm^2/s^2')
            
            height_m = calc.geopotential_to_height(met_data) # meters
            height_km = height_m * 1e-3
            
            if coord.unit == 'm': 
                data[coord.col_name] = height_m
            elif coord.unit == 'km': 
                data[coord.col_name] = height_km

        # Now calculate TP / distances to TP coordinates 
        calc_coords = [c for c in all_calc_coords if 
            all(col in data.columns for col in [c.var1, c.var2])]
        
        for coord in calc_coords: 
            if kwargs.get('verbose'): 
                print('Calculating ', coord.long_name, 'from \n', 
                  dcts.get_coord(col_name=coord.var1), '\n', # met
                  dcts.get_coord(col_name=coord.var2)) # tp
            
            met_coord = dcts.get_coord(col_name = coord.var1)
            tp_coord = dcts.get_coord(col_name = coord.var2)
            
            met_data = copy.deepcopy(data[coord.var1]) # prevents .df to be overwritten 
            tp_data = copy.deepcopy(data[coord.var2])
            
            if tp_coord.unit != met_coord.unit != coord.unit: 
                if all(unit in ['hPa', 'mbar'] for unit in [tp_coord.unit, met_coord.unit, coord.unit]):
                    pass
                elif all(unit in ['km', 'm'] for unit in [tp_coord.unit, met_coord.unit, coord.unit]): 
                    if coord.unit == 'm': 
                        if tp_coord.unit == 'km': tp_data *= 1e3
                        if met_coord.unit == 'km': met_data *= 1e3
                    elif coord.unit == 'km': 
                        if tp_coord.unit == 'm': tp_data *= 1e-3
                        if met_coord.unit == 'm': met_data *= 1e-3
                
                    if kwargs.get('verbose'): 
                        print('UNIT MISMATCH when calculating ', coord.long_name, 'from \n', 
                        dcts.get_coord(col_name=coord.var1), '\n', # met
                        dcts.get_coord(col_name=coord.var2)) # tp
                        
                        print('Fixed by readjusting: \n',
                              data[coord.var2].dropna().iloc[0], f' [{tp_coord.unit}] -> ', tp_data.dropna().iloc[0], f' [{coord.unit}]\n', 
                              data[coord.var1].dropna().iloc[0], f' [{met_coord.unit}] -> ', met_data.dropna().iloc[0], f' [{coord.unit}]')
                else: 
                    print(f'HALT STOPP: units do not match on {met_coord} and {tp_coord}.')
                    continue
            
            coord_data = (met_data - tp_data)
            data[coord.col_name] = coord_data

        self.data['df'] = data
        return data
    
# --- Helper for comparing datasets properly ---
    def get_shared_indices(self, tps=None, df=False):
        """ Make reference for shared indices of chosen tropopause definitions. """
        if not 'df_sorted' in self.data:
            self.create_df_sorted()
            
        data = self.df_sorted if not df else self.df
        prefix = 'tropo_' if not df else ''
        
        if not tps:
            tps = self.tps #tools.minimise_tps(dcts.get_coordinates(tp_def='not_nan', source=self.source))
        
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
        tps = (self.tps if not kwargs.get('tps') else kwargs.get('tps'))
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


# --- Manipulate substances mixing ratios ---
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

# --- Filters for stratosphere / troposphere ---
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
        loc_obj = MaunaLoa(self.years) if not kwargs.get('loc_obj') else kwargs.get('loc_obj')
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

    def o3_filter_lt60(self, **kwargs) -> pd.DataFrame:
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
            n2o_sorted = self.n2o_filter(coord=tp)
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

# --- Make selections based on strato / tropo characteristics --- 
    def sel_atm_layer(self, atm_layer: str, tp=None, **kwargs):
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

        if not tp: 
            if 'source' not in kwargs and self.source in ['Caribic', 'EMAC']:
                kwargs.update({'source': self.source})
    
            if 'rel_to_tp' not in kwargs and any(
                    c.rel_to_tp for c in self.get_coords(**kwargs)):
                kwargs.update({'rel_to_tp': True})
    
            [tp] = self.get_coords(**kwargs)

        if 'df_sorted' not in self.data:
            df_sorted = self.create_df_sorted(save=False, **kwargs)
        else:
            df_sorted = self.df_sorted

        if f'{atm_layer}_{tp.col_name}' not in df_sorted:
            raise KeyError(f'Could not find {tp.col_name} in df_sorted.')

        atm_layer_col = f'{atm_layer}_{tp.col_name}'

        # Filter all dataframes to only include indices in df_sorted
        df_list = [k for k in self.data
                   if isinstance(self.data[k], pd.DataFrame)
                   and k not in ['df_sorted']]  # list of all datasets to cut
        for k in df_list:
            out.data[k] = out.data[k][out.data[k].index.isin(df_sorted.index)]
            out.data[k] = out.data[k][df_sorted[atm_layer_col].loc[out.data[k].index]]

        if self.source == 'Caribic':
            out.pfxs = [k for k in out.data if k in self.pfxs]

        # update object status
        out.status.update({atm_layer: True})
        if 'TP filter' in out.status:
            out.status['TP filter'] = (out.status['TP filter'], atm_layer_col)
        else:
            out.status['TP filter'] = atm_layer_col

        return out

    def sel_tropo(self, tp=None, **kwargs):
        """ Returns GlobalData object containing only tropospheric data points. """
        return self.sel_atm_layer('tropo', tp, **kwargs)

    def sel_strato(self, tp=None, **kwargs):
        """ Returns GlobalData object containing only tropospheric data points. """
        return self.sel_atm_layer('strato', tp, **kwargs)

    def sel_LMS(self, tp=None, nr_of_bins = 3, **kwargs): #!!!
        """ Returns GlobalData object containing only lowermost stratospheric data points. """
        strato_data = self.sel_strato(tp, **kwargs).df
        
        zbsize = tp.get_bsize() if not kwargs.get('zbsize') else kwargs.get('zbsize')
        
        LMS_data = strato_data[strato_data[tp.col_name] <= zbsize * nr_of_bins]
        
        return LMS_data

    def identify_bins_relative_to_tropopause(self, subs, tp, **kwargs) -> pd.DataFrame:
        """ Flag each datapoint according to its distance to specific tropopause definitions. """
        # TODO implement this
        # !!! implement this

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


