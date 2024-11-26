# -*- coding: utf-8 -*-
""" Class definitions for data import and analysis from various sources.

@Author: Sophie Bauchinger, IAU
@Date: Fri Apr 28 14:13:28 2023

"""
from abc import abstractmethod
import copy
import dill
import pandas as pd
import matplotlib.patheffects as mpe
from metpy import calc
from metpy.units import units
import numpy as np
import os
import warnings
import xarray as xr

import dataTools.dictionaries as dcts
from dataTools import tools
from dataTools.data.mixin_analysis import AnalysisMixin, BinningMixin, TropopauseSorterMixin 
from dataTools.data.mixin_selection import SelectionMixin

# #!! TODO: fix the underlying problem in toolpac rather than just suppressing stuff
# from pandas.errors import SettingWithCopyWarning
# warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

def outline(): 
    """ Helper function to add outline to lines in plots. """
    return mpe.withStroke(linewidth=2, foreground='white')

class ModelDataMixin:
    """ Import / Calculate new dataframes  
    
    Methods: 
        get_clams_data(met_dir, save_ds, recalculate)
            Creates dataframe for ERA5 / CLaMS data from netcdf files. 
        calc_coordinates(**kwargs)
            Calculate additional coordinates as specified through .var1 and .var2.
    """

    @abstractmethod
    def get_met_data(self):
        """ Require existance of dataframe creation method for child classes. """
        if self.ID in ['CAR', 'ATOM', 'HIPPO', 'SHTR', 'PGS', 'WISE', 'PHL']: 
            return self.get_clams_data()
        else: 
            raise NotImplementedError(f'Subclass of GlobalData ( - {self.ID}): need to specifically implement .get_met_data()')

    def get_clams_data(self, met_pdir=None, save_ds=False, recalculate=False) -> pd.DataFrame:
        """ Creates dataframe for CLaMS data from netcdf files. """
        if self.ID not in ['CAR', 'SHTR', 'WISE', 'ATOM', 'HIPPO', 'PGS', 'PHL']:
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
                'CAR'  : ('CaribicTPChange',    5),
                'SHTR' : ('SouthtracTPChange',  5),
                'WISE' : ('WiseTPChange',       5),
                'ATOM' : ('AtomTPChange',       5),
                'HIPPO': ('HippoTPChange',      5),
                'PGS'  : ('PolstraccTPChange',  5),
                'PHL' : ('PhileasTPChange',     5),
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
            self.pfxs = self.pfxs+['CLAMS'] if hasattr(self, 'pfxs') else ['CLAMS']
        else: 
            self.data['met_data'] = met_df

        return met_df

    def calc_coordinates(self, **kwargs): # Calculates mostly tropopause coordinates
        """ Calculate coordinates as specified through .var1 and .var2. """
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


# %% Global data
class GlobalData(SelectionMixin, BinningMixin, TropopauseSorterMixin, AnalysisMixin, ModelDataMixin):
    """ Contains global datasets with longitude/latitude for each datapoint.

    Attributes:
        years(List[int]) : years included in the stored data
        source (str) : source of the input data, e.g. 'Caribic'
        grid_size (int) : default grid size for binning
        status (dict) : stores information on operations that change the stored data

    Methods:
    
    --- AnalysisMixin
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
    
    --- ModelDataMixin
        get_clams_data(met_dir, save_ds, recalculate)
            Creates dataframe for ERA5 / CLaMS data from netcdf files. 
        calc_coordinates(**kwargs)
            Calculate additional coordinates as specified through .var1 and .var2.
    
    --- TropopauseMixin
        n2o_filter(**kwargs)
            Use N2O data to create strato/tropo reference for data
        create_df_sorted(**kwargs)
            Use all chosen tropopause definitions to create strato/tropo reference
        calc_ratios(group_vc=False)
            Calculate ratio of tropo/strato datapoints

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
        sel_atm_layer(atm_layer, **kwargs)
            Remove all data not in the chosen atmospheric layer (tropo/strato)
        sel_tropo()
            Remove all stratospheric datapoints
        sel_strato()
            Remove all tropospheric datapoints

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
            pdir = tools.get_path() + '\\misc_data\\pickled_dicts\\'
        
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
            (c.vcoord =='mxr' or str(c.rel_to_tp) != 'nan') ) ]

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

    def get_var_data(self, var, **kwargs) -> np.array: 
        """ Returns variable data including from geometry columns. 
        Args: 
            var (dcts.Coordinate, dcts.Substance)
            key df (pd.DataFrame): Data from this dataframe will be returned. Optional. 
        """
        if var.col_name == 'geometry.y': 
            data = kwargs.get('df', self.df).geometry.y
        elif var.col_name == 'geometry.x': 
            data = kwargs.get('df', self.df).geometry.x
        else: 
            data = np.array(kwargs.get('df', self.df)[var.col_name])
        return data

    def get_var_lims(self, var, bsize=None, **kwargs) -> tuple[float]: 
        """ Returns outer limits based on variable data and (optional) bin size. 
        Args: 
            var (dcts.Coordinate, dcts.Substance)
            bsize (float): Bin size. Optional. 
            databased (bool): Toggle calculating limits from available data. Default True for everything but Lon/Lat. 

            key df (pd.DataFrame): Limits will be calculated from data in this dataframe. Optional. 
        """
        if isinstance(var, dcts.Coordinate) and not kwargs.get('databased'): 
            try: 
                return var.get_lims()
            except ValueError: 
                pass
        
        v_data = self.get_var_data(var, **kwargs)
        vmin = np.nanmin(v_data)
        vmax = np.nanmax(v_data)
        
        if bsize is None: 
            return vmin, vmax

        vbmin = (vmin // bsize) * bsize
        vbmax = ((vmax // bsize) + 1) * bsize
        return vbmin, vbmax

# --- Calculate additional variables from existing information ---
    def create_tp_coords(self) -> pd.DataFrame:
        """ Add calculated relative / absolute tropopause values to .met_data """
        df = self.met_data.copy()
        new_coords = dcts.get_coordinates(**{'ID': 'int_calc', 'source': 'Caribic'})
        new_coords = new_coords + dcts.get_coordinates(**{'ID': 'int_calc', 'source': 'CLAMS'})
        new_coords = new_coords + dcts.get_coordinates(**{'ID': 'CLAMS_calc', 'source': 'CLAMS'})
        new_coords = new_coords + dcts.get_coordinates(**{'ID': 'CLAMS_calc', 'source': 'Caribic'})

        for coord in new_coords:
            # met = tp + rel -> MET - MINUS for either one
            met_col = coord.var1
            met_coord = dcts.get_coord(col_name = met_col)
            minus_col = coord.var2

            if met_col in df.columns and minus_col in df.columns:
                df[coord.col_name] = df[met_col] - df[minus_col]

            elif met_coord.var == 'geopot' and met_col in df.columns:
                met_data = df[met_col].values * units(met_coord.unit)
                height_m = calc.geopotential_to_height(met_data)
                height_km = height_m * 1e-3

                if coord.unit == 'm': 
                    df[coord.col_name] = height_m
                elif coord.unit == 'km': 
                    df[coord.col_name] = height_km

            else:
                print(f'Could not generate {coord.col_name} as precursors are not available')

        self.data['met_data'] = df
        if 'df' in self.data: 
            self.create_df() # Recompile self.df with new TP coordinates
        return df

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
