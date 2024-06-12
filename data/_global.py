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
import dill
import pandas as pd
import matplotlib.patheffects as mpe
import warnings

import dataTools.dictionaries as dcts
from dataTools import tools

from dataTools.data.mixin_selection import SelectionMixin
from dataTools.data.mixin_binning import BinningMixin
from dataTools.data.mixin_tropopause import TropopauseSorterMixin
from dataTools.data.mixin_analysis import AnalysisMixin
from dataTools.data.mixin_model_data import ModelDataMixin

# #!! TODO: fix the underlying problem in toolpac rather than just suppressing stuff
# from pandas.errors import SettingWithCopyWarning
# warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

def outline(): 
    """ Helper function to add outline to lines in plots. """
    return mpe.withStroke(linewidth=2, foreground='white')

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
