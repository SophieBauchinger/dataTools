# -*- coding: utf-8 -*-
""" Caribic data class definition

@Author: Sophie Bauchinger, IAU
@Date: Tue Jun 11 17:35:00 2024
"""
import pandas as pd
import traceback

from dataTools.data._global import GlobalData
import dataTools.dictionaries as dcts
from dataTools import tools
from dataTools.data import data_getter

# cols = list(self.df.columns)
# cols.sort()
# coord_cols = ['Flight number', 'p', 'alt', 'int_Theta'] # ... ['geometry']
# ghg_cols = ['CH4', 'd_CH4', 'CO2', 'd_CO2', 'SF6', 'd_SF6', 'N2O', 'd_N2O']

# my_cols = coord_cols + ghg_cols + [c for c in cols if c not in coord_cols+ghg_cols+['geometry']] + ['geometry']
# len(my_cols), len(cols)

# Caribic
class Caribic(GlobalData): 
    """ Stores all available information from Caribic datafiles and allows further analsis. 
    
    Attributes: 
        pfxs (List[str]): Prefixes of tored Caribic data files
    
    Methods: 
        coord_combo()
            Create met_data from available meteorological data
        create_substance_df(detr=False):
            Combine met_data with all substance info, optionally incl. detrended
    """

    def __init__(self, years=range(2005, 2021), pfxs=('GHG', 'INTtpc'),
                 grid_size=5, verbose=False, recalculate=False):
        """ Constructs attributes for Caribic object and creates data dictionary.
        
        Parameters:
            years (List[int]) : import data only for selected years
            pfxs (List[str]) : prefixes of Caribic files to import
            grid_size (int) : grid size in degrees to use for binning
            verbose (bool) : print additional debugging information
            recalculate (bool) : get data from pre-combined file or parent directory
        """
        # no caribic phase-2 data before 2005
        super().__init__([yr for yr in years if yr > 2004], grid_size)

        self.source = 'Caribic'
        self.ID = 'CAR'
        self.pfxs = pfxs
        self.get_data(verbose=verbose, recalculate=recalculate)  # creates self.data dictionary
        if 'df' not in self.data: 
            self.create_df()
        
        if 'met_data' not in self.data:
            try:
                self.data['df'] = data_getter.create_tp_coords(self.df)
                self.data['met_data'] = self.coord_combo()  # reference for met data for all msmts
            except Exception:
                traceback.print_exc()

    def __repr__(self):
        return f"""{self.__class__}
    data: {self.pfxs}
    years: {self.years}
    status: {self.status}"""

    def get_data(self, recalculate=False, fname:str=None, 
                 verbose=False, source_pdir=None): 
        """ Imports Caribic data in the form of geopandas dataframes.

        Returns data dictionary containing dataframes for each file source and
        dictionaries relating column names with Coordinate / Substance instances.

        Parameters:
            recalculate (bool): Data is imported from source instead of using pickled dictionary.
            fname (str): specify File name of data dictionary if default should not be used.
            verbose (bool): Makes function more talkative.
            source_pdir (str): Parent directory of stored Caribic AMES files. 
        """
        if not recalculate: # Load and check the saved DATA dictionary
            data_dict, updated_status, filepath = data_getter.load_DATA_dict(self.ID, self.status, fname)
            self.status = updated_status
            if verbose: print(f'Loaded CARIBIC data from {filepath}')

            # Check data for pfxs and years: 
            if all(pfx in data_dict.keys() for pfx in self.pfxs):
                data = {k:data_dict[k] for k in self.pfxs} # choose only the requested pfxs
            else: 
                print(f'Warning: Could not load {[pfx for pfx in self.pfxs if pfx not in data.keys()]}')
            data = self.sel_year(*self.years).data
            
            for special_item, generator in [
                ('df', '.create_df()'),
                ('met_data', '.get_met_data()'),
                ('df_sorted', '.get_df_sorted()'),
                ('MODEL', '.get_model_data()')]:
                if special_item in data_dict: 
                    data[special_item] = data_dict[special_item]
                    if verbose: print(f'Loaded \'{special_item}\' from saved data. Call {generator} to generate anew. ')
            self.data = data
            return data

        if 'MODEL' in self.pfxs: 
            model_dataframe = data_getter.import_era5_data(self.ID)
            self.data['MODEL'] = model_dataframe

        # Recalculate from AMES files
        data, year_tracker = data_getter.CARIBIC_AMES_data(
            self.pfxs, self.years, verbose, parent_dir=source_pdir)

        self.years = [yr for yr in year_tracker.keys() if year_tracker[yr]] 
        if not all(year_tracker.values()): 
            # show tracker if there were unsuccessful years 
            print(year_tracker) 
        self.data = data

        return data

    def _prepare_for_tp_comp(self):
        """ Set default state for tropopause comparisons:
            
            select latitudes > 30°N
            set tropopause coordinates 
            remove non-shared timestamps
            set count_limit to 3
            (re)calculate N2O tropopause with ol_limit = 0.01
        """
        self.sel_latitude(30, 90, inplace=True)
        self.count_limit = 3
        
        self.create_df_sorted() # creates N2O_baseline
        
        self.set_tps(rel_to_tp=True, vcoord='z', model = 'ERA5')
        o3_tps = self.get_coords(rel_to_tp=True, crit='o3')
        if len(o3_tps) > 1 and all([tp.model=='MSMT' for tp in o3_tps]): 
            o3_tps.sort(key=lambda x:x.ID); o3_tps.pop(0)
        self.tps += o3_tps
        self.tps += self.get_coords(col_name = 'N2O')
        self.tps.sort(key=lambda x: x.tp_def)

        self.sel_shared_indices(inplace=True) # only tps in df_sorted
        
        if any(tp.crit=='n2o' for tp in self.tps):
            # Recalculate N2O tropopause with reduced data set
            self.create_df_sorted(ol_limit = 0.01)
        if 'season' not in self.df.columns:
            self.df['season'] = tools.make_season(self.df.index.month)

        return self

    def coord_combo(self) -> pd.DataFrame:
        """ Create dataframe with all possible coordinates but
        no measurement / substance values """
        # merge lists of coordinates for all pfxs in the object
        if 'GHG' in self.pfxs:
            # copy bc don't want to overwrite data
            df = self.data['GHG'].copy()
        else:
            df = pd.DataFrame()
        for pfx in [pfx for pfx in self.pfxs if pfx != 'GHG']:
            df = df.combine_first(self.data[pfx].copy())

        essentials = [c for c in df.columns if c in ['Flight number', 'p', 'geometry']]
        coords = [c for c in df.columns if c in [i.col_name for i in dcts.get_coordinates()]] 

        drop_cols = [c for c in df.columns if c not in list(coords + essentials)] 
        df.drop(drop_cols, axis=1, inplace=True)  # remove non-met / non-coord data

        # reorder columns
        columns = list(['Flight number'] 
                        + (['p'] if 'p' in df.columns else [])
                        + [col for col in df.columns
                           if col not in ['Flight number', 'p', 'geometry']]
                        + ['geometry'])

        self.data['met_data'] = df[columns]
        return self.data['met_data']

    def create_df(self) -> pd.DataFrame:
        """ Join together data from all pfx-sources. """
        df = self.met_data.copy() # CLAMS data should be included here already
        
        merge_kwargs = dict(
            how='outer', 
            sort=True,
            left_index=True, 
            right_index=True, 
            )

        for pfx in self.pfxs:
            df = pd.merge(df, self.data[pfx],
                          suffixes = [None, f'_{pfx}'],
                          **merge_kwargs)
            
            for c in df.columns:
                if f'{c}_{pfx}' in df.columns:
                    df[c] = df[c].combine_first(df[f'{c}_{pfx}']) # Note: Future warning, watch but should be okay
                    df = df.drop(columns=f'{c}_{pfx}')
        if 'geometry' in df.columns: 
            df = df[df.index.isin(df.geometry.dropna().index)]
        
        # df = data_getter.create_tp_coords(df)

        self.data['df'] = df
        return df

    @property
    def GHG(self) -> pd.DataFrame:
        if 'GHG' in self.data:
            return self.data['GHG']
        raise Warning('No GHG data available')

    @property
    def INTtpc(self) -> pd.DataFrame:
        if 'INTtpc' in self.data:
            return self.data['INTtpc']
        raise Warning('No INTtpc data available')

    @property
    def MS(self) -> pd.DataFrame: 
        if 'MS' in self.data: 
            return self.data['MS']
        raise Warning('No MS data available')

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
