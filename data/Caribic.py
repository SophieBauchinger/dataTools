# -*- coding: utf-8 -*-
""" Caribic data class definition

@Author: Sophie Bauchinger, IAU
@Date: Tue Jun 11 17:35:00 2024
"""
import dill
import geopandas
import numpy as np
import os
import pandas as pd
from shapely.geometry import Point
import traceback

from toolpac.readwrite import find # type: ignore
from toolpac.readwrite.FFI1001_reader import FFI1001DataReader # type: ignore

from dataTools.data._global import GlobalData
import dataTools.dictionaries as dcts
from dataTools import tools

# Caribic
class Caribic(GlobalData): 
    """ Stores all available information from Caribic datafiles and allows further analsis. 
    
    Attributes: 
        pfxs (List[str]): Prefixes of tored Caribic data files
    
    Methods: 
        coord_combo()
            Create met_data from available meteorological data
        create_tp_coordinates()
            Calculate tropopause height etc. from available met data
        create_substance_df(detr=False):
            Combine met_data with all substance info, optionally incl. detrended
    """

    def __init__(self, years=range(2005, 2021), pfxs=('GHG', 'INT', 'INTtpc'),
                 grid_size=5, verbose=False, recalculate=False, fname=None, tps_dict = {}, **kwargs):
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
        # self.flights = ()
        self.get_data(verbose=verbose, recalculate=recalculate, fname=fname)  # creates self.data dictionary
        if 'df' not in self.data: 
            self.create_df()
        self.set_tps(**tps_dict)
        
        if 'met_data' not in self.data:
            try:
                self.data['met_data'] = self.coord_combo()  # reference for met data for all msmts
                self.create_tp_coords()
            except Exception:
                traceback.print_exc()

    def __repr__(self):
        return f"""{self.__class__}
    data: {self.pfxs}
    years: {self.years}
    status: {self.status}"""

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
        self.tps += self.get_coords(rel_to_tp=True, crit='o3')
        self.tps += self.get_coords(col_name = 'N2O')
        self.tps.sort(key=lambda x: x.tp_def)

        self.remove_non_shared_indices(inplace=True)
        
        if any(tp.crit=='n2o' for tp in self.tps):
            # Recalculate N2O tropopause with reduced data set
            self.create_df_sorted(ol_limit = 0.01)
        
        return self

    def get_year_data(self, pfx: str, yr: int, parent_dir: str, verbose: bool) -> tuple[pd.DataFrame, dict]:
        """ Data import for a single year """
        if not any(find.find_dir("*_{}*".format(yr), parent_dir)):
            # removes current year from class attribute if there's no data
            self.years = np.delete(self.years, np.where(self.years == yr))
            if verbose: print(f'No data found for {yr} in {self.source}. \
                              Removing {yr} from list of years')
            return pd.DataFrame(), dict()

        print(f'Reading Caribic - {pfx} - {yr}')
        # Collect data from individual flights for current year
        df_yr = pd.DataFrame()
        
        for current_dir in find.find_dir("Flight*_{}*".format(yr), parent_dir):  # [1:]:
            flight_nr = int(str(current_dir)[-12:-9])
            # flight_nr = int(str(current_dir).split('_')[0].removeprefix('Flight'))

            f = find.find_file(f'{pfx}_*', current_dir)
            if not f or len(f) == 0:  # no files found
                if verbose: print(f'No {pfx} File found for \
                                  Flight {flight_nr} in {yr}')
                continue
            if len(f) > 1:
                f.sort()  # sort to get most recent version with indexing from end

            f_data = FFI1001DataReader(f[-1], df=True, xtype='secofday',
                                       sep_variables=';')
            df_flight = f_data.df  # index = Datetime
            df_flight.insert(0, 'Flight number',
                             [flight_nr] * df_flight.shape[0])

            col_name_dict = tools.rename_columns(f_data.VNAME)
            # set names to their short version
            df_flight.rename(columns=col_name_dict, inplace=True)
            df_yr = pd.concat([df_yr, df_flight])

        # Convert longitude and latitude into geometry objects
        lat_col, lon_col = ('lat', 'lon') if pfx!='MS' else ('PosLat', 'PosLong')
        
        geodata = [Point(lon, lat) for lon, lat in zip(
            df_yr[lon_col], df_yr[lat_col])]
        gdf_yr = geopandas.GeoDataFrame(df_yr, geometry=geodata)

        # Drop cols which are saved within datetime, geometry
        if not gdf_yr['geometry'].empty:
            
            filter_cols = [c for c in gdf_yr.columns 
                           if c in ['TimeCRef', 'year', 'month', 'day',
                           'hour', 'min', 'sec', lon_col, lat_col, 'type']]
            try: #TODO cannot remember what I wanted to achieve here
                del_column_names = [gdf_yr.filter(
                    regex='^' + c).columns[0] for c in filter_cols]
                gdf_yr.drop(del_column_names, axis=1, inplace=True)
            except: 
                pass

        return gdf_yr

    def get_pfx_data(self, pfx, parent_dir, verbose) -> pd.DataFrame:
        """ Data import for chosen prefix. """
        gdf_pfx = geopandas.GeoDataFrame()
        for yr in self.years:
            gdf_yr = self.get_year_data(pfx, yr, parent_dir, verbose)

            gdf_pfx = pd.concat([gdf_pfx, gdf_yr])
            # Remove case-sensitive distinction in Caribic data 
            if pfx == 'GHG':
                cols = ['SF6', 'CH4', 'CO2', 'N2O']
                for col in cols + ['d_' + c for c in cols]:
                    if col.lower() in gdf_pfx.columns:
                        if not col in gdf_pfx.columns:
                            gdf_pfx[col] = np.nan
                        gdf_pfx[col] = gdf_pfx[col].combine_first(gdf_pfx[col.lower()])
                        gdf_pfx.drop(columns=col.lower(), inplace=True)

            elif pfx == 'MS': 
                MS_cols = ['CO', 'CO2', 'CH4', 'CH4_Err']
                MS_col_dict = {c:'MS_'+c for c in MS_cols}
                gdf_pfx.rename(columns = MS_col_dict, inplace=True)

            # In Integrated data, drop Acetone and Acetonitrile columns
            columns_ac_an = ['int_acetone', 'int_acetonitrile',
                            'int_CARIBIC2_Ac', 'int_CARIBIC2_AN', 
                            'int_CARIBIC2_ACE', 'int_CARIBIC2_ACN']
            
            gdf_pfx.drop(columns=[c for c in gdf_pfx.columns if c in columns_ac_an], inplace=True)

        return gdf_pfx

    def get_data(self, verbose=False, recalculate=False, fname=None, pdir=None) -> dict:
        """ Imports Caribic data in the form of geopandas dataframes.
    
        Returns data dictionary containing dataframes for each file source and
        dictionaries relating column names with Coordinate / Substance instances.

        Parameters:
            recalculate (bool): Data is imported from source instead of using pickled dictionary.
            fname (str): specify File name of data dictionary if default should not be used.
            pdir (str): specify Parent directory of source files if default should not be used.
            verbose (bool): Makes function more talkative.
        """
        self.data = {}  # easiest way of keeping info which file the data comes from
        
        if not recalculate: 
            # Check if data is already available in compact form
            lowres_fname = tools.get_path() + "misc_data\\pickled_dicts\\caribic_data_dict.pkl"
            highres_fname = tools.get_path() + "misc_data\\pickled_dicts\\caribic_10s_data.pkl"
            
            data_dict = {}
            
            if any(pfx in self.pfxs for pfx in ['MS']) \
                and os.path.exists(highres_fname): 
                    with open(highres_fname, 'rb') as f:
                        data_dict.update(dill.load(f))

            if any(pfx in self.pfxs for pfx in ['GHG', 'INT', 'INTtpc']) \
                and os.path.exists(lowres_fname): 
                    with open(lowres_fname, 'rb') as f:
                        data_dict.update(dill.load(f))
            
            # check if loaded data contains given pfxs and vice versa
            if all(pfx in data_dict.keys() for pfx in self.pfxs):
                self.data = {k:data_dict[k] for k in self.pfxs} # choose only pfxs as specified
                self.data = self.sel_year(*self.years).data
                
                for special_item, generator in [('df', '.create_df()'),
                                                ('met_data', '.get_met_data()'),
                                                ('df_sorted', '.get_df_sorted()'),
                                                ('CLAMS', '.get_clams_data(recalculate=True)')]:
                    if special_item in data_dict: 
                        self.data[special_item] = data_dict[special_item]
                        if verbose: 
                            print(f'Loaded \'{special_item}\' from saved data. Call {generator} to generate anew. ')
                    
                return self.data

            elif 'Y' != input(f'Some pfxs not found in saved data, complile data structure from source? [Y/N]').upper():
                return {}

        parent_dir = r'E:\CARIBIC\Caribic2data' if not pdir else pdir
        print('Importing Caribic Data from remote files.')
        for pfx in self.pfxs:  # can include different prefixes here too
            gdf_pfx = self.get_pfx_data(pfx, parent_dir, verbose)
            if gdf_pfx.empty: print("Data extraction unsuccessful. \
                                    Please check your input data"); return
            self.data[pfx] = gdf_pfx
        return self.data

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
        
        # keep = essentials + [
        #     c.col_name for ID in self.pfxs for c in dcts.get_coordinates(ID=ID)
        #     if (c.col_name not in essentials and c in df.columns)]

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
        df = self.met_data.copy() # CLAMS data should be included here already
        
        merge_kwargs = dict(
            how='outer', 
            sort=True,
            left_index=True, 
            right_index=True, 
            )

        for pfx in self.pfxs:
            # df = df.sjoin(self.data[pfx])
            df = pd.merge(df, self.data[pfx],
                          suffixes = [None, f'_{pfx}'],
                          **merge_kwargs)
            
            for c in df.columns:
                if f'{c}_{pfx}' in df.columns:
                    df[c] = df[c].combine_first(df[f'{c}_{pfx}']) # Note: Future warning, watch but should be okay
                    df = df.drop(columns=f'{c}_{pfx}')
        if 'geometry' in df.columns: 
            df = df[df.index.isin(df.geometry.dropna().index)]
            
        self.data['df'] = df
        return df

    def create_substance_df(self, subs, detr=True):
        """ Create dataframe containing all met.+ msmt. data for a substance """
        if detr:
            self.detrend_substance(subs)
        subs_cols = [c for c in self.df
                     if any(i in [s.col_name for s in dcts.get_substances(short_name=subs)]
                            for i in [c, c[5:], c[6:], c[8:]])]

        df = self.df[list(self.met_data.columns) + subs_cols]
        df.dropna(subset=subs_cols, how='all', inplace=True)

        try:  # reordering the columns
            df = df[['Flight number', 'p']
                    + [c for c in df.columns
                       if c not in ['Flight number', 'p', 'geometry']]
                    + ['geometry']]

        except KeyError:
            pass
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
    def INTtpc(self) -> pd.DataFrame:
        if 'INTtpc' in self.data:
            return self.data['INTtpc']
        raise Warning('No INTtpc data available')

    @property
    def INT2(self) -> pd.DataFrame:
        if 'INT2' in self.data:
            return self.data['INT2']
        raise Warning('No INT2 data available')

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

    def interpolate_emac(self, merge_df: bool = True):
        """ Interpolates EMAC data onto .df timestamps

        Parameters:
            merge_df (bool): Merges interpolated EMAC data into main dataframe
        """
        if 'EMAC' not in self.data:
            self.get_emac_data()

        # Interpolate and return EMAC data on Caribic timestamps
        int_emac = tools.interpolate_onto_timestamps(self.data['EMAC'], self.df.index.values)
        self.data['int_EMAC'] = int_emac

        if merge_df:
            df = pd.merge(self.df, int_emac, how='outer', sort=True,
                          left_index=True, right_index=True)
            self.data['df'] = df
        return int_emac
    
        # data = self.df.copy()
        # tps_emac = [i.col_name for i in dcts.get_coordinates(source='EMAC') if i.col_name in self.df.columns] + [
        #     i for i in ['ECHAM5_tm1_at_fl', 'ECHAM5_tpoteq_at_fl', 'ECHAM5_press_at_fl'] if i in self.df.columns]
        # subs_emac = [i.col_name for i in dcts.get_substances(source='EMAC') if i.col_name in self.df.columns]

        # nan_count_i = data[tps_emac[0]].isna().value_counts().loc[True]
        # for c in tps_emac + subs_emac:
        #     if method == 'b':
        #         data[c].interpolate(method='linear', inplace=True, limit=2)
        #     elif method == 'n':
        #         data[c].interpolate(method='nearest', inplace=True, limit=2)
        #     else:
        #         raise KeyError('Please choose either b-linear or n-nearest neighbour interpolation.')
        #     data[c] = data[c].astype(float)
        # nan_count_f = data[tps_emac[0]].isna().value_counts().loc[True]

        # if verbose: print('{} NaNs in EMAC data filled using {} interpolation'.format(
        #     nan_count_i - nan_count_f, 'nearest neighbour' if method == 'n' else 'linear'))

        # self.data['df'] = data
        # self.status['interp_emac'] = True
        # return data

