# -*- coding: utf-8 -*-
""" EMAC data class definition

@Author: Sophie Bauchinger, IAU
@Date: Tue Jun 11 17:35:00 2024
"""
import dill
import geopandas
import keyring
import numpy as np
import os
import pandas as pd
from shapely.geometry import Point
import xarray as xr

from toolpac.readwrite.sql_data_import import client_data_choice # type: ignore
from toolpac.readwrite.FFI1001_reader import FFI1001DataReader # type: ignore

from dataTools.data._global import GlobalData
import dataTools.dictionaries as dcts
from dataTools import tools

# HALO and ATOM campaigns from SQL Database
class CampaignData(GlobalData):
    """ Stores data for a single HALO / ATom campaign.
    
    Class attributes:
        years: arr
        source: str
        ID: str - campaign (STH, ATOM, PGS, TACTS, WISE)
        campaign: str
        df: Pandas GeoDataFrame
    """

    def __init__(self, campaign, grid_size=5, tps_dict={}, **kwargs):
        """ Initialise HALO_campaign object. """
        years = dcts.years_per_campaign(campaign)
        super().__init__(years, grid_size)

        self.years = years

        source_dict = {
            'SHTR' : 'HALO',
            'WISE' : 'HALO',
            'PGS'  : 'HALO',
            'TACTS': 'HALO',
            'ATOM' : 'ATOM',
            'HIPPO': 'HIAPER', 
            'PHL'  : 'HALO'}
        self.source = source_dict[campaign]

        self.ID = campaign
        self.instruments = list(dcts.get_instruments(self.ID))
        self.data = {}
        self.get_data(**kwargs)

        if not 'df' in self.data: 
            self.create_df()
        self.calc_coordinates()

        # if 'flight_id' in self.df:
        #     self.flights = set(self.data['df']['flight_id'].values)
        # if 'Flight number' in self.df:
        #     self.flights = set(self.data['df']['Flight number'].values)

        self.years = list(set(self.data['df'].index.year))
        self.set_tps(**tps_dict)

    def __repr__(self):
        """ Show instance details representing dataset. """
        if 'instruments' in self.__dict__:
            return (
                f"""{self.__class__}
    instruments: {self.instruments}
    status: {self.status}""")

        else:
            return (
                f"""{self.__class__}
    status: {self.status}""")

    @property
    def _log_in(self):
        user = keyring.get_password('IAU_SQL', 'username_key')
        log_in = {"host": '141.2.225.99',
                  "user": user,
                  "pwd" : keyring.get_password('IAU_SQL', user)}
        return log_in

    @property
    def _special(self):
        """ Get special kwarg for campaign data import. """
        special_dct = {
            'SHTR' : 'ST all',
            'WISE' : 'WISE all',
            'PGS'  : 'PGS all',
            'ATOM' : None,
            'TACTS': None,
            'HIPPO': None,
            }
        return special_dct.get(self.ID)

    @property
    def _default_flights(self):
        """ Get default flight kwarg for campaign data import. """
        all_flights_dct = {
            'ATOM' :
                [f'AT1_{i}' for i in range(1, 12)] + [
                    f'AT2_{i}' for i in range(1, 12)] + [
                    f'AT3_{i}' for i in range(1, 14)] + [
                    f'AT4_{i}' for i in range(1, 14)],

            'TACTS':
                [f'T{i}' for i in range(1, 7)],

            'HIPPO':
                [f'H1_{i}' for i in range(2, 12)] + [
                    f'H2_{i}' for i in range(-1, 12)] + [
                    f'H3_{i}' for i in range(1, 12)] + [
                    f'H4_{i}' for i in range(0, 13)] + [
                    f'H5_{i}' for i in range(1, 15)],
            }
        return all_flights_dct.get(self.ID)

    def get_data(self, **kwargs) -> dict:
        """ Import campaign data per instrument from SQL Database.
    
        campaign dictionary:
            ST all - SOUTHTRAC
            PGS all - PGS
            WISE all - WISE
            TACTS / CARIBIC
    
        """
        fname = f'{self.ID.lower()}_data_dict.pkl' if not kwargs.get('fname') else kwargs.get('fname')
        dict_path = tools.get_path() + 'misc_data\\pickled_dicts\\' + fname
        if not kwargs.get('recalculate') and os.path.exists(dict_path):
            with open(dict_path, 'rb') as f:
                self.data = dill.load(f)

            if not 'df' in self.data:
                if input('Merged dataframe not found. Recalculate? [Y/N]').upper() == 'Y':
                    self.get_data(recalculate=True)

        else:
            if self.ID == 'PHL': 
                print('Importing PHILEAS data from Flights 07 and 19.')
                data_dict = get_phileas_data(time_res = kwargs.get('time_res', '10s'))
                self.data = data_dict
                return self.data
            
            print('Importing Campaign Data from SQL Database.')

            time_data = client_data_choice(
                log_in=self._log_in,
                campaign=self.ID,
                special=self._special,
                time=True,
                flights=self._default_flights,
                )

            if kwargs.get('verbose'):
                print('Imported time data')
            time_df = time_data._data['DATETIME']
            time_df.index += 1  # measurement_id
            time_df.index.name = 'measurement_id'

            self.data['time'] = time_df

            for instr in self.instruments:
                if kwargs.get('verbose'):
                    print('Importing data for ', instr)
                self.get_instrument_data(instr, time=time_df, **kwargs)

            # self.merge_instr_data()  # Merge data from all instruments, create .df
            self.get_met_data()  # Get meteorological data, create .met_data

        return self.data

    def get_instrument_data(self, instr: str, time: pd.Series, **kwargs) -> pd.DataFrame:
        """ Import data for given instrument. """
        variables = list(dcts.get_variables(self.ID, instr))
        if kwargs.get('verbose'):
            print('  ', variables)
        variables += ['measurement_id', 'flight_id']

        print(instr, variables)

        data = client_data_choice(log_in=self._log_in,
                                  instrument=instr,
                                  campaign=self.ID,
                                  substances=variables,
                                  flights=self._default_flights,
                                  special=self._special,
                                  )

        df = data._data['data']
        df.index = time

        for col in df.columns:
            if kwargs.get('verbose'):
                print(f'Renaming: {col} -> {dcts.harmonise_variables(instr, col)}')
            df[dcts.harmonise_variables(instr, col)] = df.pop(col)

        data_key = dcts.harmonise_instruments(instr)
        if data_key in self.data: 
            joined_data = self.data[data_key].join(df, rsuffix='_r')
            joined_data.drop(columns=[c for c in joined_data.columns if c.endswith('_r')], inplace=True)
            self.data[data_key] = joined_data
        else: 
            self.data[data_key] = df
        return df

    def merge_instr_data(self) -> pd.DataFrame:
        """ Combine available data into single dataframe. """

        time = self.data['time']
        measurement_id = time.index

        data = pd.DataFrame(measurement_id, index=time)

        # met_data = self.get_met_data()
        # data.join(met_data, rsuffix='_dupe')

        dataframes = [df for k, df in self.data.items() if (
            isinstance(df, pd.DataFrame) and k!='df')]
        # combine data
        for df in dataframes:
            data = data.join(df, rsuffix='_dupe')
            data = data.drop(columns=[c for c in data.columns if 'dupe' in c])

        if 'flight_id' in data.columns:
            data['Flight number'] = data.pop('flight_id')

        # Create GeoDataFrame using available geodata
        lon_cols = [c for c in data.columns if 'LON' in c]
        lat_cols = [c for c in data.columns if 'LAT' in c]

        if not (len(lon_cols) > 0 and len(lat_cols) > 0):
            self.data['df'] = data
            return data

        geodata = [Point(lon, lat) for lon, lat in zip(
            data[lon_cols[0]], data[lat_cols[0]])]
        gdf = geopandas.GeoDataFrame(data, geometry=geodata)
        self.data['df'] = gdf
        return gdf

    def get_met_data(self) -> pd.DataFrame:
        """ Creates dataframe for CLaMS data from netcdf files. """
        if self.ID in ['SHTR', 'WISE', 'ATOM', 'HIPPO', 'PGS']:
            return self.get_clams_data()

        elif self.ID in ['TACTS'] and 'df' in self.data.keys():
            met_cols = [c for c in self.df.columns if c in [
                c.col_name for c in dcts.get_coordinates()
                if not c.col_name == 'geometry']]
            met_df = self.df[met_cols]

        else:
            raise NotImplementedError(f'Cannot create met_data for {self.ID}')

        self.data['met_data'] = met_df
        return met_df

    def create_df(self) -> pd.DataFrame:
        """ Combine available data into single dataframe, interpolate ERA5 data. """

        df = self.df if 'df' in self.data else self.merge_instr_data()
        met_data = self.data['met_data'] if 'met_data' in self.data else self.get_met_data()
        times = self.data['time'].values if 'time' in self.data else df.index

        try:
            interpolated_met_data = tools.interpolate_onto_timestamps(met_data, times, 'int_')
        except:
            print('Interpolation unsuccessful. ')
            interpolated_met_data = met_data

        # Drop non-int columns
        int_cols = [c[4:] for c in interpolated_met_data.columns if (
            c.startswith('int_') and c[4:] in df.columns)]
        df.drop(columns = int_cols, inplace=True)
        
        # met_data = self.get_met_data()
        df = df.join(interpolated_met_data, rsuffix='_dupe')
        df.drop(columns = [c for c in df.columns if '_dupe' in c], inplace=True)

        self.data['df'] = df
        return df

    @property
    def df(self) -> pd.DataFrame:
        """ Combined dataframe with measurement and modelled data. """
        if 'df' in self.data:
            return self.data['df']
        return self.create_df()

    @property
    def met_data(self) -> pd.DataFrame:
        """ Meteorological Parameters along the flight track. """
        if 'met_data' in self.data:
            return self.data['met_data']
        return self.get_met_data()

def get_phileas_data(time_res = '10s'): 
    """ Temporary function for creating merge files for the PHILEAS campaign. """  
    # GHOST_ECD
    fname = r"C:\Users\sophie_bauchinger\Documents\GitHub\dataTools\dataTools\misc_data\PHILEAS\PHILEAS_F07_Frankfurt_20230821_HALO_GHOST_ECD_v1.csv"
    ghost_7 = FFI1001DataReader(fname, df=True, xtype='secofday').df
    ghost_7['Flight number'] = 7
    fname = r"C:\Users\sophie_bauchinger\Documents\GitHub\dataTools\dataTools\misc_data\PHILEAS\PHILEAS_F19_Solingen_20230922_HALO_GHOST_ECD_v1.csv"
    ghost_19 = FFI1001DataReader(fname, df=True, xtype='secofday').df
    ghost_19['Flight number'] = 19
    ghost = pd.concat([ghost_7, ghost_19])
    ghost = ghost.drop(columns = ['Mean', 'Time_Start', 'Time_End'])
    ghost.index = ghost.index.round('s')
    ghost.dropna(how = 'all', inplace = True)
    
    # FAIRO
    fname = r"C:\Users\sophie_bauchinger\Documents\GitHub\dataTools\dataTools\misc_data\PHILEAS\PHILEAS_F07a_2023-08-21_HALO_FAIRO_O3_V02.ames"
    fairo_7a = FFI1001DataReader(fname, df=True, xtype='secofday').df
    fairo_7a['Flight number'] = 7
    fname = r"C:\Users\sophie_bauchinger\Documents\GitHub\dataTools\dataTools\misc_data\PHILEAS\PHILEAS_F07b_2023-08-21_HALO_FAIRO_O3_V02.ames"
    fairo_7b = FFI1001DataReader(fname, df=True, xtype='secofday').df
    fairo_7b['Flight number'] = 7
    fname = r"C:\Users\sophie_bauchinger\Documents\GitHub\dataTools\dataTools\misc_data\PHILEAS\PHILEAS_F19_2023-09-22_HALO_FAIRO_O3_V02.ames"
    fairo_19 = FFI1001DataReader(fname, df=True, xtype='secofday').df
    fairo_19['Flight number'] = 19
    fairo = pd.concat([fairo_7a, fairo_7b, fairo_19])
    fairo.drop(columns = ['Mid_UTC;'], inplace = True)
    fairo.index = fairo.index.round('s')
    fairo.rename(columns = {c : c.split(';')[0] for c in fairo.columns}, inplace = True)
    
    # UMAQS
    fname = r"C:\Users\sophie_bauchinger\Documents\GitHub\dataTools\dataTools\misc_data\PHILEAS\PHILEAS_F07_Frankfurt_20230821_HALO_UMAQS_v1.ames"
    umaqs_7 = FFI1001DataReader(fname, df=True, xtype='secofday').df
    umaqs_7['Flight number'] = 7
    fname = r"C:\Users\sophie_bauchinger\Documents\GitHub\dataTools\dataTools\misc_data\PHILEAS\PHILEAS_F19_Solingen_20230922a_HALO_UMAQS_v1.ames"
    umaqs_19 = FFI1001DataReader(fname, df=True, xtype='secofday').df
    umaqs_19['Flight number'] = 19
    umaqs = pd.concat([umaqs_7, umaqs_19])
    umaqs.index = umaqs.index.round('s')
    umaqs.drop(columns = ['UTC_seconds;'], inplace = True)
    
    # ERA5 Data
    def process_ERA5_PHILEAS(ds): 
        """ Preprocess datasets for ERA5 / CLaMS renalayis data for PHILEAS - include BAHAMAS. """
        def flatten_TPdims(ds):
            TP_vars = [v for v in ds.variables if any(d.endswith('TP') for d in ds[v].dims)]
            TP_qualifier_dict = {0 : '_Main', 1 : '_Second', 2 : '_Third'}
            for variable in TP_vars: 
                # get secondary dimension for the current multi-dimensional variable
                [TP_dim] = [d for d in ds[variable].dims if d.endswith('TP')] # should only be a single one!
                for TP_value in ds[variable][TP_dim].values: 
                    ds[variable + TP_qualifier_dict[TP_value]] = ds[variable].isel({TP_dim : TP_value})
                ds = ds.drop_vars(variable)
            return ds
        # Flatten variables that have multiple tropoause dimensions (thermTP, dynTP)
        ds = flatten_TPdims(ds)
        vars = set(list(tools.ERA5_variables()) + ['Lat', 'Lon', 'PAlt', 'Pres', 'Theta']) # include Bahamas MET data
        return ds[[v for v in vars if v in ds.variables]]
    
    era5_7a = r"C:\Users\sophie_bauchinger\Documents\GitHub\dataTools\dataTools\misc_data\PHILEAS\PHILEAS_20230821_F07a_TPC_V04.nc"
    era5_7b = r"C:\Users\sophie_bauchinger\Documents\GitHub\dataTools\dataTools\misc_data\PHILEAS\PHILEAS_20230821_F07b_TPC_V04.nc"
    era5_19 = r"C:\Users\sophie_bauchinger\Documents\GitHub\dataTools\dataTools\misc_data\PHILEAS\PHILEAS_20230922_F19_TPC_V04.nc"
    with xr.open_mfdataset([era5_7a, era5_7b, era5_19], preprocess = process_ERA5_PHILEAS) as ds: 
        era5_ds = ds
    era5 = era5_ds.to_dataframe()
    era5_rename_vars = {
        'Lat' : 'BAHAMAS_LAT',
        'Lon' : 'BAHAMAS_LON',
        'PAlt' : 'BAHAMAS_ALT',
        'Pres' : 'BAHAMAS_PSTAT', # NB_PSIA
        'Theta' : 'BAHAMAS_POT', # source Bahamas?
        }
    # era5_rename_tps = {c:c[:-5] for c in era5.columns if '_Main' in c}
    # era5_rename = dict(era5_rename_vars, **era5_rename_tps)
    era5.rename(columns = era5_rename_vars, inplace=True)

    # Interpolate onto 
    times = era5.resample(time_res).mean().index
    umaqs_resampled = tools.interpolate_onto_timestamps(umaqs, times)
    ghost_resampled = tools.interpolate_onto_timestamps(ghost, times)
    fairo_resampled = tools.interpolate_onto_timestamps(fairo, times)
    era5_resampled = tools.interpolate_onto_timestamps(era5, times)
    
    for instr, df in {'UMAQS' : umaqs_resampled, 
                      'GHOST_ECD' : ghost_resampled, 
                      'FAIRO' : fairo_resampled}.items():
        for col in df.columns:
            # if kwargs.get('verbose'):
            #     print(f'Renaming: {col} -> {dcts.harmonise_variables(instr, col)}')
            df[dcts.harmonise_variables(instr, col)] = df.pop(col)
   
    msmt_data = pd.concat([umaqs_resampled, ghost_resampled, fairo_resampled], axis = 'columns').dropna(how = 'all')
    era5_resampled.drop([i for i in times if i not in msmt_data.index], inplace = True)
    df_resampled = pd.concat([msmt_data, era5_resampled], axis = 'columns') # this results in duplicate values (most likely)

    geodata = [Point(lon, lat) for lon, lat in zip(
        df_resampled['BAHAMAS_LON'], df_resampled['BAHAMAS_LAT'])]
    df = geopandas.GeoDataFrame(df_resampled, geometry=geodata)
    df = df[[c for c in df.columns if c not in ['C2H6', 'CFC12']]][df['BAHAMAS_LAT'].notna()]
    # df.rename(columns = {c:c[4:] for c in df.columns if 'BAHAMAS' in c}, inplace = True)
    
    # Quite possibly the ugliest way of dealing with duplicate flight number columns but behold it works: 
    def same_merge(x): return np.nanmin(x)
    flight_nr = df['Flight number'].T.groupby(level=0).apply(lambda x: x.apply(same_merge,)).T
    df.drop(columns = ['Flight number'], inplace = True)
    df['Flight number'] = flight_nr

    # Create output     
    data_dictionary = {
        'GHOST' : ghost_resampled, 
        'UMAQS' : umaqs_resampled, 
        'FAIRO' : fairo_resampled, 
        'met_data' : era5_resampled, 
        'df' : df,
    }
    return data_dictionary
