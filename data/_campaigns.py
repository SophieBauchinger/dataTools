# -*- coding: utf-8 -*-
""" Aircraft Campaign and Data collection Types.

@Author: Sophie Bauchinger, IAU
@Date: Tue Jun 11 17:35:00 2024
"""
import dill
import geopandas
import keyring
import pandas as pd
from pathlib import Path
from shapely.geometry import Point
import traceback

from toolpac.readwrite.sql_data_import import client_data_choice # type: ignore

from dataTools.data._global import GlobalData
import dataTools.dictionaries as dcts
from dataTools import tools
from dataTools.data import data_getter

TPCHANGE_WITH_OBS = { # Observations and model within TPChange data set
    'CAR'  : 'CaribicTPChange',
    'SHTR' : 'SouthtracTPChange',
    'WISE' : 'WiseTPChange',
    'ATOM' : 'AtomTPChange',
    'HIPPO': 'HippoTPChange',
    'PGS'  : 'PolstraccTPChange',
    'PHL'  : 'PhileasTPChange',
    }

TPCHANGE_MODEL = { # Model data only
    'Attrex' : 'MODEL_DATA/ATTREX_TPChange', # N2O available
    'Envisat' : 'MODEL_DATA/ENVISAT_TPChange', # N2O available
    'Euplex' : 'MODEL_DATA/EUPLEX_TPChange', # N2O available
    'SPURT' : 'MODEL_DATA/SPURT_TPChange', # ? Obs data ?
    'TACTS' : 'MODEL_DATA/TACTS_TPChange', # ! Obs data via SQL DB 
    'TC4' : 'MODEL_DATA/TC4_TPChange', # N2O available
    }

SOURCES = { # 'Source' per Campaign
    'TACTS': 'HALO',
    'WISE' : 'HALO',
    'PGS'  : 'HALO',
    'SHTR' : 'HALO',
    'PHL'  : 'HALO', 
    'ATOM' : 'ATOM',
    'HIPPO': 'HIAPER', 
    'TC4' : 'Aircraft', 
    'ATTREX' : 'GlobalHawk', 
    'ENVISAT' : 'Balloon', 
    'EUPLEX' : 'Falcon', 
    'SPURT' : 'LearJet ',
    'StratoClim' : 'Geophysica',
    }

#%% Any data type
class DataCollection(GlobalData):
    """ Combined data sets incl. Sonde and Aircraft data. 
    Needs to contain geoinformation for each point. """ 
    def __init__(self, dataframe:pd.DataFrame, ID:str, calc_coords=True, **kwargs): 
        """ Initialise data collection object with a datetime-indexed dataframe. """
        if not 'Datetime' in str(type(dataframe.index)): 
            raise Warning("Given dataframe needs to be datetime-indexed.")
        years = set(dataframe.index.year)
        super().__init__(years, **kwargs)
        self.source = self.ID = ID

        if calc_coords:
            try: 
                dataframe = data_getter.calc_coordinates(dataframe, recalculate=True)
            except Exception: 
                traceback.print_exc()
        self.data['df'] = dataframe
        self.df.sort_index()
        
        # Optional parameters
        self.data.update(**kwargs.get('data_dict', {}))
        self.status.update(**kwargs.get('status', {}))

    def __repr__(self):
        return f"""{self.__class__}
    years: {self.years}
    data: {list(self.data.keys())}
    status: {self.status}"""

    def dropna(self, cols=None):
        """ Remove NaN values from the dataframe. """
        self.df.dropna(subset = cols, how='all', inplace=True)
        
#%% Aircraft campaigns (HALO, ATOM, HIAPER)
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
        """ Initialise HALO_campaign object. 
    
        Parameters 
        ---
        campaign (str): SHTR, WISE, PGS, TACTS, ATOM, HIPPO, PHL, StratoClim, 
        """
        years = dcts.years_per_campaign(campaign)
        super().__init__(years, grid_size)

        self.years = years
        self.source = SOURCES[campaign]
        self.ID = campaign
        self.instruments = list(dcts.get_instruments(self.ID))
        self.data = {}
        self.get_data(**kwargs)

        if not 'df' in self.data: 
            self.create_df()
        try: 
            self.data["df"] = data_getter.calc_coordinates(
                self.data["df"], recalculate=kwargs.get('recalculate', True))
        except Exception: 
            traceback.print_exc()

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

    def get_data(self, recalculate=False, **kwargs): 
        """ Either import from TPChange netcdf or join together observational and model data. """
        # Check for importing from pickled DATA dictionary
        if not recalculate: 
            data_dict, updated_status,_ = data_getter.load_DATA_dict(
                self.ID, self.status, fname=kwargs.get("fname", None), pdir=kwargs.get("pdir", None))

            if 'df' not in data_dict: 
                if input('Merged dataframe not found. Recalculate? [Y/N]').upper() == 'Y':
                    self.get_data(recalculate=True)
            self.status = updated_status
            self.data = data_dict
            return self.data

        # Recalculate from TPChange obs + model files
        if self.ID in TPCHANGE_WITH_OBS.keys(): 
            print('Importing data from TPChange interpolation files.')
            fnames = Path('E:/TPChange') / TPCHANGE_WITH_OBS[self.ID] 
            dataframe = data_getter.get_TPChange_gdf(fnames)
            self.data['df'] = dataframe
            return self.data

        # Recalculate from TPChange model files (+ MISSING OBS DATA)
        elif self.ID in TPCHANGE_MODEL.keys(): 
            print('Importing meteo data from TPChange interpolation files.')
            fnames = 'E:/TPChange/' + Path('E:/TPChange') / TPCHANGE_MODEL[self.ID] 
            dataframe = data_getter.get_TPChange_gdf(fnames)

            self.data['met_data'] = dataframe

            print('No observational data available in interpolated files. ')
            return self.data
        else: 
            raise Warning(f"Could neither find a stored DATA_dict nor find TPChange files for ID {self.ID}. ")

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

#%% SQL Database Import - Aircraft Campaigns
class CampaignSQLData(CampaignData):
    """ Class for campaigns where observational data is available only through the SQL database. """

    def __init__(self, campaign, grid_size=5, tps_dict={}, **kwargs):
        """ Initialise campaign object. """
        super().__init__(campaign, grid_size=grid_size, tps_dict=tps_dict, **kwargs)
        self.get_obs_data()
        self.create_df()

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

    def get_obs_data(self, **kwargs) -> dict:
        """ Import campaign data per instrument from SQL Database.
    
        campaign dictionary:
            ST all - SOUTHTRAC
            PGS all - PGS
            WISE all - WISE
            TACTS / CARIBIC
    
        """
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

        return self.data

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

