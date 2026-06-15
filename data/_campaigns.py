# -*- coding: utf-8 -*-
""" Aircraft Campaign and Data collection Types.

@Author: Sophie Bauchinger, IAU
@Date: Tue Jun 11 17:35:00 2024
"""
import pandas as pd
import traceback

from dataTools.data._global import GlobalData
from dataTools.data import data_getter

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
        campaign (str): SHTR, WISE, PGS, TACTS, ATOM, HIPPO, PHL, StratoClim, ...
        """
        super().__init__(None, grid_size)

        self.ID = campaign
        self.source = SOURCES.get(campaign, 'NA')
        self.data = {}
        self.get_data(**kwargs)
        self.update_years()
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
            if isinstance(data_dict, pd.DataFrame): 
                data_dict = {'df':data_dict}
            if 'df' not in data_dict: 
                if input('Merged dataframe not found. Recalculate instead? [Y/N]').upper() == 'Y':
                    self.get_data(recalculate=True)
            self.status = updated_status
            self.data = data_dict
            return self.data
        
        campaign_list = [c.lower() for c in data_getter.CAMPAIGN_LIST]
        camp_from_kry = [c.lower() for c in data_getter.CAMP_FROM_KRY]

        # Caribic: data in yearly subfolders (YYYY)
        if self.ID.lower() in ['car', 'caribic']:
            raise NotImplementedError('Please use dataTools.data.Caribic instead, which includes GHG data.')
            ppdir = data_getter.find_TPCfolder(self.ID)
            pdirs = [i for i in ppdir.iterdir() if len(i.name)==4] 
            dataframe = pd.concat([data_getter.get_TPChange_gdf(
                pdir, drop_variables=['CARIBIC2_LocalTime']) for pdir in pdirs])
        # ATOM: drop_variables 
        elif self.ID.lower()=='atom': 
            pdir = data_getter.find_TPCfolder(self.ID)
            dataframe = data_getter.get_TPChange_gdf(
                pdir, drop_variables=['ATom_UTC_Start', 'ATom_UTC_Stop', 'ATom_End_LAS'])
        # Campaigns: Get TPChange gdf
        elif self.ID.lower() in campaign_list + ['shtr', 'pgs', 'phl']:
            pdir = data_getter.find_TPCfolder(self.ID)
            dataframe = data_getter.get_TPChange_gdf(pdir)
        elif self.ID.lower() in camp_from_kry:
            fpaths = data_getter.find_TPCfolder(self.ID)
            dataframe = pd.concat([data_getter.get_TPChange_gdf(fp) for fp in fpaths])
        else: 
            raise Warning(f"Could not find {self.ID} in available campaigns")
            
        self.data['df'] = dataframe
        return self.data

    def create_df(self, recalculate=False, inplace=True): 
        """ Apply `calc_coordinates` to self.df """
        if not 'df' in self.data: 
            self.get_data()
        df_exp = data_getter.calc_coordinates(self.data['df'], 
                                              recalculate=recalculate, 
                                              inplace=inplace)
        if inplace and (len(self.df.columns)<len(df_exp.columns)): 
            self.data['df'] = df_exp
        return df_exp

    @property
    def met_data(self) -> pd.DataFrame:
        """ Meteorological Parameters along the flight track. """
        if 'met_data' in self.data:
            return self.data['met_data']
        return self.df[[c.col_name for c in self.coordinates if not c.name.startswith('geo')]+['geometry']]
