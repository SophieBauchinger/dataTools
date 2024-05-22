# -*- coding: utf-8 -*-
"""
@Author: Sophie Bauchinger, IAU
@Date Thu Aug 10 16:30:17 2023

Test functions on miscellaneous data 
"""
import numpy as np

from toolpac.readwrite import find
from toolpac.readwrite.FFI1001_reader import FFI1001DataReader

#%% Test changes to FFI1001ReadHeader, FFI1001DataReader
# =============================================================================
# fnames = [
#     r'C:/Users/sophie_bauchinger/Documents/GitHub/iau-caribic/misc_data/b47_cryosampler_GCMS_ECD_PIC_w_cat.csv',
#     r'C:/Users/sophie_bauchinger/Documents/GitHub/iau-caribic/misc_data/ACinst_GUF003_202108122119_RA.ict',
#     r'E:\CARIBIC\Caribic2data\Flight148_20060428\GHG_20060428_148_MNL_CAN_V11.txt',
#     ]
# 
# cryo = FFI1001DataReader(fnames[0], sep_header=';', df=True, flatten_vnames=False)
# ac = FFI1001DataReader(fnames[1], sep_header=',', df=True)
# ac_flat = FFI1001DataReader(fnames[1], sep_header=',', df=True, flatten_vnames=False)
# c = FFI1001DataReader(fnames[2], sep_variables=';', df=True)
# 
# print(cryo.df.shape, cryo.df.columns[:4])
# print(ac.df.shape, ac.df.columns[:4])
# print(ac_flat.df.shape, ac_flat.df.columns[:4])
# print(c.df.shape, c.df.columns[:4])
# =============================================================================
pdir = r'C:\Users\sophie_bauchinger\Documents\GitHub\dataTools\dataTools\misc_data\\'
# fname = r'C:\Users\sophie_bauchinger\Documents\GitHub\dataTools\dataTools\misc_data\INTtpc_20150115_492_GRU_MUC_10s_V02.txt'
fnameV02 = pdir + 'INTtpc_20190501_569_PVG_MUC_10s_V02.txt'
fnameV03 = pdir + 'INTtpc_20190501_569_PVG_MUC_10s_V03.txt'
fnameV04 = pdir + 'INTtpc_20190501_569_PVG_MUC_10s_V04(1).txt'
c2 = FFI1001DataReader(fnameV02, sep_variables=';', df=True)
c3 = FFI1001DataReader(fnameV02, sep_variables=';', df=True)
c4 = FFI1001DataReader(fnameV02, sep_variables=';', df=True)

#%%
from data import GlobalData, Caribic
import pandas as pd
import tools
import dictionaries as dcts
import geopandas
from shapely.geometry import Point
import os
import dill
import xarray as xr
"""

"""
def process_caribic(ds): 
    # ds = ds.drop_dims([d for d in ds.dims if 'header_lines' in d])
    variables = [v for v in ds.variables if ds[v].dims == ('time',)]
    
    ds['time'] = pd.to_datetime(ds['time'])
    
    return ds[variables]

class CaribicNetCDF(Caribic):
    """ """
    def __init__(self, years=range(2005, 2020), pfxs=('GHG', 'HCF', 'INT', 'INTtpc')):
        
        self.source = 'Caribic'
        self.pfxs = pfxs

        self.data = {}
        self.get_data()
        
        self.get_df()
        self.years = set(self.df.index.year)
        
    def get_df(self): 
        """ Create dataframe from all pfxs"""
        # df = pd.DataFrame()
        # for pfx in [pfx for pfx in self.pfxs]:
        #     df = df.combine_first(self.data[pfx].copy())
            
        # self.data['df'] = df
        self.data['met_data'] = self.df
        
    def get_data(self, **kwargs):
        """ """
        if not ('ds' in locals()): 
            with xr.open_mfdataset('misc_data/WSM_output_20240110/WSM*.nc', parallel=True, preprocess = process_caribic) as ds: 
                ds = ds

        self.data['ds'] = ds

        if not ('df' in locals()): 
            df = ds.to_dataframe()
        
        self.data['df'] = df
        
        geodata = [Point(lon, lat) for lon, lat in zip(
            df['lon'], df['lat'])]
        df = geopandas.GeoDataFrame(df, geometry=geodata)
        df.drop(columns=['lon', 'lat'], inplace=True)

        # self.data['df'] = df
        
        ghg_vars = ['p', 'alt', 'geometry'] + [c for c in df.columns if c.startswith('ghg_')]
        ghg_df = df[ghg_vars]
        ghg_df.rename(columns = {k:k[4:] for k in [c for c in df.columns if c.startswith('ghg_')]}, 
                      inplace=True)
        
        self.data['GHG'] = ghg_df
        
        int_vars = ['p', 'alt', 'geometry'] + [c for c in df.columns if c.startswith('int_')]
        int_df = df[int_vars]
        int_df.rename(columns = {k:k[4:] for k in [c for c in df.columns if c.startswith('int_')]}, 
                      inplace=True)
        
        self.data['INT'] = int_df
        
        hcf_vars = ['p', 'alt', 'geometry'] + [c for c in df.columns if c.startswith('hcf_')]
        hcf_df = df[hcf_vars]
        hcf_df.rename(columns = {k:k[4:] for k in [c for c in df.columns if c.startswith('hcf_')]}, 
                      inplace=True)

        self.data['HCF'] = hcf_df

        if not ('tpc_df' in locals()): 
            caribic = Caribic()
            tpc_df = caribic.INTtpc
            self.data['INTtpc'] = tpc_df
            del caribic
        else: 
            self.data['INTpc'] = tpc_df

class CaribicINTtpc(GlobalData): 
    """ """
    def __init__(self, years=range(2005, 2020), pfxs=('INTtpc')):
        super().__init__([yr for yr in years if yr > 2004], 5)
        self.source = 'Caribic'
        self.pfxs = pfxs
        self.flights = ()
        # self.get_data(verbose=False, recalculate=False)
    
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
        for current_dir in find.find_dir("Flight*_{}*".format(yr), parent_dir):# [1:]:
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

            rename_dict = tools.rename_columns(f_data.VNAME)
            # set names to their short version
            df_flight.rename(columns=rename_dict, inplace=True)
            df_yr = pd.concat([df_yr, df_flight])

        # Convert longitude and latitude into geometry objects
        try: 
            geodata = [Point(lon, lat) for lon, lat in zip(
                df_yr['lon'], df_yr['lat'])]
        except KeyError: 
            geodata = [Point(lon, lat) for lon, lat in zip(
                df_yr['PosLong'], df_yr['PosLat'])]
        gdf_yr = geopandas.GeoDataFrame(df_yr, geometry=geodata)

        # Drop cols which are saved within datetime, geometry
        if not gdf_yr['geometry'].empty:
            try: 
                filter_cols = ['TimeCRef', 'year', 'month', 'day',
                               'hour', 'min', 'sec', 'lon', 'lat', 'type']
                del_column_names = [gdf_yr.filter(
                    regex='^' + c).columns[0] for c in filter_cols]
                gdf_yr.drop(del_column_names, axis=1, inplace=True)
            except IndexError: 
                pass

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
        parent_dir = r'misc_data'

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
    