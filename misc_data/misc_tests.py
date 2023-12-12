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
fnames = [
    r'C:/Users/sophie_bauchinger/Documents/GitHub/iau-caribic/misc_data/b47_cryosampler_GCMS_ECD_PIC_w_cat.csv',
    r'C:/Users/sophie_bauchinger/Documents/GitHub/iau-caribic/misc_data/ACinst_GUF003_202108122119_RA.ict',
    r'E:\CARIBIC\Caribic2data\Flight148_20060428\GHG_20060428_148_MNL_CAN_V11.txt',
    ]

cryo = FFI1001DataReader(fnames[0], sep_header=';', df=True, flatten_vnames=False)
ac = FFI1001DataReader(fnames[1], sep_header=',', df=True)
ac_flat = FFI1001DataReader(fnames[1], sep_header=',', df=True, flatten_vnames=False)
c = FFI1001DataReader(fnames[2], sep_variables=';', df=True)

print(cryo.df.shape, cryo.df.columns[:4])
print(ac.df.shape, ac.df.columns[:4])
print(ac_flat.df.shape, ac_flat.df.columns[:4])
print(c.df.shape, c.df.columns[:4])

#%%
from data import GlobalData
import pandas as pd
import tools
import dictionaries as dcts
import geopandas
from shapely.geometry import Point
import os
import dill

class CaribicTest(GlobalData): 
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

            col_dict, rename_dict = tools.rename_columns(f_data.VNAME)
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
    