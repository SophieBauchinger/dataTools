# -*- coding: utf-8 -*-
""" Mixin for importing model data (CLaMS, ERA5 into GobalData instances)

@Author: Sophie Bauchinger, IAU
@Date: Wed Jun 12 13:16:00 2024
"""

from abc import abstractmethod
import copy
from metpy import calc
from metpy.units import units
import os
import pandas as pd
import xarray as xr

import dataTools.dictionaries as dcts
from dataTools import tools

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
        if self.ID in ['CAR', 'ATOM', 'HIPPO', 'SHTR', 'PGS', 'WISE']: 
            return self.get_clams_data()
        else: 
            raise NotImplementedError(f'Subclass of GlobalData ( - {self.ID}): need to specifically implement .get_met_data()')

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
    