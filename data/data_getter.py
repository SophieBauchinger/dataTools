# -*- coding: utf-8 -*-
""" Data import for different formats

@Author: Sophie Bauchinger, IAU
@Date: Tue May 27 11:36:54 2025

"""
import pandas as pd
import xarray as xr

from toolpac.conv.times import datetime_to_fractionalyear as dt_to_fy  # type: ignore
from toolpac.readwrite import find

from dataTools import tools


def import_era5_data(
        ID:str,
        fnames:str=None, 
        version:int=5) -> pd.DataFrame:
    """ Creates dataframe for ERA5/CLaMS data from netcdf files. """
    if fnames is None:
        met_pdir = r'E:/TPChange/'
        campaign_dir_dict = { # campaign_pdir, version
            'CAR'  : 'CaribicTPChange',
            'SHTR' : 'SouthtracTPChange',
            'WISE' : 'WiseTPChange',
            'ATOM' : 'AtomTPChange',
            'HIPPO': 'HippoTPChange',
            'PGS'  : 'PolstraccTPChange',
            'PHL'  : 'PhileasTPChange',
            'HPS'  : 'HPS_o3_sonde_TPChange' 
            }
        campaign_pdir = met_pdir+campaign_dir_dict[ID]
        
        fnames = campaign_pdir + "/*.nc"
        if ID in ['CAR', 'HPS']: # organised by year!
            fnames = campaign_pdir + "/2*/*.nc"
   
    drop_variables = {'CAR' : ['CARIBIC2_LocalTime'],
                      'ATOM' : ['ATom_UTC_Start', 'ATom_UTC_Stop', 'ATom_End_LAS']}

    # extract data, each file goes through preprocess first to filter variables & convert units
    with xr.open_mfdataset(fnames, 
                            preprocess = tools.process_TPC if not version==2 else tools.process_TPC_V02,
                            drop_variables = drop_variables.get(ID),
                            ) as ds:
        ds = ds
    met_df = ds.to_dataframe()
    return met_df

def process_CORE(ds):
    ds.time.attrs["unit"] = "seconds since 2000-01-01"
    return xr.decode_cf(ds)

def import_CORE_data(fnames=None):
    """ Import monthly O3 and CO with interpolated ERA5 met. parameters. """
    if fnames is None: 
        fnames = r'E:/TPChange/iagosCoreTPChange' + '/*.nc'
    with xr.open_mfdataset(fnames, 
                           preprocess = process_CORE, 
                           drop_variables = ['trop2_z'],
                           ) as ds:
        ds = ds
    met_df = ds.to_dataframe()
    return met_df


# "E:\TPChange\iagosCoreTPChange\mozaic_2008_3_o3_co.nc"