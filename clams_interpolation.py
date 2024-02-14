# -*- coding: utf-8 -*-
"""
@Author: Sophie Bauchinger, IAU
@Date Thu Feb  8 15:22:02 2024

"""
import os
import pandas as pd
import xarray as xr#
import datetime as dt

from toolpac.conv.times import secofday_to_datetime
from toolpac.readwrite.sql_data_import import client_data_choice

def process_clams(ds): 
    variables = ['ERA5_TEMP',
                 'ERA5_PRESS',
                 'ERA5_THETA',
                 'ERA5_GPH',
                 'ERA5_PV',
                 'ERA5_EQLAT',

                 'ERA5_TROP1_THETA',
                 'ERA5_TROP1_PRESS',
                 'ERA5_TROP1_Z',
                 'ERA5_PRESS_2_0_Main',
                 'ERA5_PRESS_3_5_Main',
                 'ERA5_THETA_2_0_Main',
                 'ERA5_THETA_3_5_Main',
                 'ERA5_GPH_2_0_Main',
                 'ERA5_GPH_3_5_Main',]

    if 'ATom_UTC_Start' in ds.variables: 
        variables.append('ATom_UTC_Start')

    return ds[variables]

def process_atom_clams(ds):
    """ Additional time values for ATom as otherwise the function breaks """    
    ds = process_clams(ds)
    
    # find flight date from file name
    filepath = ds['ATom_UTC_Start'].encoding['source']
    fname = os.path.basename(filepath)
    date_string = fname.split('_')[1]
    date = dt.datetime(year = int(date_string[:4]), 
                        month = int(date_string[4:6]), 
                        day = int(date_string[-2:]))
    
    # generate datetimes for each timestamp
    datetimes = [secofday_to_datetime(date, secofday + 5) for secofday in ds['ATom_UTC_Start'].values]
    ds = ds.assign(Time = datetimes) 
    
    ds = ds.drop_vars('ATom_UTC_Start')
    
    return ds

def get_clams_data(campaign: str):
    """ Creates dataframe for CLaMS data from netcdf files. """#

    campaign_dir_dict = {
        'SHTR': 'SouthtracTPChange',
        'WISE': 'WiseTPChange',
        'ATOM': 'AtomTPChange',
        'HIPPO': 'HippoTPChange',
        'PGS' : 'PolstraccTPChange',
    }

    fnames = r'E:/TPChange/' + campaign_dir_dict[campaign] + "/*.nc"

    decode_times = True
    preprocess = process_clams

    if campaign == 'ATOM':
        decode_times = False
        preprocess = process_atom_clams

    # extract data, each file goes through preprocess first to filter variables & convert units
    with xr.open_mfdataset(fnames, decode_times=decode_times, preprocess=preprocess) as ds:
        met_ds = ds

    met_df = met_ds.to_dataframe()

    return met_df

def get_resampled_reanalysis(met_data: pd.DataFrame, msmt_times: pd.DataFrame):
    """ Import and interpolate / resample CLaMS data for aircraft campaigns. """
    time = msmt_times.values
    new_indices = [i for i in time if i not in met_data.index]

    # add measurement timestamps to met_data
    expanded_df = pd.concat([met_data, pd.Series(index=new_indices, dtype='object')])
    expanded_df.drop(columns=[0], inplace=True)
    expanded_df.sort_index(inplace=True) # need to sort by time otherwise interpolation limit cannot be used
    expanded_df.interpolate(method='time', inplace=True, limit=2)  # , limit=500)

    met_data_on_msmt_times = expanded_df.loc[time]

    return met_data_on_msmt_times

if __name__ == '__main__': 
    
    # define parameters for sql import 
    log_in = {'host': '141.2.225.99', 
              'user': 'Datenuser', 
              'pwd': 'AG-Engel1!'}
    
    special_dct = {
        'SHTR': 'ST all',
        'WISE': 'WISE all',
        'PGS': 'PGS all',
        'ATOM': None,
        'TACTS': None,
    }
    
    default_flights = {
        'ATOM':
            [f'AT1_{i}' for i in range(1, 12)] + [
                f'AT2_{i}' for i in range(1, 12)] + [
                f'AT3_{i}' for i in range(1, 14)] + [
                f'AT4_{i}' for i in range(1, 14)],

        'TACTS':
            [f'T{i}' for i in range(1, 7)],
    }
    
    resampled_data_dict = {}
                        
    for campaign in ['PGS']: 

        met_data = get_clams_data(campaign)
        
        time_data = client_data_choice(
            log_in = log_in,
            campaign = campaign,
            special = special_dct[campaign],
            time = True,
            flights= default_flights.get(campaign),
        )
        
        time_df = time_data._data['DATETIME']
        time_df.index += 1  # measurement_id
        time_df.index.name = 'measurement_id'
        
        # msmt_times = pd.DataFrame(time_df.index, index=time_df)
        
        resampled_data = get_resampled_reanalysis(met_data, time_df)
        
        resampled_data_dict[campaign] = resampled_data
        
        

#%%
import mysql.connector

def get_measurement_times(campaign):

    log_in = {'host': '141.2.225.99', 
              'user': 'Datenuser', 
              'pwd': 'AG-Engel1!'}

    connection = mysql.connector.connect(
        host=log_in['host'], user=log_in['user'], passwd=log_in['pwd']
    )
    
    cursor = connection.cursor()
    
    # get measurement times as seconds from start of day of flight 
    cursor.execute(f'SELECT * FROM {campaign}.times;')
    results = cursor.fetchall()
    columns = cursor.column_names
    times = pd.DataFrame(results, columns=columns)
    #  ['measurement_id', 'flight_id', 'flight_name', 'measurement_time']
    
    # get flight dates
    cursor.execute(f"SELECT flight, flight_date, flight_start, flight_end, flight_duration FROM {campaign}.flight_dates;")
    results = cursor.fetchall()
    columns = cursor.column_names
    flight_dates = pd.DataFrame(results, columns=columns)
    # ['flight', 'flight_date', 'flight_start', 'flight_end', 'flight_duration']
    
    
    # add associated flight date (from flight_dates) to times
    
    # add 1 to the flight date if measurement_time is larger than 86400 (when a flight took place across UTC midnight)
    # remove 86400 seconds from those timestamps
    
    # use the timestamp in seconds and the flight date to create a datetime object

# From Markus' code (but not currently in working order)
# =============================================================================
#     times['date'] =[flight_dates[flight_dates["flight"] == times["flight_name"][i]]["flight_date"].values[0] for i in range(len(times))]#
#     
#     for i in range(len(times)):
#         if times["measurement_time"][i] >= 86400:
#             times["measurement_time"].iat[i] = times["measurement_time"][i] - 86400
#             times["date"].iat[i] = str(int(times["date"][i]) + 1)
# 
#     times["time"] = [str(dt.timedelta(seconds=float(times["measurement_time"][i]))) for i in range(len(times))]
#     
#     times["datetime"] = [pd.to_datetime(times["date"][i] + " " + times["time"][i], format="%Y%m%d %H:%M:%S") for i in range(len(times))]
#     
# =============================================================================
    
    # print(times, flight_dates)
    return (times, flight_dates)