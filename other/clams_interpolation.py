# -*- coding: utf-8 -*-
"""
@Author: Sophie Bauchinger, IAU
@Date Thu Feb  8 15:22:02 2024

"""
import pandas as pd

from toolpac.readwrite.sql_data_import import client_data_choice # type: ignore

from dataTools.data.Model import Era5ModelData

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
    campaigns = ['PGS', 'SHTR', 'WISE', 'ATOM']
    # available: ['PGS', 'SHTR', 'WISE', 'ATOM']
    
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
                        
    # Get Time data for chosen campaigns
   
    for campaign in campaigns: # !!!

        met_data = Era5ModelData(campaign).df
        
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
        
        resampled_data = get_resampled_reanalysis(met_data, time_df)
        
        resampled_data_dict[campaign] = resampled_data

    fname_dict = {
        'ATOM' : 'atom', 
        'CAR' : 'caribic', 
        'HIPPO' : 'hippo',
        'PHIL' : 'phileas', 
        'PGS' : 'polstracc', 
        'SHTR' : 'southtrac',
        'WISE' : 'wise'
    }
    
    # Save resampled data in .csv files
    for campaign in resampled_data_dict: 
        
        pdir = 'E:\\TPChange\\'
        fname = f'{fname_dict[campaign]}_met_data.csv'
        
        df = resampled_data_dict[campaign]
        df.to_csv(pdir+fname)
