# -*- coding: utf-8 -*-
"""
@Author: Sophie Bauchinger, IAU
@Date Tue Dec  5 11:37:36 2023

"""

import pandas as pd

# get data from the SQL database
from toolpac.readwrite.sql_data_import import client_data_choice

import dictionaries as dcts
# import dictionaries as dcts

log_in = {"host": "141.2.225.99", "user": "Datenuser", "pwd": "AG-Engel1!"}

# southtrac_defs = dcts.campaign_definitions('SOUTHTRAC')

from SF6_GhOST_SQL_read_write import definitions, read
import SF6_GhOST_resample

def do_campaign(ghost_campaign):
    # Database access
    user = {"host": "141.2.225.99", "user": "Datenuser", "pwd": "AG-Engel1!"}
    
    instruments = dcts.instruments_per_campaign(ghost_campaign)
    
    special, flights, ghost_ms_substances, n2o_instr, n2o_substances = definitions(ghost_campaign)
    time, fractional_year, meteo, ecd, ms, n2o, ozone = read(user, ghost_campaign, special, flights, ghost_ms_substances, n2o_instr, n2o_substances)

    data = {}

    for instr in [ecd, ms]:

        # create one dataframe for ECD and one for MS
        Ghost=pd.concat([time, fractional_year, meteo[['P', 'LAT', 'LON', 'TH', 'STRAT_O3']], instr, n2o, ozone], axis=1)
        # simpler than resampling:
        Ghost=Ghost.dropna(subset=["P"])

        if instr.name=='MS':
            # some magic to have only one measurement of GhOST with mean value of UMAQS data
            # resampling for GhOST MS data from Markus
            time_cols=['DATETIME', 'year_frac']
            Ghost = SF6_GhOST_resample.ghost_resample_mod(Ghost, ghost_ms_substances, time_cols)
            Ghost = Ghost.dropna(subset=["P"])  # should not be necessary
        elif instr.name=='ECD':
            Ghost = Ghost.dropna(subset=["SF6"])

        # have two datasets: ST_ECD and ST_MS
        # save
        return Ghost
        # Ghost.to_csv(f'{ghost_campaign}_{instr.name}.csv', index=False, na_rep='NaN', sep=';')
    return data

#%%
def campaign_definitions(campaign):
    """  Returns parameters needed for client_data_choice per campaign.

    Parameters: 
        ghost_campaign (str): Name of the campaign, e.g. SOUTHTRAC
    """
    
    campaign_dicts = {
        "SOUTHTRAC" : dict(
            special = "ST all",
            ghost_ms_substances = ['HFC125', 'HFC134a', 'H1211', 'HCFC22'],
            n2o_instr = "UMAQS",
            n2o_substances = ["N2O", "CO", "CH4", "CO2"],
            flights=None),

        "TACTS" : dict(
            ghost_ms_substances = ['H1211'],
            flights = ["T1", "T2", "T3", "T4", "T5", "T6"],
            special = None,
            n2o_instr = ["TRIHOP_N2O", "TRIHOP_CO", "TRIHOP_CO2"],
            n2o_substances = [["N2O"], ["CO"], ["CO2"]]),
        
        "WISE" : dict(
            special = "WISE all",
            ghost_ms_substances = ['H1211'],
            n2o_instr = "UMAQS",
            n2o_substances = ["N2O", "CO"],
            flights=None),

        "PGS" : dict(
            special = "PGS all",
            ghost_ms_substances = ['H1211'],
            flights=None,
            n2o_instr = "TRIHOP",
            n2o_substances = ["N2O", "CO", "CH4"]),
        }

    if campaign not in campaign_dicts: 
        raise KeyError(f'{campaign} is not a valid GHoST campaign for SQL database access.')
    
    return campaign_dicts[campaign]

#%% 

def read(campaign): 
    """ Read campaign data from SQL database. """
    
    defs = dcts.campaign_definitions(campaign)
    
    meteo_data = client_data_choice(
          log_in,
          campaign = campaign,
          special = defs.get('special'),  # all flights
          meteo = True,
          flights = defs.get('flights'))
    
    time_data = client_data_choice(
            log_in,
            campaign=campaign,
            special = defs.get('special'),   # all flights
            time=True,
            flights = defs.get('flights'))
    
    ghost_ms_data = client_data_choice(log_in = log_in,
                                        instrument="GHOST_MS", 
                                        campaign = campaign, 
                                        special = defs.get('special'),
                                        substances = defs.get('ghost_ms_substances'),
                                        flights = defs.get('flights'),
                                        **defs)

    ghost_ecd_data = client_data_choice(log_in = log_in, 
                                        instrument="GHOST_ECD",
                                        special = defs.get('special'),
                                        substances = ['SF6', 'CFC12'],
                                        flights = defs.get('flights'),
                                        **defs)
    
    

client_data_choice()