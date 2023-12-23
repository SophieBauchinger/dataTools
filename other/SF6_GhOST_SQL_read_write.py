"""
Created at 14.04.2022

@ authors: Markus Jesswein, IAU


Get SF6 and N2O from SouthTRAC flights

"""
# =============================================================================
# Packages
# =============================================================================

import numpy as np
import pandas as pd
import datetime

from toolpac.readwrite.sql_data_import import client_data_choice

import SF6_GhOST_resample

from toolpac.readwrite.merge_reader import MergeReader



# =============================================================================
# gathering data
# =============================================================================


def definitions(ghost_campaign):

    if ghost_campaign == "SOUTHTRAC":
        special = "ST all"
        ghost_ms_substances = ['HFC125', 'HFC134a', 'H1211', 'HCFC22']
        n2o_instr = "UMAQS"
        n2o_substances = ["N2O", "CO", "CH4", "CO2"]
        flights=None
    elif ghost_campaign == "TACTS":
        ghost_ms_substances = ['H1211']
        flights = ["T1", "T2", "T3", "T4", "T5", "T6"]
        special = None
        # TACTS has tables TRIHOP_N2O
        # TACTS has tables TRIHOP_CO
        # TACTS has tables TRIHOP_CO2
        n2o_instr = ["TRIHOP_N2O", "TRIHOP_CO", "TRIHOP_CO2"]
        n2o_substances = [["N2O"], ["CO"], ["CO2"]]
    elif ghost_campaign == "WISE":
        special = "WISE all"
        ghost_ms_substances = ['H1211']
        n2o_instr = "UMAQS"
        n2o_substances = ["N2O", "CO"]
        flights=None
    elif ghost_campaign == "PGS":
        special = "PGS all"
        ghost_ms_substances = ['H1211']
        flights=None
        n2o_instr = "TRIHOP"
        n2o_substances = ["N2O", "CO", "CH4"]

    return special, flights, ghost_ms_substances, n2o_instr, n2o_substances


def read(user, ghost_campaign, special, flights, ghost_ms_substances, n2o_instr, n2o_substances):
    ghost_ms_data = client_data_choice(
        user,
        campaign=ghost_campaign,
        special=special,   # all flights
        instrument="GHOST_MS",
        substances=ghost_ms_substances,
        flights=flights
    )

    ghost_ecd_data = client_data_choice(
        user,
        campaign=ghost_campaign,
        special=special,   # all flights
        instrument="GHOST_ECD",
        substances=["SF6", "CFC12"],
        flights=flights
    )

    if ghost_campaign == "TACTS":
        n2o_data = [None for i in range(len(n2o_instr))]
        for i in range(len(n2o_instr)):
            n2o_data[i] = client_data_choice(
                user,
                campaign=ghost_campaign,
                special=special,   # all flights
                instrument=n2o_instr[i],
                substances=n2o_substances[i],
                flights=flights
                )
    else:
        n2o_data = client_data_choice(
            user,
            campaign=ghost_campaign,
            special=special,   # all flights
            instrument=n2o_instr,
            substances=n2o_substances,
            flights=flights
            )

    tmp_ozone = client_data_choice(
         user,
         campaign=ghost_campaign,
         special=special,  # all flights
         instrument='FAIRO',
         substances=['O3'],
         flights=flights
    )

    meteo_data = client_data_choice(
         user,
         campaign=ghost_campaign,
         special=special,  # all flights
         meteo=True,
         flights=flights
    )

    time_data = client_data_choice(
        user,
        campaign=ghost_campaign,
        special=special,   # all flights
        time=True,
        flights=flights
    )

    #####
    # von Johannes
    ######

    # print('HERE')
    if 'data' in ghost_ms_data._data.keys():
        ms = ghost_ms_data._data['data']      # pandas dataframe
    else:
        ms = ghost_ms_data._data['H1211']      # pandas dataframe

    ms.name = 'MS'

    ecd = ghost_ecd_data._data['data']    # pandas dataframe
    ecd.name = 'ECD'

    if ghost_campaign == "TACTS":
        n2o = pd.concat([n2o_data[0]._data['N2O'], n2o_data[1]._data['CO'], n2o_data[2]._data['CO2']], axis=1)
    else:
        n2o = n2o_data._data['data']      # pandas dataframe

    ozone = tmp_ozone._data['O3']

    meteo = meteo_data._data['meteo']     # use P LAT and LON # pandas dataframe
    time = time_data._data['DATETIME']    # pandas series


    def year_fraction(date):
        start = datetime.date(date.year, 1, 1).toordinal()
        year_length = datetime.date(date.year+1, 1, 1).toordinal() - start
        return date.year + float(date.toordinal() - start) / year_length


    fractional_year = np.empty(len(time))
    fractional_year[:] = np.nan
    for tt in range(0, len(time)):
        fractional_year[tt] = year_fraction(time[tt])
    fractional_year = pd.DataFrame({'year_frac': fractional_year[:]})

    return time, fractional_year, meteo, ecd, ms, n2o, ozone


def do_campaign(ghost_campaign):
    # Database access
    user = {"host": "141.2.225.99", "user": "Datenuser", "pwd": "AG-Engel1!"}
    special, flights, ghost_ms_substances, n2o_instr, n2o_substances = definitions(ghost_campaign)
    time, fractional_year, meteo, ecd, ms, n2o, ozone = read(user, ghost_campaign, special, flights, ghost_ms_substances, n2o_instr, n2o_substances)

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
        Ghost.to_csv(f'{ghost_campaign}_{instr.name}.csv', index=False, na_rep='NaN', sep=',')


ghost_campaign = "PGS"  # "SOUTHTRAC" "TACTS" "PGS" "WISE"
