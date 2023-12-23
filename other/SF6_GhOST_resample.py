
# Original routine from Markus Jesswein
# modified by Tanja Schuck

"""
Created at 07.06.2021

@ authors: Markus Jesswein, IAU

Get Tracer from database

"""
# =============================================================================
# Packages
# =============================================================================

import numpy as np
import pandas as pd


def ghost_resample_mod(mission_data, block_cols, time_cols):
    """
    Resampling GhOST-MS measurements (maybe rename function as it can be used for other instruments as well)
    :param mission_data: DataFrames of combined data
    :param block_cols: columns of DataFrame to be searched for blocks
    :param time_cols: columns with time parameters not to be averaged over block time interval
    returns a subset of the DataFrame with all non-time columns averaged
    """
    # List of indexes where we have measurements
    measurement_indexes = mission_data.dropna(subset=block_cols).index.values

    # lists for measurements block starts and ends
    sample_starts = [measurement_indexes[0]] # first values is set manually
    sample_ends = []

    # every time there is a jump in the measurements_indexes list,
    # the i+1 value is added to the samples_starts list
    # and the i value is added to the sample_ends list
    for i in range(len(measurement_indexes) - 1):
        if measurement_indexes[i + 1] != measurement_indexes[i] + 1:
            sample_starts.append(measurement_indexes[i + 1])
            sample_ends.append(measurement_indexes[i])

    # manually adding the last value to sample_ends
    sample_ends.append(measurement_indexes[-1])

    # middle_index for the time (time is set to the middle of the enrichment)
    middle_index = [int((sample_starts[i] + sample_ends[i]) / 2) for i in range(len(sample_starts))]

    # new dataFrames
    data_resample = pd.DataFrame()

    no_time_cols = [column for column in mission_data.columns if column not in time_cols]

    # values at centre of interval for time variables
    for column in time_cols:
        data_resample[column] = [mission_data[column].loc[middle_index[i]] for i in range(len(middle_index))]

    # mean values for the sample data
    for column in no_time_cols:
        data_resample[column] = [np.mean(mission_data[column].loc[sample_starts[i]:sample_ends[i]]) for i
                                 in range(len(sample_starts))]

    return data_resample

