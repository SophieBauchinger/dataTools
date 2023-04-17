
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from toolpac.ext.ccgcrv import ccg_filter
import C_tools


def read_NOAA_ts_mlo_co2(daily=False, firstyear=None, lastyear=None):
    ref_path = Path(r'D:\Daten_andere\NOAA')
    if daily:
        co2_fname = 'co2_mlo_surface-insitu_1_ccgg_DailyData.txt'
        headerlines=149
        cols = ['site_code', 'year', 'month', 'day', 'hour', 'minute', 'second', 'year_frac',
                'co2', 'stdev', 'nvalue', 'latitude', 'longitude', 'altitude', 'elevation', 'intake_height', 'qcflag']
    else:
        co2_fname = 'co2_mm_mlo.txt'
        headerlines=51
        cols = ['year', 'month', 'year_frac', 'co2', 'co2_deseas', 'days', 'stdev', 'unc_ppm']

    # C_read.read_NOAA_ts does not work for MLO co2 data
    ref_data_df = pd.read_csv(ref_path/co2_fname, header=headerlines, skiprows=1, delim_whitespace=True)
    ref_data_df.columns=cols

    for x in ['co2', 'days', 'stdev', 'unc_ppm']:
        if x in ref_data_df.columns:
            ref_data_df.loc[(ref_data_df[x] < 0), x] = np.nan

    if firstyear and not lastyear:
        return ref_data_df.loc[ref_data_df['year_frac'] > firstyear].reset_index(drop=True)
    elif lastyear and not firstyear:
        return ref_data_df.loc[ref_data_df['year_frac'] < lastyear+1].reset_index(drop=True)
    elif firstyear and lastyear:
        return ref_data_df.loc[(lastyear+1 > ref_data_df['year_frac']) & (ref_data_df['year_frac'] > firstyear)]\
            .reset_index(drop=True)
    else:
        return ref_data_df


def read_NOAA_ts_nh_co2(firstyear=2005., lastyear=2020.):
    ref_path = Path(r'D:\Daten_andere\NOAA')
    co2_fname = 'zone_nh.mbl.co2'
    cols = ['year_frac', 'co2']

    ref_data_df = pd.read_csv(ref_path/co2_fname, delim_whitespace=True)
    ref_data_df.columns = cols

    if firstyear and not lastyear:
        return ref_data_df.loc[ref_data_df['year_frac'] > firstyear].reset_index(drop=True)
    elif lastyear and not firstyear:
        return ref_data_df.loc[ref_data_df['year_frac'] < lastyear+1].reset_index(drop=True)
    elif firstyear and lastyear:
        return ref_data_df.loc[(lastyear+1 > ref_data_df['year_frac']) & (ref_data_df['year_frac'] > firstyear)]\
            .reset_index(drop=True)
    else:
        return ref_data_df


def appl_filt(ref_data_df, numharm=2):
    xp = ref_data_df.year_frac.to_numpy()
    yp = ref_data_df.co2.to_numpy()

    # drop nan from both arrays
    xp_nonan = xp[np.logical_not(np.isnan(yp))]
    yp_nonan = yp[np.logical_not(np.isnan(yp))]

    filt = ccg_filter.ccgFilter(xp_nonan, yp_nonan, numharmonics=numharm, debug=False)
    for x in dir(filt):
        print(x)

    return filt


def ctrl_plot_fit(filt, filt_CAR=None):
    fig = plt.figure(figsize=(10, 5))
    ax = plt.subplot(111)
    ax.scatter(filt.xp, filt.yp)
    ax.plot(filt.xp, filt.getFunctionValue(filt.xp), color='red')
    plt.ylabel('CO2 [ppm]')
    plt.tight_layout()
    if filt_CAR is not None:
        ax.scatter(filt_CAR.xp, filt_CAR.yp)
        ax.plot(filt_CAR.xp, filt_CAR.getFunctionValue(filt_CAR.xp), color='green')
        plt.plot(filt.xp, filt_CAR.getFunctionValue(filt.xp), color='black')
        plt.xlim(left=2004.5, right=max(filt.xp)+0.5)
        ax.get_xlim()
        plt.ylim(bottom=filt.getFunctionValue(2005)*0.97)


def eval_seas_max_min(filt, filt_CAR):
    MLO_seas = pd.DataFrame(filt.getAmplitudes(),
                            columns=['year', 'amp', 'max_date', 'max_val', 'min_date', 'min_val'])
    CAR_seas = pd.DataFrame(filt_CAR.getAmplitudes(),
                            columns=['year', 'amp', 'max_date', 'max_val', 'min_date', 'min_val'])


"""
    ref_data_df = read_NOAA_ts_mlo_co2(daily=False, firstyear=2005., lastyear=2020.)    
    ref_data_df = read_NOAA_ts_nh_co2(firstyear=2005., lastyear=2020.)
    
    CAR_data_df = C_tools.extract_data(df_flights, Fdata, ['co2','tropo','year_frac', 'lat'], flight_numbers,
                                       select_var=['tropo', 'lat'], select_value=[True, 30.], select_cf=['EQ', 'GE'])
    # CAR_data_df = C_tools.conc_all_flights(Fdata, flight_numbers, 'GHG') # does not contain tropo
   
    numharm=2
    filt = appl_filt(ref_data_df, numharm=numharm)
    filt_CAR = appl_filt(CAR_data_df, numharm=numharm)

    ctrl_plot_fit(filt, filt_CAR)
    
"""