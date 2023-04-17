# -*- coding: utf-8 -*-
"""
Simple examples on reading data from CARIBIC, NOAA, AGAGE and MOZART files

@Author: Sophie Bauchimger, IAU
@Date: Mon Feb 13 11:51:02 2023

"""
import datetime as dt
import geopandas
import numpy as np
import pandas as pd
from shapely.geometry import Point
import xarray as xr

from toolpac.readwrite import find
from toolpac.readwrite.FFI1001_reader import FFI1001DataReader

#%% Define monthly mean
def monthly_mean(df, first_of_month=False):
    """ 
    df: Pandas DataFrame with datetime index
    first_of_month: bool, if True sets monthly mean timestamp to first of that month
    
    Returns dataframe with monthly averages of all values
    """
    df_MM = df.groupby(pd.PeriodIndex(df.index, freq="M")).mean(numeric_only=True)
    if first_of_month: 
        df_MM['Date_Time'] = [dt.datetime(y, m, 1) for y, m in zip(df_MM.index.year, df_MM.index.month)]
        df_MM.set_index('Date_Time', inplace=True)
    return df_MM

#%% Mauna Loa
def mlo_data(path = r'C:\Users\sophie_bauchinger\toolpac tutorial\mlo_SF6_Day.dat', year = 2012):
    """ Create dataframe for given mlo data (.dat) for a speficied year """
    # extract and stitch together names and units for column headers
    with open(path) as f:
        for i, line in enumerate(f):
            if i == 38: title = line.split()

    # df = pd.read_csv(path, sep=" ", skiprows=39, dtype=float)
    # print(df)
    # print(df.columns, '\n', title)
    # df.columns = title

    mlo_data = np.genfromtxt(path, delimiter="", skip_header=39)
    df = pd.DataFrame(mlo_data, columns=title, dtype=float)
    df = df.loc[df.SF6catsMLOyr < year+1].loc[df.SF6catsMLOyr > year-1].reset_index()
    if 'SF6catsMLOday' in df.columns:
        time = [dt.datetime(int(y), int(m), int(d)) for y, m, d in zip(df.SF6catsMLOyr, df.SF6catsMLOmon, df.SF6catsMLOday)]
        df = df.drop('SF6catsMLOday', axis=1)
    else: time = [dt.datetime(int(y), int(m), 1) for y, m in zip(df.SF6catsMLOyr, df.SF6catsMLOmon)]
    df = df.drop(df.iloc[:, :3], axis=1)
    df.astype(float)
    fix_time = 'Date_Time'
    df[fix_time] = time
    df.set_index(fix_time, inplace=True)
    return df

if __name__=='__main__':
    mlo_file = r'C:\Users\sophie_bauchinger\toolpac tutorial\mlo_SF6_Day.dat'
    mlo_2012 = mlo_data(mlo_file, 2012)

    mlo_file_MM = r'C:\Users\sophie_bauchinger\toolpac tutorial\mlo_SF6_MM.dat'
    mlo_2012_MM = mlo_data(mlo_file_MM, 2012)

#%% Mace Head
def mhd_data(file = r'C:\Users\sophie_bauchinger\toolpac tutorial\MHD-medusa_2012.dat'):
    """ Create dataframe when given path to mace head data file (.dat)"""
    # extract and stitch together names and units for column headers
    with open(file) as f:
        for i, line in enumerate(f):
            if i == 14: units = line.split()
            if i == 15: title = line.split(); break
    column_headers = [name + "[" + unit + "]" for name, unit in zip(title, units)]

    mhd_data = np.genfromtxt(file, delimiter="", skip_header=16)

    time = [dt.datetime(int(mhd_data[i][3]), int(mhd_data[i][2]), int(mhd_data[i][1]),
                     int(mhd_data[i][4]), int(mhd_data[i][5])) for i in range(0, len(mhd_data))]

    df = pd.DataFrame(mhd_data, columns=column_headers, dtype=float)
    df = df.replace(0, float('NaN')); df = df.replace(-99.990, float('NaN'))
    df = df.drop(df.iloc[:, :7], axis=1)
    df = df.astype(float)
    fix_time = 'Date_Time'
    df[fix_time] = time
    df.set_index(fix_time, inplace=True)
    return df

if __name__=='__main__':
    mhd_2012 = mhd_data()

#%% CARIBIC
def caribic_gdf(year):
    df = pd.DataFrame()
    parent_dir = r'C:\Users\sophie_bauchinger\toolpac tutorial\Caribic-2 data'
    for current_dir in find.find_dir("*_{}*".format(year), parent_dir)[1:]:
        flight_nr = int(str(current_dir)[-12:-9])
        try:
            f = find.find_file("GHG*", current_dir)[-1]
            f_data = FFI1001DataReader(f, df=True, xtype = 'secofday')
            df_temp = f_data.df # index = Datetime
            df_temp.insert(0, 'Flight number',
                           [flight_nr for i in range(0, df_temp.shape[0])])
            df = pd.concat([df, df_temp])
        except: print(f'No GHG data collected on Flight {flight_nr} in {year}')
    # Convert longitude and latitude into geometry objects -> GeoDataFrame
    geodata = [Point(lat, lon) for lon, lat in zip(
        df['lon; longitude (mean value); [deg]\n'],
        df['lat; latitude (mean value); [deg]\n'])]
    gdf = geopandas.GeoDataFrame(df, geometry=geodata)

    # Drop all unnecessary columns [info within datetime, geometry]
    gdf = gdf.drop(['TimeCRef; CARIBIC_reference_time_since_0_hours_UTC_on_first_date_in_line_7; [s]',
                'year; date of sampling: year; [yyyy]\n',
                'month; date of sampling: month; [mm]\n',
                'day; date of sampling: day; [dd]\n',
                'hour; time of sampling (mean value): hour; [HH]\n',
                'min; time of sampling (mean value): minutes; [MM]\n',
                'sec; time of sampling (mean value): seconds; [SS]\n',
                'lon; longitude (mean value); [deg]\n',
                'lat; latitude (mean value); [deg]\n',
                'type; type of sample collector: 0 glass flask from TRAC, 1 metal flask from HIRES; [0-1]\n'],
               axis=1)
    gdf = gdf[gdf['SF6; SF6 mixing ratio; [ppt]\n'].notna()]
    return gdf

if __name__=='__main__':
    caribic_2012 = caribic_gdf(2012)
    caribic_2008 = caribic_gdf(2008)
# if df['SF6; SF6 mixing ratio; [ppt]\n'].isnull().sum() == len(df.index): pass

#%% MOZART
""" Data & Coordinates

Data variables:
    hyam     Hybrid A coordinate
    hybm     Hybrid B coordinate
    P0       Reference pressure [Pa
    PS       Surface Pressure [Pa]
    SF6      Annual mean SF6 dry air mole fraction [pmol/mol]

Coordinates
    time        39  [year]                  1970 to 2008
    level       28  [hybrid sigma level]    2.7 to 995.0
    latitude    36  [degrees_north]         -90 to 90
    longitude   72  [degrees_east]          0 to 355
"""
def mozart_data(year=None, level = 27,
                file = r'C:\Users\sophie_bauchinger\toolpac tutorial\RIGBY_2010_SF6_MOLE_FRACTION_1970_2008.nc'):
    """ Returns xarray Dataset and GeoPandas GeoDataFrame of MOZART model data """
    with xr.open_dataset(file) as ds:
        # choose a hybrid sigma level (uppermost or lowermost)
        ds = ds.isel(level=level)
        if year: ds = ds.sel(time=year)
        df = ds.to_dataframe()

    # Convert longitude and latitude into geometry objects -> GeoDataFrame
    geodata = [Point(lat, lon) for lon, lat in zip(
        df.index.to_frame()['longitude'], df.index.to_frame()['latitude'])]
    gdf = geopandas.GeoDataFrame(
        df.reset_index().drop(['longitude', 'latitude', 'scalar', 'P0'], axis=1),
        geometry=geodata)
    return ds, gdf

reference_pressure = 100000 # ds.P0

if __name__=='__main__':
    mozart_file = r'C:\Users\sophie_bauchinger\toolpac tutorial\RIGBY_2010_SF6_MOLE_FRACTION_1970_2008.nc'
    mozart_2008 = mozart_data(2008, 27, mozart_file)
