# -*- coding: utf-8 -*-

# routines to read CARIBIC data

import C_tools

# =============================================================================
# import general  modules 
# =============================================================================

import pandas as pd
import numpy as np
import os
import fnmatch
from pathlib import Path


# %%
def read_flight_list(path, name):
    fname = Path(path, name + '.txt')

    if not os.path.isfile(fname):
        print('File ', fname, ' not found.')
        return
    
    df_flights = pd.read_csv(fname, sep='\t', na_filter=False)  # string NA should not be changed to NaN

    df_flights.index = df_flights['flight']
    del df_flights['flight']

    df_flights['date'] = df_flights['date'].astype(str)
    
    return df_flights


# %%
def find_most_recent(path, flight, prefix, high_res=False, verbose=False):
    os.chdir(path)
    subdirlist = next(os.walk('.'))[1]
    subdirlist.sort()    # not really needed
    flight_dir_i = [i for i, s in enumerate(subdirlist) if 'Flight'+str(flight) in s]
    # flight_dir_i is a list

    # print(flight_dir_i,subdirlist[flight_dir_i[0]])

    os.chdir(Path(os.getcwd(), subdirlist[flight_dir_i[0]]))
    all_files = [name for name in os.listdir('.') if os.path.isfile(name)]
    all_files.sort()  # not really needed
    # print(all_files)

    file_list = fnmatch.filter(all_files, prefix+'_*')  # not case sensitive
    # print(file_list)
    if high_res:  # use only highest resolution files if existing
        file_list_no_10s = [x for x in file_list if '_10s_' not in x]
        # check if sublist is empty (for some prefix only 10s files exist)
        if len(file_list) != 0 and len(file_list_no_10s) != 0:
            file_list = file_list_no_10s
    else:  # use only 10s resolution files
        file_list = [x for x in file_list if '_10s_' in x]

    file_list.sort()  # NEEDED

    if len(file_list) < 1:
        print('No '+prefix+' File found for flight '+str(flight)+'.')
        return None
    else:
        if verbose:
            print(Path(os.getcwd(), file_list[-1]))
        return Path(os.getcwd(), file_list[-1])


# %%
def read_ames_to_flight_df(path, flight, prefix, high_res=False):
    fname = find_most_recent(path, flight, prefix, high_res=high_res)
    if fname is None and prefix == 'MS':
        print('checking SU file ...')
        fname = find_most_recent(path, flight, 'SU')
    
    if fname is None:
        return None
    
    print('Reading in Files') # print(fname)

    lun = open(fname, "r")
    header_lines = int(str.split(lun.readline())[0])

    line = ''
    for i in range(11):
        line = lun.readline()

    vmiss_list = [int(x) for x in str.split(line)]
    # read this line into a list
    # na_values of read_csv can be a list
    vmiss_list = [vmiss_list[0]] + vmiss_list  # add one element for time column, no vmiss by default there

    lun.close()
    
    df_data = pd.read_csv(fname, sep='\t', skiprows=header_lines-1, na_values=vmiss_list)
    df_data.columns = map(str.lower, df_data.columns)
    
    if all([item in df_data.columns for item in ['year', 'month', 'day', 'hour', 'min', 'sec']]):
        df_data['year_frac'] = C_tools.make_year_cont(
                df_data['year'],
                df_data['month'],
                df_data['day'],
                df_data['hour'],
                df_data['min'],
                df_data['sec'])

    if 'month' in df_data.columns:
        df_data['season'] = C_tools.make_season(df_data['month'])
    
    return df_data


# %%
def read_flights_to_dict(path, flight_numbers, prefix, high_res=False):
    if prefix in ['INT', 'INT2', 'HCF', 'GHG', 'HFO']:
        high_res = True

    data_dict = {}
    
    for x in flight_numbers:
        # read file into dataframe
        df = read_ames_to_flight_df(path, x, prefix, high_res=high_res)

        # add dictionary to Fdata
        data_dict.update({'F'+str(x) + '_' + prefix.upper(): df})

    return data_dict

# %%
def read_WAS_trajs(path, flight, sample_no):
    # e.g. read_WAS_trajs(Path('D:\CARIBIC\Trajectories'), 554, 5)

    if type(flight) is not int:
        print('Only one flight can be read. Parameter flight has to be integer.')
        return None

    if type(sample_no) is int:
        sample_no = [sample_no]

    # create strings from sample numbers to locate them in file names
    sample_no_str = [str(x).zfill(2) for x in sample_no]
    # can also be used as dictionary keys
    TrajData=dict.fromkeys(sample_no_str)
    TrajData['headers']=[]

    flight_path=path/str(flight)
    os.chdir(flight_path)
    # check what files there are an extract the sample number string
    all_files = [name for name in os.listdir('.') if os.path.isfile(name)
                 and name.startswith('MA_') and name.endswith('_TR')]

    # check if directory was empty
    if len(all_files) == 0:
        print('Empty directory', flight_path)
        return None

    all_sample_str = [x.split('_')[5][1:3] for x in all_files]

    files_to_read = [x for i, x in enumerate(all_files) if all_sample_str[i] in sample_no_str]
    # check if files found match samples to be read
    if len(all_files) == 0:
        print('No files matching sample numbers found in directory', flight_path)
        return None

    files_to_read.sort()  # not really needed

    # read trajectory files and cut them into single trajectories
    # there are 31 trajectories per sample in one file
    for fname_idx, fname in enumerate(files_to_read):
        with open(flight_path/fname, 'r') as traj_file:
            data = pd.read_csv(traj_file, header=None, delim_whitespace=True)
            traj_file.close()
            print(f'File  {fname} read.')

        data.columns = ['TTTT', 'p0', 'lon', 'lat', 'p', 'u', 'v',
                        'w', 'T', 'q', 'PV', 'dPV', 'err', 'dummy']
        # Explanation from http://projects.knmi.nl/campaign_support/CARIBIC/descriptionTR_CARIBIC.txt
        #        TTTT     : time in minutes relative to start time yymmddhh2
        #        PRES0    : pressure at bottom of model=surface (0.1 hPa)
        #        LON, LAT : horizontal trajectory position (in 0.1 degrees)
        #        PRES     : vertical trajectory position (0.1 hPa)
        #        U, V     : horizontal winds in EW resp. NS direction at trajectory position
        #                   (in 0.1 m/s)
        #        W        : vertical wind at trajectory position (in mPa/s)
        #        T        : temperature in 0.1 K
        #        q        : moisure  mixing ratio in mg / kg
        #        PV       : Potential vorticity in 0.001 PVU      (1 PVU = 10**-6 K m**2 / kg / s )
        #        dPV/dp   : Potential vorticity gradient in PVU/Pa
        #        errflg   : error flag. If errflg=0 everything is OK.

        # column dummy arises from an empty column read by read_csv

        # find rows were new trajectories start:
        traj_headers = data.loc[data['TTTT'] == 'WTK']
        traj_headers.columns = ['m', 'lp', 'yymmddhh1', 'yymmddhh2', 'yymmddhh3', 'ts', 'npt',
                                'lon', 'lat', 'p', 'dt', 'dx', 'dy', 'frac']

        # remove volume trajectories
        max_lat=np.max(traj_headers['lat'])
        min_lat=np.min(traj_headers['lat'])

        traj_idx = data.index[data['TTTT'] == 'WTK'].tolist()
        traj_len = int(np.mean(np.diff(traj_idx)))
        is_vol_idx = []
        for i in traj_idx:
            if traj_headers.lat[i] in [min_lat, max_lat]:
                is_vol_idx.append(i)
        for i in is_vol_idx:
            data.drop(data.loc[i:i+traj_len-1].index, inplace=True)
            # default inplace=False returns copy, original unchanged

        traj_headers = data.loc[data['TTTT'] == 'WTK']
        traj_headers.columns = ['m', 'lp', 'yymmddhh1', 'yymmddhh2', 'yymmddhh3', 'ts', 'npt',
                                'lon', 'lat', 'p', 'dt', 'dx', 'dy', 'frac']

        data=data.reset_index(drop=True)
        traj_idx = data.index[data['TTTT'] == 'WTK'].tolist()
        print(traj_idx)

        # make values nicer for lat, lon and p:
        data['lon'] = data['lon']/10
        data['lat'] = data['lat']/10
        data['p'] = data['p']/10

        # check for negative values of longitude
        # and change longitude range from 0-360 to -180-+180
        data['lon'] = data['lon'].apply(lambda x: x if x <= 180 else x - 360.)

        print(f'\t{len(traj_idx)} trajectories found for sample {sample_no_str[fname_idx]}. '
              f'{len(is_vol_idx)} volume trajectories removed')

        TrajData[sample_no_str[fname_idx]] = [data.loc[i+1:i+traj_len-1] for i in traj_idx]
        TrajData['headers'].append(traj_headers)

    return TrajData


# %%
def read_NOAA_ts(path, fname, colname, header=51, dropnan=False, df_return=False):
    # colname supplies base column name in NOAA file, e.g. SF6catsMLO
    ref_data_df = pd.read_csv(path / fname, header=header, skiprows=1, delim_whitespace=True)
    # polyfit for lagtime calculation cannot deal with missing values in y array
    if dropnan:
        ref_data_df = ref_data_df.dropna(how='any', subset=[f'{colname}m'])
    year = ref_data_df[f'{colname}yr'].values
    month = ref_data_df[f'{colname}mon'].values
    year_frac = year + (month - 0.5) / 12
    mxr = ref_data_df[f'{colname}m'].values
    print('File ', path / fname, ' read.')
    if df_return:
        return ref_data_df
    else:
        return year_frac, year, month, mxr


# %%
def read_AGAGE_ts(path, fname, colname, header=13, dropnan=False, df_return=False):
    # colname supplies base column name in NOAA file, e.g. SF6catsMLO
    ref_data_df = pd.read_csv(path / fname, header=header)
    # polyfit for lagtime calculation cannot deal with missing values in y array
    ref_data_df.columns = ref_data_df.columns.str.replace(' ', '')  # data has spaces in column labels

    if dropnan:
        ref_data_df = ref_data_df.dropna(how='any', subset=[f'{colname}'])

    year = ref_data_df['year'].values
    month = ref_data_df['month'].values
    year_frac = ref_data_df['time'].values
    mxr = ref_data_df[f'{colname}'].values
    print('File ', path / fname, ' read.')
    if df_return:
        return ref_data_df
    else:
        return year_frac, year, month, mxr

#%%% Testing
if __name__=='__main__':# create empty dictionary
    caribic2data = Path('E:\CARIBIC\Caribic2data')
    flight_list_name = 'flight_list_GHG'

    df_flights = read_flight_list(caribic2data, flight_list_name)
    flight_numbers = df_flights.index.tolist()
    
    Fdata = {}
    
    dict_new = read_flights_to_dict(caribic2data, flight_numbers, 'GHG')
    Fdata.update(dict_new)

    dict_new = read_flights_to_dict(caribic2data, flight_numbers, 'INT')
    Fdata.update(dict_new)
    
    dict_new = read_flights_to_dict(caribic2data, flight_numbers, 'INT2', high_res=True)
    Fdata.update(dict_new)