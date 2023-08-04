# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 08:54:48 2023

@author: sophie_bauchinger
"""

from netCDF4 import Dataset as dt
import xarray as xr
import times
import glob

def fill_Fdata_nc(flight_numbers, path_caribic2_nc_data, VMISS):
    # read nc file from TPChange project
    # name convention CARIBIC2_YYYYMMDD_XXX_TPC_VYY.1.nc
    # =============================================================================
    print(flight_numbers)

    # only one flight_number supplied as integer
    if type(flight_numbers) is int:
        flight_numbers = [flight_numbers]
    # create empty dictionary
    Fdata = {}
    # fill directory
    for x in flight_numbers:
        if find_TPC_nc_file(x, path_caribic2_nc_data) is not None:
            ncfile = dt(find_TPC_nc_file(x, path_caribic2_nc_data), 'r')
            # get date from filename
            Fdata.update({f'F{x}_DATE': date_from_flight_info(ncfile)})
            # for compatibility with Nasa Ames
            Fdata.update({f'F{x}_SCOM_MA': [f'{item[0]}: {item[1]}' for item in ncfile.__dict__.items()]})

            ds = xr.open_dataset(find_TPC_nc_file(x, path_caribic2_nc_data))
            df = ds.to_dataframe()
            df = df.reset_index()
            df = df.replace(-9999, VMISS)
            for key in df.keys():
                if 'CLaMS' in key:
                    if 'F11' in key or 'F12' in key:        # convert to ppt
                        df[key] = df[key]*1e12
                    elif 'CO2' in key or 'H2O' in key:      # convert to ppm
                        df[key] = df[key]*1e6
                    else:                                   # convert to ppb
                        df[key] = df[key]*1e9

            df['TimeCRef'] = times.datetime_to_secofday(df['Time'], refdatetime=None)
            Fdata.update({f'F{x}_MS_TPC': df})

    return Fdata



def find_TPC_nc_file(flight_no, path_caribic2_nc_data):
    # x = flight number as integer
    # path_caribic2_nc_data: path to CARIBIC-2 TPChange data, subdirectory structure by year

    nc_file_list = [file for file in glob.glob(str(path_caribic2_nc_data) + '\**\*'
                                               '_'+str(flight_no)+'_'+ '*.nc', recursive=True)]

    if len(nc_file_list)==0:
        print(f'No file found for flight {flight_no}')
        return None
    if len(nc_file_list)==1:
        return nc_file_list[0]
    elif len(nc_file_list)>1:
        version = [(fname.replace('.nc', '')).split('_V')[-1] for fname in nc_file_list]
        version = [vers.replace('.', '') for vers in version]
        version = [int(vers) for vers in version]
        return nc_file_list[np.argmin(version)]


def date_from_nc_filename(fname):

    tmp = fname.split('\\')[-1]
    tmp = tmp.replace('.nc', '')
    print(tmp)
    tmp_date_str = tmp.split('_')[1]
    year = int(tmp_date_str[0:4])
    month = int(tmp_date_str[4:6])
    day = int(tmp_date_str[6:8])

    return [year, month, day]


def date_from_flight_info(ncfile):

    # flight_info looks like this:
    # 'Campaign: CARIBIC2; Flightnumber: 544; Start: 11:09:55 22.03.2018; End: 21:20:55 22.03.2018'
    tmp_str = ncfile.flight_info.split(';')[2]
    tmp_date_str = tmp_str.split(' ')[-1]

    year = int(tmp_date_str[6:10])
    month = int(tmp_date_str[3:5])
    day = int(tmp_date_str[0:2])

    return [year, month, day]

