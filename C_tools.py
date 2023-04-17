# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 10:53:02 2019

@author: Schuck
"""

# =============================================================================
# functions in file:
#
# extract_data(df_flights,Fdata, var,flight=None,route=None,select_var=None,select_value=None,select_cf=None))
# check_select(select_var,select_value,select_cf)
# get_op(op)
# make_year_cont(year,month,day,hour,minute,sec)
# create_flight_list_file(path, prefix) #create_flight_list_file('D:\\CARIBIC\\Caribic2data','HCF')
# which_flight(df_flights, var, value)
# rebin_data(df_data, xbmin, xbmax, xbin,
#              bin2d = True,
#              ybmin=None, ybmax=None, ybin=None()
# make_flight_list(df_flights,flight=None,route=None)
#
# =============================================================================


import operator
import pandas as pd
import datetime
import os
import math
import numpy as np
import scipy.odr as odr
from pathlib import Path
import fnmatch

from textwrap import fill

from toolpac.calc import bin_1d_2d
import C_read


# %%
def extract_data(df_flights, Fdata, var, flight, route=None,
                 # flight can be an integer or list of integers
                 # route can be a string or list of strings
                 select_var=None, select_value=None, select_cf=None
                 # can be string variables of lists of strings
                 # select_cf can be LT, GT, EQ, LE, GE, case insensitive
                 ):

    if type(var) != list:
        varlist = [var.lower()]
    else:
        varlist = [x.lower() for x in var]

    if 'day' not in varlist:
        varlist.insert(0, 'day')
    if 'month' not in varlist:
        varlist.insert(0, 'month')
    if 'year' not in varlist:
        varlist.insert(0, 'year')
    if 'timecref' not in varlist:
        varlist.insert(0, 'timecref')

    check_res = check_select(select_var, select_value, select_cf)
    if check_res[0] is False:
        print('No data extracted.')
        return None

    select_varlist = check_res[1]
    select_valuelist = check_res[2]
    select_cflist = check_res[3]

    # select flights for one or more route(s)
    if route is not None:
        print('Parameter route is set, ignoring parameter flight.')
        if type(route) == list:
            flight = []
            for i in range(len(route)):
                flight = flight + df_flights.index[df_flights['route'] == route[i]].tolist()
        else:
            flight = df_flights.index[df_flights['route'] == route].tolist()

    flight_list = []
    if type(flight) is int:  # only one flight given by number
        flight_list = [flight]
    if type(flight) is list:  # only one flight
        flight_list = flight

    prefixes = set([x.split('_', 1)[1] for x in Fdata.keys()])
    if 'MS' in prefixes:
        prefixes.remove('MS')
    df_merge = do_data_merge(Fdata, flight_list, prefixes=list(prefixes))
    varlist.insert(0, 'flight')  # flight is not a variable in initial data but is present to df_merge

    var_not_found = [var for var in varlist if var not in df_merge.columns]
    if len(var_not_found) > 0:
        print(f'Variable(s) {var_not_found} not found in data. Check name.')
        print('Variables in data:')
        print(fill(str(df_merge.columns.tolist()), width=90))

        for x in var_not_found:
            varlist.remove(x)

    sel_var_not_found = [var for var in select_varlist if var not in df_merge.columns]
    if len(sel_var_not_found) > 0:
        print(f'Selection variable(s) {sel_var_not_found} not found in data. Check name.')
        print('Variables in data:')
        print(fill(str(df_merge.columns.tolist()), width=90))

        for x in sel_var_not_found:
            idx = select_varlist.index(x)
            select_varlist.remove(x)
            select_valuelist.pop(idx)
            select_cflist.pop(idx)

    if len(select_varlist) == 0:
        select = False
    else:
        select=True

    df_merge = df_merge[list(set(varlist + select_varlist))]
    # print(df_merge.columns)
    # print(varlist)

    all_select=[]
    if select:
        for i, (var, value, cf) in enumerate(zip(select_varlist, select_valuelist, select_cflist)):
            # print(i, var, value, cf, type(value))
            select_index = df_merge.index[get_op(cf)(df_merge[var], value)].tolist()
            # all_select = all_select + select_index  # results in multi condition OR
            if i == 0:
                all_select = select_index
            else:
                all_select = [x for x in all_select if x in select_index]
            # all_select = [x for x in all_select if x in select_index]

        df_data_sub = df_merge[varlist].iloc[all_select].copy()
    else:
        df_data_sub = df_merge[varlist].copy()

    df_data_sub.reset_index(drop=True, inplace=True)

    return df_data_sub


# %%
def check_select(select_var: object, select_value: object, select_cf: object) -> object:
    len_check = True

    select_varlist = []
    select_valuelist = []
    select_cflist = []

    if select_var is not None:
        if type(select_var) != list:
            select_varlist = [select_var.lower()]
        else:
            select_varlist = [x.lower() for x in select_var]

    if select_value is not None:
        if type(select_value) != list:
            select_valuelist = [select_value]
        else:
            select_valuelist = [x for x in select_value]

    if select_cf is not None:
        if type(select_cf) != list:
            select_cflist = [select_cf.lower()]
        else:
            select_cflist = [x.lower() for x in select_cf]

    if len(select_varlist) != len(select_valuelist):
        print(select_varlist)
        print(select_valuelist)
        print(select_cflist)
        print('Number of select values does not match number of select variables')
        len_check = False

    if len(select_varlist) != len(select_cflist):
        print(select_varlist)
        print(select_valuelist)
        print(select_cflist)
        print('Number of select operators does not match number of select variables')
        len_check = False

    return len_check, select_varlist, select_valuelist, select_cflist


# %%
def get_op(op):
    op = op.lower()
    return {
        'lt': operator.lt,
        'le': operator.le,
        'eq': operator.eq,
        'ge': operator.ge,
        'gt': operator.gt,
        }[op]


# %%
def make_year_cont(year, month, day, hour, minute, sec):
    saved_args = locals()

    int_in = False
    if type(year) == int:
        int_in = True
        year = [year]
        month = [month]
        day = [day]
        hour = [hour]
        minute = [minute]
        sec = [sec]
    else:
        arg_len = [len(x) for x in saved_args.values()]
        if sum(arg_len) != len(arg_len * arg_len[0]):
            print('varying length of parameters')
            return None

    # print(locals().types())
    year_frac = len(year)*[None]
    for i in range(len(year)):
        year_start = datetime.datetime(year[i], 1, 1, 0, 0, 0)
        year_end = datetime.datetime(year[i]+1, 1, 1, 0, 0, 0)
        now = datetime.datetime(year[i], month[i], day[i], hour[i], minute[i], sec[i])
        year_frac[i] = year[i] + (now - year_start)/(year_end - year_start)

    # if input was integer numbers rather than lists of integers
    # then return float instead of list of floats
    if int_in:
        year_frac = year_frac[0]

    return year_frac


# %%
def make_season(month):

    season = len(month)*[None]
    for i, m in enumerate(month):
        if m in [3, 4, 5]:
            season[i] = 1
        elif m in [6, 7, 8]:
            season[i] = 2
        elif m in [9, 10, 11]:
            season[i] = 3
        elif m in [12, 1, 2]:
            season[i] = 4

    return season


# %%
def create_flight_list_file(path, prefix):

    prefix = prefix.upper()
    print(path)
    os.chdir(path)
    outfile = Path(path, 'flight_list_'+ prefix+ '.txt')
    if not os.path.isfile(outfile):
        print('File ', outfile, ' does not exist and will be created.')
    else:
        confirm = input('File ' + str(outfile) + ' does exist and will be overwritten. (y/n) ')
        if confirm != 'y':
            print('Stopped.')
            return None

    airportlist = Path(path, 'CARIBIC_Airports.csv')
    df_airports = pd.read_csv(airportlist, sep=';')
    df_airports = df_airports.set_index('IATA')

    subdirlist = next(os.walk('.'))[1]
    subdirlist.sort()    # not really needed
    flight_dir = [x for x in subdirlist if 'Flight' in x]
    file_list = []
    for y in flight_dir:
        os.chdir(y)
        all_files = [name for name in os.listdir('.') if
                     os.path.isfile(name) and prefix in name]
        all_files.sort()

        if len(all_files) == 1:
            file_list.append(all_files[0])
        elif len(all_files) > 1:
            file_list.append(all_files[-1])
        os.chdir('..')

    flightno = [x.split('_')[prefix.count('_')+2] for x in file_list]
    dest1 = [x.split('_')[prefix.count('_')+3] for x in file_list]
    dest2 = [x.split('_')[prefix.count('_')+4] for x in file_list]
    date_str = [x.split('_')[prefix.count('_')+1] for x in file_list]
    route = [' '] * len(file_list)

    for i in range(len(flightno)):
        continent1 = df_airports.loc[dest1[i]]['Continent']
        continent2 = df_airports.loc[dest2[i]]['Continent']
        # print(continent1,continent2)
        if continent1 == continent2:
            route[i] = continent1
        elif continent1 == 'Europe':
            route[i] = continent2
        elif continent2 == 'Europe':
            route[i] = continent1

        if route[i] == 'South_America':
            if dest2[i] in ['BOG', 'CCS', 'PMV'] or dest1[i] in ['BOG', 'CCS', 'PMV']:
                route[i] = 'South_Am_north'
            else:
                route[i] = 'South_Am_south'

        if route[i] == 'Asia':
            if dest2[i] in ['BKK', 'KUL', 'MAA'] or dest1[i] in ['BKK', 'KUL', 'MAA']:
                route[i] = 'Asia_south'
            else:
                route[i] = 'Asia_east'

    lun = open(outfile, "w+")
    lun.write('flight\tdest1\tdest2\troute\tdate\n')
    for i in range(len(file_list)):
        lun.write(flightno[i] + '\t' +
                  dest1[i] + '\t' +
                  dest2[i] + '\t' +
                  route[i] + '\t' +
                  date_str[i] + '\n')

    lun.close()


# %%
def which_flight(df_flights, var, value):
    # returns a list of flight numbers with value of var
    # var: variable may be any variable present in df_flights
    # var may additionally be year or month or dest to search both destinations in parallel

    if var not in df_flights.columns and var not in ['year', 'month']:
        print('Variable not found in df_flights.')
        print(df_flights.columns.tolist())
        return None

    which = []

    if var in df_flights.columns.tolist():
        lst = df_flights[var].values.tolist()
        if type(value) is str:
            which = [i for i, x in enumerate(lst) if fnmatch.fnmatch(x, value)]
        else:
            which = [i for i, x in enumerate(lst) if x == value]
    elif var == 'dest':
        which1 = [i for i in range(len(df_flights))
                  if value in df_flights['dest1'].iloc[i]]
        which2 = [i for i in range(len(df_flights))
                  if value in df_flights['dest2'].iloc[i]]
        which = which1 + which2
        which.sort()
    elif var in ['year', 'month']:
        # dat in df_flights is an integer number, not a string
        if type(value) == str:
            value = int(value)

        temp = []
        if var == 'year':
            temp = [math.floor(int(x)/10000)
                    for x in df_flights['date'].values.tolist()]
        elif var == 'month':
            temp = [math.floor(100*math.modf(int(x)/10000)[0])
                    for x in df_flights['date'].values.tolist()]

        which = [i for i in range(len(temp))
                 if temp[i] == int(value)]

    flight_nos = [df_flights.index.tolist()[x] for x in which]

    return flight_nos


# %%
def rebin_data(df_data, xbmin, xbmax, xbin,
               bin2d=True,
               ybmin=None, ybmax=None, ybin=None):
    # df_data is dataframe with either 2 (for 1d binning) or 3 (2d) columns
    # in case of 2 columns data for 2d binning number of points in bins will be counted
    # in case of 3 columns mean of 3rd column will be calculated

    # uses bin_2d and bin_1d by Nils Schohl

    if bin2d:
        if (ybmin is None) or (ybmax is None) or (ybin is None):
            print('Missing parameter ybmin, ybmax, ybin')
            return None

    if len(df_data.columns) > 3:
        print("Dataframe has more than 3 columns; only first 2/3 (1d/2d binning) will be used")

    if not np.issubdtype(df_data.iloc[:, 0].dtype, np.number):
        print("First column of dataframe needs to be numeric.")
        return None

    if len(df_data.columns) > 1:
        if not np.issubdtype(df_data.iloc[:, 1].dtype, np.number):
            print("Second column of dataframe needs to be numeric.")
            return None

    if len(df_data.columns) > 2:
        if not np.issubdtype(df_data.iloc[:, 2].dtype, np.number):
            print("Second column of dataframe needs to be numeric.")
            return None

    print(df_data.columns)
    x_data = df_data.iloc[:, 0].values

    y_data = None
    if len(df_data.columns) == 1:
        y_data = np.empty(len(df_data), dtype=float)
    if len(df_data.columns) > 1:
        y_data = df_data.iloc[:, 1].values
    if len(df_data.columns) > 2:
        z_data = df_data.iloc[:, 2].values
    else:
        z_data = np.empty(len(df_data), dtype=float)

    if bin2d:
        rebinned = bin_1d_2d.bin_2d(z_data, x_data, y_data, xbmin, xbmax, xbin, ybmin, ybmax, ybin)
        # returns object
        # use 2d_plot for plotting
    else:
        rebinned = bin_1d_2d.bin_1d(y_data, x_data, xbmin, xbmax, xbin)

    return rebinned


# %%
def do_data_cut(df, Fdata, prefix, columns, over=True, over_all=False):
    # data from dataframe df will be distributed into subdataframes of Fdata
    # based on flight number --> df has to contain a column 'flight',
    # flights not present in df will be filled with NaN in Fdata if not Fdata[f'F{flight}_{prefix}] is None
    # target dataframes in Fdata are determined by prefix, only one prefix can be done at a time
    # if a prefix does not exist it will be created
    # over_all=True overwrites complete dataframe
    # over=True overwrites columns

    if(len(df)) == 0:
        print('No data found in dataframe.')
        return

    if 'flight' not in df.columns:
        print('Dataframe does not contain column \'flight\'. Nothing will be done.')
        return

    if 'timecref' not in df.columns:
        print('Dataframe does not contain column \'timecref\'. Nothing will be done.')
        return

    if 'timecref' not in columns:
        columns.extend(['timecref'])
        print(columns)
        return

    if not all(i in df.columns for i in columns):
        print('Dataframe does not contain all column names. Nothing will be done.')
        return

    keys = [x for x in Fdata.keys() if x.endswith(f'_{prefix}')]
    print(keys)
    if len(keys) == 0:
        print(f'Dictionary does not contain keys ending with _{prefix}. Keys will be created')

    unique_flights_Fdata = [int(x[1:4]) for x in Fdata.keys() if x.endswith('_INT')]
    # print(unique_flights_Fdata)
    unique_flights_df = list(set(df['flight']))
    # print(unique_flights_df)

    for f in unique_flights_Fdata:
        if f in unique_flights_df:
            # print(f)
            df_sub = df.loc[df['flight'] == f][columns].copy()
            if f'F{f}_{prefix}' not in Fdata.keys() or over_all is True or Fdata[f'F{f}_{prefix}'] is None:
                # if key does not exist then create and add sub-dataframe
                # over=True forces this
                Fdata[f'F{f}_{prefix}'] = df_sub
            else:
                # print('here')
                # if key does exist then merge new columns from sub-dataframe into it
                tmp_merge = Fdata[f'F{f}_{prefix}'].copy()
                Fdata.update({f'F{f}_{prefix}': None})
                # print(Fdata[f'F{f}_{prefix}'])
                tmp_merge = pd.merge(tmp_merge, df_sub, on=['timecref'], how="outer", suffixes=('', '_copy'))
                # print(tmp_merge.columns)
                if over is False:
                    # do not overwrite existing data, only non-existing columns will be appended
                    drop_list = tmp_merge.filter(regex='_copy$').columns.tolist()
                else:
                    # overwrite existing columns with new data if they are listed in input parameter columns
                    # columns not specified in parameter columns will remain unchanged
                    drop_list = tmp_merge.filter(regex='_copy$').columns.tolist()
                    drop_list = [x.replace('_copy', '') for x in drop_list if x.replace('_copy', '') in columns]

                # print(drop_list)
                tmp_merge.drop(drop_list, axis=1, inplace=True)
                tmp_merge.columns = [x.replace('_copy', '') for x in tmp_merge.columns]
                # print(tmp_merge)
                Fdata.update({f'F{f}_{prefix}': tmp_merge})
        else:
            print(f'Flight {f}: adding None')
            Fdata.update({f'F{f}_{prefix}': None})


# %%
def do_data_merge(Fdata, flight_list, prefixes=['INT', 'GHG', 'HCF', 'HFO'], verbose=False):
    """ 
    given dictionary of dataframe for each flight (with different suffix for msmt types) 
    I think we're now creating one dataframe for all the different flights? 
    """

    if not type(prefixes) is list:
        print('Prefixes have to be supplied as a list of strings. Nothing done.')
        return None

    if 'MS' in prefixes:
        print('\"MS\" found in prefixes, data merge only works for sample data. \"MS\" will be omitted')
        prefixes.remove('MS')

    if verbose:
        print('Merging ', prefixes)

    # reordering prefixes to make sure that the prefix with the maximum entries is on 0th position
    # otherwise pd-merge below will result in erroneous NaN entries
    len_list = [None]*len(prefixes)
    for prefix in prefixes:
        len_list[prefixes.index(prefix)] = sum([len(Fdata[x]) for x in Fdata.keys()
                                                if x.endswith(f'_{prefix}') and Fdata[x] is not None
                                                and int(x[1:4]) in flight_list])
    temp_zip = sorted(zip(len_list, prefixes), key=lambda t: t[0], reverse=True)
    prefixes = list(zip(*temp_zip))[1]
    # print(prefixes)

    max_len = max(len_list)

    df_merge = pd.DataFrame(columns=['flight', 'timecref'])

    for prefix in prefixes:
        if verbose:
            print(f'Merging {prefix}.')
        Fdata_keys = [x for x in Fdata.keys() if x.endswith(f'_{prefix}') and Fdata[x] is not None]
        if len(Fdata_keys) > 0:
            df_all_flights = conc_all_flights(Fdata, flight_list, prefix)
            if len(df_all_flights) > 0:
                df_merge = pd.merge(df_merge, df_all_flights, on=['flight', 'timecref'],
                                    how="outer", suffixes=('', '_copy'))
        else:
            if verbose:
                print(f'No {prefix} data found.')

    df_merge.drop(df_merge.filter(regex='_copy$').columns.tolist(), axis=1, inplace=True)
    if len(df_merge) != max_len:
        print('Merged dataframe is ' + str(len(df_merge) - max_len) +
              ' lines different than longest of the originals. Check TimeCRef.')

    return df_merge


# %%
def conc_all_flights(Fdata, flight_numbers, prefix):
    keys = [x for x in Fdata.keys() if x.endswith(prefix)]
    all_flights_found = [int(x[1:4]) for x in keys]
    flights_to_do = list(set(flight_numbers) & set(all_flights_found))

    df_all_flights = pd.DataFrame()
    flight_nos = []
    for i in range(len(flights_to_do)):
        df = Fdata[f'F{flights_to_do[i]}_{prefix}']
        if df is not None:
            flight_nos = flight_nos + len(df)*[flights_to_do[i]]
            df_all_flights = pd.concat([df_all_flights, df], sort=False)

    # df_all_flights['flight'] = flight_nos  # will append column at end of dataframe
    df_all_flights.insert(0, 'flight', flight_nos)  # insert column at front of dataframe
    df_all_flights.reset_index(drop=True, inplace=True)
    return df_all_flights


# %%
def detrend_subst(t_obs, c_obs, subst, ref_path, degree=2):
    # calculation of difference based on a polynomial fit function of degree order
    # adopted from calculate_lag from toolpac
    # will only work if reference time series can be fit with one single polynomial
    # if time series is more complicate (e.g. CH4) then rewrite to do it stepwise as in calculate_lag
    # t_ref: reference time series time
    # c_ref: reference time series mixing ratios
    # t_obs: time series of observation times
    # c_obs: observed mixing ratio time series

    start_t_obs = min(t_obs)

    if 'NOAA' in str(ref_path):
        noaa_subst=subst.replace('_', '').upper()
        if 'CFC' in noaa_subst:
            noaa_subst=noaa_subst.replace('CFC', 'F')
        elif 'HALON' in noaa_subst:
            noaa_subst=noaa_subst.replace('HALON', 'H')
        colname_base = f'{noaa_subst}catsMLO'
        ref_fname = f'mlo_{noaa_subst}_MM.dat'

        tmp = C_read.read_NOAA_ts(ref_path, ref_fname, colname_base, header=51, dropnan=True, df_return=False)

    if 'AGAGE' in str(ref_path):
        agage_subst = subst.replace('_', '-').upper()
        if subst == 'sulfuryl_fluoride':
            agage_subst = 'SO2F2'
        ref_fname = 'JFJ-medusa_mon.csv'

        tmp = C_read.read_AGAGE_ts(ref_path, ref_fname, agage_subst, header=13, dropnan=True, df_return=False)

    print(tmp)
    t_ref = tmp[0]
    c_ref = tmp[3]
    # ignore reference data earlier and later than two years before/after measurements
    # indices of array to use:
    wt = np.where(np.logical_or((t_ref > (min(t_obs) - 2.)), (t_ref < (max(t_obs) + 2.))))[0]
    t_ref = t_ref[wt]
    c_ref = c_ref[wt]

    ts_fit = np.polyfit(t_ref, c_ref, degree)
    c_fit = np.poly1d(ts_fit)

    detrend_correction = c_fit(t_obs) - c_fit(start_t_obs)
    c_obs_detr = c_obs - detrend_correction

    return c_obs_detr


# %%
def make_flight_list(df_flights, flight=None, route=None):

    # select all_flights
    if (flight is None) and (route is None):
        flight = df_flights.index.values.tolist()

    # select flights for one or more route(s)
    if flight is None:
        if type(route) == list:
            flight = []
            for i in range(len(route)):
                flight = flight + df_flights.index[df_flights['route'] == route[i]].tolist()
        else:
            flight = df_flights.index[df_flights['route'] == route].tolist()
            if len(flight) == 0:
                print('Route ', route, ' not found.')
                print(df_flights['route'].unique())

    print('Flights: ', flight)

    flight_list = []
    if type(flight) is int:  # only one flight given by number
        flight_list = [flight]
    if type(flight) is list:  # only one flight
        flight_list = flight

    return flight_list


def simple_odr_linear(xdata, ydata):
    function = odr.Model(linear)

    # remove naNs
    non_nan_index = [i for i in range(len(xdata)) if not np.isnan(xdata[i]) and not np.isnan(ydata[i])]
    mydata = odr.Data(xdata[non_nan_index], ydata[non_nan_index])
    myodr = odr.ODR(mydata, function, beta0=[1., 0.])
    out = myodr.run()
    out.pprint()
    slope = out.beta[0]
    axis = out.beta[1]

    return slope, axis


def linear(B, x):
    # Linear function y = m*x + b
    # B is a vector of the parameters.
    # x is an array of the current x values.
    # x is in the same format as the x passed to Data or RealData.
    #
    # Return an array in the same format as y passed to Data or RealData.
    return B[0]*x + B[1]


def find_nearest_sorted(array, value, idx_return=True):
    # https://stackoverflow.com/questions/2566412/find-nearest-value-in-numpy-array
    value = np.where((value > max(array)) | (value < min(array)), np.nan, value)
    idx = np.searchsorted(array, value, side="left")
    idx = idx - (np.abs(value - array[idx - 1]) < np.abs(value - array[idx]))
    if idx_return:
        return idx
    else:
        return array[idx]

