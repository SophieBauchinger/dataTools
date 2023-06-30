# -*- coding: utf-8 -*-
"""
@Author: Sophie Bauchimger, IAU
@Date: Fri Apr 28 14:13:28 2023

Defines classes used as basis for data structures
"""
import datetime as dt
import geopandas
import numpy as np
import pandas as pd
from shapely.geometry import Point
import xarray as xr
import dill
from os.path import exists

# from toolpac.calc import bin_1d_2d
from toolpac.readwrite import find
from toolpac.readwrite.FFI1001_reader import FFI1001DataReader
from toolpac.conv.times import fractionalyear_to_datetime

from tools import monthly_mean, daily_mean, ds_to_gdf, rename_columns, bin_1d, bin_2d

#%% GLobal data
class GlobalData(object):
    """
    Global data that can be averaged on longitude / latitude grid
    Choose years, size of the grid and adjust the colormap settings
    """
    def __init__(self, years, grid_size=5, v_limits=None):
        """
        years: array or list of integers
        grid_size: int
        v_limits: tuple
        """
        self.years = years
        self.grid_size = grid_size

    def get_data(self, c_pfxs=['GHG'], remap_lon=True,
                 mozart_file = r'C:\Users\sophie_bauchinger\sophie_bauchinger\toolpac_tutorial\RIGBY_2010_SF6_MOLE_FRACTION_1970_2008.nc',
                 verbose=False):
        """
        If Caribic: Create geopandas df from data files for all available substances
            get all files starting with prefixes in c_pfxs - each in one dataframe
            lon / lat data is put into a geometry column
            Index is set to datetime of the sampling / modelled times
            a column with flight number is created

        If Mozart: Create dataset from given file
            if remap_lon, longiture is remapped to ±180 degrees
        """
        if self.source=='Caribic':
            self.data = {} # easiest way of keeping info which file the data comes from
            parent_dir = r'E:\CARIBIC\Caribic2data'

            for pfx in c_pfxs: # can include different prefixes here too
                gdf_pfx = geopandas.GeoDataFrame()
                for yr in self.years:
                    if not any(find.find_dir("*_{}*".format(yr), parent_dir)):
                        # removes current year from class attribute if there's no data
                        self.years = np.delete(self.years, np.where(self.years==yr))
                        if verbose: print(f'No data found for {yr} in {self.source}. \
                                          Removing {yr} from list of years')
                        continue

                    print(f'Reading Caribic - {pfx} - {yr}')

                    # Collect data from individual flights for current year
                    df_yr = pd.DataFrame()
                    for current_dir in find.find_dir("Flight*_{}*".format(yr), parent_dir)[1:]:
                        flight_nr = int(str(current_dir)[-12:-9])

                        f = find.find_file(f'{pfx}_*', current_dir)
                        if not f or len(f)==0: # no files found
                            if verbose: print(f'No {pfx} File found for \
                                              Flight {flight_nr} in {yr}')
                            continue
                        elif len(f) > 1: f.sort() # sort to get most recent v

                        f_data = FFI1001DataReader(f[0], df=True, xtype = 'secofday')
                        df_flight = f_data.df # index = Datetime
                        df_flight.insert(0, 'Flight number',
                                       [flight_nr for i in range(df_flight.shape[0])])

                        # for some years, substances are in lower case rather
                        # than upper. need to adjust to combine them
                        new_names, col_dict, col_dict_rev = rename_columns(
                            df_flight.columns)
                        # set names to the short version
                        df_flight.rename(columns = col_dict_rev, inplace=True)
                        df_yr = pd.concat([df_yr, df_flight])

                    # Convert longitude and latitude into geometry objects
                    geodata = [Point(lat, lon) for lon, lat in zip(
                        df_yr['LON [deg]'],
                        df_yr['LAT [deg]'])]
                    gdf_yr = geopandas.GeoDataFrame(df_yr, geometry=geodata)

                    # Drop cols which are saved within datetime, geometry
                    if not gdf_yr['geometry'].empty:
                        filter_cols = ['TimeCRef', 'year', 'month', 'day',
                                       'hour', 'min', 'sec', 'lon', 'lat', 'type']
                        # upper bc renamed those columns
                        del_column_names = [gdf_yr.filter(
                            regex='^'+c.upper()).columns[0] for c in filter_cols]
                        gdf_yr.drop(del_column_names, axis=1, inplace=True)

                    gdf_pfx = pd.concat([gdf_pfx, gdf_yr])

                if gdf_pfx.empty: print("Data extraction unsuccessful. \
                                        Please check your input data"); return

                # Remove dropped columns from dictionary
                pop_cols = [i for i in col_dict.keys() if i not in gdf_pfx.columns]
                for key in pop_cols: col_dict.pop(key)

                self.data[pfx] = gdf_pfx
                self.data[f'{pfx}_dict'] = col_dict

            self.flights = list(set(pd.concat(
                [self.data[pfx]['Flight number'] for pfx in self.pfxs])))

            return self.data

        elif self.source=='Mozart':
            with xr.open_dataset(mozart_file) as ds:
                ds = ds.isel(level=27)
            try: ds = ds.sel(time = self.years)
            except: # keep only data for specified years
                ds = xr.concat([ds.sel(time=y) for y in self.years
                                if y in ds.time], dim='time')
                if verbose: print(f'No data found for \
                                  {[y for y in self.years if y not in ds.time]} \
                                  in {self.source}')
                self.years = [y for y in ds.time.values] # only include available years

            if remap_lon: # set longitudes between 180 and 360 to start at -180 towards 0
                new_lon = (((ds.longitude.data + 180) % 360) - 180)
                ds = ds.assign_coords({'longitude':('longitude', new_lon,
                                                    ds.longitude.attrs)})
                ds = ds.sortby(ds.longitude) # reorganise values

            self.ds = ds
            self.df = ds_to_gdf(self.ds)
            try: self.SF6 = self.df['SF6']
            except: pass

            return ds # xr.concat(datasets, dim = 'time')

    def binned_1d(self, **kwargs):
        """
        Returns 1D binned objects for each year as lists (lat / lon)
        Parameters:
            substance (str): e.g. 'sf6'
            single_yr (int): if specified, use only data for that year
        """
        return bin_1d(self, **kwargs) # out_x_list, out_y_list

    def binned_2d(self, **kwargs):
        """
        Returns 2D binned object for each year as a list
        Parameters:
            substance (str): if None, uses default substance for the object
            single_yr (int): if specified, uses only data for that year
        """
        return bin_2d(self, **kwargs) # out_list

    def sel_year(self, *yr_list):
        """ Returns GlobalData object containing only data for selected years
            yr_list (int / list) """
        for yr in yr_list:
            if yr not in self.years:
                print(f'No data available for {yr}')
                yr_list = np.delete(yr_list, np.where(yr_list==yr))

        out = type(self).__new__(self.__class__) # new class instance
        for attribute_key in self.__dict__.keys(): # copy attributes
            out.__dict__[attribute_key] = self.__dict__[attribute_key]
        out.data = self.data.copy() # stops self.data being overwritten

        if self.source == 'Caribic':
            df_list = [k for k in self.data.keys()
                       if isinstance(self.data[k], pd.DataFrame)] # or Geodf
            for k in df_list: # only take data from chosen years
                out.data[k] = out.data[k][out.data[k].index.year.isin(yr_list)]
                out.data[k].sort_index(inplace=True)

            out.years = list(yr_list)
            out.flights = list(set([fl for fl in out.data[k]['Flight number']]))
            out.flights.sort()

        else:
            out.df =  out.df[out.df.index.year.isin(yr_list)].sort_index()
            out.years = yr_list

            if hasattr(out, 'ds'):
                out.ds = out.ds.sel(time=yr_list)
            if hasattr(out, 'SF6'):
                out.SF6 = out.SF6[out.SF6.index.years.isin(yr_list)].sort_index()

        out.years.sort()
        return out

    def sel_latitude(self, lat_min, lat_max):
        """ Returns GlobalData object containing only data for selected years
            yr_list (int / list) """
        # copy everything over without changing the original class instance
        out = type(self).__new__(self.__class__)
        for attribute_key in self.__dict__.keys():
            out.__dict__[attribute_key] = self.__dict__[attribute_key]
        out.data = self.data.copy()

        if self.source == 'Caribic':
            df_list = [k for k in self.data.keys()
                       if isinstance(self.data[k], pd.DataFrame)] # valid for gdf
            for k in df_list: # delete everything that isn't the chosen year
                out.data[k] = out.data[k].cx[lat_min:lat_max, -180:180]
                out.data[k].sort_index(inplace=True)

            # update available years, flights
            out.years = list(set([yr for yr in out.data[k].index.year]))
            out.flights = list(set([fl for fl in out.data[k]['Flight number']]))
            out.years.sort(); out.flights.sort()

        else:
            out.df =  out.df.query(f'latitude > {lat_min}')
            out.df =  out.df.query(f'latitude < {lat_max}')
            out.years = list(set([yr for yr in out.df.index.year]))

            if hasattr(out, 'ds'):
                out.ds = out.ds.sel(latitude=slice(lat_min, lat_max))
            if hasattr(out, 'SF6'):
                out.SF6 = out.df['SF6']

        return out

# Caribic
class Caribic(GlobalData):
    """ Stores relevant Caribic data

    Class attributes:
        pfxs (list of str): prefixes, e.g. GHG, INT, INT2
        data (dict):
            {pfx} : DataFrame
            {pfx}_dict : dictionary (col_name_now : col_name_original)
        years (list of int)
        flights (list of int)
    """

    def __init__(self, years, grid_size=5, flight_nr = None,
               pfxs=['GHG'], verbose=False):
        # no caribic data before 2005, takes too long to check so cheesing it
        super().__init__([yr for yr in years if yr > 2004], grid_size)
        self.source = 'Caribic'
        self.pfxs = pfxs
        self.get_data(pfxs, verbose=verbose)

    def sel_flight(self, *flights_list):
        """ Returns Caribic object containing only data for selected flights
            flight_list (int / list) """
        for flight_nr in flights_list:
            if flight_nr not in self.flights:
                print(f'No available data for {flight_nr}')
                flights_list = np.delete(flights_list, np.where(flights_list==flight_nr))

        out = type(self).__new__(self.__class__) # create new class instance
        for attribute_key in self.__dict__.keys(): # copy stuff like pfxs
            out.__dict__[attribute_key] = self.__dict__[attribute_key]
        # very important so that self.data doesn't get overwritten
        out.data = self.data.copy()

        df_list = [k for k in self.data.keys()
                   if isinstance(self.data[k], pd.DataFrame)] # list of all datasets to cut
        for k in df_list: # delete everything but selected flights
            out.data[k] = out.data[k][
                out.data[k]['Flight number'].isin(flights_list)]
            out.data[k].sort_index(inplace=True)

        out.flights = list(flights_list) # now only actually available flights
        out.years = list(set([yr for yr in out.data[k].index.year]))
        out.years.sort(); out.flights.sort()

        return out

# Mozart
class Mozart(GlobalData):
    """ Stores relevant Mozart data

    Class attributes:
        years: arr
        source: str
        substance: str
        ds: xarray DataFrame
        df: Pandas GeoDataFrame
        x: arr, latitude
        y: arr, longitude (remapped to +-180 deg)
    """

    def __init__(self, years, grid_size=5, v_limits=None):
        """ Initialise Mozart object """
        super().__init__(years, grid_size)
        self.years = years
        self.source = 'Mozart'; self.source_print = 'MZT'
        self.substance = 'SF6'
        self.v_limits = v_limits # colorbar normalisation limits
        self.get_data()

#%% Local data
class LocalData(object):
    """ Defines structure for ground-based station data """
    def __init__(self, years, data_Day=False, substance='sf6'):
        self.years = years
        self.substance = substance.upper()
        self.source = None

    def get_data(self, path):
        """ Create dataframe from file """
        if not exists(path): print(f'File {path} does not exists.'); return pd.DataFrame() # empty dataframe

        if self.source=='Mauna_Loa':
            header_lines = 0 # counter for lines in header
            with open(path) as f:
                for line in f:
                    if line.startswith('#'): header_lines += 1
                    else: title = line.split(); break # first non-header line has column names

            with open(path) as f: # get units from 2nd to last line of header
                if self.substance=='co': # get col names from last line
                    # print(f.readlines()[header_lines-1])
                    self.description = f.readlines()[header_lines-1]
                    title = self.description.split()[2:]
                else: self.description = f.readlines()[header_lines-2]

            mlo_data = np.genfromtxt(path, skip_header=header_lines)
            df = pd.DataFrame(mlo_data, columns=title, dtype=float)

            # get names of year and month column (depends on substance)
            if self.data_format == 'CATS':
                yr_col = [x for x in df.columns if 'catsMLOyr' in x][0]
                mon_col = [x for x in df.columns if 'catsMLOmon' in x][0]
            elif self.data_format == 'ccgg': yr_col = 'year'; mon_col = 'month'

            # keep only specified years
            df = df.loc[df[yr_col] > min(self.years)-1].loc[df[yr_col] < max(self.years)+1].reset_index()

            if any('catsMLOday' in s for s in df.columns): # check if data has day column
                day_col = [x for x in df.columns if 'catsMLOday' in x][0]
                time = [dt.datetime(int(y), int(m), int(d)) for y, m, d in zip(df[yr_col], df[mon_col], df[day_col])]
                df = df.drop(day_col, axis=1) # get rid of day column
            else: time = [dt.datetime(int(y), int(m), 15) for y, m in zip(df[yr_col], df[mon_col])] # choose middle of month for monthly data

            if self.data_format == 'CATS': df = df.drop(df.iloc[:, :3], axis=1) # get rid of now unnecessary time data
            elif self.data_format == 'ccgg' and self.substance !='co':
                filter_cols = ['index', 'site_code', 'year', 'month', 'day', 'hour', 'minute', 'second', 'time_decimal', 'latitude', 'longitude', 'altitude', 'elevation', 'intake_height', 'qcflag']
                df.drop(filter_cols, axis=1, inplace=True)
                unit_dic = {'co2':'[ppm]', 'ch4' : '[ppb]'}
                df.rename(columns = {'value' : f'{self.substance} {unit_dic[self.substance]}', 'value_std_dev' : f'{self.substance}_std_dev {unit_dic[self.substance]}'}, inplace=True)

            elif self.data_format == 'ccgg' and self.substance == 'co':
                filter_cols = ['index', 'site', 'year', 'month']
                df.drop(filter_cols, axis=1, inplace=True)
                df.dropna(how='any', subset='value', inplace=True)
                unit_dic = {'co':'[ppb]'}
                df.rename(columns = {'value' : f'{self.substance} {unit_dic[self.substance]}'}, inplace=True)

            df.astype(float)
            df['Date_Time'] = time
            df.set_index('Date_Time', inplace=True) # make the datetime object the new index
            if self.data_format == 'CATS':
                try: df.dropna(how='any', subset=str(self.substance.upper()+'catsMLOm'), inplace=True)
                except: print('didnt drop NA. ', str(self.substance.upper()+'catsMLOm'))
            if self.data_format == 'ccgg' and self.substance !='co':
                df.replace([-999.999, -999.99, -99.99, -9], np.nan, inplace=True)
                df.dropna(how='any', subset=f'{self.substance} {unit_dic[self.substance]}', inplace=True)
            return df

        elif self.source == 'Mace_Head': # make col names with space (like caribic)
            header_lines = 0
            with open(path) as f:
                for i, line in enumerate(f):
                    if line.split()[0] == 'unit:':
                        units = line.split()
                        title = list(f)[0].split() # takes next row for some reason
                        header_lines = i+2; break
            column_headers = [name + " [" + unit + "]" for name, unit in zip(title, units)] # eg. 'SF6 [ppt]'

            mhd_data = np.genfromtxt(path, skip_header=header_lines)

            df = pd.DataFrame(mhd_data, columns=column_headers, dtype=float)
            df = df.replace(0, np.nan) # replace 0 with nan for statistics
            df = df.drop(df.iloc[:, :7], axis=1) # drop unnecessary time columns
            df = df.astype(float)

            df['Date_Time'] = fractionalyear_to_datetime(mhd_data[:,0])
            df.set_index('Date_Time', inplace=True) # new index is datetime
            return df

class Mauna_Loa(LocalData):
    """ Mauna Loa data, plotting, averaging """
    def __init__(self, years, substance='sf6', data_Day = False,
                 path_dir =  r'C:\Users\sophie_bauchinger\Documents\GitHub\iau-caribic\misc_data'):
        """ Initialise Mauna Loa with (daily and) monthly data in dataframes """
        super().__init__(years, data_Day, substance)
        self.source = 'Mauna_Loa'
        self.substance = substance

        if substance in ['sf6', 'n2o']:
            self.data_format = 'CATS'
            fname_MM = r'\mlo_{}_MM.dat'.format(self.substance.upper())
            self.df = self.get_data(path_dir+fname_MM)

            self.df_monthly_mean = self.df_Day = pd.DataFrame() # create empty df
            if data_Day: # user input saying if daily data should exist
                fname_Day = r'\mlo_{}_Day.dat'.format(self.substance.upper())
                self.df_Day = self.get_data(path_dir + fname_Day)
                try: self.df_monthly_mean = monthly_mean(self.df_Day)
                except: pass

        if substance in ['co2', 'ch4', 'co']:
            self.data_format = 'ccgg'
            fname = r'\{}_mlo_surface-insitu_1_ccgg_MonthlyData.txt'.format(
                self.substance)
            if substance=='co': fname = r'\co_mlo_surface-flask_1_ccgg_month.txt'
            self.df = self.get_data(path_dir+fname)

class Mace_Head(LocalData):
    """ Mauna Loa data, plotting, averaging """
    def __init__(self, years=[2012], substance='sf6', data_Day = False,
                 path =  r'C:\Users\sophie_bauchinger\sophie_bauchinger\misc_data\MHD-medusa_2012.dat'):
        """ Initialise Mace Head with (daily and) monthly data in dataframes """
        super().__init__(years, data_Day, substance)
        self.years = years
        self.source = 'Mace_Head'
        self.substance = substance

        self.df = self.get_data(path)
        self.df_Day = daily_mean(self.df)
        self.df_monthly_mean = monthly_mean(self.df)



#%% Fctn calls
if __name__=='__main__':
    year_range = np.arange(2000, 2020)

    # only calculate caribic if necessary
    calc_c = False
    if calc_c and exists('caribic_dill.pkl'): # Avoid long file loading times
        with open('caribic_dill.pkl', 'rb') as f: caribic = dill.load(f)
        del f
    elif calc_c: caribic = Caribic(year_range, pfxs = ['GHG', 'INT', 'INT2'])

    mozart = Mozart(year_range)

    mlo_sf6 = Mauna_Loa(year_range, data_Day = True)
    mlo_n2o = Mauna_Loa(year_range, substance='n2o')
    mlo_co2 = Mauna_Loa(year_range, 'co2')

    mhd = Mace_Head() # 2012
