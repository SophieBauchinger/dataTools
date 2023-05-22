# -*- coding: utf-8 -*-
"""
@Author: Sophie Bauchimger, IAU
@Date: Fri Apr 28 14:13:28 2023

Defines classes used as basis for data structures
"""
import datetime as dt
import geopandas
import numpy as np
from os.path import exists
import pandas as pd
from shapely.geometry import Point
import xarray as xr

from toolpac.calc import bin_1d_2d
from toolpac.readwrite import find
from toolpac.readwrite.FFI1001_reader import FFI1001DataReader
from toolpac.conv.times import fractionalyear_to_datetime

from aux_fctns import monthly_mean, daily_mean, ds_to_gdf, rename_columns #,  same_col_merge
from dictionaries import get_col_name # , get_vlims, get_default_unit

# Note: Flights [340, 344, 346, 360, 364, 400, 422, 424, 440, 442, 444, 446, 467] have the wrong day saved

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
        self.v_limits = v_limits # colorbar normalisation limits

    def select_year(self, yr, df=None):
        """ Returns dataframe of selected year only """
        if hasattr(self, df): df = self.df
        elif df == 'None': print('Invalid operation. Cannot find a valid dataframe'); return

        try: return df[df.index.year == yr]
        except: print(f'No data found for {yr} in {self.source}'); return

    def get_data(self, c_pfxs=['GHG'], remap_lon=True,
                 mozart_file = r'C:\Users\sophie_bauchinger\sophie_bauchinger\toolpac_tutorial\RIGBY_2010_SF6_MOLE_FRACTION_1970_2008.nc',
                 verbose=False):
        """
        If Caribic: Create geopandas dataframes from data files for all available substances
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
                        self.years = np.delete(self.years, np.where(self.years==yr)) # removes current year if there's no data
                        if verbose: print(f'No data found for {yr} in {self.source}. Removing {yr} from list of years')
                        continue

                    print(f'Reading Caribic - {pfx} - {yr}')

                    # Collect data from individual flights for current year
                    df_yr = pd.DataFrame()
                    for current_dir in find.find_dir("Flight*_{}*".format(yr), parent_dir)[1:]:
                        flight_nr = int(str(current_dir)[-12:-9])

                        f = find.find_file(f'{pfx}_*', current_dir)
                        if not f or len(f)==0: # show error msg and go directly to next loop
                            if verbose: print(f'No {pfx} File found for Flight {flight_nr} in {yr}')
                            continue
                        elif len(f) > 1: f.sort() # sort list of files, then take latest (to get latest version)

                        f_data = FFI1001DataReader(f[0], df=True, xtype = 'secofday')
                        df_flight = f_data.df # index = Datetime
                        df_flight.insert(0, 'Flight number',
                                       [flight_nr for i in range(df_flight.shape[0])])

                        # for some years, substances are in lower case rather than upper. need to adjust to combine them
                        new_names, col_dict, col_dict_rev = rename_columns(df_flight.columns)
                        df_flight.rename(columns = col_dict_rev, inplace=True) # set names to the short version
                        df_yr = pd.concat([df_yr, df_flight])

                    # Convert longitude and latitude into geometry objects -> GeoDataFrame
                    geodata = [Point(lat, lon) for lon, lat in zip(
                        df_yr['LON [deg]'],
                        df_yr['LAT [deg]'])]
                    gdf_yr = geopandas.GeoDataFrame(df_yr, geometry=geodata)

                    # Drop all unnecessary columns [info is saved within datetime, geometry]
                    if not gdf_yr['geometry'].empty: # check that geometry column was filled
                        filter_cols = ['TimeCRef', 'year', 'month', 'day', 'hour', 'min', 'sec', 'lon', 'lat', 'type']
                        del_column_names = [gdf_yr.filter(regex='^'+c.upper()).columns[0] for c in filter_cols] # upper bc renamed those columns
                        gdf_yr.drop(del_column_names, axis=1, inplace=True)

                    gdf_pfx = pd.concat([gdf_pfx, gdf_yr]) # hopefully also merges names....

                if gdf_pfx.empty: print("Data extraction unsuccessful. Please check your input data"); return

                # Remove dropped columns from dictionary 
                pop_cols = [i for i in col_dict.keys() if i not in gdf_pfx.columns]
                for key in pop_cols: col_dict.pop(key)

                self.data[pfx] = gdf_pfx
                self.data[f'{pfx}_dict'] = col_dict

            return self.data

        elif self.source=='Mozart':
            with xr.open_dataset(mozart_file) as ds:
                ds = ds.isel(level=27)
            try: ds = ds.sel(time = self.years)
            except: # keep only data for specified years
                ds = xr.concat([ds.sel(time=y) for y in self.years if y in ds.time], dim='time')
                if verbose: print(f'No data found for {[y for y in self.years if y not in ds.time]} in {self.source}')
                self.years = [y for y in ds.time.values] # only include available years

            if remap_lon: # set longitudes between 180 and 360 to start at -180 towards 0
                new_lon = (((ds.longitude.data + 180) % 360) - 180)
                ds = ds.assign_coords({'longitude':('longitude', new_lon, ds.longitude.attrs)})
                ds = ds.sortby(ds.longitude) # reorganise values

            self.ds = ds
            self.df = ds_to_gdf(self.ds)
            try: self.SF6 = self.df['SF6']
            except: pass

            return ds # xr.concat(datasets, dim = 'time')

    def binned_1d(self, subs, single_yr=None, c_pfx=None):
        """
        Returns 1D binned objects for each year as lists (lat / lon)
        Parameters:
            substance (str): e.g. 'sf6'
            single_yr (int): if specified, use only data for that year [default=None]
        """
        substance = get_col_name(subs, self.source, c_pfx)

        out_x_list, out_y_list = [], []
        if single_yr is not None: years = [int(single_yr)]
        else: years = self.years

        if self.source == 'Caribic': df = self.data[c_pfx] # for Caribic, need to choose the df
        else: df = self.df 

        for yr in years: # loop through available years if possible
            try: df = df[df.index.year == yr]
            except: df = df

            if not any(df[df.index.year == yr][substance].notna()): continue # check for "empty" data array
            if df.empty: continue # go on to next year

            x = np.array([df.geometry[i].x for i in range(len(df.index))]) # lat
            y = np.array([df.geometry[i].y for i in range(len(df.index))]) # lon

            xbmin, xbmax = min(x), max(x)
            ybmin, ybmax = min(y), max(y)

            # average over lon / lat
            out_x = bin_1d_2d.bin_1d(df[substance], x,
                                     xbmin, xbmax, self.grid_size)
            out_y = bin_1d_2d.bin_1d(df[substance], y,
                                     ybmin, ybmax, self.grid_size)

            out_x_list.append(out_x); out_y_list.append(out_y)

        return out_x_list, out_y_list

    def binned_2d(self, subs, single_yr=None, c_pfx=None):
        """
        Returns 2D binned object for each year as a list
        Parameters:
            substance (str): if None, uses default substance for the object
            single_yr (int): if specified, uses only data for that year [default=None]
        """
        substance = get_col_name(subs, self.source, c_pfx)

        out_list = []
        if single_yr is not None: years = [int(single_yr)]
        else: years = self.years

        if self.source == 'Caribic': df = self.data[c_pfx] # for Caribic, need to choose the df
        else: df = self.df 

        for yr in years: # loop through available years if possible
            try: df = df[df.index.year == yr]
            except: df = df

            if not any(df[df.index.year == yr][substance].notna()): continue
            if df[substance].empty: continue

            x = np.array([df.geometry[i].x for i in range(len(df.index))]) # lat
            y = np.array([df.geometry[i].y for i in range(len(df.index))]) # lon

            xbmin, xbmax, xbsize = min(x), max(x), self.grid_size
            ybmin, ybmax, ybsize = min(y), max(y), self.grid_size

            out = bin_1d_2d.bin_2d(np.array(df[substance]), x, y,
                                   xbmin, xbmax, xbsize, ybmin, ybmax, ybsize)
            out_list.append(out)
        return out_list


class Caribic(GlobalData):
    """ CARIBIC data, plotting, averaging """

    def __init__(self, years, grid_size=5, v_limits=None, flight_nr = None, # subst='sf6', 
               pfxs=['GHG'], verbose=False):
        super().__init__([yr for yr in years if yr > 2004], grid_size, v_limits) # no caribic data before 2005, takes too long to actually check so cheesing it
        self.source = 'Caribic'; self.source_print = 'CAR'
        self.pfxs = pfxs

        # self.substance_short = subst
        # self.substance = get_col_name(self.substance_short, self.source, self.pfxs[0]) # default substance

        self.get_data(pfxs, verbose=verbose)
        if flight_nr: self.select_flight()

    def select_flight(self, flight_nr):
        """ Returns dataframe for selected flight only """
        for pfx, df in self.data.items():
            try: self.data[pfx] = df[df.values == flight_nr]
            except: print('No {pfx} data found for Flight {flight_nr}')

class Mozart(GlobalData):
    """
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
        """ Initialise MOZART object """
        super().__init__( years, grid_size, v_limits)
        self.years = years
        self.source = 'Mozart'; self.source_print = 'MZT'
        self.substance = 'SF6'
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
                self.description = f.readlines()[header_lines-2]

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
            else: time = [dt.datetime(int(y), int(m), 15) for y, m in zip(df[yr_col], df[mon_col])]
            if self.data_format == 'CATS': df = df.drop(df.iloc[:, :3], axis=1) # get rid of now unnecessary time data
            elif self.data_format == 'ccgg': 
                filter_cols = ['index', 'site_code', 'year', 'month', 'day', 'hour', 'minute', 'second', 'time_decimal', 'latitude', 'longitude', 'altitude', 'elevation', 'intake_height', 'qcflag']
                df.drop(filter_cols, axis=1, inplace=True)
                df.dropna(how='any', subset='value', inplace=True)
                unit_dic = {'co2':'[ppm]', 'ch4' : '[ppb]'}
                df.rename(columns = {'value' : f'{self.substance} {unit_dic[self.substance]}', 'value_std_dev' : f'{self.substance}_std_dev {unit_dic[self.substance]}'}, inplace=True)

            df.astype(float)
            df['Date_Time'] = time
            df.set_index('Date_Time', inplace=True) # make the datetime object the new index
            if self.data_format == 'CATS':
                try: df.dropna(how='any', subset=str(self.substance.upper()+'catsMLOm'), inplace=True)
                except: print('didnt drop NA. ', str(self.substance.upper()+'catsMLOm'))
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
        self.source = 'Mauna_Loa'; self.source_print = 'MLO'
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

        if substance in ['co2', 'ch4']:
            self.data_format = 'ccgg'
            fname = r'\{}_mlo_surface-insitu_1_ccgg_MonthlyData.txt'.format(self.substance)
            self.df = self.get_data(path_dir+fname)

class Mace_Head(LocalData):
    """ Mauna Loa data, plotting, averaging """
    def __init__(self, years=[2012], substance='sf6', data_Day = False,
                 path =  r'C:\Users\sophie_bauchinger\sophie_bauchinger\misc_data\MHD-medusa_2012.dat'):
        """ Initialise Mace Head with (daily and) monthly data in dataframes """
        super().__init__(years, data_Day, substance)
        self.years = years
        self.source = 'Mace_Head'; self.source_print = 'MHD'
        self.substance = substance

        self.df = self.get_data(path)
        self.df_Day = daily_mean(self.df)
        self.df_monthly_mean = monthly_mean(self.df)

#%% Fctn calls
if __name__=='__main__':
    c_years = np.arange(2005, 2020)
    caribic = Caribic(c_years, pfxs = ['GHG', 'INT', 'INT2'])

    mzt_years = np.arange(2000, 2020)
    mozart = Mozart(years=mzt_years)

    mlo_years = np.arange(2000, 2020)
    mlo_sf6 = Mauna_Loa(mlo_years, data_Day = True)
    mlo_n2o = Mauna_Loa(mlo_years, substance='n2o')
    mlo_co2 = Mauna_Loa(range(2000, 2010), 'co2')

    mhd = Mace_Head() # 2012
