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
from os.path import exists
import matplotlib.pyplot as plt
from functools import partial

# from toolpac.calc import bin_1d_2d
from toolpac.readwrite import find
from toolpac.readwrite.FFI1001_reader import FFI1001DataReader
from toolpac.conv.times import fractionalyear_to_datetime
from toolpac.outliers import outliers
from toolpac.conv.times import datetime_to_fractionalyear

from dictionaries import get_col_name, substance_list, get_fct_substance
from tools import monthly_mean, daily_mean, ds_to_gdf, rename_columns, bin_1d, bin_2d, coord_merge_substance, coord_combo, EMAC_vars_filter
from tropFilter import chemical, dynamical, thermal
# from detrend import detrend_substance

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
        self.status = {} # use this dict to keep track of changes made to data

    def binned_1d(self, subs, **kwargs):
        """
        Returns 1D binned objects for each year as lists (lat / lon)
        Parameters:
            substance (str): e.g. 'sf6'
            single_yr (int): if specified, use only data for that year
        """
        return bin_1d(self, subs, **kwargs) # out_x_list, out_y_list

    def binned_2d(self, subs, **kwargs):
        """
        Returns 2D binned object for each year as a list
        Parameters:
            substance (str): if None, uses default substance for the object
            single_yr (int): if specified, uses only data for that year
        """
        return bin_2d(self, subs, **kwargs) # out_list

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
        """ Returns GlobalData object containing only data for selected latitudes """
        # copy everything over without changing the original class instance
        out = type(self).__new__(self.__class__)
        for attribute_key in self.__dict__.keys():
            out.__dict__[attribute_key] = self.__dict__[attribute_key]
        out.data = self.data.copy()

        if self.source == 'Caribic':
            df_list = [k for k in self.data.keys()
                       if isinstance(self.data[k], pd.DataFrame)] # valid for gdf
            for k in df_list: # delete everything that isn't the chosen lat range
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

    def sel_eqlat(self, eql_min, eql_max, model='ERA5'):
        """ Returns GlobalData object containing only data for selected equivalent latitudes """
        # copy everything over without changing the original class instance
        out = type(self).__new__(self.__class__)
        for attribute_key in self.__dict__.keys():
            out.__dict__[attribute_key] = self.__dict__[attribute_key]
        out.data = self.data.copy()

        if self.source != 'Caribic': raise Exception('Not implemented')
        else:
            if model == 'ERA5': eql_col = 'int_ERA5_EQLAT [deg N]'
            elif model == 'ECMWF': eql_col = 'int_eqlat [deg]'

            df = self.met_data.copy()
            df = df[df[eql_col] > eql_min]
            df = df[df[eql_col] < eql_max]
            self.met_data = df

            df_list = [k for k in self.data.keys()
                       if isinstance(self.data[k], pd.DataFrame)] # all dataframes

            for k in df_list: # delete everything outside eql range
                out.data[k] = out.data[k][out.data[k].index.isin(df.index)]

        # caribic.data['INT2'][caribic.data['INT2']['int_ERA5_EQLAT [deg N]'] > 10].dropna(subset='int_ERA5_EQLAT [deg N]')

        return out

# Caribic
class CaribicData(GlobalData):
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
        self.get_data(verbose=verbose) # creates self.data dictionary
        self.met_data = coord_combo(self) # reference for met data for all msmts

    def get_data(self, verbose=False):
        """
        If Caribic: Create geopandas df from data files for all available substances
            get all files starting with prefixes in c_pfxs - each in one dataframe
            lon / lat data is put into a geometry column
            Index is set to datetime of the sampling / modelled times
            a column with flight number is created

        If Mozart: Create dataset from given file
            if remap_lon, longiture is remapped to ±180 degrees
        """
        self.data = {} # easiest way of keeping info which file the data comes from
        parent_dir = r'E:\CARIBIC\Caribic2data'

        for pfx in self.pfxs: # can include different prefixes here too
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

                    f_data = FFI1001DataReader(f[-1], df=True, xtype = 'secofday')
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
                if pfx=='INT': gdf_pfx.drop(columns=['int_acetone [ppt]',
                                 'int_acetonitrile [ppt]'], inplace=True)
                if pfx=='INT2': gdf_pfx.drop(columns=['int_CARIBIC2_Ac [pptV]',
                                 'int_CARIBIC2_AN [pptV]'], inplace=True)

            if gdf_pfx.empty: print("Data extraction unsuccessful. \
                                    Please check your input data"); return

            # Remove dropped columns from dictionary
            pop_cols = [i for i in col_dict.keys() if i not in gdf_pfx.columns]
            for key in pop_cols: col_dict.pop(key)

            self.data[pfx] = gdf_pfx
            self.data[f'{pfx}_dict'] = col_dict

        self.flights = list(set(pd.concat(
            [self.data[pfx]['Flight number'] for pfx in self.pfxs])))

        self.data

class Caribic(CaribicData):
    def __init__(self, years, grid_size=5, flight_nr = None, pfxs=['GHG'], 
                 verbose=False):
        super().__init__(years, grid_size, flight_nr, pfxs, verbose)

        for subs in ['sf6', 'n2o', 'co2', 'ch4']: 
            self.create_substance_df(subs)

    def sel_flight(self, flights, verbose=False):
        """ Returns Caribic object containing only data for selected flights
            flight_list (int / list) """
        if isinstance(flights, int): flights = [flights]
        # elif isinstance(flights, range): flights = list(flights)
        invalid = [f for f in flights if f not in self.flights]
        if len(invalid)>0 and verbose:
            print(f'No data found for flights {invalid}. Proceeding without.')
        flights = [f for f in flights if f in self.flights]

        out = type(self).__new__(self.__class__) # create new class instance
        for attribute_key in self.__dict__.keys(): # copy stuff like pfxs
            out.__dict__[attribute_key] = self.__dict__[attribute_key]
        # very important so that self.data doesn't get overwritten
        out.data = self.data.copy()

        df_list = [k for k in self.data.keys()
                   if isinstance(self.data[k], pd.DataFrame)] # list of all datasets to cut
        for k in df_list: # delete everything but selected flights
            out.data[k] = out.data[k][
                out.data[k]['Flight number'].isin(flights)]
            out.data[k].sort_index(inplace=True)

        out.flights = flights # update to chosen & available flights
        out.years = list(set([yr for yr in out.data[k].index.year]))
        out.years.sort(); out.flights.sort()

        return out

    def sel_tropo(self, **kwargs):
        """ Returns Caribic object containing only tropospheric data points.
        Parameters:
            filter_type (str): 'chem', 'therm' or 'dyn'

            crit (str): 'n2o', 'o3'
            coord (str): 'pt', 'dp', 'z'
            pvu (float): 1.5, 2.0, 3.5
            limit (float): pre-flag limit for chem. TP sorting

            subs (str): substance for plotting
            c_pfx (str): 'GHG', 'INT', 'INT2'
            verbose (bool)
            plot (bool)
        """
        out = type(self).__new__(self.__class__) # create new class instance
        for attribute_key in self.__dict__.keys(): # copy stuff like pfxs
            out.__dict__[attribute_key] = self.__dict__[attribute_key]

        if not 'c_pfx' in kwargs.keys(): 
            print('Using INT2 Data for filtering'); kwargs['c_pfx'] = 'INT2'

        out.data = {k:v.copy() for k,v in self.data.items() if isinstance(v, pd.DataFrame)} # only using OG msmt data
        functions = {'chem' : chemical, 'dyn' : dynamical, 'therm' : thermal}

        df_sorted = functions[kwargs.get('tp_def')](out, **kwargs)

        for df_key in out.data.keys():
            col = [col for col in df_sorted.columns if col.startswith('tropo')][0]
            # only keep rows that are in df_sorted, then only tropospheric data
            out.data[df_key] = out.data[df_key][out.data[df_key].index.isin(df_sorted.index)]
            out.data[df_key][col] = df_sorted[col] # copy to keep track of tp def
            out.data[df_key] = out.data[df_key][out.data[df_key][col]]

        out.pfxs = [k for k in out.data.keys()]
        return out

    def sel_strato(self, **kwargs):
        """ Returns Caribic object containing only tropospheric data points.
        Parameters:
            filter_type (str): 'chem', 'therm' or 'dyn'

            crit (str): 'n2o', 'o3'
            coord (str): 'pt', 'dp', 'z'
            pvu (float): 1.5, 2.0, 3.5
            limit (float): pre-flag limit for chem. TP sorting

            subs (str): substance for plotting
            c_pfx (str): 'GHG', 'INT', 'INT2'
            verbose (bool)
            plot (bool)
        """
        out = type(self).__new__(self.__class__) # create new class instance
        for attribute_key in self.__dict__.keys(): # copy stuff like pfxs
            out.__dict__[attribute_key] = self.__dict__[attribute_key]

        out.data = {k:v.copy() for k,v in self.data.items() if k in self.pfxs} # only using OG msmt data
        functions = {'chem' : chemical, 'dyn' : dynamical, 'therm' : thermal}

        if not 'c_pfx' in kwargs.keys(): pfxs = set(out.data.keys())
        else: pfxs = [kwargs['c_pfx']]; del kwargs['c_pfx']

        for pfx in pfxs:
            try: df_sorted = functions[kwargs.get('tp_def')](out, c_pfx=pfx, **kwargs)
            except: # remove pfxs that can't be sorted
                print(f'Sorting of {pfx} with selected TP definition unsuccessful')
                del out.data[pfx]; continue

            col = [col for col in df_sorted.columns if col.startswith('strato')][0]
            # only keep rows that are in df_sorted, then only tropospheric data
            out.data[pfx] = out.data[pfx][out.data[pfx].index.isin(df_sorted.index)]
            out.data[pfx][col] = df_sorted[col]
            out.data[pfx] = out.data[pfx][out.data[pfx][col]]

        out.pfxs = [k for k in out.data.keys()]
        return out

    def filter_extreme_events(self, **kwargs):
        """ Filter out all tropospheric extreme events.

        Returns new Caribic object where trop. extreme events have been removed.
        Result depends on tropopause definition for trop / strat sorting.

        Parameters:
            filter_type (str): 'chem', 'therm' or 'dyn'

            crit (str): 'n2o', 'o3'
            coord (str): 'pt', 'dp', 'z'
            pvu (float): 1.5, 2.0, 3.5
            limit (float): pre-flag limit for chem. TP sorting

            subs (str): substance for plotting
            c_pfx (str): 'GHG', 'INT', 'INT2'
            verbose (bool)
            plot (bool)
        """
        out = type(self).__new__(self.__class__) # create new class instance
        for attribute_key in self.__dict__.keys(): # copy stuff like pfxs
            out.__dict__[attribute_key] = self.__dict__[attribute_key]

        # Find and filter tropospheric extreme events
        tp_def = kwargs.get('tp_def')
        if tp_def == 'chem' and 'ref_obj' not in kwargs.keys(): 
            kwargs['ref_obj'] = Mauna_Loa(self.years, 'n2o')
        if tp_def in ['dyn', 'therm'] and 'c_pfx' not in kwargs.keys():
            raise Exception('Please supply a pfx to choose data source for TP sorting.')
        tropo_obj = self.sel_tropo(**kwargs)
        out.pfxs = tropo_obj.pfxs # only take trop. sorted pfx data
        out.data = {k:v.copy() for k,v in self.data.items() if k in tropo_obj.pfxs} # only using OG msmt data

        #!!! use subs data if available. Need to make a decision on which subs data to use though ? 

        for pfx in tropo_obj.pfxs:
            data = tropo_obj.data[pfx].sort_index()
            if 'subs' in kwargs.keys() and isinstance(kwargs['subs'], str): subs_list = [kwargs['subs']]
            elif 'subs' in kwargs.keys() and isinstance(kwargs['subs'], list): subs_list = kwargs['subs']
            else: subs_list = substance_list(pfx)

            for subs in subs_list:
                try: substance = get_col_name(subs, tropo_obj.source, pfx)
                except: substance = None
                if substance not in data.columns: continue
                time = np.array(datetime_to_fractionalyear(data.index, method='exact'))
                mxr = data[substance].tolist()
                if f'd_{substance}' in data.columns:
                    d_mxr = data[f'd_{substance}'].tolist()
                else: d_mxr = None # integrated values of high resolution data

                func = get_fct_substance(subs)
                # Find extreme events
                tmp = outliers.find_ol(func, time, mxr, d_mxr, flag=None, # here
                                       direction='p', verbose=False,
                                       plot=False, limit=0.1, ctrl_plots=False)

                subs_cols = [c for c in data.columns if substance in c] # detrended etc

                # Set rows that were flagged as extreme events to 9999, then nan
                for c in subs_cols:
                    data.loc[(flag != 0 for flag in tmp[0]), c] = 9999
                out.data[pfx].update(data) # essential to update before setting to nan
                out.data[pfx].replace(9999, np.nan, inplace=True)

            # delete unfiltered data
            drop_subs = [subs for subs in substance_list(pfx) if subs not in subs_list]
            drop_cols = [get_col_name(subs, out.source, pfx) for subs in drop_subs]
            out.data[pfx].drop(columns = drop_cols, inplace=True)

        # also filter / create the substance dataframes (not elegantly)
        for subs in ['sf6', 'n2o', 'co2', 'ch4']:
            if subs in self.data.keys() and 'subs_pfx' in kwargs.keys():
                data = out.data[subs] = self.data[subs].copy()
                substance = get_col_name(subs, tropo_obj.source, kwargs['subs_pfx'])

                time = np.array(datetime_to_fractionalyear(data.index, method='exact'))
                mxr = data[substance].tolist()
                if f'd_{substance}' in data.columns:
                    d_mxr = data[f'd_{substance}'].tolist()
                else: d_mxr = None # integrated values of high resolution data

                func = get_fct_substance(subs)
                # Find extreme events
                tmp = outliers.find_ol(func, time, mxr, d_mxr, flag=None, # here
                                       direction='p', verbose=False,
                                       plot=False, limit=0.1, ctrl_plots=False)
                subs_cols = [c for c in data.columns if substance in c] # include detr etc
                for c in subs_cols:
                    data.loc[(flag != 0 for flag in tmp[0]), substance] = 9999
                out.data[subs].update(data)
                out.data[subs].replace(9999, np.nan, inplace=True)

            elif subs in self.data.keys():
                try: out.create_substance_df(subs)
                except: print(f'Failed trying to create substance df for {subs}')

        return out

    def detrend(self, subs, **kwargs):
        """ Remove linear trend for the substance & add detr. data to all dataframes """
        detrend_substance(self, subs, **kwargs)

    def create_substance_df(self, subs):
        """ Create dataframe containing all met.+ msmt. data for a substance """
        self.data[f'{subs}'] = coord_merge_substance(self, subs)

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
        self.source = 'Mozart'
        self.substance = 'SF6'
        self.v_limits = v_limits # colorbar normalisation limits
        self.get_data()

    def get_data(self, remap_lon=True, verbose=False,
                 fname = r'C:\Users\sophie_bauchinger\sophie_bauchinger\toolpac_tutorial\RIGBY_2010_SF6_MOLE_FRACTION_1970_2008.nc'):
        """ Create dataset from given file
        
        if remap_lon, longitude is remapped to ±180 degrees
        """
        with xr.open_dataset(fname) as ds:
            ds = ds.isel(level=27)
        try: ds = ds.sel(time = self.years)
        except: # keep only data for specified years
            ds = xr.concat([ds.sel(time=y) for y in self.years
                            if y in ds.time], dim='time')
            if verbose: print(f'No data found for \
                              {[y for y in self.years if y not in ds.time]} \
                              in {self.source}')
            self.years = [y for y in ds.time.values] # only include act. available years

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

class EMAC(GlobalData):
    def __init__(self, years=[y for y in range(2000, 2020)], 
                 interp='n', pdir=None, subsam=False):
        super().__init__(years)
        self.years = years
        self.source = 'EMAC'


        self.get_data(years, interp, pdir, subsam)

    def get_data(self, years, interp, pdir, subsam):
        """ Extract s4d EMAC Data from netCDF files for Caribic """
        if subsam: 
            if pdir is None: pdir = 'E:/MODELL/EMAC/TPChange/s4d_subsam_CARIBIC/'
            fnames = pdir +  f'*{interp}CARIB2_s.nc'
        else: 
            if pdir is None: pdir = 'E:/MODELL/EMAC/TPChange/s4d_CARIBIC/'
            fnames = pdir + f'*{interp}CARIB2.nc'

        with xr.open_mfdataset(fnames, preprocess=partial(EMAC_vars_filter)) as dataset:
            ds = dataset

        ds.rename({'tlon':'longitude', 'tlat':'latitude'})

        self.ds = ds

    def make_df(self):
        self.df = self.ds.to_dataframe()

# =============================================================================
# if __name__=='__main__': test = EMAC()
# =============================================================================

def single_file():
    parent_dir = 'E:/MODELL/EMAC/TPChange/s4d_CARIBIC/'
    fnames = parent_dir + '200506_s4d_nCARIB2.nc' # f'*{interp}CARIB2.nc'
    with xr.open_mfdataset(fnames, preprocess=partial(EMAC_vars_filter)) as ds_mon:
        return ds_mon

# =============================================================================
# if __name__=='__main__': ds_filtered = single_file()
# =============================================================================

class EMAC_subsam(EMAC):
    def __init__(self, years):
        super().__init__(years)
        self.years = years
        self.source = 'EMAC'
        
        self.get_data()
        
    def get_data(self, years, interp='n',
                 parent_dir = r'E:\MODELL\EMAC\TPChange\s4d_subsam_CARIBIC'):
        """ Extract s4d subsampled EMAC Data from netCDF files for Caribic """

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
    def __init__(self, years, subs='sf6', data_Day = False,
                 path_dir =  r'C:\Users\sophie_bauchinger\Documents\GitHub\iau-caribic\misc_data'):
        """ Initialise Mauna Loa with (daily and) monthly data in dataframes """
        super().__init__(years, data_Day, subs)
        self.source = 'Mauna_Loa'
        self.substance = subs

        if subs in ['sf6', 'n2o']:
            self.data_format = 'CATS'
            fname_MM = r'\mlo_{}_MM.dat'.format(self.substance.upper())
            self.df = self.get_data(path_dir+fname_MM)

            self.df_monthly_mean = self.df_Day = pd.DataFrame() # create empty df
            if data_Day: # user input saying if daily data should exist
                fname_Day = r'\mlo_{}_Day.dat'.format(self.substance.upper())
                self.df_Day = self.get_data(path_dir + fname_Day)
                try: self.df_monthly_mean = monthly_mean(self.df_Day)
                except: pass

        elif subs in ['co2', 'ch4', 'co']:
            self.data_format = 'ccgg'
            fname = r'\{}_mlo_surface-insitu_1_ccgg_MonthlyData.txt'.format(
                self.substance)
            if subs=='co': fname = r'\co_mlo_surface-flask_1_ccgg_month.txt'
            self.df = self.get_data(path_dir+fname)

        else: raise KeyError(f'Mauna Loa data not available for {subs.upper()}')

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


#%% detrending 
def detrend_substance(c_obj, subs, loc_obj=None, degree=2, save=True, plot=False,
                      as_subplot=False, ax=None, c_pfx=None, note=''):
    """ Remove linear trend of substances using free troposphere as reference.
    (redefined from C_tools.detrend_subs)

    Parameters:
        c_obj (GlobalData/Caribic)
        subs (str): substance to detrend e.g. 'sf6'
        loc_obj (LocalData): free troposphere data, defaults to Mauna_Loa
    """
    if loc_obj is None: 
        try: loc_obj = Mauna_Loa(c_obj.years, subs)
        except: raise ValueError(f'Cannot detrend as ref. data could not be found for {subs.upper()}') 
    out_dict = {}

    if c_pfx: pfxs = [c_pfx]
    else: pfxs = [pfx for pfx in c_obj.pfxs if subs in substance_list(pfx)]

    if plot:
        if not as_subplot:
            fig, axs = plt.subplots(len(pfxs), dpi=250, figsize=(6,4*len(pfxs)))
            if len(pfxs)==1: axs = [axs]
        elif ax is None:
            ax = plt.gca()

    for c_pfx, i in zip(pfxs, range(len(pfxs))):
        df = c_obj.data[c_pfx]
        substance = get_col_name(subs, c_obj.source, c_pfx)
        if substance is None: continue

        c_obs = df[substance].values
        t_obs =  np.array(datetime_to_fractionalyear(df.index, method='exact'))

        ref_df = loc_obj.df
        ref_subs = get_col_name(subs, loc_obj.source)
        if ref_subs is None: raise ValueError(f'No reference data found for {subs}')
        # ignore reference data earlier and later than two years before/after msmts
        ref_df = ref_df[min(df.index)-dt.timedelta(356*2)
                        : max(df.index)+dt.timedelta(356*2)]
        ref_df.dropna(how='any', subset=ref_subs, inplace=True) # remove NaN rows
        c_ref = ref_df[ref_subs].values
        t_ref = np.array(datetime_to_fractionalyear(ref_df.index, method='exact'))

        popt = np.polyfit(t_ref, c_ref, degree)
        c_fit = np.poly1d(popt) # get popt, then make into fct

        detrend_correction = c_fit(t_obs) - c_fit(min(t_obs))
        c_obs_detr = c_obs - detrend_correction
        # get variance (?) by substracting offset from 0
        c_obs_delta = c_obs_detr - c_fit(min(t_obs))

        df_detr = pd.DataFrame({f'detr_{substance}' : c_obs_detr,
                                 f'delta_{substance}' : c_obs_delta,
                                 f'detrFit_{substance}' : c_fit(t_obs)}, 
                                index = df.index)
        # maintain relationship between detr and fit columns
        df_detr[f'detrFit_{substance}'] = df_detr[f'detrFit_{substance}'].where(
            ~df_detr[f'detr_{substance}'].isnull(), np.nan)

        out_dict[f'detr_{c_pfx}_{subs}'] = df_detr
        out_dict[f'popt_{c_pfx}_{subs}'] = popt

        if save:
            columns = [f'detr_{substance}', 
                       f'delta_{substance}', 
                       f'detrFit_{substance}']
            
            c_obj.data[c_pfx][columns] = df_detr[columns]
            # move geometry column to the end again
            c_obj.data[c_pfx]['geometry'] =  c_obj.data[c_pfx].pop('geometry')

        if plot:
            if not as_subplot: ax = axs[i]
            # ax.annotate(f'{c_pfx} {note}', xy=(0.025, 0.925), xycoords='axes fraction',
            #                       bbox=dict(boxstyle="round", fc="w"))
            ax.scatter(df_detr.index, c_obs, color='orange', label='Flight data', marker='.')
            ax.scatter(df_detr.index, c_obs_detr, color='green', label='trend removed', marker='.')
            ax.scatter(ref_df.index, c_ref, color='gray', label='MLO data', alpha=0.4, marker='.')
            ax.plot(df_detr.index, c_fit(t_obs), color='black', ls='dashed',
                      label='trendline')
            ax.set_ylabel(f'{substance}') # ; ax.set_xlabel('Time')
            handles, labels = ax.get_legend_handles_labels()
            leg = ax.legend(title=f'{c_pfx} {note}')
            leg._legend_box.align = "left"

    if plot and not as_subplot:
        fig.tight_layout()
        fig.autofmt_xdate()
        plt.show()

    return out_dict
