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

        self.substance = None
        self.substances = None # list of substances to use for plotting etc
        self.source = None

    def select_year(self, yr):
        """ Returns dataframe of selected year only """
        try: return self.df[self.df.index.year == yr]
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
                        new_names, col_dict = rename_columns(df_flight.columns)
                        df_flight.rename(columns = col_dict, inplace=True) # set names to the short version
                        df_yr = pd.concat([df_yr, df_flight])

                    print(df_yr.columns())
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
                print(col_dict.keys())
                pop_cols = [i for i in col_dict.keys() if i not in gdf_pfx.columns]
                print(pop_cols)
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

    def binned_1d(self, substance=None, single_yr=None, c_pfx=None):
        """
        Returns 1D binned objects for each year as lists (lat / lon)
        Parameters:
            substance (str): if None, use default substance for the object
            single_yr (int): if specified, use only data for that year [default=None]
        """
        out_x_list, out_y_list = [], []
        if substance is None: substance = self.substance
        if single_yr is not None: years = [int(single_yr)]
        else: years = self.years

        if c_pfx is not None: df = self.data[c_pfx] # for Caribic, need to choose the df
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

    def binned_2d(self, substance=None, single_yr=None, c_pfx=None):
        """
        Returns 2D binned object for each year as a list
        Parameters:
            substance (str): if None, uses default substance for the object
            single_yr (int): if specified, uses only data for that year [default=None]
        """
        out_list = []
        if substance is None: substance = self.substance
        if single_yr is not None: years = [int(single_yr)]
        else: years = self.years

        if c_pfx is not None: df = self.data[c_pfx] # for Caribic, need to choose the df
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
            yr_col = [x for x in df.columns if 'catsMLOyr' in x][0]
            mon_col = [x for x in df.columns if 'catsMLOmon' in x][0]

            # keep only specified years
            df = df.loc[df[yr_col] > min(self.years)-1].loc[df[yr_col] < max(self.years)+1].reset_index()

            if any('catsMLOday' in s for s in df.columns): # check if data has day column
                day_col = [x for x in df.columns if 'catsMLOday' in x][0]
                time = [dt.datetime(int(y), int(m), int(d)) for y, m, d in zip(df[yr_col], df[mon_col], df[day_col])]
                df = df.drop(day_col, axis=1) # get rid of day column
            else: time = [dt.datetime(int(y), int(m), 15) for y, m in zip(df[yr_col], df[mon_col])]
            df = df.drop(df.iloc[:, :3], axis=1) # get rid of now unnecessary time data
            df.astype(float)
            df['Date_Time'] = time
            df.set_index('Date_Time', inplace=True) # make the datetime object the new index

            try: df.dropna(how='any', subset=str(self.substance.upper()+'catsMLOm'), inplace=True)
            except: print('didnt drop na. ', str(self.substance.upper()+'catsMLOm'))
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
        self.years = years
        self.source = 'Mauna_Loa'; self.source_print = 'MLO'
        self.substance = substance

        fname_MM = r'\mlo_{}_MM.dat'.format(self.substance.upper())
        self.df = self.get_data(path_dir+fname_MM)


        self.df_monthly_mean = self.df_Day = pd.DataFrame() # create empty df
        if data_Day: # user input saying if daily data should exist
            fname_Day = r'\mlo_{}_Day.dat'.format(self.substance.upper())
            self.df_Day = self.get_data(path_dir + fname_Day)
            try: self.df_monthly_mean = monthly_mean(self.df_Day)
            except: pass

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
    # caribic = Caribic(c_years)
    caribic = Caribic(c_years, pfxs = ['GHG', 'INT', 'INT2'])
    # caribic_int = Caribic(c_years, subst='co2', pfxs = ['INT'])
    # caribic_int2 = Caribic(c_years, subst='n2o', pfxs = ['INT2'])

    mzt_years = np.arange(2000, 2020)
    mozart = Mozart(years=mzt_years)

    mlo_years = np.arange(2000, 2020)
    mlo = Mauna_Loa(mlo_years, data_Day = True)
    mlo_n2o = Mauna_Loa(mlo_years, substance='n2o')

    mhd = Mace_Head() # 2012

#%%
    # def plot_scatter(self, substance=None, single_yr=None):
    #     if self.source=='Caribic':
    #         if substance is None: substance = self.substance
    #         if single_yr is not None:
    #             df = self.select_year(single_yr)
    #             df_mm = monthly_mean(df).notna()
    #         else: df = self.df; df_mm = self.df_monthly_mean

    #         # Plot mixing ratio msmts and monthly mean
    #         fig, ax = plt.subplots(dpi=250)
    #         plt.title(f'{self.source} {self.substance_short.upper()} measurements')
    #         if hasattr(self, 'pfxs'): plt.title(f'{self.source} {self.substance_short.upper()} measurements {self.pfxs}')
    #         ymin = np.nanmin(df[substance])
    #         ymax = np.nanmax(df[substance])

    #         cmap = plt.cm.viridis_r
    #         extend = 'neither'
    #         if self.v_limits: vmin, vmax = self.v_limits# ; extend = 'both'
    #         else: vmin = ymin; vmax = ymax
    #         norm = Normalize(vmin, vmax)

    #         plt.scatter(df.index, df[substance],
    #                     # label=f'{self.substance_short.upper()} {self.years}',
    #                     marker='x', zorder=1,
    #                     c = df[substance],
    #                     cmap = cmap, norm = norm)
    #         for i, mean in enumerate(df_mm[substance]):
    #             y,m = df_mm.index[i].year, df_mm.index[i].month
    #             xmin, xmax = dt.datetime(y, m, 1), dt.datetime(y, m, monthrange(y, m)[1])
    #             ax.hlines(mean, xmin, xmax, color='black',
    #                       linestyle='dashed', zorder=2)
    #         plt.colorbar(sm(norm=norm, cmap=cmap), aspect=50, ax = ax, extend=extend)
    #         plt.ylabel(f'{substance}')
    #         plt.ylim(ymin-0.15, ymax+0.15)
    #         fig.autofmt_xdate()

    #         plt.show() # for some reason there's a matplotlib user warning here: converting a masked element to nan. xys = np.asarray(xys)

    #     elif self.source=='Mozart':
    #         self.plot_1d(substance, single_yr)

    # def plot_1d(self, substance=None, single_yr=None, plot_mean=False, single_graph=False):
    #     """
    #     Plots 1D averaged values over latitude / longitude including colormap
    #     Parameters:
    #         substance (str): if None, plots default substance for the object
    #         single_yr (int): if specified, plots only data for that year [default=None]
    #         plot_mean (bool): choose whether to plot the overall average over all years
    #         single_graph (bool): choose whether to plot all years on one graph
    #     """
    #     if substance is None: substance = self.substance
    #     if single_yr is not None: years = [int(single_yr)]
    #     else: years = self.years

    #     out_x_list, out_y_list = self.binned_1d(substance, single_yr)

    #     if not single_graph:
    #         # Plot mixing ratios averages over lats / lons for each year separately
    #         for out_x, out_y, year in zip(out_x_list, out_y_list, years):
    #             fig, ax = plt.subplots(dpi=300, ncols=2, sharey=True, figsize=(8,3.5))
    #             fig.suptitle('{} {} modeled SF$_6$ concentration. Gridsize={}'.format(
    #                 self.source, year, self.grid_size))

    #             cmap = plt.cm.viridis_r
    #             if self.v_limits: vmin, vmax = self.v_limits
    #             else:
    #                 vmin = min([np.nanmin(out_x.vmean), np.nanmin(out_y.vmean)])
    #                 vmax = max([np.nanmin(out_x.vmean), np.nanmin(out_y.vmean)])
    #             norm = Normalize(vmin, vmax) # allows mapping colormap onto available values

    #             ax[0].plot(out_x.xintm, out_x.vmean, zorder=1, color='black', lw = 0.5)
    #             ax[0].scatter(out_x.xintm, out_x.vmean, # plot across latitude
    #                           c = out_x.vmean, cmap = cmap, norm = norm, zorder=2)
    #             ax[0].set_xlabel('Latitude [deg]'); plt.xlim(out_x.xbmin, out_x.xbmax)
    #             ax[0].set_ylabel('Mean SF$_6$ mixing ratio [ppt]')

    #             ax[1].plot(out_y.xintm, out_y.vmean, zorder=1, color='black', lw = 0.5)
    #             ax[1].scatter(out_y.xintm, out_y.vmean, # plot across longitude
    #                           c = out_y.vmean, cmap = cmap, norm = norm, zorder=2)
    #             ax[1].set_xlabel('Longitude [deg]'); plt.xlim(out_y.xbmin, out_y.xbmax)
    #             ax[1].set_ylabel('Mean SF$_6$ mixing ratio [ppt]')

    #             fig.colorbar(sm(norm=norm, cmap=cmap), aspect=50, ax = ax[1])
    #             plt.show()

    #     if single_graph:
    #         # Plot averaged mixing ratios for all years on one graph
    #         fig, ax = plt.subplots(dpi=300, ncols=2, sharey=True, figsize=(8,3.5))
    #         fig.suptitle(f'{self.source} {self.years[0]} - {self.years[-1]} modeled {substance} mixing ratio. Gridsize={self.grid_size}')

    #         cmap = cm.get_cmap('plasma_r')
    #         vmin, vmax = self.years[0], self.years[-1]
    #         norm = Normalize(vmin, vmax)

    #         for out_x, out_y, year in zip(out_x_list, out_y_list, self.years): # add each year to plot
    #             ax[0].plot(out_x.xintm, out_x.vmean, label=year)#, c = cmap(norm(year)))
    #             ax[0].set_xlabel('Latitude [deg]'); plt.xlim(out_x.xbmin, out_x.xbmax)
    #             ax[0].set_ylabel(f'Mean {substance} mixing ratio [ppt]')

    #             ax[1].plot(out_y.xintm, out_y.vmean, label=year)# , c = cmap(norm(year)))
    #             ax[1].set_xlabel('Longitude [deg]'); plt.xlim(out_y.xbmin, out_y.xbmax)
    #             ax[1].set_ylabel(f'Mean {substance} mixing ratio [ppt]')

    #         if plot_mean: # add average over available years to plot
    #             total_x_vmean = np.mean([i.vmean for i in out_x_list], axis=0)
    #             total_y_vmean = np.mean([i.vmean for i in out_y_list], axis=0)
    #             ax[0].plot(out_x.xintm, total_x_vmean, label='Mean', c = 'k', ls ='dashed')
    #             ax[1].plot(out_y.xintm, total_y_vmean, label='Mean', c = 'k', ls ='dashed')

    #         handles, labels = ax[0].get_legend_handles_labels()
    #         plt.legend(reversed(handles), reversed(labels), # reversed so that legend aligns with graph
    #                    bbox_to_anchor=(1,1), loc='upper left')
    #         plt.show()
    #     return

    # def plot_2d(self, substance=None, single_yr=None):
    #     """
    #     Create a 2D plot of binned mixing ratios for each available year.
    #     Parameters:
    #         substance (str): if None, plots default substance for the object
    #         single_yr (int): if specified, plots only data for that year [default=None]
    #     """
    #     if substance is None: substance = self.substance
    #     if single_yr is not None: years = [int(single_yr)]
    #     else: years = self.years
    #     out_list = self.binned_2d(substance, single_yr)

    #     for out, yr in zip(out_list, years):
    #         plt.figure(dpi=300, figsize=(8,3.5))
    #         plt.gca().set_aspect('equal')

    #         cmap = plt.cm.viridis_r # create colormap
    #         if self.v_limits: vmin, vmax = self.v_limits # set colormap limits
    #         else: vmin = np.nanmin(out.vmin); vmax = np.nanmax(out.vmax)
    #         norm = Normalize(vmin, vmax) # normalise color map to set limits

    #         world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
    #         world.boundary.plot(ax=plt.gca(), color='black', linewidth=0.3)

    #         plt.imshow(out.vmean, cmap = cmap, norm=norm, origin='lower',  # plot values
    #                    extent=[out.ybmin, out.ybmax, out.xbmin, out.xbmax])
    #         cbar = plt.colorbar(ax=plt.gca(), pad=0.08, orientation='vertical') # colorbar
    #         cbar.ax.set_xlabel('Mean SF$_6$ [ppt]')

    #         plt.title('{} {} SF$_6$ concentration measurements. Gridsize={}'.format(
    #             self.source, yr, self.grid_size))
    #         plt.xlabel('Longitude  [degrees east]'); plt.xlim(-180,180)
    #         plt.ylabel('Latitude [degrees north]'); plt.ylim(-60,100)
    #         plt.show()
    #     return


    # def plot_1d_LonLat(self, lon_values = [10, 60, 120, 180],
    #                    lat_values = [70, 30, 0, -30, -70],
    #                    single_yr=None, substance=None):
    #     """
    #     Plots mixing ratio with fixed lon/lat over lats/lons side-by-side
    #     Parameters:
    #         lon_values (list of ints): longitude values to average over
    #         lat_values (list of ints): latitude values to average over
    #         substance (str): if None, plots default substance for the object
    #         single_yr (int): if specified, plots only data for that year [default=None]
    #     """
    #     if substance is None: substance = self.substance
    #     if single_yr is not None: years = [int(single_yr)]
    #     else: years = self.years

    #     out_x_list, out_y_list = self.binned_1d(substance, single_yr)

    #     for out_x, out_y, year in zip(out_x_list, out_y_list, years):
    #         fig, (ax1, ax2) = plt.subplots(dpi=250, ncols=2, figsize=(9,5), sharey=True)
    #         fig.suptitle(f'MOZART {year} SF$_6$ at fixed longitudes / latitudes', size=17)
    #         self.ds.SF6.sel(time = year, longitude=lon_values,
    #                         method='nearest').plot.line(x = 'latitude', ax=ax1)
    #         ax1.plot(out_x.xintm, out_x.vmean, c='k', ls='dashed', label='average')

    #         self.ds.SF6.sel(time = year, latitude=lat_values,
    #                         method="nearest").plot.line(x = 'longitude', ax=ax2)
    #         ax2.plot(out_y.xintm, out_y.vmean, c='k', ls='dashed', label='average')

    #         ax1.set_title(''); ax2.set_title('')
    #         ax2.set_ylabel('')
    #         plt.show()

    #     return


    # def plot(self, substance=None, greyscale=True, v_limits = (6,9)):
    #     """
    #     Plot all available data as timeseries
    #     Parameters:
    #         substance (str): specify substance (optional)
    #         greyscale (bool): toggle plotting in greyscale or viridis colormap
    #         v_limits (tuple(int, int)): change limits for colormap
    #     """
    #     if greyscale: colors = {'day':lcm(['grey']), 'msmts': lcm(['silver'])} # defining monoscale colormap for greyscale plots
    #     else: colors = {'msmts':plt.cm.viridis_r, 'day': plt.cm.viridis_r}

    #     if not substance: substance = self.substance
    #     col_name = get_col_name(substance, self.source)
    #     vmin, vmax = get_vlims(substance)
    #     norm = Normalize(vmin, vmax)
    #     dflt_unit = get_default_unit(substance)

    #     # Plot all available info on one graph
    #     fig, ax = plt.subplots(figsize = (5,3.5), dpi=250)
    #     # Measurement data
    #     plt.scatter(self.df.index, self.df[col_name], c=self.df[col_name], zorder=0,
    #                     cmap=colors['msmts'], norm=norm, marker='+',
    #                     label=f'{self.source_print} {substance.upper()}')

    #     # Daily mean
    #     if not self.df_Day.empty: # check if there is data in the daily df
    #         plt.scatter(self.df_Day.index, self.df_Day[col_name], c = self.df_Day[col_name],
    #                     cmap=colors['day'], norm=norm, marker='+', zorder=2,
    #                     label=f'{self.source_print} {substance.upper()} (D)')

    #     # Monthly mean
    #     if not self.df_monthly_mean.empty: # check for data in the monthly df
    #         for i, mean in enumerate(self.df_monthly_mean[col_name]): # plot monthly mean
    #             y, m = self.df_monthly_mean.index[i].year, self.df_monthly_mean.index[i].month
    #             xmin = dt.datetime(y, m, 1)
    #             xmax = dt.datetime(y, m, monthrange(y, m)[1])
    #             ax.hlines(mean, xmin, xmax, color='black', linestyle='dashed', zorder=2)
    #         ax.hlines(mean, xmin, xmax, color='black', ls='dashed',
    #                   label=f'{self.source_print} {substance.upper()} (M)') # needed for legend, just plots on top

    #     plt.ylabel(f'{self.substance.upper()} mixing ratio [{dflt_unit}]')
    #     plt.xlim(min(self.df.index), max(self.df.index))
    #     plt.xlabel('Time')

    #     from matplotlib.patches import Patch

    #     if not greyscale:
    #         plt.colorbar(sm(norm=norm, cmap=colors['day']), aspect=50, ax=ax, extend='neither')

    #     # Slightly weird code to create a legend showing the range of the colormap)
    #     handles, labels = ax.get_legend_handles_labels()
    #     step = 0.2
    #     pa = [ Patch(fc=colors['msmts'](norm(v))) for v in np.arange(vmin, vmax, step)]
    #     pb = [ Patch(fc=colors['day'](norm(v))) for v in np.arange(vmin, vmax, step)]
    #     pc = [ Patch(fc='black') for v in np.arange(vmin, vmax, step)]

    #     h = [] # list of handles
    #     for a, b, c in zip(pa, pb, pc): # need to do this to have them in the right order
    #         h.append(a); h.append(b); h.append(c)
    #     l = [''] * (len(h) - len(labels)) + labels # needed to have multiple color patches for one proper label
    #     ax.legend(handles=h, labels=l, ncol=len(h)/3, handletextpad=1/(len(h)/2)+0.2, handlelength=0.15, columnspacing=-0.3)

    #     fig.autofmt_xdate()
    #     plt.show()


# from calendar import monthrange
# import matplotlib.cm as cm
# import matplotlib.pyplot as plt
# from matplotlib.colors import Normalize
# from matplotlib.cm import ScalarMappable as sm
# from matplotlib.colors import ListedColormap as lcm
# supress a gui backend userwarning, not really advisible
# import warnings; warnings.filterwarnings("ignore", category=UserWarning, module='matplotlib')

# caribic_int = Caribic(c_years, grid_size, v_limits, pfxs=['INT'])

    # caribic.plot_scatter()
    # caribic.plot_scatter(single_yr=2014)
    # caribic.plot_1d()
    # caribic.plot_2d()
    # mozart.plot_1d()
    # mozart.plot_2d()
    # lon_values = [0, 10, 50, 120, 150]
    # lat_values = [70, 30, 0, -30, -70]
    # mozart.plot_1d_LonLat(lon_values, lat_values)
    # mlo.plot()
    # mlo_n2o.plot()
    # mhd.plot()
