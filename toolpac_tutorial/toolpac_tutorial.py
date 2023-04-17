# -*- coding: utf-8 -*-
"""
@Author: Sophie Bauchimger, IAU
@Date: Mon Feb 27 13:10:50 2023

Toolpac Tutorial

"""
import datetime as dt
import geopandas
import numpy as np
import pandas as pd
from shapely.geometry import Point
from calendar import monthrange
import xarray as xr

import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable as sm

from toolpac.calc import bin_1d_2d
from toolpac.readwrite import find
from toolpac.readwrite.FFI1001_reader import FFI1001DataReader
from toolpac.outliers import outliers, ol_fit_functions
from toolpac.age import calculate_lag as cl
from toolpac.conv.times import datetime_to_fractionalyear, fractionalyear_to_datetime

# monthly_mean
def monthly_mean(df, first_of_month=True):
    """
    df: Pandas DataFrame with datetime index
    first_of_month: bool, if True sets monthly mean timestamp to first of that month

    Returns dataframe with monthly averages of all values
    """
    # group by month then calculate mean
    df_MM = df.groupby(pd.PeriodIndex(df.index, freq="M")).mean(numeric_only=True)

    if first_of_month: # reset index to first of month
        df_MM['Date_Time'] = [dt.datetime(y, m, 1) for y, m in zip(df_MM.index.year, df_MM.index.month)]
        df_MM.set_index('Date_Time', inplace=True)
    return df_MM

def ds_to_gdf(ds):
    """ Convert xarray Dataset to GeoPandas GeoDataFrame """ 
    df = ds.to_dataframe()
    geodata = [Point(lat, lon) for lon, lat in zip(
        df.index.to_frame()['longitude'], df.index.to_frame()['latitude'])]

    # create geodataframe using lat and lon data from indices
    gdf = geopandas.GeoDataFrame(
        df.reset_index().drop(['longitude', 'latitude', 'scalar', 'P0'], axis=1),
        geometry=geodata)
    return gdf

class global_data(object):
    """ 
    Global data that can be averaged on longitude / latitude grid 
    Choose years, size of the grid and adjust the colormap settings 
    """
    def __init__(self, years, grid_size, v_limits):
        """ 
        years: array or list of integers
        grid_size: int
        v_limits: tuple
        """
        self.years = years
        self.grid_size = grid_size
        self.v_limits = v_limits

        self.substance = None
        self.substance_short = None
        self.source = None

    def select_year(self, yr):
        """ Returns dataframe of selected year only """
        try: df = self.df[self.df.index.year == yr]; return df
        except: print(f'No data found for {yr}')

    def select_flight(self, nr):
        """ Returns dataframe for selected flight only """
        try: df = self.df[self.df["Flight_nr"] == nr]; return df
        except: 
            if self.source == 'Caribic': print('No data found for Flight {nr}')
            else: print('Invalid operation')

    def plot_1d(self):
        """ 
        Plot 1D averaged values over latitude / longitude including 
        colormap for all years separately.
        Returns 1D binned objects for each year as lists (lat / lon) 
        """
        out_x_list, out_y_list = [], []
        for year in self.years: # loop through available years if possible
            try: df = self.df[self.df.index.year == year]
            except: df = self.df

            if not any(self.df[self.df.index.year == year][self.substance].notna()): continue # check for "empty" data array
            if df.empty: continue # go on to next year
            print(df[self.substance])

            x = np.array([df.geometry[i].x for i in range(len(df.index))]) # lat
            y = np.array([df.geometry[i].y for i in range(len(df.index))]) # lon

            xbmin, xbmax = min(x), max(x)
            ybmin, ybmax = min(y), max(y)

            # average over lon / lat 
            out_x = bin_1d_2d.bin_1d(df[self.substance], x,
                                     xbmin, xbmax, self.grid_size)
            out_y = bin_1d_2d.bin_1d(df[self.substance], y,
                                     ybmin, ybmax, self.grid_size)

            fig, ax = plt.subplots(dpi=300, ncols=2, sharey=True, figsize=(8,3.5))
            fig.suptitle('{} {} modeled SF$_6$ concentration. Gridsize={}'.format(
                self.source, year, self.grid_size))

            cmap = plt.cm.viridis_r
            if self.v_limits: vmin, vmax = self.v_limits
            else:
                vmin = min([np.nanmin(out_x.vmean), np.nanmin(out_y.vmean)])
                vmax = max([np.nanmin(out_x.vmean), np.nanmin(out_y.vmean)])
            norm = Normalize(vmin, vmax) # allows mapping colormap onto available values  

            ax[0].plot(out_x.xintm, out_x.vmean, zorder=1, color='black', lw = 0.5)
            ax[0].scatter(out_x.xintm, out_x.vmean, # plot across latitude
                          c = out_x.vmean, cmap = cmap, norm = norm, zorder=2)
            ax[0].set_xlabel('Latitude [deg]'); plt.xlim(xbmin, xbmax)
            ax[0].set_ylabel('Mean SF$_6$ mixing ratio [ppt]')

            ax[1].plot(out_y.xintm, out_y.vmean, zorder=1, color='black', lw = 0.5)
            ax[1].scatter(out_y.xintm, out_y.vmean, # plot across longitude
                          c = out_y.vmean, cmap = cmap, norm = norm, zorder=2)
            ax[1].set_xlabel('Longitude [deg]'); plt.xlim(ybmin, ybmax)
            ax[1].set_ylabel('Mean SF$_6$ mixing ratio [ppt]')

            fig.colorbar(sm(norm=norm, cmap=cmap), aspect=50, ax = ax[1])
            plt.show()

            # add current year to 1D binned list 
            out_x_list.append(out_x); out_y_list.append(out_y) 

        # Plot all averaged mixing ratios on one graph
        fig, ax = plt.subplots(dpi=300, ncols=2, sharey=True, figsize=(8,3.5))
        fig.suptitle(f'{self.source} {self.years[0]} - {self.years[-1]} modeled {self.substance} mixing ratio. Gridsize={self.grid_size}')

        cmap = cm.get_cmap('viridis_r')
        vmin, vmax = self.years[0], self.years[-1]
        norm = Normalize(vmin, vmax)

        for out_x, out_y, year in zip(out_x_list, out_y_list, self.years): # add to plot
            ax[0].plot(out_x.xintm, out_x.vmean, label=year, c = cmap(norm(year)))
            ax[0].set_xlabel('Latitude [deg]'); plt.xlim(xbmin, xbmax)
            ax[0].set_ylabel(f'Mean {self.substance} mixing ratio [ppt]')

            ax[1].plot(out_y.xintm, out_y.vmean, label=year, c = cmap(norm(year)))
            ax[1].set_xlabel('Longitude [deg]'); plt.xlim(ybmin, ybmax)
            ax[1].set_ylabel(f'Mean {self.substance} mixing ratio [ppt]')

        handles, labels = ax[0].get_legend_handles_labels()
        plt.legend(reversed(handles), reversed(labels), 
                   bbox_to_anchor=(1,1), loc='upper left')
        plt.show()

        return out_x_list, out_y_list

    def plot_2d(self):
        """ 
        Create a 2D plot of binned mixing ratios. 
        Returns binned dataframes for each year as list 
        """
        out_list = []
        for year in self.years:
            try: df = self.df[self.df.index.year == year]
            except: df = self.df
            if df[self.substance].empty: continue

            x = np.array([df.geometry[i].x for i in range(len(df.index))]) # lat
            y = np.array([df.geometry[i].y for i in range(len(df.index))]) # lon

            xbmin, xbmax, xbsize = min(x), max(x), self.grid_size
            ybmin, ybmax, ybsize = min(y), max(y), self.grid_size

            out = bin_1d_2d.bin_2d(np.array(df[self.substance]), x, y,
                                   xbmin, xbmax, xbsize, ybmin, ybmax, ybsize)
            out_list.append(out)

            plt.figure(dpi=300, figsize=(8,3.5))
            plt.gca().set_aspect('equal')

            cmap = plt.cm.viridis_r # create colormap
            if self.v_limits: vmin, vmax = self.v_limits # set colormap limits
            else: vmin = np.nanmin(out.vmin); vmax = np.nanmax(out.vmax)
            norm = Normalize(vmin, vmax) # normalise color map to set limits

            world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
            world.boundary.plot(ax=plt.gca(), color='black', linewidth=0.3)

            plt.imshow(out.vmean, cmap = cmap, norm=norm, # plot values 
                             origin='lower', extent=[ybmin, ybmax, xbmin, xbmax])
            cbar = plt.colorbar(ax=plt.gca(), pad=0.08, orientation='vertical') # colorbar
            cbar.ax.set_xlabel('Mean $SF_6$ [ppt]')

            plt.title('{} {} SF$_6$ concentration measurements. Gridsize={}'.format(
                self.source, year, self.grid_size))
            plt.xlabel('Longitude  [degrees east]'); plt.xlim(-180,180)
            plt.ylabel('Latitude [degrees north]'); plt.ylim(-60,100)
            plt.show()

        return out_list

#%% Mauna Loa
class Mauna_Loa():
    """ Mauna Loa data, plotting, averaging """

    def __init__(self, years, path = None, path_MM = None, substance='sf6'):
        """ Initialise Mauna Loa with (daily and) monthly data in dataframes """
        self.years = years
        self.substance = substance

        parent = r'C:\Users\sophie_bauchinger\sophie_bauchinger\misc_data'
        if (not path and substance == 'sf6'): path = parent + '\mlo_SF6_Day.dat'
        if (not path_MM and substance == 'sf6'): path_MM = parent + '\mlo_SF6_MM.dat'
        if (not path and substance == 'n2o'): path_MM = parent + '\mlo_N2O_MM.dat'
        if not path: path = path_MM
        if not path_MM: path_MM = path
        print(path, path_MM)

        self.df = pd.concat([self.mlo_data(y, path) for y in years])
        self.df_MM = pd.concat([self.mlo_data(y, path_MM) for y in years])
        self.df_monthly_mean = monthly_mean(self.df)

    def mlo_data(self, yr, path):
        """ Create dataframe for given mlo data (.dat) for a speficied year """
        header_lines = 0 # counter for lines in header
        with open(path) as f:
            for line in f: 
                if line.startswith('#'): header_lines += 1
                else: title = line.split(); break

        mlo_data = np.genfromtxt(path, skip_header=header_lines)
        df = pd.DataFrame(mlo_data, columns=title, dtype=float)

        yr_col = [x for x in df.columns if 'catsMLOyr' in x][0]
        mon_col = [x for x in df.columns if 'catsMLOmon' in x][0]

        df = df.loc[df[yr_col] < yr+1].loc[df[yr_col] > yr-1].reset_index()
        if any('catsMLOday' in s for s in df.columns): # if data has day column
            day_col = [x for x in df.columns if 'catsMLOday' in x][0]
            time = [dt.datetime(int(y), int(m), int(d)) for y, m, d in zip(df[yr_col], df[mon_col], df[day_col])]
            df = df.drop(day_col, axis=1)
        else: time = [dt.datetime(int(y), int(m), 15) for y, m in zip(df[yr_col], df[mon_col])]
        df = df.drop(df.iloc[:, :3], axis=1)
        df.astype(float)
        df['Date_Time'] = time
        df.set_index('Date_Time', inplace=True)
        return df

    def plot_MM(self):
        # print(self.df.index, self.df.loc[:, self.df.columns.str.endswith('catsMLOm')])
        # print(self.df_MM.index, self.df_MM.loc[:, self.df_MM.columns.str.endswith('catsMLOm')])
        fig, ax = plt.subplots(dpi=250)
        plt.scatter(self.df.index, self.df.loc[:, self.df.columns.str.endswith('catsMLOm')],
                    color='silver', label='Mauna Lao', marker='+')

        plt.plot(self.df_MM.index, self.df_MM.loc[:, self.df_MM.columns.str.endswith('catsMLOm')],
                 'red', zorder=1, linestyle='dashed', label='Mauna Lao, MM')

        for i, mean in enumerate(np.array( # make array, otherwise 'enumerate' gives mean as the col name
                self.df_monthly_mean.loc[:, self.df_monthly_mean.columns.str.endswith('catsMLOm')])): # plot MLO mean
            y, m = self.df_monthly_mean.index[i].year, self.df_monthly_mean.index[i].month
            xmin = dt.datetime(y, m, 1)
            xmax = dt.datetime(y, m, monthrange(y, m)[1])
            ax.hlines(mean, xmin, xmax, color='black', linestyle='dashed', zorder=2)
        ax.hlines(mean, xmin, xmax, color='black', linestyle='dashed', label='Calculated MM') # needed for legend, nothing else

        plt.title(f'Ground-based {self.substance} measurements {self.years[0]} - {self.years[-1]}')
        plt.ylabel(f'Measured {self.substance} mixing ratio [ppt]')
        plt.xlim(min(self.df.index), max(self.df.index))
        plt.xlabel('Measurement time')
        plt.legend()
        fig.autofmt_xdate()
        plt.show()

if __name__=='__main__':
    years = np.arange(2011, 2012)
    mlo = Mauna_Loa(years=years)
    mlo.plot_MM()
    mlo_n2o = Mauna_Loa(years, substance='n2o')
    mlo_n2o.plot_MM()

#%% Mace Head
class Mace_Head():
    """ Mace Head data, plotting, averaging """

    def __init__(self, path = None):
        self.years = 2012
        if not path: path = r'C:\Users\sophie_bauchinger\sophie_bauchinger\misc_data\MHD-medusa_2012.dat'
        self.df = self.mhd_data(path)
        self.df_monthly_mean = monthly_mean(self.df)

    def mhd_data(self, path):
        """ Create dataframe from Mace Head data in .dat file"""
        # extract and stitch together names and units for column headers
        header_lines = 0
        with open(path) as f:
            for i, line in enumerate(f):
                if line.split()[0] == 'unit:': 
                    units = line.split()
                    title = list(f)[0].split() # takes next row for some reason
                    header_lines = i+2; break
        column_headers = [name + "[" + unit + "]" for name, unit in zip(title, units)]

        mhd_data = np.genfromtxt(path, skip_header=header_lines)

        df = pd.DataFrame(mhd_data, columns=column_headers, dtype=float)
        df = df.replace(0, np.nan) # replace 0 with nan for statistics
        df = df.drop(df.iloc[:, :7], axis=1) # drop unnecessary time columns
        df = df.astype(float) 

        df['Date_Time'] = fractionalyear_to_datetime(mhd_data[:,0]) 
        df.set_index('Date_Time', inplace=True) # new index is datetime
        return df

    def plot_mhd(self):
        """ Plot Mace Head meausurements and monthly means over time """ 
        fig, ax = plt.subplots(dpi=250)
        plt.scatter(self.df.index, self.df['SF6[ppt]'],
                    color='grey', label='Mace Head', marker='+')

        for i, mean in enumerate(self.df_monthly_mean['SF6[ppt]']): # plot MHD mean
            y, m = self.df_monthly_mean.index[i].year, self.df_monthly_mean.index[i].month
            xmin = dt.datetime(y, m, 1)
            xmax = dt.datetime(y, m, monthrange(y, m)[1])
            ax.hlines(mean, xmin, xmax, color='black', linestyle='dashed', zorder=2)

        plt.title('Ground-based SF$_6$ measurements 2012')
        plt.ylabel('Measured SF$_6$ mixing ratio [ppt]')
        plt.xlabel('Measurement time')
        plt.legend()
        plt.show()

if __name__=='__main__':
    mhd = Mace_Head()
    mhd.plot_mhd()

#%% CARIBIC
class Caribic(global_data):
    """ CARIBIC data, plotting, averaging """

    def __init__(self, years, grid_size=5, v_limits=None, flight_nr = None,
               subst='sf6', pfxs=['GHG']):

        super().__init__(years, grid_size, v_limits)
        self.source = 'Caribic'
        self.substance_short = subst
        self.substance = self.get_col_name(self.substance_short)

        self.df, self.column_dict = self.caribic_data(pfxs)

        if flight_nr: self.df = self.df[self.df.values == flight_nr]
        self.df_monthly_mean = monthly_mean(self.df)

        self.x = np.array([self.df.geometry[i].x for i in range(len(self.df.index))]) # lat
        self.y = np.array([self.df.geometry[i].y for i in range(len(self.df.index))]) # lon

    def caribic_data(self, pfxs):
        """ 
        Create geopandas dataframe with caribic data for given year and file prefixes
        Automatically adds flight number as column and sets index to 
        datetime object of the sampling time

        Returns tuple of geodataframe and dictionary of short and long column names
        """ 
        gdf = geopandas.GeoDataFrame() # initialise GeoDataFrame
        parent_dir = r'E:\CARIBIC\Caribic2data'

        for yr in self.years:
            if not any(find.find_dir("*_{}*".format(yr), parent_dir)): 
                print(f'No data found for {yr}'); continue
            df = pd.DataFrame()
            print(f'Reading in Caribic data for {yr}')

            # First collect data from individual flights 
            for current_dir in find.find_dir("*_{}*".format(yr), parent_dir)[1:]:
                flight_nr = int(str(current_dir)[-12:-9])
                for pfx in pfxs:
                    f = find.find_file(f'{pfx}_*', current_dir)
                    if not f: # show error msg and go directly to next loop
                        print(f'No {pfx} File found for Flight {flight_nr} in {yr}'); continue
                    elif len(f) > 1: f.sort() # sort list of files, then take latest
    
                    f_data = FFI1001DataReader(f[0], df=True, xtype = 'secofday')
                    df_temp = f_data.df # index = Datetime
                    df_temp.insert(0, 'Flight number',
                                   [flight_nr for i in range(df_temp.shape[0])])
                    df = pd.concat([df, df_temp])

            # Convert longitude and latitude into geometry objects -> GeoDataFrame
            geodata = [Point(lat, lon) for lon, lat in zip(
                df['lon; longitude (mean value); [deg]\n'],
                df['lat; latitude (mean value); [deg]\n'])]
            gdf_temp = geopandas.GeoDataFrame(df, geometry=geodata)
    
            # Drop all unnecessary columns [info is saved within datetime, geometry]
            if not gdf_temp['geometry'].empty:
                gdf_temp = gdf_temp.drop(['TimeCRef; CARIBIC_reference_time_since_0_hours_UTC_on_first_date_in_line_7; [s]',
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
            else: print('Geodata creation was unsuccessful. Please check your input.')
            # try: gdf_temp = gdf_temp[gdf_temp[self.get_new_column_name(self.substance_short)].notna()]
            # except: print(f'{self.substance_short} data not available in {pfxs} in {yr}')

            # Add current year to final dataframe
            gdf = pd.concat([gdf, gdf_temp])

        if gdf.empty: print("Data extraction unsuccessful. Please check your input data"); return 

        # Create a dictionary for translating short column names to description
        new_names = [x.split(";")[0] + x.split(";")[-1][:-1] for x in gdf.columns if len(x.split(";")) == 3] # remove \n with [-1]
        description = [x.split(";")[1] for x in gdf.columns if len(x.split(";")) == 3]
        gdf.rename(columns = dict(zip([x for x in gdf.columns if len(x.split(";")) == 3], new_names)), inplace=True)
        col_names_dict = dict(zip(new_names, description))

        return gdf, col_names_dict

    # def get_col_name_original(self, substance):
    #     """ Returns name of original column name - obsolete? """
    #     column_names = {
    #         'sf6': 'SF6; SF6 mixing ratio; [ppt]\n',
    #         'n2o': 'N2O; N2O mixing ratio; [ppt]\n',
    #         'co2': 'CO2; CO2 mixing ratio; [ppm]\n',
    #         'ch4': 'CH4; CH4 mixing ratio; [ppb]\n',
    #         }
    #     return column_names[substance]

    def get_col_name(self, substance):
        """ Returns column name for substance as saved in dataframe """
        new_names = {
            'sf6': 'SF6 [ppt]',
            'n2o': 'N2O [ppb]',
            'co2': 'CO2 [ppm]',
            'ch4': 'CH4 [ppb]',
            }
        return new_names[substance]

    def plot_scatter(self):
        """ Plot msmts and monthly mean for specified years [list] """

        # Plot mixing ratio msmts and monthly mean
        fig, ax = plt.subplots(dpi=250)
        plt.title(f'CARIBIC {self.substance_short} measurements')
        ymin = self.df[self.substance].min()
        ymax = self.df[self.substance].max()

        cmap = plt.cm.viridis_r
        extend = 'neither'
        if self.v_limits: vmin, vmax = self.v_limits# ; extend = 'both'
        else: vmin = ymin; vmax = ymax
        norm = Normalize(vmin, vmax)

        plt.scatter(self.df.index, self.df[self.substance],
                    label=f'{self.substance_short} {self.years}', marker='x', zorder=1,
                    c = self.df[self.substance],
                    cmap = cmap, norm = norm)

        for i, mean in enumerate(self.df_monthly_mean[self.substance]):
            y = self.df_monthly_mean.index[i].year
            m = self.df_monthly_mean.index[i].month
            xmin = dt.datetime(y, m, 1)
            xmax = dt.datetime(y, m, monthrange(y, m)[1])
            ax.hlines(mean, xmin, xmax, color='black',
                      linestyle='dashed', zorder=2)

        plt.colorbar(sm(norm=norm, cmap=cmap), aspect=50, ax = ax, extend=extend)

        plt.ylabel(f'{self.substance_short} mixing ratio [ppt]')
        plt.ylim(ymin-0.15, ymax+0.15)
        fig.autofmt_xdate()
        plt.show()

    def try_plot_2d(self):
        out_list = []
        for year in self.years:
            try: df = self.df[self.df.index.year == year]
            except: df = self.df
            if df.empty: continue

            x = np.array([df.geometry[i].x for i in range(len(df.index))]) # lat
            y = np.array([df.geometry[i].y for i in range(len(df.index))]) # lon

            xbmin, xbmax, xbsize = min(x), max(x), self.grid_size
            ybmin, ybmax, ybsize = min(y), max(y), self.grid_size

            out = bin_1d_2d.bin_2d(np.array(df[self.substance]), x, y,
                                   xbmin, xbmax, xbsize, ybmin, ybmax, ybsize)
            out_list.append(out)

            # cmap, normalisation, geopandas world
            if self.v_limits: vmin, vmax = self.v_limits
            else: vmin = np.nanmin(out.vmin); vmax = np.nanmax(out.vmax)
            norm = Normalize(vmin, vmax)
            cmap = plt.cm.viridis_r
            world = geopandas.read_file(
                geopandas.datasets.get_path('naturalearth_lowres'))

            # plot mixing ratio, colorbar, world map
            plt.figure(dpi=300, figsize=(8,3.5))
            plt.gca().set_aspect('equal')
            world.boundary.plot(ax = plt.gca(), color='black', linewidth=0.3)
            plt.imshow(out.vmean, cmap = cmap, norm=norm, #interpolation='nearest', 
                             origin='lower', extent=[ybmin, ybmax, xbmin, xbmax])

            cbar = plt.colorbar(ax=plt.gca(), pad=0.08, orientation='vertical')
            cbar.ax.set_xlabel(f'Mean {self.substance_short} [ppt]')

            plt.xlabel('Longitude  [deg]'); plt.xlim(-180,180)
            plt.ylabel('Latitude [deg]'); plt.ylim(-60,100)
            plt.title('{} {} {} concentration measurements. Gridsize={}'.format(
                self.source, year, self.substance_short, self.grid_size))
            plt.show()

        return out_list

if __name__=='__main__':
    years = np.arange(2016, 2023)
    v_limits = (6,9)
    grid_size = 5
    pfxs=['GHG', 'INT']

    caribic = Caribic(years, v_limits = v_limits, grid_size=grid_size, pfxs = pfxs)
    caribic.plot_scatter()
    caribic.plot_1d()
    caribic.plot_2d()
    caribic.try_plot_2d()

#%% MOZART
class Mozart(global_data):
    """
    Class attributes:
        years: arr
        source: str
        substance: str
        ds: xarray DataFrame
        df: Pandas GeoDataFrame
        x: arr, latitude
        y: arr, longitude (remapped to +-180 deg)
        SF6: Pandas DataSeries of SF6 mixing ratios
    """

    def __init__(self, years, grid_size=5, v_limits=None):
        """ Initialise MOZART object """
        super().__init__( years, grid_size, v_limits)
        self.years = years
        self.source = 'Mozart'
        self.substance = 'sf6'

        self.ds = xr.concat([self.mozart_data(year = y) for y in self.years], dim='time')
        self.df = ds_to_gdf(self.ds)
        self.SF6 = self.df['SF6']

    def mozart_data(self, year, level = 27, remap = True, 
                    file = r'C:\Users\sophie_bauchinger\sophie_bauchinger\toolpac_tutorial\RIGBY_2010_SF6_MOLE_FRACTION_1970_2008.nc'):
        """ 
        Returns xarray Dataset of MOZART model data
        If remap is set True, longitude values are remapped for pos / neg symmetry

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
        with xr.open_dataset(file) as ds:
            ds = ds.isel(level=27)
            ds = ds.sel(time = year)

        if remap: # set longitudes between 180 and 360 to start at -180 towards 0
            ds_remap = ds
            ds_remap['longitude'] = np.array([i for i in ds.longitude if i<=180] +
                                       [i - 360 for i in ds.longitude if i>180])
            ds = ds_remap.sortby(ds_remap.longitude) # reorganise values             
        return ds

    def plot_scatter(self, total=False):
        """ Plot 1D averaged data over latitude or longitude for each years.
        If total is set True, plot the average mixing ratio for all years """
        if total: 
            x = np.array([self.df.geometry[i].x for i in range(len(self.df.index))]) # lat
            y = np.array([self.df.geometry[i].y for i in range(len(self.df.index))]) # lon
            
            xbmin, xbmax = min(x), max(x)
            ybmin, ybmax = min(y), max(y)
    
            out_x = bin_1d_2d.bin_1d(self.SF6, x, xbmin, xbmax, self.grid_size)
            out_y = bin_1d_2d.bin_1d(self.SF6, y, ybmin, ybmax, self.grid_size)

            fig, ax = plt.subplots(dpi=300, ncols=2, sharey=True, figsize=(8,3.5))
            fig.suptitle('{} {} - {} modeled SF$_6$ concentration. Gridsize={}'.format(
                self.source, self.years[0], self.years[-1], self.grid_size))
    
            ax[0].plot(out_x.xintm, out_x.vmean, zorder=1, color='black', lw = 0.5)
            ax[0].scatter(out_x.xintm, out_x.vmean)#,
                          #c = out_x.vmean, cmap = cmap, norm = norm, zorder=2)
            ax[0].set_xlabel('Latitude [degrees north]'); plt.xlim(xbmin, xbmax)
            ax[0].set_ylabel('Mean SF$_6$ mixing ratio [ppt]')
    
            ax[1].plot(out_y.xintm, out_y.vmean, zorder=1, color='black', lw = 0.5)
            ax[1].scatter(out_y.xintm, out_y.vmean)#,
                          #c = out_y.vmean, cmap = cmap, norm = norm, zorder=2)
            ax[1].set_xlabel('Longitude [degrees east]'); plt.xlim(ybmin, ybmax)
            ax[1].set_ylabel('Mean SF$_6$ mixing ratio [ppt]')
            plt.show()

        else: 
            for y in self.years:
                ds = self.mozart_data(year = y)
                df = ds_to_gdf(ds)
        
                x = np.array([df.geometry[i].x for i in range(len(df.index))]) # lat
                y = np.array([df.geometry[i].y for i in range(len(df.index))]) # lon
                y = np.array([i for i in y if i<=180] +
                                  [i - 360 for i in y if i>180])                

                out_x = bin_1d_2d.bin_1d(df['SF6'], x, min(x), max(x), self.grid_size)
                out_y = bin_1d_2d.bin_1d(df['SF6'], y, min(y), max(y), self.grid_size)

                fig, ax = plt.subplots(dpi=300, ncols=2, sharey=True, figsize=(8,3.5))
                fig.suptitle('{} {} modeled SF$_6$ concentration. Gridsize={}'.format(
                    self.source, y, self.grid_size))
        
                ax[0].plot(out_x.xintm, out_x.vmean, zorder=1, color='black', lw = 0.5)
                ax[0].scatter(out_x.xintm, out_x.vmean)#,
                              #c = out_x.vmean, cmap = cmap, norm = norm, zorder=2)
                ax[0].set_xlabel('Latitude [degrees north]'); plt.xlim(min(x), max(x))
                ax[0].set_ylabel('Mean SF$_6$ mixing ratio [ppt]')
        
                ax[1].plot(out_y.xintm, out_y.vmean, zorder=1, color='black', lw = 0.5)
                ax[1].scatter(out_y.xintm, out_y.vmean)#,
                              #c = out_y.vmean, cmap = cmap, norm = norm, zorder=2)
                ax[1].set_xlabel('Longitude [degrees east]'); plt.xlim(min(y), max(y))
                ax[1].set_ylabel('Mean SF$_6$ mixing ratio [ppt]')
                plt.show()

    def plot_1d_LonLat(self, lon_values = [10, 60, 120, 180], lat_values = [70, 30, 0, -30, -70]):
        """ plot change over lat, lon with fixed lon / lat """
        for year in self.years:
            fig, (ax1, ax2) = plt.subplots(dpi=250, ncols=2, figsize=(9,5), sharey=True)
            fig.suptitle(f'MOZART {year} SF$_6$ at fixed longitudes / latitudes', size=17)
            self.ds.SF6.sel(time = year, longitude=lon_values, method='nearest').plot.line(x = 'latitude', ax=ax1)
            self.ds.SF6.sel(time = year, latitude=lat_values, method="nearest").plot.line(x = 'longitude', ax=ax2) # ax = ax
            ax1.set_title(''); ax2.set_title('')
            ax2.set_ylabel('')
            plt.show()

if __name__=='__main__':
    years = np.arange(2000, 2008)
    v_limits = (6,9)
    grid_size = 10

    mozart = Mozart(years=years, v_limits = v_limits, grid_size = grid_size)
    mozart.plot_scatter()
    mozart.plot_scatter(total=True)
    mozart.plot_1d_LonLat()
    mozart.plot_1d()
    mozart.plot_2d()

#%% Function calls
if __name__=='__main__':
    v_limits = (6,9)
    grid_size = 5
    
    mlo_years = np.arange(2011, 2012)
    mlo = Mauna_Loa(years=mlo_years)
    mlo.plot_MM()

    mhd = Mace_Head()
    mhd.plot_mhd()

    c_years = np.arange(2008, 2014)
    caribic = Caribic(c_years, v_limits = v_limits, grid_size=grid_size)
    caribic.plot_scatter()
    caribic.plot_1d()
    caribic.plot_2d()
    caribic.try_plot_2d()
    
    m_years = np.arange(2000, 2008)
    mozart = Mozart(years=m_years, v_limits = v_limits)
    mozart.plot_scatter()
    mozart.plot_1d()
    mozart.plot_2d()

#%% Outliers
if __name__=='__main__':
    for y in range(2008, 2010): # Caribic
        for dir_val in ['np', 'p', 'n']:
            data = Caribic([y]).df
            sf6_mxr = data['SF6; SF6 mixing ratio; [ppt]\n']
            ol = outliers.find_ol(ol_fit_functions.simple, data.index, sf6_mxr, None, None, 
                                  plot=True, limit=0.1, direction = dir_val)
            
    for y in range(2008, 2010): 
        for dir_val in ['np', 'p', 'n']:
            data = Mauna_Loa([y]).df
            sf6_mxr = data['SF6catsMLOm']
            ol = outliers.find_ol(ol_fit_functions.simple, data.index, sf6_mxr, None, None, 
                                  plot=True, limit=0.1, direction = dir_val)
   
    for dir_val in ['np', 'p', 'n']: # Mace Head
        data = Mace_Head().df
        sf6_mxr = data['SF6[ppt]']
        ol = outliers.find_ol(ol_fit_functions.simple, data.index, sf6_mxr, None, None, 
                              plot=True, limit=0.1, direction = dir_val)

#%% Time Lag calculations
if __name__=='__main__':
    mlo_time_lims = (2000, 2020)
    mlo_MM = Mauna_Loa(years = np.arange(*mlo_time_lims)).df #.df_monthly_mean
    mlo_MM.resample('1M') # add rows for missing months, filled with NaN 
    mlo_MM.interpolate(inplace=True) # linearly interpolate missing data

    t_ref = np.array(datetime_to_fractionalyear(mlo_MM.index, method='exact'))
    c_ref = np.array(mlo_MM['SF6catsMLOm'])

    for c_year in range(2012, 2014):
        c_data = Caribic([c_year]).df
        t_obs_tot = np.array(datetime_to_fractionalyear(c_data.index, method='exact'))
        c_obs_tot = np.array(c_data['SF6; SF6 mixing ratio; [ppt]\n'])
    
        lags = []
        for t_obs, c_obs in zip(t_obs_tot, c_obs_tot):
            lag = cl.calculate_lag(t_ref, c_ref, t_obs, c_obs, plot=True)
            lags.append((lag))
    
        fig, ax = plt.subplots(dpi=300)
        plt.scatter(c_data.index, lags, marker='+')
        plt.title('CARIBIC SF$_6$ time lag {} wrt. MLO {} - {}'.format(c_year, *mlo_time_lims))
        plt.ylabel('Time lag [yr]')
        plt.xlabel('CARIBIC Measurement time')
        fig.autofmt_xdate()
