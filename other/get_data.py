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
from toolpac.conv.times import datetime_to_fractionalyear, fractionalyear_to_datetime

from aux_fctns import monthly_mean, ds_to_gdf, get_col_name

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
        Create a 2D plot of binned mixing ratios for each available year. 
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


class Caribic(global_data):
    """ CARIBIC data, plotting, averaging """

    def __init__(self, years, grid_size=5, v_limits=None, flight_nr = None,
               subst='sf6', pfxs=['GHG']):

        super().__init__(years, grid_size, v_limits)
        self.source = 'Caribic'
        self.substance_short = subst
        self.substance = get_col_name(self.substance_short, 'car')

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