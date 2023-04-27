# -*- coding: utf-8 -*-
"""
@Author: Sophie Bauchimger, IAU
@Date: Mon Feb 27 13:10:50 2023

Defines functions handling data extraction and timeline plotting for ground-based 
measurement stations Mauna Loa Observatory and Mace Head Observatory

"""
import datetime as dt
import numpy as np
import pandas as pd
from calendar import monthrange
import matplotlib.pyplot as plt

from toolpac.outliers import outliers, ol_fit_functions
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

#%% Mauna Loa
class Mauna_Loa():
    """ Mauna Loa data, plotting, averaging """

    def __init__(self, years, data_Day = False, substance='sf6', 
                 path = r'C:\Users\sophie_bauchinger\Documents\GitHub\iau-caribic\misc_data'):
        """ Initialise Mauna Loa with (daily and) monthly data in dataframes """
        self.years = years
        self.substance = substance
        fname = r'\mlo_{}_MM.dat'.format(substance.upper())
        self.df = pd.concat([self.mlo_data(y, path+fname) for y in years])

        if data_Day: 
            try: # try finding daily msmt data for the substance 
                fname = r'\mlo_{}_Day.dat'.format(substance.upper())
                self.df_Day = pd.concat([self.mlo_data(y, path+fname) for y in years])
                self.df_monthly_mean = monthly_mean(self.df_Day)
            except: 
                self.df_Day = self.df_monthly_mean = False # set both to False
                print(f'No daily data found for {substance}. Please check your files')
            
        else: self.df_Day = self.df_monthly_mean = False # set both to False

    def mlo_data(self, yr, path):
        """ Create dataframe for given mlo data (.dat) for a speficied year """
        header_lines = 0 # counter for lines in header
        with open(path) as f:
            for line in f: 
                if line.startswith('#'): header_lines += 1
                else: title = line.split(); break

        mlo_data = np.genfromtxt(path, skip_header=header_lines)
        df = pd.DataFrame(mlo_data, columns=title, dtype=float)

        # get names of year and month column (depends on substance)
        yr_col = [x for x in df.columns if 'catsMLOyr' in x][0]
        mon_col = [x for x in df.columns if 'catsMLOmon' in x][0]

        df = df.loc[df[yr_col] < yr+1].loc[df[yr_col] > yr-1].reset_index() #  select only chosen year, then let index start from 0
        if any('catsMLOday' in s for s in df.columns): # check if data has day column
            day_col = [x for x in df.columns if 'catsMLOday' in x][0]
            time = [dt.datetime(int(y), int(m), int(d)) for y, m, d in zip(df[yr_col], df[mon_col], df[day_col])]
            df = df.drop(day_col, axis=1) # get rid of day column
        else: time = [dt.datetime(int(y), int(m), 15) for y, m in zip(df[yr_col], df[mon_col])]
        df = df.drop(df.iloc[:, :3], axis=1) # get rid of now unnecessary time data
        df.astype(float)
        df['Date_Time'] = time
        df.set_index('Date_Time', inplace=True) # make the datetime object the new index
        return df

    def plot(self):
        # print(self.df.index, self.df.loc[:, self.df.columns.str.endswith('catsMLOm')])
        # print(self.df_MM.index, self.df_MM.loc[:, self.df_MM.columns.str.endswith('catsMLOm')])
        fig, ax = plt.subplots(dpi=250)


        if self.df_Day is not False: # if cond is fulfilled, the data exists 
            plt.scatter(self.df_Day.index, self.df_Day.loc[:, self.df_Day.columns.str.endswith('catsMLOm')],
                        color='silver', label=f'MLO daily {self.substance.upper()}', marker='+', zorder=2)

            for i, mean in enumerate(np.array( # make array, otherwise 'enumerate' gives calc MM = name of the column
                    self.df_monthly_mean.loc[:, self.df_monthly_mean.columns.str.endswith('catsMLOm')])): # plot MLO mean
                y, m = self.df_monthly_mean.index[i].year, self.df_monthly_mean.index[i].month
                xmin = dt.datetime(y, m, 1)
                xmax = dt.datetime(y, m, monthrange(y, m)[1])
                ax.hlines(mean, xmin, xmax, color='black', ls='dashed', zorder=3)
            ax.hlines(mean, xmin, xmax, color='black', ls='dashed', label=f'MLO calc MM {self.substance.upper()}') # needed for legend, nothing else

        plt.plot(self.df.index, self.df.loc[:, self.df.columns.str.endswith('catsMLOm')],
                 'orange', zorder=2, ls='dashed', label=f'MLO {self.substance.upper()} MM')

        # plt.title(f'Ground-based {self.substance.upper()} measurements {self.years[0]} - {self.years[-1]}')
        plt.ylabel(f'Measured {self.substance.upper()} mixing ratio [ppt]')
        plt.xlim(min(self.df.index), max(self.df.index))
        plt.xlabel('Measurement time')
        plt.legend()
        fig.autofmt_xdate()
        plt.show()

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

    def plot(self):
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

#%% Function calls

if __name__=='__main__':
    v_limits = (6,9)
    grid_size = 5
    
    mlo_years = np.arange(2011, 2012)
    mlo = Mauna_Loa(mlo_years, True)
    mlo.plot()

    mlo_n2o = Mauna_Loa(mlo_years, substance='n2o')
    mlo_n2o.plot()

    mhd = Mace_Head()
    mhd.plot()

#%% Outliers
if __name__=='__main__':
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
