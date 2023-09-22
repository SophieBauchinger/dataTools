# -*- coding: utf-8 -*-
"""
@Author: Sophie Bauchinger, IAU
@Date: Fri Apr 28 14:13:28 2023

Defines classes used as basis for data structures
"""
import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from os.path import exists

from toolpac.conv.times import fractionalyear_to_datetime, datetime_to_fractionalyear

from tools import monthly_mean, daily_mean
from dictionaries import get_col_name, substance_list

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
                fname_Day = r'\mlo_{}_Day.dat'.format(self.subs.upper())
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
