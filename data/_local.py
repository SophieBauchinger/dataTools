# -*- coding: utf-8 -*-
""" Class definitions for data import and analysis from ground-based stations.

@Author: Sophie Bauchinger, IAU
@Date: Fri Apr 28 14:13:28 2023

class LocalData
class Mauna_Loa(LocalData)
class Mace_Head(LocalData)
"""

import datetime as dt
import numpy as np
import pandas as pd

from toolpac.conv.times import fractionalyear_to_datetime as fy_to_dt # type: ignore

import dataTools.dictionaries as dcts
from dataTools import tools

# %% Local data
class LocalData():
    """ Contains time-series data from ground-based stations.
    
    Attributes:
        years (List[int]) : years included in the stored data
        source (str) : data source, e.g. 'Mauna_Loa'
        ID (str) : short identifier of the data source
        substances (List[str]) : compounds for which measurements are available
        data (Dict[str:pd.DataFrame]) : dictionary of dataframes, keys of form 'co2'
    
    Methods:
        df (pd.DataFrame) : combined information from all dataframes
        get_data(path):
            import time-series data from files in given path
    """

    def __init__(self, years, substances):
        self.years = years
        if not hasattr(self, 'ID'): self.ID = None
        if not hasattr(self, 'source'): self.source = None
        if substances == 'all':
            self.substances = [s.short_name for s in
                               dcts.get_substances(source=self.source)
                               if not s.short_name.startswith('d_')]
        elif isinstance(substances, (list, set)):
            self.substances = substances
        else:
            raise TypeError(f'<substances> needs to be one of \'all\', \'list\', \'set\', not {type(substances)}')
        self.data = {}

    def create_df(self) -> pd.DataFrame:
        """ Combine all available dataframes in data into one. """
        df = pd.DataFrame()
        for data_value in self.data.values():
            if isinstance(data_value, pd.DataFrame):
                df = pd.concat([df, data_value], axis=1)
        return df

    @property
    def df(self) -> pd.DataFrame:
        if 'df' in self.data:
            return self.data['df']  #
        return self.create_df()

class MaunaLoa(LocalData):
    """ Stores data for all substances measured at Mauna Loa. """

    def __init__(self, years=range(1999, 2021), substances='all', data_D=False, path_dir=None):
        """ Initialise Mauna Loa object with all available substance data. """
        self.source = 'Mauna_Loa'
        self.ID = 'MLO'
        super().__init__(years=years, substances=substances)
        if data_D: self.data_D = {}
        if not path_dir: path_dir = tools.get_path() + "misc_data\\reference_data"
        self.get_data(path_dir, data_D)

    def __repr__(self):
        return f'Mauna Loa - {self.substances}'

    def get_data(self, path_dir: str, data_D=False):
        """ Add data for all substances in the given directory to data dictionary. """
        for subs in self.substances:
            self.data[subs] = self.get_subs_data(path_dir, subs, freq='M')
            if data_D:
                try:
                    self.data_D[subs] = self.get_subs_data(path_dir, subs, freq='D')
                finally:
                    print(f'No daily MLO data found for {subs}')
                    continue
        return self

    def get_subs_data(self, path_dir: str, subs: dcts.Substance, freq='M') -> pd.DataFrame:
        """ Import data from Mauna Loa files.
    
        Parameters:
            path_dir (str): Parent directory containing data files
            subs (str): short name of substance data to import
            freq (str): data frequency, 'M' / 'D'
        """
        if subs.short_name not in ['sf6', 'n2o', 'ch4', 'co2', 'co']:
            raise NotImplementedError(f'Data format and filepaths not defined for {subs}')

        if freq == 'D' and subs != 'sf6':
            raise Warning('Daily data from Mauna Loa only available for sf6.')

        # get correct path for the chosen substance
        fnames = {
            'sf6': r'\\mlo_SF6_{}.dat'.format('Day' if freq == 'D' else 'MM'),
            'n2o': '\\mlo_N2O_MM.dat',
            'co' : '\\co_mlo_surface-flask_1_ccgg_month.txt',
            'co2': '\\co2_mlo_surface-insitu_1_ccgg_MonthlyData.txt',
            'ch4': '\\ch4_mlo_surface-insitu_1_ccgg_MonthlyData.txt',
            }

        path = path_dir + fnames[subs]

        # 'ch4', 'co', 'co2' : 1st line has header_lines
        if subs in ['co2', 'ch4', 'co']:
            data_format = 'ccgg'
            with open(path, encoding='utf-8') as f:
                header_lines = int(f.readline().split(' ')[-1].strip())
                title = f.readlines()[header_lines - 2].split()
                if title[0].startswith('#'): title = title[2:]  # CO data

        elif subs in ['sf6', 'n2o']:
            data_format = 'CATS'
            header_lines = 0
            with open(path, encoding='utf-8') as f:
                for line in f:
                    if line.startswith('#'):
                        header_lines += 1
                    else:
                        title = line.split()
                        break

        else:
            raise Exception('This cannot happen')

        mlo_data = np.genfromtxt(path, skip_header=header_lines)
        df = pd.DataFrame(mlo_data, columns=title, dtype=float)

        # get names of year and month column (depends on substance)
        if data_format == 'CATS':
            yr_col = [x for x in df.columns if 'catsMLOyr' in x][0]
            mon_col = [x for x in df.columns if 'catsMLOmon' in x][0]
        elif data_format == 'ccgg':
            yr_col = 'year'
            mon_col = 'month'
        else:
            raise Exception('Data Format not valid. ')

        # keep only specified years
        df = df.loc[df[yr_col] > min(self.years) - 1].loc[df[yr_col] < max(self.years) + 1].reset_index()

        if any('catsMLOday' in s for s in df.columns):  # check if data has day column
            day_col = [x for x in df.columns if 'catsMLOday' in x][0]
            time = [dt.datetime(int(y), int(m), int(d)) for y, m, d in zip(df[yr_col], df[mon_col], df[day_col])]
            df = df.drop(day_col, axis=1)  # get rid of day column
        else:
            time = [dt.datetime(int(y), int(m), 15) for y, m in
                    zip(df[yr_col], df[mon_col])]  # choose middle of month for monthly data

        if data_format == 'CATS':
            df = df.drop(df.iloc[:, :3], axis=1)  # get rid of now unnecessary time data

        elif data_format == 'ccgg':
            filter_cols = [c for c in df.columns if c not in ['value', 'value_std_dev']]
            df.drop(filter_cols, axis=1, inplace=True)
            df.dropna(how='any', subset='value', inplace=True)
            df.rename(columns={'value': f'{subs}_{self.ID}', 'value_std_dev': f'{subs}_std_dev_{self.ID}'},
                      inplace=True)

        df.astype(float)
        df['Date_Time'] = time
        df.set_index('Date_Time', inplace=True)  # make the datetime object the new index
        if data_format == 'CATS':
            df.dropna(how='any', subset=str(str(subs).upper() + 'catsMLOm'), inplace=True)
        if data_format == 'ccgg' and subs != 'co':
            df.replace([-999.999, -999.99, -99.99, -9], np.nan, inplace=True)
            df.dropna(how='any', subset=f'{subs}_{self.ID}', inplace=True)

        if 'dict' not in self.data:
            self.data['dict'] = {}
        self.data['dict'].update({k: dcts.get_subs(col_name=k) for k in df.columns
                                  if k in [s.col_name for s in dcts.get_substances()]})
        return df

class MaceHead(LocalData):
    """ Stores data for SF6 measurements at Mace Head. """

    def __init__(self, years=(2012), substances='all', path=None):
        """ Initialise Mace Head with (daily and) monthly data in dataframes """

        self.source = 'Mace_Head'
        self.ID = 'MHD'
        super().__init__(years, substances=substances)
        if not path: path = tools.get_path('misc_data\\reference_data\\MHD-medusa_2012.dat') 
        self.path = path
        self.data_Hour = {}
        self.get_data()

    def __repr__(self):
        return f'Mace Head - {self.substances}'

    def get_data(self) -> pd.DataFrame:
        """ Import data from path definitely in init """
        if not hasattr(self, 'data'): self.data = {}
        header_lines = 0
        with open(self.path, encoding='utf-8') as f:
            for i, line in enumerate(f):
                if line.split()[0] == 'unit:':
                    title = list(f)[0].split()  # takes row below units, which is chill
                    header_lines = i + 2
                    break

        # take only relevant substances and feed column info into dictionary
        column_dict = {f'{name}_{self.ID}':
                           dcts.get_subs(short_name=name.lower(), source=self.source)
                       for name in title if name.lower() in self.substances}

        if 'dict' not in self.data:
            self.data['dict'] = column_dict

        mhd_data = np.genfromtxt(self.path, skip_header=header_lines)
        df = pd.DataFrame(mhd_data, dtype=float,
                          columns=[f'{name}_{self.ID}' for name in title])
        df = df.replace(0, np.nan)  # replace 0 with nan for statistics
        df = df.drop(df.iloc[:, :7], axis=1)  # drop unnecessary time columns
        df = df.astype(float)

        df['Date_Time'] = fy_to_dt(mhd_data[:, 0])
        df.set_index('Date_Time', inplace=True)  # new index is datetime
        cols = list(column_dict)
        df = df[cols]

        self.data_Hour['df'] = df
        # self.data_Day = {'df': tools.time_mean(df, f='D')}
        # self.data['df'] = tools.time_mean(df, f='M')
        return df
