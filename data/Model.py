# -*- coding: utf-8 -*-
""" Model data class definitions: ERa5, EMAC, Mozart

@Author: Sophie Bauchinger, IAU
@Date: Tue Jun 11 17:35:00 2024
"""
from abc import abstractmethod
import copy
import datetime as dt
import dill
from functools import partial
import geopandas
from metpy import calc
from metpy.units import units
import os
import pandas as pd
from shapely.geometry import Point
import xarray as xr

from dataTools.data._global import GlobalData
from dataTools.data.mixin_analysis import AnalysisMixin
import dataTools.dictionaries as dcts
from dataTools import tools

class ModelDataMixin:
    """ Import / Calculate new dataframes  
    
    Methods: 
        get_clams_data(met_dir, save_ds, recalculate)
            Creates dataframe for ERA5 / CLaMS data from netcdf files. 
        calc_coordinates(**kwargs)
            Calculate additional coordinates as specified through .var1 and .var2.
    """

    @abstractmethod
    def get_met_data(self):
        """ Require existance of dataframe creation method for child classes. """
        if self.ID in ['CAR', 'ATOM', 'HIPPO', 'SHTR', 'PGS', 'WISE']: 
            return self.get_clams_data()
        else: 
            raise NotImplementedError(f'Subclass of GlobalData ( - {self.ID}): need to specifically implement .get_met_data()')

    def get_clams_data(self, met_pdir=None, save_ds=False, recalculate=False) -> pd.DataFrame:
        """ Creates dataframe for CLaMS data from netcdf files. """
        if self.ID not in ['CAR', 'SHTR', 'WISE', 'ATOM', 'HIPPO', 'PGS']:
            raise KeyError(f'Cannot import CLaMS data for ID {self.ID}')

        alldata_fname = {
            'CAR' : 'caribic_clams_V03.nc'
            }
        if (self.ID in alldata_fname \
            and os.path.exists(tools.get_path() + 'misc_data/' + alldata_fname.get(self.ID)) \
            and not recalculate):
                with xr.open_dataset(tools.get_path() + 'misc_data/' + alldata_fname.get(self.ID)) as ds: 
                    ds = ds
        else: 
            print('Importing CLAMS data')
            campaign_dir_version_dict = { # campaign_pdir, version
                'CAR'  : ('CaribicTPChange',    4),
                'SHTR' : ('SouthtracTPChange',  2),
                'WISE' : ('WiseTPChange',       2), #!!! 4
                'ATOM' : ('AtomTPChange',       4),
                'HIPPO': ('HippoTPChange',      2),
                'PGS'  : ('PolstraccTPChange',  2),
                'PHL' : ('PhileasTPChange',     4),
                }
            campaign_pdir, version = campaign_dir_version_dict[self.ID]
            met_pdir = r'E:/TPChange/' + campaign_pdir
            
            fnames = met_pdir + "/*.nc"
            if self.ID == 'CAR': 
                fnames = met_pdir + "/2*/*.nc"
                
            drop_variables = {'CAR' : ['CARIBIC2_LocalTime'], 
                                'ATOM' : ['ATom_UTC_Start', 'ATom_UTC_Stop', 'ATom_End_LAS']}
                
            # extract data, each file goes through preprocess first to filter variables & convert units
            with xr.open_mfdataset(fnames, 
                                    preprocess = tools.process_TPC if not version==2 else tools.process_TPC_V02,
                                    drop_variables = drop_variables.get(self.ID),
                                    ) as ds:
                ds = ds

        if save_ds: 
            self.data['met_ds'] = ds

        met_df = ds.to_dataframe()

        if self.ID=='CAR': 
            self.data['CLAMS'] = met_df
            self.pfxs = self.pfxs+['CLAMS'] if hasattr(self, 'pfxs') else ['CLAMS']
        else: 
            self.data['met_data'] = met_df

        return met_df

    def calc_coordinates(self, **kwargs): # Calculates mostly tropopause coordinates
        """ Calculate coordinates as specified through .var1 and .var2. """
        data = self.df
        
        if kwargs.get('recalculate'): 
            data.drop(columns = [c.col_name for c in self.coordinates if c.ID=='calc'], 
                      inplace=True)
        
        all_calc_coords = dcts.get_coordinates(ID='calc') \
                        + dcts.get_coordinates(ID='CLAMS_calc') \
                        + dcts.get_coordinates(ID='MS_calc') \
                        + dcts.get_coordinates(ID='EMAC_calc')

        # Firstly calculate geopotential height from geopotential
        geopot_coords = [c for c in all_calc_coords if (
            c.var1 in data.columns and str(c.var2) == 'nan' )]
        
        for coord in geopot_coords: 
            met_data = data[coord.var1].values * units(dcts.get_coord(coord.var1).unit)
            height_m = calc.geopotential_to_height(met_data) # meters
            height_km = height_m * 1e-3
            
            if coord.unit == 'm': 
                data[coord.col_name] = height_m
            elif coord.unit == 'km': 
                data[coord.col_name] = height_km

        # Now calculate TP / distances to TP coordinates 
        calc_coords = [c for c in all_calc_coords if 
            all(col in data.columns for col in [c.var1, c.var2])]
        
        for coord in calc_coords: 
            if kwargs.get('verbose'): 
                print('Calculating ', coord.long_name, 'from \n', 
                  dcts.get_coord(col_name=coord.var1), '\n', # met
                  dcts.get_coord(col_name=coord.var2)) # tp
            
            met_coord = dcts.get_coord(col_name = coord.var1)
            tp_coord = dcts.get_coord(col_name = coord.var2)
            
            met_data = copy.deepcopy(data[coord.var1]) # prevents .df to be overwritten 
            tp_data = copy.deepcopy(data[coord.var2])
            
            if tp_coord.unit != met_coord.unit != coord.unit: 
                if all(unit in ['hPa', 'mbar'] for unit in [tp_coord.unit, met_coord.unit, coord.unit]):
                    pass
                elif all(unit in ['km', 'm'] for unit in [tp_coord.unit, met_coord.unit, coord.unit]): 
                    if coord.unit == 'm': 
                        if tp_coord.unit == 'km': tp_data *= 1e3
                        if met_coord.unit == 'km': met_data *= 1e3
                    elif coord.unit == 'km': 
                        if tp_coord.unit == 'm': tp_data *= 1e-3
                        if met_coord.unit == 'm': met_data *= 1e-3
                
                    if kwargs.get('verbose'): 
                        print('UNIT MISMATCH when calculating ', coord.long_name, 'from \n', 
                        dcts.get_coord(col_name=coord.var1), '\n', # met
                        dcts.get_coord(col_name=coord.var2)) # tp
                        
                        print('Fixed by readjusting: \n',
                              data[coord.var2].dropna().iloc[0], f' [{tp_coord.unit}] -> ', tp_data.dropna().iloc[0], f' [{coord.unit}]\n', 
                              data[coord.var1].dropna().iloc[0], f' [{met_coord.unit}] -> ', met_data.dropna().iloc[0], f' [{coord.unit}]')
                else: 
                    print(f'HALT STOPP: units do not match on {met_coord} and {tp_coord}.')
                    continue
            
            coord_data = (met_data - tp_data)
            data[coord.col_name] = coord_data

        self.data['df'] = data
        return data

# ERA5 / CLAMS
class Era5ModelData(AnalysisMixin, ModelDataMixin): 
    """ Holds ERA5 reanalysis / CLaMS model data as available from TPChange. """
    def __init__(self, campaign, met_pdir=None, recalculate=False): 
        """ Initialise object with imported reanalysis model data """
        if not campaign=='CAR': 
            years = dcts.years_per_campaign(campaign)
        else: 
            years = range(2005, 2021)
            self.pfxs = []
        self.years = years
        
        self.ID = campaign
        source_dict = {
            'SHTR' : 'HALO',
            'WISE' : 'HALO',
            'PGS'  : 'HALO',
            'TACTS': 'HALO',
            'ATOM' : 'ATOM',
            'HIPPO': 'HIAPER', 
            'CAR'  : 'Caribic'}
        self.source = source_dict[campaign]

        self.data = {}
        met_data = self.get_clams_data(met_pdir, recalculate)
        self.data['df'] = met_data
        self.calc_coordinates()

    @property
    def df(self): 
        return self.data['df']

# EMAC
class EMAC(GlobalData):
    """ Data class holding information on Caribic-specific EMAC Model output.
    
    Methods:
        create_tp()
            Create dataset with tropopause relevant parameters
        create_df()
            Create pandas dataframe from time-dependent data
    """

    def __init__(self, years=range(2005, 2020), s4d=True, s4d_s=True, tp=True, df=True, pdir=None, tps_dict={}):
        if isinstance(years, int): years = [years]
        super().__init__([yr for yr in years if 2000 <= yr <= 2019])
        self.source = 'EMAC'
        self.ID = 'EMAC'
        self.pdir = '{}'.format(r'E:/MODELL/EMAC/TPChange/' if pdir is None else pdir)
        self.get_data(years, s4d, s4d_s, tp, df)
        self.set_tps(**tps_dict)

    def __repr__(self):
        self.years.sort()
        return f'EMACData object\n\
            years: {self.years}\n\
            status: {self.status}'

    def get_data(self, years, s4d: bool, s4d_s: bool, tp: bool, df: bool, recalculate=False) -> dict:
        """ Preprocess EMAC model output and create datasets """

        if not recalculate:
            if s4d:
                with xr.open_dataset(tools.get_path()+r'misc_data\EMAC\emac_ds.nc') as ds:
                    self.data['s4d'] = ds
            if s4d_s:
                with xr.open_dataset(tools.get_path()+r'misc_data\EMAC\emac_ds_s.nc') as ds_s:
                    self.data['s4d_s'] = ds_s
            if tp:
                if os.path.exists(tools.get_path()+r'misc_data\EMAC\emac_tp.nc'):
                    with xr.open_dataset(tools.get_path()+r'misc_data\EMAC\emac_tp.nc') as tp:
                        self.data['tp'] = tp
                else:
                    self.create_tp()
            if df:
                if os.path.exists(tools.get_path()+r'misc_data\EMAC\emac_df.pkl'):
                    with open(tools.get_path()+r'misc_data\EMAC\emac_df.pkl', 'rb') as f:
                        self.data['df'] = dill.load(f)
                elif tp:
                    self.create_df()

        else:
            # print('No ready-made files found. Calculating it anew')
            if s4d:  # preprocess: process_s4d
                fnames = self.pdir + "s4d_CARIBIC/*bCARIB2.nc"
                # extract data, each file goes through preprocess first to filter variables & convert units
                with xr.open_mfdataset(fnames, preprocess=partial(tools.process_emac_s4d), mmap=False) as ds:
                    self.data['s4d'] = ds
            if s4d_s:  # preprocess: process_s4d_s
                fnames_s = self.pdir + "s4d_subsam_CARIBIC/*bCARIB2_s.nc"
                # extract data, each file goes through preprocess first to filter variables
                with xr.open_mfdataset(fnames_s, preprocess=partial(tools.process_emac_s4d_s), mmap=False) as ds:
                    self.data['s4d_s'] = ds
            if tp: self.create_tp()
            if tp and df: self.create_df()

        # update years according to available data
        self.years = list(set(pd.to_datetime(self.data['{}'.format('s4d' if s4d else 's4d_s')]['time'].values).year))

        self.data = self.sel_year(*years).data  # needed if data is loaded from file
        return self.data

    def create_tp(self) -> xr.Dataset:
        """ Create dataset with tropopause relevant parameters from s4d and s4d_s"""
        ds = self.data['s4d'].copy()
        ds_s = self.data['s4d_s'].copy()  # subsampled flight level values

        # remove floating point errors for datasets that mess up joining them up
        ds['time'] = ds.time.dt.round('S')
        ds_s['time'] = ds_s.time.dt.round('S')

        for dataset in [ds, ds_s]:
            for var in dataset.variables:
                if hasattr(dataset[var], 'units'):
                    if dataset[var].units == 'Pa':
                        dataset[var] = dataset[var].metpy.convert_units(units.hPa)
                    elif dataset[var].units == 'm':
                        dataset[var] = dataset[var].metpy.convert_units(units.km)
                    dataset[var] = dataset[var].metpy.dequantify()  # makes units an attribute again

        # get geopotential height from geopotential (s4d & s4d_s for _at_fl)
        # ^ metpy.dequantify allows putting the units back into being attributes
        print('Calculating ECHAM5_height')
        ds = ds.assign(ECHAM5_height=calc.geopotential_to_height(ds['ECHAM5_geopot'])).metpy.dequantify()
        ds['ECHAM5_height'] = ds['ECHAM5_height'].metpy.convert_units(units.km)
        ds['ECHAM5_height'] = ds['ECHAM5_height'].metpy.dequantify()

        print('Calculating ECHAM5_height_at_fl')
        ds_s = ds_s.assign(
            ECHAM5_height_at_fl=calc.geopotential_to_height(ds_s['ECHAM5_geopot_at_fl'])).metpy.dequantify()
        ds_s['ECHAM5_height_at_fl'] = ds_s['ECHAM5_height_at_fl'].metpy.convert_units(units.km)
        ds_s['ECHAM5_height_at_fl'] = ds_s['ECHAM5_height_at_fl'].metpy.dequantify()

        new_coords = dcts.get_coordinates(**{'ID': 'calc', 'source': 'EMAC', 'var1': 'not_tpress', 'var2': 'not_nan'})
        abs_coords = [c for c in new_coords if c.var2.endswith('_i')]  # get eg. value of pt at tp
        rel_coords = list(dcts.get_coordinates(**{'ID': 'calc', 'source': 'EMAC', 'var1': 'tpress', 'var2': 'not_nan'})
                          + [c for c in new_coords if c not in abs_coords])  # eg. pt distance to tp

        # copy relevant data into new dataframe
        vars_at_fl = ['longitude', 'latitude', 'tpress',
                      'tropop_PV_at_fl', 'e5vdiff_tpot_at_fl',
                      'ECHAM5_tm1_at_fl', 'ECHAM5_tpoteq_at_fl',
                      'ECHAM5_press_at_fl', 'ECHAM5_height_at_fl'] + [
                         v for v in ds_s.variables if v.startswith('tracer_')]
        tp_ds = ds_s[vars_at_fl].copy()

        tp_vars = [v.col_name for v in dcts.get_coordinates(**{'ID': 'EMAC', 'tp_def': 'not_nan'})
                   if not v.col_name.endswith(('_i', '_f')) and v.col_name in ds.variables]
        for var in tp_vars:
            tp_ds[var] = ds[var].copy()

        for coord in abs_coords:
            # e.g. potential temperature at the tropopause (get from index)
            print(f'Calculating {coord.col_name}')
            met = ds[coord.var1]
            met_at_tp = met.sel(lev=ds[coord.var2], method='nearest')
            # remove unnecessary level dimension when adding to tp_ds
            tp_ds[coord.col_name] = met_at_tp.drop_vars('lev')

        for coord in rel_coords:
            # mostly depend on abs_coords so order matters (eg. dp wrt. tp_dyn)
            print(f'Calculating {coord.col_name}')
            met = tp_ds[coord.var1]
            if coord.var2 in ds.variables:
                tp = ds[coord.var2]
            elif coord.var2 in ds_s.variables:
                tp = ds_s[coord.var2]
            elif coord.var2 in tp_ds.variables:
                tp = tp_ds[coord.var2]
            else:
                raise ValueError(f'Could not find {coord.var2} in the data')
            # units aren't propagated when subtracting so add them back
            rel = (met - tp) * units(met.units)
            tp_ds[coord.col_name] = rel.metpy.dequantify()  # makes attr 'units'

        self.data['tp'] = tp_ds
        return self.data['tp']

    def create_df(self, tp=True) -> pd.DataFrame:
        """ Create dataframe from time-dependent variables in dataset """
        if tp:
            if 'tp' not in self.data:
                print('Tropopause dataset not found, generating it now.')
                self.create_tp()
            dataset = self.data['tp']
        else:
            dataset = self.ds_s

        df = dataset.to_dataframe()
        # drop rows without geodata
        df.dropna(subset=['longitude', 'latitude'], how='any', inplace=True)
        # geodata = [Point(lat, lon) for lat, lon in zip(
        #     df['latitude'], df['longitude'])]
        geodata = [Point(lon, lat) for lon, lat in zip(
            df['longitude'], df['latitude'])]
        df.drop(['longitude', 'latitude'], axis=1, inplace=True)
        df.index = df.index.round('S')
        self.data['df'] = geopandas.GeoDataFrame(df, geometry=geodata)
        return self.data['df']

    @property
    def ds(self) -> xr.Dataset:
        """ Allow accessing dataset as class attribute """
        if 's4d' in self.data:
            return self.data['s4d']
        raise Warning('No s4d dataset found')

    @property
    def ds_s(self) -> xr.Dataset:
        """ Allow accessing dataset as class attribute """
        if 's4d_s' in self.data:
            return self.data['s4d_s']
        raise Warning('No s4d_s found')

    @property
    def tp(self) -> xr.Dataset:
        """ Returns dataset with tropopause relevant parameters  """
        if 'tp' not in self.data:
            self.create_tp()
        return self.data['tp']  # tropopause relevant parameters, only time-dependent

    @property
    def df(self) -> pd.DataFrame:
        """ Return dataframe based on subsampled EMAC Data """
        if 'df' not in self.data:
            choice = input('No dataframe found. Generate it now? [Y/N]\n')
            if choice.upper() == 'Y':
                return self.create_df()
            return pd.DataFrame()
        return self.data['df']

    def save_to_dir(self):
        """ Save emac data etc to files """
        dt.datetime.now().strftime("%Y_%m_%d-%p%I_%M_%S")
        pdir = f'misc_data/EMAC/Emac-{dt.datetime.now().strftime("%Y_%m_%d-%I_%M_%S")}'
        os.mkdir(pdir)

        with open(pdir + '/Emac_inst.pkl', 'wb') as f:
            dill.dump(self, f)
        if 's4d' in self.data: self.ds.to_netcdf(pdir + '/ds.nc')
        if 's4d_s' in self.data: self.ds_s.to_netcdf(pdir + '/ds_s.nc')
        if 'tp' in self.data: self.tp.to_netcdf(pdir + '/tp.nc')
        if 'df' in self.data:
            with open(pdir + '/df.pkl', 'wb') as f:
                dill.dump(self.df, f)
        return self

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

    def __init__(self, years=range(1980, 2009), grid_size=5, v_limits=None):
        """ Initialise Mozart object """
        super().__init__(years, grid_size)
        self.years = years
        self.source = 'Mozart'
        self.ID = 'MZT'
        self.substance = 'SF6'
        self.v_limits = v_limits  # colorbar normalisation limits
        self.data = {}
        self.get_data()

    def __repr__(self):
        return f'Mozart data, subs = {self.substance}'

    def get_data(self, remap_lon=True, verbose=False, fname=None):
        """ Create dataset from given file
    
        Parameters:
            remap_lon (bool): convert longitude from 0-360 to ±180 degrees
            verbose (bool): make the function more talkative
        """
        if not fname: fname = tools.get_path('misc_data\\reference_data\\RIGBY_2010_SF6_MOLE_FRACTION_1970_2008.nc')
        
        with xr.open_dataset(fname) as ds:
            ds = ds.isel(level=27)
        try:
            ds = ds.sel(time=self.years)
        finally:  # keep only data for specified years
            ds = xr.concat([ds.sel(time=y) for y in self.years
                            if y in ds.time], dim='time')
            if verbose: print(f'No data found for \
                              {[y for y in self.years if y not in ds.time]} \
                              in {self.source}')
            self.years = list(ds.time.values)  # only include actually available years

        if remap_lon:  # set longitudes between 180 and 360 to start at -180 towards 0
            new_lon = (((ds.longitude.data + 180) % 360) - 180)
            ds = ds.assign_coords({'longitude': ('longitude', new_lon,
                                                 ds.longitude.attrs)})
            ds = ds.sortby(ds.longitude)  # reorganise values

        self.data['ds'] = ds
        df = tools.ds_to_gdf(self.ds)
        df.rename(columns={'SF6': 'SF6_MZT'}, inplace=True)
        self.data['df'] = df
        try:
            self.data['SF6'] = self.data['df']['SF6_MZT']
        finally:
            pass

        return ds  # xr.concat(datasets, dim = 'time')

    @property
    def ds(self) -> xr.Dataset:
        return self.data['ds']

    @property
    def df(self) -> pd.DataFrame:
        return self.data['df']

    @property
    def SF6(self) -> xr.Dataset:
        return self.data['SF6']
