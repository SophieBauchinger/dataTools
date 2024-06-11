# -*- coding: utf-8 -*-
""" Mozart data class definition

@Author: Sophie Bauchinger, IAU
@Date: Tue Jun 11 17:35:00 2024
"""
import pandas as pd
import xarray as xr

from dataTools import tools

from dataTools.data._global import GlobalData

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
        if not fname: fname = tools.get_path('misc_data\\RIGBY_2010_SF6_MOLE_FRACTION_1970_2008.nc')
        
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


