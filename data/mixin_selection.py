# -*- coding: utf-8 -*-
""" Mixin for adding sub-selection methods to GlobalData objects 

@Author: Sophie Bauchinger, IAU
@Date: Tue Jun 11 16:45:00 2024

class SelectionMixin

#TODO: UTLS Selection
 * 400 - 10 hPa 
 *  6.5 - 30 km
 * +- 5km around the tropopause

"""

import copy 
import pandas as pd
import xarray as xr
import geopandas 

from dataTools import dictionaries as dcts
from dataTools import tools

class SelectionMixin: 
    """ Holds methods that remove all data that does not fit the given sub-selection criteria. 
    
    Methods:
        sel_subset(**kwargs)
            Combine multiple selection methods using keyword arguments to deduce the desired functionality
        sel_year(*years)
            Remove all data not in the chosen years
        sel_latitude(lat_min, lat_max)
            Remove all data not in the chosen latitude range
        sel_eqlat(eql_min, eql_max)
            Remove all data not in the chosen equivalent latitude range
        sel_season(season)
            Remove all data not in the chosen season
        sel_flight(flights)
            Remove all data that is not from the chosen flight numbers
        
        sel_atm_layer(atm_layer, **kwargs)
            Remove all data not in the chosen atmospheric layer (tropo/strato)
        sel_tropo()
            Remove all stratospheric datapoints
        sel_strato()
            Remove all tropospheric datapoints
            
        
    """

    def sel_subset(self, inplace:bool = False, **kwargs):
        """ Allows making multiple selections at once.

        Parameters:
            key year (list or int)
            key latitude (Tuple[float, float])
            key eqlat (Tuple[float, float])
            key flight (list or int)
            key tropo / strato (dict)
        """
        # check function input
        if any([f'sel_{k}' not in self.__dir__() for k in kwargs]):
            raise KeyError('Subset selection is not possible with the following parameters:\
                            {}'.format([k for k in kwargs if f'sel_{k}' not in self.__dir__()]))

        out = type(self).__new__(self.__class__)  # new class instance
        for attribute_key in self.__dict__:  # copy attributes
            out.__dict__[attribute_key] = copy.deepcopy(self.__dict__[attribute_key])
        out.data = self.data.copy()  # stops self.data being overwritten

        def get_fctn(class_inst, selection):
            if f'sel_{selection}' in class_inst.__dir__():
                return getattr(class_inst, f'sel_{selection}')
            else:
                raise KeyError(f'sel_{selection}')

        for selection in kwargs:
            if isinstance(kwargs.get(selection), (int, str)):
                out = get_fctn(out, selection)(kwargs.get(selection))
            if isinstance(kwargs.get(selection), (set, list, tuple)):
                out = get_fctn(out, selection)(*kwargs.get(selection))
            if isinstance(kwargs.get(selection), dict):
                out = get_fctn(out, selection)(**kwargs.get(selection))

        if inplace: 
            self.__dict__.update(out.__dict__)

        return out

    def sel_year(self, *years:int, inplace:bool=False):
        """ Returns GlobalData object containing only data for selected years. """

        # input validation, choose only years that are actually available
        yr_list = [yr for yr in years if yr in self.years]
        if len(yr_list) == 0:
            raise KeyError(f'No valid data for any of the given years: {years}')
        if len(yr_list) != len(years):
            print(f'Note: No data available for {[yr for yr in years if yr not in self.years]}')

        out = type(self).__new__(self.__class__)  # new class instance
        for attribute_key in self.__dict__:  # copy attributes
            out.__dict__[attribute_key] = copy.deepcopy(self.__dict__[attribute_key])
        out.data = self.data.copy()  # stops self.data being overwritten

        if self.source == 'MULTI': 
            out.data['df'] = out.data['df'][out.data['df'].index.year.isin(yr_list)]

        elif self.source in ['Caribic', 'EMAC', 'TP']:
            # Dataframes
            df_list = [k for k in self.data
                       if isinstance(self.data[k], pd.DataFrame)]  # or Geo-dataframe
            for k in df_list:  # only take data from chosen years
                out.data[k] = out.data[k][out.data[k].index.year.isin(yr_list)]
                out.data[k].sort_index(inplace=True)

            # Datasets
            ds_list = [k for k in self.data
                       if isinstance(self.data[k], xr.Dataset)]
            for k in ds_list:
                out.data[k] = out.data[k].sel(time=out.data[k].time.dt.year.isin(yr_list))

        elif self.source == 'Mozart':  # yearly data
            for k in [k for k in self.data if isinstance(self.data[k], pd.DataFrame)]:
                out.data[k] = out.data[k][out.data[k].index.year.isin(yr_list)].sort_index()
            for k in [k for k in self.data if isinstance(self.data[k], xr.Dataset)]:
                out.data[k] = out.data[k].sel(time=yr_list)

        yr_list.sort()
        out.years = yr_list
        
        if inplace: 
            self.__dict__.update(out.__dict__)
        return out

    def sel_latitude(self, lat_min:float, lat_max:float, inplace:bool=False):
        """ Returns GlobalData object containing only data for selected latitudes """
        # copy everything over without changing the original class instance
        out = type(self).__new__(self.__class__)
        for attribute_key in self.__dict__:
            out.__dict__[attribute_key] = copy.deepcopy(self.__dict__[attribute_key])

        out.data = self.data.copy()
        
        if self.source == 'MULTI' and isinstance(self.data['df'], geopandas.GeoDataFrame): 
            out.data['df'] = out.data['df'].cx[-180:180, lat_min:lat_max]

        elif self.source in ['Caribic', 'EMAC', 'TP', 'HALO', 'ATOM']:
            df_list = [k for k in self.data
                       if isinstance(self.data[k], geopandas.GeoDataFrame)]  # needed for latitude selection
            for k in df_list:  # delete everything that isn't the chosen lat range
                out.data[k] = out.data[k].cx[-180:180, lat_min:lat_max]
                out.data[k].sort_index(inplace=True)
            for k in [k for k in self.data if
                      k not in df_list and isinstance(self.data[k], pd.DataFrame)]:  # non-geodataframes
                indices = [index for df_indices in [out.data[k].index for k in df_list] for index in
                           df_indices]  # all indices in the Geo-dataframes
                out.data[k] = out.data[k].loc[out.data[k].index.isin(indices)]

            # update available years, flights
            if len(df_list) != 0:
                out.years = list(set(out.data[df_list[-1]].index.year))
                out.years.sort()

            # Datasets
            ds_list = [k for k in self.data
                       if (isinstance(self.data[k], xr.Dataset) 
                           and 'latitude' in out.data[k].variables)]

            for k in ds_list:
                out.data[k] = out.data[k].where(out.data[k]['latitude'] > lat_min)
                out.data[k] = out.data[k].where(out.data[k]['latitude'] < lat_max)

            # update years if it hasn't happened with the dataframe already
            if 'df' not in self.data and self.source == 'EMAC':  # only dataset exists
                self.years = list(set(pd.to_datetime(self.data['ds']['time'].values).year))
                
            # update object status
            if 'latitude' in out.status:
                out.status['latitude'] = (max([out.status['latitude'][0], lat_min]),
                                          min([out.status['latitude'][1], lat_max]))
            else:
                out.status['latitude'] = (lat_min, lat_max)

        elif self.source == 'Mozart':
            out.data['df'] = out.df.query(f'latitude > {lat_min}')
            out.data['df'] = out.df.query(f'latitude < {lat_max}')
            out.years = list(set(out.df.index.year))

            if hasattr(out, 'ds'):
                out.ds = out.ds.sel(latitude=slice(lat_min, lat_max))
            if hasattr(out, 'SF6'):
                out.SF6 = out.df['SF6']

        else:
            raise Warning(f'Not implemented for {self.source}')

        if inplace: 
            self.__dict__.update(out.__dict__)
        return out

    def sel_eqlat(self, eql_min:float, eql_max:float, model='ERA5', inplace:bool=False):
        """ Returns GlobalData object containing only data for selected equivalent latitudes """
        # copy everything over without changing the original class instance
        out = type(self).__new__(self.__class__)
        for attribute_key in self.__dict__:
            out.__dict__[attribute_key] = copy.deepcopy(self.__dict__[attribute_key])
        out.data = self.data.copy()

        if self.source not in ['Caribic', 'TP']:
            raise NotImplementedError('Action not yet supported for non-Caribic data')
        [eql] = self.get_coords(model=model, hcoord='eql')
        df = self.data['df'].copy()
        df = df[df[eql.col_name] > eql_min]
        df = df[df[eql.col_name] < eql_max]
        out.data['df'] = df

        df_list = [k for k in self.data
                   if isinstance(self.data[k], pd.DataFrame)]  # all dataframes

        for k in df_list:  # delete everything outside eql range
            out.data[k] = out.data[k][out.data[k].index.isin(df.index)]

        # update object status
        if 'eq_lat' in out.status:
            out.status['eq_lat'] = (max([out.status['eq_lat'][0], eql_min]),
                                    min([out.status['eq_lat'][1], eql_max]))
        else:
            out.status['eq_lat'] = (eql_min, eql_max)

        if inplace: 
            self.__dict__.update(out.__dict__)
        return out

    def sel_season(self, *seasons, inplace:bool=False):
        """ Return GlobalData object containing only pd.DataFrames for the chosen season

        Parameters:
            seasons (List[int]): list of multiple of 1,2,3,4
                1 - spring, 2 - summer, 3 - autumn, 4 - winter """
        if 'season' in self.status:
            if any(s == dcts.dict_season()[f'name_{s}'] for s in self.status['season']):
                return self
            raise Warning('Cannot select {} as already filtered for {}'.format(
                [dcts.dict_season()[f'name_{s}'] for s in seasons], self.status['season']))
        out = type(self).__new__(self.__class__)  # new class instance
        for attribute_key in self.__dict__:  # copy attributes
            out.__dict__[attribute_key] = copy.deepcopy(self.__dict__[attribute_key])

        out.data = {}
        # Dataframes
        df_list = [k for k in self.data
                   if isinstance(self.data[k], pd.DataFrame)]  # or Geodf
        for k in df_list:  # only take data from chosen years
            out.data[k] = self.data[k].copy()
            out.data[k]['season'] = tools.make_season(out.data[k].index.month)

            mask = [i in seasons for i in out.data[k]['season']]
            out.data[k] = out.data[k].loc[mask]
            out.data[k] = out.data[k].drop(columns=['season'])
            out.data[k].sort_index(inplace=True)

        out.status['season'] = [dcts.dict_season()[f'name_{s}'] for s in seasons]
        if inplace: 
            self.__dict__.update(out.__dict__)
        return out

    def sel_flight(self, flights, verbose=False, inplace:bool=False):
        """ Returns Caribic object containing only data for selected flights
            flight_list (int / list) """
        if not hasattr(self, 'flights'):
            raise NotImplementedError(f'Flight selection not available for {self.source}.')
        if isinstance(flights, int): flights = [flights]
        # elif isinstance(flights, range): flights = list(flights)
        invalid = [f for f in flights if f not in self.flights]
        if len(invalid) > 0 and verbose:
            print(f'No data found for flights {invalid}. Proceeding without.')
        flights = [f for f in flights if f in self.flights]

        out = type(self).__new__(self.__class__)  # create new class instance
        for attribute_key in self.__dict__:  # copy stuff like pfxs
            out.__dict__[attribute_key] = copy.deepcopy(self.__dict__[attribute_key])
        # very important so that self.data doesn't get overwritten
        out.data = self.data.copy()

        df_list = [k for k in self.data
                   if isinstance(self.data[k], pd.DataFrame)
                   and 'Flight number' in self.data[k].columns]  # list of all datasets to cut
        for k in df_list:  # delete everything but selected flights
            out.data[k] = out.data[k][
                out.data[k]['Flight number'].isin(flights)]
            out.data[k].sort_index(inplace=True)

        # out.flights = flights  # update to chosen & available flights
        out.years = list(set(out.data['df'].index.year))
        out.years.sort()
        # out.flights.sort()

        if inplace: 
            self.__dict__.update(out.__dict__.copy())
        return out

# --- Make selections based on strato / tropo characteristics --- 
    def sel_atm_layer(self, atm_layer: str, tp=None, inplace:bool=False, **kwargs):
        """ Create GlobalData object with strato / tropo sorting.

        Parameters:
            atm_layer (str): atmospheric layer: 'tropo' or 'strato'

            key tp_def (str): 'chem', 'therm' or 'dyn'
            key crit (str): 'n2o', 'o3'
            key coord (str): 'pt', 'dp', 'z'
            key pvu (float): 1.5, 2.0, 3.5
            key limit (float): pre-flag limit for chem. TP sorting
        """
        out = type(self).__new__(self.__class__)  # create new class instance
        for attribute_key in self.__dict__:  # copy stuff like pfxs
            out.__dict__[attribute_key] = copy.deepcopy(self.__dict__[attribute_key])

        if not tp: 
            if 'source' not in kwargs and self.source in ['Caribic', 'EMAC']:
                kwargs.update({'source': self.source})
    
            if 'rel_to_tp' not in kwargs and any(
                    c.rel_to_tp for c in self.get_coords(**kwargs)):
                kwargs.update({'rel_to_tp': True})
    
            [tp] = self.get_coords(**kwargs)

        if 'df_sorted' not in self.data:
            df_sorted = self.create_df_sorted(save=False, **kwargs)
        else:
            df_sorted = self.df_sorted

        if f'{atm_layer}_{tp.col_name}' not in df_sorted:
            raise KeyError(f'Could not find {tp.col_name} in df_sorted.')

        atm_layer_col = f'{atm_layer}_{tp.col_name}'

        # Filter all dataframes to only include indices in df_sorted
        df_list = [k for k in self.data
                   if isinstance(self.data[k], pd.DataFrame)
                   and k not in ['df_sorted']]  # list of all datasets to cut
        for k in df_list:
            out.data[k] = out.data[k][out.data[k].index.isin(df_sorted.index)]
            out.data[k] = out.data[k][df_sorted[atm_layer_col].loc[out.data[k].index]]

        if self.source == 'Caribic':
            out.pfxs = [k for k in out.data if k in self.pfxs]

        # update object status
        out.status.update({atm_layer: True})
        if 'TP filter' in out.status:
            out.status['TP filter'] = (out.status['TP filter'], atm_layer_col)
        else:
            out.status['TP filter'] = atm_layer_col

        if inplace: 
            self.__dict__.update(out.__dict__)
        return out

    def sel_tropo(self, tp=None, inplace:bool=False, **kwargs):
        """ Returns GlobalData object containing only tropospheric data points. """
        return self.sel_atm_layer('tropo', tp, inplace=inplace, **kwargs)

    def sel_strato(self, tp=None, inplace:bool=False, **kwargs):
        """ Returns GlobalData object containing only tropospheric data points. """
        return self.sel_atm_layer('strato', tp, inplace=inplace, **kwargs)

    def sel_LMS(self, tp=None, nr_of_bins = 3, inplace:bool=False, **kwargs): #!!!
        """ Returns GlobalData object containing only lowermost stratospheric data points. """
        strato_data = self.sel_strato(tp, inplace=inplace, **kwargs).df
        
        zbsize = tp.get_bsize() if not kwargs.get('zbsize') else kwargs.get('zbsize')
        
        LMS_data = strato_data[strato_data[tp.col_name] <= zbsize * nr_of_bins]
        
        return LMS_data

