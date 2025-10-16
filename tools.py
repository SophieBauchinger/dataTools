# -*- coding: utf-8 -*-

""" Auxiliary functions for data extraction and handling.

@Author: Sophie Bauchinger, IAU
@Date: Fri Apr 28 09:51:49 2023

Functions: 
    # --- Data extraction --- 
    time_mean(df, f, first_of_month, minmax)
    ds_to_gdf(ds)
    rename_columns(columns)
    
    # --- EMAC data handling --- 
    process_emac_s4d(ds, incl_model, incl_tropop, incl_subs)
    process_emac_s4d_s(ds, incl_model, incl_tropop, incl_subs)
    
    # --- TPChange ERA5 / CLaMS reanalysis ---
    ERA5_variables()
    process_TPC_V02(ds)
    process_TPC(ds)
    interpolate_onto_timestamps(dataframe, times, prefix)

    # --- Data selection ---
    minimise_tps(tps, vcoord)
    
    # --- Data handling ---
    make_season(month)
    assign_t_s(df, TS, coordinate, tp_val)
    get_lin_fit(series, degree)
    pre_flag(data_arr, ref_arr, crit, limit, **kwargs)
    conv_molarity_PartsPer(x, unit)
    conv_PartsPer_molarity(x, unit)
    
    # --- Plotting helpers ---
    add_zero_line(ax, axis)
    
    # --- Binning of geodataframes ---
    bin_1d(glob_obj, subs, **kwargs)
    bin_2d(glob_obj, subs, **kwargs)
    
    # --- Create animations ---
    make_gif(pdir, fnames)
      
"""

import datetime as dt
import dill
import geopandas
import glob
import numpy as np
from pathlib import Path
import pandas as pd
from PIL import Image
from scipy import stats
from scipy.ndimage import zoom, gaussian_filter
import traceback
import warnings
import xarray as xr

import toolpac.calc.binprocessor as bp # type: ignore
from toolpac.conv.times import datetime_to_fractionalyear as dt_to_fy # type: ignore

import dataTools.dictionaries as dcts

# %% 
def get_path():
    return dcts.get_path()

# %% Data extraction
def time_mean(df, f, first_of_month=True, minmax=False) -> pd.DataFrame:
    """ Group values by time and return the respective averages.
    Parameters:
        df (pd.DataFrame): Input data to be averaged
        f (str): Frequency. One of 'D', 'M', 'Y'
        first_of_month (bool): Set day column to 1
        minmax (bool): Return dataframe containing mean, max and min of the grouped data
    """
    # Only average substance columns
    cols = [c for c in df.columns if c in [s.col_name for s in dcts.get_substances()]]
    df = df[cols]

    df_mean = df.groupby(pd.PeriodIndex(df.index, freq=f)).mean(numeric_only=True)
    df_mean.reset_index(inplace=True)
    
    if f == 'D': 
        df_mean['Date_Time'] = df_mean['Date_Time'].apply(lambda x: dt.datetime(x.year, x.month, x.day))
    elif f == 'M' and first_of_month: 
        df_mean['Date_Time'] = df_mean['Date_Time'].apply(lambda x: dt.datetime(x.year, x.month, 1))
    else: 
        df_mean['Date_Time'] = df_mean['Date_Time'].apply(lambda x: dt.datetime(x.year, x.month, 15))
        
    df_mean.set_index('Date_Time', inplace=True)
    
    # df_mean['Date_Time', i] = dt.datetime(y, m, d)
    
    # df_mean['Date_Time'] = np.nan
    # for i, (y, m, d) in enumerate(zip(
    #         df_mean.index.year,
    #         df_mean.index.month if f != 'Y' else None,
    #         df_mean.index.day if f == 'D' else (
    #             [1] * len(df_mean.index) if first_of_month else None))):
    #     df_mean['Date_Time', i] = dt.datetime(y, m, d)

    # df_mean.set_index('Date_Time', inplace=True)
    
    if minmax:
        df_min = df.groupby(pd.PeriodIndex(df.index, freq=f)).min(numeric_only=True)
        df_max = df.groupby(pd.PeriodIndex(df.index, freq=f)).max(numeric_only=True)

        df_mean = df_mean.rename(columns={c: 'mean_' + c for c in df_mean.columns})
        df_min = df_min.rename(columns={c: 'min_' + c for c in df_min.columns})
        df_max = df_max.rename(columns={c: 'max_' + c for c in df_max.columns})

        return pd.concat([df_mean, df_min, df_max], axis=1)

    return df_mean

# def ds_to_gdf(ds) -> pd.DataFrame:
#     """ Convert xarray Dataset to GeoPandas GeoDataFrame: Mostly for TPChange .nc files currently. Can be generalised """
#     df = ds.to_dataframe()
#     df.reset_index(inplace = True)
#     df.rename(columns = {
#         **{c : "latitude_degN" for c in ["Lat", "latitude", "lat_degN"]},
#         **{c : "longitude_degE" for c in ["Lon", "longitude", "lon_degE"]},
#         "PAlt" : "barometric_altitude_m",
#         "Pres" : "pressure_hPa",
#         "Temp" : "temperature_K",
#         "Theta" : "theta_K",
#         "Time" : 'Datetime', "time" : "Datetime", 
#         }, inplace = True)
    
#     df.dropna(subset=['longitude_degE', 'latitude_degN'], how='any', inplace=True)
#     geodata = [Point(lat, lon) for lat, lon in zip(
#         df['latitude_degN'], df['longitude_degE'])]

#     # create geodataframe using lat and lon data from indices
#     df.drop([c for c in ['scale', 'P0'] if c in df.columns], axis=1, inplace=True)
#     gdf = geopandas.GeoDataFrame(df, geometry=geodata)

#     if not gdf.Datetime.dtype == '<M8[ns]':  # mzt, check if time is not in datetime format
#         index_time = [dt.datetime(y, 1, 1) for y in gdf.Datetime]
#         gdf['Datetime'] = index_time
#     gdf.set_index('Datetime', inplace=True).sort_index(inplace=True)
#     gdf.index = gdf.index.floor('s')  # remove micro/nanoseconds
    
#     # Convert CLaMS N2O to ppb
#     if "CLaMS_N2O" in gdf.columns: 
#         gdf["CLaMS_N2O_ppb"] = conv_molarity_PartsPer(gdf["CLaMS_N2O"].values, 'ppb')
#         gdf.drop(columns = ["CLaMS_N2O"], inplace = True)

#     # WOUDC: Convert Ozone partial pressure to ppb
#     if any('O3_mPa' in v for v in gdf.columns): 
#         [pPress_col] = [v for v in gdf.columns if 'O3_mPa' in v]
#         gdf['O3_ppb'] = conv_pPress_PartsPer(gdf[pPress_col], gdf['pressure_hPa'])
    
#     woudc_cols = [c for c in gdf.columns if "WOUDC" in c]
#     gdf.rename(columns = {col : col[12:] for col in woudc_cols}, inplace=True) # remove WOUDC_STNxxx prefix

#     # Reorder columns
#     ordered_cols = list(gdf.columns)
#     ordered_cols.sort(key = lambda x: x if not x.startswith(('ERA5', 'CLaMS', 'geo')) else 'z'+x)

#     gdf = gdf[[c for c in ordered_cols if not c == "RH_%"]]

#     return gdf

# def rename_columns(columns) -> dict:
#     """ Create dictionary relating column name with AMES_variable object

#     Relate dataframe column name with all information in

#     Get new column names and col_name_dict for AMES data structure.
#     Get only short name + unit; description found in coordinate instance for specific col_name
#     Standardise names via case changes
#     """
#     col_name_dict = {}
#     for x in columns:
#         if len(x.split(';')) == 3:
#             col_name, long_name, unit = [i.strip() for i in x.split(';')]
#         else:
#             col_name = x.split(";")[0].strip()
#         col_name_dict.update({x: col_name})
#     return col_name_dict

# def add_theta(ds, t_var = "t", p_var = "isobaricInhPa", theta_name='theta'): 
#     """ Calculate ERA5 potential temperature from pressure and temperature.
#     Unless specified, assumes variables 'isobaricInhPa' and 't' to be present in ds. 
#     """
#     pressure = ds[p_var] * units.hPa
#     temperature = ds[t_var].values * units.kelvin

#     # Make sure pressure info is available for all temperature points
#     p_broadcasted = np.broadcast_to(pressure, temperature.shape)
#     p_broadcasted = p_broadcasted * units.hPa

#     # Calculate potential temperature
#     theta = calc.potential_temperature(p_broadcasted, temperature)

#     # Add to dataset
#     ds[theta_name] = (ds[t_var].dims, theta.magnitude)
#     ds[theta_name].attrs["units"] = "K"
#     ds[theta_name].attrs["long_name"] = "Potential Temperature"
#     return ds


# # TPChange ERA5 / CLaMS reanalysis interpolated onto flight tracks
# def process_TPC(ds): # from V04
#     """ Preprocess datasets for ERA5 / CLaMS renalayis data from version .04 onwards. 
    
#     NB CARIBIC: drop_variables = ['CARIBIC2_LocalTime']
#     NB ATom:    drop_variables = ['ATom_UTC_Start', 'ATom_UTC_Stop', 'ATom_End_LAS']

#     """
#     def flatten_TPdims(ds):
#         """ Flatten additional dimensions corresponding to Main / Second / ... Tropopauses.  
#         Used for ERA5 / CLaMS reanalysis datasets from version .03
#         """
#         TP_vars = [v for v in ds.variables if any(d.endswith('TP') for d in ds[v].dims)]
#         TP_qualifier_dict = {0 : '_Main', 
#                             1 : '_Second', 
#                             2 : '_Third'}
#         for variable in TP_vars: 
#             # get secondary dimension for the current multi-dimensional variable
#             [TP_dim] = [d for d in ds[variable].dims if d.endswith('TP')] # should only be a single one!
#             for TP_value in ds[variable][TP_dim].values: 
#                 ds[variable + TP_qualifier_dict[TP_value]] = ds[variable].isel({TP_dim : TP_value})
#             ds = ds.drop_vars(variable)
#         return ds
    
#     # Flatten variables that have multiple tropoause dimensions (thermTP, dynTP)
#     ds = flatten_TPdims(ds)
#     if "Time" in ds.variables:
#         ds = ds.sortby("Time")
#         ds = ds.dropna(dim="Time", how = "all")
#         ds = ds.dropna(dim="Time", subset = ["Time"])
#     else: 
#         print("Cannot find variable `Time`, please check the data files. ")

#     variables = [v for v in ds.variables if (
#         ("N2O" in v or "WOUDC" in v) or v in ["Lat", "Lon", "Theta", "Temp", "Pres", "PAlt"])
#         ] + dcts.TPChange_variables()

#     return ds[[v for v in variables if v in ds.variables]]

# def get_TPChange_gdf(fname_or_pdir): 
    """ Returns flattened and geo-referenced dataframe of TPChange data (dir or fname). """
    if Path(fname_or_pdir).is_dir(): 
        fnames = [f for f in fname_or_pdir.glob("*.nc")]
        with xr.open_mfdataset(fnames, preprocess = process_TPC) as ds: 
            ds = ds
    elif Path(fname_or_pdir).is_file(): 
        with xr.open_dataset(fname_or_pdir) as ds: 
            ds = process_TPC(ds)
    else: 
        raise ValueError(f"Not a valid filepath or parent directory: {fname_or_pdir}")
    try: 
        return ds_to_gdf(ds)
    except Exception: 
        warnings.warn("Could not generate geodata, check your input!" + traceback.format_exc())
        return ds.to_dataframe()

# Interpolation
def interpolate_onto_timestamps(dataframe, times, prefix='') -> pd.DataFrame:
    """ Interpolate met data onto given measurement timestamps. 
    
    Parameters: 
        dataframe (pd.DataFrame): data to be interpolated
        times (array, list): Timestamps to be used for interpolating onto
    """
    if isinstance(dataframe, geopandas.GeoDataFrame):
        dataframe = pd.DataFrame(dataframe[[c for c in dataframe.columns if c not in ['geometry']]])

    # add measurement timestamps to met_data
    new_indices = [i for i in times if i not in dataframe.index]

    expanded_df = pd.concat([dataframe, pd.Series(index=new_indices, dtype='object')])
    expanded_df.drop(columns=[0], inplace=True)
    expanded_df.sort_index(inplace=True)

    try:
        expanded_df.interpolate(method='time', inplace=True, limit=2)  # , limit=500)
    except TypeError:
        print(f'Check if type {type(dataframe)} is suitable for time-wise interpolation!')

    regridded_data = expanded_df.loc[times]  # return only measurement timestamps
    
    # Rename columns using prefix
    regridded_data.rename(columns = {col:prefix+col for col in regridded_data.columns}, 
                          inplace=True)
    
    return regridded_data

# %% Data selection
def minimise_tps(tps, vcoord=None) -> list:
    """ Returns a reduced list of tropopause coordinates.

    1. Remove tps with other vertical coords if vcoord is specified
    2. remove all cpt, combo tp
    3. remove all ECMWF tps
    4. Remove modelled N2O tp
    5. Remove duplicates of O3 tp
    6. Remove 1.5 PVU ERA5 dyn tp
    7. Remove non-relative tps if relative exists in tps
    """
    # 1. Remove all non-tropopause related coordinates
    while len([tp for tp in tps if str(tp.tp_def)=='nan']) > 0: # repeat until none left 
        print('hi')
        [tps.remove(tp) for tp in tps if str(tp.tp_def) == 'nan']
    
    # 2. Remove all coordinates with vcoords other than the specified one
    if vcoord: [tps.remove(tp) for tp in tps if tp.vcoord != vcoord]
    
    [tps.remove(tp) for tp in tps if (
        tp.model in ['EMAC', 'ECMWF'] or
        tp.tp_def in ['cpt', 'combo'] or
        tp.pvu == 1.5
        )]
    
    # check if coord exists with pt, remove if it does
    tp_to_remove = []
    for tp in tps:  # 1
        try:
            dcts.get_coord(vcoord='pt', model=tp.model, source=tp.source,
                           tp_def=tp.tp_def, ID=tp.ID, 
                           pvu=tp.pvu, crit=tp.crit, rel_to_tp=tp.rel_to_tp)
        except KeyError:
            continue
        except ValueError: 
            print('ValueError', dcts.get_coordinates(vcoord='pt', model=tp.model, source=tp.source,
                           tp_def=tp.tp_def,
                           pvu=tp.pvu, crit=tp.crit, rel_to_tp=tp.rel_to_tp)); continue
        if not tp.vcoord == 'pt': tp_to_remove.append(tp)
    [tp_to_remove.append(tp) for tp in tps if tp.tp_def in ['cpt', 'combo']]  # 2
    [tp_to_remove.append(tp) for tp in tps if tp.model in ['ECMWF', 'EMAC']]  # 3
    [tp_to_remove.append(tp) for tp in tps
     if tp.col_name in [tp.col_name for tp in dcts.get_coordinates(tp_def='chem', crit='n2o', model='not_MSMT')]]  # 4
    [tp_to_remove.append(tp) for tp in tps
     if tp.col_name in ['int_CARIBIC2_H_rel_TP', 'int_O3']]
    [tp_to_remove.append(tp) for tp in tps if tp.pvu == 1.5]  # 6
    for tp in tps:  # 7
        try:
            rel_tp = dcts.get_coord(rel_to_tp=True, model=tp.model, source=tp.source,
                                    tp_def=tp.tp_def, ID=tp.ID,
                                    pvu=tp.pvu, crit=tp.crit, vcoord=tp.vcoord)
        except KeyError:
            continue
        except ValueError: 
            print('ValueError2'); continue
        if (any(tp.col_name == rel_tp.col_name for tp in tps)
                and tp.rel_to_tp is False):
            tp_to_remove.append(tp)
    tps = tps.copy()
    for tp in set(tp_to_remove): tps.remove(tp)
    tps.sort(key=lambda x: x.tp_def)
    return tps

def get_shared_indices(data, variables):
    """ Make reference for shared indices of chosen variables / tps. 
    Parameters: 
        data (pd.DataFrame): data to get shared indices from
        variables (list[dcts.Substance|dcts.Coordinate]): 
            data needs to be available for these variables
    """
    prefix = ''
    if not any(v.col_name in data.columns for v in variables):
        prefix = 'tropo_'
    tropo_cols = [prefix + v.col_name for v in variables if prefix + v.col_name in data]
    indices = data.dropna(subset=tropo_cols, how='any').index
    return indices

# %% Data Handling
def make_season(month) -> np.array:
    """ If given array of months, return integer representation of seasons
    1 - spring, 2 - summer, 3 - autumn, 4 - winter """
    season = len(month) * [None]
    for i, m in enumerate(month):
        if m in [3, 4, 5]:
            season[i] = 1  # spring
        elif m in [6, 7, 8]:
            season[i] = 2  # summer
        elif m in [9, 10, 11]:
            season[i] = 3  # autumn
        elif m in [12, 1, 2]:
            season[i] = 4  # winter
    return season

def assign_t_s(df, TS, coordinate, tp_val=0) -> pd.Series:
    """ Returns the bool series of t / s after applying appropriate comparison for a chosen vcoord.

    Parameters:
        df (DataFrame): reference data - e.g. track pressure / TP p distance to track p
        TS (str): 't' / 's';  indicates troposphere / stratosphere
        coordinate (str): dp, pt, z

        tp_val (float): value of tropopause in chosen coordinates. For non-relative coords
    """
    if ((coordinate in ['p', 'dp'] and TS == 't')
            or (coordinate in ['pt', 'z'] and TS == 's')):
        return df.gt(tp_val)

    elif ((coordinate in ['p', 'dp'] and TS == 's')
          or (coordinate in ['pt', 'z'] and TS == 't')):
        return df.lt(tp_val)

    else:
        raise KeyError(f'Strat/Trop assignment undefined for {coordinate}')

def get_lin_fit(series, degree=2, verbose=False) -> np.array:  # previously get_mlo_fit
    """ Given one year of reference data, find the fit parameters for
    the substance (col name) """
    year, month = series.index.year, series.index.month
    t_ref = year + (month - 0.5) / 12  # obtain frac year for middle of the month
    mxr_ref = series.values
    fit = np.poly1d(np.polyfit(t_ref, mxr_ref, degree))
    if verbose: 
        print(f'Fit parameters obtained: {fit}')
    return fit

def pre_flag(data_arr, ref_arr, crit='n2o', limit=0.97, **kwargs) -> pd.DataFrame:
    """ Sort data into strato / tropo based on difference to ground obs.

    Parameters:
        data_arr (pd.Series) : msmt data to be sorted into strato / tropo air, index should be datetime
        ref_arr (pd.Series) : reference data to use for filtering (background)
        crit (str) : substance to use for flagging
        limit (float) : tracer mxr fraction below which air is classified
                        as stratospheric

    Returns: time-indexed dataframe with strato/tropo/pre_flag columns
    """
    data_arr.sort_index(inplace=True)
    df_flag = pd.DataFrame({f'strato_{data_arr.name}': np.nan,
                            f'tropo_{data_arr.name}': np.nan},
                           index=data_arr.index, dtype=object)

    fit = get_lin_fit(ref_arr)
    t_obs_tot = np.array(dt_to_fy(df_flag.index, method='exact'))
    df_flag.loc[data_arr < limit * fit(t_obs_tot),
    (f'strato_{data_arr.name}', f'tropo_{data_arr.name}')] = (True, False)

    df_flag[f'flag_{crit}'] = 0
    df_flag.loc[df_flag[f'strato_{data_arr.name}'] == True, f'flag_{crit}'] = 1

    if kwargs.get('verbose'):
        print('Result of pre-flagging: \n',
              df_flag[f'flag_{crit}'].value_counts())
    return df_flag

def ratio_to_fraction(ratio):
    """ Black magic but it works - thank you Matthew. 
    Calc: ratio = a/b. fraction = a/(a+b)
        (a/b)^-1 = b/a
        b/a + 1 = (a+b)/a
        ((a+b)/a)^-1 = a/(a+b)
    <3
    """
    return ((ratio)**-1 + 1)**-1

def conv_molarity_PartsPer(x, unit):
    """ Convert molarity (mol/mol) to given unit (eg. ppb). """
    factor = {'ppm': 1e6,  # per million
              'ppb': 1e9,  # per billion
              'ppt': 1e12,  # per trillion
              'ppq': 1e15,  # per quadrillion
              }
    # e.g. n2o: 300 ppb, 3e-7 mol/mol
    return x * factor[unit]

def conv_PartsPer_molarity(x, unit):
    """ Convert x from [unit] to molarity (mol/mol) """
    factor = {'ppm': 1e-6,  # per million
              'ppb': 1e-9,  # per billion
              'ppt': 1e-12,  # per trillion
              'ppq': 1e-15,  # per quadrillion
              }
    return x * factor[unit]

def conv_pPress_PartsPer(pPress, airPress): 
    """ Converts partial pressure 'pPress' (mPa) using airPress (hPa) to mxr (ppb) """
    return pPress / airPress * 1.656e4

class LognormFit: 
    """ Holds information on Lognorm Fits on distributions of the given dataset. """
    def __init__(self, data_arr, **kwargs): 
        """ Get lognormal fit for 1D data array. 
        Parameters: 
            data_arr (array): Flattened array of v data
            fit_bins (int or List[float]): Histogram bins to base lognorm fit onto

            key normalise (bool). Toggle normalising the histogram. Defaults to False 
            # key *(shape, loc, scale) (float): Initial guesses for lognorm fit. Optional       
        """
        self.x = data_arr[~np.isnan(data_arr)] 
        self.normalise = kwargs.get('normalise', False)
        self.bins = kwargs.get('fit_bins', 30)
        self.counts, self.bin_edges = np.histogram(self.x, bins=self.bins)
        self._get_fit()
        
    def _get_fit(self) -> tuple[tuple, np.array]: 
        """ Get scipy lognorm fit for the given data. 
        
        Parameters: 
            key normalise (bool). Toggle normalising the histogram. Defaults to False 
            key *(shape, loc, scale) (float): Initial guesses for lognorm fit. Optional   
        """
        bin_center = self.bin_edges[:-1] + np.diff(self.bin_edges) / 2
        self.bin_center = bin_center

        # need to fix loc to zero to be able to calculate the mode etc.  
        self.shape, self.loc, self.scale = stats.lognorm.fit(self.x, floc = 0) 
        self.fit_params = (self.shape, self.loc, self.scale)
        normed_lognorm_fit = stats.lognorm.pdf(bin_center, *self.fit_params)
 
        area = sum(np.diff(self.bin_edges)*self.counts)
        norm_factor = area if not self.normalise else 1

        lognorm_fit = normed_lognorm_fit * norm_factor
        self.lognorm_fit = lognorm_fit
        
        return self.fit_params, lognorm_fit

    def show_fit(self, ax): 
        """ PLot the data and the lognorm fit onto one axis. """
        # Plot the pre-computed histogram (see ax.hist for documentation)
        ax.hist(self.bin_edges[:-1],
                self.bin_edges, 
                weights = self.counts,
                orientation = 'horizontal',
                edgecolor = 'white', 
                alpha = 0.7)
        
        # Plot the lognorm fit on the data       
        ax.plot(self.lognorm_fit, self.bin_center, c = 'k')

        # Indicate the mode and median of the fit
        ax.hlines(self.mode, 0, max(self.lognorm_fit), color = 'k', ls = 'dashed',
                label = 'Mode: {:.2f}'.format(self.mode))
        ax.hlines(self.median, 0, max(self.counts), color = 'g', ls = 'dashed',
                label = 'Median: {:.2f}'.format(self.median))

        # Show sigma and mu of the fit
        sigma_mu = r'$\sigma$ = {:.2f}, '.format(self.sigma) + r'$\mu$ = {:.2f}\n'.format(self.mu)
        ax.legend(title = sigma_mu)

    @property
    def mu(self) -> float: 
        """ Get the mean of the corresponding normal distributed Y = log(X). """
        return np.log(self.median)
    
    @property
    def sigma(self) -> float: 
        """ Get the std. of the corresponding normal distributed Y = log(X). """
        return self.shape

    @property
    def multiplicative_std(self) -> float: 
        """ Multiplicative standard deviation of the lognorm 
        so that with sigma* = e^(sigma) and mu* = e^(mu):

        68.3% interval: [mu* / sigma*, mu* x sigma*]
        95.5% interval: [mu* / (sigma*)^2, mu* x (sigma*)^2]        
        """
        return np.exp(self.sigma)

    @property
    def median(self) -> float: 
        return stats.lognorm.median(*self.fit_params)

    @property
    def mode(self) -> float: 
        """ Get mode of fitted lognorm distribution. """
        return np.exp(self.mu - self.sigma**2)
    
    @property
    def variance(self) -> tuple[float]: 
        """ Get the variance of the lognorm distribution. """
        var = stats.lognorm.var(*self.fit_params)
        var_calc = np.exp(2*(self.mu + self.sigma**2)) - np.exp(2*self.mu + self.sigma**2)
        # return var, var_calc
        return stats.lognorm.var(*self.fit_params)
    
    def stats(self, prec=1) -> pd.Series: 
        """ Returns a pandas series containing the relevant lognorm fit parameters. """

        var_dict = dict(
            Mode = self.mode,
            Median = self.median, 
            Sigma = self.sigma, 
            Std = np.sqrt(self.variance),
            Variance = self.variance, 
            Mult_std = self.multiplicative_std,
            )

        stats_dict = {k:float('{0:.{1}f}'.format(v, prec)) for k,v in var_dict.items()}
        stats_dict['int_68'] = (float('{0:.{1}f}'.format(self.median/self.multiplicative_std, prec)), 
                                float('{0:.{1}f}'.format(self.median*self.multiplicative_std, prec)))
        stats_dict['int_95'] = (float('{0:.{1}f}'.format(self.median/self.multiplicative_std/2, prec)), 
                                float('{0:.{1}f}'.format(self.median*self.multiplicative_std*2, prec)))
        return pd.Series(stats_dict)

class Bin2DFitted(bp.Simple_bin_2d): 
    """ Extending Bin2D class to hold lognorm fits for distributions. """
    def __init__(self, v, x, y, binclassinstance, count_limit=2, **fit_kwargs): 
        super().__init__(v, x, y, binclassinstance, count_limit)
        self.calc_lognorm_fits(**fit_kwargs)
    
    def calc_lognorm_fits(self, **fit_kwargs): 
        """ Add lognormal fits to distribution of values in 3D bins. """
        self.vmean_fit = LognormFit(self.vmean, **fit_kwargs)
        self.vstdv_fit = LognormFit(self.vstdv, **fit_kwargs)
        self.rvstd_fit = LognormFit(self.rvstd, **fit_kwargs)

        return self.vmean_fit, self.vstdv_fit, self.rvstd_fit

class Bin3DFitted(bp.Simple_bin_3d): 
    """ Extending Bin3D class to hold lognorm fits for distributions. """
    def __init__(self, v, x, y, z, binclassinstance, count_limit=2, **fit_kwargs): 
        super().__init__(v, x, y, z, binclassinstance, count_limit)
        self.calc_lognorm_fits(**fit_kwargs)
    
    def calc_lognorm_fits(self, **kwargs): 
        """ Add lognormal fits to distribution of values in 3D bins. """
        self.vmean_fit = LognormFit(self.vmean, **kwargs)
        self.vstdv_fit = LognormFit(self.vstdv, **kwargs)
        self.rvstd_fit = LognormFit(self.rvstd, **kwargs)

        return self.vmean_fit, self.vstdv_fit, self.rvstd_fit

def load_reload_Bin3D_df(action, pickle_obj = None, fname = None):
    """ Either saves Bin3D_df to file or reloads it from there. 
    
    Current files: 
        caribic_Bin3D_df.pkl
        caribic_10s_Bin3D_df.pkl
    """
    
    fname = dcts.get_path() + 'misc_data\\cache\\' + ('caribic_Bin3D_df.pkl' 
                                                      if fname is None else fname)

    if action == 'pickle' and pickle_obj is not None: 
        with open(fname, 'wb') as f: 
            dill.dump(pickle_obj.Bin3D_df, f)
        return pickle_obj.Bin3D_df
    elif action == 'load': 
        with open(fname, 'rb') as f: 
            output = dill.load(f)
        
        # reload LognormFit instances
        for tp_dict in output['strato_Bin3D_dict']: 
            for k,v in tp_dict.items(): 
                v.calc_lognorm_fits(50) 
        for tp_dict in output['tropo_Bin3D_dict']: 
            for k,v in tp_dict.items(): 
                v.calc_lognorm_fits(50)

    return output

# %% Fun with stats
def prep_x_n(x): 
    """ Drop NaN values and get length"""
    x = x.flatten()
    x = x[~np.isnan(x)]
    n = len(x)
    return x,n

def kth_moment(x, c, k):
    """ Calculate a distribution's moments. """
    x,n = prep_x_n(x)
    sum_to_k = sum([(xi-c)**k for xi in x])
    moment = 1/(n) * sum_to_k
    return moment

def mean(x):
    """ raw 2st moment """
    return kth_moment(x, c=0, k= 1)

def variance(x):
    """ Central 2nd moment """
    return kth_moment(x, c = kth_moment(x, c=0, k= 1), k=2)

def skewness(x):
    """ Standardised 3rd moment"""
    return kth_moment(x, c = kth_moment(x, c=0, k= 1), k=3) / kth_moment(x, c = kth_moment(x, c=0, k= 1), k=2)**(3/2)

# %% Plotting tools
def add_zero_line(ax):
    """ Highlight the gridline at 0 for the chosen axis on the given Axes object.

    Call when everything else has been plotted already, otherwise the limits will be messy.
    """
    zero_lines = np.delete(ax.get_ygridlines(), ax.get_yticks() != 0)
    for l in zero_lines:
        l.set_color('k')
        l.set_linestyle('-.')

    if len(zero_lines) == 0:
        xlims = ax.get_xlim()
        ax.hlines(0, *xlims)
        ax.set_xlim(*xlims)

def world(fname = \
    'c:/Users/sophie_bauchinger/Documents/GitHub/libraries/110m_cultural_511/ne_110m_admin_0_map_units.shp'): 
    return geopandas.read_file(fname)

def add_world(axs, fname = \
    'c:/Users/sophie_bauchinger/Documents/GitHub/libraries/110m_cultural_511/ne_110m_admin_0_map_units.shp'): 

    """ Adds country outlines to the given axis with zorder 0 as thin grey lines. """
    world = geopandas.read_file(fname)
    if isinstance(axs, list): 
        for ax in axs: 
            world.boundary.plot(ax=ax, color='grey', linewidth=0.3, zorder=0)
    else: 
        world.boundary.plot(ax=axs, color='grey', linewidth=0.3, zorder=0)

def nan_zoom(input, factor, order=1, mode='grid-constant', grid_mode=False, **kwargs):
    """ The array is zoomed using spline interpolation of the requested order and may contain NaN values.
    Args: 
        input (array_like): The input array
        zoom (float or sequence): The zoom factor along the axes. 
    Based on scipy.ndimage.zoom: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.zoom.html
    """
    array_copy = np.copy(input)
    nan_mask = np.isnan(array_copy)
    array_copy[nan_mask] = 0

    # Create a weight array with 0 at NaN positions and 1 elsewhere
    weights = np.ones_like(input)
    weights[nan_mask] = 0

    # Apply zoom to both arrays
    zoomed_array = zoom(array_copy, factor, order=order, mode=mode, grid_mode=grid_mode, **kwargs)
    zoomed_weights = zoom(weights, factor, order=order, mode=mode, grid_mode=grid_mode, **kwargs)

    # Restore NaN values
    if not all(i == 1 for i in zoomed_weights.flatten()):
        zoomed_array[zoomed_weights == 0] = np.nan

    return zoomed_array

def nan_gaussian_filter(input, sigma, **kwargs):
    """ Multidimensional Gaussian filter for arrays containing NaN values.
    
    Args: 
        input (array_like): The input array
        sigma (scalar / sequence of scalars): Standard deviation for Gaussian kernel. 
    
    Based on scipy.ndimage.gaussian_filter: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_filter.html
    """
    array_copy = np.copy(input)
    nan_mask = np.isnan(array_copy)
    array_copy[nan_mask] = 0

    # Create a weight array with 0 at NaN positions and 1 elsewhere
    weights = np.ones_like(input)
    weights[nan_mask] = 0

    # Apply Gaussian filter to both arrays
    filtered_array = gaussian_filter(array_copy, sigma=sigma, **kwargs)
    filtered_weights = gaussian_filter(weights, sigma=sigma, **kwargs)

    if all(weight == 1 for weight in weights.flatten()): 
        return filtered_array

    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        filtered_array /= filtered_weights
        filtered_array[filtered_weights == 0] = np.nan  # Restore NaNs where weights are 0

    return filtered_array

#%% Miscellaneous
def make_gif(pdir=None, fnames=None): # Animate changes over years
    if not pdir: 
        pdir = r'C:\Users\sophie_bauchinger\sophie_bauchinger\Figures\tp_scatter_2d'
    for vc in ['p', 'pt', 'z']:
        tps = dcts.get_coordinates(vcoord=vc, tp_def='not_nan', rel_to_tp=False)
        for tp in tps:
            # fn = pdir+.format(, '_'+str(year) if year else ''))
            frames = [Image.open(image) for image in glob.glob(f'{pdir}/{tp.col_name}*_*.png')]
            if len(frames)==0: frames = [Image.open(image) for image in glob.glob(f'{pdir}/{tp.col_name[:-1]}*_*.png')]

            # frames = [Image.open(image) for image in glob.glob(f"{pdir}/*.JPG")]
            frame_one = frames[0]
            frame_one.save(f'C:/Users/sophie_bauchinger/sophie_bauchinger/Figures/tp_scatter_2d_GIFs/{tp.col_name}.gif',
                           format="GIF", append_images=frames,
                           save_all=True, duration=200, loop=0)

def gif_from_images(images, output_path='test_output.gif', duration=500):
    """ Saves given images to the output path as a gif. """
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=0
    )

class InitialisationError(Exception): 
    """ Raised when initialisation of a class is not intended. """
    pass
