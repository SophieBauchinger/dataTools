# -*- coding: utf-8 -*-
""" Data import for different formats

@Author: Sophie Bauchinger, IAU
@Date: Tue May 27 11:36:54 2025

"""
import copy
import datetime as dt
import dill
import geopandas
from metpy.units import units
from metpy import calc
import numpy as np
import pandas as pd
from pathlib import Path
from shapely.geometry import Point
import traceback
import warnings
import xarray as xr

# from toolpac.conv.times import datetime_to_fractionalyear as dt_to_fy  # type: ignore
from toolpac.readwrite import find
from toolpac.readwrite.FFI1001_reader import FFI1001DataReader # type: ignore

from dataTools import tools
import dataTools.dictionaries as dcts

#%% Create new coordinates
def calc_coordinates(df, recalculate=False, verbose=False): # Calculates mostly tropopause coordinates
    """ Calculate coordinates as specified through .var1 and .var2. """
    all_calc_coords = dcts.get_coord_df().dropna(subset=['var1', 'var2'], how = 'any').col_name.values
    if recalculate: # Toggle recalculating existing coordinates
        df.drop(columns = [c.col_name for c in df.columns if c.col_name in all_calc_coords],
                inplace=True)

    # Firstly calculate geopotential height from geopotential
    geopot_coords = [c for c in all_calc_coords if (
        c.var1 in df.columns and str(c.var2) == 'nan' )]
    
    for coord in geopot_coords: 
        met_df = df[coord.var1].values * units(dcts.get_coord(coord.var1).unit)
        height_m = calc.geopotential_to_height(met_df) # meters
        height_km = height_m * 1e-3
        
        if coord.unit == 'm': 
            df[coord.col_name] = height_m
        elif coord.unit == 'km': 
            df[coord.col_name] = height_km

    # Now calculate TP / distances to TP coordinates 
    calc_coords = [c for c in all_calc_coords if 
        all(col in df.columns for col in [c.var1, c.var2])]
    
    for coord in calc_coords: 
        if verbose: 
            print('Calculating ', coord.long_name, 'from \n', 
                dcts.get_coord(col_name=coord.var1), '\n', # met
                dcts.get_coord(col_name=coord.var2)) # tp
        
        met_coord = dcts.get_coord(col_name = coord.var1)
        tp_coord = dcts.get_coord(col_name = coord.var2)
        
        met_data = copy.deepcopy(df[coord.var1]) # prevents .df to be overwritten 
        tp_data = copy.deepcopy(df[coord.var2])
        
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
            
                if verbose: 
                    print('UNIT MISMATCH when calculating ', coord.long_name, 'from \n', 
                    dcts.get_coord(col_name=coord.var1), '\n', # met
                    dcts.get_coord(col_name=coord.var2)) # tp
                    
                    print('Fix by readjusting: \n',
                            df[coord.var2].dropna().iloc[0], f' [{tp_coord.unit}] -> ', tp_data.dropna().iloc[0], f' [{coord.unit}]\n', 
                            df[coord.var1].dropna().iloc[0], f' [{met_coord.unit}] -> ', met_data.dropna().iloc[0], f' [{coord.unit}]')
            else: 
                print(f'HALT STOPP: units do not match on {met_coord} and {tp_coord}.')
                continue
        
        coord_data = (met_data - tp_data)
        df[coord.col_name] = coord_data

    return df

# Does not currently work for CARIBIC, presumably because the int_ and _Main shit fucks shit up. Damn
def create_tp_coords(df, verbose=False) -> pd.DataFrame: # TODO: Fix this 
    """ Add calculated relative / absolute tropopause values to .met_data """
    new_coords = dcts.get_coordinates(**{'ID': 'int_calc', 'source': 'Caribic'})
    new_coords = new_coords + dcts.get_coordinates(**{'ID': 'int_calc', 'source': 'CLAMS'})
    new_coords = new_coords + dcts.get_coordinates(**{'ID': 'CLAMS_calc', 'source': 'CLAMS'})
    new_coords = new_coords + dcts.get_coordinates(**{'ID': 'CLAMS_calc', 'source': 'Caribic'})

    success_counter=0; fail_counter=0
    for coord in new_coords:
        # met = tp + rel -> MET - MINUS for either one
        met_col = coord.var1
        met_coord = dcts.get_coord(col_name = met_col)
        minus_col = coord.var2

        if met_col in df.columns and minus_col in df.columns:
            df[coord.col_name] = df[met_col] - df[minus_col]
            success_counter+=1

        elif met_coord.var == 'geopot' and met_col in df.columns:
            met_data = df[met_col].values * units(met_coord.unit) # [m^2/s^2]
            height_m = calc.geopotential_to_height(met_data) # [m]

            if coord.unit == 'm': 
                df[coord.col_name] = height_m
                success_counter+=1
            elif coord.unit == 'km': 
                df[coord.col_name] = height_m * 1e-3
                success_counter+=1
            else: 
                fail_counter+=1

        else:
            fail_counter+=1
            if verbose: print(f'Could not generate {coord.col_name} as precursors are not available')
    print(f"Succesfully calculated {success_counter} coordinates, skipped {fail_counter}")
    return df

#%% Pickled data dictionaries in .data.store

def load_DATA_dict(ID, status, fname="None"): 
    """ Load locally saved data within dataTools from pickled DATA_dict.pkl. """
    if not fname:
        pdir = Path(tools.get_path()+ 'data\\store\\')
        fnames = [i for i in pdir.iterdir() if f'{ID.lower()}_DATA' in i]
        fnames.sort(key=lambda x: x.name[-10:]) # sort by date
        fname = fnames[-1] # get latest file

    filepath = Path(pdir)/fname
    if not filepath.exists(): 
        raise Warning(f"Could not found requested file at {filepath}")

    with open(filepath, 'rb') as f:
        data = dill.load(f)

    if not 'df' in data: 
        print(f"No merged dataframe found for ID {ID}. Check file or call .create_df() to calculate.")

    status.update(dict(path = status.get('path', []) + [filepath])) # add path to status
    return data, status, filepath

#%% TPChange ERA5 / CLaMS reanalysis interpolated onto flight tracks
def process_TPC(ds): 
    """ Preprocess datasets for ERA5 / CLaMS renalayis data from version .04 onwards. 
    
    NB CARIBIC: drop_variables = ['CARIBIC2_LocalTime']
    NB ATom:    drop_variables = ['ATom_UTC_Start', 'ATom_UTC_Stop', 'ATom_End_LAS']

    """
    def flatten_TPdims(ds):
        """ Flatten additional dimensions corresponding to Main / Second / ... Tropopauses.  
        Used for ERA5 / CLaMS reanalysis datasets from version .03
        """
        TP_vars = [v for v in ds.variables if any(d.endswith('TP') for d in ds[v].dims)]
        TP_qualifier_dict = {0 : '_Main', 
                            1 : '_Second', 
                            2 : '_Third'}
        for variable in TP_vars: 
            # get secondary dimension for the current multi-dimensional variable
            [TP_dim] = [d for d in ds[variable].dims if d.endswith('TP')] # should only be a single one!
            for TP_value in ds[variable][TP_dim].values: 
                ds[variable + TP_qualifier_dict[TP_value]] = ds[variable].isel({TP_dim : TP_value})
            ds = ds.drop_vars(variable)
        return ds
    
    # Flatten variables that have multiple tropoause dimensions (thermTP, dynTP)
    ds = flatten_TPdims(ds)
    if "Time" in ds.variables:
        ds = ds.sortby("Time")
        ds = ds.dropna(dim="Time", how = "all")
        ds = ds.dropna(dim="Time", subset = ["Time"])
    else: 
        print("Cannot find variable `Time`, please check the data files. ")

    variables = [v for v in ds.variables if (
        ("N2O" in v or "WOUDC" in v) or v in ["Lat", "Lon", "Theta", "Temp", "Pres", "PAlt"])
        ] + dcts.TPChange_variables()

    return ds[[v for v in variables if v in ds.variables]]

def ds_to_gdf(ds) -> pd.DataFrame:
    """ Convert xarray Dataset to GeoPandas GeoDataFrame: Mostly for TPChange .nc files currently. Can be generalised """
    df = ds.to_dataframe()
    df.reset_index(inplace = True)
    df.rename(columns = {
        **{c : "latitude_degN" for c in ["Lat", "latitude", "lat_degN"]},
        **{c : "longitude_degE" for c in ["Lon", "longitude", "lon_degE"]},
        "PAlt" : "barometric_altitude_m",
        "Pres" : "pressure_hPa",
        "Temp" : "temperature_K",
        "Theta" : "theta_K",
        "Time" : 'Datetime', "time" : "Datetime", 
        }, inplace = True)
    
    df.dropna(subset=['longitude_degE', 'latitude_degN'], how='any', inplace=True)
    geodata = [Point(lat, lon) for lat, lon in zip(
        df['latitude_degN'], df['longitude_degE'])]

    # create geodataframe using lat and lon data from indices
    df.drop([c for c in ['scale', 'P0'] if c in df.columns], axis=1, inplace=True)
    gdf = geopandas.GeoDataFrame(df, geometry=geodata)

    if not gdf.Datetime.dtype == '<M8[ns]':  # mzt, check if time is not in datetime format
        index_time = [dt.datetime(y, 1, 1) for y in gdf.Datetime]
        gdf['Datetime'] = index_time
    gdf = gdf.set_index('Datetime').sort_index()
    gdf.index = gdf.index.floor('s')  # remove micro/nanoseconds
    
    # Convert CLaMS N2O to ppb
    if "CLaMS_N2O" in gdf.columns: 
        gdf["CLaMS_N2O_ppb"] = tools.conv_molarity_PartsPer(gdf["CLaMS_N2O"].values, 'ppb')
        gdf.drop(columns = ["CLaMS_N2O"], inplace = True)

    # WOUDC: Convert Ozone partial pressure to ppb
    if any('O3_mPa' in v for v in gdf.columns): 
        [pPress_col] = [v for v in gdf.columns if 'O3_mPa' in v]
        gdf['O3_ppb'] = tools.conv_pPress_PartsPer(gdf[pPress_col], gdf['pressure_hPa'])
    
    woudc_cols = [c for c in gdf.columns if "WOUDC" in c]
    gdf.rename(columns = {col : col[12:] for col in woudc_cols}, inplace=True) # remove WOUDC_STNxxx prefix

    # Reorder columns
    ordered_cols = list(gdf.columns)
    ordered_cols.sort(key = lambda x: x if not x.startswith(('ERA5', 'CLaMS', 'geo')) else 'z'+x)

    gdf = gdf[[c for c in ordered_cols if not c == "RH_%"]]

    return gdf

def get_TPChange_gdf(fname_or_pdir): 
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

def import_era5_data(ID:str, fnames:str=None, single_year=None) -> pd.DataFrame:
    """ Creates dataframe for ERA5/CLaMS data from netcdf files. 
    Params:
        single_year (int): Only appplies to CARIBIC
    """
    if fnames is None:
        met_pdir = r'E:/TPChange/'
        campaign_dir_dict = { # campaign_pdir, version
            'CAR'  : 'CaribicTPChange',
            'SHTR' : 'SouthtracTPChange',
            'WISE' : 'WiseTPChange',
            'ATOM' : 'AtomTPChange',
            'HIPPO': 'HippoTPChange',
            'PGS'  : 'PolstraccTPChange',
            'PHL'  : 'PhileasTPChange',
            } # HPS now in OZONE_SONDES
        campaign_pdir = met_pdir+campaign_dir_dict[ID]
        
        fnames = campaign_pdir + "/*.nc"
        if ID in ['CAR']: # organised by year!
            fnames = campaign_pdir + "/2*/*.nc"
            if single_year is not None: 
                fnames = campaign_pdir + f"/{single_year}/*.nc"
   
    drop_variables = {'CAR' : ['CARIBIC2_LocalTime'],
                      'ATOM' : ['ATom_UTC_Start', 'ATom_UTC_Stop', 'ATom_End_LAS']}

    # extract data, each file goes through preprocess first to filter variables & convert units
    with xr.open_mfdataset(fnames, 
                           preprocess = process_TPC,
                           drop_variables = drop_variables.get(ID),
                           ) as ds:
        ds = ds
    met_df = ds_to_gdf(ds)
    return met_df

#%% MOZAIC (IAGOS-Core) with interpolated ERA5 variables
UNITS_MOZAIC= {
    'time' : 'seconds since 2000-01-01 00:00', #  UTC
    'lat' : 'degN', 
    'lon' : 'degE',
    'press' : 'Pa',
    'height' : 'm',
    'temp'  : 'K',
    'eqlat' : 'degN',
    'pv' : 'PVU',
    'trop1_z' : 'km',
    'trop2_z' : 'km',
    'trop1_theta' : 'K',
    'trop2_theta' : 'K',
    'o3' : 'ppb',
    'co' : 'ppb',
    }

def get_mozaic_ds(year, month, pdir=Path(r"E:\TPChange\iagosCoreTPChange")): 
    """ Import and prepare IAGOS x ERA5 data from monthly files. 
    
    Available time range: 1995-05 to 2014-09
    1 = Jan, ... 0 = Dec
    """
    path = pdir / f"mozaic_{year}_{month}_o3_co.nc"
    if not path.exists(): 
        raise Warning(f"Found no IAGOS-Core / Mozaic data for {year}-{month}. ")
    
    # Import data and get prep pressure_hPa/theta_K
    with xr.open_dataset(path) as ds: 
        ds = ds
    # if not "readable time": 
    #     ds.time.attrs["unit"] = "seconds since 2000-01-01"
    #     ds = xr.decode_cf(ds)
    ds = ds.rename({v:f'{v}_{u}' for v,u in UNITS_MOZAIC.items() if not v=='time'})
    
    ds['pressure_hPa'] = ds['press_Pa'] / 100
    ds = ds.drop_vars("press_Pa")
    ds = tools.add_theta(ds, t_var = 'temp_K', p_var = 'pressure_hPa', theta_name='theta_K') # adds variable 'theta' [K]
    
    # Lower bound for all: -200
    for var in [v for v in ds.variables if not v=='time']: # should be positive
        ds[var] = ds[var].where(ds[var] > -200) 
        
    # Upper bounds: 200 for coords, 1000 for theta
    for var in ['lat_degN', 'lon_degE', 'eqlat_degN', 'theta_K']: # should be between -180 and 180
        ds[var] = ds[var].where(ds[var] < (1000 if var=='theta_K' else 200))

    return ds
    
def get_moazaic_df(year, month, pdir=Path(r"E:\TPChange\iagosCoreTPChange")):
    """ Get prepared dataset and create cleaned dataframe. 
    
    Available time range: 1995-05 to 2014-09
    1 = Jan, ... 0 = Dec
    """
    ds = get_mozaic_ds(year, month, pdir)
    df = ds.to_dataframe(
        ).dropna(subset=['lat_degN', 'lon_degE']
        ).set_index('time'
        ).sort_index()
    return df
    
#%% CARIBIC data import
def CARIBIC_year_data(pfx: str, yr: int, parent_dir: str, verbose: bool) -> tuple[pd.DataFrame, dict]:
    """ Data import for a single year """
    if not any(find.find_dir("*_{}*".format(yr), parent_dir)):
        if verbose: 
            print(f'No data found for {yr}, remove from list of years')
        return pd.DataFrame()

    # Collect data from individual flights for current year
    print(f'Reading Caribic - {pfx} - {yr}')
    df_yr = pd.DataFrame()
    
    for current_dir in find.find_dir("Flight*_{}*".format(yr), parent_dir):  # [1:]:
        flight_nr = int(str(current_dir)[-12:-9])
        f = find.find_file(f'{pfx}_*', current_dir)
        if not f or len(f) == 0:  # no files found
            if verbose: print(f'No {pfx} File found for \
                                Flight {flight_nr} in {yr}')
            continue
        if len(f) > 1:
            f.sort()  # sort to get most recent version with indexing from end

        f_data = FFI1001DataReader(f[-1], df=True, xtype='secofday',
                                    sep_variables=';')
        df_flight = f_data.df  # index = Datetime
        df_flight.insert(0, 'Flight number',
                            [flight_nr] * df_flight.shape[0])

        col_name_dict = rename_AMES_columns(f_data.VNAME)
        # set names to their short version
        df_flight.rename(columns=col_name_dict, inplace=True)
        df_yr = pd.concat([df_yr, df_flight])
    
    if df_yr.empty: 
        print(f'No data found for {pfx} - {yr}!')
        return df_yr

    # Convert longitude and latitude into geometry objects
    lat_col, lon_col = ('lat', 'lon') if pfx!='MS' else ('PosLat', 'PosLong')
    
    geodata = [Point(lon, lat) for lon, lat in zip(
        df_yr[lon_col], df_yr[lat_col])]
    gdf_yr = geopandas.GeoDataFrame(df_yr, geometry=geodata)

    # Drop cols which are saved within datetime, geometry
    if not gdf_yr['geometry'].empty:
        filter_cols = [c for c in gdf_yr.columns 
                        if c in ['TimeCRef', 'year', 'month', 'day',
                        'hour', 'min', 'sec', lon_col, lat_col, 'type']]
        try: #TODO cannot remember what I wanted to achieve here
            del_column_names = [gdf_yr.filter(
                regex='^' + c).columns[0] for c in filter_cols]
            gdf_yr.drop(del_column_names, axis=1, inplace=True)
        except: 
            pass

    return gdf_yr

def CARIBIC_pfx_data(pfx, years, parent_dir, verbose) -> pd.DataFrame:
    """ Data import for chosen prefix. """
    gdf_pfx = geopandas.GeoDataFrame()
    year_tracker = {yr:False for yr in years}
    for yr in years:
        gdf_yr = CARIBIC_year_data(pfx, yr, parent_dir, verbose)
        if not gdf_yr.empty: 
            year_tracker.update({yr:True})
        gdf_pfx = pd.concat([gdf_pfx, gdf_yr])
        # Remove case-sensitive distinction in Caribic data 
        if pfx == 'GHG':
            cols = ['SF6', 'CH4', 'CO2', 'N2O']
            for col in cols + ['d_' + c for c in cols]:
                if col.lower() in gdf_pfx.columns:
                    if not col in gdf_pfx.columns:
                        gdf_pfx[col] = np.nan
                    gdf_pfx[col] = gdf_pfx[col].combine_first(gdf_pfx[col.lower()])
                    gdf_pfx.drop(columns=col.lower(), inplace=True)

        elif pfx == 'MS': 
            MS_cols = ['CO', 'CO2', 'CH4', 'CH4_Err']
            MS_col_dict = {c:'MS_'+c for c in MS_cols}
            gdf_pfx.rename(columns = MS_col_dict, inplace=True)

        # Drop Acetone and Acetonitrile, drop 4.0 PVU Tropopause and AgeSpec variables 
        gdf_pfx.drop(columns=[c for c in gdf_pfx.columns 
                              if c in dcts.remove_from_CARIBIC_variables()],
                     inplace=True)

    return gdf_pfx, year_tracker

def CARIBIC_AMES_data(pfxs, years, verbose=False, parent_dir=None) -> dict:
    """ Imports Caribic data in the form of geopandas dataframes.

    Returns data dictionary containing dataframes for each file source and
    dictionaries relating column names with Coordinate / Substance instances.

    Parameters:
        recalculate (bool): Data is imported from source instead of using pickled dictionary.
        fname (str): specify File name of data dictionary if default should not be used.
        pdir (str): specify Parent directory of source files if default should not be used.
        verbose (bool): Makes function more talkative.
    """
    data_dict = {}
    parent_dir = r'E:\CARIBIC\Caribic2data' if not parent_dir else parent_dir
    print('Importing Caribic Data from remote files.')
    for pfx in pfxs:  # can include different prefixes here too
        gdf_pfx, year_tracker = CARIBIC_pfx_data(pfx, years, parent_dir, verbose)
        if gdf_pfx.empty: print("Data extraction unsuccessful. \
                                Please check your inputs"); return
        data_dict[pfx] = gdf_pfx
    return data_dict, year_tracker

def rename_AMES_columns(columns) -> dict: # prev tools.rename_columns()
    """ Create dictionary relating column name with AMES_variable object

    Relate dataframe column name with all information in

    Get new column names and col_name_dict for AMES data structure.
    Get only short name + unit; description found in coordinate instance for specific col_name
    Standardise names via case changes
    """
    col_name_dict = {}
    for x in columns:
        if len(x.split(';')) == 3:
            col_name, long_name, unit = [i.strip() for i in x.split(';')]
        else:
            col_name = x.split(";")[0].strip()
        col_name_dict.update({x: col_name})
    return col_name_dict

#%% EMAC curtain and flight track data
def process_emac_s4d(ds, incl_model=True, incl_tropop=True, incl_subs=True):
    """ Choose which variables to keep when importing EMAC data .

    Parameters:
        ds: currrent xarray dataset
        incl_subs (bool): keep tracer substances
        incl_model (bool): keep modelled meteorological data
        incl_tropop (bool): keep tropopause-relevant variabels

    Variable description:
        time - datetime [ns]
        tlon - track longitude [degrees_east]
        tlat - track latitude [degrees_north]
        tpress - track pressure [hPa]
        tps - track surface pressure [Pa]
        tracer_* - modelled substances [mol/mol]
        tropop_* - tropopause relevant variables
        ECHAM5_* - modelled met. data
        e5vdiff_tpot* - potential temperature [K]
    """
    variables = ['time', 'tlon', 'lev', 'tlat', 'tpress', 'tps']
    if incl_model:
        variables.extend([v for v in ds.variables
                          if v.startswith(('ECHAM5_', 'e5vdiff_tpot'))
                          and not v.endswith(('m1', 'aclc'))])
    if incl_tropop:
        variables.extend([v for v in ds.variables
                          if v.startswith('tropop_') and not v.endswith('_f')
                          and not any([x in v for x in ['_clim', 'pblh']])])
    if incl_subs:
        tracers = [s.col_name for s in dcts.get_substances(**{'ID': 'EMAC'})]
        tracers_at_fl = [t + '_at_fl' for t in tracers]
        variables.extend([v for v in ds.variables if
                          (v in tracers or v in tracers_at_fl)])
    # only keep specified variables
    ds = ds[variables]
    for var in ds.variables:  # streamline units
        if hasattr(ds[var], 'units'):
            if ds[var].units == 'Pa':
                ds[var] = ds[var].metpy.convert_units(units.hPa)
            elif ds[var].units == 'm':
                ds[var] = ds[var].metpy.convert_units(units.km)
            ds[var] = ds[var].metpy.dequantify()  # makes units an attribute again
    # if either lon or lat are nan, drop that timestamp
    ds = ds.dropna(subset=['tlon', 'tlat'], how='any', dim='time')
    ds = ds.rename({'tlon': 'longitude', 'tlat': 'latitude'})
    ds['time'] = ds.time.dt.round('S')  # rmvs floating pt errors
    return ds

def process_emac_s4d_s(ds, incl_model=True, incl_tropop=True, incl_subs=True):
    """ Keep only variables that depend only on time and are available in subsampled data """
    ds = process_emac_s4d(ds, incl_model, incl_tropop, incl_subs)
    variables = [v for v in ds.variables if ds[v].dims == ('time',)]
    return ds[variables]

# %% PHILEAS-specific things for the TP paper case studies
# Create curtain from observation df and model ds
def get_curtain_ds(gdf, model_ds, time_interval = "10min", calc_theta=True): 
    """ Create an interpolated curtain dataset on the specified time resolution. 
    Args: 
        model_ds (ERA5 dataset): Assumes variables 'time', 'latitude' and 'longitude'
        gdf (gpd.GeoDataFrame): Time-indexed, geometry-column  
    """
    # 1. Resample the flight data to specified intervals
    gdf_resampled = gdf.resample(time_interval).first().dropna(subset=["geometry"])

    # Extract coordinates and time
    res_times = gdf_resampled.index
    res_lons = gdf_resampled.geometry.x.values
    res_lats = gdf_resampled.geometry.y.values

    # era_times = model_ds.time.values
    curtain_profiles = []

    for i in range(len(res_times)):
        t = res_times[i]
        lon = res_lons[i]
        lat = res_lats[i]

        profile = model_ds.interp(time=t, longitude=lon, latitude=lat)
        profile = profile.expand_dims(time=[t])
        curtain_profiles.append(profile)

    # Combine into single Dataset
    curtain_ds = xr.concat(curtain_profiles, dim="time")
    
    if calc_theta: 
        curtain_ds = tools.add_theta(curtain_ds)
    
    return curtain_ds

# Phileas Fl07 and Fl19 specific imports
def get_phileas_era5():
    """ Get ERA5, UMAQS and FAIRO data from TPChange .nc files"""
    met_pdir = r'E:/TPChange/' + 'PhileasTPChange'
    fnames = met_pdir + "/*.nc"

    def process_ERA5_PHILEAS(ds): 
        """ Preprocess datasets for ERA5 / CLaMS renalayis data for PHILEAS - include BAHAMAS. """
        def flatten_TPdims(ds):
            TP_vars = [v for v in ds.variables if any(d.endswith('TP') for d in ds[v].dims)]
            TP_qualifier_dict = {0 : '_Main', 1 : '_Second', 2 : '_Third'}
            for variable in TP_vars: 
                # get secondary dimension for the current multi-dimensional variable
                [TP_dim] = [d for d in ds[variable].dims if d.endswith('TP')] # should only be a single one!
                for TP_value in ds[variable][TP_dim].values: 
                    ds[variable + TP_qualifier_dict[TP_value]] = ds[variable].isel({TP_dim : TP_value})
                ds = ds.drop_vars(variable)
            return ds
        # Flatten variables that have multiple tropoause dimensions (thermTP, dynTP)
        ds = flatten_TPdims(ds)
        
        # Add flight number column (take everything after first string, then take stuff up until next ;)
        fl_nr = ds.flight_info.split('Flightnumber: F')[1].split(';')[0]
        fl_nr = float(fl_nr.replace('a', '').replace('b', ''))
        
        flight_arr = xr.DataArray(
            fl_nr, 
            coords=ds.coords,  # Use the same coordinates as the existing dataset
            dims=ds.dims,  # Use the same dimension names as the existing dataset
            )
        ds['Flight number'] = flight_arr

        vars = set(list(tools.ERA5_variables()) + ['Flight number']
                   + ['Lat', 'Lon', 'PAlt', 'Pres', 'Theta'] # include Bahamas MET data   
                   + ['PHILEAS_N2O', 'PHILEAS_O3', 'PHILEAS_CO']) # Include observations

        return ds[[v for v in vars if v in ds.variables]]
    
    with xr.open_mfdataset(fnames, preprocess = process_ERA5_PHILEAS) as ds: 
        df = ds.to_dataframe()
    df.dropna(subset = [c for c in df.columns if not c.endswith('_Second')], how = 'any', inplace = True)
        
    df.rename(columns = {
        'Lat' : 'BAHAMAS_LAT',
        'Lon' : 'BAHAMAS_LON',
        'PAlt' : 'BAHAMAS_ALT',
        'Pres' : 'BAHAMAS_PSTAT', # NB_PSIA
        'Theta' : 'BAHAMAS_POT', # source Bahamas?
        }, inplace = True)

    geodata = [Point(lon, lat) for lon, lat in zip(
        df['BAHAMAS_LON'], df['BAHAMAS_LAT'])]
    df = geopandas.GeoDataFrame(df, geometry=geodata)
    
    met_df = df[[c for c in df.columns if not c.startswith('PHILEAS')]]

    df.rename(columns = dict(
        PHILEAS_CO = 'UMAQS_CO',
        PHILEAS_N2O = 'UMAQS_N2O',
        PHILEAS_O3 = 'FAIRO_O3'), 
                inplace = True)

    data_dictionary = {
        'met_data' : met_df, 
        'df' : df
        }
    return data_dictionary

def get_phileas_data_fl07_fl19(time_res = '10s'): 
    """ Temporary function for creating merge files for the PHILEAS campaign. """  
    # GHOST_ECD
    fname = r"C:\Users\sophie_bauchinger\Documents\GitHub\dataTools\dataTools\misc_data\PHILEAS\PHILEAS_F07_Frankfurt_20230821_HALO_GHOST_ECD_v1.csv"
    ghost_7 = FFI1001DataReader(fname, df=True, xtype='secofday').df
    ghost_7['Flight number'] = 7
    fname = r"C:\Users\sophie_bauchinger\Documents\GitHub\dataTools\dataTools\misc_data\PHILEAS\PHILEAS_F19_Solingen_20230922_HALO_GHOST_ECD_v1.csv"
    ghost_19 = FFI1001DataReader(fname, df=True, xtype='secofday').df
    ghost_19['Flight number'] = 19
    ghost = pd.concat([ghost_7, ghost_19])
    ghost = ghost.drop(columns = ['Mean', 'Time_Start', 'Time_End'])
    ghost.index = ghost.index.round('s')
    ghost.dropna(how = 'all', inplace = True)
    
    # FAIRO
    fname = r"C:\Users\sophie_bauchinger\Documents\GitHub\dataTools\dataTools\misc_data\PHILEAS\PHILEAS_F07a_2023-08-21_HALO_FAIRO_O3_V02.ames"
    fairo_7a = FFI1001DataReader(fname, df=True, xtype='secofday').df
    fairo_7a['Flight number'] = 7
    fname = r"C:\Users\sophie_bauchinger\Documents\GitHub\dataTools\dataTools\misc_data\PHILEAS\PHILEAS_F07b_2023-08-21_HALO_FAIRO_O3_V02.ames"
    fairo_7b = FFI1001DataReader(fname, df=True, xtype='secofday').df
    fairo_7b['Flight number'] = 7
    fname = r"C:\Users\sophie_bauchinger\Documents\GitHub\dataTools\dataTools\misc_data\PHILEAS\PHILEAS_F19_2023-09-22_HALO_FAIRO_O3_V02.ames"
    fairo_19 = FFI1001DataReader(fname, df=True, xtype='secofday').df
    fairo_19['Flight number'] = 19
    fairo = pd.concat([fairo_7a, fairo_7b, fairo_19])
    fairo.drop(columns = ['Mid_UTC;'], inplace = True)
    fairo.index = fairo.index.round('s')
    fairo.rename(columns = {c : c.split(';')[0] for c in fairo.columns}, inplace = True)
    
    # UMAQS
    fname = r"C:\Users\sophie_bauchinger\Documents\GitHub\dataTools\dataTools\misc_data\PHILEAS\PHILEAS_F07_Frankfurt_20230821_HALO_UMAQS_v1.ames"
    umaqs_7 = FFI1001DataReader(fname, df=True, xtype='secofday').df
    umaqs_7['Flight number'] = 7
    fname = r"C:\Users\sophie_bauchinger\Documents\GitHub\dataTools\dataTools\misc_data\PHILEAS\PHILEAS_F19_Solingen_20230922a_HALO_UMAQS_v1.ames"
    umaqs_19 = FFI1001DataReader(fname, df=True, xtype='secofday').df
    umaqs_19['Flight number'] = 19
    umaqs = pd.concat([umaqs_7, umaqs_19])
    umaqs.index = umaqs.index.round('s')
    umaqs.drop(columns = ['UTC_seconds;'], inplace = True)
    
    # ERA5 Data
    def process_ERA5_PHILEAS(ds): 
        """ Preprocess datasets for ERA5 / CLaMS renalayis data for PHILEAS - include BAHAMAS. """
        def flatten_TPdims(ds):
            TP_vars = [v for v in ds.variables if any(d.endswith('TP') for d in ds[v].dims)]
            TP_qualifier_dict = {0 : '_Main', 1 : '_Second', 2 : '_Third'}
            for variable in TP_vars: 
                # get secondary dimension for the current multi-dimensional variable
                [TP_dim] = [d for d in ds[variable].dims if d.endswith('TP')] # should only be a single one!
                for TP_value in ds[variable][TP_dim].values: 
                    ds[variable + TP_qualifier_dict[TP_value]] = ds[variable].isel({TP_dim : TP_value})
                ds = ds.drop_vars(variable)
            return ds
        # Flatten variables that have multiple tropoause dimensions (thermTP, dynTP)
        ds = flatten_TPdims(ds)
        vars = set(list(tools.ERA5_variables()) + ['Lat', 'Lon', 'PAlt', 'Pres', 'Theta']) # include Bahamas MET data
        return ds[[v for v in vars if v in ds.variables]]
    
    era5_7a = r"C:\Users\sophie_bauchinger\Documents\GitHub\dataTools\dataTools\misc_data\PHILEAS\PHILEAS_20230821_F07a_TPC_V04.nc"
    era5_7b = r"C:\Users\sophie_bauchinger\Documents\GitHub\dataTools\dataTools\misc_data\PHILEAS\PHILEAS_20230821_F07b_TPC_V04.nc"
    era5_19 = r"C:\Users\sophie_bauchinger\Documents\GitHub\dataTools\dataTools\misc_data\PHILEAS\PHILEAS_20230922_F19_TPC_V04.nc"
    with xr.open_mfdataset([era5_7a, era5_7b, era5_19], preprocess = process_ERA5_PHILEAS) as ds: 
        era5_ds = ds
    era5 = era5_ds.to_dataframe()
    era5_rename_vars = {
        'Lat' : 'BAHAMAS_LAT',
        'Lon' : 'BAHAMAS_LON',
        'PAlt' : 'BAHAMAS_ALT',
        'Pres' : 'BAHAMAS_PSTAT', # NB_PSIA
        'Theta' : 'BAHAMAS_POT', # source Bahamas?
        }
    # era5_rename_tps = {c:c[:-5] for c in era5.columns if '_Main' in c}
    # era5_rename = dict(era5_rename_vars, **era5_rename_tps)
    era5.rename(columns = era5_rename_vars, inplace=True)

    # Interpolate onto 
    times = era5.resample(time_res).mean().index
    umaqs_resampled = tools.interpolate_onto_timestamps(umaqs, times)
    ghost_resampled = tools.interpolate_onto_timestamps(ghost, times)
    fairo_resampled = tools.interpolate_onto_timestamps(fairo, times)
    era5_resampled = tools.interpolate_onto_timestamps(era5, times)
    
    for instr, df in {'UMAQS' : umaqs_resampled, 
                      'GHOST_ECD' : ghost_resampled, 
                      'FAIRO' : fairo_resampled}.items():
        for col in df.columns:
            # if kwargs.get('verbose'):
            #     print(f'Renaming: {col} -> {dcts.harmonise_variables(instr, col)}')
            df[dcts.harmonise_variables(instr, col)] = df.pop(col)
   
    msmt_data = pd.concat([umaqs_resampled, ghost_resampled, fairo_resampled], axis = 'columns').dropna(how = 'all')
    era5_resampled.drop([i for i in times if i not in msmt_data.index], inplace = True)
    df_resampled = pd.concat([msmt_data, era5_resampled], axis = 'columns') # this results in duplicate values (most likely)

    geodata = [Point(lon, lat) for lon, lat in zip(
        df_resampled['BAHAMAS_LON'], df_resampled['BAHAMAS_LAT'])]
    df = geopandas.GeoDataFrame(df_resampled, geometry=geodata)
    df = df[[c for c in df.columns if c not in ['C2H6', 'CFC12']]][df['BAHAMAS_LAT'].notna()]
    # df.rename(columns = {c:c[4:] for c in df.columns if 'BAHAMAS' in c}, inplace = True)
    
    # Quite possibly the ugliest way of dealing with duplicate flight number columns but behold it works: 
    def same_merge(x): return np.nanmin(x)
    flight_nr = df['Flight number'].T.groupby(level=0).apply(lambda x: x.apply(same_merge,)).T
    df.drop(columns = ['Flight number'], inplace = True)
    df['Flight number'] = flight_nr

    # Create output     
    data_dictionary = {
        'GHOST' : ghost_resampled, 
        'UMAQS' : umaqs_resampled, 
        'FAIRO' : fairo_resampled, 
        'met_data' : era5_resampled, 
        'df' : df,
    }
    return data_dictionary

def get_HrelTP_fl07_fl19_from_file(): 
    """ """
    fnames = [
        r"data\PHILEAS_FAIRO-UV_w_HrelTP\PHILEAS_F07a_2023-08-21_HALO_FAIRO_O3_V02.csv",
        r"data\PHILEAS_FAIRO-UV_w_HrelTP\PHILEAS_F07b_2023-08-21_HALO_FAIRO_O3_V02.csv",
        r"data\PHILEAS_FAIRO-UV_w_HrelTP\PHILEAS_F19_2023-09-22_HALO_FAIRO_O3_V02.csv"
        ]
    
    paths = [dcts.get_path() + fname for fname in fnames]
    
    df = pd.DataFrame()
    
    for path in paths: 
        df_flight = pd.read_csv(path, sep = ';')
        df = pd.concat([df, df_flight])
    df.rename(columns = {"Unnamed: 0": 'Datetime'}, inplace = True)
    df['Datetime'] = pd.to_datetime(df['Datetime'], format = 'ISO8601')
    df.set_index('Datetime', inplace = True)
    df.index = df.index.round('s')
    return df

