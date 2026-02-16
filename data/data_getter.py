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
import re
from shapely.geometry import Point
import traceback
import warnings
import xarray as xr

# from toolpac.conv.times import datetime_to_fractionalyear as dt_to_fy  # type: ignore
from toolpac.readwrite import find
from toolpac.readwrite.FFI1001_reader import FFI1001DataReader # type: ignore

from dataTools import tools
import dataTools.dictionaries as dcts

#%% Create new coordinates [height and tropopause-relative]
def calc_coordinates(df, recalculate=False, verbose=False): 
    """ Calculate coordinates as specified through .var1 and .var2. """    
    df = copy.deepcopy(df)
    
    current_coords = []
    for col in df.columns: 
        try: current_coords.append(dcts.get_coord(col))
        except: continue
    
    # TODO: Fix problem if a coord is duplicated (e.g. one with and one without _Main)
    
    def get_var_coord(coords, var1: str): 
        """ Get the correct coordinate from given coords (otherwise col_name is incorrect)"""
        [var] = [c for c in coords if c == dcts.get_coord(var1)]
        return var
   
    if recalculate: # Toggle recalculating existing coordinates
        drop_cols = [c.col_name for c in current_coords if c in dcts.get_coordinates(var1='not_nan')]
        df.drop(columns = drop_cols, inplace=True)
        if verbose: 
            print(f"Dropped {drop_cols} to recalculate")
        [current_coords.remove(dcts.get_coord(col)) for col in drop_cols]

    # Firstly calculate geopotential height from geopotential
    geopot_new = dcts.get_coordinates(var1='not_nan', var2='nan')
    geopot_new = [c for c in geopot_new if dcts.get_coord(c.var1) in current_coords]

    for coord in geopot_new:
        if not recalculate and coord in current_coords: 
            if verbose: print(f"Found`{coord.col_name}`, skipping. ")
            continue
        if verbose: 
            print(f"Calculating `{coord.col_name}` from `{coord.var1}`")
        var1 = get_var_coord(current_coords, coord.var1)
        met_df = df[var1.col_name].values * units(var1.unit)
        height_m = calc.geopotential_to_height(met_df) # meters
        height_km = height_m * 1e-3
        
        if coord.unit == 'm': 
            df[coord.col_name] = height_m
        elif coord.unit == 'km': 
            df[coord.col_name] = height_km
        else: 
            continue
        current_coords += [coord]

    # Now calculate TP / distances to TP coordinates 
    other_new = dcts.get_coordinates(var1='not_nan', var2='not_nan')
    other_new = [c for c in other_new if all(
        dcts.get_coord(var) in current_coords for var in [c.var1, c.var2])]

    for coord in other_new: 
        if not recalculate and coord in current_coords: 
            if verbose: print(f"Found`{coord.col_name}`, skipping. ")
            continue
        if verbose: 
            print(f"Calculating `{coord.col_name}` = `{coord.var1}` - `{coord.var2}`")
        
        met_coord = get_var_coord(current_coords, coord.var1)
        tp_coord = get_var_coord(current_coords, coord.var2)
        
        met_data = copy.deepcopy(df[met_coord.col_name]) # prevents .df to be overwritten 
        tp_data = copy.deepcopy(df[tp_coord.col_name])
        
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
        
        current_coords += [coord]

    return df

def combine_coords(df, verbose=False): 
    """ Combine columns for int_* and *_Main coordinates -> * column. """
    current_coords = []
    for col in df.columns: 
        try: current_coords.append(dcts.get_coord(col))
        except: continue
    
    dupes = [x for x in current_coords if current_coords.count(x) > 1]
    new_cols =  list(set([c.name for c in dupes]))
    if verbose: 
        dupes.sort(key=lambda x: x.name)
        print(f"Combining to make the following columns:\n{(new_cols)}")

    comb_df = pd.DataFrame(index=df.index, columns = new_cols)

    for coord in dupes: 
        same_coords = [c for c in dupes if c==coord]
        for c in same_coords:
            new_series = pd.Series(df[c.col_name], name=coord.name)
            comb_df.update(new_series)
    
    out = df.drop(columns=[c.col_name for c in dupes])
    out = pd.concat([out, comb_df], axis=1)

    return out

#%% Pickled data dictionaries in .data.store
WOUDC_STATION_LIST = [
    'HPS'       # 1.1 GB 
    '007',      # 198 MB
    '012',      # 1.1 GB
    '014',      # 1.5 GB
    '018',      # 1.5 GB
    '021',      # 1.7 GB
    '024',      # 1.1 GB
    '029',      # 432 MB
    '053',      # 1.7 GB
    '067',      # 2.7 GB
    '076',      # 1.5 GB
    '077',      # 1.1 GB
    '089',      # 1.0 GB
    '101',      # 1.2 GB
    '107',      # 516 MB
    # '156',      # TODO
    '190',      # 911 MB
    '205',      #  11 MB
    '233',      # 1.5 GB
    '254',      #  42 MB 
    '256',      # 3.8 GB
    '308',      # 2.6 GB
    '315',      # 2.3 GB
    '328',      # 1.7 GB
    '339',      # 322 MB
    '344',      # 1.6 GB
    '394',      # 431 MB
    '435',      # 225 MB 
    '436',      # 1.3 GB
    '450',      # 272 MB
    '458',      # 1.6 GB
    ]

CAMPAIGN_LIST = [
    'airtoss', 
    'atom', 
    'attrex-awas', 
    'attrex-ucats', 
    'caribic', 
    'envisat-spirale', 
    'esmval', 
    'euplex-asur', 
    'gwlcycle', 
    'hippo', 
    'phileas', 
    'polstracc', 
    'southtrac', 
    'spurt', 
    'start08', 
    'stratoclim', 
    'tacts', 
    'tc4_dc8', 
    'wise'
]

def load_DATA_dict(ID, status=None, fname=None, pdir=None): 
    """ Load locally saved data within dataTools from pickled DATA_dict.pkl.
    
    Arguments: 
        ID (str): short name of the chosen campaign / measurement station. 
        status (dict): For saving information on the flight path. 
        fname (str): Specify a file to load. 
        pdir (str|Path): Specify the parent directory to look for the file. 
     
    For aircraft campaigns, returns a dictionary containing pandas dataframes.
    For ozone sondes measurements, returns a pandas dataframe. 
    """
    # pdir = pdir or Path(tools.get_path())
    pdir = pdir or Path(r"C:\Users\sophie_bauchinger\Documents\GitHub\chemTPanalyses\chemTPanalyses\data\store")
    if not fname:
        fnames = [i for i in Path(pdir).iterdir() if (
            f'{ID.lower()}_DATA' in str(i) or f'{ID}_DATA' in str(i))]
        fnames.sort(key=lambda x: x.name[-10:]) # sort by date
        fname = fnames[-1] # get latest file

    filepath = Path(pdir)/fname
    if not filepath.exists():
        raise FileNotFoundError(f"Could not found requested file at {filepath}")

    with open(filepath, 'rb') as f:
        data = dill.load(f)

    if not 'df' in data and not isinstance(data, pd.DataFrame): 
        print(f"No merged dataframe found for ID {ID}. Check file or call .create_df() to calculate.")

    if status is not None: 
        status.update(dict(path = status.get('path', []) + [filepath.name])) # add fname to status
    return data, status, filepath

def save_DATA_dict(data, ID, pdir=None, woudc=False): 
    """ Save pickled data dictionary to prepare import using load_DATA_dict. 
    Arguments: 
        data (pd.DataFrame or {str:pd.DataFrame})
        ID (str)
    """
    pdir = pdir or Path(r"C:\Users\sophie_bauchinger\Documents\GitHub\chemTPanalyses\chemTPanalyses\data\store")
    fname = f"{ID.lower()}_DATA_{dt.datetime.now().strftime("%y%m%d")}.pkl"
    if woudc: 
        fname = 'stn'+fname
    with open(pdir/fname, 'wb') as f: 
        dill.dump(data, f)
    print(f"Successfully saved dataframe for {('stn' if woudc else '')+ID} as {pdir/fname}")

def load_ozone_sonde_data(stn_ids, status=None, pdir=None): 
    """ Create join dataframe incl. stn/flight info from pickled WOUDC/HPS & TPChange sonde data. """
    pdir = pdir or Path(r"C:\Users\sophie_bauchinger\Documents\GitHub\chemTPanalyses\chemTPanalyses\data\store")

    def update_flight_nr(df, ID):
        df["Flight number"] = df["Flight number"].apply(
            lambda x: f"{ID}_{x}" if not x.startswith(ID) else x)
        return df

    stn_dict = {}
    for stn_id in stn_ids:
        stn_df, status, _ = load_DATA_dict('stn'+stn_id, status, pdir=Path(pdir))
        stn_df = update_flight_nr(stn_df, stn_id)
        stn_dict[stn_id] = stn_df

    station_df = pd.concat(stn_dict.values())
    station_df.drop(columns=['latitude_degN', 'longitude_degE'], inplace=True)

    return stn_dict, station_df, status

def get_binned_path(stn_id, coord):
    """ Specific reference of where station-nbased monthly weighted binned data is saved. """
    pdir = Path(r"C:\Users\sophie_bauchinger\Documents\GitHub\chemTPanalyses\chemTPanalyses\data\output")
    subdir = pdir / coord.name
    if not subdir.exists(): 
        subdir.mkdir()
    fname = f"STN{stn_id}_binned_monthly.pkl"
    return subdir / fname

#%% TPChange ERA5 / CLaMS reanalysis interpolated onto flight tracks
ERA5_VARS = dcts.ERA5_variables()

def process_TPC(ds) -> xr.Dataset: 
    """ Preprocess datasets for ERA5 / CLaMS renalayis data from version .04 onwards. 
    
    NB CARIBIC: drop_variables = ['CARIBIC2_LocalTime']
    NB ATom:    drop_variables = ['ATom_UTC_Start', 'ATom_UTC_Stop', 'ATom_End_LAS']

    """   
    def flatten_TPdims(ds):
        """ Flatten additional dimensions corresponding to Main / Second / ... Tropopauses.  
        Used for ERA5 / CLaMS reanalysis datasets from version .03
        """
        new_vars = {}
        drop_vars = [] # multidimensional TP variables
        for var in ds.data_vars: 
            tp_dims = [d for d in ds[var].dims if d.endswith("TP")]
            if not tp_dims: continue 
            [TP_dim] = tp_dims

            for i, suffix in {0: "_Main", 1: "_Second", 2: "_Third"}.items(): 
                if i in ds[TP_dim]: 
                    new_vars[var + suffix] = ds[var].isel({TP_dim: i})
            drop_vars.append(var)

        return ds.drop_vars(drop_vars).assign(new_vars)

    # Flatten variables that have multiple tropoause dimensions (thermTP, dynTP)
    ds = flatten_TPdims(ds)
    if "Time" in ds.variables:
        ds = ds.sortby("Time")
        ds = ds.dropna(dim="Time", subset = ["Time"])
    else: 
        print("Cannot find variable `Time`, please check the data files. ")

    # Add 'Flight number' column
    flight_info = ds.attrs.get("flight_info") # None if not available
    if "Flightnumber" in flight_info:
        flight_id = flight_info.split("Flightnumber: ")[1][:6]
        flight_nr = "".join(re.findall(r'\d+', flight_id))
        ds['Flight number'] = flight_nr

    variables = [v for v in ds.variables if (
        any(i in v for i in ["N2O", "O3", "WOUDC"]) # WOUDC, N2O and O3 variables are kept
            or v in ["Flight number", "Lat", "Lon", "Theta", 
                     "Temp", "Pres", "PAlt", "horWind", "WindDir",
                     "CLaMS_ST"]) # stratospheric air mass tracer
        ] + ERA5_VARS

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
        "horWind" : "horWind_m_per_s",
        "WindDir" : "WindDir_degrees",
        }, inplace = True)

    df.dropna(subset=['longitude_degE', 'latitude_degN'], how='any', inplace=True)
    # lon = x, lat = y 
    # geodata = [Point(lat, lon) for lat, lon in zip(
    #     df['latitude_degN'], df['longitude_degE'])]
    geodata = [Point(lon, lat) for lon, lat in zip(
        df['longitude_degE'], df['latitude_degN'], )]

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

    woudc_cols = [c for c in gdf.columns if "WOUDC" in c]
    gdf.rename(columns = {col : col[12:] for col in woudc_cols}, inplace=True) # remove WOUDC_STNxxx prefix
    
    # WOUDC: Convert Ozone partial pressure to ppb
    if any(v in ['DWDO3SondeHP_OZONE', 'O3_mPa'] for v in gdf.columns):
        [pPress_col] = [v for v in gdf.columns if v in ['DWDO3SondeHP_OZONE', 'O3_mPa']]
        gdf['O3_ppb'] = tools.conv_pPress_PartsPer(gdf[pPress_col], gdf['pressure_hPa'])

    # Reorder columns
    ordered_cols = list(gdf.columns)
    ordered_cols.sort(key = lambda x: x if not x.startswith(('ERA5', 'CLaMS', 'geo')) else 'z'+x)

    gdf = gdf[[c for c in ordered_cols if not c == "RH_%"]]

    return gdf

def get_TPChange_gdf(fname_or_pdir): 
    """ Returns flattened and geo-referenced dataframe of TPChange data (dir or fname). """
    if Path(fname_or_pdir).is_dir(): 
        fnames = [f for f in fname_or_pdir.glob("*.nc")]
        from dask.diagnostics import ProgressBar
        with xr.open_mfdataset(fnames, 
                               preprocess = process_TPC, 
                               parallel=True) as ds, ProgressBar(): 
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

