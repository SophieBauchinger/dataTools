# -*- coding: utf-8 -*-
"""
@Author: Sophie Bauchimger, IAU
@Date: Tue May  9 15:39:11 2023

Dictionaries for finding fctnbs, col names, v lims, default unit
"""
from toolpac.outliers import ol_fit_functions as fct

def get_fct_substance(substance):
    """ Returns appropriate fct from toolpac.outliers.ol_fit_functions to a substance """
    df_func_dict = {'co2': fct.higher,
                    'ch4': fct.higher,
                    'n2o': fct.simple, 
                    'sf6': fct.quadratic, 
                    'trop_sf6_lag': fct.quadratic, 
                    'sulfuryl_fluoride': fct.simple, 
                    'hfc_125': fct.simple, 
                    'hfc_134a': fct.simple, 
                    'halon_1211': fct.simple, 
                    'cfc_12': fct.simple, 
                    'hcfc_22': fct.simple, 
                    'int_co': fct.quadratic}
    return df_func_dict[substance.lower()]

def get_col_name(substance, source, c_pfx='GHG', CLaMS=True):
    """ 
    Returns column name for substance as saved in dataframe 
        source (str) 'Caribic', 'Mauna_Loa', 'Mace_Head', 'Mozart' 
        substance (str): sf6, n2o, co2, ch4
    """
    cname=None

    if source=='Caribic' and c_pfx == 'GHG': # caribic / ghg
        col_names = { 
            'sf6': 'SF6 [ppt]',
            'n2o': 'N2O [ppb]',
            'no' : 'NO [ppb]',
            'noy': 'NOy [ppb]',
            'no2': 'NO2 [ppb]',
            'co' : 'CO [ppm]',
            'co2': 'CO2 [ppb]',
            'ch4': 'CH4 [ppb]'}

    elif source=='Caribic' and c_pfx=='INT': # caribic / int
        col_names = {
            # CH4 [ppb]
            # d_CH4 [ppb]
            # CO2 [ppm]
            # d_CO2 [ppm]
            # N2O [ppb]
            # d_N2O [ppb]
            # SF6 [ppt]
            # d_SF6 [ppt]
            'co' : 'int_CO [ppb]',
            'o3' : 'int_O3 [ppb]',
            'h2o': 'int_H2O_gas [ppm]',
            'no' : 'int_NO [ppb]',
            'noy': 'int_NOy [ppb]',
            'co2': 'int_CO2 [ppm]',
            'ch4': 'int_CH4 [ppb]'
            # 'ch4_clams' : 'int_CLaMS_CH4 [ppb]', 
            # 'co_clams' : 'int_CLaMS_CO [ppb]', 
            # 'co2_clams' : 'int_CLaMS_CO2 [ppm]',
            # 'f11_clams' : 'int_CLaMS_F11 [ppt]', 
            # 'f12_clams' : 'int_CLaMS_F12 [ppt]', 
            # 'h2o_clams' : 'int_CLaMS_H2O [ppm]',
            # 'n2o' : 'int_CLaMS_N2O [ppb]', 
            # 'o3_clams' : 'int_CLaMS_O3 [ppb]', 
            }

    elif source=='Caribic' and c_pfx=='INT2': # caribic / int2
        col_names = {
            # CH4 [ppb]
            # d_CH4 [ppb]
            # CO2 [ppm]
            # d_CO2 [ppm]
            # N2O [ppb]
            # d_N2O [ppb]
            # SF6 [ppt]
            # d_SF6 [ppt]
            'noy': 'int_CARIBIC2_NOy [ppbv]',
            'no' : 'int_CARIBIC2_NO [ppbv]',
            'ch4': 'int_CLaMS_CH4 [ppb]',
            'co' : 'int_CLaMS_CO [ppb]',
            'co2': 'int_CLaMS_CO2 [ppm]',
            'h2o': 'int_CLaMS_H2O [ppm]',
            'n2o': 'int_CLaMS_N2O [ppb]',
            'o3' : 'int_CLaMS_O3 [ppb]'}

    elif source=='Mauna_Loa': # mauna loa. monthly or daily median
        col_names = {
            'sf6': 'SF6catsMLOm',
            'n2o': 'N2OcatsMLOm',
            'co2': 'CO2catsMLOm',
            'ch4': 'CH4catsMLOm'}

    elif source=='Mace_Head': # mace head
        col_names={'sf6': 'SF6 [ppt]',
                   'ch2cl2': 'CH2Cl2 [ppt]'}
        
    elif source=='Mozart': # mozart
        col_names = {'sf6': 'SF6'}

    try: cname = col_names[substance.lower()]
    except: print(f'Column name not found for {substance} in {source}'); return None
    return cname

def get_coord_name(coord, source, c_pfx='INT', CLaMS=True):
    """ Get name of eq. lat, rel height wrt therm/dyn tp, ..."""

    if source=='Caribic' and c_pfx=='INT': # caribic / int
        pass
        col_names = {
            'p' : 'p [mbar]',
            'h_rel_tp' : 'int_h_rel_TP [km]',
            'pv' : 'int_PV [PVU]',
            'to_air_tmp' : 'int_ToAirTmp [degC]', # Total Air Temperature
            'tpot' : 'int_Tpot [K]', # potential temperature derived from measured pressure and temperature
            'z' : 'int_z_km [km]', # geopotential height of sample from ECMWF
            'dp_tp_therm' : 'int_dp_strop_hpa [hPa]', # pressure difference relative to thermal tropopause from ECMWF
            'dp_tp_dym' : 'int_dp_dtrop_hpa [hPa]', # pressure difference relative to dynamical (PV=3.5PVU) tropopause from ECMWF
            'pt_rel_therm' : 'int_pt_rel_sTP_K [K]', #  potential temperature difference relative to thermal tropopause from ECMWF
            'pt_rel_dyn' : 'int_pt_rel_dTP_K [K]', #  potential temperature difference relative to  dynamical (PV=3.5PVU) tropopause from ECMWF
            'z_rel_therm' : 'int_z_rel_sTP_km [km]', # geopotential height relative to thermal tropopause from ECMWF
            'z_rel_dyn' : 'int_z_rel_dTP_km [km]', # geopotential height relative to dynamical (PV=3.5PVU) tropopause from ECMWF
            'eq_lat' : 'int_eqlat [deg]', # equivalent latitude in degrees north from ECMWF
            # 'int_AgeSpec_AGE [year]',
            # 'int_AgeSpec_MODE [year]', 
            # 'int_AgeSpec_MEDIAN_AGE [year]',
            # 'int_CARIBIC2_H_rel_TP; H_rel_TP; replacement for H_rel_TP; [km]; [km]\n',
            # 'int_ERA5_PV [PVU]', 
            # 'int_Theta [K]', 
            # 'int_ERA5_PRESS [hPa]',
            # 'int_ERA5_TEMP [K]', 
            # 'int_ERA5_EQLAT [deg N]',
            # 'int_ERA5_TROP1_PRESS [hPa]', 
            # 'int_ERA5_TROP1_THETA [K]',
            }

    elif source=='Caribic' and c_pfx=='INT2': # caribic / int2
        col_names = {
            'p' : 'p [mbar]', # pressure (mean value) 
            'h_rel_tp' : 'int_CARIBIC2_H_rel_TP [km]', # H_rel_TP; replacement for H_rel_TP
            'pv' : 'int_ERA5_PV [PVU]', 
            'theta' : 'int_Theta [K]', # Potential temperature
            'p_era5' : 'int_ERA5_PRESS [hPa]', # Pressure (ERA5)
            't' : 'int_ERA5_TEMP [K]', # Temperature (ERA5)
            'eq_lat' : 'int_ERA5_EQLAT [deg N]', # Equivalent latitude (ERA5)
            'tp_p': 'int_ERA5_TROP1_PRESS [hPa]', # Pressure of local lapse rate tropopause (ERA5)
            'tp_theta' : 'int_ERA5_TROP1_THETA [K]', # Pot. temperature of local lapse rate tropopause (ERA5)
            'mean_age' : 'int_AgeSpec_AGE [year]',
            'modal_age' : 'int_AgeSpec_MODE [year]', 
            'median_age' : 'int_AgeSpec_MEDIAN_AGE [year]'
            }

    elif source=='Mozart': # mozart
        col_names = {
            'sf6': 'SF6'}

    try: cname = col_names[coord.lower()]
    except: print(f'Column name not found for {coord} in {source}'); return None
    return cname

def get_vlims(substance):
    """ Get default limits for colormaps per substance """
    v_limits = {
        'sf6': (6,9),
        'n2o': (0,10),
        'co2': (0,10),
        'ch4': (0,10)}
    return v_limits[substance.lower()]

def get_default_unit(substance):
    unit = {
        'sf6': 'ppt',
        'n2o': 'ppb',
        'co2': 'ppm',
        'ch4': 'ppb'}
    return unit[substance.lower()]

#%% Input choice and validation

def validated_input(prompt, valid_values):
    valid_input = False
    while not valid_input:
        value = input(prompt)
        if int(value) in valid_values: 
            yn = input(f'Confirm your choice ({value}): Y/N \n')
            if yn.upper() =='Y': valid_input = int(value) in valid_values
            else: value = None; pass

        try: valid_input = int(value) in valid_values
        except: print('')
    return value    

def choose_column(df, var='subs'):
    """ Let user choose one of the available column names """
    choices = dict(zip(range(0, len(df.columns)), df.columns))
    for k, v in choices.items(): print(k, ':', v)
    x = validated_input(f'Select a {var} column by choosing a number between 0 and {len(df.columns)}: \n', choices.keys())
    return choices[int(x)]
