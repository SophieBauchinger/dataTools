# -*- coding: utf-8 -*-
""" Collection of dictionaries (as functions) for CARIBIC analysis.

@Author: Sophie Bauchimger, IAU
@Date: Tue May  9 15:39:11 2023

substance_list: available substances for different CARIBIC datasets
get_fct_substance: Fitting function for specific substance trends
get_col_name: Long column name from abbreviated substance
get_coord_name: Long column name from abbreviated coordinate 
coord_dict: all available coordinates
get_vlims: default limits for colormap representation per substance
get_default_units: default unit per substance
validated_input: let user try again if input was not in choices
choose_column: let user choose from available columns per specification
"""
from toolpac.outliers import ol_fit_functions as fct
import numpy as np

def substance_list(ID):
    """ Get all possible substances according to identifier (ID) """
    if ID == 'GHG':    return ['ch4', 'co2', 'n2o', 'sf6']
    if ID == 'INT':    return ['co', 'o3', 'h2o', 'no', 'noy', 'co2', 'ch4']
    if ID == 'INT2':   return ['co', 'o3', 'h2o', 'no', 'noy', 'co2', 'ch4',
                               'n2o', 'f11', 'f12']
    if ID == 'c_total':return ['o3', 'co2', 'n2o', 'co', 'sf6', 'ch4', 'no',
                               'noy', 'h2o', 'f11', 'f12']
    if ID == 'MLO':    return ['ch4', 'co2', 'n2o', 'sf6', 'co']

def get_fct_substance(substance):
    """ Returns corresponding fct from toolpac.outliers.ol_fit_functions """
    fct_dict = {'co2': fct.higher,
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
                'co': fct.quadratic, # prev. int_co
                }
    try: return fct_dict[substance.lower()]
    except:
        print(f'No default fctn for {substance}. Using simple harmonic')
        return fct.simple

def get_col_name(substance, source, c_pfx='', CLaMS=False):
    """
    Returns column name for substance as saved in dataframe
        source (str) 'Caribic', 'Mauna_Loa', 'Mace_Head', 'Mozart'
        substance (str): e.g. sf6, n2o, co2, ch4
    """
    cname=None

    if source=='Mauna_Loa': # mauna loa
        col_names = {
            'sf6': 'SF6catsMLOm',
            'n2o': 'N2OcatsMLOm',
            'co2': 'co2 [ppm]',
            'ch4': 'ch4 [ppb]',
            'co' : 'co [ppb]',
            }

    elif source=='Mace_Head': # mace head
        col_names={'sf6': 'SF6 [ppt]',
                   'ch2cl2': 'CH2Cl2 [ppt]',
                   }

    elif source=='Mozart': # mozart
        col_names = {'sf6': 'SF6'}

    elif source=='Caribic':
        if 'GHG' in c_pfx: # caribic / ghg
            col_names = {
                'ch4': 'CH4 [ppb]',
                'co2': 'CO2 [ppm]',
                'n2o': 'N2O [ppb]',
                'sf6': 'SF6 [ppt]',
                }

        elif 'INT' in c_pfx and c_pfx!='INT2': # caribic / int
            col_names = {
                'co' : 'int_CO [ppb]',
                'o3' : 'int_O3 [ppb]',
                'h2o': 'int_H2O_gas [ppm]',
                'no' : 'int_NO [ppb]',
                'noy': 'int_NOy [ppb]',
                'co2': 'int_CO2 [ppm]',
                'ch4': 'int_CH4 [ppb]',
                }

        elif 'INT2' in c_pfx: # caribic / int2
            col_names = {
                'co' : 'int_CARIBIC2_CO [ppbv]',
                'o3' : 'int_CARIBIC2_Ozone [ppbV]',
                'h2o': 'int_CARIBIC2_H2Ogas [ppmv]',
                'no' : 'int_CARIBIC2_NO [ppbv]',
                'noy': 'int_CARIBIC2_NOy [ppbv]',
                'co2': 'int_CARIBIC2_CO2 [ppmV]',
                'ch4': 'int_CARIBIC2_CH4 [ppbV]',
                'n2o': 'int_CLaMS_N2O [ppb]',
                'f11' : 'int_CLaMS_F11 [ppt]',
                'f12' : 'int_CLaMS_F12 [ppt]',
                }

            if CLaMS:
                col_names.update({
                'ch4' : 'int_CLaMS_CH4 [ppb]',
                'co'  : 'int_CLaMS_CO [ppb]',
                'co2' : 'int_CLaMS_CO2 [ppm]',
                'h2o' : 'int_CLaMS_H2O [ppm]',
                'o3'  : 'int_CLaMS_O3 [ppb]',
                })

        # after having gotten the 'standard' col names,
        # create the detrended / lag col names
        if c_pfx.startswith('detr'):
            col_names = {k:'detr_'+v for (k, v) in col_names.items()}
        elif c_pfx.startswith('lag_'):
            col_names = {subs:f'lag_{subs} [yr]' for subs
                         in col_names.keys()}

    try: cname = col_names[substance.lower()]
    except:
        print(f'No {substance} data in {source} ({c_pfx})')
        return None
    return cname

def get_v_coord(c_pfx, coord, tp_def, pvu=3.5):
    """ Coordinates relative to tropopause 
    coord (str): 'pt', 'dp', 'z'
    tp_def (str): 'chem', 'dyn', 'therm'
    pvu (float): 1.5, 2.0, 3.5
    """
    if c_pfx=='INT':
        if tp_def == 'chem': 
            col_names = {
                'z'       : 'int_h_rel_TP [km]'}                               # height above O3 tropopause according to Zahn et al. (2003), Atmospheric Environment, 37, 439-440
        if tp_def == 'therm':
            col_names = {
                'dp'      : 'int_dp_strop_hpa [hPa]',                          # pressure difference relative to thermal tropopause from ECMWF
                'pt'      : 'int_pt_rel_sTP_K [K]',                            # potential temperature difference relative to thermal tropopause from ECMWF
                'z'       : 'int_z_rel_sTP_km [km]'}                           # geopotential height relative to thermal tropopause from ECMWF
                
        elif tp_def == 'dyn':
            col_names = {
                'dp'        : 'int_dp_dtrop_hpa [hPa]',                        # pressure difference relative to dynamical (PV=3.5PVU) tropopause from ECMWF
                'pt'        : 'int_pt_rel_dTP_K [K]',                        # potential temperature difference relative to  dynamical (PV=3.5PVU) tropopause from ECMWF
                'z'         : 'int_z_rel_dTP_km [km]'}                         # geopotential height relative to dynamical (PV=3.5PVU) tropopause from ECMWF
        return col_names[coord]

    elif c_pfx=='INT2' and tp_def == 'dyn' and coord=='pt':
        col_names = {
            'pt_dyn_1_5'    : 'int_ERA5_D_1_5PVU_BOT [K]',                     # THETA-Distance to local 1.5 PVU surface (ERA5)
            'pt_dyn_2_0'    : 'int_ERA5_D_2_0PVU_BOT [K]',                     # -"- 2.0 PVU
            'pt_dyn_3_5'    : 'int_ERA5_D_3_5PVU_BOT [K]'}                     # -"- 3.5 PVU
        try: return col_names['pt_dyn_{}_{}'.format(str(pvu)[0], str(pvu)[2])]
        except: 
            print(f'No vertical coordinate found for {c_pfx} {coord} {tp_def} ({pvu})')
            return None

def get_h_coord(c_pfx, coord):
    """ coord: eql, """
    if c_pfx == 'INT':
        col_names = {
            'eql' : 'int_eqlat [deg]',                               # equivalent latitude in degrees north from ECMWF
            }
    elif c_pfx == 'INT2':
        col_names = {
            'eql' : 'int_ERA5_EQLAT [deg N]',                        # Equivalent latitude (ERA5)
            }
    try: return col_names[coord]
    except: 
        print(f'No horizontal coordinate found for {c_pfx} {coord}')
        return None

def get_val_coord(c_pfx, val):
    """ val(str): t, p, pv, pt """
    if c_pfx == 'GHG':
        col_names = {
            'p' : 'p [mbar]'}
    elif c_pfx == 'INT':
        col_names = {
            'p'  : 'p [mbar]',                                      # pressure (mean value)
            't'  : 'int_ToAirTmp [degC]',                           # Total Air Temperature
            'pv' : 'int_PV [PVU]',                                  # PV from ECMWF (integral)
            'pt' : 'int_Tpot [K]',                                  # potential temperature derived from measured pressure and temperature
            'z'  : 'int_z_km [km]',                                 # geopotential height of sample from ECMWF
            }
    elif c_pfx == 'INT2':
        col_names = {
            'p' : 'p [mbar]',                                      # pressure (mean value)
            't' : 'int_ERA5_TEMP [K]',                             # Temperature (ERA5)
            'pv': 'int_ERA5_PV [PVU]',                             # Potential vorticity (ERA5)
            'pt': 'int_Theta [K]',                                 # Potential temperature
            }
    try: return col_names[val]
    except: 
        print(f'No coordinate found for {c_pfx} {val}')
        return None
# =============================================================================
def get_coord_name(coord, source, c_pfx=None, CLaMS=True):
    """ Get name of eq. lat, rel height wrt therm/dyn tp, ..."""

    if source=='Caribic' and c_pfx=='GHG':
        col_names = {
            'p' : 'p [mbar]'}

    if source=='Caribic' and c_pfx=='INT':
        col_names = {
            'p'             : 'p [mbar]',                                      # pressure (mean value)
            'z_chem'        : 'int_h_rel_TP [km]',  	                       # height above O3 tropopause according to Zahn et al. (2003), Atmospheric Environment, 37, 439-440
            'pv'            : 'int_PV [PVU]',                                  # PV from ECMWF (integral)
            'to_air_tmp'    : 'int_ToAirTmp [degC]',                           # Total Air Temperature
            'tpot'          : 'int_Tpot [K]',                                  # potential temperature derived from measured pressure and temperature
            'z'             : 'int_z_km [km]',                                 # geopotential height of sample from ECMWF
            'dp_therm'      : 'int_dp_strop_hpa [hPa]',                        # pressure difference relative to thermal tropopause from ECMWF
            'dp_dym'        : 'int_dp_dtrop_hpa [hPa]',                        # pressure difference relative to dynamical (PV=3.5PVU) tropopause from ECMWF
            'pt_therm'      : 'int_pt_rel_sTP_K [K]',                          # potential temperature difference relative to thermal tropopause from ECMWF
            'pt_dyn'        : 'int_pt_rel_dTP_K [K]',                          # potential temperature difference relative to  dynamical (PV=3.5PVU) tropopause from ECMWF
            'z_therm'       : 'int_z_rel_sTP_km [km]',                         # geopotential height relative to thermal tropopause from ECMWF
            'z_dyn'         : 'int_z_rel_dTP_km [km]',                         # geopotential height relative to dynamical (PV=3.5PVU) tropopause from ECMWF
            'eq_lat'        : 'int_eqlat [deg]',                               # equivalent latitude in degrees north from ECMWF
            }

    elif source=='Caribic' and c_pfx=='INT2':
        col_names = {
            'p'             : 'p [mbar]',                                      # pressure (mean value)
            'z_chem'        : 'int_CARIBIC2_H_rel_TP [km]',                    # H_rel_TP; replacement for H_rel_TP => O3 tropopause 
            'pv'            : 'int_ERA5_PV [PVU]',                             # Potential vorticity (ERA5)
            'pt'            : 'int_Theta [K]',                                 # Potential temperature
            'p_era5'        : 'int_ERA5_PRESS [hPa]',                          # Pressure (ERA5)
            't'             : 'int_ERA5_TEMP [K]',                             # Temperature (ERA5)
            'eq_lat'        : 'int_ERA5_EQLAT [deg N]',                        # Equivalent latitude (ERA5)
            'p_tp'          : 'int_ERA5_TROP1_PRESS [hPa]',                    # Pressure of local lapse rate tropopause (ERA5)
            'pt_tp'         : 'int_ERA5_TROP1_THETA [K]',                      # Pot. temperature of local lapse rate tropopause (ERA5)
            'mean_age'      : 'int_AgeSpec_AGE [year]',                        # Mean age from age-spectrum (10 yr)
            'modal_age'     : 'int_AgeSpec_MODE [year]',                       # Modal age from age-spectrum (10 yr)
            'median_age'    : 'int_AgeSpec_MEDIAN_AGE [year]',                 # Median age from age-spectrum
            'pt_dyn_1_5'    : 'int_ERA5_D_1_5PVU_BOT [K]',                     # THETA-Distance to local 1.5 PVU surface (ERA5)
            'pt_dyn_2_0'    : 'int_ERA5_D_2_0PVU_BOT [K]',                     # -"- 2.0 PVU
            'pt_dyn_3_5'    : 'int_ERA5_D_3_5PVU_BOT [K]',                     # -"- 3.5 PVU
            }

    elif source=='Mozart': # mozart
        col_names = {
            'sf6': 'SF6'}

    try: cname = col_names[coord.lower()]
    except:
        print(f'Coordinate error: No {coord} in {source} ({c_pfx})')
        return None
    return cname

def coord_dict():
    """ Collection of coordinate column names in Caribic data.
    Currently available for GHG, INT, INT2 """
    ghg_coords = ['p [mbar]']
    int_coords = ['p [mbar]', 'int_h_rel_TP [km]', 'int_PV [PVU]',
                  'int_ToAirTmp [degC]', 'int_Tpot [K]', 'int_z_km [km]',
                  'int_dp_strop_hpa [hPa]', 'int_dp_dtrop_hpa [hPa]',
                  'int_pt_rel_sTP_K [K]', 'int_pt_rel_dTP_K [K]',
                  'int_z_rel_sTP_km [km]', 'int_z_rel_dTP_km [km]',
                  'int_eqlat [deg]']
    int2_coords = ['p [mbar]', 'int_CARIBIC2_H_rel_TP [km]',
                   'int_ERA5_PV [PVU]', 'int_Theta [K]',
                   'int_ERA5_PRESS [hPa]', 'int_ERA5_TEMP [K]',
                   'int_ERA5_EQLAT [deg N]', 'int_ERA5_TROP1_PRESS [hPa]',
                   'int_ERA5_TROP1_THETA [K]', 'int_AgeSpec_AGE [year]',
                   'int_AgeSpec_MODE [year]', 'int_AgeSpec_MEDIAN_AGE [year]',
                   'int_ERA5_D_1_5PVU_BOT [K]', 'int_ERA5_D_2_0PVU_BOT [K]',
                   'int_ERA5_D_3_5PVU_BOT [K]']
    coord_dict = {'GHG': ghg_coords,
                  'INT': int_coords,
                  'INT2': int2_coords}
    return coord_dict

def dict_season():
    return {'name_1': 'Spring (MAM)', 'name_2': 'Summer (JJA)',
            'name_3': 'Autumn (SON)', 'name_4': 'Winter (DJF)',
            'color_1': 'blue', 'color_2': 'orange',
            'color_3': 'green', 'color_4': 'red'}

def trop_filter_dict(tp_def, pvu=None, c_pfx=None):
    """ Return available criteria per tropopause definition (tp_def) """
    if tp_def == 'chem':
        crits = {'GHG' : ['n2o'],
                 'INT' : ['o3'],
                 'INT2' : ['n2o', 'o3']}
    elif tp_def == 'therm':
        crits = {'INT' : ['dp', 'pt'],
                 'INT2' : ['dp', 'pt', 'z']}

    elif tp_def == 'dyn' and pvu==3.5:
        crits = {'INT' : ['dp', 'pt', 'z'],
                 'INT2' : ['pt']}
    elif tp_def == 'dyn' and pvu in [1.5, 2.0]:
        crits = {'INT2' : ['pt']}

    if c_pfx: return crits[c_pfx]
    else: return crits
        

def get_vlims(substance):
    """ Get default limits for colormaps per substance """
    v_limits = {
        'sf6': (6,9),
        'n2o': (310,340),
        'co2': (320,380),
        'ch4': (1600,1950),
        'co' : (50, 160)}
    try: v_lims = v_limits[substance.lower()]
    except: v_lims = (np.nan, np.nan); print('no default v_lims found')
    return v_lims

def get_default_unit(substance):
    unit = {
        'sf6': 'ppt',
        'n2o': 'ppb',
        'co2': 'ppm',
        'ch4': 'ppb',
        'co' : 'ppb'}
    return unit[substance.lower()]

# def default_parameters(substance):
#     defaults = {
#         'n2o' : {
#             }}

#%% Input choice and validation
def validated_input(prompt, choices):
    valid_values = choices.keys()
    valid_input = False
    while not valid_input:
        value = input(prompt)
        if int(value) == 99: return None
        if int(value) in valid_values:
            yn = input(f'Confirm your choice ({choices[int(value)]}): Y/N \n')
            if yn.upper() =='Y': valid_input = int(value) in valid_values
            else: value = None; pass

        try: valid_input = int(value) in valid_values
        except: print('')
    return value

def choose_column(df, var='subs'):
    """ Let user choose one of the available column names """
    choices = dict(zip(range(0, len(df.columns)), df.columns))
    for k, v in choices.items(): print(k, ':', v)
    print('99 : pass')
    x = validated_input(f'Select a {var} column by choosing a number between \
                        0 and {len(df.columns)}: \n', choices)
    if not x: return None
    return choices[int(x)]
