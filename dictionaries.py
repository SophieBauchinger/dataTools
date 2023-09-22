# -*- coding: utf-8 -*-
""" Collection of dictionaries and look-up tables for air sample analysis.

@Author: Sophie Bauchinger, IAU
@Date: Tue May  9 15:39:11 2023

substance_list: Get all available substances for different datasets
get_fct_substance: Fitting function for specific substance trends
get_col_name: Long column name from abbreviated substance
get_tp_params: Get all available parameters for TP filtering dep. on conditions
get_v_coord: Long column name acc. to vertical coordinate parameters
get_h_coord: Long column name acc. to horizontal coordinate parameters
get_coord_name: Long column name from abbreviated coordinate name

dict_season: dictionary linking name and color to Nrs. 1-4
get_vlims: default limits for colormap representation per substance
get_default_units: default unit per substance

validated_input: let user try again if input was not in choices
choose_column: let user choose from available columns per specification

"""
from toolpac.outliers import ol_fit_functions as fct
import numpy as np
import pandas as pd

#%% Coordinates
class Coordinate():
    def __init__(self, **kwargs):
        """ Correlate column names with corresponding descriptors

        col_name (str)
        long_name (str)
        unit (str)
        ID (str): 'INT', 'INT2', 'EMAC'

        vcoord (str): p, z, pt, pv, eqpt
        hcoord (str): lat, lon, eql
        tp_def (str): chem, dyn, therm, cpt
        rel_to_tp (bool): coordinate relative to tropopause 
        model (str): msmt, ECMWF, ERA5, EMAC
        pvu (float): 1.5, 2.0, 3.5
        """
        self.__dict__.update(kwargs)

    def __repr__(self):
        return f'Coordinate: {self.col_name} [{self.unit}] from {self.ID}'

def coordinate_df():
    """ Get dataframe containing all info about all coordinate variables """
    with open('coordinates.csv', 'rb') as f: 
        coord_df = pd.read_csv(f)
    return coord_df

def get_coordinates(**kwargs):
    """Return dictionary of col_name:Coordinate for all items where conditions are met 
    Exclusion conditions need to have 'not_' prefix """
    df = coordinate_df()
    for cond, val in kwargs.items():
        if cond not in df.columns: continue
        # keep only rows where all conditions are fulfilled
        if not str(val).startswith('not_') and not str(val) == 'nan': 
            df = df[df[cond] == kwargs[cond]]
        elif str(val) == 'nan':
            df = df[df[cond].isna()]
        # also take out coords that are specifically excluded
        elif val == 'not_nan':
            df = df[~df[cond].isna()]
        elif val.startswith('not_'):
            df = df[df[cond] != val[4:]]

    if len(df) == 0: 
        raise KeyError('No data found using the given specifications')
    # df.set_index('col_name', inplace=True)
    coord_dict = df.to_dict(orient='index')
    coord = [Coordinate(**v) for k,v in coord_dict.items()]
    return coord

def coord_dict(*IDs):
    """ Return coordinate column names corresponding to list of IDs """
    return [y.col_name for id in IDs for y in get_coordinates(**{'ID':id})]

def get_coord(**kwargs):
    # if not any(v in kwargs for v in ['vcoord', 'hcoord', 'var']):
    #     raise KeyError('Please supply at least one of vcoord, hcoord, var.')
    coordinates = get_coordinates(**kwargs) # dict i:Coordinate
    if len(coordinates) > 1: 
        raise Warning(f'Multiple columns fulfill the conditions: {[i.col_name for i in coordinates]}')
        return [i.col_name for i in coordinates]
    return coordinates[0]

def make_coord_label(coordinates):
    """ Returns string to be used as axis label for a specific Coordinate object. """
    if not isinstance(coordinates, (list, set)): coordinates = [coordinates]
    labels=[]
    for coord in coordinates:
        if coord.vcoord is not np.nan:
            pv = '%s' % (f', {coord.pvu}' if coord.tp_def=='dyn' else '')
            model = coord.model
            # model = '%s' % (' - ECMWF' if (coordinate.ID=='INT' and coordinate.tp_def in ['dyn', 'therm']) else '')
            # model += '%s' % (' - ERA5' if (coordinate.ID=='INT2' and coordinate.tp_def in ['dyn', 'therm']) else '')
            tp = '%s' % (f', {coord.tp_def}' if coord.tp_def is not np.nan else '')
            vcoord = f'$\Delta\,${coord.vcoord}' if coord.rel_to_tp else f'{coord.vcoord}'
            if coord.vcoord == 'pt': vcoord = '$\Delta\,\Theta$' if coord.rel_to_tp else '$\Theta$'
            
            label = f'{vcoord} ({model+tp+pv}) [{coord.unit}]'
            
        elif coord.hcoord is not np.nan:
            if coord.hcoord == 'lat' and coord.unit=='degrees_north': 
                label = 'Latitude [°N]'
            elif coord.hcoord == 'lon' and coord.unit == 'degrees_east':
                label = 'Longitude [°E]'
            elif coord.hcoord == 'eql' and coord.unit=='degrees_north': 
                label = f'Equivalent Latitude [°N] ({coord.model})'
            elif coord.hcoord == 'geometry':
                label = 'Geometry [°N, °E]'
        
        elif coord.var is not np.nan and coord.vcoord is np.nan:
            print(coord)
            label = f'{coord.var} {coord.unit}'
        labels.append(label)

    if len(coordinates)==1: return labels[0]
    else: return labels

#%% Substances
class Substance():
    def __init__(self, col_name, **kwargs):
        """ Correlate substance column names with corresponding desriptors
        col_name (str)
        long_name (str)
        short_name (str)
        unit (str)
        ID (str): INT, INT2, EMAC, MLO, MZT, MHD
        model (str): msmt, CLAMS, EMAC, MOZART
        function (str): h, s, q
        """
        self.col_name = col_name
        self.__dict__.update(kwargs)

    def __repr__(self):
        return f'Substance : {self.short_name} [{self.unit}] - \'{self.col_name}\' from {self.ID}'

def substance_df():
    """ Get dataframe containing all info about all substance variables """
    with open('substances.csv', 'rb') as f: 
        substances = pd.read_csv(f)
    # update fct column with proper functions 
    fctn_dict = {'h' : fct.higher, 's' : fct.simple, 'q' : fct.quadratic}
    substances['function'] = [fctn_dict.get(f) for f in substances['function']]
    return substances

def get_substances(**kwargs):
    """ Return dictionary of col_name:Substance for all items were conditions are met """
    df = substance_df()
    # keep only rows where all conditions are fulfilled
    for cond in kwargs: 
        df = df[df[cond] == kwargs[cond]]
    if len(df) == 0: 
        raise KeyError('No substance column found using the given specifications')
    df.set_index('col_name', inplace=True)
    subs_dict = df.to_dict(orient='index')
    subs = [Substance(k, **v) for k,v in subs_dict.items()]
    return subs

def substance_list(ID):
    """ Returns list of available substances for a specific datset as short name """
    df = substance_df()
    if ID not in df.ID.values: 
        raise KeyError(f'Unable to provide subs list for {ID}')
    else: 
        df = df[df['ID'] == ID]
        df = df[[not name.startswith('d_') for name in df['short_name']]]
    return set(df['short_name'])

def get_subs(substance, ID, clams=False, **kwargs):
    """ Return single Substance object with the given specifications """
    conditions = {'short_name' : substance, 'ID' : ID}
    if ID=='INT2' and substance not in ['f11', 'f12', 'n2o', 'no', 'noy']: 
        conditions.update({'model' : 'CLAMS' if clams else 'msmt'})
    substances = get_substances(**conditions)
   
    if len(substances) > 1: 
        raise Warning(f'Multiple columns fulfill the conditions: {substances}')
        return list(substances)
    return substances[0]

def get_col_name(substance, ID, clams=False):
    #TODO change source, c_pfx to ID in all other scripts
    return get_subs(substance, ID, clams).col_name

def make_subs_label(substances):
    """ Returns string to be used as axis label for a specific Coordinate object. """
    if not isinstance(substances, (list, set)): substances = [substances]
    labels=[]
    for subs in substances:
        special_names = {'ch2cl2':'CH2Cl2', 'noy':'NOy'}
        name = '%s' % f'{subs.short_name.upper()}' if not subs.short_name in special_names else special_names[subs.short_name]
        if name.startswith('D_'): name = 'd_'+name[2:]
        source = '%s' % subs.source if subs.model=='msmt' else subs.model
        label = f'{name} [{subs.unit}] ({source})'
        labels.append(label)
    if len(substances)==1: return labels[0]
    else: return labels

def get_fct_substance(substance, verbose=False):
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
        if verbose: print(f'No default fctn for {substance}. Using simple harmonic')
        return fct.simple

def get_tp_params(tp_def=None, ID=None, crit=None, vcoord=None, pvu=None):
    """ Return a list of all TP params possible given the constraints """
    #TODO implement EMAC tropopauses in here!! 
    c_keys = ['tp_def', 'ID', 'crit']
    c1 = {k:v for k,v in zip(c_keys, ['chem', 'GHG', 'n2o'])}
    c2 = {k:v for k,v in zip(c_keys, ['chem', 'INT', 'o3'])}
    c3 = {k:v for k,v in zip(c_keys, ['chem', 'INT2', 'n2o'])}
    c4 = {k:v for k,v in zip(c_keys, ['chem', 'INT2', 'o3'])}

    t_keys = ['tp_def', 'ID', 'vcoord']
    t1 = {k:v for k,v in zip(t_keys, ['therm', 'INT', 'dp'])}
    t2 = {k:v for k,v in zip(t_keys, ['therm', 'INT', 'pt'])}
    t3 = {k:v for k,v in zip(t_keys, ['therm', 'INT', 'z'])}
    t4 = {k:v for k,v in zip(t_keys, ['therm', 'INT2', 'dp'])}
    t5 = {k:v for k,v in zip(t_keys, ['therm', 'INT2', 'pt'])}

    d_keys = ['tp_def', 'ID', 'vcoord', 'pvu']
    d1 = {k:v for k,v in zip(d_keys, ['dyn', 'INT2', 'pt', 1.5])}
    d2 = {k:v for k,v in zip(d_keys, ['dyn', 'INT2', 'pt', 2.0])}
    d3 = {k:v for k,v in zip(d_keys, ['dyn', 'INT2', 'pt', 3.5])}
    d4 = {k:v for k,v in zip(d_keys, ['dyn', 'INT', 'dp', 3.5])}
    d5 = {k:v for k,v in zip(d_keys, ['dyn', 'INT', 'pt', 3.5])}
    d6 = {k:v for k,v in zip(d_keys, ['dyn', 'INT', 'z',  3.5])}

    # de_keys = ['tp_def', 'coord']
    # de1 = {k:v for k,v in zip(de_keys, ['dyn', 'p'])}
    # de2 = {k:v for k,v in zip(de_keys, ['therm', 'p'])}
    # de3 = {k:v for k,v in zip(de_keys, ['cpt', 'p'])}

    param_dicts = [
        c1, c2, c3, c4,
        t1, t2, t3, t4, t5,
        d1, d2, d3, d4, d5, d6]

    for var in [tp_def, ID, crit, vcoord, pvu]: # e.g. 'therm'
        if var is not None:
            param_dicts = [d for d in param_dicts if var in d.values()]

    if len(param_dicts)==0:
        given_params = ''.join([f"{name} ({val}), " for name, val in zip(['tp_def', 'ID', 'crit', 'coord', 'pvu'],
                            [tp_def, ID, crit, vcoord, pvu]) if val is not None])
        raise KeyError(f'No TP params with the following constraints: {given_params}')

    return param_dicts

def dict_season():
    """ Use to get name_s, color_s for season s"""
    return {'name_1': 'Spring (MAM)', 'name_2': 'Summer (JJA)',
            'name_3': 'Autumn (SON)', 'name_4': 'Winter (DJF)',
            'color_1': 'blue', 'color_2': 'orange',
            'color_3': 'green', 'color_4': 'red'}

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



#%% Input choice and validation
def validated_input(prompt, choices):
    valid_values = choices
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
    x = validated_input(f'Select a {var} column by choosing a number between \0 and {len(df.columns)}: \n', choices)
    if not x: return None
    return choices[int(x)]

#%% ARCHIVE (for now )
# def get_default_unit(substance):
#     unit = {
#         'sf6': 'ppt',
#         'n2o': 'ppb',
#         'co2': 'ppm',
#         'ch4': 'ppb',
#         'co' : 'ppb'}
#     return unit[substance.lower()]

# def get_v_coord(ID, coord, tp_def, pvu=3.5):
#     """ Coordinates relative to tropopause
    
#     ID (str): c_pfx or .source (e.g. EMAC)
#     tp_def (str): 'chem', 'dyn', 'therm'
#     coord (str): 'pt', 'dp', 'z'
#     pvu (float): 1.5, 2.0, 3.5
#     """
#     # chemical tropopause (O3)
#     if tp_def == 'chem':
#         if ID == 'INT' and coord =='z':
#             col_names = {
#                 'z'       : 'int_h_rel_TP [km]'}                               # height above O3 tropopause according to Zahn et al. (2003), Atmospheric Environment, 37, 439-440
#         elif ID == 'INT2' and coord == 'z':
#             col_names = {
#                 'z'       : 'int_h_rel_TP [km]'}                               # height above O3 tropopause according to Zahn et al. (2003), Atmospheric Environment, 37, 439-440
#         else: raise KeyError(f'No {ID} {coord}-coord available for chemical TP')

#     # thermal tropopause
#     elif tp_def == 'therm':
#         if ID == 'INT' and coord in ['dp', 'pt']:
#             col_names = {
#                 'dp'      : 'int_dp_strop_hpa [hPa]',                          # pressure difference relative to thermal tropopause from ECMWF
#                 'pt'      : 'int_pt_rel_sTP_K [K]',                            # potential temperature difference relative to thermal tropopause from ECMWF
#                 'z'       : 'int_z_rel_sTP_km [km]'}                           # geopotential height relative to thermal tropopause from ECMWF
#         elif ID == 'INT2' and coord in ['dp', 'pt', 'z']:
#             col_names = {
#                 'dp' : 'int_dp_strop_hpa_ERA5 [hPa]',                          # pressure difference relative to thermal tropopause from ERA5
#                 'pt' : 'int_pt_rel_sTP_K_ERA5 [K]'}                            # potential temperature difference relative to thermal tropopause from ERA5
#         elif ID=='EMAC' and coord in ['dp', 'pt', 'p']:
#             col_names = {
#                 }
#         else: raise KeyError(f'No {ID} {coord}-coord available for thermal TP')

#     # dynamical tropopause
#     elif tp_def == 'dyn':
#         if ID == 'INT':
#             col_names = {
#                 'dp'        : 'int_dp_dtrop_hpa [hPa]',                         # pressure difference relative to dynamical (PV=3.5PVU) tropopause from ECMWF
#                 'pt'        : 'int_pt_rel_dTP_K [K]',                           # potential temperature difference relative to  dynamical (PV=3.5PVU) tropopause from ECMWF
#                 'z'         : 'int_z_rel_dTP_km [km]'}                          # geopotential height relative to dynamical (PV=3.5PVU) tropopause from ECMWF
#         elif ID == 'INT2' and coord == 'pt' and pvu in [1.5, 2.0, 3.5]:
#             if pvu==1.5: col_names = {'pt' : 'int_ERA5_D_1_5PVU_BOT [K]'}       # THETA-Distance to local 1.5 PVU surface (ERA5)
#             elif pvu==2.0: col_names = {'pt' : 'int_ERA5_D_2_0PVU_BOT [K]'}     # -"- 2.0 PVU
#             elif pvu==3.5: col_names = {'pt' : 'int_ERA5_D_3_5PVU_BOT [K]'}     # -"- 3.5 PVU
#         else: raise KeyError(f'No {ID} {coord}-coord with pvu{pvu} available for dynamical TP')

#     else: raise KeyError(f'Cannot provide a v-coordinate for {tp_def} TP')
#     return col_names[coord]


# def get_h_coord(c_pfx, coord):
#     """ coord: eql, """
#     if c_pfx == 'INT' and coord=='eql':
#         col_names = {
#             'eql' : 'int_eqlat [deg]'}                                          # equivalent latitude in degrees north from ECMWF
#     elif c_pfx == 'INT2' and coord=='eql':
#         col_names = {
#             'eql' : 'int_ERA5_EQLAT [deg N]'}                                   # Equivalent latitude (ERA5)
#     else: raise KeyError(f'No {coord}-coord available for {c_pfx}')

#     return col_names[coord]

# def get_coord_name(coord, source, c_pfx=None, CLaMS=True):
#     """ Get name of eq. lat, rel height wrt therm/dyn tp, ..."""

#     if source=='Caribic' and c_pfx=='GHG':
#         col_names = {
#             'p' : 'p [mbar]'}

#     if source=='Caribic' and c_pfx=='INT':
#         col_names = {
#             'p'             : 'p [mbar]',                                       # pressure (mean value)
#             'z_chem'        : 'int_h_rel_TP [km]',  	                        # height above O3 tropopause according to Zahn et al. (2003), Atmospheric Environment, 37, 439-440
#             'pv'            : 'int_PV [PVU]',                                   # PV from ECMWF (integral)
#             'to_air_tmp'    : 'int_ToAirTmp [degC]',                            # Total Air Temperature
#             'tpot'          : 'int_Tpot [K]',                                   # potential temperature derived from measured pressure and temperature
#             'z'             : 'int_z_km [km]',                                  # geopotential height of sample from ECMWF
#             'dp_therm'      : 'int_dp_strop_hpa [hPa]',                         # pressure difference relative to thermal tropopause from ECMWF
#             'dp_dym'        : 'int_dp_dtrop_hpa [hPa]',                         # pressure difference relative to dynamical (PV=3.5PVU) tropopause from ECMWF
#             'pt_therm'      : 'int_pt_rel_sTP_K [K]',                           # potential temperature difference relative to thermal tropopause from ECMWF
#             'pt_dyn'        : 'int_pt_rel_dTP_K [K]',                           # potential temperature difference relative to  dynamical (PV=3.5PVU) tropopause from ECMWF
#             'z_therm'       : 'int_z_rel_sTP_km [km]',                          # geopotential height relative to thermal tropopause from ECMWF
#             'z_dyn'         : 'int_z_rel_dTP_km [km]',                          # geopotential height relative to dynamical (PV=3.5PVU) tropopause from ECMWF
#             'eq_lat'        : 'int_eqlat [deg]',                                # equivalent latitude in degrees north from ECMWF
#             }

#     elif source=='Caribic' and c_pfx=='INT2':
#         col_names = {
#             'p'             : 'p [mbar]',                                       # pressure (mean value)
#             'z_chem'        : 'int_CARIBIC2_H_rel_TP [km]',                     # H_rel_TP; replacement for H_rel_TP => O3 tropopause
#             'pv'            : 'int_ERA5_PV [PVU]',                              # Potential vorticity (ERA5)
#             'pt'            : 'int_Theta [K]',                                  # Potential temperature
#             'p_era5'        : 'int_ERA5_PRESS [hPa]',                           # Pressure (ERA5)
#             't'             : 'int_ERA5_TEMP [K]',                              # Temperature (ERA5)
#             'eq_lat'        : 'int_ERA5_EQLAT [deg N]',                         # Equivalent latitude (ERA5)
#             'p_tp'          : 'int_ERA5_TROP1_PRESS [hPa]',                     # Pressure of local lapse rate tropopause (ERA5)
#             'pt_tp'         : 'int_ERA5_TROP1_THETA [K]',                       # Pot. temperature of local lapse rate tropopause (ERA5)
#             'mean_age'      : 'int_AgeSpec_AGE [year]',                         # Mean age from age-spectrum (10 yr)
#             'modal_age'     : 'int_AgeSpec_MODE [year]',                        # Modal age from age-spectrum (10 yr)
#             'median_age'    : 'int_AgeSpec_MEDIAN_AGE [year]',                  # Median age from age-spectrum
#             'pt_dyn_1_5'    : 'int_ERA5_D_1_5PVU_BOT [K]',                      # THETA-Distance to local 1.5 PVU surface (ERA5)
#             'pt_dyn_2_0'    : 'int_ERA5_D_2_0PVU_BOT [K]',                      # -"- 2.0 PVU
#             'pt_dyn_3_5'    : 'int_ERA5_D_3_5PVU_BOT [K]',                      # -"- 3.5 PVU
#             }

#     elif source=='Mozart': # mozart
#         col_names = {'sf6': 'SF6'}

#     try: cname = col_names[coord.lower()]
#     except:
#         print(f'Coordinate error: No {coord} in {source} ({c_pfx})')
#         return None
#     return cname

# def coord_dict():
#     """ Collection of coordinate column names in Caribic data.
#     Currently available for GHG, INT, INT2 """
#     ghg_coords = ['p [mbar]']
#     int_coords = ['p [mbar]', 'int_h_rel_TP [km]', 'int_PV [PVU]',
#                   'int_ToAirTmp [degC]', 'int_Tpot [K]', 'int_z_km [km]',
#                   'int_dp_strop_hpa [hPa]', 'int_dp_dtrop_hpa [hPa]',
#                   'int_pt_rel_sTP_K [K]', 'int_pt_rel_dTP_K [K]',
#                   'int_z_rel_sTP_km [km]', 'int_z_rel_dTP_km [km]',
#                   'int_eqlat [deg]']
#     int2_coords = ['p [mbar]', 'int_CARIBIC2_H_rel_TP [km]',
#                    'int_ERA5_PV [PVU]', 'int_Theta [K]',
#                    'int_ERA5_PRESS [hPa]', 'int_ERA5_TEMP [K]',
#                    'int_ERA5_EQLAT [deg N]', 'int_ERA5_TROP1_PRESS [hPa]',
#                    'int_ERA5_TROP1_THETA [K]', 'int_AgeSpec_AGE [year]',
#                    'int_AgeSpec_MODE [year]', 'int_AgeSpec_MEDIAN_AGE [year]',
#                    'int_ERA5_D_1_5PVU_BOT [K]', 'int_ERA5_D_2_0PVU_BOT [K]',
#                    'int_ERA5_D_3_5PVU_BOT [K]']
#     coord_dict = {'GHG': ghg_coords,
#                   'INT': int_coords,
#                   'INT2': int2_coords}
#     return coord_dict