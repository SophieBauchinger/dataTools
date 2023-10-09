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

def make_coord_label(coordinates, filter_label=False):
    """ Returns string to be used as axis label for a specific Coordinate object. """
    if not isinstance(coordinates, (list, set)): coordinates = [coordinates]
    labels=[]
    for coord in coordinates:
        if coord.vcoord is not np.nan:
            pv = '%s' % (f', {coord.pvu}' if coord.tp_def=='dyn' else '')
            crit = '%s' % (f', {coord.crit}' if coord.tp_def=='chem' else '')
            model = coord.model # if not coord.tp_def=='chem' else ''
            # model = '%s' % (' - ECMWF' if (coordinate.ID=='INT' and coordinate.tp_def in ['dyn', 'therm']) else '')
            # model += '%s' % (' - ERA5' if (coordinate.ID=='INT2' and coordinate.tp_def in ['dyn', 'therm']) else '')
            tp = '%s' % (f', {coord.tp_def}' if coord.tp_def is not np.nan else '')
            vcoord = f'$\Delta\,${coord.vcoord}' if coord.rel_to_tp else f'{coord.vcoord}'
            if coord.vcoord == 'pt': vcoord = '$\Delta\,\Theta$' if coord.rel_to_tp else '$\Theta$'
            
            label = f'{vcoord} ({model+tp+pv+crit}) [{coord.unit}]'
            if filter_label: label = f'{model+tp+pv+crit} ({vcoord})'
            
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

# legacy
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

def get_subs(*args, **kwargs):
    """ Return single Substance object with the given specifications """
    for i,arg in enumerate(args): # substance, ID, clams 
        pot_args = ['substance', 'ID', 'clams']
        kwargs.update({pot_args[i]:arg})
    if not 'short_name' in kwargs and 'substance' in kwargs: 
        kwargs.update({'short_name':kwargs.pop('substance')})
    if not 'model' in kwargs and 'clams' in kwargs:
        kwargs.update({'model':'CLAMS' if kwargs.pop('clams') else 'msmt'})

    # conditions = {'short_name' : substance, 'ID' : ID}
    # if kwargs.get('ID')=='INT2' and kwargs.get('substance') not in ['f11', 'f12', 'n2o', 'no', 'noy']: 
    #     if not 'model' in kwargs: 
    #         kwargs.update({'model' : 'CLAMS' if kwargs.get('clams') else 'msmt'})
    substances = get_substances(**kwargs)
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

#%% Misc
def dict_season():
    """ Use to get name_s, color_s for season s"""
    return {'name_1': 'Spring (MAM)', 'name_2': 'Summer (JJA)',
            'name_3': 'Autumn (SON)', 'name_4': 'Winter (DJF)',
            'color_1': 'blue', 'color_2': 'orange',
            'color_3': 'green', 'color_4': 'red'}

def dict_tps():
    """ Get color etc for tropopause definitions """
    return {'color_chem' : '#1f77b4',
            'color_therm' : '#ff7f0e',
            'color_dyn' : '#2ca02c'}

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

def note_dict(fig_or_ax, x=None, y=None, s=None):
    """ Return default arguments & bbox dictionary for adding notes to plots. """
    try: transform = fig_or_ax.transAxes
    except: transform = fig_or_ax.transFigure

    bbox_defaults = dict(edgecolor='lightgrey', 
                         facecolor='lightcyan', 
                         boxstyle='round')

    note_dict = dict(x = x if x else 0.97,
                     y = y if y else 0.97,
                     horizontalalignment='right', 
                     verticalalignment='center_baseline', 
                     transform = transform, 
                     bbox = bbox_defaults)
    if s: note_dict.update(dict(s=s))

    return note_dict

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
