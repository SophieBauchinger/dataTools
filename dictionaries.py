# -*- coding: utf-8 -*-
""" Collection of dictionaries and look-up tables for air sample analysis.

@Author: Sophie Bauchinger, IAU
@Date: Tue May  9 15:39:11 2023

class Coordinate
    .label: filter_label, coord_only
    .get_bsize

class Substance
    .label: name_only
    .vlims: bin_attr, atm_layer

class Instrument

Functions:
    # ---- Coordinates ---- 
    coordinate_df
    get_coordinates(**kwargs)
    get_coord(*args, **kwargs)

     # ---- Substances ---- 
    vlim_dict_per_substance(short_name)
    substance_df
    get_substances(**kwargs)
    get_subs(*args, **kwargs)
    lookup_fit_function(short_name)
    
    # ---- Instruments ----
    instrument_df
    instr_vars_per_ID_df
    get_instruments(ID)
    get_variables(ID, instr)
    variables_per_instruments(instr)
    harmonise_instruments(old_name)
    harmonise_variables(instr, var_name)
    
    # --- Aircraft campaigns ---
    campaign_definitions(campaign)
    years_per_campaign(campaign)
    instruments_per_campaign(campaign)
    MS_variables(*args)
    
    # --- Plotting / Colors ---
    dict_season
    dict_colors
    dict_tps
    note_dict(fig_or_ax, x, y, s, ha)

"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap as lsc
import cmasher as cmr
import os

from toolpac.outliers import ol_fit_functions as fct # type: ignore

def get_path():
    """ Get parent directory of current module, i.e. location of dataTools. """
    return os.path.dirname(os.path.abspath(__file__)) + "\\"

# %% Coordinates
class Coordinate:
    """ Holds information on the properties of the current coordinate variable
    
    Attributes:
        col_name (str)
        long_name (str)
        unit (str)
        ID (str): 'INT', 'INT2', 'EMAC'

        vcoord (str): p, z, pt, pv, eqpt
        hcoord (str): lat, lon, eql
        var (str): e.g. geopot

        tp_def (str): chem, dyn, therm, cpt
        rel_to_tp (bool): coordinate relative to tropopause
        model (str): msmt, ECMWF, ERA5, EMAC
        pvu (float): 1.5, 2.0, 3.5
        crit (str): n2o / o3
    
    """
    def __init__(self, **kwargs):
        """ Correlate column names with corresponding descriptors

        col_name (str)
        long_name (str)
        unit (str)
        ID (str): 'INT', 'INT2', 'EMAC'

        vcoord (str): p, z, pt, pv, eqpt
        hcoord (str): lat, lon, eql
        var (str): e.g. geopot

        tp_def (str): chem, dyn, therm, cpt
        rel_to_tp (bool): coordinate relative to tropopause
        model (str): msmt, ECMWF, ERA5, EMAC
        pvu (float): 1.5, 2.0, 3.5
        crit (str): n2o / o3
        """
        self.col_name, self.long_name, self.unit, self.ID = [None] * 4
        self.vcoord, self.hcoord, self.var = [None] * 3
        self.tp_def, self.rel_to_tp, self.model, self.pvu, self.crit = [None] * 5

        self.__dict__.update(kwargs)
        if self.pvu is not np.nan:
            self.pvu = float(self.pvu)

    def __repr__(self):
        return f'Coordinate: {self.col_name} [{self.unit}] from {self.ID}'

    def label(self, filter_label=False, coord_only=False):
        """ Returns latex-formatted string to be used as axis label. """

        tp_defs = {'chem': 'Chemical',
                   'dyn': 'Dynamic',
                   'therm': 'Thermal',
                   'cpt': 'Cold point',
                   'combo': 'Multi-definition'}

        if self.vcoord is not np.nan:
            vcs = {'p': 'Pressure',
                   'z': 'Geopotential height',
                   'pt': '$\Theta$',
                   'eqpt': '$\Theta$(eq)',
                   'mxr': 'Mixing ratio',
                   'pv': 'Potential vorticity',
                   'lev': 'Level'}

            vcoord = vcs[self.vcoord] if self.vcoord in vcs else self.vcoord

            if self.tp_def is not np.nan:
                vcoord = f'$\Delta${vcoord}$_{{TP}}$' if self.rel_to_tp else vcoord

                pv = '%s' % (f', {self.pvu}' if self.tp_def == 'dyn' else '')
                crit = '%s' % (', ' + ''.join(
                    f"$_{i}$" if i.isdigit() else i.upper() for i in self.crit) if self.tp_def == 'chem' else '')
                model = self.model
                tp = '%s' % (self.tp_def if self.tp_def is not np.nan else '')

                label = f'{vcoord} ({model}, {tp + pv + crit}) [{self.unit}]'

                if filter_label:
                    tp = tp_defs[tp]
                    vc = self.vcoord if not self.vcoord == 'pt' else '$\Theta$'
                    if self.rel_to_tp: vc = '$\Delta\,$' + vc
                    label = f'{tp + pv + crit} ({model}, {vc})'
            else:
                label = f'{vcoord} [{self.unit}]'

        elif self.hcoord is not np.nan:
            hcs = {'lon': 'Longitude',
                   'lat': 'Latitude',
                   'eql': 'Equivalent Latitude',
                   'degrees_north': '°N',
                   'degrees_east': '°E',
                   'degrees': '°N, °E'}
            if self.hcoord in hcs and self.unit in hcs:
                label = f'{hcs[self.hcoord]} [{hcs[self.unit]}]'
            else:
                label = f'{self.hcoord} [{self.unit}]'

        elif self.var is not np.nan:
            label = f'{self.var} [{self.unit}]'

        else:
            raise NotImplementedError('Cannot create label, this should not have happened.')

        if coord_only:
            vcoord = f'$\Delta${self.vcoord}$_{{TP}}$' if self.rel_to_tp else f'{self.vcoord}'
            if self.vcoord == 'pt': vcoord = '$\Delta\Theta_{{TP}}$' if self.rel_to_tp else '$\Theta$'
            label = f'{vcoord} [{self.unit}]'

        return label

    def get_bsize(self) -> float:
        """ Returns default bin size of the coordinate. """
        bsizes_dict = {
            'pt': 10,
            'p': 25,
            'z': 0.75,
            'mxr': 5,  # n2o
            'eqpt': 5,
            'eql': 5,
            'lat': 10,
            'lon': 10,
        }

        if self.vcoord in bsizes_dict:
            return bsizes_dict[self.vcoord]
        elif self.hcoord in bsizes_dict:
            return bsizes_dict[self.hcoord]
        else:
            raise KeyError(f'No default bin size for v: {self.vcoord} / h: {self.hcoord} / var: {self.var}')

def coordinate_df():
    """ Get dataframe containing all info about all coordinate variables """
    with open(get_path() + 'coordinates.csv', 'rb') as f:
        coord_df = pd.read_csv(f, sep="\s*,\s*", engine='python')
        if 'pvu' in coord_df.columns:
            coord_df['pvu'] = coord_df['pvu'].astype(object)  # allow comparison with np.nan
    return coord_df

def get_coordinates(**kwargs):
    """Return dictionary of col_name:Coordinate for all items where conditions are met
    Exclusion conditions need to have 'not_' prefix """
    df = coordinate_df()

    for cond, val in kwargs.items():
        if cond not in df.columns:
            raise KeyError(f'{cond} not recognised as valid coordinate qualifier.')

        if cond == 'pvu' and not str(val) == 'nan':
            df = df[df[cond].astype(float) == kwargs[cond]]
        # keep only rows where all conditions are fulfilled
        elif not str(val).startswith('not_') and not str(val) == 'nan':
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
    coord = [Coordinate(**v) for k, v in coord_dict.items()]
    return coord

def get_coord(*args, **kwargs):
    for i, arg in enumerate(args):  # col_name, ID
        pot_args = ['col_name', 'ID']
        kwargs.update({pot_args[i]: arg})
    coordinates = get_coordinates(**kwargs)  # dict i:Coordinate
    if len(coordinates) > 1:
        raise ValueError(f'Multiple columns fulfill the conditions: {[i.col_name for i in coordinates]}')
    return coordinates[0]

# %% Substances
class Substance:
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
        self.long_name, self.short_name, self.unit = [None] * 3
        self.ID, self.source, self.model = [None] * 3
        self.function, self.detr = [None] * 2

        self.__dict__.update(kwargs)

    def __repr__(self):
        return f'Substance : {self.short_name} [{self.unit}] - \'{self.col_name}\' from {self.ID}'

    def label(self, name_only:bool=False, bin_attr:str=None):
        """ Returns string to be used as axis label. """
        detr_qualifier = 'rel. to BGD ' if self.detr else ''

        # Define exceptions for labels with upper & lowercase substance labels
        special_names = {'ch2cl2': 'CH$_2$Cl$_2$',
                        'noy': r'NO$_\mathrm{y}$',
                        'f11': 'F11',
                        'f12': 'F12',
                        'mol mol-1': 'mol/mol'}

        # Get substance label
        subs_abbr = self.short_name.split('_')[-1]
        if subs_abbr in special_names:
            subs_abbr = special_names[subs_abbr]
        else:
            subs_abbr = subs_abbr.upper()
            subs_abbr = ''.join(f"$_{i}$" if i.isdigit() else i for i in subs_abbr)
        if self.short_name.startswith('d_'):
            subs_abbr = r'$\mathrm{\sigma}\,$' + f'({subs_abbr})'

        # name_only returns only the abbreviated substance name
        if name_only is True:
            return subs_abbr
        
        # Get data source identifier
        if self.model == 'MSMT':
            identifier = self.source if not self.source=='HALO' else self.ID
        elif self.model == 'CLAMS':
            identifier = 'CLaMS'
        else:
            identifier = self.model
        if self.source!='HALO' and len(get_substances(
                short_name=self.short_name, source=self.source, model=self.model)) > 1:
            identifier += f' - {self.ID}'

        # Get the appropriate unit label
        unit = self.unit if not self.unit in special_names else special_names[self.unit]

        # Special labels for binned data
        bin_attr_qualifiers = {
            'vmean': '', 
            'vstdv' : 'Variability of ',
            'rvstd' : 'Relative variability of '}
        bin_attr_units = {
            'vmean' : unit,
            'vstdv' : unit,
            'rvstd' : '%'}
        if bin_attr is not None: 
            qualifier = bin_attr_qualifiers[bin_attr]
            bin_attr_unit = bin_attr_units[bin_attr]
            return f'{qualifier}{subs_abbr} {detr_qualifier}[{bin_attr_unit}]'


        return f'{subs_abbr} {detr_qualifier}[{unit}]' + (f' ({identifier})' if not self.detr else '')

    def vlims(self, bin_attr='vmean', atm_layer=None) -> tuple:
        """ Default colormap normalisation limits for substance mixing ratio or variabiliy. 
        
        Args: 
            bin_attr (str): vmean / vstdv / rvstd / ... + _tropo/_strato
            atm_layer (str): Atmospheric layer, if not in bin_attr will be added as bin_attr + _  + atm_layer
        """
        # special cases for vcount 
        if self.short_name.startswith('d_'): 
            vlims = (0,1)
        
        vlim_dict = vlim_dict_per_substance(self.short_name)

        if atm_layer is not None and atm_layer not in bin_attr: # set bin_attr to reflect atm_layer
            bin_attr = f'{bin_attr}_{atm_layer}' 

        if bin_attr not in vlim_dict: # invalid bin_attr
            raise KeyError(f'Unrecognised bin_attr {bin_attr}, please check your input. ')

        if str(vlim_dict[bin_attr]) == '(nan, nan)': # values not set for substance 
            raise KeyError(f'No default vlims set for {bin_attr} in {self.short_name} data')
        
         # values and bin_attr set and valid
        vlims = vlim_dict[bin_attr] # if not atm_layer else vlim_dict[f'{bin_attr}_{atm_layer}']

        return vlims 

def vlim_dict_per_substance(short_name) -> dict[tuple]: 
    """ Returns dictionary for colormap normalisation limits for the given substance (str). 
    
    Dictionary keys: vmean / vstdv / vstdv_tropo / vstdv_strato / rvstd / rvstd_tropo / rvstd_strato
    If values are not specifically set for a substance, their dict entry will be (nan, nan). 
    """
    d_ = False
    if short_name.startswith('d_'): 
        short_name = short_name[2:]
        d_ = True
    
    vlim_dict = {
        'vmean' : (np.nan, np.nan),
        'vstdv': (np.nan, np.nan),
        'vstdv_tropo' : (np.nan, np.nan),
        'vstdv_strato' : (np.nan, np.nan),
        'rvstd' : (0, 10),
        'rvstd_tropo' : (np.nan, np.nan),
        'rvstd_strato' : (np.nan, np.nan),
        'vcount' :  (1, np.nan),
        }
    
    if short_name == 'sf6': 
        vlim_dict.update(
            dict(vmean = (5.5, 10),
                 vstdv = (0, 2),
                 vstdv_tropo = (0.05, 0.3),
                 vstdv_strato = (0, 2),
                 ))
    elif short_name == 'detr_sf6': 
        vlim_dict.update(
            dict(vmean = (0.95, 1.05),
                 vstdv = (0, 0.05),
                 vstdv_tropo = (0.005, 0.015),
                 vstdv_strato = (0.005, 0.03),
                 rvstd_tropo = (0.005, 0.015),
                 rvstd_strato = (0.015, 0.035),
                 ))
    elif short_name == 'detr_co2': 
        vlim_dict.update(
            dict(vmean = (370, 395),
                 vstdv = (0.8, 3.0),
                 vstdv_tropo = (2.0, 3.0),
                 vstdv_strato = (1.2, 1.8),
                 rvstd_tropo = (0.5, 0.8),
                 rvstd_strato = (0.25, 0.5),
                 ))
    elif short_name == 'detr_ch4': 
        vlim_dict.update(
            dict(vmean = (1650, 1970),
                 vstdv = (16, 60),
                 vstdv_tropo = (16, 26),
                 vstdv_strato = (30, 60),
                 ))
    elif short_name == 'detr_n2o': 
        vlim_dict.update(
            dict(vmean = (290, 330),
                 vstdv = (0, 13),
                 vstdv_tropo =  (0.8, 1.8),
                 vstdv_strato = (5.1, 13),
                 ))
    elif short_name == 'detr_co': 
        vlim_dict.update(
            dict(vmean = (50, 150),
                 vstdv = (10, 30),
                 vstdv_tropo = (16, 30),
                 vstdv_strato = (10, 26),
                 ))
    elif short_name == 'ch4': 
        vlim_dict.update(
            dict(vmean = (1650, 1970),
                 ))
    elif short_name == 'co2': 
        vlim_dict.update(
            dict(vmean =  (370, 420),
                 ))
    elif short_name == 'n2o': 
        vlim_dict.update(
            dict(vmean = (290, 330),
                 ))
    elif short_name == 'co': 
        vlim_dict.update(
            dict(vmean = (15, 250),
                 ))
    elif short_name == 'o3': 
        vlim_dict.update(
            dict(vmean = (0.0, 1000),
                 vstdv = (0, 200),
                 vstdv_tropo = (15, 60),
                 vstdv_strato = (90, 200), 
                 rvstd = (0,10),
                 rvstd_tropo = (0,10),
                 rvstd_strato = (0,10),
                 ))
    elif short_name == 'h2o': 
        vlim_dict.update(
            dict(vmean = (0.0, 1000),
                 ))
    elif short_name == 'no': 
        vlim_dict.update(
            dict(vmean = (0.0, 0.6),
                 ))
    elif short_name == 'noy': 
        vlim_dict.update(
            dict(vmean = (0.0, 6),
                 ))      
    elif short_name == 'f11': 
        vlim_dict.update(
            dict(vmean = (130, 250),
                 ))
    elif short_name == 'f12': 
        vlim_dict.update(
            dict(vmean = (400, 540),
                 ))   
    else: 
        raise KeyError(f'There are no default vlims for {short_name}')

    if d_: 
        vlim_dict.update(
            {k : (v[0]/100, v[1]/100) for k, v in vlim_dict.items()
             if 'vmean' in k}
        )

    return vlim_dict

def substance_df():
    """ Get dataframe containing all info about all substance variables """
    with open(get_path() + 'substances.csv', 'rb') as f:
        substances = pd.read_csv(f)
    # update fct column with proper functions
    fctn_dict = {'h': fct.higher, 's': fct.simple, 'q': fct.quadratic}
    substances['function'] = [fctn_dict.get(f) for f in substances['function']]
    return substances

def get_substances(**kwargs) -> list['Substance']:
    """ Return list of Substance for all items conditions are met for. """
    df = substance_df()
    # keep only rows where all conditions are fulfilled
    for cond in kwargs:
        if cond not in df.columns:
            print(f'{cond} not recognised as valid substance qualifier.')
        else:
            df = df[df[cond] == kwargs[cond]]
    if len(df) == 0:
        raise KeyError('No substance column found using the given specifications')
    df.set_index('col_name', inplace=True)
    subs_dict = df.to_dict(orient='index')
    subs = [Substance(k, **v) for k, v in subs_dict.items()]
    return subs

def get_subs(*args, **kwargs):
    """ Returns the unique Substance object with the given specifications. 
    
    Args (Optional): 
        1. col_name (str)
        2. ID (str)
        3. clams (bool)
    """
    for i, arg in enumerate(args):  # substance, ID, clams
        pot_args = ['col_name', 'ID', 'clams']
        kwargs.update({pot_args[i]: arg})
    if 'short_name' not in kwargs and 'substance' in kwargs:
        kwargs.update({'short_name': kwargs.pop('substance')})
    if 'model' not in kwargs and 'clams' in kwargs:
        kwargs.update({'model': 'CLAMS' if kwargs.pop('clams') else 'MSMT'})

    substances = get_substances(**kwargs)
    if len(substances) > 1:
        raise Warning(f'Multiple columns fulfill the conditions: {substances}')
    return substances[0]

def lookup_fit_function(short_name):
    """ Get appropriate fit function for the given substance. """
    f_per_subs = dict(ch4='h', co='q', co2='h', n2o='s', sf6='q')
    function_per_f = dict(h=fct.higher, s=fct.simple, q=fct.quadratic)
    return function_per_f[f_per_subs[short_name]]

#%% Aircraft campaigns
def instrument_df() -> pd.DataFrame:
    """ Get dataframe containing all info about all substance variables """
    with open(get_path() + 'instruments.csv', 'rb') as f:
        instruments = pd.read_csv(f)
    return instruments

def instr_vars_per_ID_df() -> pd.DataFrame:
    """ Import information on variables available per instrument per campaign. """
    # variable names as stored on databank
    with open(get_path() + 'instr_vars_per_ID.csv', 'rb') as f:
        instr_vars_per_ID_df = pd.read_csv(f)
    return instr_vars_per_ID_df

def get_instruments(ID: str) -> set:
    """ Return all instruments and variables for a given ID / campaign. """
    df = instr_vars_per_ID_df()
    if ID not in df.columns:
        raise KeyError(f'Could not retrieve instruments / variable info for {ID}.')
    instruments = set(df[df[ID] == True]['instrument'])
    return instruments

def get_variables(ID: str, instr: str) -> set:
    """ Return all instruments and variables for a given ID / campaign. Harmonised. """
    df = instr_vars_per_ID_df()
    if instr not in get_instruments(ID):
        raise KeyError(f'Could not retrieve variables for {instr} in {ID}.')
    df = df[df[ID]] # choose only rows where ID value is True
    df = df[df['instrument'] == instr] # choose only rows of given instrument
    variables = set(df['variable'])
    return variables

def variables_per_instrument(instr: str = None) -> list:
    """ Returns all possible measured / modelled substances for original instrument name. """
    variable_dict = {
        'GCECD' : ['N2O', 'N2Oe', 'SF6', 'SF6e', 'CH4', 'CH4e'],
        'MMS' : ['P', 'T', 'G_LAT', 'G_LON', 'G_ALT', 'POT'],
        'UCATS-O3' : ['O3'],
        'CLAMS_MET' : ['EQLAT', 'PV', 'P', 'P_WMO', 'TH', 'TH_WMO', 'TH_PV'],
        'CLAMS' : ['EQLAT', 'P', 'P_TROP', 'TH', 'TH_TROP', 'D_2_0PVU_BOT'],
        'TRAJ_2PV' : ['P', 'LAT', 'DTH', 'DP'],
        'TRAJ_WMO' : ['P', 'PV', 'LAT', 'DTH', 'DP'],
        'BAHAMAS' : ['ALT', 'LAT', 'LON', 'POT'],
        'FAIRO' : ['O3'],
        'GHOST_ECD' : ['SF6'],
        'TRIHOP' : ['N2O', 'CH4', 'CO2'],
        'TRIHOP_N2O' : ['N2O'],
        'TRIHOP_CO2' : ['CO2'],
        'TRIHOP_CH4' : ['CH4'],
        'HAGAR' : ['CO2'],
        'HAGARV_LI' : ['CO2', 'CO2_err'],
        'HAGARV_ECD' : ['CH4'],
        'HAI14' : ['P'],
        'HAI26' : ['P'],
        'UMAQS' : ['N2O', 'CO2', 'CH4'],
        'GLORIA' : ['lapse_rate', 'altitude'],
        }

    if instr not in variable_dict:
        raise KeyError(f'Could not retrieve list of variables for {instr}.')
    return variable_dict[instr]

def harmonise_instruments(old_name):
    """ Harmonise campaign-speficic instrument names. """
    new_names = {
        'CLAMS_MET' : 'CLAMS',
        'TRAJ_2PV' : 'CLAMS',
        'TRAJ_WMO' : 'CLAMS',
        'TRIHOP_N2O' : 'TRIHOP',
        'TRIHOP_CO' : 'TRIHOP',
        'TRIHOP_CO2' : 'TRIHOP',
        'TRIHOP_CH4' : 'TRIHOP',
        'HAGARV_LI' : 'HAGAR',
        'GHOST_ECD' : 'GHOST',
        'UCATS-O3' : 'UCATS',
         }
    if old_name in new_names:
        return new_names[old_name]
    return old_name

def harmonise_variables(instr, var_name):
    """ Harmonise campaign-specific variable names. """
    df = instr_vars_per_ID_df()
    df = df[df['instrument'] == instr]

    if len(df) == 0:
        raise KeyError(f'Instrument list does not contain {instr}.')
    df = df[df['variable'] == var_name]

    col_names = set(df['col_name'].values)

    if len(col_names) == 1:
        new_name = col_names.pop()

    elif len(col_names) == 0:
        if not var_name in ['flight_id', 'measurement_id']:
            print(f'Could not find an entry for {instr}, {var_name}.')
        new_name = var_name
    elif len(col_names) > 1:
        raise Exception(f'Found multiple instrument + variable combinations: {instr}, {var_name} -> {col_names}')

    return new_name

class Instrument:
    """ Defines an instrument that may have flown on an aircraft campaign. """

    def __init__(self, original_name, **kwargs):
        """ Initialise instrument class instance. """
        self.original_name = original_name
        self.name = harmonise_instruments(original_name)

        instr_info = instrument_df().loc[instrument_df()['original_name'] == original_name]

        self.campaigns = [c for c in instr_info if instr_info[c].values == True]
        self.variables = variables_per_instrument(original_name)

        self.__dict__.update(kwargs)

    def __repr__(self):
        return f'Instrument : {self.original_name} - {self.variables}'

#%% Data: Campaigns / Instruments / Variables
def campaign_definitions(campaign: str) -> dict:
    """  Returns parameters needed for client_data_choice per campaign.

    Parameters:
        ghost_campaign (str): Name of the campaign, e.g. SOUTHTRAC
    """

    campaign_dicts = {
        "SOUTHTRAC" : dict(
            special = "ST all",
            ghost_ms_substances = ['HFC125', 'HFC134a', 'H1211', 'HCFC22'],
            n2o_instr = "UMAQS",
            n2o_substances = ["N2O", "CO", "CH4", "CO2"],
            flights=None),

        "TACTS" : dict(
            ghost_ms_substances = ['H1211'],
            flights = ["T1", "T2", "T3", "T4", "T5", "T6"],
            special = None,
            n2o_instr = ["TRIHOP_N2O", "TRIHOP_CO", "TRIHOP_CO2"],
            n2o_substances = [["N2O"], ["CO"], ["CO2"]]),

        "WISE" : dict(
            special = "WISE all",
            ghost_ms_substances = ['H1211'],
            n2o_instr = "UMAQS",
            n2o_substances = ["N2O", "CO"],
            flights=None),

        "PGS" : dict(
            special = "PGS all",
            ghost_ms_substances = ['H1211'],
            flights=None,
            n2o_instr = "TRIHOP",
            n2o_substances = ["N2O", "CO", "CH4"]),
        }

    if campaign not in campaign_dicts:
        raise KeyError(f'{campaign} is not a valid GHoST campaign for SQL database access.')

    return campaign_dicts[campaign]

def years_per_campaign(campaign: str) -> tuple:
    """ Return years of specified campaign as tuple. """
    year_dict = {
        'HIPPO' : (2009, 2010, 2011),
        'TACTS' : (2012),
        'PGS' : (2015, 2016),
        'WISE' : (2017),
        'SHTR' : (2019),
        'ATOM' : (2016, 2017, 2018),
        }
    if campaign not in year_dict:
        raise NotImplementedError(f'Campaign {campaign} does not have a year list.')
    return year_dict[campaign]

def instruments_per_campaign(campaign: str) -> tuple:
    """ Returns tuple of relevant instruments for a given campaign. """
    instr_dict = {
        'TACTS' : ('BAHAMAS', 'FAIRO', 'GHOST_ECD', 'TRIHOP_CO', 'TRIHOP_N2O', 'TRIHOP_CO2', 'TRIHOP_CH4', 'TRAJ_2PV', 'TRAJ_WMO'),
        'PGS' : ('BAHAMAS', 'FAIRO', 'GHOST_ECD', 'TRIHOP', 'HAGAR', 'HAI14', 'HAI16', 'CLAMS'),
        'WISE' : ('BAHAMAS', 'FAIRO', 'GHOST_ECD', 'UMAQS', 'HAGARV_LI', 'HAI14', 'HAI16', 'CLAMS_MET', 'HAGARV_ECD'),
        'SHTR' : ('BAHAMAS', 'FAIRO', 'GHOST_ECD', 'UMAQS', 'HAGARV_LI', 'CLAMS_MET', 'GLORIA'),
        'ATOM' : ('GCECD', 'MMS', 'UCATS-O3', 'CLAMS_MET'),
        }
    if campaign not in instr_dict:
        raise NotImplementedError(f'Campaign {campaign} does not have an instrument list.')
    return instr_dict[campaign]

def MS_variables(*args): 
    """ List of current variables of interest in the Caribic MS dataset """
    measured_coords = [
        # coordinates
        'Altitude', 
        'H_rel_TP', 
        'Tpot',
        ]
    measured_subs = [
        # substances
        'Ozone', 
        'CO', 
        'CO2', 
        'CH4', 
        ]
    modelled_ECMWF = [
        # coordinates
        'temp__k_', 
        'pv__pvu_', 
        'pot_temp__k_', 
        'eq_pott_temp__k_', 
        'z__0_1_g_m_', 
        'eq_latitude_deg_n_', 
        'p_strop__hpa_', 
        'p_dtrop__hpa_', 
        't_strop__k_', 
        't_dtrop__k_', 
        'pt_strop__k_', 
        'pt_dtrop__k_', 
        'pv_strop__pvu_', 
        'z_strop__01grav_m_', 
        'z_dtrop__01grav_m_', 
        'dp_strop__hpa_',
        'dp_dtrop__hpa_',
        ]
    variables = []
    if 'measured' in args: 
        [variables.append(i) for i in measured_coords + measured_subs]
    if 'ECMWF' in args: 
        [variables.append(i) for i in modelled_ECMWF]     
    return variables

#%% Misc for plotting
def dict_season():
    """ Use to get name_s, color_s for season s"""
    return {'name_1': 'Spring (MAM)', 'color_1': '#228833',  # blue
            'name_2': 'Summer (JJA)', 'color_2': '#AA3377',  # yellow
            'name_3': 'Autumn (SON)', 'color_3': '#CCBB44',  # red
            'name_4': 'Winter (DJF)', 'color_4': '#4477AA'}  # green
            # 'color_1': 'blue', 'color_2': 'orange',
            # 'color_3': 'green', 'color_4': 'red'}

colors = [
    '#E6E6E6', # 0.0 - 0.1
    '#D5D4E8', # 0.1 - 0.2
    '#A6BDDB', 
    '#5EA4CC', 
    '#FCC2AB', # 0.4 - 0.5
    '#FC9272', 
    '#F86144', 
    '#E12D26', # 0.7 - 0.8
    '#CB181D', # 0.8 - 0.9
]

def dict_colors():
    """ Get colorbars and colors for various variables. """
    return {
        'vmean': plt.cm.viridis,
        'vstdv': cmr.get_sub_cmap('hot_r', 0.1, 1),
        'vstdv_tropo': cmr.get_sub_cmap('YlOrBr', 0, 0.75),
        'vstdv_strato': plt.cm.BuPu,
        'rvstd_tropo': cmr.get_sub_cmap('YlOrBr', 0, 0.75),
        'rvstd_strato': plt.cm.BuPu,
        'diff': plt.cm.PiYG,
        'vcount': cmr.get_sub_cmap('plasma_r', 0.1, 0.9),
        'heatmap' : plt.cm.rainbow,
        'tropo' : 'm', # magenta 
        'always_tropo' : 'pink',
        'strato' : 'b', # blue
        'always_stato' : 'c', # cyan
        # 'rvstd' : cmr.get_sub_cmap('hot_r', 0.1, 1),
        # 'vcount' : plt.cm.PuRd,
        'rvstd' : lsc.from_list('RSTD_default', colors, N=9),
        'chem' : '#1f77b4',
        'therm' : '#ff7f0e',
        'dyn' : '#2ca02c',
    }

def dict_tps():
    """ Get color etc for tropopause definitions """
    return {'color_chem': '#1f77b4',
            'color_therm': '#ff7f0e',
            'color_dyn': '#2ca02c'}

def note_dict(fig_or_ax, x=None, y=None, s=None, ha=None):
    """ Return default arguments & bbox dictionary for adding notes to plots. """
    try:
        transform = fig_or_ax.transAxes
    except:
        transform = fig_or_ax.transFigure

    bbox_defaults = dict(edgecolor='lightgrey',
                         facecolor='lightcyan',
                         boxstyle='round')

    x = x if x else 0.97
    y = y if y else 0.97
    
    if not ha: 
        ha = 'right' if x > 0.5 else 'left'

    note_dict = dict(x=x,
                     y=y,
                     horizontalalignment = ha,
                     verticalalignment='center_baseline' if y > 0.5 else 'bottom',
                     transform=transform,
                     bbox=bbox_defaults)
    if s: note_dict.update(dict(s=s))

    return note_dict
