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
import matplotlib.pyplot as plt
import cmasher as cmr

# %% Coordinates
class Coordinate:
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
        self.col_name, self.long_name, self.unit, self.ID = [None] * 4
        self.vcoord, self.hcoord, self.var = [None] * 3
        self.tp_def, self.rel_to_tp, self.model, self.pvu, self.crit = [None] * 5

        self.__dict__.update(kwargs)
        if self.pvu is not np.nan:
            self.pvu = float(self.pvu)

    def __repr__(self):
        return f'Coordinate: {self.col_name} [{self.unit}] from {self.ID}'

    def label(self, filter_label=False, coord_only=False):
        """ Returns string to be used as axis label for a specific Coordinate object. """

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

    def get_bsize(self):
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
            print(f'No default bin size for {self.vcoord} / {self.hcoord}')
            return None


def coordinate_df():
    """ Get dataframe containing all info about all coordinate variables """
    with open('coordinates.csv', 'rb') as f:
        coord_df = pd.read_csv(f)
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


def get_coord(**kwargs):
    coordinates = get_coordinates(**kwargs)  # dict i:Coordinate
    if len(coordinates) > 1:
        raise ValueError(f'Multiple columns fulfill the conditions: {[i.col_name for i in coordinates]}')
    return coordinates[0]


# %% Substances
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
        self.long_name, self.short_name, self.unit = [None] * 3
        self.ID, self.source, self.model = [None] * 3
        self.function, self.detr = [None] * 2

        self.__dict__.update(kwargs)

    def __repr__(self):
        return f'Substance : {self.short_name} [{self.unit}] - \'{self.col_name}\' from {self.ID}'

    def label(self, name_only=False, delta=False):
        """ Returns string to be used as axis label. """

        special_names = {'ch2cl2': 'CH$_2$Cl$_2$',
                         'noy': r'NO$_\mathrm{y}$',
                         'f11': 'F11',
                         'f12': 'F12',
                         'mol mol-1': 'mol/mol'}

        code = self.short_name.split('_')[-1]
        if code in special_names:
            code = special_names[code]
        else:
            code = code.upper()
            code = ''.join(f"$_{i}$" if i.isdigit() else i for i in code)
        if self.short_name.startswith('d_'):
            code = r'$\mathrm{\sigma}\,$' + f'({code})'

        if name_only:
            return code

        if self.model == 'MSMT':
            identifier = self.source if not self.source=='HALO' else self.ID
        elif self.model == 'CLAMS':
            identifier = 'CLaMS'
        else:
            identifier = self.model
        if self.source!='HALO' and len(get_substances(
                short_name=self.short_name, source=self.source, model=self.model)) > 1:
            identifier += f' - {self.ID}'

        unit = self.unit if not self.unit in special_names else special_names[self.unit]

        if not self.detr:
            return f'{code} [{unit}] ({identifier})'

        elif self.detr and not delta:
            return f'{code} detrended wrt. 2005 [{unit}]'

        elif self.detr and delta:
            return f'{code} ' + r'(-$\Delta_{2005}$)' + f' [{unit}]'

        else:
            raise KeyError('Could not generate a label')

    def vlims(self, bin_attr='vmean', atm_layer=None) -> tuple:
        """ Default colormap normalisation limits for substance mixing ratio or variabiliy. """

        def get_vlims(subs_short, bin_attr='vmean', atm_layer=None) -> tuple:
            """ Default colormap normalisation limits for substance mixing ratio or variability. """
            vlims_mxr = {  # optimised for Caribic measurements from 2005 to 2020
                'sf6': (5.5, 10),
                'co2': (370, 420),
                'ch4': (1650, 1970),
                'n2o': (290, 330),

                'co': (15, 250),
                'o3': (0.0, 1000),
                'h2o': (0.0, 1000),
                'no': (0.0, 0.6),
                'noy': (0.0, 6),
                'f11': (130, 250),
                'f12': (400, 540),

                'detr_sf6': (5.5, 6.5),
                'detr_co2': (370, 395),
                'detr_ch4': (1650, 1970),
                'detr_n2o': (290, 330),
                'detr_co': (50, 150)
            }

            if bin_attr == 'vmean':
                try:
                    return vlims_mxr[subs_short]
                except:
                    raise KeyError(f'No default vlims for {subs_short}. ')

            vlims_stdv_total = {
                'detr_sf6': (0, 0.3),
                'sf6': (0, 2),
                'detr_n2o': (0, 13),
                'detr_co': (10, 30),
                'detr_co2': (0.8, 3.0),
                'detr_ch4': (16, 60),
            }

            vlims_stdv_tropo = {
                'detr_sf6': (0.05, 0.15),
                'sf6': (0.05, 0.3),
                'detr_n2o': (0.8, 1.8),
                'detr_co': (16, 30),
                'detr_co2': (2.0, 3.0),
                'detr_ch4': (16, 26),
            }

            vlims_stdv_strato = {
                'detr_sf6': (0.05, 0.3),
                'sf6': (0, 2),
                'detr_n2o': (5.1, 13),
                'detr_co': (10, 26),
                'detr_co2': (1.2, 1.8),
                'detr_ch4': (30, 60),
            }

            if bin_attr == 'vstdv':
                if not atm_layer:
                    return vlims_stdv_total[subs_short]
                elif atm_layer == 'tropo':
                    return vlims_stdv_tropo[subs_short]
                elif atm_layer == 'strato':
                    return vlims_stdv_strato[subs_short]
                else:
                    raise KeyError(f'No default vlims for {subs_short} STDV in {atm_layer}')

            if bin_attr == 'vcount':
                return (1, np.nan)

        if not self.short_name.startswith('d_'):
            return get_vlims(self.short_name, bin_attr, atm_layer)
        else:
            return (0, 1)

def substance_df():
    """ Get dataframe containing all info about all substance variables """
    with open('substances.csv', 'rb') as f:
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
    """ Return single Substance object with the given specifications """
    for i, arg in enumerate(args):  # substance, ID, clams
        pot_args = ['short_name', 'ID', 'clams']
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
    with open('instruments.csv', 'rb') as f:
        instruments = pd.read_csv(f)
    return instruments

def instr_vars_per_ID_df() -> pd.DataFrame:
    """ Import information on variables available per instrument per campaign. """
    # variable names as stored on databank
    with open('instr_vars_per_ID.csv', 'rb') as f:
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


#%%% SQL Database / HALO stuff
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

# %% Misc
def dict_season():
    """ Use to get name_s, color_s for season s"""
    return {'name_1': 'Spring (MAM)', 'color_1': '#228833',  # blue
            'name_2': 'Summer (JJA)', 'color_2': '#AA3377',  # yellow
            'name_3': 'Autumn (SON)', 'color_3': '#CCBB44',  # red
            'name_4': 'Winter (DJF)', 'color_4': '#4477AA'}  # green
            # 'color_1': 'blue', 'color_2': 'orange',
            # 'color_3': 'green', 'color_4': 'red'}


def dict_colors():
    """ Get colorbars and colors for various variables. """
    return {
        'vmean': plt.cm.viridis,
        'vstdv': cmr.get_sub_cmap('summer_r', 0.1, 1),
        'vstdv_tropo': cmr.get_sub_cmap('YlOrBr', 0, 0.75),
        'vstdv_strato': plt.cm.BuPu,
        'diff': plt.cm.PiYG,
        'vcount': cmr.get_sub_cmap('plasma_r', 0.1, 0.9),
        # 'vcount' : plt.cm.PuRd,
    }


def axis_label(coord):
    """ Return axis label for vcoord / hcoord. """
    label_dict = {
        'pt': '$\Theta$',
        'z': 'z',
        'p': 'p',
        'lat': 'Latitude [°N]',
        'lon': 'Longitude [°E]',
        'mxr': 'N$_2$O mixing ratio'
    }
    return label_dict[coord]


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

    note_dict = dict(x=x,
                     y=y,
                     horizontalalignment='right' if x > 0.5 else 'left',
                     verticalalignment='center_baseline' if y > 0.5 else 'bottom',
                     transform=transform,
                     bbox=bbox_defaults)
    if s: note_dict.update(dict(s=s))

    return note_dict
