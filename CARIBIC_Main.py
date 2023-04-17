
# =============================================================================
# import general modules
# =============================================================================
import pandas as pd
# use modin.pandas as alternative, faster than normal pandas, for large data sets,
# https://pypi.org/project/modin/
# https://www.machinelearningplus.com/modin-speedup-pandas/

import numpy as np
from pathlib import Path
from importlib import reload

# =============================================================================
# import own modules
# =============================================================================
import C_read
import C_SF6_age
import C_filter
import C_tools

# =============================================================================
# define list of files for which data is to be imported
# =============================================================================
flight_list_name = 'flight_list_GHG'
# list created with function create_flight_list_file from C_tools

# =============================================================================
# define global variables
# =============================================================================

caribic2data = Path('E:\CARIBIC\Caribic2data')
# traj_path = Path('E:\CARIBIC\Trajectories')

# SF6 reference data for stratospheric age/time lag calculation
# NOAA MLO:
sf6_path = Path(r'C:\Users\sophie_bauchinger\sophie_bauchinger\toolpac_tutorial')
sf6_MLO_fname = 'mlo_SF6_MM.dat'

strat_age_ref = 'MLO'
n2o_path = sf6_path                               # r needed to deal with underscore in path name
n2o_fname = 'mlo_N2O_MM.dat'                    # read NOAA MLO data for strat-trop filter

# SF6 reference data for tropospheric age/time lag calculation
# sf6_NHMBL_fname = 'zone_nh.mbl.sf6'
# trop_age_ref = 'NHMBL'


# Agage Ames:
# sf6_path = Path(r'D:\Python\CARIBIC\SF6_age_data')
# sf6_fname = 'SF6_20200813_Agage_trend.ames'
# here  ref can be NH (northern hemisphere), SH (southern hemisphere), TR (tropical), GL (global mean)

# strat / trop / outlier filter:
filter_crit = 'N2O'  # options are 'N2O', 'PV', 'O3' # use N2O for GHG and HCF  data, consider O3 for HCF data


# =============================================================================
# CARIBIC flight information
df_flights = C_read.read_flight_list(caribic2data, flight_list_name)
flight_numbers = df_flights.index.tolist()

df_route_colors = pd.DataFrame(columns=['route', 'color'])
df_route_colors['route'] = ['North_America', 'South_Am_north', 'South_Am_south',
                            'Asia_south', 'Asia_east', 'Africa', 'Europe']
df_route_colors['color'] = ['red', 'darkviolet', 'blue', 'gold', 'lime', 'cyan', 'lightgrey']
df_route_colors.index = df_route_colors['route']

# =============================================================================
# read all MS, INT and HCF
# name convention df_XXX_YYY with XXX flight number and YYY prefix
# =============================================================================

# create empty dictionary
Fdata = {}

# dict_new = C_read.read_flights_to_dict(caribic2data, flight_numbers, 'MS')
# Fdata.update(dict_new)

dict_new = C_read.read_flights_to_dict(caribic2data, flight_numbers, 'INT')
Fdata.update(dict_new)

dict_new = C_read.read_flights_to_dict(caribic2data, flight_numbers, 'INT2', high_res=True)
Fdata.update(dict_new)

dict_new = C_read.read_flights_to_dict(caribic2data, flight_numbers, 'GHG')
Fdata.update(dict_new)

# # do not read HCF and HFC for testing
# dict_new = C_read.read_flights_to_dict(caribic2data, flight_numbers, 'HCF')
# Fdata.update(dict_new)

# dict_new = C_read.read_flights_to_dict(caribic2data, flight_numbers, 'HFO')
# Fdata.update(dict_new)
# del dict_new

# calculate stratospheric time lag from sf6 mixing ratios at MLO:
C_SF6_age.calc_sf6_time_lag(Fdata, sf6_path, sf6_MLO_fname, ref=strat_age_ref, trop=False)
# calculate tropospheric time lag from sf6 mixing ratios in the NH marine boundary layer:
# C_SF6_age.calc_sf6_time_lag(Fdata, sf6_path, sf6_NHMBL_fname, ref=trop_age_ref, trop=True)


# get stratosphere/troposphere flags based on N2O mixing ratio
# INT data has to be read to perform filtering, even if crit='n2o' is used
C_filter.filter_strat_trop(Fdata, 'n2o', pv_lim=2., o3_lim=100., n2o_path=n2o_path, n2o_fname=n2o_fname, plot=True)

# get tropospheric outliers
subst_list = ['co2', 'ch4', 'n2o', 'sf6', 'trop_sf6_lag', 
              'hfc_125', 'hfc_134a', 'halon_1211', 'cfc_12', 'hcfc_22', 'int_co']
df_func_list = {'co2': 'higher',
                'ch4': 'higher',
                'n2o': 'simple',
                'sf6': 'quadratic',
                'trop_sf6_lag': 'quadratic', 
                'sulfuryl_fluoride': 'simple',
                'hfc_125': 'simple',
                'hfc_134a': 'simple',
                'halon_1211': 'simple',
                'cfc_12': 'simple',
                'hcfc_22': 'simple',
                'int_co': 'quadratic'}
                
                # substances not present in df_func_list will be done with 'simple'
                # therefore it's sufficient to only have those for which 'simple' does not work well
                

# cutting flag data and filling Fdata[FXXX_FLG] for some reason does not work properly
# use df_return=True and do manually

df_tmp_flag = C_filter.filter_outliers(Fdata, subst_list, df_func_list, plot=True, df_return=True, outlier_lim=0.1)
if filter_crit == 'N2O':
    df_merge = C_tools.do_data_merge(Fdata, flight_numbers, ['GHG'])    # prefixes have to be supplied as list
    # print(df_merge['n2o'].isnull().sum())
    for subst in subst_list:
    # print(subst)
        df_tmp_flag.loc[df_merge['n2o'].isnull(), f'ol_{subst}'] = np.nan
        df_tmp_flag.loc[df_merge['n2o'].isnull(), f'ol_rel_{subst}'] = np.nan
        df_tmp_flag.loc[df_merge['n2o'].isnull(), f'fl_{subst}'] = np.nan
    
columns = [f'fl_{x}' for x in subst_list] + [f'ol_{x}' for x in subst_list] + [f'ol_rel_{x}' for x in subst_list]
# compile C_tools
# runfile('D:/Python/CARIBIC/C_tools.py', wdir='D:/Python/CARIBIC')
# do_data_cut(df_tmp_flag, Fdata, 'FLG', columns, over=True, over_all=False)
# do_data_cut(df_tmp_flag, Fdata, 'FLG', columns, over=True, over_all=False)

# Fdata['FXXX_FLG'] should now contain ol-values and fl-numbers but may for some unknown reason not. Try again then.

