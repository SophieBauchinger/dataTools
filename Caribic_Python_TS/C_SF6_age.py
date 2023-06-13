# =============================================================================
# import general modules
# =============================================================================
import numpy as np
import pandas as pd
# import sys

# import matplotlib.cm as cm
# import matplotlib.pyplot as plt
# from matplotlib.colors import Normalize
# from matplotlib.cm import ScalarMappable as sm

# =============================================================================
# import own modules
# =============================================================================
from toolpac.readwrite import FFI1001_reader as tp_read
from toolpac.age import calculate_lag as tp_lag

# from toolpac.calc import bin_1d_2d
# from toolpac.readwrite import find
# from toolpac.readwrite.FFI1001_reader import FFI1001DataReader
# from toolpac.outliers import outliers, ol_fit_functions
# from toolpac.age import calculate_lag as cl
# from toolpac.conv.times import datetime_to_fractionalyear, fractionalyear_to_datetime

# sys.path.insert(0, r'C:\Users\sophie_bauchinger\sophie_bauchinger\toolpac_tutorial')
# from toolpac_tutorial import Mauna_Loa, Mace_Head, Caribic, Mozart

# def calc_sf6_time_lag(dataframe, ref_years, substance, ):
#     """ Calculate time lag for ames data when given reference time etc""" 
    
    
    
#     mlo_time_lims = (2000, 2020)
#     mlo_MM = Mauna_Loa(years = np.arange(*mlo_time_lims)).df #.df_monthly_mean
#     mlo_MM.resample('1M') # add rows for missing months, filled with NaN 
#     mlo_MM.interpolate(inplace=True) # linearly interpolate missing data
    
#     t_ref = np.array(datetime_to_fractionalyear(mlo_MM.index, method='exact'))
#     c_ref = np.array(mlo_MM['SF6catsMLOm'])
    
#     for c_year in range(2012, 2014):
#         c_data = Caribic([c_year]).df
#         t_obs_tot = np.array(datetime_to_fractionalyear(c_data.index, method='exact'))
#         c_obs_tot = np.array(c_data['SF6; SF6 mixing ratio; [ppt]\n'])
    
#         lags = []
#         for t_obs, c_obs in zip(t_obs_tot, c_obs_tot):
#             lag = cl.calculate_lag(t_ref, c_ref, t_obs, c_obs, plot=True)
#             lags.append((lag))
    
#         fig, ax = plt.subplots(dpi=300)
#         plt.scatter(c_data.index, lags, marker='+')
#         plt.title('CARIBIC SF$_6$ time lag {} wrt. MLO {} - {}'.format(c_year, *mlo_time_lims))
#         plt.ylabel('Time lag [yr]')
#         plt.xlabel('CARIBIC Measurement time')
#         fig.autofmt_xdate()

def calc_sf6_time_lag(Fdata, fpath, fname, ref='GL', trop=False):
    """ """
    # ref can be NH (northern hemisphere), SH (southern hemisphere)
    # TR (tropical), GL (global mean), or MLO (Mauna Loa)

    if not trop:
        if ref == 'MLO':
            ref_data_df = pd.read_csv(fpath / fname, header=51, skiprows=1, delim_whitespace=True)
            # polyfit for lagtime calculation cannot deal with missing values in y array
            print(ref_data_df)
            mod_ref_data_df = ref_data_df.dropna(how='any', subset=['SF6catsMLOm'])
            year = mod_ref_data_df['SF6catsMLOyr'].values
            month = mod_ref_data_df['SF6catsMLOmon'].values
            t_ref = year + (month-0.5)/12
            c_ref = mod_ref_data_df['SF6catsMLOm'].values
            print('File ', fpath / fname, ' read.')
        else:
            ref_data_df = read_Agage_ref_data(fpath, fname)
            t_ref = ref_data_df.iloc[:, 0].values  # eq. to ref_data_df[ref_data_df.columns[0]]
            c_ref = ref_data_df[ref + '_SF6\t[ppt]\n'].values
            print('File ', fpath / fname, ' read.')
    else:
        if ref == 'NHMBL':
            ref_data_df = pd.read_csv(fpath / fname, header=None, delim_whitespace=True)
            # polyfit for lagtime calculation cannot deal with missing values in y array
            mod_ref_data_df = ref_data_df.dropna(axis=0, how='any')
            t_ref = ref_data_df[0]
            c_ref = ref_data_df[1]
            print('File ', fpath / fname, ' read.')

    np.seterr(invalid='ignore')
    # suppress warnings for NaN elements contained in SF6 array in > < comparisons in calculate_lag

    ghg_keys = [x for x in Fdata.keys() if x.endswith('_GHG')]
    for key in ghg_keys:
        GHG_df = Fdata[key]
        if GHG_df is not None:
            t_obs = GHG_df.year_frac.values.mean()  # only one point in time
            c_obs = GHG_df['sf6'].values
            # print(key, t_obs)
            lag = tp_lag.calculate_lag(t_ref, c_ref, t_obs, c_obs, degree=3, fitint=10)
            # default is degree=2, fitint=10
            # if fitint is too small
            if trop:
                Fdata[key]['trop_sf6_lag'] = lag.tolist()
            else:
                Fdata[key]['strat_sf6_lag'] = lag.tolist()

    return


# read reference time series for lag time calculation from Agage Ames file
def read_Agage_ref_data(path, fname):
    sf6_ames_dict = tp_read.FFI1001DataReader(path / fname, sep_data="\t", df=True)
    df = sf6_ames_dict.df

    return df

