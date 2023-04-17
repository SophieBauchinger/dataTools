# =============================================================================
# import general modules
# =============================================================================
import pandas as pd
import numpy as np
from pathlib import Path

# =============================================================================
# import own modules
# =============================================================================
from toolpac.outliers import ol_fit_functions as fct
from toolpac.outliers import outliers
from toolpac.conv.times import datetime_to_fractionalyear, fractionalyear_to_datetime

import C_tools


# %%
def filter_strat_trop(Fdata, ref_data, crit, pv_lim=2., o3_lim=100., plot=True,
                      return_merge=False):
    """ 
    crit can be n2o, o3, or pv
    for crit == n2o path and filename have to be supplied
    pv_lim sets default value for PV filter
    """

    if crit == 'n2o':
        # if n2o_path == '' or n2o_fname == '':
        #     print('Parameters n2o_path and/or n2o_fname empty.')
        #     return

        mlo_lim = 0.97  # percentage below which measured N2O values will be defined stratospheric
        # print(f'Filtering by N2O using data file {n2o_path}\{n2o_fname}. Using {mlo_lim} as pre-filter percentage.')
        # read MLO N2O data
        df_n2o_mlo = ref_data # pd.read_csv(Path(n2o_path, n2o_fname), header=37, skiprows=1, delim_whitespace=True)
        # drop missing values to make sure that always a results is obtained
        mod_ref_data_df = df_n2o_mlo.dropna(how='any', subset=['N2OcatsMLOm'])
        year = mod_ref_data_df.index.year
        month = mod_ref_data_df.index.month
        mlo_t_ref = year + (month - 0.5) / 12
        mlo_mxr_ref = mod_ref_data_df['N2OcatsMLOm'].values
        mlo_fit = np.poly1d(np.polyfit(mlo_t_ref, mlo_mxr_ref, 2))
        print(f'MLO fit parameters obtained: {mlo_fit}')
        ref_prefix = 'INT'

    if crit.lower() == 'pv':
        print(f'Filtering by PV using limit of {pv_lim} PVU.')
        ref_prefix = 'INT'

    if crit.lower() == 'o3':
        print(f'Filtering by O3 using in_h_rel_tp or a limit of {o3_lim}ppbV.')
        ref_prefix = 'INT'

    ref_keys = ['GHG'] # [x for x in Fdata.keys() if x.endswith(f'_{ref_prefix}')]
    print('ref keys', ref_keys)
    # some roundabout way of keeping the source information in the dictionary, but also flagging stuff depending on the mlo_fit function
    for key in ref_keys:
        flag_key = key.replace(ref_prefix, 'FLG')  # add key to dictionary, using information from INT or GHG-dataframes
        Fdata[flag_key] = Fdata[key][['timecref', 'year', 'month', 'day', 'hour', 'min', 'sec',
                                      'year_frac', 'season']].copy()

        Fdata[flag_key]['strato'] = np.nan
        print('Fdata', Fdata)
        Fdata[flag_key]['tropo'] = np.nan

        # if crit.lower() == 'pv':
        #     Fdata[flag_key].loc[Fdata[key]['int_pv'] > pv_lim, ('strato', 'tropo')] = (True, False)
        #     Fdata[flag_key].loc[Fdata[key]['int_pv'] <= pv_lim, ('strato', 'tropo')] = (False, True)

        # if crit.lower() == 'o3':
        #     # if int_h_rel_tp has a value then decide strato/tropo by that
        #     Fdata[flag_key].loc[Fdata[key]['int_h_rel_tp'] > 0., ('strato', 'tropo')] = (True, False)
        #     Fdata[flag_key].loc[Fdata[key]['int_h_rel_tp'] < 0., ('strato', 'tropo')] = (False, True)
        #     # if int_h_rel_tp is only calculated north of 35N (slightly different for early flights)
        #     # and for O3 > 60 ppbV
        #     # for other cases use a crude limit of 100ppb
        #     Fdata[flag_key].loc[((Fdata[key]['int_h_rel_tp'].isna()) & (Fdata[key]['int_o3'] > o3_lim)),
        #                         ('strato', 'tropo')] = (True, False)
        #     Fdata[flag_key].loc[((Fdata[key]['int_h_rel_tp'].isna()) & (Fdata[key]['int_o3'] <= o3_lim)),
        #                         ('strato', 'tropo')] = (False, True)
        #     Fdata[flag_key].loc[Fdata[key]['int_o3'].isna(),
        #                         ('strato', 'tropo')] = (np.nan, np.nan)

        if crit == 'n2o':
            print('replaying ghg key')
            ghg_key = key.replace('INT', 'GHG')
            if Fdata[ghg_key] is not None:
                Fdata[flag_key].loc[Fdata[ghg_key]['n2o'] < mlo_lim * mlo_fit(Fdata[key]['year_frac']),
                                    ('strato', 'tropo')] = (True, False)
            print('Fdata line 87', Fdata)

    if crit == 'n2o':
        print('Pre-flagging done.')
        outlier_lim = 0.1
        ref_year = 2005.

        flights_in_Fdata = Fdata["Flight number"].unique() # list(set([int(x.split('_', 1)[0][1:4]) for x in Fdata.keys()]))

        pref_in_Fdata = ['GHG'] # list(set([x.split('_', 1)[1] for x in Fdata.keys()]))
        print('pref in Fdata', pref_in_Fdata)
        df_merge = C_tools.do_data_merge(Fdata, flights_in_Fdata, pref_in_Fdata)
        print('df_merge, line 99', df_merge)

        df_merge['year_delta'] = datetime_to_fractionalyear(df_merge.index)
        df_merge['year_delta'] = df_merge['year_delta'] - ref_year
        # create new dataframe for flags and set initial flags to 0
        data_flag = pd.DataFrame(df_merge, columns=['flight', 'timecref', 'year', 'month', 'day'])
        data_flag['n2o'] = 0
        # set flag of pre-filtered samples to -1 to exclude them from fit following later
        data_flag.loc[df_merge['strato'] == True, 'n2o'] = -1
        # needed for pre-flagging, not for later use
        # can be checked e. g. by
        # data_flag['n2o'].value_counts()

        # to find stratospheric values it is sufficient to look for negative outliers, direction = 'n'
        # which is also faster than direction='pn', outlier filter will still find outliers below baseline later
        # independent of 'n' or 'pn' used here
        # function modifies data_flag
        print('prior flag count:\n', data_flag['n2o'].value_counts())
        outliers.ol_iteration_for_subst('n2o', df_merge, data_flag,
                                        func=fct.simple, direction='n', limit=outlier_lim, plot=plot)

        print('post flag count:\n', data_flag['n2o'].value_counts())

        df_merge['strato'] = np.nan
        df_merge['tropo'] = np.nan
        df_merge.loc[data_flag['n2o'] < 0, ('strato', 'tropo')] = (True, False)
        df_merge.loc[data_flag['n2o'] == 0, ('strato', 'tropo')] = (False, True)
        print('N2O flagging done. Distributing data.')

        # do some magic to integrate strato/tropo data into GHG dataframes in Fdata dictionary:
        C_tools.do_data_cut(df_merge, Fdata, 'FLG', ['timecref', 'year', 'month', 'day', 'year_frac', 'season',
                                                      'strato', 'tropo'], over_all=True)  # overwrite existing FLG data

        # this should not be necessary
        # for key in int_keys:
        #     ghg_key = key.replace('INT', 'GHG')
        #     flag_key = key.replace('INT', 'FLG')
        #     # samples with no valid n2o value have undefined strato/tropo
        #     Fdata[flag_key].loc[Fdata[ghg_key]['n2o'].isna(),
        #                         ('strato', 'tropo')] = (np.nan, np.nan)
        # return
        if return_merge:
            return df_merge
    return


# %%
def reset_flags(df_merge):
    data_flag = pd.DataFrame(df_merge, columns=['flight', 'timecref', 'year', 'month', 'day'])
    data_flag['n2o'] = 0
    # set flag of pre-filtered samples to -1 to exclude them from fit following later
    data_flag.loc[df_merge['strato'] == True, 'n2o'] = -1

    return data_flag


# %%
def filter_outliers(Fdata, subst_list, df_func_list, plot=False, df_return=False, outlier_lim=0.1):
    # merge datasets int, ghg, hcf, hfo
    # check for subst in column names of int, ghg, hcf, hfo
    # take tropospheric data and search outliers direction = 'pn'
    # returns one dataframe containing flags
    # add flags to Fdata['FXXX_FLG'][subst]

    ref_year = 2005.  # reference year, shouldn' matter too much, used to get smaller numbers on time axis

    # merge all data, all flights, all sample data types
    print('Merging data ...')
    flights_in_Fdata = list(set([int(x.split('_', 1)[0][1:4]) for x in Fdata.keys()]))
    pref_in_Fdata = list(set([x.split('_', 1)[1] for x in Fdata.keys()]))
    if 'MS' in pref_in_Fdata:
        pref_in_Fdata.remove("MS")
    if 'SU' in pref_in_Fdata:
        pref_in_Fdata.remove("SU")

    df_merge = C_tools.do_data_merge(Fdata, flights_in_Fdata, pref_in_Fdata)
    # create alternative time axis
    df_merge['year_delta'] = df_merge['year_frac']
    df_merge['year_delta'] = df_merge['year_delta'] - ref_year

    # create new dataframe for flags
    data_flag = pd.DataFrame(df_merge, columns=['flight', 'timecref', 'year', 'month', 'day', 'strato', 'tropo'])
    data_flag.columns = [f'fl_{x}' if x in subst_list else x for x in data_flag.columns]

    # loop over all substances supplied
    for subst in subst_list:
        print(subst)
        # check if subst is in any data column
        if subst not in df_merge.columns:
            print(f'\n{subst} not found in any dataframe, check spelling')
            subst_list.remove(subst)
        else:
            # check if a function is supplied in dictionary df_func_list (to be defined in CARIBIC_Main)
            if subst in df_func_list.keys():
                func = 'fct.' + df_func_list[subst]
            else:
                print(f'No function found for {subst} - using 2nd order polynomial with simple harmonic '
                      f'if subst is found in data.')
                func = 'fct.simple'
            # create columns for flag and residual in flag dataframe and set all flags to 0
            data_flag[f'ol_{subst}'] = np.nan
            data_flag[f'ol_rel_{subst}'] = np.nan
            data_flag[f'fl_{subst}'] = 0

            # set all strato flags to a value != 0 to exclude them
            data_flag.loc[df_merge['strato'] == True, f'fl_{subst}'] = -20

            # do the outlier magic
            # func is a string that needs to be evaluated to find the matching function definition
            # find_ol returns flag, residual, warning, popt1
            time = df_merge['year_delta'].tolist()
            mxr = df_merge[subst].tolist()
            if f'd_{subst}' in df_merge.columns:
                d_mxr = df_merge[f'd_{subst}'].tolist()
            else:    # case for integrated values of high resolution data
                d_mxr = None
            flag = data_flag[f'fl_{subst}'].tolist()
            tmp = outliers.find_ol(eval(func), time, mxr, d_mxr, flag, direction='pn',
                                   plot=plot, limit=outlier_lim)

            data_flag[f'fl_{subst}'] = tmp[0]  # flag
            data_flag[f'ol_{subst}'] = tmp[1]  # residual
            # only looking for tropospheric outliers:
            data_flag.loc[data_flag['strato'] == True, f'fl_{subst}'] = np.nan
            data_flag.loc[data_flag['strato'] == True, f'ol_{subst}'] = np.nan

            # no residual value for non-outliers
            # data_flag.loc[data_flag[f'fl_{subst}'] == 0, f'ol_{subst}'] = np.nan

            fit_result = [eval(func)(t, *tmp[3]) for t in time]
            # print(len(fit_result), len(data_flag))
            data_flag[f'ol_rel_{subst}'] = data_flag[f'ol_{subst}'] / fit_result

    # cut data into flights and add to Fdata['FXXX_FLG']
    columns = [f'fl_{x}' for x in subst_list] + [f'ol_{x}' for x in subst_list] + [f'ol_rel_{x}' for x in subst_list]

    # C_tools.do_data_cut(data_flag, Fdata, 'FLG', columns, over=True, over_all=False)
    # calling here does not work because flags need to be checked again taking into account filter criterion
    # overwrites fl_subst and ol_subst columns of existing dataframe in dictionary

    if df_return:
        return data_flag

    return
