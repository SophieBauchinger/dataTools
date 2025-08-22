# -*- coding: utf-8 -*-
"""Ozone data tools, e.g. calculation of height relative to tropopause."""
from ast import literal_eval
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d

import yaml


# ------------------------------------------------------------------------------


def load_tp_hght_data(path: Path, sep=";", v_scal=1) -> dict:
    """
    Load rel. TP height data from csv file.

    Parameters
    ----------
    path : str or Path
        DESCRIPTION.
    sep : str, optional
        DESCRIPTION. The default is ";".
    v_scal : int, optional
        DESCRIPTION. The default is 1.

    Returns
    -------
    dict
        height relative to tropopause data.
    """
    with open(path, "r") as file_obj:
        tp_data = file_obj.readlines()

    for i, line in enumerate(tp_data):
        tp_data[i] = line.strip().rsplit(sep)

    tp_hght = np.array(tp_data[0][1:], dtype=float)
    time = []
    ozone = []

    for line in tp_data[1:-1]:
        time.append(int(line[0]))
        ozone.append([float(o) * v_scal for o in line[1:]])

    time = np.array(time)
    ozone = np.array(ozone)

    result = {}
    result["tp_hght"] = tp_hght
    result["montly_avg"] = {}
    result["montly_avg"]["month"] = time
    result["montly_avg"]["ozone"] = ozone

    # convert time given as monthly mean to days of year
    doy = time * 365 / 12 - (365 / 12 / 2)  # ignore leap years and so on...
    time = np.array(list(range(1, 366)))
    tmp_o3 = np.zeros([len(time), len(tp_hght)])
    for i in range(len(tp_hght)):  # interpolate O3 for each TP height
        f_ip = interp1d(
            doy,
            ozone[:, i],
            kind="linear",
            bounds_error=False,
            fill_value="extrapolate",
        )
        tmp_o3[:, i] = f_ip(time)
    ozone = tmp_o3

    result["DOY_interp"] = {}
    result["DOY_interp"]["DOY"] = time
    result["DOY_interp"]["ozone"] = ozone

    return result


# ------------------------------------------------------------------------------


def calc_o3tp_relhght(
    tpdata: dict,
    v_ozone: float,
    sel_month: int,
    sel_day_of_month=False,
    vmiss=9999,
    v_ozone_min=60.0,
    hreltp_min=-1.5,
) -> float:
    """
    Find relative tropopause height for a given ozone value, v_ozone.
    result is based on linear interpolation of rel. TP heights that correspond
    to the ozone values closest to v_ozone.

    Parameters
    ----------
    tpdata : dict
        DESCRIPTION.
    v_ozone : float
        DESCRIPTION.
    sel_month : int
        DESCRIPTION.
    sel_day : TYPE, optional
        if specified, monthly tpdata is interpolated linearly to days of
            year. The default is False.
    vmiss : TYPE, optional
        DESCRIPTION. The default is 9999.
    v_ozone_min : TYPE, optional
        DESCRIPTION. The default is 60.0.
    hreltp_min : TYPE, optional
        DESCRIPTION. The default is -1.5.

    Returns
    -------
    float
        height relative to tropopause or VMISS.
    """
    if v_ozone <= v_ozone_min:
        return vmiss  # v_ozone too low, return vmiss

    # first line: TP heights
    tp_hght = tpdata["tp_hght"]

    if sel_day_of_month:
        sel_time = (
            datetime(2010, sel_month, sel_day_of_month) - datetime(2010, 1, 1)  # doy from time-
        ).days + 1  # delta object
        time = tpdata["DOY_interp"]["DOY"]
        ozone = tpdata["DOY_interp"]["ozone"]
    else:
        sel_time = sel_month
        time = tpdata["montly_avg"]["month"]
        ozone = tpdata["montly_avg"]["ozone"]

    # find corresponding time index
    ix_t = np.arange(len(time))[np.where(time == sel_time)]

    # search corresponding ozone array for adjacent values
    sel_ozone = ozone[ix_t]
    sel_ozone = sel_ozone.reshape(len(sel_ozone[0]))  # flatten...
    ix_close = np.array([np.argmin(np.abs(sel_ozone - v_ozone))], dtype=int)

    if ix_close in (0, len(sel_ozone) - 1):
        # no bracketing value! interpolation not possible,
        return vmiss

    # there is a bracketing value...
    # find the next closest / = bracketing o3 value
    o3_close = sel_ozone[ix_close]
    sel_ozone[ix_close] = vmiss
    ix_brack = np.array([np.argmin(np.abs(sel_ozone - v_ozone))], dtype=int)

    # interpolate H_rel_TP based on rel. distance to given ozone value
    rel_dist_o3 = np.abs(o3_close - v_ozone) / np.abs(o3_close - sel_ozone[ix_brack])
    result = (tp_hght[ix_close] + (tp_hght[ix_brack] - tp_hght[ix_close]) * rel_dist_o3)[0]

    if result <= hreltp_min:
        result = vmiss  # h_rel_TP too low, return vmiss

    return result


# ------------------------------------------------------------------------------


def FAIROeval_load_settings(fname: Path, nkeys_line=5, delimiter=";", verbose=False) -> dict:
    """
    Load old FAIROeval configuration from config file.

    parameters:
        fname - filepath of config file
    keywords:
        nkeys_line - line index in config file where to look for the number of
                     defined keys. 1-based!
        delimiter - ; by default for csv
        verbose - activates some print statements to the console

    Parameters
    ----------
    fname : [str, Path]
        DESCRIPTION.
    nkeys_line : TYPE, optional
        DESCRIPTION. The default is 5.
    delimiter : TYPE, optional
        DESCRIPTION. The default is ";".
    verbose : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    settings : dict
        FAIROeval settings.
    """
    with open(fname, "r") as file_obj:
        set_raw = file_obj.readlines()

    if "***\n" in set_raw:
        nkeys_line = set_raw.index("***\n") + 2

    header = set_raw[0:nkeys_line]
    data = set_raw[nkeys_line:]

    dict_string = "{"

    for line in data:
        key, value = line[:-1].split(delimiter)[:2]

        dict_string += "'" + key + "': "

        if key == "delimiter":
            dict_string += value + ", "
            continue

        if value in ["True", "False"]:  # is boolean
            dict_string += value + ", "
            continue

        if value.find("[") > -1:  # is a list...
            dict_string += value + ", "
            continue

        try:
            float(value)
        except ValueError:  # not boolean or list: must be string
            dict_string += "'" + value + "', "
        else:  # is a number!
            dict_string += value + ", "

    dict_string = dict_string[0:-2] + "}"
    settings = literal_eval(dict_string)

    settings["header"] = header

    # definition which keys are essential
    keys_ess = ["platform", "path_data", "path_output"]

    for key in keys_ess:  # check if essential parameters are given
        if key not in settings:
            print("failed! missing essential key: " + key)
            return None

    keys_opt = [  # definition which keys are optional
        "instrument",
        "exp_name",
        "exp_date_UTC",
        "preselect_t_range",
        "exp_t_range",
        "cuevettes",
        "LED_2_Hg_rat",
        "T_fitparms",
        "O3_thresh",
        "OMC_noise_thresh",
        "OMC_thresh_dI1I2",
        "OMC_thresh_dI",
        "OMC_thresh_follow",
        "OMC_I_use",
        "OMC_use_Zahn_smooth",
        "OMC_noise_filter",
        "OMC_sg",
        "coerce_state",
        "OMC_unc",
        "OSC_eval",
        "OSC_Ijump_filter",
        "OSC_I_maxjump",
        "OSC_I_range",
        "OSC_p_range",
        "OSC_pjump_filter",
        "OSC_dp_range",
        "OSC_avg",
        "OSC_S_range",
        "OSC_S_ipsect",
        "OSC_cut_edges",
        "OSC_smooth_freq",
        "OSC_unc",
        "apply_t_corr",
        "use_t_ref",
        "t_ref_max_delta",
        "t_Offset_DataTransfer",
        "t_Offset_UserDef",
        "t_corr_parms_UserDef",
        "t_corr_PlatSpec",
        "no_I_val",
        "no_O3_val",
        "data_cut_sect",
        "OMCfile_del",
        "OSCfile_del",
        "ts_fmt",
        "delimiter",
        "colhdr_ix",
        "key_dI1",
        "key_dI2",
        "key_dI1I2",
        "key_I_C1",
        "key_I_C2",
        "key_Imed_C1",
        "key_Imed_C2",
        "key_p",
        "key_state",
        "key_time",
        "key_T_C1",
        "key_T_C2",
        "key_Tp_T_Cuev1",
        "key_Tp_T_Cuev2",
        "key_Tp_p_Cuev",
        "key_Tp_p_Inlet",
    ]

    def_set = {  # default values of optional parameters
        "instrument": "undefined",
        "exp_name": "undefined",
        "exp_date_UTC": False,
        "preselect_t_range": False,
        "exp_t_range": False,
        "cuevettes": ["C1", "C2"],
        "LED_2_Hg_rat": 0.95,
        "T_fitparms": [
            86585.1,  # parameters for cuevette length of 37.84 cm
            # and ozone acs of 1.136*10E-17 cm**2 at 20°C
            3.09e-3,
            6.63e-5,
            3.88e-7,
            3.18e-8,
            4.01e-10,
        ],
        "O3_thresh": [3, 5000],
        "OMC_noise_thresh": ["dI1_I2", "dI_C1", "dI_C2"],
        "OMC_thresh_dI1I2": 20000,
        "OMC_thresh_dI": 1000,
        "OMC_thresh_follow": 2,
        "OMC_I_use": "mean",
        "OMC_use_Zahn_smooth": True,
        "OMC_noise_filter": False,
        "OMC_sg": [9, 4],
        "coerce_state": "MS",
        "OMC_unc": 0.02,
        "OSC_eval": True,
        "OSC_Ijump_filter": False,
        "OSC_I_maxjump": 10000.0,
        "OSC_I_range": [500, 5000000],
        "OSC_p_range": [10, 1000],
        "OSC_pjump_filter": False,
        "OSC_dp_range": [2, 20],
        "OSC_avg": 110.0,
        "OSC_S_range": [0.5, 50],
        "OSC_S_ipsect": [0],
        "OSC_cut_edges": False,
        "OSC_smooth_freq": 1.0,
        "OSC_unc": 0.005,
        "apply_t_corr": True,
        "use_t_ref": True,
        "t_ref_max_delta": 6400,
        "t_Offset_DataTransfer": 0,
        "t_Offset_UserDef": False,
        "t_corr_parms_UserDef": False,
        "t_corr_PlatSpec": False,
        "no_I_val": -999999,
        "no_O3_val": -1,
        "data_cut_sect": False,
        "OMCfile_del": [0, 0],
        "OSCfile_del": [0, 0],
        "ts_fmt": "%d.%m.%y %H:%M:%S.%f",
        "delimiter": "\t".encode("UTF-8"),
        "colhdr_ix": 0,
        "key_dI1": "dI_C1",
        "key_dI2": "dI_C2",
        "key_dI1I2": "dI1_I2",
        "key_I_C1": "I_C1",
        "key_I_C2": "I_C2",
        "key_Imed_C1": "I_Med_C1",
        "key_Imed_C2": "I_Med_C2",
        "key_p": "p",
        "key_state": "State",
        "key_time": "Time",
        "key_T_C1": "T_C1",
        "key_T_C2": "T_C2",
        "key_Tp_T_Cuev1": "T_Cuev1",
        "key_Tp_T_Cuev2": "T_Cuev2",
        "key_Tp_p_Cuev": "p_Cuev",
        "key_Tp_p_Inlet": "p_Inlet",
    }

    for key in keys_opt:  # apply default if key not defined in config file
        if key not in settings:
            settings[key] = def_set[key]
            if verbose:
                print(f"added missing optional key: {key}")

    return settings


# ------------------------------------------------------------------------------


def load_settings(fname: [str, Path], verbose=False) -> dict:
    """
    Load pyFairoproc configuration from config file.

    Parameters
    ----------
    fname : str or pathlib.Path
        filepath of config file.
    verbose : bool, optional
        print verbose stuff to console. The default is False.

    Raises
    ------
    ValueError
        if essential keys are missing in the config file.

    Returns
    -------
    settings : dict
        pyFairoproc configuration.
    """
    with open(fname, "r") as f:
        settings = yaml.load(f, Loader=yaml.FullLoader)

    # definition which keys are essential
    keys_ess = ["platform", "path_data", "path_output"]

    for key in keys_ess:  # check if essential parameters are given
        if key not in settings:
            raise ValueError(f"failed! missing essential key: {key}")

    keys_opt = [  # definition which keys are optional
        "instrument",
        "exp_name",
        "exp_date_UTC",
        "preselect_t_range",
        "exp_t_range",
        "cuevettes",
        "coerce_state",
        "LED_2_Hg_rat",
        "T_fitparms",
        "O3_thresh",
        "OMC_noise_thresh",
        "OMC_thresh_dI1I2",
        "OMC_thresh_dI",
        "OMC_thresh_follow",
        "OMC_I_use",
        "OMC_T_use",
        "OMC_use_Zahn_smooth",
        "OMC_noise_filter",
        "OMC_sg",
        "OMC_t_offset",
        "OMC_unc",
        "OMC_MR_zero_offset",
        "OMC_freq",
        "OSC_freq",
        "OSC_eval",
        "OSC_Ijump_filter",
        "OSC_I_maxjump",
        "OSC_I_range",
        "OSC_p_range",
        "OSC_Scorr_maxdiff",
        "OSC_dp_range",
        "OSC_avg",
        "OSC_S_range",
        "OSC_S_ipsect",
        "OSC_cut_edges",
        "OSC_smooth_freq",
        "OSC_unc",
        "apply_t_corr",
        "use_t_ref",
        "t_delta_range",
        "t_Offset_UserDef",
        "t_corr_parms_UserDef",
        "t_corr_PlatSpec",
        "no_I_val",
        "no_O3_val",
        "OMCoutput_cut",
        "OSCoutput_cut",
        "ts_fmt",
        "delimiter",
        "colhdr_ix",
        "key_dI1",
        "key_dI2",
        "key_dI1I2",
        "key_I_C1",
        "key_I_C2",
        "key_Imed_C1",
        "key_Imed_C2",
        "key_p",
        "key_state",
        "key_time",
        "key_T_C1",
        "key_T_C2",
        "key_Tp_T_Cuev1_in",
        "key_Tp_T_Cuev2_in",
        "key_Tp_p_Cuev",
        "key_Tp_p_Inlet",
        "key_Tp_p_preOSC",
        "key_Tp_dp_OSC",
        "t_ref_max_delta",  # deprecated since v2021.06.26a
    ]

    def_set = {  # default values of optional parameters
        "instrument": "undefined",
        "exp_name": "undefined",
        "exp_date_UTC": False,
        "preselect_t_range": False,
        "exp_t_range": False,
        "coerce_state": "MS",
        "cuevettes": ["C1", "C2"],
        "LED_2_Hg_rat": 0.95,
        # parameters for cuevette length of 37.84 cm
        # and ozone acs of 1.136e-17 cm**2 at 20°C :
        "T_fitparms": [
            86585.1,
            3.09e-3,
            6.63e-5,
            3.88e-7,
            3.18e-8,
            4.01e-10,
        ],
        "O3_thresh": [2, 5000],
        "OMC_noise_thresh": ["dI1_I2", "dI_C1", "dI_C2"],
        "OMC_thresh_dI1I2": 20000,
        "OMC_thresh_dI": 1000,
        "OMC_thresh_follow": 2,
        "OMC_I_use": "mean",
        "OMC_T_use": "outside",
        "OMC_use_Zahn_smooth": True,
        "OMC_noise_filter": False,
        "OMC_sg": [9, 4],
        "OMC_unc": 0.02,
        "OMC_MR_zero_offset": 0.0,
        "OMC_t_offset": 0.08,
        "OMC_freq": -1,
        "OSC_eval": True,
        "OSC_Ijump_filter": False,
        "OSC_I_maxjump": 10000.0,
        "OSC_I_range": [10000, 10000000],
        "OSC_p_range": [10, 1000],
        "OSC_dp_range": [2, 20],
        "OSC_avg": 110.0,
        "OSC_S_range": [0.5, 50],
        "OSC_S_ipsect": [0],
        "OSC_Scorr_maxdiff": 0.01,
        "OSC_cut_edges": False,
        "OSC_smooth_freq": False,
        "OSC_unc": 0.005,
        "OSC_freq": -1,
        "apply_t_corr": True,
        "use_t_ref": True,
        "t_ref_max_delta": 6400,  # deprecated since v2021.06.26a
        "t_delta_range": [-3600, 3600],
        "t_Offset_UserDef": False,
        "t_corr_parms_UserDef": False,
        "t_corr_PlatSpec": False,
        "no_I_val": -999999,
        "no_O3_val": -1,
        "OMCoutput_cut": False,
        "OSCoutput_cut": False,
        "ts_fmt": "%d.%m.%y %H:%M:%S.%f",
        "delimiter": "\t".encode("UTF-8"),
        "colhdr_ix": 0,
        "key_dI1": "dI_C1",
        "key_dI2": "dI_C2",
        "key_dI1I2": "dI1_I2",
        "key_I_C1": "I_C1",
        "key_I_C2": "I_C2",
        "key_Imed_C1": "I_Med_C1",
        "key_Imed_C2": "I_Med_C2",
        "key_p": "p",
        "key_state": "State",
        "key_time": "Time",
        "key_T_C1": "T_C1",  # sensor on the outside of the cuevette
        "key_T_C2": "T_C2",  # sensor on the outside of the cuevette
        "key_Tp_T_Cuev1_in": "T_C1_In",  # old: T_Cuev1
        "key_Tp_T_Cuev2_in": "T_C2_In",  # old: T_Cuev2
        "key_Tp_p_Cuev": "p_Cuev",
        "key_Tp_p_Inlet": "p_Inlet",
        "key_Tp_p_preOSC": "p_preOSC",
        "key_Tp_dp_OSC": "dp_Oscar",
    }

    for key in keys_opt:  # apply default if key not defined in config file
        if key not in settings:
            settings[key] = def_set[key]
            if verbose:
                print(f"added missing optional key: {key}")

    # check if path_data or path_output are defined as relative paths;
    # relative to that of the config file is assumed.
    a = Path(fname).parent
    for i, p in enumerate(settings["path_data"]):
        if not Path(p).is_absolute():
            settings["path_data"][i] = str((a / Path(p)).resolve())
    if not Path(settings["path_output"]).is_absolute():
        settings["path_output"] = str((a / Path(settings["path_output"])).resolve())

    return settings
