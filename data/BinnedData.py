# -*- coding: utf-8 -*-

""" Class definitions for binned global data objects. Can be extended with plotting functionality. 

@Author: Sophie Bauchinger, IAU
@Date: Fri Dec 20 15:56:12 2024

SimpleBin objects have the following main attributes: 
    .vmean
    .*intm (x/y/z)
"""
import pandas as pd
import numpy as np

import toolpac.calc.binprocessor as bp

from dataTools import tools


# Getting DATA and LIMS
def get_var_data(df, var) -> np.array:
    """ Returns variable data including from geometry columns. 
    Args:
        df (pd.DataFrame): Input data to extract variable data from
        var (dcts.Coordinate, dcts.Substance)
    """
    if var.col_name == 'geometry.y':
        data = df.geometry.y
    elif var.col_name == 'geometry.x':
        data = df.geometry.x
    else:
        data = np.array(df[var.col_name])
    return data


def get_var_lims(var, bsize=None, gdf=None, **kwargs) -> tuple[float]:
    """ Returns outer limits based on variable data and (optional) bin size. 

    Parameters: 
        var (dcts.Coordinate, dcts.Substance)
        bsize (float): Bin size. Optional. 
        gdf (pd.DataFrame): Limits will be calculated from data in this dataframe. 
            Needs to be supplied if var.get_lims() is not possible or if 'databased' is specified. 
        
        key databased (bool): Toggle calculating limits from available data. 
            Default True for everything but Lon/Lat. 
    """
    if not kwargs.get('databased', False):
        try:
            return var.get_lims()
        except ValueError:
            pass
    if gdf is not None:
        v_data = get_var_data(gdf, var)
        vmin = np.nanmin(v_data)
        vmax = np.nanmax(v_data)

        if bsize is None:
            return vmin, vmax

        vbmin = (vmin // bsize) * bsize
        vbmax = ((vmax // bsize) + 1) * bsize
        return vbmin, vbmax
    else:
        raise ValueError('Could not generate variable limits.')


# BINCLASSINSTANCE
def make_bci(xcoord, ycoord=None, zcoord=None, **kwargs) -> bp.Bin:
    """ Create n-dimensional binning structure using standard coordinate limits / bin sizes. 
    
    Args: 
        *coord (dcts.Coordinate)
        
        key *bsize (float): Size of the bin
        key *bmin, *bmax (float): Outer bounds for bins. Optional 
        
    Returns equi-distant binning structure for all given dimensions.  
    """
    if isinstance(kwargs.get('bci'), (bp.Bin1D, bp.Bin2D, bp.Bin3D)):
        # Useful in cases where bci should stay the same across upstream function calls
        return kwargs.get('bci')

    dims = sum([dim is not None for dim in [xcoord, ycoord, zcoord]])

    if dims not in [1, 2, 3]:
        raise ValueError('Something went wrong when evaluating dimension numbers. ')

    xbsize = kwargs.get('xbsize', xcoord.get_bsize())
    def_xbmin, def_xbmax = get_var_lims(xcoord, bsize=xbsize, **kwargs)
    xbmin = kwargs.get('xbmin', def_xbmin)
    xbmax = kwargs.get('xbmax', def_xbmax)

    if dims == 1:
        return bp.Bin1D(
            xbmin, xbmax, xbsize)

    ybsize = kwargs.get('ybsize', ycoord.get_bsize())
    def_ybmin, def_ybmax = get_var_lims(ycoord, bsize=ybsize, **kwargs)
    ybmin = kwargs.get('ybmin', def_ybmin)
    ybmax = kwargs.get('ybmax', def_ybmax)

    if dims == 2:
        return bp.Bin2D(
            xbmin, xbmax, xbsize,
            ybmin, ybmax, ybsize)

    zbsize = kwargs.get('zbsize', zcoord.get_bsize())
    def_zbmin, def_zbmax = get_var_lims(zcoord, bsize=zbsize, **kwargs)
    zbmin = kwargs.get('zbmin', def_zbmin)
    zbmax = kwargs.get('zbmax', def_zbmax)

    return bp.Bin3D(
        xbmin, xbmax, xbsize,
        ybmin, ybmax, ybsize,
        zbmin, zbmax, zbsize)

# BINNING
def binning(df, var, xcoord, ycoord=None, zcoord=None, count_limit=5, **kwargs):
    """ From values in df, bin the given variable onto the available coordinates. 
    Parameters: 
        df (pd.DataFrame): Hold the variable and coordinate data. 
        var, x/y/zcoord (dcts.Substance|dcts.Coordinate)
        count_limit (int): Bins with fewer data points are excluded from the output. 

        key bci (bp.Bin**D): Binclassinstance
        key *bsize (float): if bci is not specified, controls the size of the *d-bins.

    Returns a single BinnedData*d object. 
    """
    dims = sum([dim is not None for dim in [xcoord, ycoord, zcoord]])
    v, x, y, z = [(get_var_data(df, i) if i is not None else None) for i in [var, xcoord, ycoord, zcoord]]

    if dims == 1:
        bci_1d = make_bci(xcoord, **kwargs)
        out = bp.Binned1D(v, x, bci_1d, count_limit=count_limit)

    elif dims == 2:
        bci_2d = make_bci(xcoord, ycoord, **kwargs)
        out = bp.Binned2D(v, x, y, bci_2d, count_limit=count_limit)

    elif dims == 3:
        bci_3d = make_bci(xcoord, ycoord, zcoord, **kwargs)
        out = bp.Binned3D(v, x, y, z, bci_3d, count_limit=count_limit)

    else:
        raise Exception(f'Invalid dimensions. Found dims = {dims}')

    return out

def seasonal_binning(df, var, xcoord, ycoord=None, zcoord=None,
                     count_limit=5, **kwargs):
    """ Bin the given dataframe for all available seasons. 

    Parameters: 
        df (pd.DataFrame): Hold the variable and coordinate data. 
        var, x/y/zcoord (dcts.Substance|dcts.Coordinate)
        count_limit (int): Bins with fewer data points are excluded from the output. 

        key bci (bp.Bin**d): Binning structure
        key *bsize (float): if bci is not specified, controls the size of the *d-bins. 

    Returns a dictionary of {season : BinnedData*d object}. 
    """
    if 'season' not in df.columns:
        df['season'] = tools.make_season(df.index.month)

    # Want the same binclassinstance across all seasons
    bci = make_bci(xcoord, ycoord, zcoord, **kwargs)

    seasonal_dict = {}
    for s in set(df['season'].values):
        df_season = df[df['season'] == s]
        bin_s_out = binning(df_season, var, xcoord, ycoord, zcoord,
                            count_limit=count_limit, bci=bci, **kwargs)
        seasonal_dict[s] = bin_s_out

    return seasonal_dict

def monthly_binning(df, var, xcoord, ycoord=None, zcoord=None, **kwargs): 
    """ Separate into months and bin data on the given grid. 
    
    Parameters:    
        df (pd.DataFrame): Hold the variable and coordinate data. 
        var, x/y/zcoord (dcts.Substance|dcts.Coordinate)
        count_limit (int): Bins with fewer data points are excluded from the output. 

        key bci (bp.Bin*d): Binning structure
        key *bsize (float): if bci is not specified, controls the size of the *d-bins. 

    Returns a dictionary of {month : BinnedData*d object}. 
    """
    # raise NotImplementedError("Tough cookie.")
    if not isinstance(df.index, pd.DatetimeIndex):
        # TODO: if not DateTime-indexed df, need to check for 'month' column
        raise NotImplementedError("Non Datetime-Index not yet supported.")

    bci = make_bci(xcoord, ycoord, zcoord, **kwargs)
    monthly_dict = {}
    monthly_dfs = {m: df[df.index.month == m] for m in np.arange(1,13)}
    for month in np.arange(1,13): 
        m_df = monthly_dfs[month]
        bin_m_out = binning(m_df, var, xcoord, ycoord, zcoord, 
                            bci=bci, **kwargs)
        monthly_dict[month] = bin_m_out
    return monthly_dict

def weighted_binning(data_list, xcoord, ycoord=None, zcoord=None, **kwargs):
    """ Return weighted binned data where each item in data_list contributes equally. """
    dims = sum([dim is not None for dim in [xcoord, ycoord, zcoord]])
    if len(data_list[0]) != dims: 
        raise Exception("Dimensions of data_list are not equal to given parameters. ")

    if dims == 1:
        bci_1d = make_bci(xcoord, **kwargs)
        return bp.WeightedBinning1D(data_list, bci_1d, **kwargs)

    elif dims == 2:
        bci_2d = make_bci(xcoord, ycoord, **kwargs)
        out = bp.WeightedBinning2D(data_list, bci_2d, **kwargs)

    elif dims == 3:
        raise NotImplementedError("Feel free to extend the WeightedBinning class to 3D. ")

    else:
        raise Exception(f'Invalid dimensions. Found dims = {dims}')

    return out

def monthly_weighted_binning(GlobalObj, var, xcoord, **kwargs): 
    """ Create monthly binned profiles of subs (give data for a latitude band)
    
    Parameters: 
        GlobalObject (dataTools.data.GlobalData)
        subs (dcts.Substance)
        vcoord (dcts.Coordinate): Vertical coord. for profile
        lat_bsize (float): Size of latitude bands (even)
    """    
    bci = make_bci(xcoord)
    
    binned_monthly = {}
    for month in set(GlobalObj.df.index.month):
        data_month = GlobalObj.sel_month(month)
        data_list = []
        for flight_id in data_month.flights:
            data_ascent = data_month.sel_flight(flight_id)
            
            v = data_ascent.get_var_data(var)
            x = data_ascent.get_var_data(xcoord)
            data_list.append((v,x))

        binned_m = bp.WeightedBinning1D(data_list, bci, **kwargs)
        # TODO: Add statistical values: Q1, Q3 and IQR available
        binned_monthly[month] = binned_m.weighted_mean, binned_m.weighted_std

    return binned_monthly, bci
    

#%% Stratosphere / Troposphere binning
def get_ST_binDict(GlobalObj, strato_params, tropo_params, **kwargs):
    """ Create binned data dicts for strato / tropo with specified parameters.
    Parameters: 
        tropo/strato_params (dict): Required var, xcoord, (y/zcoord Optional).
            Optional keys: xbsize, ybsize, ...

        key bin_attr (str): Bin attribute to base statistics on
        key tps (list[dcts.Coordinate]): Tropopause definitions. Defaults to GlobalObj.tps
    """
    for params in [tropo_params, strato_params]:
        if not all(i in params for i in ['var', 'xcoord']):
            raise Exception('Need to supply at least `var` and `xcoord` as parameters.')

    strato_Bin_dict, tropo_Bin_dict = {}, {}

    for tp in kwargs.get('tps', GlobalObj.tps):
        strato_df = GlobalObj.sel_strato(tp).df
        strato_Bin_dict[tp.col_name] = binning(
            strato_df, lognorm = kwargs.get('lognorm', False), 
            **strato_params)

        tropo_df = GlobalObj.sel_tropo(tp).df
        tropo_Bin_dict[tp.col_name] = binning(
            tropo_df, 
            lognorm = kwargs.get('lognorm', False), 
            **tropo_params)

    if 'bin_attr' not in kwargs:
        return strato_Bin_dict, tropo_Bin_dict

    strato_attr = extract_attr(
        strato_Bin_dict, kwargs.get('bin_attr'))
    tropo_attr = extract_attr(
        tropo_Bin_dict, kwargs.get('bin_attr'))

    return strato_attr, tropo_attr


def get_ST_seass_binDict(GlobalObj, strato_params, tropo_params, **kwargs):
    """ Create seasonal binned data dicts for strato / tropo with specified parameters.
    Parameters:
        GlobalObj (GlobalData)
        tropo/strato_params (dict): Required var, xcoord, (y/zcoord Optional).
            Optional keys: xbsize, ybsize, ...

        key bin_attr (str): Bin attribute to base statistics on
        key tps (list[dcts.Coordinate]): Tropopause definitions. Defaults to GlobalObj.tps
    """

    for params in [tropo_params, strato_params]:
        if not all(i in params for i in ['var', 'xcoord']):
            raise Exception('Need to supply at least `var` and `xcoord` as parameters.')

    strato_Binseas_dict, tropo_Binseas_dict = {}, {}

    for tp in kwargs.get('tps', GlobalObj.tps)[::-1]:
        strato_df = GlobalObj.sel_strato(tp).df
        strato_Binseas_dict[tp.col_name] = seasonal_binning(
            strato_df, **strato_params,
            lognorm = kwargs.get('lognorm', False))

        tropo_df = GlobalObj.sel_tropo(tp).df
        tropo_Binseas_dict[tp.col_name] = seasonal_binning(
            tropo_df, **tropo_params,
            lognorm = kwargs.get('lognorm', False))

    if 'bin_attr' not in kwargs:
        return strato_Binseas_dict, tropo_Binseas_dict

    strato_3D_attr_seas = {k: {s: getattr(v[s], kwargs.get('bin_attr')) for s in v.keys()
                               } for k, v in strato_Binseas_dict.items()}
    tropo_3D_attr_seas = {k: {s: getattr(v[s], kwargs.get('bin_attr')) for s in v.keys()
                              } for k, v in tropo_Binseas_dict.items()}

    return strato_3D_attr_seas, tropo_3D_attr_seas


def make_ST_bins(GlobalObj, tropo_params, strato_params, **kwargs):
    """ Create tropo_BinDict and strato_BinDict. 
    Data is first sorted into troposphere and stratosphere, then binned according to 
    the given params. 
    
    Parameters:
        GlobalObj (GlobalData instance)
        tropo/strato_params (dict['var', 'xcoord', 'ycoord']):
            Specify which parameters to use as variable and coordinates for binning.
            Additional parameters: bci, 
        
        key tps (list[dcts.Coordinate]): Tropopause definitions for sorting. Default GlobalObj.tps
    """
    tropo_BinDict, strato_BinDict = {}, {}

    for tp in kwargs.get('tps', GlobalObj.tps):
        tropo_df = GlobalObj.sel_tropo(tp).df
        tropo_BinDict[tp.col_name] = binning(
            tropo_df, **tropo_params)
        strato_df = GlobalObj.sel_strato(tp).df
        strato_BinDict[tp.col_name] = binning(
            strato_df, **strato_params)

    return tropo_BinDict, strato_BinDict


# %% Now dealing with the OUTPUT
def extract_attr(data_dict, bin_attr):
    """ Returns data_dict with bin_attr dimension removed. """
    if isinstance(list(data_dict.values())[0], dict):  # seasonal / monthly
        return {k: {s: getattr(v[s], bin_attr) for s in v.keys()
                    } for k, v in data_dict.items()}
    # Simple bins
    return {k: getattr(v, bin_attr) for k, v in data_dict.items()}


# %%
# --- Lognorm fitted histograms and things --- # 
def get_lognorm_stats_df(data_dict: dict, lognorm_attr: str, prec: int = 1, use_percentage=False) -> pd.DataFrame:
    """ Create combined lognorm-fit statistics dataframe for all tps when given binned data. 
    Relative standard deviations are multiplied by 100 to return percentages instead of fractions. 
    
    Parameters: 
        data_dict (dict[tools.Bin*DFitted]): Binned data incl. lognorm fits (all variables)
        lognorm_attr (str): vmean_fit / vsdtv_fit / rvstd_fit
        use_percentage (bool): Toggle stats being expressed as percentages.
    """

    if all(isinstance(v, np.ndarray) for v in data_dict.values()):
        return data_dict

    if 'rvstd' not in lognorm_attr and not use_percentage:
        if isinstance(list(data_dict.values())[0], dict):  # seasonal
            raise NotImplementedError('Cannot yet do this for seasonal data_dicts.')
        return pd.DataFrame({k: getattr(v, lognorm_attr).stats(prec=prec) for k, v in data_dict.items()})

    # else: need to multiply by 100 to get percentage
    # (and initially return values with increased precision)
    stats_df = pd.DataFrame({k: getattr(v, lognorm_attr).stats(prec=prec + 2) for k, v in data_dict.items()})

    df = stats_df.T.convert_dtypes()
    non_float_cols = [c for c in df.columns if c not in df.select_dtypes([int, float]).columns]
    for c in df.select_dtypes(float).columns:
        df[c] = df[c].apply(lambda x: round(100 * x, prec + 2))
    for c in non_float_cols:
        df[c] = df[c].apply(lambda x: tuple([round(100 * i, prec + 2) for i in x]))
    return df.T
