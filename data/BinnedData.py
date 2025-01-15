# -*- coding: utf-8 -*-

""" Class definitions for binned global data objects. Can be extended with plotting functionality. 

@Author: Sophie Bauchinger, IAU
@Date: Fri Dec 20 15:56:12 2024

"""
import pandas as pd
import numpy as np

import toolpac.calc.binprocessor as bp 

import dataTools.dictionaries as dcts
from dataTools import tools

#!!! self.get_var_lims()
#!!! self.get_var_data()

# help for getting DATA and LIMS
def get_var_data(df, var) -> np.array: 
    """ Returns variable data including from geometry columns. 
    Args: 
        var (dcts.Coordinate, dcts.Substance)
        key df (pd.DataFrame): Data from this dataframe will be returned. Optional. 
    """
    if var.col_name == 'geometry.y': 
        data = df.geometry.y
    elif var.col_name == 'geometry.x': 
        data = df.geometry.x
    else: 
        data = np.array(df[var.col_name])
    return data

def get_var_lims(var, bsize=None, gdf = None, **kwargs) -> tuple[float]: 
    """ Returns outer limits based on variable data and (optional) bin size. 

    Parameters: 
        var (dcts.Coordinate, dcts.Substance)
        bsize (float): Bin size. Optional. 
        gdf (pd.DataFrame): Limits will be calculated from data in this dataframe. 
            Needs to be supplied if var.get_lims() is not possible or if 'databased' is specified. 
        
        key databased (bool): Toggle calculating limits from available data. 
            Default True for everything but Lon/Lat. 
    """
    if isinstance(var, dcts.Coordinate) and not kwargs.get('databased', True): 
        try: 
            return var.get_lims()
        except ValueError: 
            pass
    
    v_data = get_var_data(gdf, var)
    vmin = np.nanmin(v_data)
    vmax = np.nanmax(v_data)
    
    if bsize is None: 
        return vmin, vmax

    vbmin = (vmin // bsize) * bsize
    vbmax = ((vmax // bsize) + 1) * bsize
    return vbmin, vbmax

# BINCLASSINSTANCE
def make_bci(xcoord, ycoord=None, zcoord=None, **kwargs): 
    """ Create n-dimensional binclassinstance using standard coordinate limits / bin sizes. 
    
    Args: 
        *coord (dcts.Coordinate)
        
        key *bsize (float): Size of the bin
        key *bmin, *bmax (float): Outer bounds for bins. Optional 
        
    Returns Bin_equi*d binning structure for all given dimensions.  
    """
    if isinstance(kwargs.get('bci'), (bp.Bin_equi1d,bp.Bin_equi2d,bp.Bin_equi3d,
                                      bp.Bin_notequi1d, bp.Bin_notequi2d, bp.Bin_notequi3d)):
        # Useful in cases where bci should stay the same across upstream function calls
        return kwargs.get('bci')
    
    dims = sum([dim is not None for dim in [xcoord, ycoord, zcoord]])

    if dims not in [1,2,3]:
        raise ValueError('Something went wrong when evaluating dimension numbers. ') 

    xbsize = kwargs.get('xbsize', xcoord.get_bsize())
    def_xbmin, def_xbmax = get_var_lims(xcoord, bsize = xbsize, **kwargs)
    xbmin = kwargs.get('xbmin', def_xbmin)
    xbmax = kwargs.get('xbmax', def_xbmax)
    
    if dims == 1: 
        return bp.Bin_equi1d(xbmin, xbmax, xbsize)

    ybsize = kwargs.get('ybsize', ycoord.get_bsize())
    def_ybmin, def_ybmax = get_var_lims(ycoord, bsize = ybsize, **kwargs)
    ybmin = kwargs.get('ybmin', def_ybmin)
    ybmax = kwargs.get('ybmax', def_ybmax)
    
    if dims == 2: 
        return bp.Bin_equi2d(xbmin, xbmax, xbsize, 
                                ybmin, ybmax, ybsize)

    zbsize = kwargs.get('zbsize', zcoord.get_bsize())
    def_zbmin, def_zbmax = get_var_lims(zcoord, bsize = zbsize, **kwargs)
    zbmin = kwargs.get('zbmin', def_zbmin)
    zbmax = kwargs.get('zbmax', def_zbmax)
    
    return bp.Bin_equi3d(xbmin, xbmax, xbsize,
                            ybmin, ybmax, ybsize,
                            zbmin, zbmax, zbsize)

# actually now BINNING
def foo_binning(df, var, xcoord, ycoord=None, zcoord=None, count_limit=None, **kwargs):
    """ From values in df, bin the given variable onto the available coordinates. 
    Parameters: 
        df (pd.DataFrame): Hold the variable and coordinate data. 
        var, x/y/zcoord (dcts.Substance|dcts.Coordinate)
        count_limit (int): Bins with fewer data points are excluded from the output. 

        key bci (bp.Bin_**d): Binclassinstance
        key *bsize (float): if bci is not specified, controls the size of the *d-bins. 

    Returns a single Simple_bin_*d object. 
    """
    dims = sum([dim is not None for dim in [xcoord, ycoord, zcoord]])
    v, x, y, z = [(get_var_data(df, i) if i is not None else None) for i in [var, xcoord, ycoord, zcoord]]
    
    if dims == 1:
        bci_1d = make_bci(xcoord, **kwargs)
        out = bp.Simple_bin_1d(v, x, bci_1d, count_limit=count_limit)
    
    elif dims == 2: 
        bci_2d = make_bci(xcoord, ycoord, **kwargs)
        out = bp.Simple_bin_2d(v, x, y, bci_2d, count_limit=count_limit)

    elif dims == 3: 
        bci_3d = make_bci(xcoord, ycoord, zcoord, **kwargs)
        out = bp.Simple_bin_3d(v, x, y, z, bci_3d, count_limit=count_limit)

    return out 

def seasonal_binning(df, var, xcoord, ycoord=None, zcoord=None, 
                     count_limit=None, **kwargs): 
    """ Do foo_binning for all available seasons. """
    if 'season' not in df.columns:
        df['season'] = tools.make_season(df.index.month)
    
    # Want the same binclassinstance across all seasons
    bci = make_bci(xcoord, ycoord, zcoord, **kwargs)
    
    seasonal_dict = {}
    for s in set(df['season'].values): 
        bin_s_out = foo_binning(df, var, xcoord, ycoord, zcoord, 
                                count_limit=count_limit, bci = bci, **kwargs)
        seasonal_dict[s] = bin_s_out

    return seasonal_dict

def stratospheric_binning(df, tps):
    pass
    #TODO: This is where TropopauseMixin needs to be decoupled from GlobalData !!!


#%% Now dealing with the OUTPUT 
def extract_attr(self, data_dict, bin_attr):
    """ Returns data_dict with bin_attr dimension removed. """
    if isinstance(list(data_dict.values())[0], dict): # seasonal
        return {k: {s: getattr(v[s], bin_attr) for s in v.keys()
                                } for k,v in data_dict.items()}
    return {k:getattr(v, bin_attr) for k,v in data_dict.items()}
