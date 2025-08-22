# -*- coding: utf-8 -*-
""" BinningMixin functionality can now be found in data.BinnedData
This is its graveyard. 

@Author: Sophie Bauchinger, IAU
@Date: Thu Jan 30 11:05:00 2025

"""
import numpy as np
import pandas as pd

import toolpac.calc.binprocessor as bp  # type: ignore

from dataTools import tools


# %% Mixin for adding binning methods to GlobalData objects
class BinningMixin:
    """ Holds methods for binning global data in 1D/2D/3D in selected coordinates. 
    
    Methods:
        bin_1d(subs, coord, bci_1d, xbsize, df)

        bin_2d(subs, xcoord, ycoord, bci_2d, xbsize, ybsize)
        
        bin_3d(subs, zcoord, bci_3d, xbsize, ybsize, zbsize, eql)
        
        bin_LMS(subs, tp, df, bci_3d, zbsize, nr_of_bins)
        
        bin_1d_seasonal(subs, coord, bci_1d, xbsize, df)
        
        bin_2d_seasonal(subs, xcoord, ycoord, bci_2d, xbsize, ybsize, df)
            Bin substance data onto a 2D-grid of the given coordinates for each season. 
        
    """

    def make_bci(self, xcoord, ycoord=None, zcoord=None, **kwargs): 
        """ Create n-dimensional binclassinstance using standard coordinate limits / bin sizes. 
        
        Args: 
            *coord (dcts.Coordinate)
            
            key *bsize (float): Size of the bin
            key *bmin, *bmax (float): Outer bounds for bins. Optional 
            
        Returns Bin_equi*d binning structure for all given dimensions.  
        """
        dims = sum([dim is not None for dim in [xcoord, ycoord, zcoord]])

        if dims not in [1,2,3]:
            raise ValueError('Something went wrong when evaluating dimension numbers. ') 

        xbsize = kwargs.get('xbsize', xcoord.get_bsize())
        def_xbmin, def_xbmax = self.get_var_lims(xcoord, bsize = xbsize, **kwargs)
        xbmin = kwargs.get('xbmin', def_xbmin)
        xbmax = kwargs.get('xbmax', def_xbmax)
        
        if dims == 1: 
            return bp.Bin_equi1d(xbmin, xbmax, xbsize)

        ybsize = kwargs.get('ybsize', ycoord.get_bsize())
        def_ybmin, def_ybmax = self.get_var_lims(ycoord, bsize = ybsize, **kwargs)
        ybmin = kwargs.get('ybmin', def_ybmin)
        ybmax = kwargs.get('ybmax', def_ybmax)
        
        if dims == 2: 
            return bp.Bin_equi2d(xbmin, xbmax, xbsize, 
                                 ybmin, ybmax, ybsize)

        zbsize = kwargs.get('zbsize', zcoord.get_bsize())
        def_zbmin, def_zbmax = self.get_var_lims(zcoord, bsize = zbsize, **kwargs)
        zbmin = kwargs.get('zbmin', def_zbmin)
        zbmax = kwargs.get('zbmax', def_zbmax)
        
        return bp.Bin_equi3d(xbmin, xbmax, xbsize,
                             ybmin, ybmax, ybsize,
                             zbmin, zbmax, zbsize)

    def extract_attr(self, data_dict, bin_attr):
        """ Returns data_dict with bin_attr dimension removed. """
        if isinstance(list(data_dict.values())[0], dict): # seasonal
            return {k: {s: getattr(v[s], bin_attr) for s in v.keys()
                                  } for k,v in data_dict.items()}
        return {k:getattr(v, bin_attr) for k,v in data_dict.items()}

    # --- 1D binning -- 
    def bin_1d(self, var, xcoord, **kwargs) -> bp.Simple_bin_1d:
        """ Bin substance data in self.df onto 1D-bins of the given coordinate. 
        
        Args: 
            var (dcts.Substance, dcts.Coordinate)
            
            xcoord (dcts.Substance, dcts.Coordinate) - 1st bin dimension 
            ycoord (dcts.Substance, dcts.Coordinate) - 2nd bin dimension 
            
            key bci_3d (bp.Bin_equi3d, bp.Bin_notequi3d): 3D-Binning structure
            key xbsize (float): 1st dim bin size
            key ybsize (float): 2nd dim bin size
        
        Returns bp.Simple_bin_1d object
        """
        df = kwargs.get('df', self.df)
        x = self.get_var_data(xcoord, df=df)
        bci_1d = kwargs.get('bci_1d', self.make_bci(xcoord, **kwargs))

        out = bp.Simple_bin_1d(np.array(df[var.col_name]), x,
                               bci_1d, count_limit=self.count_limit)

        return out

    def bin_1d_seasonal(self, var, xcoord, **kwargs) -> dict[bp.Simple_bin_1d]:
        """ Bin substance data onto the given coordinate for each season. 
        Args: 
            subs (dcts.Substance)
            coord (dcts.Coordinate)
            
            key bci_1d (bp.Bin_equi1d, bp.Bin_notequi1d): 1D-binning structure
            key xbsize (float)
        
        Returns dictionary of bp.Simple_bin_1d objects for each season. 
        """

        df = kwargs.get('df', self.df)
        if 'season' not in df.columns:
            df['season'] = tools.make_season(df.index.month)

        bci_1d = self.make_bci(xcoord, **kwargs)

        out_dict = {}
        for s in set(self.df['season']): 
            out_dict[s] = self.sel_season(s).bin_1d(var, xcoord, 
                                                    bci_1d = bci_1d, 
                                                    **kwargs)
        return out_dict

    # --- 2D binning -- 
    def bin_2d(self, var, xcoord, ycoord, **kwargs) -> bp.Simple_bin_2d:
        """ Bin substance data in self.df onto an x-y grid spanned by the given coordinates. 

        Args: 
            var (dcts.Substance, dcts.Coordinate)
            
            xcoord (dcts.Substance, dcts.Coordinate) - 1st bin dimension 
            ycoord (dcts.Substance, dcts.Coordinate) - 2nd bin dimension 
            
            key bci_2d (bp.Bin_equi2d, bp.Bin_notequi2d): 2D-Binning structure
            key xbsize (float): 1st dim bin size
            key ybsize (float): 2nd dim bin size
            key lognorm_fit (bool): Toggle fitting a lognorm distr. to the binned data. Default True
        
        Returns bp.Simple_bin_2d object or tools.Bin2DFitted object
        """
        x = self.get_var_data(xcoord)
        y = self.get_var_data(ycoord)
        bci_2d = kwargs.get('bci_2d', self.make_bci(xcoord, ycoord, **kwargs))

        if kwargs.get('lognorm_fit', True): 
            out = tools.Bin2DFitted(np.array(self.df[var.col_name]),
                                    x, y, bci_2d,
                                    count_limit=self.count_limit,
                                    **kwargs)
        else: 
            out = bp.Simple_bin_2d(np.array(self.df[var.col_name]), 
                                   x, y, bci_2d, 
                                   count_limit=self.count_limit)

        return out

    def get_data_2d_dicts(self, var, xcoord, ycoord, bin_attr=None, **kwargs) -> tuple[dict, dict]:
        """ Get strato/tropo dictionary with binned 2D variables bin_attr data using given coords for all tps. 
        
        Parameters: 
            var (dcts.Substance|dcts.Coordinate): Data to be binned
            x/ycoord (dcts.Coordinate): x- and y-coordinates for 2D binning
            
            bin_attr (str): e.g. vmean|vstdv|rvstd. Optional
                if None, the dictionary contains data on all available variables
            
            key x/ybsize: Bin size of x/y-coordinate
            
        Returns two dictionaries of the form {tp : DATA}.
                
        """
        strato_BinDict, tropo_BinDict = {}, {}

        for tp in kwargs.get('tps', self.tps):
            strato_BinDict[tp.col_name] = self.sel_strato(tp).bin_2d(
                var, xcoord, ycoord, **kwargs)
            tropo_BinDict[tp.col_name] = self.sel_tropo(tp).bin_2d(
                var, xcoord, ycoord, **kwargs)

        if bin_attr is None: 
            return strato_BinDict, tropo_BinDict

        strato_attr_dict = self.extract_attr(strato_BinDict, bin_attr)
        tropo_attr_dict = self.extract_attr(tropo_BinDict, bin_attr)
        return strato_attr_dict, tropo_attr_dict

    def bin_2d_seasonal(self, var, xcoord, ycoord, **kwargs) -> dict[bp.Simple_bin_2d]: 
        """ Seasonal binning of var along xyz coordinates. """
        if 'season' not in self.df.columns:
            self.df['season'] = tools.make_season(self.df.index.month)
            
        bci_2d = self.make_bci(xcoord, ycoord, **kwargs)

        out_dict = {}
        for s in set(self.df['season']): 
            out_dict[s] = self.sel_season(s).bin_2d(var, xcoord, ycoord, 
                                                    bci_2d = bci_2d, 
                                                    **kwargs)
        return out_dict

    def get_seasonal_2d_dict(self, var, xcoord, ycoord, bin_attr=None, **kwargs) -> tuple[dict, dict]: 
        """ Get seasonal 2D binned data for strato/tropo for all TPs. 
        
        Parameters: 
            var (dcts.Substance|dcts.Coordinate): Data to be binned
            x/ycoord (dcts.Coordinate): x- and y-coordinates for 2D binning

            bin_attr (str): e.g. vmean|vstdv|rvstd. Optional
                if None, the dictionary contains data on all available variables
            
            key x/ybsize: Bin size of x/y-coordinate
        
        Returns two dictionaries with bin_attr values in the form of {tp: {s : DATA}}.
        """
        strato_Bin2Dseas_dict, tropo_Bin2Dseas_dict = {}, {}
        
        for tp in self.tps:
            strato_Bin2Dseas_dict[tp.col_name] = self.sel_strato(tp).bin_2d_seasonal(
                var, xcoord, ycoord, **kwargs)
            tropo_Bin2Dseas_dict[tp.col_name] = self.sel_tropo(tp).bin_2d_seasonal(
                var, xcoord, ycoord, **kwargs)
        
        if bin_attr is None: 
            return strato_Bin2Dseas_dict, tropo_Bin2Dseas_dict
        
        strato_2D_attr_seas = {k: {s: getattr(v[s], bin_attr) for s in v.keys()
                                   } for k,v in strato_Bin2Dseas_dict.items()}
        tropo_2D_attr_seas = {k: {s: getattr(v[s], bin_attr) for s in v.keys()
                                  } for k,v in tropo_Bin2Dseas_dict.items()}
        
        return strato_2D_attr_seas, tropo_2D_attr_seas

    # --- 3D binning -- 
    def bin_3d(self, var, xcoord, ycoord, zcoord, **kwargs) -> bp.Simple_bin_3d:
        """ Bin variable data onto a 3D-grid given by z-coordinate / (equivalent) latitude / longitude. 

        Args: 
            var (dcts.Substance, dcts.Coordinate)
            
            xcoord (dcts.Substance, dcts.Coordinate) - 1st bin dimension 
            ycoord (dcts.Substance, dcts.Coordinate) - 2nd bin dimension 
            zcoord (dcts.Substance, dcts.Coordinate) - 3rd bin dimension 
            
            key bci_3d (bp.Bin_equi3d, bp.Bin_notequi3d): 3D-Binning structure
            key xbsize (float): 1st dim bin size
            key ybsize (float): 2nd dim bin size
            key zbsize (float): 3rd dim bin size
            key lognorm_fit (bool): Toggle fitting a lognorm distr. to the binned data. Default True
        
        Returns a Simple_bin_2d object or tools.Bin3DFitted object including LogNorm distribution fits.
        """

        x = self.get_var_data(xcoord)
        y = self.get_var_data(ycoord)
        z = self.get_var_data(zcoord)
        bci_3d = kwargs.get('bci_3d', self.make_bci(xcoord, ycoord, zcoord, **kwargs))
        
        if kwargs.get('lognorm_fit', True): 
            out = tools.Bin3DFitted(np.array(self.df[var.col_name]),
                                    x, y, z, bci_3d,
                                    count_limit=self.count_limit,
                                    **kwargs)
        else: 
            out = bp.Simple_bin_3d(np.array(self.df[var.col_name]), 
                                   x, y, z, bci_3d, 
                                   count_limit=self.count_limit)
        return out

    def add_to_Bin3D_df(self, *binning_params, **kwargs): 
        """ Create dataframe containing all precalculated Bin3DFitted instances. 
        
        Parameters: 
            arg binning_params (list[dcts.Substance, dcts.Coordinate, bool])
        """
        if not hasattr(self, 'Bin3D_df'): 
            self.Bin3D_df = pd.DataFrame(columns = [
                'subs', 'zcoord', 'eql', 'strato_Bin3D_dict', 'tropo_Bin3D_dict'])

        for params in binning_params:
            subs = params.get('subs')
            zcoord = params.get('zcoord')
            eql = params.get('eql')

            df = self.Bin3D_df
            if len(df[(df.subs == str(subs)) & (df.zcoord == str(zcoord)) & (df.eql == str(eql))]) == 1: 
                continue

            strato_data, tropo_data = self.make_Bin3D(
                params.get('subs'), 
                params.get('zcoord'), 
                eql = params.get('eql'))

            df_data = dict(subs = str(params.get('subs')), 
                zcoord = str(params.get('zcoord')), 
                eql = str(params.get('eql')), 
                strato_Bin3D_dict = strato_data, 
                tropo_Bin3D_dict = tropo_data)

            self.Bin3D_df.loc[len(self.Bin3D_df)] = df_data

        return self.Bin3D_df

    def make_Bin3D(self, var, zcoord, eql, **kwargs):
        """ Create Bin3DFitted instances for tropospheric data sorted with each tps. 
        
        Parameters: 
            var (dcts.Substance)
            zcoord (dcts.Coordinate): Vertical coordinate used for binning
            eql (bool): Use equivalent latitude instead of latitude
            
            key bci_3d (bp.Bin_equi3d, bp.Bin_notequi3d): 3D-Binning structure
            key *(xbsize, ybsize, zbsize) (float): Binsize for x/y/z dimensions. Optional

        Returns stratospheric, tropospheric 3D-bin dictionaries keyed by tropopause definition.
        """
        strato_Bin3D_dict, tropo_Bin3D_dict = {}, {}
        
        if eql: 
            ycoord = self.get_coords(hcoord='lat', model = 'MSMT')[0]
        else:
            ycoord = self.get_coords(hcoord='lat', model = 'MSMT')[0]
        xcoord = self.get_coords(hcoord='lon', model = 'MSMT')[0]
        
        for tp in self.tps:
            strato_plotter = self.sel_strato(tp)
            strato_Bin3D_dict[tp.col_name] = strato_plotter.bin_3d(
                var, xcoord, ycoord, zcoord, **kwargs)

            tropo_plotter = self.sel_tropo(tp)
            tropo_Bin3D_dict[tp.col_name] = tropo_plotter.bin_3d(
                var, xcoord, ycoord, zcoord, **kwargs)
        
        return strato_Bin3D_dict, tropo_Bin3D_dict 

    def get_Bin3D_dict(self, var, zcoord, eql, **kwargs) -> tuple[dict[tools.Bin3DFitted]]:
        """ Returns tuple of Bin3DFitted instances for the given parameters. """

        if not hasattr(self, 'Bin3D_df'):
            self.add_to_Bin3D_df(dict(subs=var, zcoord=zcoord, eql=eql, **kwargs))
            
        df = self.Bin3D_dfs
        data = df.loc[(df.subs == str(var)) 
                      & (df.zcoord == str(zcoord)) 
                      & (df.eql==str(eql))]
        if len(data) == 0: 
            df = self.add_to_Bin3D_df(dict(subs=var, zcoord=zcoord, eql=eql))
            data = df.loc[(df.var == str(var)) 
                          & (df.zcoord == str(zcoord)) 
                          & (df.eql==str(eql))]
        
        strato_Bin3D_dict = data['strato_Bin3D_dict'].values[0]
        tropo_Bin3D_dict = data['tropo_Bin3D_dict'].values[0]

        return strato_Bin3D_dict, tropo_Bin3D_dict
 
    def get_data_3d_dicts(self, var, zcoord, eql, bin_attr=None, **kwargs) -> tuple[dict, dict]: 
        """ Extract specific attributes from Bin3D dictionaries. 
        
        Parameters:
            var (dcts.Substance|dcts.Coordinate): Data to be binned
            x/y/z coord (dcts.Coordinate): x/y/z-coordinates for 2D binning

            bin_attr (str): e.g. vmean|vstdv|rvstd. Optional
                if None, the dictionary contains data on all available variables
        
        Returns data dictionaries such that {tp_col : np.ndarray} 
        """
        strato_Bin3D_dict, tropo_Bin3D_dict = self.get_Bin3D_dict(var, zcoord, eql, **kwargs)
        if bin_attr is None: 
            return strato_Bin3D_dict, tropo_Bin3D_dict

        strato_attr_dict = {k:getattr(v, bin_attr) for k,v in strato_Bin3D_dict.items()}
        tropo_attr_dict = {k:getattr(v, bin_attr) for k,v in tropo_Bin3D_dict.items()}
        
        if bin_attr == 'rvstd': 
            # Multiply everything by 100 to get spercentages
            strato_attr_dict = {k:v*100 for k,v in strato_attr_dict.items()}
            tropo_attr_dict = {k:v*100 for k,v in tropo_attr_dict.items()}
        
        return strato_attr_dict, tropo_attr_dict

    def bin_3d_seasonal(self, var, xcoord, ycoord, zcoord, **kwargs) -> dict[bp.Simple_bin_3d]: 
        """ Seasonal binning of var along xyz coordinates. """
        if 'season' not in self.df.columns:
            self.df['season'] = tools.make_season(self.df.index.month)
            
        bci_3d = self.make_bci(xcoord, ycoord, zcoord, **kwargs)
        
        out_dict = {}
        for s in set(self.df['season']): 
            out_dict[s] = self.sel_season(s).bin_3d(var, xcoord, ycoord, zcoord, 
                                                    bci_3d = bci_3d, 
                                                    **kwargs)
        return out_dict

    def get_seasonal_3d_dict(self, var, xcoord, ycoord, zcoord, bin_attr=None, **kwargs) -> tuple[dict, dict]: 
        """ Get seasonal 3D binned data for strato/tropo for all TPs. 
        
        Parameters: 
            var (dcts.Substance|dcts.Coordinate): Data to be binned
            x/ycoord (dcts.Coordinate): x- and y-coordinates for 2D binning

            bin_attr (str): e.g. vmean|vstdv|rvstd. Optional
                if None, the dictionary contains data on all available variables
            
            key x/ybsize: Bin size of x/y-coordinate
        
        Returns two dictionaries with bin_attr values in the form of {tp: {s : DATA}}.
        """
        strato_Bin3Dseas_dict, tropo_Bin3Dseas_dict = {}, {}
        
        for tp in self.tps:
            strato_Bin3Dseas_dict[tp.col_name] = self.sel_strato(tp).bin_3d_seasonal(
                var, xcoord, ycoord, zcoord, **kwargs)
            tropo_Bin3Dseas_dict[tp.col_name] = self.sel_tropo(tp).bin_3d_seasonal(
                var, xcoord, ycoord, zcoord, **kwargs)
        
        if bin_attr is None: 
            return strato_Bin3Dseas_dict, tropo_Bin3Dseas_dict
        
        strato_3D_attr_seas = {k: {s: getattr(v[s], bin_attr) for s in v.keys()
                                   } for k,v in strato_Bin3Dseas_dict.items()}
        tropo_3D_attr_seas = {k: {s: getattr(v[s], bin_attr) for s in v.keys()
                                  } for k,v in tropo_Bin3Dseas_dict.items()}
        
        return strato_3D_attr_seas, tropo_3D_attr_seas

    def bin_LMS(self, var, tp, df=None, nr_of_bins=3, **kwargs) -> bp.Simple_bin_3d:
        """ Bin data onto lon-lat-tp grid, then return only the lowermost stratospheric bins. 
        
        Args: 
            var (dcts.Substance|dcts.Coordinate): Variable data to bin
            tp (dcts.Coordinate): Tropopause Definition used to select LMS data

            df (pd.DataFrame): Stratospheric dataset (filtered using TP). Optional
            nr_of_bins (int): Max nr. of bins over the tropopause that should be returned

            key bci_3d(bp.Bin_equi3d, bp.Bin_notequi3d): Binned data
            key zbsize (float): Size of vertical bins

        Returns bp.Simple_bin_3d object
        """

        if not tp.rel_to_tp:
            raise Exception('tp has to be relative to tropopause')

        xbsize = ybsize = self.grid_size
        zbsize = kwargs.get('zbsize', tp.get_bsize())

        if not isinstance(df, pd.DataFrame):
            df = self.sel_strato(**tp.__dict__).df

        x = df.geometry.x
        y = df.geometry.y
        xbmin, xbmax = -180, 180
        ybmin, ybmax = -90, 90

        z = df[tp.col_name]

        # nr_of_bins = min(out.nz, nr_of_bins)
        zbmax = ((np.nanmax(z) // zbsize) + 1) * zbsize
        zbmax = min(zbsize * nr_of_bins, zbmax)
        zbmin = (np.nanmin(z) // zbsize) * zbsize

        bci_3d = kwargs.get('bci_3d', bp.Bin_equi3d(xbmin, xbmax, xbsize,
                                                    ybmin, ybmax, ybsize,
                                                    zbmin, zbmax, zbsize))
        out = bp.Simple_bin_3d(np.array(df[var.col_name]),
                               x, y, z, bci_3d,
                               count_limit=self.count_limit)
        return out

