# -*- coding: utf-8 -*-
""" Mixin for adding binning methods to GlobalData objects 

@Author: Sophie Bauchinger, IAU
@Date: Tue Jun 11 16:45:00 2024

class BinningMixin
"""

import pandas as pd
import numpy as np
import matplotlib.patheffects as mpe

import toolpac.calc.binprocessor as bp # type: ignore

from dataTools import tools

class BinningMixin: 
    """ Holds methods for binning global data in 1D/2D/3D in selected coordinates. 
    
    
    """
    
    def binned_1d(self, subs, **kwargs) -> tuple[list, list]:
        """
        Returns 1D binned objects for each year as lists (lat / lon)
        Parameters:
            subs (Substance): dcts.Substance instance
        """
        return tools.bin_1d(self, subs, **kwargs)  # out_x_list, out_y_list

    def binned_2d(self, subs, **kwargs) -> list:
        """
        Returns 2D binned object for each year as a list
        Parameters:
            subs (Substance): dcts.Substance instance
        """
        return tools.bin_2d(self, subs, **kwargs)  # out_list

    @property
    def outline(self): 
        """ Helper function to add outline to lines in plots. """
        return mpe.withStroke(linewidth=2, foreground='white')

    def bin_1d(self, subs, coord, bci_1d=None, xbsize=None, df=None) -> bp.Simple_bin_1d: 
        """ Bin substance data in self.df onto 1D-bins of the given coordinate. 
        
        Args: 
            subs (dcts.Substance)
            coord (dcts.Coordinate)
            
            bci_1d (bp.Bin_equi1d, bp.Bin_notequi1d): 1D-Binning structure
            xbsize (float)
        
        Returns bp.Simple_bin_1d object
        """
        df = self.df if (df is None) else df
        xbsize = coord.get_bsize() if not xbsize else xbsize

        if coord.col_name == 'geometry.y': # latitude
            x = df.geometry.y
        elif coord.col_name == 'geometry.x':
            x = df.geometry.x
        else:
            x = np.array(df[coord.col_name])

        # get bins as multiples of the bin size
        xbmax = ((np.nanmax(x) // xbsize) + 1) * xbsize
        xbmin = (np.nanmin(x) // xbsize) * xbsize
        
        if not isinstance(bci_1d, (bp.Bin_equi1d, bp.Bin_notequi1d)): 
            bci_1d = bp.Bin_equi1d(xbmin, xbmax, xbsize)

        out = bp.Simple_bin_1d(np.array(df[subs.col_name]), x,
                               bci_1d, count_limit=self.count_limit)

        return out

    def bin_2d(self, subs, xcoord, ycoord, 
               bci_2d=None, xbsize = None, ybsize= None) -> bp.Simple_bin_2d: 
        """ Bin substance data in self.df onto an x-y grid spanned by the given coordinates. 
        
        Args: 
            subs (dcts.Substance)
            xcoord (dcts.Coordinate)
            ycoord (dcts.Coordinate)
            
            bci_2d (bp.Bin_equi2d, bp.Bin_notequi2d): 2D-Binning structure
            xbsize (float)
            ybsize (float)
        
        Returns bp.Simple_bin_2d object
        """
        
        xbsize = xcoord.get_bsize() if not xbsize else xbsize
        ybsize = ycoord.get_bsize() if not ybsize else ybsize
        
        if xcoord.col_name.startswith('geometry.'):
            x = self.df.geometry.y if xcoord.col_name == 'geometry.y' else self.df.geometry.x
        else:
            x = np.array(self.df[xcoord.col_name])

        if ycoord.col_name.startswith('geometry.'):
            y = self.df.geometry.y if ycoord.col_name == 'geometry.y' else self.df.geometry.x
        else: 
            y = np.array(self.df[ycoord.col_name])

        # get bins as multiples of the bin size
        xbmax = ((np.nanmax(x) // xbsize) + 1) * xbsize
        xbmin = (np.nanmin(x) // xbsize) * xbsize

        ybmax = ((np.nanmax(y) // ybsize) + 1) * ybsize
        ybmin = (np.nanmin(y) // ybsize) * ybsize

        if not isinstance(bci_2d, (bp.Bin_equi2d, bp.Bin_notequi2d)):
            bci_2d = bp.Bin_equi2d(xbmin, xbmax, xbsize,
                                   ybmin, ybmax, ybsize)

        out = bp.Simple_bin_2d(np.array(self.df[subs.col_name]), x, y,
                               bci_2d, count_limit=self.count_limit)
        return out

    def bin_3d(self, subs, zcoord, bci_3d=None, 
               xbsize = None, ybsize = None, zbsize = None,
               eql=False) -> bp.Simple_bin_3d: 
        """ Bin substance data onto a 3D-grid given by z-coordinate / (equivalent) latitude / longitude. 
        
        Args: 
            subs (dcts.Substance)
            zcoord (dcts.Coordinate) - vertical coordinate
            
            bci_3d (bp.Bin_equi3d, bp.Bin_notequi3d): 3D-Binning structure
            xbsize (float)
            ybsize (float)
            zbsize (float)
            
            eql (bool): Use equivalalent latitude instead of latitude as y-coordinate 
        
        Returns a bp.Simple_bin_3d object. 
        """
        xbsize = self.grid_size if not xbsize else xbsize 
        ybsize = self.grid_size if not ybsize else ybsize 
        zbsize = zcoord.get_bsize() if not zbsize else zbsize
        
        x = self.df.geometry.x # longitude
        y = self.df.geometry.y # latitude
        
        if eql:
            [ycoord] = self.get_coords(hcoord='eql', model='ERA5')
            y = self.df[ycoord.col_name]
        
        xbmin, xbmax = -180, 180
        ybmin, ybmax = -90, 90

        z = self.df[zcoord.col_name]
        zbmax = ((np.nanmax(z) // zbsize) + 1) * zbsize
        zbmin = (np.nanmin(z) // zbsize) * zbsize
        
        if not isinstance(bci_3d, (bp.Bin_equi3d, bp.Bin_notequi3d)):
            bci_3d = bp.Bin_equi3d(xbmin, xbmax, xbsize,
                                   ybmin, ybmax, ybsize,
                                   zbmin, zbmax, zbsize)
        
        out = bp.Simple_bin_3d(np.array(self.df[subs.col_name]), 
                               x, y, z, bci_3d)
        
        return out

    def bin_LMS(self, subs, tp, df=None, bci_3d=None, zbsize=None, nr_of_bins = 3) -> bp.Simple_bin_3d: 
        """ Bin data onto lon-lat-tp grid, then return only the lowermost stratospheric bins. 
        
        Args: 
            subs (dcts.Substance): Substance data to bin
            tp (dcts.Coordinate): Tropopause Definition used to select LMS data
            
        Optional:
            df (pd.DataFrame): Stratospheric dataset (filtered using TP)
            bci_3d(bp.Bin_equi3d, bp.Bin_notequi3d): Binned data
            zbsize (float): Size of vertical bins
            nr_of_bins (int): Max nr. of bins over the tropopause that sshould be returned 

        Returns bp.Simple_bin_3d object
        """
        
        if not tp.rel_to_tp: 
            raise Exception('tp has to be relative to tropopause')
        
        xbsize = ybsize = self.grid_size
        zbsize = tp.get_bsize() if not zbsize else zbsize

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

        if not isinstance(bci_3d, (bp.Bin_equi3d, bp.Bin_notequi3d)):
            bci_3d = bp.Bin_equi3d(xbmin, xbmax, xbsize,
                                   ybmin, ybmax, ybsize,
                                   zbmin, zbmax, zbsize)

        out = bp.Simple_bin_3d(np.array(df[subs.col_name]), 
                               x, y, z, bci_3d)
        return out

    def bin_1d_seasonal(self, subs, coord, bci_1d=None, xbsize=None, df=None) -> dict[bp.Simple_bin_1d]:
        """ Bin substance data onto the given coordinate for each season. 
        Args: 
            subs (dcts.Substance)
            coord (dcts.Coordinate)
            
            bci_1d (bp.Bin_equi1d, bp.Bin_notequi1d): 1D-binning structure
            xbsize (float)
        
        Returns dictionary of bp.Simple_bin_1d objects for each season. 
        """

        df = self.df if (df is None) else df
        
        if not 'season' in df.columns: 
            df['season'] = tools.make_season(df.index.month)
        
        out_dict = {}
        xbsize = coord.get_bsize() if not xbsize else xbsize

        for s in set(df['season'].tolist()):
            df_s = df[df['season'] == s]

            if coord.col_name == 'geometry.y': # latitude
                x = df_s.geometry.y
            elif coord.col_name == 'geometry.x':
                x = df_s.geometry.x
            else:
                x = np.array(df_s[coord.col_name])
            
            # skip seasons that have no data
            if all(str(xi) == 'nan' for xi in x): continue
            
            # get bins as multiples of the bin size
            xbmax = ((np.nanmax(x) // xbsize) + 1) * xbsize
            xbmin = (np.nanmin(x) // xbsize) * xbsize

            if not isinstance(bci_1d, (bp.Bin_equi1d, bp.Bin_notequi1d)):
                bci_1d = bp.Bin_equi1d(xbmin, xbmax, xbsize)

            out = bp.Simple_bin_1d(np.array(df_s[subs.col_name]), x,
                                   bci_1d, count_limit=self.count_limit)
            out_dict[s] = out

        return out_dict

    def bin_2d_seasonal(self, subs, xcoord, ycoord,
                        bci_2d=None, xbsize=None, ybsize=None, df=None,
                        **kwargs) -> dict[bp.Simple_bin_1d]:
        """ Bin substance data onto a 2D-grid of the given coordinates for each season. 
        Args: 
            subs (dcts.Substance)
            xcoord (dcts.Coordinate)
            ycoord (dcts.Coordinate)
            
            bci_2d (bp.Bin_equi2d, bp.Bin_notequi2d): 1D-binning structure
            xbsize (float)
            ybsize (float)
        
        Returns dictionary of bp.Simple_bin_2d objects for each season. 
        """
        
        df = self.df if (df is None) else df
        
        if not 'season' in df.columns: 
            df['season'] = tools.make_season(df.index.month)
        
        out_dict = {}
        xbsize = xcoord.get_bsize() if not xbsize else xbsize
        ybsize = ycoord.get_bsize() if not ybsize else ybsize

        # calculate binned output per season
        for s in set(df['season'].tolist()):
            df_s = df[df['season'] == s]

            if xcoord.col_name == 'geometry.y': # latitude
                x = df_s.geometry.y
            else:
                x = np.array(df_s[xcoord.col_name])
            
            # skip seasons that have no data
            if all(str(xi) == 'nan' for xi in x): continue

            y = np.array(df_s[ycoord.col_name])

            # get bins as multiples of the bin size
            xbmax = ((np.nanmax(x) // xbsize) + 1) * xbsize
            xbmin = (np.nanmin(x) // xbsize) * xbsize

            ybmax = ((np.nanmax(y) // ybsize) + 1) * ybsize
            ybmin = (np.nanmin(y) // ybsize) * ybsize

            if not isinstance(bci_2d, (bp.Bin_equi2d, bp.Bin_notequi2d)):
                bci_2d = bp.Bin_equi2d(xbmin, xbmax, xbsize,
                                           ybmin, ybmax, ybsize)

            out = bp.Simple_bin_2d(np.array(df_s[subs.col_name]), x, y,
                                   bci_2d, count_limit=self.count_limit)
            out_dict[s] = out

        return out_dict

