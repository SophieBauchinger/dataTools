# -*- coding: utf-8 -*-
"""
@Author: Sophie Bauchinger, IAU
@Date: Tue Jun  6 13:59:31 2023

Showing mixing ratios per season on a plot of coordinate relative to the
tropopause (in km or K) versus equivalent latitude (in deg N)

BinPlotter1D: 
    - Histograms of nr of datapoints over distance to tropopause bins
    - Vertical profiles: 
        seasonal or not
        variability or mean mixing ratios
    - Bar plots for Troposphere | Stratosphere: vmean, vstdv, vcount
    - Matrices of variability per latitude bin

"""

import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.patheffects as mpe
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec
import pandas as pd
import itertools
from mpl_toolkits.axes_grid1 import AxesGrid
import geopandas

import toolpac.calc.binprocessor as bp

import dataTools.dictionaries as dcts
from dataTools import tools
from dataTools.data import GlobalData

import warnings
warnings.filterwarnings("ignore", message="Boolean Series key will be reindexed to match DataFrame index. result = super().__getitem__(key)")

# outline = mpe.withStroke(linewidth=2, foreground='white')

#TODO map of distance to tropopause (stratosphere only?)
#TODO might want to implement a logarithmic scale for pressure at some point

class SimpleBinPlotter(): 
    def plot_1d(self, simple_bin_1d, bin_attr='vmean'): 
        """ scatter plot of binned data. """
        data = getattr(simple_bin_1d, bin_attr)

        fig, ax = plt.subplots(dpi=150, figsize=(6,7))
        ax.plot(simple_bin_1d.xintm, data, label=bin_attr)
        ax.legend()
        plt.show()

    def plot_2d(self, simple_bin_2d, bin_attr='vmean'): 
        """ Imshow 2D plot of binned data. """
        data = getattr(simple_bin_2d, bin_attr)
        vlims = np.nanmin(data), np.nanmax(data)
        norm = Normalize(*vlims)

        fig, ax = plt.subplots(dpi=150, figsize=(8,9))
        ax.set_title(bin_attr)
        cmap = dcts.dict_colors()[bin_attr]
        img = ax.imshow(data.T, cmap = cmap, norm=norm,
                        aspect='auto', origin='lower',
                        extent=[simple_bin_2d.xbmin, simple_bin_2d.xbmax, 
                                simple_bin_2d.ybmin, simple_bin_2d.ybmax])
        fig.subplots_adjust(right=0.9)
        fig.tight_layout(pad=2.5)
        fig.colorbar(img, ax = ax, aspect=30, pad=0.09, orientation='horizontal')
        plt.show()


class BinPlotter():
    """ Plotting class to facilitate creating binned 2D plots for any choice of x and y.

    Attributes:
        data (pd.DataFrame): Input data

    Methods:
        get_vlimit(subs, bin_attr)
        get_coord_lims(coord, xyz)
        _get_bsize(coord, xyz)
        filter_non_shared_indices(tps)
        bin_1d(subs, coord)
        bin_2d(subs, xcoord, ycoord)
        bin_3d(subs, zcoord)
        bin_1d_seasonal(subs, coord, bin_equi1d, xbsize)
        bin_2d_seasonal(subs, xcoord, ycoord, bin_equi2d, xbsize, ybsize)
    """
    def __init__(self, glob_obj: GlobalData, filter_tps = None, **kwargs):
        """ Initialise class instances. 
        Paramters: 
            glob_obj (GlobalData)
            detr (bool): Use data with linear trend wrt MLO05 removed
            
            key xbsize / ybsize (float)
            key ybsize (float)
            key vlims / xlims / ylims (Tuple[float])
            """
        self.glob_obj = glob_obj
        
        # if not kwargs.get('all_latitudes'): 
        #     self.glob_obj = self.glob_obj.sel_latitude(30, 90)

        self.data = {'df' : glob_obj.df} # dataframe

        # filter_tps = kwargs.pop('filter_tps') if 'filter_tps' in kwargs else False
        if filter_tps: 
            # tps = tools.minimise_tps(dcts.get_coordinates(tp_def='not_nan'))
            self.filter_non_shared_indices(filter_tps)
            glob_obj.data['df'] = self.df.copy()

        self.count_limit = glob_obj.count_limit

        self.data['df']['season'] = tools.make_season(self.data['df'].index.month)
        self.outline = mpe.withStroke(linewidth=2, foreground='white')

        self.kwargs = kwargs

    def __repr__(self):
        return f'<class eqlat.BinPlotter> with minimum points per bin: {self.glob_obj.count_limit} \n\
based on {self.glob_obj.__repr__()}'

    @property
    def df(self) -> pd.DataFrame:
        return self.data['df']

    def get_vlimit(self, subs: dcts.Substance, 
                   bin_attr: str) -> tuple: 
        """ Get colormap limits for given substance and bin attribute. """
        if 'vlims' in self.kwargs:
            vlims = self.kwargs.get('vlims')
        else:
            try:
                vlims = subs.vlims(bin_attr=bin_attr)
            except KeyError:
                if bin_attr=='vmean':
                    vlims = (np.nanmin(self.df[subs.col_name]), np.nanmax(self.df[subs.col_name]))
                else:
                    raise KeyError('Could not generate colormap limits.')
            except: 
                raise KeyError('Could not generate colormap limits.')
        return vlims

    def get_coord_lims(self, coord, xyz: str = None) -> tuple: 
        """ Get coordinate limits for plotting. """
        if xyz=='x': 
            if 'xlims' in self.kwargs: 
                lims = self.kwargs.get('xlims')
            else: 
                lims = (-90, 90)
        elif xyz=='y': 
            if 'ylims' in self.kwargs: 
                lims = self.kwargs.get('ylims')
            else: 
                lims = (np.floor(np.nanmin(self.df[coord.col_name])),
                         np.ceil(np.nanmax(self.df[coord.col_name])))
        else: 
            lims = (np.floor(np.nanmin(self.df[coord.col_name])),
                     np.ceil(np.nanmax(self.df[coord.col_name])))
        return lims

    def _get_bsize(self, coord, xyz: str = None) -> float: 
        """ Get bin size for given coordinate. """
        if xyz=='x' and 'xbsize' in self.kwargs: 
            bsize = self.kwargs.get('xbsize')
        elif xyz=='y' and 'ybsize' in self.kwargs: 
            bsize = self.kwargs.get('ybsize')
        elif xyz=='z' and 'zbsize' in self.kwargs: 
            bsize = self.kwargs.get('zbsize')

        else: 
            bsize = coord.get_bsize()# dcts.get_default_bsize(coord.hcoord)
            if not bsize and xyz=='x': 
                bsize = 10
            if not bsize:    
                lims = self.get_coord_lims(coord, xyz)
                bsize = 5 * ( np.ceil((lims[1]-lims[0])/10) / 5 )
                if (lims[1]-lims[0])/10<1: 
                    bsize=0.5
        return bsize

    def filter_non_shared_indices(self, tps: list):
        """ Filter dataframe for datapoints that don't exist for all tps or are zero. """
        print('Filtering out non-shared indices. ')
        cols = [tp.col_name for tp in tps]
        self.data['df'].dropna(subset=cols, how='any', inplace=True)
        self.data['df'] = self.data['df'][self.data['df'] != 0].dropna(subset=cols)
        return self.data['df']

    def bin_1d(self, subs, coord, bin_equi1d=None, xbsize: float = None): 
        """ Bin substance data onto bins of coord. """
        if not xbsize:
            xbsize = self._get_bsize(coord)

        df = self.df

        if coord.col_name == 'geometry.y': # latitude
            x = df.geometry.y
        elif coord.col_name == 'geometry.x':
            x = df.geometry.x
        else:
            x = np.array(df[coord.col_name])

        # get bins as multiples of the bin size
        xbmax = ((np.nanmax(x) // xbsize) + 1) * xbsize
        xbmin = (np.nanmin(x) // xbsize) * xbsize
        
        if not bin_equi1d:
            bin_equi1d = bp.Bin_equi1d(xbmin, xbmax, xbsize)

        out = bp.Simple_bin_1d(np.array(df[subs.col_name]), x,
                               bin_equi1d, count_limit=self.count_limit)

        return out

    def bin_2d(self, subs, xcoord, ycoord, bin_equi2d=None, 
               xbsize: float = None, ybsize: float = None, 
               df: pd.DataFrame = None): 
        """ Bin substance data onto x-y grid. """
        if not xbsize:
            xbsize = self._get_bsize(xcoord, 'x')
        if not ybsize:
            ybsize = self._get_bsize(ycoord, 'y')

        # calculate binned output per season
        if not isinstance(df, pd.DataFrame): 
            df = self.df

        if xcoord.col_name.startswith('geometry.'):
            x = df.geometry.y if xcoord.col_name == 'geometry.y' else df.geometry.x
        else:
            x = np.array(df[xcoord.col_name])

        if ycoord.col_name.startswith('geometry.'):
            y = df.geometry.y if ycoord.col_name == 'geometry.y' else df.geometry.x
        else: 
            y = np.array(df[ycoord.col_name])

        # get bins as multiples of the bin size
        xbmax = ((np.nanmax(x) // xbsize) + 1) * xbsize
        xbmin = (np.nanmin(x) // xbsize) * xbsize

        ybmax = ((np.nanmax(y) // ybsize) + 1) * ybsize
        ybmin = (np.nanmin(y) // ybsize) * ybsize

        if not bin_equi2d:
            bin_equi2d = bp.Bin_equi2d(xbmin, xbmax, xbsize,
                                       ybmin, ybmax, ybsize)

        out = bp.Simple_bin_2d(np.array(df[subs.col_name]), x, y,
                               bin_equi2d, count_limit=self.count_limit)
        return out

    def bin_3d(self, subs, zcoord, bin_equi3d=None, 
               xbsize: float = None, ybsize: float = None, zbsize: float = None, 
               df: pd.DataFrame = None): 
        """ Bin substance data onto longitude-latitude-z grid. """
        if not xbsize:
            xbsize = self.glob_obj.grid_size
        if not ybsize:
            ybsize = self.glob_obj.grid_size
        if not zbsize: 
            zbsize = self._get_bsize(zcoord)
        
        # calculate binned output per season
        if not isinstance(df, pd.DataFrame): 
            df = self.df
        
        x = df.geometry.x
        y = df.geometry.y
        xbmin, xbmax = -180, 180
        ybmin, ybmax = -90, 90

        z = df[zcoord.col_name]
        zbmax = ((np.nanmax(z) // zbsize) + 1) * zbsize
        zbmin = (np.nanmin(z) // zbsize) * zbsize
        
        if not bin_equi3d:
            bin_equi3d = bp.Bin_equi3d(xbmin, xbmax, xbsize,
                                       ybmin, ybmax, ybsize,
                                       zbmin, zbmax, zbsize)
        
        out = bp.Simple_bin_3d(np.array(df[subs.col_name]), x, y, z, bin_equi3d)
        
        return out

    def bin_LMS(self, subs, tp, df=None): 
        """ Bin data onto lon-lat-z grid without count limit, then take lowest stratospheric bins. """
        if not tp.rel_to_tp: 
            raise Exception('tp has to be relative to tropopause')
        
        xbsize = ybsize = self.glob_obj.grid_size
        zbsize = self._get_bsize(tp)

        # calculate binned output per season
        if not isinstance(df, pd.DataFrame): 
            df = self.glob_obj.sel_strato(**tp.__dict__).df
        

        x = df.geometry.x
        y = df.geometry.y
        xbmin, xbmax = -180, 180
        ybmin, ybmax = -90, 90

        z = df[tp.col_name]
        zbmax = ((np.nanmax(z) // zbsize) + 1) * zbsize
        zbmin = (np.nanmin(z) // zbsize) * zbsize

        bin_equi3d = bp.Bin_equi3d(xbmin, xbmax, xbsize,
                                   ybmin, ybmax, ybsize,
                                   zbmin, zbmax, zbsize)

        out = bp.Simple_bin_3d(np.array(df[subs.col_name]), x, y, z, bin_equi3d)
        
        return out

    def bin_1d_seasonal(self, subs, coord, bin_equi1d=None, xbsize: float = None) -> dict:
        """ Bin the data onto coord for each season. """
        out_dict = {}
        if not xbsize:
            xbsize = self._get_bsize(coord, 'x')

        for s in set(self.df['season'].tolist()):
            df = self.df[self.df['season'] == s]

            if coord.col_name == 'geometry.y': # latitude
                x = df.geometry.y
            elif coord.col_name == 'geometry.x':
                x = df.geometry.x
            else:
                x = np.array(df[coord.col_name])
            
            # skip seasons that have no data
            if all(str(xi) == 'nan' for xi in x): continue
            
            # get bins as multiples of the bin size
            xbmax = ((np.nanmax(x) // xbsize) + 1) * xbsize
            xbmin = (np.nanmin(x) // xbsize) * xbsize

            if not bin_equi1d:
                bin_equi1d = bp.Bin_equi1d(xbmin, xbmax, xbsize)

            out = bp.Simple_bin_1d(np.array(df[subs.col_name]), x,
                                   bin_equi1d, count_limit=self.count_limit)
            out_dict[s] = out

        return out_dict

    def bin_2d_seasonal(self, subs, xcoord, ycoord,
                     bin_equi2d = None,
                     xbsize: float = None, ybsize: float = None) -> dict:
        """ Bin the dataframe per season. """
        out_dict = {}
        if not xbsize:
            xbsize = self._get_bsize(xcoord, 'x')
        if not ybsize:
            ybsize = self._get_bsize(ycoord, 'y')

        # calculate binned output per season
        for s in set(self.df['season'].tolist()):
            df = self.df[self.df['season'] == s]

            if xcoord.col_name == 'geometry.y': # latitude
                x = df.geometry.y
            else:
                x = np.array(df[xcoord.col_name])
            
            # skip seasons that have no data
            if all(str(xi) == 'nan' for xi in x): continue

            y = np.array(df[ycoord.col_name])

            # get bins as multiples of the bin size
            xbmax = ((np.nanmax(x) // xbsize) + 1) * xbsize
            xbmin = (np.nanmin(x) // xbsize) * xbsize

            ybmax = ((np.nanmax(y) // ybsize) + 1) * ybsize
            ybmin = (np.nanmin(y) // ybsize) * ybsize

            if not bin_equi2d:
                bin_equi2d = bp.Bin_equi2d(xbmin, xbmax, xbsize,
                                           ybmin, ybmax, ybsize)

            out = bp.Simple_bin_2d(np.array(df[subs.col_name]), x, y,
                                   bin_equi2d, count_limit=self.count_limit)
            out_dict[s] = out

        return out_dict

    def rms_seasonal_vstdv(self, subs, coord, **kwargs) -> pd.DataFrame:
        """ Root mean squared of seasonal variability for given substance and tp. """
        data_dict = self.bin_1d_seasonal(subs, coord, **kwargs)
        seasons = list(data_dict.keys())

        df = pd.DataFrame(index = data_dict[seasons[0]].xintm)
        df['rms_vstdv'] = np.nan
        df['rms_rvstd'] = np.nan
        
        for s in data_dict: 
            df[f'vstdv_{s}'] = data_dict[s].vstdv
            df[f'rvstd_{s}'] = data_dict[s].rvstd
            df[f'vcount_{s}'] = data_dict[s].vcount

        s_cols_vstd = [c for c in df.columns if c.startswith('vstdv')]
        s_cols_rvstd = [c for c in df.columns if c.startswith('rvstd')]
        n_cols = [c for c in df.columns if c.startswith('vcount')]
        
        # for each bin, calculate the root mean square of the season's standard deviations
        for i in df.index: 
            n = df.loc[i][n_cols].values
            nom = sum(n) - len([i for i in n if i])
            
            s_std = df.loc[i][s_cols_vstd].values
            denom_std = np.nansum([( n[j]-1 ) * s_std[j]**2 for j in range(len(seasons))])
            df['rms_vstdv'].loc[i] = np.sqrt(denom_std / nom) if not nom==0 else np.nan
            
            s_rstd = df.loc[i][s_cols_rvstd].values
            denom_rstd = np.nansum([( n[j]-1 ) * s_rstd[j]**2 for j in range(len(seasons))])
            df['rms_rvstd'].loc[i] = np.sqrt(denom_rstd / nom) if not nom==0 else np.nan
        
        return df

class BinPlotter1D(BinPlotter):
    """ Single dimensional binning & plotting. 
    
    Methods: 
        plot_1d_gradient(subs, coord, bin_attr)
        make_bar_plot(subs, xcoord, tp, bin_attr)
        plot_bar_plots(subs, xcoord, bin_attr)
        matrix_plot_stdv_subs()
        matrix_plot_stdv()
    """
    def __init__(self, glob_obj, **kwargs):
        super().__init__(glob_obj, **kwargs)

    def flight_height_histogram(self, tp, alpha: float = 0.7, ax=None, **kwargs): 
        """ Make a histogram of the number of datapoints for each tp bin. """
        if ax is None: 
            fig, ax = plt.subplots(dpi=250, figsize=(6,4))
            ax.set_ylabel(tp.label())
        
        data = self.glob_obj.df[tp.col_name].dropna()

        ax.set_title(f'Distribution of {self.glob_obj.source} measurements')
        rlims = (-70, 70) if (tp.vcoord=='pt' and tp.rel_to_tp) else (data.min(), 
                                                                      data.max())
        hist = ax.hist(data.values, 
                        bins=30, range=rlims, alpha=alpha, 
                        orientation='horizontal',
                        **kwargs)
        ax.grid(ls='dotted')
        if (tp.rel_to_tp is True) and ax is not None: 
            ax.hlines(0, max(hist[0]), 0, color='k', ls='dashed')
        ax.set_xlabel('# Datapoints')
        
        if tp.crit == 'n2o': 
            ax.invert_yaxis()
            ax.hlines(320.459, max(hist[0]), 0, color='k', ls='dashed', lw=0.5)
        return hist

    def overlapping_histograms(self, tps: list): 
        """ """
        # tps = [tp for tp in tps if tp.vcoord=='pt']
        fig, ax = plt.subplots(dpi=250, figsize=(6,4))
        for tp in tps: 
            hist = self.flight_height_histogram(tp, alpha=0.6, ax=ax,
                                                label=tp.label(True))
            
        ax.hlines(0, max(hist[0]), 0, color='k', ls='dashed')
        if all((tp.rel_to_tp and tp.vcoord=='pt') for tp in tps): 
            ax.set_ylabel('$\Delta\,\Theta$ relative to Tropopause')
        ax.legend(fontsize=7)

    def plot_vertial_profile_variability_comparison(self, subs, tps: list, 
                                                    rel: bool = True, 
                                                    bin_attr: str = 'vstdv', 
                                                    seasons: list[int] = [1,3],
                                                    **kwargs): 
        """ Compare relative mixing ratio varibility between tropopause definitions. """
        fig, ax = plt.subplots(dpi=500, figsize=(4,5))
        outline = mpe.withStroke(linewidth=2, foreground='white')
        
        for i, tp in enumerate(tps):
            bin_dict = self.bin_1d_seasonal(subs, tp)
            
            ls = list(['--', '-.', ':', '-']*5)[i]

            for s in seasons: 
                if s not in bin_dict.keys(): continue
                vdata = getattr(bin_dict[s], bin_attr)
                if rel: vdata = vdata / bin_dict[s].vmean * 100
                y = bin_dict[s].xintm

                ax.plot(vdata, y, ls=ls,
                        c=dcts.dict_season()[f'color_{s}'],
                        label=tp.label(True) if s==1 else None,
                        path_effects=[outline], zorder=2, lw=2.5)
                
                ax.scatter(vdata, y,marker='.', 
                        c=dcts.dict_season()[f'color_{s}'],
                        zorder=3)

                if s==3:
                    yticks = [i for i in y if i<0] + [0] + [i for i in y if i > 0] + [-55, -65]
                    ax.set_yticks(y if not tp.rel_to_tp else yticks)
            
        ax.grid('both')
        ax.set_ylabel(f'$\Delta\,\Theta$ [{tps[0].unit}]')
        
        if bin_attr == 'vstdv': 
            ax.set_xlabel(('Relative variability' if rel else 'Variability')+ f' of {subs.label(name_only=True)} [%]')
        elif bin_attr == 'vmean': 
            ax.set_xlabel(subs.label())
        
        if tps[0].rel_to_tp: 
            xlims = plt.axis()[:2]
            ax.hlines(0, *xlims, ls='dashed', color='k', lw=1, label = 'Tropopause', zorder=1)
            ax.set_xlim(*xlims)

        ax.legend()
        ax.grid('both', ls='dashed', lw=0.5)
        ax.set_axisbelow(True)
        ax.set_ylim(-70, 70)

    def plot_1d_seasonal_gradient(self, subs, coord, 
                                  bin_attr: str = 'vmean', 
                                  add_stdv: bool = False, 
                                  **kwargs):
        """ Plot gradient per season onto one plot. """
        big = kwargs.pop('big') if 'big' in kwargs else False
        bin_dict = self.bin_1d_seasonal(subs, coord, **kwargs)
        
        # fig = plt.figure(dpi=500, figsize=(8,8/3))
        # ax = fig.add_subplot(132)
        
        fig, ax = plt.subplots(dpi=500, 
                               figsize= (6,4) if not big else (3,4))
        outline = mpe.withStroke(linewidth=2, foreground='white')
    
        if coord.vcoord=='pt' and coord.rel_to_tp: 
            ax.set_yticks(np.arange(-60, 75, 20) + [0])

        # ax.set_xlim(5.1,6.2)

        if add_stdv: 
            ax_stdv = ax.twiny()
            ax_stdv.set_xlim(0, (6.2-5.1))

        for s in bin_dict.keys():
            color = dcts.dict_season()[f'color_{s}']
            label = dcts.dict_season()[f'name_{s}']

            vdata = getattr(bin_dict[s], bin_attr)
            y = bin_dict[s].xintm

            if bin_attr=='vmean': 
                if add_stdv: 
                    ax_stdv.plot(bin_dict[s].vstdv, y, 
                            c = color, label = label,
                            linewidth=1, ls='dashed',
                            alpha=0.5,
                            path_effects = [outline], zorder = 2)
                    
                    ax_stdv.tick_params(labelcolor='grey')
                ax.errorbar(vdata, y, 
                            xerr = bin_dict[s].vstdv, 
                            c = color, lw = 1, alpha=0.7, 
                            path_effects=[outline],
                            capsize = 1.5, zorder = 1)
            marker = 'd'
            ax.plot(vdata, y, 
                    marker=marker,
                    c = color, label = label,
                    linewidth=2,# if not kwargs.get('big') else 3,
                    path_effects = [outline], zorder = 2)

            ax.scatter(vdata, y, 
                    marker=marker,
                    c = color, zorder = 3)
            
            # markers should be on top of all other information in the plot 
            # ax.scatter(getattr(bin_dict[s], bin_attr), 
            #             bin_dict[s].xintm, 
            #             marker= marker,# '*', 
            #             lw=1,
            #             c=color, zorder=20)

            # if s==3:
            #     yticks = [i for i in y if i<0] + [0] + [i for i in y if i > 0]
            #     ax.set_yticks(y if not coord.rel_to_tp else yticks)
        
        ax.set_title(coord.label(filter_label=True))
        ax.set_ylabel(coord.label(coord_only = True))

        # cl = f'{coord.vcoord} [{coord.unit}]'
        # if coord.vcoord=='pt': 
        #     cl = f'$\Theta$ [{coord.unit}]'
        # if coord.rel_to_tp: 
        #     cl = '$\Delta$' + cl
        # ax.set_ylabel((f'{cl} - ' if coord.tp_def in ['dyn', 'therm'] else '') + f'{coord.label(True)}')

        if bin_attr=='vmean':
            ax.set_xlabel(subs.label())
        elif bin_attr=='vstdv': 
            ax.set_xlabel('Relative variability of '+subs.label(name_only=True))

        # xmin = np.nanmin([np.nanmin(bin_inst.vmean) for bin_inst in bin_dict.values()])
        # xmax = np.nanmax([np.nanmax(bin_inst.vmean) for bin_inst in bin_dict.values()])
        
        if coord.vcoord in ['mxr', 'p'] and not coord.rel_to_tp: 
            ax.invert_yaxis()
        if coord.vcoord=='p': 
            ax.set_yscale('symlog' if coord.rel_to_tp else 'log')

        if coord.rel_to_tp: 
            xlims = plt.axis()[:2]
            ax.hlines(0, *xlims, ls='dashed', color='k', lw=1, 
                      label = 'Tropopause', zorder=1)
            # if coord.crit=='o3': 
            #     ax.set_ylim(-4, 5.1)

        # if not big: 
        #     ax.legend(loc='lower left')
        ax.grid('both', ls='dashed', lw=0.5)
        ax.set_axisbelow(True)
        
        if coord.rel_to_tp: 
            tools.add_zero_line(ax)
            # zero_lines = np.delete(ax.get_ygridlines(), ax.get_yticks()!=0)
            # for l in zero_lines: 
            #     l.set_color('k')
            #     l.set_linestyle('-.')
        
        return bin_dict, fig 

    def plot_1d_seasonal_gradient_with_vstdv(self, subs, coord): 
        """ Add second axis to vertical gradient plots to indicate variability. """
        bin_dict, fig = self.plot_1d_seasonal_gradient(subs, coord)

        ax1 = fig.add_subplot(133)
        outline = mpe.withStroke(linewidth=2, foreground='white')

        for s in set(self.df['season'].tolist()):
            y = bin_dict[s].xintm

            ax1.plot(bin_dict[s].vstdv, y, '-', 
                    c=dcts.dict_season()[f'color_{s}'],
                    label=dcts.dict_season()[f'name_{s}'],
                    path_effects=[outline], zorder=2)
        ax1.grid('both')
        ax1.tick_params(labelleft=False)

        fig.set_size_inches(8,5)

    def plot_vertical_profile_mean_vstdv(self, subs, tps: list = None, rstd=False, **kwargs): 
        """ Scatter plot of mean seasnoal variability for troopause definitions """
        if tps is None: 
            tps = tools.minimise_tps(dcts.get_coordinates(tp_def='not_nan', rel_to_tp=True))
            tps.sort(key=lambda x: x.col_name, reverse=True)
            tps = [tp for tp in tps if tp.vcoord=='pt']
            tps.append(dcts.get_coord(vcoord='pt', tp_def='nan', model='MSMT'))

        pt_range = 70
        # figsize = [6.4, 4.8] # default
        figsize = [5.5, 4.5]

        if self.glob_obj.ID == 'PGS': 
            pt_range = 140
            figsize = [6.4, 4.8 * 1.5]

        fig, ax = plt.subplots(dpi=250, figsize = figsize)
        ax.set_ylim(-pt_range, pt_range)

        for tp in tps:                              
            df = self.rms_seasonal_vstdv(subs, tp, **kwargs)
            print(df)

            if not rstd: 
                x_data = df['rms_vstdv']
            else: 
                x_data = df['rms_rvstd'] * 100 # %
            
            
            y_data = df.index
            
            if tp.model=='ERA5' and str(tp.tp_def) != 'nan': 
            
                ax.plot(x_data, y_data, 
                        '-', marker='d', 
                        # c=dcts.dict_season()[f'color_{s}'],
                        label=tp.label(True),
                        path_effects=[self.outline], zorder=10)
                
                ax.set_ylabel('$\Delta\Theta_{TP}$ [K]') 
                

            elif (tp.model=='MSMT' or not (tp.rel_to_tp is True)):
                if tp.tp_def == 'chem':
                    color = 'yellow'
                    ax2 = ax.twinx()
                    ax2.set_ylabel(tp.label(),
                                   color = color) 
                    # ax2.set_ylim(330-70, 330+70)
                    ax.plot(x_data, y_data,
                            '-', marker='d', 
                            # c=dcts.dict_season()[f'color_{s}'],
                            c = color,
                            label=tp.label(True),
                            path_effects=[self.outline], zorder=10)
    
                    ax.set_ylabel('$\Theta$-Distance to TP [K]')
                    ax2.tick_params(axis ='y', labelcolor = color, color=color)
                    ax2.spines['right'].set_color(color)

                else: # non-relative coords for reference
                    color = 'xkcd:grey'
                    ax2 = ax.twinx()
                    ax2.set_ylabel(tp.label(),
                                   color = color) 
                    ax2.set_ylim(330-pt_range, 330+pt_range)
                    
                    ax2.plot(x_data, y_data, 
                             '--', marker='d', 
                            c = color, 
                            # label=tp.label(True),
                            path_effects=[self.outline], zorder=0,
                            label='Potential Temperature')
                    ax2.tick_params(axis ='y', labelcolor = color, color=color)
                    ax2.spines['right'].set_color(color)
                
            else: 
                raise KeyError(f'No default plotting for {tp}')
        
        # ax.set_xlim(0, 0.25)
        # ax.hlines(0, 0, 0.25, color='k', ls='dashed', zorder=0)
        
        ax.grid('both', ls='dotted')
        # ax.set_title(f'{self.glob_obj.ID}')
        if not rstd: 
            ax.set_xlabel(f'Mean seasonal variability of {subs.label(name_only=True)} [{subs.unit}]')
        else: 
            ax.set_xlabel(f'Mean seasonal relative variability of {subs.label(name_only=True)} [%]')
        ax.set_zorder(3)
        ax.patch.set_visible(False)
        
        h,l = ax.get_legend_handles_labels()
        
        if 'ax2' in locals():
            ax2.set_zorder(2)
            ax2.patch.set_visible(True)
            h2,l2 = ax2.get_legend_handles_labels()
            
            ax.legend(handles = h+h2, 
                      labels=l+l2,
                      loc='lower right')
        
        else: 
            ax.legend(loc='lower right')

        tools.add_zero_line(ax)

    def stdv_rms_non_pt(self, subs, tp): 
        """ Same as plot_vertical_profile_mean_vstdv but for other vcoords. """
        fig, ax = plt.subplots(dpi=250, figsize=(2.5, 6))

        for tp in [tp]: 
            df = self.rms_seasonal_vstdv(subs, tp)
            ax.plot(df['rms_vstdv'], df.index, '-', marker='d', 
                    # c=dcts.dict_season()[f'color_{s}'],
                    label=tp.label(True),
                    path_effects=[self.outline], zorder=2)

            ax.set_ylabel(tp.label()) 

            ax.grid('both', ls='dotted')
            ax.set_xlabel(f'Mean variability of {subs.label(name_only=True)}')
            ax.legend(loc='lower right')

    def make_bar_plot(self, subs, xcoord, tp, bin_attr: str, 
                      percent_deviation: bool = False, **kwargs) -> tuple: 
        """ Plot histograms showing differences between TPs. 
        bin over xcoord
        overall alpha plot for average value, split up into latitude bands ? 
        """
        # same bins for both tropo and strato
        if not isinstance(kwargs.get('xbsize'), float):
            xbsize = self._get_bsize(xcoord, 'x')
        else: 
            xbsize = kwargs.get('xbsize')

        if xcoord.col_name == 'geometry.y': # latitude
            x = self.df.geometry.y
        elif xcoord.col_name == 'geometry.x':
            x = self.df.geometry.x
        else:
            x = self.df[xcoord.col_name]

        # get bins as multiples of the bin size
        xbmax = ((np.nanmax(x) // xbsize) + 1) * xbsize
        xbmin = (np.nanmin(x) // xbsize) * xbsize

        if not isinstance(kwargs.get('bin_equi1d'), (bp.Bin_equi1d, bp.Bin_notequi1d)): 
            bin_equi1d = bp.Bin_equi1d(xbmin, xbmax, xbsize)
        else: 
            bin_equi1d = kwargs.get('bin_equi1d')

        s_data = self.glob_obj.sel_strato(**tp.__dict__).df
        t_data = self.glob_obj.sel_tropo(**tp.__dict__).df

        if bin_attr=='vcount':
            t_total = len(t_data[t_data[subs.col_name].notna()])
            s_total = len(s_data[s_data[subs.col_name].notna()])
            return t_total, s_total

        t_binned_1d = bp.Simple_bin_1d(np.array(t_data[subs.col_name]), x[x.index.isin(t_data.index)],
                                      bin_equi1d, count_limit=self.count_limit if not bin_attr=='vcount' else 1)

        s_binned_1d = bp.Simple_bin_1d(np.array(s_data[subs.col_name]), x[x.index.isin(s_data.index)],
                                      bin_equi1d, count_limit=self.count_limit if not bin_attr=='vcount' else 1)

        t_values = getattr(t_binned_1d, bin_attr)
        s_values = getattr(s_binned_1d, bin_attr)
        
        t_nans = np.isnan(t_values)
        s_nans = np.isnan(s_values)
        
        t_av = np.average(t_values[~t_nans], weights = t_binned_1d.vcount[~t_nans])
        s_av = np.average(s_values[~s_nans], weights = s_binned_1d.vcount[~s_nans])

        if bin_attr=='vstdv' and percent_deviation: 
            t_av = t_av / np.average(t_binned_1d.vmean[~t_nans], weights = t_binned_1d.vcount[~t_nans]) * 100
            s_av = s_av / np.average(s_binned_1d.vmean[~s_nans], weights = s_binned_1d.vcount[~s_nans]) * 100

        return t_av, s_av

    def plot_bar_plots(self, subs, xcoord, bin_attr, percent_deviation=False, **kwargs):
        """ All tp defs. """
        #TODO vstdv as percentage of the mean
        fig, (ax_t, ax_label, ax_s) = plt.subplots(1, 3, dpi=200, 
                                         figsize=(9,4), sharey=True)
        
        bin_description = f'{self.glob_obj.grid_size}°' + ('latitude' if xcoord.hcoord=='lat' else ('longitude' if xcoord.hcoord=='lon' else 'HUH')) + ' bins'
        
        if bin_attr=='vmean': 
            description = f'Average mixing ratio in {bin_description}'
        elif bin_attr=='vstdv': 
            if not percent_deviation: 
                description = f'Variability in {bin_description}'
            else: 
                description = f'Percent deviation in {bin_description}'
        elif bin_attr == 'vcount': 
            description = 'Total number of datapoints'
        else: 
            description = bin_attr
        
        fig.suptitle(f'{description} of {subs.label()}' if not bin_attr=='vcount' else description)
        fig.subplots_adjust(top=0.85)
        ax_t.set_title('Troposphere', fontsize=9, loc='right')
        ax_s.set_title('Stratosphere', fontsize=9, loc='left')
        
        tropo_bar_vals = []
        strato_bar_vals = []
        bar_labels = []
        
        for i, tp in enumerate(self.glob_obj.tp_coords()):
            t_av, s_av = self.make_bar_plot(subs, xcoord, tp, bin_attr, 
                                            percent_deviation, **kwargs)

            tropo_bar_vals.append(t_av)
            strato_bar_vals.append(s_av)
            
            bar_labels.append(tp.label(True))
            
        bar_params = dict(align='center', edgecolor='k',  rasterized=True,
                          alpha=0.65, zorder=10)
        
        nums = range(len(bar_labels))

        t_bars = ax_t.barh(nums, tropo_bar_vals, **bar_params)
        s_bars = ax_s.barh(nums, strato_bar_vals, **bar_params)
        for i in nums: 
            ax_label.text(0.5, i, bar_labels[i], 
                          horizontalalignment='center', verticalalignment='center')
        ax_label.axis('off')

        maximum = np.nanmax(tropo_bar_vals+strato_bar_vals)
        minimum = np.nanmin(tropo_bar_vals+strato_bar_vals) if not percent_deviation else 0
        for decimal_place in [4,3,2,1,0]:
            if all(i>np.round(minimum, decimal_place) for i in tropo_bar_vals+strato_bar_vals): 
                minimum = np.round(minimum, decimal_place)
            else: 
                break
        padding = (maximum-minimum)/3 * (2 if percent_deviation else 1)

        ax_t.set_xlim(maximum +padding , minimum-padding if not minimum-padding<0 else 0)
        ax_s.set_xlim(minimum-padding if not minimum-padding<0 else 0, maximum +padding)
        
        ax_t.grid('both', ls='dotted')
        ax_s.grid('both', ls='dotted')
        # ax_t.axis('off')
        # ax_s.axis('off')
        
        if not bin_attr=='vcount':
            ax_t.bar_label(t_bars, ['{0:.3f}'.format(t_val)+('%' if percent_deviation else '')
                                    for t_val in tropo_bar_vals], 
                           padding=2)
            ax_s.bar_label(s_bars, ['{0:.3f}'.format(s_val)+('%' if percent_deviation else '')
                                    for s_val in strato_bar_vals], 
                           padding=2)

        else:
            ax_t.bar_label(t_bars, ['{0:.0f}'.format(t_val) for t_val in tropo_bar_vals], 
                           padding=2)
            ax_s.bar_label(s_bars, ['{0:.0f}'.format(s_val) for s_val in strato_bar_vals], 
                           padding=2)

        for ax in [ax_t, ax_s]: 
            ax.yaxis.set_major_locator(ticker.NullLocator())
        fig.subplots_adjust(wspace=0)

    def matrix_plot_stdev_subs(self, substance, note='', tps=None, minimise_tps=True,
                               atm_layer='both', savefig=False) -> (np.array, np.array):
        """
        Create matrix plot showing variability per latitude bin per tropopause definition
    
        Parameters:
            glob_obj (GlobalObject): Contains the data in self.df
            key short_name (str): Substance short name to show, e.g. 'n2o'

        Returns:
            tropospheric, stratospheric standard deviations within each bin as list for each tp coordinate
        """
        if not tps: 
            tps = [tp for tp in dcts.get_coordinates(tp_def='not_nan')
                   if 'tropo_'+tp.col_name in self.glob_obj.df_sorted.columns]
    
        if minimise_tps:
            tps = tools.minimise_tps(tps)
    
        lat_bmin, lat_bmax = -90, 90 # np.nanmin(lat), np.nanmax(lat)
        lat_bci = bp.Bin_equi1d(lat_bmin, lat_bmax, self.glob_obj.grid_size)
    
        tropo_stdevs = np.full((len(tps), lat_bci.nx), np.nan)
        tropo_av_stdevs = np.full(len(tps), np.nan)
        strato_stdevs = np.full((len(tps), lat_bci.nx), np.nan)
        strato_av_stdevs = np.full(len(tps), np.nan)
    
        tropo_out_list = []
        strato_out_list = []
    
        for i, tp in enumerate(tps):
            # troposphere
            tropo_data = self.glob_obj.sel_tropo(**tp.__dict__).df
            tropo_lat = np.array([tropo_data.geometry[i].y for i in range(len(tropo_data.index))]) # lat
            tropo_out_lat = bp.Simple_bin_1d(tropo_data[substance.col_name], tropo_lat, 
                                             lat_bci, count_limit = self.glob_obj.count_limit)
            tropo_out_list.append(tropo_out_lat)
            tropo_stdevs[i] = tropo_out_lat.vstdv if not all(np.isnan(tropo_out_lat.vstdv)) else tropo_stdevs[i]
            
            # weighted average stdv
            tropo_nonan_stdv = tropo_out_lat.vstdv[~ np.isnan(tropo_out_lat.vstdv)]
            tropo_nonan_vcount = tropo_out_lat.vcount[~ np.isnan(tropo_out_lat.vstdv)]
            tropo_weighted_average = np.average(tropo_nonan_stdv, weights = tropo_nonan_vcount)
            tropo_av_stdevs[i] = tropo_weighted_average 
            
            # stratosphere
            strato_data = self.glob_obj.sel_strato(**tp.__dict__).df
            strato_lat = np.array([strato_data.geometry[i].y for i in range(len(strato_data.index))]) # lat
            strato_out_lat = bp.Simple_bin_1d(strato_data[substance.col_name], strato_lat, 
                                              lat_bci, count_limit = self.glob_obj.count_limit)
            strato_out_list.append(strato_out_lat)
            strato_stdevs[i] = strato_out_lat.vstdv if not all(np.isnan(strato_out_lat.vstdv)) else strato_stdevs[i]
            
            # weighted average stdv
            strato_nonan_stdv = strato_out_lat.vstdv[~ np.isnan(strato_out_lat.vstdv)]
            strato_nonan_vcount = strato_out_lat.vcount[~ np.isnan(strato_out_lat.vstdv)]
            strato_weighted_average = np.average(strato_nonan_stdv, weights = strato_nonan_vcount)
            strato_av_stdevs[i] = strato_weighted_average 
    
        # Plotting
        # -------------------------------------------------------------------------
        pixels = self.glob_obj.grid_size # how many pixels per imshow square
        yticks = np.linspace(0, (len(tps)-1)*pixels, num=len(tps))[::-1] # order was reversed for some reason
        tp_labels = [tp.label(True)+'\n' for tp in tps]
        xticks = np.arange(lat_bmin, lat_bmax+self.glob_obj.grid_size, self.glob_obj.grid_size)
    
        fig = plt.figure(dpi=200, figsize=(lat_bci.nx*0.825, 10))#len(tps)*2))
    
        gs = gridspec.GridSpec(5, 2, figure=fig,
                               height_ratios = [1, 0.1, 0.02, 1, 0.1],
                               width_ratios = [1, 0.09])
        axs = gs.subplots()
    
        [ax.remove() for ax in axs[2, 0:]]
        middle_ax = plt.subplot(gs[2, 0:])
        middle_ax.axis('off')
    
        ax_strato1 = axs[0,0]
        ax_strato2 = axs[0,1]
        [ax.remove() for ax in  axs[1, 0:]]
        cax_s = plt.subplot(gs[1, 0:])
        
        ax_tropo1 = axs[3,0]
        ax_tropo2 = axs[3,1]
        [ax.remove() for ax in axs[4, 0:]]
        cax_t = plt.subplot(gs[4, 0:])
    
        # Plot STRATOSPHERE
        # -------------------------------------------------------------------------
        try: 
            vmin, vmax = substance.vlims('vstdv', 'strato')
        except KeyError: 
            vmin, vmax = np.nanmin(strato_stdevs), np.nanmax(strato_stdevs)
            
        norm = Normalize(vmin, vmax)  # normalise color map to set limits
        strato_cmap = dcts.dict_colors()['vstdv_strato'] # plt.cm.BuPu  # create colormap
        ax_strato1.set_title(f'Stratospheric variability of {substance.label()}{note}', fontsize=14)
    
        img = ax_strato1.matshow(strato_stdevs, alpha=0.75,
                         extent = [lat_bmin, lat_bmax,
                                   0, len(tps)*pixels],
                         cmap = strato_cmap, norm=norm)
        ax_strato1.set_yticks(yticks, labels=tp_labels)
        ax_strato1.set_xticks(xticks, loc='bottom')
        ax_strato1.tick_params(axis='x', top=False, labeltop=False, labelbottom=True)
    
        for label in ax_strato1.get_yticklabels():
            label.set_verticalalignment('bottom')
    
        ax_strato1.grid('both')
        ax_strato1.set_xlabel('Latitude [°N]')
    
        # add numeric values
        for j,x in enumerate(xticks[:-1]):
            for i,y in enumerate(yticks):
                value = strato_stdevs[i,j]
                if str(value) != 'nan':
                    ax_strato1.text(x+0.5*self.glob_obj.grid_size,
                            y+0.5*pixels,
                            '{0:.2f}'.format(value) if value>vmax/100 else '<{0:.2f}'.format(vmax/100),
                            va='center', ha='center')
        cbar = plt.colorbar(img, cax=cax_s, orientation='horizontal')
        cbar.set_label(f'Standard deviation of {substance.label(name_only=True)} within bin [{substance.unit}]')
        # make sure vmin and vmax are shown as colorbar ticks
        cbar_vals = cbar.get_ticks()
        cbar_vals = [vmin] + cbar_vals[1:-1].tolist() + [vmax]
        cbar.set_ticks(cbar_vals)
    
        # Stratosphere average variability
        img = ax_strato2.matshow(np.array([strato_av_stdevs]).T, alpha=0.75,
                         extent = [0, self.glob_obj.grid_size,
                                   0, len(tps)*pixels],
                         cmap = strato_cmap, norm=norm)
        for i,y in enumerate(yticks): 
            value = strato_av_stdevs[i]
            if str(value) != 'nan':
                ax_strato2.text(0.5*self.glob_obj.grid_size,
                        y+0.5*pixels,
                        '{0:.2f}'.format(value) if value>vmax/100 else '<{0:.2f}'.format(vmax/100),
                        va='center', ha='center')
        ax_strato2.tick_params(axis='both', bottom=False, top=False, labeltop=False, left=False, labelleft=False)
        ax_strato2.set_xlabel('Average')
    
        # Plot TROPOSPHERE
        # -------------------------------------------------------------------------
        try: 
            vmin, vmax = substance.vlims('vstdv', 'tropo')
        except KeyError: 
            vmin, vmax = np.nanmin(strato_stdevs), np.nanmax(strato_stdevs)
        norm = Normalize(vmin, vmax)  # normalise color map to set limits
        tropo_cmap = dcts.dict_colors()['vstdv_tropo'] # cmr.get_sub_cmap('YlOrBr', 0, 0.75) # create colormap
        ax_tropo1.set_title(f'Tropospheric variability of {substance.label()}{note}', fontsize=14)
    
        img = ax_tropo1.matshow(tropo_stdevs, alpha=0.75,
                         extent = [lat_bmin, lat_bmax,
                                   0, len(tps)*pixels],
                         cmap = tropo_cmap, norm=norm)
        ax_tropo1.set_yticks(yticks, labels=tp_labels)
        ax_tropo1.set_xticks(xticks, loc='bottom')
        ax_tropo1.tick_params(axis='x', top=False, labeltop=False, labelbottom=True)
        ax_tropo1.set_xlabel('Latitude [°N]')
    
        for label in ax_tropo1.get_yticklabels():
            label.set_verticalalignment('bottom')
    
        ax_tropo1.grid('both')
        # ax1.set_xlim(-40, 90)
    
        # add numeric values
        for j,x in enumerate(xticks[:-1]):
            for i,y in enumerate(yticks):
                value = tropo_stdevs[i,j]
                if str(value) != 'nan':
                    ax_tropo1.text(x+0.5*self.glob_obj.grid_size,
                            y+0.5*pixels,
                            '{0:.2f}'.format(value) if value>vmax/100 else '<{0:.2f}'.format(np.ceil(vmax/100)),
                            va='center', ha='center')
        cbar = plt.colorbar(img, cax=cax_t, orientation='horizontal')
        cbar.set_label(f'Standard deviation of {substance.label(name_only=True)} within bin [{substance.unit}]')
        # make sure vmin and vmax are shown as colorbar ticks
        cbar_vals = cbar.get_ticks()
        cbar_vals = [vmin] + cbar_vals[1:-1].tolist() + [vmax]
        cbar.set_ticks(cbar_vals)
        
        # Tropopsphere average variability
        img = ax_tropo2.matshow(np.array([tropo_av_stdevs]).T, alpha=0.75,
                         extent = [0, self.glob_obj.grid_size,
                                   0, len(tps)*pixels],
                         cmap = tropo_cmap, norm=norm)
    
        for i,y in enumerate(yticks): 
            value = tropo_av_stdevs[i]
            if str(value) != 'nan':
                ax_tropo2.text(0.5*self.glob_obj.grid_size,
                        y+0.5*pixels,
                        '{0:.2f}'.format(value) if value>vmax/100 else '<{0:.2f}'.format(np.ceil(vmax/100)),
                        va='center', ha='center')
    
    
        ax_tropo2.set_xlabel('Average')
        ax_tropo2.tick_params(axis='both', bottom=False, top=False, labeltop=False, left=False, labelleft=False)
    
        # -------------------------------------------------------------------------
        fig.tight_layout()
        fig.subplots_adjust(top=0.8)
    
        if savefig:
            plt.savefig(f'E:/CARIBIC/Plots/variability_lat_binned/variability_{substance.col_name}.png', format='png')
        fig.show()
    
        return tropo_out_list, strato_out_list
    
    def matrix_plot_stdev(self, note='', atm_layer='both', savefig=False,
                          minimise_tps=True, **subs_kwargs):
        substances = [s for s in self.glob_obj.substances
                      if not s.col_name.startswith('d_')]
    
        for subs in substances:
            self.matrix_plot_stdev_subs(subs, note=note, minimise_tps=minimise_tps,
                                       atm_layer=atm_layer, savefig=savefig)


def plot_seasonal_vstdv_n2o_filter(glob_obj, subs, theta, rstd=False, 
                                   tp_kwargs = dict(vcoord='mxr', ID='GHG')): 
    """ Scatter plot of mean seasnoal variability for troopause definitions """
    
    bp1 = BinPlotter1D(glob_obj)
    bp_tropo = BinPlotter1D(glob_obj.sel_tropo(**tp_kwargs))
    bp_strato = BinPlotter1D(glob_obj.sel_strato(**tp_kwargs))
    
    pt_range = 70
    # figsize = [6.4, 4.8] # default
    figsize = [5, 3]

    # if glob_obj.ID == 'PGS': 
    #     pt_range = 140
    #     figsize = [6.4, 4.8 * 1.5]

    fig, ax = plt.subplots(dpi=250, figsize = figsize)
    # ax.set_ylim(280, 400)
    ax.set_ylim(280, 370)
    
    for instance, c, label in zip([bp1, bp_tropo, bp_strato],
                           ['grey', 'teal', 'orange'], 
                           ['Unfiltered', 'Tropo ', 'Strato ']):
        # label = theta.label(True)
        
        df = instance.rms_seasonal_vstdv(subs, theta)
        
        if not rstd: 
            x_data = df['rms_vstdv']
        else: 
            x_data = df['rms_rvstd'] * 100 # %
        
        y_data = df.index
        
        ax.plot(x_data, y_data, 
                '-', marker='d', 
                c=c,
                # c=dcts.dict_season()[f'color_{s}'],
                label=label,
                path_effects=[instance.outline], zorder=10)
        
        ax.set_ylabel('$\Theta$ [K]') 
        
        ax.grid('both', ls='dotted')
        # ax.set_title(f'{self.glob_obj.ID}')
        if not rstd: 
            ax.set_xlabel(f'Mean seasonal variability of {subs.label(name_only=True)} [{subs.unit}]')
        else: 
            ax.set_xlabel(f'Relative seasonal variability of {subs.label(name_only=True)} [%]')
        ax.set_zorder(3)
        ax.patch.set_visible(False)
        
        h,l = ax.get_legend_handles_labels()
        ax.legend(loc='lower right')

        tools.add_zero_line(ax)


class BinPlotter2D(BinPlotter): 
    """ Two-dimensional binning & plotting. 
    
    Methods: 
        calc_average(bin2d_inst, bin_attr)
        calc_ts_averages(bin2d_inst, bin_attr)
        single_2d_plot(ax, bin2d_inst, bin_attr, xcoord, ycoord,
                       cmap, norm, xlims, ylims)
        seasonal_2d_plots(subs, xcoord, ycoord, bin_attr, cmap)
        plot_2d_mxr(subs, xcoord, ycoord)
        plot_2d_stdv(subs, xcoord, ycoord)
        plot_mixing_ratios()
        plot_stdv_subset()
        plot_total_2d(subs, xcoord, ycoord, bin_attr)
        plot_mxr_diff(params_1, params2)
        plot_differences()
    """
    def __init__(self, glob_obj, **kwargs): 
        """ Initialise bin plotter. """
        super().__init__(glob_obj, **kwargs)

    def calc_average(self, bin2d_inst, bin_attr='vstdv'):
        """ Calculate weighted overall average. """
        data = getattr(bin2d_inst, bin_attr)
        data = data[~np.isnan(data)]

        weights = bin2d_inst.vcount
        weights = bin2d_inst.vcount[[i!=0 for i in weights]]

        try: weighted_average = np.average(data, weights = weights)
        except ZeroDivisionError: weighted_average = np.nan
        return weighted_average

    def calc_ts_averages(self, bin2d_inst, bin_attr = 'vstdv'):
        """ Calculate tropospheric and stratospheric weighted averages for rel_to_tp coord data. """
        data = getattr(bin2d_inst, bin_attr)

        tropo_mask = bin2d_inst.yintm < 0
        tropo_data = data[[tropo_mask]*bin2d_inst.nx]
        tropo_data = tropo_data[~ np.isnan(tropo_data)]

        tropo_weights = bin2d_inst.vcount[[tropo_mask]*bin2d_inst.nx]
        tropo_weights = tropo_weights[[i!=0 for i in tropo_weights]]

        try: tropo_weighted_average =  np.average(tropo_data, weights = tropo_weights)
        except ZeroDivisionError: tropo_weighted_average = np.nan

        # stratosphere
        strato_mask = bin2d_inst.yintm > 0
        strato_data = data[[strato_mask]*bin2d_inst.nx]
        strato_data = strato_data[~ np.isnan(strato_data)]

        strato_weights = bin2d_inst.vcount[[strato_mask]*bin2d_inst.nx]
        strato_weights = strato_weights[[i!=0 for i in strato_weights]]

        try: strato_weighted_average = np.average(strato_data, weights = strato_weights)
        except ZeroDivisionError: strato_weighted_average = np.nan

        return tropo_weighted_average, strato_weighted_average

    def yearly_maps(self, subs, bin_attr):
        # glob_obj, subs, single_yr=None, c_pfx=None, years=None, detr=False):
        """ Create binned 2D plots for each available year on a grid. """
        
        nplots = len(self.glob_obj.years)
        nrows = nplots if nplots <= 4 else math.ceil(nplots / 3)
        ncols = 1 if nplots <= 4 else 3

        fig = plt.figure(dpi=100, figsize=(6 * ncols, 3 * nrows))

        grid = AxesGrid(fig, 111, # similar to subplot(142)
                        nrows_ncols=(nrows, ncols),
                        axes_pad=0.4,
                        share_all=True,
                        label_mode="all",
                        cbar_location="bottom",
                        cbar_mode="single")

        if nplots >= 4:
            data_type = 'measured' if subs.model == 'MSMT' else 'modeled'
            fig.suptitle(f'{bin_attr} of binned global {data_type} mixing ratios of {subs.label()}',
                         fontsize=25)
            plt.subplots_adjust(top=0.96)

        world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))

        xcoord = dcts.get_coord(col_name='geometry.x')
        ycoord = dcts.get_coord(col_name='geometry.y')
        xbsize, ybsize = self._get_bsize(subs, xcoord, ycoord)
        bin_equi2d = bp.Bin_equi2d(-180, 180, xbsize, 
                                   -90, 90, ybsize)
        
        vlims = self.get_vlimit(subs, bin_attr)
        norm = Normalize(*vlims)  # normalise color map to set limits
        
        for i, (ax, year) in enumerate(zip(grid, self.glob_obj.years)):
            ax.text(**dcts.note_dict(ax, 0.13, 0.1, f'{year}'), weight='bold')
            world.boundary.plot(ax=ax, color='grey', linewidth=0.3)

            # label outer plot axes
            if grid._get_col_row(i)[0] == 0:
                ax.set_ylabel('Latitude [°N]')
            if grid._get_col_row(i)[0] == ncols:
                ax.set_xlabel('Longitude [°E]')

            ax.set_xlim(-180, 180)
            ax.set_ylim(-60, 100)

            # plot data
            df_year = self.glob_obj.sel_year(year).df
            if df_year.empty: 
                continue
            out = self.bin_2d(subs, xcoord, ycoord, bin_equi2d, df=df_year)         
            data = getattr(out, bin_attr)

            img = ax.imshow(data.T, origin='lower',
                            cmap=dcts.dict_colors()[bin_attr], norm=norm,
                            extent=[bin_equi2d.xbmin, bin_equi2d.xbmax, 
                                    bin_equi2d.ybmin, bin_equi2d.ybmax])

        for i, ax in enumerate(grid):  # hide extra plots
            if i >= nplots:
                ax.axis('off')

        if 'img' in locals(): 
            cbar = grid.cbar_axes[0].colorbar(img, aspect=5, pad=0.1) # colorbar
            cbar.ax.tick_params(labelsize=15)
            cbar.ax.minorticks_on()
            cbar.ax.set_xlabel(subs.label(), fontsize=15)
        
        fig.tight_layout()
        plt.show()

    def seasonal_2d_plots(self, subs, xcoord, ycoord, bin_attr, **kwargs):
        """
        Parameters:
            bin_attr (str): 'vmean', 'vstdv', 'vcount'
        """
        
        try: 
            cmap = dcts.dict_colors()[bin_attr]
        except: 
            cmap = plt.viridis

        binned_seasonal = self.bin_2d_seasonal(subs, xcoord, ycoord, **kwargs)

        if not any(bin_attr in bin2d_inst.__dict__ for bin2d_inst in binned_seasonal.values()):
            raise KeyError(f'\'{bin_attr}\' is not a valid attribute of Bin2D objects.')

        vlims = self.get_vlimit(subs, bin_attr)
        xlims = self.get_coord_lims(xcoord, 'x')
        ylims = self.get_coord_lims(ycoord, 'y')
        
        # vlims, xlims, ylims = self.get_limits(subs, xcoord, ycoord, bin_attr)
        norm = Normalize(*vlims)
        fig, axs = plt.subplots(2, 2, dpi=100, figsize=(8,9),
                                sharey=True, sharex=True)

        fig.subplots_adjust(top = 1.1)

        data_title = 'Mixing ratio' if bin_attr=='vmean' else 'Varibility'
        # fig.suptitle(f'{data_title} of {subs.label()}', y=0.95)

        seasons = binned_seasonal.keys()

        for season, ax in zip(seasons, axs.flatten()):
            bin2d_inst = binned_seasonal[season]
            ax.set_title(dcts.dict_season()[f'name_{season}'])
            
            img = self.single_2d_plot(ax, bin2d_inst, bin_attr, xcoord, ycoord, 
                               cmap, norm, xlims, ylims, **kwargs)

        fig.subplots_adjust(right=0.9)
        fig.tight_layout(pad=2.5)

        cbar = fig.colorbar(img, ax = axs.ravel().tolist(), aspect=30, pad=0.08, 
                            orientation='horizontal', 
                            # location='top', ticklocation='bottom'
                            )
        cbar.ax.set_xlabel(data_title+' of '+subs.label(), 
                           # fontsize=13
                           )

        cbar_vals = cbar.get_ticks()
        cbar_vals = [vlims[0]] + cbar_vals[1:-1].tolist() + [vlims[1]]
        cbar.set_ticks(cbar_vals, ticklocation='bottom')

        plt.show()

    def plot_2d_mxr(self, subs, xcoord, ycoord, **kwargs):
        """ Plot binned average mixing ratios on an x vs. y plot. """
        bin_attr = 'vmean'
        self.seasonal_2d_plots(subs, xcoord, ycoord, bin_attr, **kwargs)

    def plot_2d_stdv(self, subs, xcoord, ycoord, averages=True, **kwargs):
        """ Plot binned substance standard deviation on an x vs. y plot. """
        bin_attr = 'vstdv'
        self.seasonal_2d_plots(subs, xcoord, ycoord, bin_attr, averages=averages, **kwargs)

    def plot_mixing_ratios(self, **kwargs):
        """ Plot all possible permutations of subs, xcoord, ycoord. """
        permutations = list(itertools.product(self.glob_obj.substances,
                                              # self.x_coordinates,
                                              [dcts.get_coord(col_name='int_ERA5_EQLAT')],
                                              # self.y_coordinates
                                              tools.minimise_tps(dcts.get_coordinates(tp_def='not_nan'))
                                              ))
        for perm in permutations:
            self.plot_2d_mxr(*perm, **kwargs)

    def plot_stdv_subset(self, **subs_kwargs):
        """ Plot a small subset of standard deviation plots. """
        tps = tools.minimise_tps(dcts.get_coordinates(tp_def='not_nan'))
        # xcoords = [dcts.get_coord(col_name='geometry.y'), dcts.get_coord(col_name='int_ERA5_EQLAT')]
        xcoords = [dcts.get_coord(col_name='int_ERA5_EQLAT')]
        substances = [s for s in dcts.get_substances(ID='GHG', **subs_kwargs) if s.short_name.startswith('detr_')]

        for subs in substances:
            print(subs)
            for tp in tps:
                for xcoord in xcoords:
                    self.plot_2d_stdv(subs, xcoord, tp)

    def plot_total_2d(self, subs, xcoord, ycoord, bin_attr='vstdv', **kwargs):
        """ Single 2D plot of varibility of given substance. """
    
        df = self.df

        if xcoord.col_name == 'geometry.x': 
            x = df.geometry.x
        elif xcoord.col_name == 'geometry.y': 
            x = df.geometry.y
        else:
            x = np.array(df[xcoord.col_name])

        if ycoord.col_name == 'geometry.y': 
            y =  df.geometry.y
        else: 
            y = np.array(df[ycoord.col_name])

        xbsize = self._get_bsize(xcoord, 'x')
        ybsize = self._get_bsize(ycoord, 'y')

        # get bins as multiples of the bin size
        xbmax = ((np.nanmax(x) // xbsize) + 1) * xbsize
        xbmin = (np.nanmin(x) // xbsize) * xbsize

        ybmax = ((np.nanmax(y) // ybsize) + 1) * ybsize
        ybmin = (np.nanmin(y) // ybsize) * ybsize

        bin_equi2d = bp.Bin_equi2d(xbmin, xbmax, xbsize,
                                   ybmin, ybmax, ybsize)

        bin2d_inst = bp.Simple_bin_2d(np.array(df[subs.col_name]), x, y,
                               bin_equi2d, count_limit=self.count_limit)
        
        vlims = self.get_vlimit(subs, bin_attr)
        xlims = self.get_coord_lims(xcoord, 'x')
        ylims = self.get_coord_lims(ycoord, 'y')
        
        # vlims, xlims, ylims = self.get_limits(subs, xcoord, ycoord, bin_attr=bin_attr)
        norm = Normalize(*vlims)
        fig, ax = plt.subplots(dpi=250, figsize=(8,9))
        fig.subplots_adjust(top = 1.1)

        data_title = 'Mixing ratio' if bin_attr=='vmean' else 'Varibility'
        # fig.suptitle(f'{data_title} of {subs.label()}', y=0.95)

        cmap = dcts.dict_colors()[bin_attr]

        img = self.single_2d_plot(ax, bin2d_inst, bin_attr, xcoord, ycoord, 
                           cmap, norm, xlims, ylims, **kwargs)

        fig.subplots_adjust(right=0.9)
        fig.tight_layout(pad=2.5)

        cbar = fig.colorbar(img, ax = ax, aspect=30, pad=0.09, orientation='horizontal')
        cbar.ax.set_xlabel(data_title+' of '+subs.label())

        cbar_vals = cbar.get_ticks()
        cbar_vals = [vlims[0]] + cbar_vals[1:-1].tolist() + [vlims[1]]
        cbar.set_ticks(cbar_vals)

        plt.show()

    def plot_mxr_diff(self, params_1, params_2, **kwargs):
        """ Plot difference between two plots. 
        
        NOT the final version, not particularly useable right now. 
        """
        subs1, xcoord1, ycoord1 = params_1
        subs2, xcoord2, ycoord2 = params_2

        xbsize, ybsize = self._get_bsize(xcoord1, 'x'), self._get_bsize(ycoord1, 'y')

        bin_equi2d = bp.Bin_equi2d(np.nanmin(self.df[xcoord1.col_name]),
                                   np.nanmax(self.df[xcoord1.col_name]),
                                   xbsize,
                                   np.nanmin(self.df[ycoord1.col_name]),
                                   np.nanmax(self.df[ycoord1.col_name]),
                                   ybsize)

        binned_seasonal_1 = self.bin_2d_seasonal(*params_1, bin_equi2d=bin_equi2d)
        binned_seasonal_2 = self.bin_2d_seasonal(*params_2, bin_equi2d=bin_equi2d)

        # vlims = self.get_vlimit(subs1, 'vmean')
        xlims = self.get_coord_lims(xcoord1, 'x')
        ylims = self.get_coord_lims(ycoord1, 'y')

        # vlims, xlims, ylims = self.get_limits(*params_1)
        cmap = plt.cm.PiYG

        fig, axs = plt.subplots(2, 2, dpi=250, figsize=(8,9),
                                sharey=True, sharex=True)

        for season, ax in zip([1,2,3,4], axs.flatten()):
            ax.set_title(dcts.dict_season()[f'name_{season}'])
            ax.set_facecolor('lightgrey')

            # note simple substraction filters out everything where either is nan
            vmean = binned_seasonal_1[season].vmean - binned_seasonal_2[season].vmean
            vmax_abs = max(abs(np.nanmin(vmean)), abs(np.nanmax(vmean)))
            norm = Normalize(-vmax_abs, vmax_abs)

            bci = binned_seasonal_1[season].binclassinstance

            img = ax.imshow(vmean.T, cmap = cmap, norm=norm,
                            aspect='auto', origin='lower',
                            extent=[bci.xbmin, bci.xbmax, bci.ybmin, bci.ybmax])
            ax.set_xlim(xlims[0]*0.95, xlims[1]*1.05)
            ax.set_ylim(ylims[0]*0.95, ylims[1]*1.05)

            ax.set_xlabel(xcoord1.label())
            ax.set_ylabel(ycoord1.label())

            if kwargs.get('note'):
                ax.text(**dcts.note_dict(ax, s=kwargs.get('note')))

        fig.subplots_adjust(right=0.9)
        fig.tight_layout(pad=2.5)

        cbar = fig.colorbar(img, ax = axs.ravel().tolist(), aspect=30,
                            pad=0.09, orientation='horizontal')
        cbar.ax.set_xlabel(
            f'{subs1.label()} {xcoord1.label()} {ycoord1.label()} \n vs.\n\
{subs2.label()} {xcoord2.label()} {ycoord2.label()}')
        plt.show()

    def plot_differences(self, subs_params={}, **kwargs):
        """ Plot the mixing ratio difference between different substance cols and coordinates. """
        
        substances = dcts.get_substances(ID='GHG', detr=True, **subs_params)
        tps = tools.minimise_tps(dcts.get_coordinates(tp_def='not_nan'))
        eql = dcts.get_coord(hcoord='eql', model='ERA5')
        
        permutations = list(itertools.product(substances, [eql], tps))

        for params_1, params_2 in itertools.combinations(permutations, 2):
            if (params_1[0].short_name == params_2[0].short_name
                and params_1[1].hcoord == params_2[1].hcoord
                and params_1[2].vcoord == params_2[2].vcoord):
                # only compare same substance in same coordinate system
                self.plot_mxr_diff(params_1, params_2, **kwargs)

    def single_2d_plot(self, ax, bin2d_inst, bin_attr, xcoord, ycoord, 
                       cmap, norm, xlims, ylims, **kwargs):
        """ Plot binned data with imshow. """

        bci = bin2d_inst.binclassinstance
        data = getattr(bin2d_inst, bin_attr) # atttribute: 'vmean', 'vstdv'

        img = ax.imshow(data.T,
                        cmap = cmap, norm=norm,
                        aspect='auto', origin='lower',
                        # if not ycoord.vcoord in ['p', 'mxr'] else 'upper',
                        extent=[bci.xbmin, bci.xbmax, bci.ybmin, bci.ybmax] 
                        # if not ycoord.vcoord in ['p', 'mxr'] else [bci.xbmin, bci.xbmax, bci.ybmax, bci.ybmin]
                        )

        ax.set_xlabel(xcoord.label())
        ax.set_ylabel(ycoord.label())

        ax.set_xlim(*xlims)
        #TODO with count_limit > 1, might not have data in all bins - get dynamic bins?
        ax.set_ylim(ylims[0] - bci.ybsize*1.5, ylims[1] + bci.ybsize*1.5)

        ax.set_xticks(np.arange(-90, 90+30, 30)) # stop+30 to include stop

        if bci.ybmin < 0:
            # make sure 0 is included in ticks, evenly spaced away from 0
            ax.set_yticks(list(np.arange(0, abs(bci.ybmin) + bci.ybsize*3, bci.ybsize*3) * -1)
                          + list(np.arange(0, bci.ybmax + bci.ybsize, bci.ybsize*3)))
        else:
            ax.set_yticks(np.arange(bci.ybmin, bci.ybmax + bci.ybsize*3, bci.ybsize*3))
        ax.set_yticks(np.arange(bci.ybmin, bci.ybmax+bci.ybsize, bci.ybsize), minor=True)

        if ycoord.rel_to_tp:
            ax.hlines(0, *xlims, color='k', ls='dashed', zorder=1, lw=1)

        if kwargs.get('averages'):
            if ycoord.rel_to_tp:
                tropo_av, strato_av = self.calc_ts_averages(bin2d_inst, bin_attr)
                ax.text(**dcts.note_dict(ax, x=0.275, y = 0.9,
                                             s=str('S-Av: {0:.2f}'.format(strato_av)
                                                   + '\n' + 'T-Av: {0:.2f}'.format(tropo_av))))
            else: 
                average = self.calc_average(bin2d_inst, bin_attr)
                ax.text(**dcts.note_dict(ax, x=0.225, y = 0.9,
                                             s=str('Av: {0:.2f}'.format(average))))

        if kwargs.get('note'):
            ax.text(**dcts.note_dict(ax, s=kwargs.get('note')))

        ax.grid('both', lw=0.4)
        
        return img

class BinPlotter3D(BinPlotter): 
    """ Three-dimensional binning & plotting. 
    
    Methods: 
        stratosphere_map(subs, tp, bin_attr)
    """

    def z_crossection(self, subs, tp, bin_attr, threshold = 3, zbsize=None): 
        """ Create lat/lon gridded plots for all z-bins. """
        binned_data = self.bin_3d(subs, tp, zbsize=zbsize)
        
        data3d = getattr(binned_data, bin_attr)

        lon_bins = binned_data.xintm
        lat_bins = binned_data.yintm
        z_bins = binned_data.zintm

        vcounts_per_z_level = [sum(j) for j in [sum(i) for i in binned_data.vcount]]
        
        # first z level: binned_data.vcount[:,:,0]
        
        for ix in range(binned_data.nx):
            for iy in range(binned_data.ny): 
                for iz in range(binned_data.nz):
                    datapoint = data3d[ix, iy, iz]

        
        # fig = plt.figure(dpi=150)
        
        vlims = self.get_vlimit(subs, bin_attr)
        norm = Normalize(*vlims)
        cmap = dcts.dict_colors()[bin_attr]

        data_title = 'Mixing ratio' if bin_attr=='vmean' else 'Varibility'
        # fig.suptitle(f'{data_title} of {subs.label()}', y=0.95)

        cmap = dcts.dict_colors()[bin_attr]
        world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
        
        if tp.rel_to_tp:
            title = f'Cross section binned relative to {tp.label(filter_label=True)} Tropopause'
        else: 
            title = '' # f' in {tp.label()}'

        for iz in range(binned_data.nz):
            data2d = data3d[:,:,iz]
            if sum(~np.isnan(data2d.flatten())) > threshold: 
                fig, ax = plt.subplots(dpi=200)
                world.boundary.plot(ax=ax, color='grey', linewidth=0.3, zorder=0)
                ax.set_title(title)
                ax.text(s = '{:.0f} to {:.0f} {}'.format(binned_data.zbinlimits[iz], binned_data.zbinlimits[iz+1], tp.unit),
                        **dcts.note_dict(ax, x=0.025, y=0.05))

                img = ax.imshow(data2d.T,
                                cmap = cmap, norm=norm,
                                aspect='auto', origin='lower',
                                # if not ycoord.vcoord in ['p', 'mxr'] else 'upper',
                                extent=[binned_data.xbmin, binned_data.xbmax, 
                                        binned_data.ybmin, binned_data.ybmax],
                                zorder = 1)

                cbar = fig.colorbar(img, ax = ax, aspect=30, pad=0.09, orientation='horizontal')
                cbar.ax.set_xlabel(f'{data_title} of {subs.label()}')
                
                plt.show()
    
        return binned_data
        

    def stratosphere_map(self, subs, tp, bin_attr): 
        """ Plot (first two ?) stratospheric bins on a lon-lat binned map. """
        df = self.glob_obj.sel_strato(**tp.__dict__).df
        # df = self.glob_obj.sel_tropo(**tp.__dict__).df

        fig, ax = plt.subplots(figsize=(9,9))
        ax.set_title(tp.label(True))
        world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
        world.boundary.plot(ax=ax, color='grey', linewidth=0.3)
        
        xcoord = dcts.get_coord(col_name='geometry.x')
        ycoord = dcts.get_coord(col_name='geometry.y')
        
        ax.set_ylabel('Latitude [°N]')
        ax.set_xlabel('Longitude [°E]')
        ax.set_xlim(-180, 180)
        ax.set_ylim(-60, 100)
        
        xbsize = self._get_bsize(xcoord, 'x')
        ybsize = self._get_bsize(ycoord, 'y')
        zbsize = self._get_bsize(tp)

        
        bin_equi3d = bp.Bin_equi2d(-180, 180, xbsize, 
                                   -90, 90, ybsize, 
                                   0, zbsize*2, zbsize)
        
        vlims = self.get_vlimit(subs, bin_attr)
        
        # vlims,_,_ = self.get_limits(subs, bin_attr = bin_attr)
        norm = Normalize(*vlims)  # normalise color map to set limits

        # ---------------------------------------------------------------------

        out = self.bin_3d(subs, xcoord, ycoord, tp, bin_equi3d, df=df)         
        data = getattr(out, bin_attr)

        img = ax.imshow(data.T, origin='lower',
                        cmap=dcts.dict_colors()[bin_attr], norm=norm,
                        extent=[bin_equi3d.xbmin, bin_equi3d.xbmax, 
                                bin_equi3d.ybmin, bin_equi3d.ybmax])
        # cbar = 
        plt.colorbar(img, ax=ax, pad=0.1, orientation='horizontal') # colorbar
        plt.show()

def n2o_tp_stdv_rms(glob_obj, subs, rstd=False): 
    n2o = dcts.get_coord(col_name='N2O')
    vcoord = dcts.get_coord(col_name='int_ERA5_THETA')
    
    attr = 'rms_rvstd' if rstd else 'rms_vstdv'
    
    # bin_equi1d = bp.Bin_equi1d(4, 13, 0.75)
    
    bp = BinPlotter1D(glob_obj)
    var_df = bp.rms_seasonal_vstdv(subs, vcoord)
    
    strato = BinPlotter1D(glob_obj.sel_strato(**n2o.__dict__))
    s_var_df = strato.rms_seasonal_vstdv(subs, vcoord) #, 
                                         # bin_equi1d=bin_equi1d)
    
    tropo = BinPlotter1D(glob_obj.sel_tropo(**n2o.__dict__))
    t_var_df = tropo.rms_seasonal_vstdv(subs, vcoord) #,
                                        # bin_equi1d=bin_equi1d)
    
    fig, ax = plt.subplots(dpi=250, figsize=(4, 6))
    
    ax.plot(var_df[attr]*(100 if rstd else 1), var_df.index, 
            '-', marker='d', 
            c='grey', ls='dashed',
            label='Unfiltered',
            path_effects=[mpe.withStroke(linewidth=2, foreground='white')], zorder=2)

    ax.plot(t_var_df[attr]*(100 if rstd else 1), t_var_df.index, 
            '-', marker='d', 
            c='red',
            label='Tropospheric',
            path_effects=[mpe.withStroke(linewidth=2, foreground='white')], zorder=2)
    
    ax.plot(s_var_df[attr]*(100 if rstd else 1), s_var_df.index, 
            '-', marker='d', 
            c='purple',
            label='Stratospheric',
            path_effects=[mpe.withStroke(linewidth=2, foreground='white')], zorder=2)

    ax.set_ylabel(vcoord.label()) 

    ax.grid('both', ls='dotted')
    rel = 'relative ' if rstd else ''
    ax.set_xlabel(f'Mean {rel}variability of {subs.label(name_only=True)} [%]')
    ax.legend(loc='lower right')
    
    
