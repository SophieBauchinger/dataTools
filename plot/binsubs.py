# -*- coding: utf-8 -*-
"""
@Author: Sophie Bauchinger, IAU
@Date: Tue Jun  6 13:59:31 2023

Showing mixing ratios per season on a plot of coordinate relative to the
tropopause (in km or K) versus equivalent latitude (in deg N)
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


import toolpac.calc.dev_binprocessor as bp

import dictionaries as dcts
import tools

import warnings
warnings.filterwarnings("ignore", message="Boolean Series key will be reindexed to match DataFrame index. result = super().__getitem__(key)")

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
        bin_1d_seasonal(self, subs, coord, bin_equi1d, xbsize)
        bin_2d_seasonal(subs, xcoord, ycoord, bin_equi2d, xbsize, ybsize)
    """
    def __init__(self, glob_obj, **kwargs):
        """ Initialise class instances. 
        Paramters: 
            glob_obj (GlobalData)
            detr (bool): Use data with linear trend wrt MLO05 removed
            
            key xbsize / ybsize (float)
            key ybsize (float)
            key vlims / xlims / ylims (Tuple[float])
            """
        self.glob_obj = glob_obj
        
        if not kwargs.get('all_latitudes'): 
            self.glob_obj = self.glob_obj.sel_latitude(30, 90)

        self.data = {'df' : glob_obj.df} # dataframe

        filter_tps = kwargs.pop('filter_tps') if 'filter_tps' in kwargs else True
        if filter_tps: 
            tps = tools.minimise_tps(dcts.get_coordinates(tp_def='not_nan'))
            self.filter_non_shared_indices(tps)
            print(self.data['df'])
            glob_obj.data['df'] = self.df.copy()

        self.count_limit = glob_obj.count_limit

        self.data['df']['season'] = tools.make_season(self.data['df'].index.month)

        self.kwargs = kwargs

    def __repr__(self):
        return f'<class eqlat.BinPlotter> with minimum points per bin: {self.glob_obj.count_limit} \n\
based on {self.glob_obj.__repr__()}'

    @property
    def df(self) -> pd.DataFrame:
        return self.data['df']

    def get_vlimit(self, subs, bin_attr) -> tuple: 
        """ Get colormap limits for given substance and bin attribute. """
        if 'vlims' in self.kwargs:
            vlims = self.kwargs.get('vlims')
        else:
            try:
                vlims = dcts.get_vlims(subs.short_name, bin_attr=bin_attr)
            except KeyError:
                if bin_attr=='vmean':
                    vlims = (np.nanmin(self.df[subs.col_name]), np.nanmax(self.df[subs.col_name]))
                else:
                    raise KeyError('Could not generate colormap limits.')
            except: 
                raise KeyError('Could not generate colormap limits.')
        return vlims

    def get_coord_lims(self, coord, xyz=None) -> tuple: 
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

    def _get_bsize(self, coord, xyz=None) -> float: 
        """ Get bin size for given coordinate. """
        if xyz=='x' and 'xbsize' in self.kwargs: 
            bsize = self.kwargs.get('xbsize')
        elif xyz=='y' and 'ybsize' in self.kwargs: 
            bsize = self.kwargs.get('ybsize')
        elif xyz=='z' and 'zbsize' in self.kwargs: 
            bsize = self.kwargs.get('zbsize')

        else: 
            try: 
                bsize = dcts.get_default_bsize(coord.hcoord)
            except KeyError: 
                try: 
                    bsize = dcts.get_default_bsize(coord.vcoord)
                except KeyError: 
                    if xyz=='x': 
                        bsize = 10
                    else:
                        lims = self.get_coord_lims(coord, xyz)
                        bsize = 5 * ( np.ceil((lims[1]-lims[0])/10) / 5 )
                        if (lims[1]-lims[0])/10<1: 
                            bsize=0.5
        return bsize

    def filter_non_shared_indices(self, tps):
        """ Filter dataframe for datapoints that don't exist for all tps or are zero. """
        # print('Filtering out non-shared indices. ')
        cols = [tp.col_name for tp in tps]
        self.data['df'].dropna(subset=cols, how='any', inplace=True)
        self.data['df'] = self.data['df'][self.data['df'] != 0].dropna(subset=cols)
        return self.data['df']

    def bin_1d(self, subs, coord, bin_equi1d=None, xbsize=None): 
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

    def bin_2d(self, subs, xcoord, ycoord, bin_equi2d=None, xbsize=None, ybsize=None, df=None): 
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
               xbsize=None, ybsize=None, zbsize=None, df=None): 
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

    def bin_1d_seasonal(self, subs, coord, bin_equi1d=None, xbsize=None) -> dict:
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
                     xbsize=None, ybsize=None) -> dict:
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

    def plot_1d_gradient(self, subs, coord, bin_attr='vmean', **kwargs):
        """ Plot gradient per season onto one plot. """
        bin_dict = self.bin_1d_seasonal(subs, coord)
        
        fig, ax = plt.subplots(dpi=200, figsize=(6,5))
        outline = mpe.withStroke(linewidth=2, foreground='white')

        for s in set(self.df['season'].tolist()):
            vmean = bin_dict[s].vmean
            y = bin_dict[s].xintm
            ax.plot(vmean, y, '-', marker='o', c=dcts.dict_season()[f'color_{s}'],
                      label=dcts.dict_season()[f'name_{s}'],
                      path_effects=[outline])

            if s==3: 
                ax.set_yticks(y if not coord.rel_to_tp else [0]+y)
                print(y)
        
        ax.set_ylabel(dcts.make_coord_label(coord))
        ax.set_xlabel(dcts.make_subs_label(subs))

        xmin = np.nanmin([np.nanmin(bin_inst.vmean) for bin_inst in bin_dict.values()])
        xmax = np.nanmax([np.nanmax(bin_inst.vmean) for bin_inst in bin_dict.values()])

        if coord.rel_to_tp: 
            ax.hlines(0, xmin, xmax, ls='dashed', color='k', lw=1)

        # ax.set_yticks([0] + bin_dict[1].xintm)
        # ax.set_yticks(y, minor=True)
        # print(y)
        
        if coord.vcoord in ['mxr', 'p'] and not coord.rel_to_tp: 
            ax.invert_yaxis()
        if coord.vcoord=='p': 
            ax.set_yscale('symlog' if coord.rel_to_tp else 'log')

        # ax.set_yticks(np.concatenate([y[:-1], y[:-1] * 3]), minor=True)

        # y = np.exp(np.linspace(min(y),max(y),1000))
        # ax.tick_params(axis='y', which='minor')
        
        # y_minor = ticker.LogLocator(base = 10.0, subs = np.arange(min(y), max(y)), numticks = 10)
        # ax.yaxis.set_minor_locator(y_minor)
        # ax.yaxis.set_minor_formatter(ticker.NullFormatter())
        
        # ax.yaxis.set_minor_formatter(FormatStrFormatter("%.1f"))

        # ax.set_yticks(minor=True, ticks = np.arange())

        ax.grid('both', ls='dashed', lw=0.5)
        ax.legend()

    def make_bar_plot(self, subs, xcoord, tp, bin_attr, percent_deviation=False, **kwargs) -> tuple: 
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
        
        bin_description = f'{self.glob_obj.grid_size}°' + ('latitude' if xcoord.hcoord=='lat' else 'longitude') + ' bins'
        
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
        
        fig.suptitle(f'{description} of {dcts.make_subs_label(subs)}')
        fig.subplots_adjust(top=0.85)
        ax_t.set_title('Troposphere', fontsize=9, loc='right')
        ax_s.set_title('Stratosphere', fontsize=9, loc='left')
        
        tropo_bar_vals = []
        strato_bar_vals = []
        bar_labels = []
        
        for i, tp in enumerate(tools.minimise_tps(dcts.get_coordinates(tp_def='not_nan'))):
            t_av, s_av = self.make_bar_plot(subs, xcoord, tp, bin_attr, 
                                            percent_deviation, **kwargs)

            tropo_bar_vals.append(t_av)
            strato_bar_vals.append(s_av)
            
            bar_labels.append(dcts.make_coord_label(tp, True))
            
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

    def matrix_plot_stdev_subs(self, substance, note='', minimise_tps=True,
                               atm_layer='both', savefig=False) -> (np.array, np.array):
        """
        Create matrix plot showing variability per latitude bin per tropopause definition
    
        Parameters:
            glob_obj (GlobalObject): Contains the data in self.df
            key short_name (str): Substance short name to show, e.g. 'n2o'

        Returns:
            tropospheric, stratospheric standard deviations within each bin as list for each tp coordinate
        """
        tps = [tp for tp in dcts.get_coordinates(tp_def='not_nan')
               if 'tropo_'+tp.col_name in self.glob_obj.df_sorted.columns]
    
        if minimise_tps:
            tps = tools.minimise_tps(tps)
    
        lat_bmin, lat_bmax = 30, 90 # np.nanmin(lat), np.nanmax(lat)
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
        tp_labels = [dcts.make_coord_label(tp, True)+'\n' for tp in tps]
        xticks = np.arange(lat_bmin, lat_bmax+self.glob_obj.grid_size, self.glob_obj.grid_size)
    
        fig = plt.figure(dpi=200, figsize=(lat_bci.nx*0.825, len(tps)*2))
    
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
            vmin, vmax = dcts.get_vlims(substance.short_name, 'vstdv', 'strato')
        except KeyError: 
            vmin, vmax = np.nanmin(strato_stdevs), np.nanmax(strato_stdevs)
            
        norm = Normalize(vmin, vmax)  # normalise color map to set limits
        strato_cmap = dcts.dict_colors()['vstdv_strato'] # plt.cm.BuPu  # create colormap
        ax_strato1.set_title(f'Stratospheric variability of {dcts.make_subs_label(substance)}{note}', fontsize=14)
    
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
        cbar.set_label(f'Standard deviation of {dcts.make_subs_label(substance, name_only=True)} within bin [{substance.unit}]')
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
            vmin, vmax = dcts.get_vlims(substance.short_name, 'vstdv', 'tropo')
        except KeyError: 
            vmin, vmax = np.nanmin(strato_stdevs), np.nanmax(strato_stdevs)
        norm = Normalize(vmin, vmax)  # normalise color map to set limits
        tropo_cmap = dcts.dict_colors()['vstdv_tropo'] # cmr.get_sub_cmap('YlOrBr', 0, 0.75) # create colormap
        ax_tropo1.set_title(f'Tropospheric variability of {dcts.make_subs_label(substance)}{note}', fontsize=14)
    
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
        cbar.set_label(f'Standard deviation of {dcts.make_subs_label(substance, name_only=True)} within bin [{substance.unit}]')
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
        substances = [s for s in dcts.get_substances(**subs_kwargs, detr=True)
                      if (s.col_name in self.glob_obj.df.columns
                          and not s.col_name.startswith('d_'))]
    
        for subs in substances:
            self.matrix_plot_stdev_subs(subs, note=note, minimise_tps=minimise_tps,
                                       atm_layer=atm_layer, savefig=savefig)

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
            fig.suptitle(f'{bin_attr} of binned global {data_type} mixing ratios of {dcts.make_subs_label(subs)}',
                         fontsize=25)
            plt.subplots_adjust(top=0.96)

        world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))

        xcoord = dcts.get_coord(col_name='geometry.x')
        ycoord = dcts.get_coord(col_name='geometry.y')
        xbsize, ybsize = self.get_bsize(subs, xcoord, ycoord)
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
            cbar.ax.set_xlabel(dcts.make_subs_label(subs), fontsize=15)
        
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

        binned_seasonal = self.bin_2d_seasonal(subs, xcoord, ycoord)

        if not any(bin_attr in bin2d_inst.__dict__ for bin2d_inst in binned_seasonal.values()):
            raise KeyError(f'\'{bin_attr}\' is not a valid attribute of Bin2D objects.')

        vlims = self.get_vlimit(subs, bin_attr)
        xlims = self.get_coord_lims(xcoord, 'x')
        ylims = self.get_coord_lims(ycoord, 'y')
        
        # vlims, xlims, ylims = self.get_limits(subs, xcoord, ycoord, bin_attr)
        norm = Normalize(*vlims)
        fig, axs = plt.subplots(2, 2, dpi=250, figsize=(8,9),
                                sharey=True, sharex=True)

        fig.subplots_adjust(top = 1.1)

        data_title = 'Mixing ratio' if bin_attr=='vmean' else 'Varibility'
        # fig.suptitle(f'{data_title} of {dcts.make_subs_label(subs)}', y=0.95)

        for season, ax in zip([1,2,3,4], axs.flatten()):
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
        cbar.ax.set_xlabel(data_title+' of '+dcts.make_subs_label(subs), 
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
        permutations = list(itertools.product(self.substances,
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

        if xcoord.col_name == 'geometry.y': # latitude
            x = df.geometry.y
        else:
            x = np.array(df[xcoord.col_name])
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
        # fig.suptitle(f'{data_title} of {dcts.make_subs_label(subs)}', y=0.95)

        cmap = dcts.dict_colors()[bin_attr]

        img = self.single_2d_plot(ax, bin2d_inst, bin_attr, xcoord, ycoord, 
                           cmap, norm, xlims, ylims, **kwargs)

        fig.subplots_adjust(right=0.9)
        fig.tight_layout(pad=2.5)

        cbar = fig.colorbar(img, ax = ax, aspect=30, pad=0.09, orientation='horizontal')
        cbar.ax.set_xlabel(data_title+' of '+dcts.make_subs_label(subs))

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

            ax.set_xlabel(dcts.make_coord_label(xcoord1))
            ax.set_ylabel(dcts.make_coord_label(ycoord1))

            if kwargs.get('note'):
                ax.text(**dcts.note_dict(ax, s=kwargs.get('note')))

        fig.subplots_adjust(right=0.9)
        fig.tight_layout(pad=2.5)

        cbar = fig.colorbar(img, ax = axs.ravel().tolist(), aspect=30,
                            pad=0.09, orientation='horizontal')
        cbar.ax.set_xlabel(
            f'{dcts.make_subs_label(subs1)} {dcts.make_coord_label(xcoord1)} {dcts.make_coord_label(ycoord1)} \n vs.\n\
{dcts.make_subs_label(subs2)} {dcts.make_coord_label(xcoord2)} {dcts.make_coord_label(ycoord2)}')
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

        ax.set_xlabel(dcts.make_coord_label(xcoord))
        ax.set_ylabel(dcts.make_coord_label(ycoord))

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

    def stratosphere_map(self, subs, tp, bin_attr): 
        """ Plot (first two ?) stratospheric bins on a lon-lat binned map. """
        df = self.glob_obj.sel_strato(**tp.__dict__).df
        # df = self.glob_obj.sel_tropo(**tp.__dict__).df

        fig, ax = plt.subplots(figsize=(9,9))
        ax.set_title(dcts.make_coord_label(tp, True))
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

