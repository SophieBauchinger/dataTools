# -*- coding: utf-8 -*-
""" Visualisation Mixins for binned aircraft campaign data / wrt. local tropopause. 

@Author: Sophie Bauchinger, IAU
@Date: Tue Jun  6 13:59:31 2023
 
class SimpleBinPlotter

class BinPlotterBaseMixin
> class BinPlotter1DMixin
> class BinPlotter2DMixin
> class BinPlotter3DMixin
>> class BinPlotterMixin
"""
#%% Imports
from abc import ABCMeta, abstractmethod
import geopandas
import math
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines
import matplotlib.patheffects as mpe
# from matplotlib.patches import Patch
import matplotlib.ticker as ticker
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from mpl_toolkits.axes_grid1 import AxesGrid

import numpy as np
import pandas as pd
from PIL import Image
import io
import itertools
import warnings

import toolpac.calc.binprocessor as bp # type: ignore

import dataTools.dictionaries as dcts
from dataTools import tools

warnings.filterwarnings(action='ignore', message='Mean of empty slice')

#%%% BinPlotter classes for multiple dimensionalities
class SimpleBinPlotter: 
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

class BinPlotterBaseMixin:
    """ Structure for binning & plotting classes for any choice of x/y/z.

    Methods:
        set_kwargs(**kwargs)
        get_vlimit(subs, bin_attr)
        get_coord_lims(coord, xyz)
        _get_bsize(coord, xyz)
    """

    def __init__(self, filter_tps = None, **kwargs):
        """ Initialise class instances. 
        Paramters:
            filter_tps (List[Coordinate]): Reduce dataset to only points that all TP coords share
         
            key xbsize / ybsize (float)
            key ybsize (float)
            key vlims / xlims / ylims (Tuple[float])
            """
        super().__init__(**kwargs)

        if filter_tps: 
            self.remove_non_shared_indices(filter_tps, inplace=True)

        self.data['df']['season'] = tools.make_season(self.data['df'].index.month)
        self.outline = mpe.withStroke(linewidth=2, foreground='white')
        self.set_kwargs(**kwargs)

    def set_kwargs(self, **kwargs): 
        """ Set shared kwargs used by various class methods. """
        self._kwargs = kwargs
        return self

    def get_vlimit(self, subs: dcts.Substance, bin_attr: str) -> tuple: 
        """ Get colormap limits for given substance and bin attribute. """
        if hasattr(self, '_kwargs'): 
            if 'vlims' in self._kwargs:
                return self._kwargs.get('vlims')

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
        """ Returns default maximum / minimum boundary of the coordinate for plotting. """

        if coord.hcoord == 'lat': 
            return (-90, 90)
        elif coord.hcoord == 'lon': 
            return (-180, 180)

        if xyz=='x': 
            if 'xlims' in self._kwargs: 
                lims = self._kwargs.get('xlims')
            else: 
                lims = (-90, 90)
        elif xyz=='y': 
            if 'ylims' in self._kwargs: 
                lims = self._kwargs.get('ylims')
            else: 
                lims = (np.floor(np.nanmin(self.df[coord.col_name])),
                         np.ceil(np.nanmax(self.df[coord.col_name])))
        else: 
            lims = (np.floor(np.nanmin(self.df[coord.col_name])),
                     np.ceil(np.nanmax(self.df[coord.col_name])))
        return lims

    def _get_bsize(self, coord, xyz: str = None) -> float: 
        """ Get bin size for given coordinate. """
        bsize = None

        if hasattr(self, '_kwargs'):
            if xyz=='x' and 'xbsize' in self._kwargs: 
                bsize = self._kwargs.get('xbsize')
            elif xyz=='y' and 'ybsize' in self._kwargs: 
                bsize = self._kwargs.get('ybsize')
            elif xyz=='z' and 'zbsize' in self._kwargs: 
                bsize = self._kwargs.get('zbsize')
            if bsize: 
                return bsize 

        bsize = coord.get_bsize()# dcts.get_default_bsize(coord.hcoord)
        if not bsize and xyz=='x': 
            bsize = 10
        if not bsize:    
            lims = self.get_coord_lims(coord, xyz)
            bsize = 5 * ( np.ceil((lims[1]-lims[0])/10) / 5 )
            if (lims[1]-lims[0])/10<1: 
                bsize=0.5
        return bsize

class BinPlotter1DMixin(BinPlotterBaseMixin):
    """ Single dimensional binning & plotting. 
    
    Methods: 
        flight_height_histogram
        overlapping_histograms
        plot_vertial_profile_variability_comparison
        plot_1d_seasonal_gradient
        plot_1d_gradient
        plot_1d_seasonal_gradient_with_vstdv
        stdv_rms_non_pt
        calc_bin_avs
        plot_bar_plots
        matrix_plot_stdev_subs
        matrix_plot_stdev
    """

    def flight_height_histogram(self, tp, alpha: float = 0.7, ax=None, **kwargs): 
        """ Make a histogram of the number of datapoints for each tp bin. """
        if ax is None: 
            fig, ax = plt.subplots(dpi=250, figsize=(6,4))
            ax.set_ylabel(tp.label())
        
        data = self.df[tp.col_name].dropna()

        ax.set_title(f'Distribution of {self.source} measurements')
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
        fig, ax = plt.subplots(dpi=500, figsize=(5,6))
        outline = mpe.withStroke(linewidth=2, foreground='white')
        
        for i, tp in enumerate(tps):
            bin_dict = self.bin_1d_seasonal(subs, tp)
            
            ls = list(['--', '-.', ':', '-']*5)[i]
            marker = list(['o', 'X', 'd', 'p', '*', '+', '1']*5)[i]

            for s in seasons: 
                if s not in bin_dict.keys(): continue
                vdata = getattr(bin_dict[s], bin_attr)
                if rel: vdata = vdata / bin_dict[s].vmean * 100
                y = bin_dict[s].xintm

                ax.plot(vdata, y, 
                        # ls=ls,
                        c=dcts.dict_season()[f'color_{s}'],
                        path_effects=[outline], 
                        zorder=2, lw=1.5)
                
                ax.scatter(vdata, y,
                           marker=marker, 
                           label=tp.label(True) if s==1 else None,
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
        if all(tp.vcoord=='pt' for tp in tps): 
            ax.set_ylim(-70, 70)

    def plot_1d_seasonal_gradient(self, subs, coord, 
                                  bin_attr: str = 'vmean', 
                                  add_stdv: bool = False, 
                                  **kwargs):
        """ Plot gradient per season onto one plot. """
        big = kwargs.pop('big') if 'big' in kwargs else False
        bin_dict = self.bin_1d_seasonal(subs, coord, **kwargs)
        
        fig, ax = plt.subplots(dpi=500, 
                               figsize= (6,4) if not big else (3,4))
        
    
        if coord.vcoord=='pt' and coord.rel_to_tp: 
            ax.set_yticks(np.arange(-60, 75, 20) + [0])



        for s in bin_dict.keys():
            self.plot_1d_gradient(ax, s, bin_dict[s], bin_attr, add_stdv)
            
        ax.set_title(coord.label(filter_label=True))
        ax.set_ylabel(coord.label(coord_only = True))

        if bin_attr=='vmean':
            ax.set_xlabel(subs.label())
        elif bin_attr=='vstdv': 
            ax.set_xlabel('Relative variability of '+subs.label(name_only=True))

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

        if not big: 
            ax.legend(loc='lower left')
        ax.grid('both', ls='dashed', lw=0.5)
        ax.set_axisbelow(True)
        
        if coord.rel_to_tp: 
            tools.add_zero_line(ax)

        return bin_dict, fig 

    def plot_1d_gradient(self, ax, s, bin_obj,
                         bin_attr: str = 'vmean', 
                         add_stdv: bool = False):
        """ Create scatter/line plot for the given binned parameter for a single season. """
        
        outline = mpe.withStroke(linewidth=2, foreground='white')
        
        color = dcts.dict_season()[f'color_{s}']
        label = dcts.dict_season()[f'name_{s}']

        vdata = getattr(bin_obj, bin_attr)
        y = bin_obj.xintm

        if bin_attr=='vmean': 
            if add_stdv: 
                ax_stdv = ax.twiny()
                ax_stdv.set_xlim(0, (6.2-5.1))
                ax_stdv.plot(bin_obj.vstdv, y, 
                            c = color, label = label,
                            linewidth=1, ls='dashed',
                            alpha=0.5,
                            path_effects = [outline], zorder = 2)
                    
                ax_stdv.tick_params(labelcolor='grey')
            (_, caps, _) = ax.errorbar(vdata, y, 
                            xerr = bin_obj.vstdv, 
                            c = color, lw = 0.8, alpha=0.7, 
                            # path_effects=[outline],
                            capsize = 2, zorder = 1)
            for cap in caps: 
                cap.set_markeredgewidth(1)
                cap.set(alpha=1, zorder=20)

        marker = 'd'
        ax.plot(vdata, y, 
                    marker=marker,
                    c = color, label = label,
                    linewidth=2,# if not kwargs.get('big') else 3,
                    path_effects = [outline], zorder = 2)

        ax.scatter(vdata, y, 
                    marker=marker,
                    c = color, zorder = 3)

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

    def calc_bin_avs(self, data: pd.DataFrame, 
                     subs, xcoord, bin_attr: str,
                     **kwargs) -> float: 
        """ Data is 1D-binned along xcoord and the average bin_atrr value is calculated for the dataset. 
        
        Parameters: 
            data (pd.DataFrame): Data to be binned
            subs (dcts.Substance): Substance instance to be binned
            xcoord (dcts.Coordinate): Coordinate instance along which to bin
            bin_attr (str): e.g. vmean/vstdv/rvstd, binned output to be averaged
            
            key xbsize (float): Bin-size along x-axis
            key bin_equi1d (bp.Bin_equi1d, bp.Bin_notequi1d): Binning structure
        
        Returns the averaged value of bin_attr for the whole dataset. 
        """

        if not isinstance(kwargs.get('xbsize'), float):
            xbsize = xcoord.get_bsize()

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

        if bin_attr=='vcount':
            total = len(data[data[subs.col_name].notna()])
            return total

        binned_1d = bp.Simple_bin_1d(np.array(data[subs.col_name]), x[x.index.isin(data.index)],
                                      bin_equi1d, count_limit=self.count_limit if not bin_attr=='vcount' else 1)
        values = getattr(binned_1d, bin_attr)
        nans = np.isnan(values)
        av = np.average(values[~nans], weights = binned_1d.vcount[~nans])

        return av

    def plot_bar_plots(self, subs, xcoord, bin_attr, **kwargs):
        """ Plot a bar plot to compare average bin_attr values across tropopause definitions. 
        1. Data is separated into stratospheric / tropospheric values for each TP definiton
        2. Data is binned along the x-axis as defined by xcoord
        3. The average value of bin_attr for the specified substance is calculated for each atm_layer
        
        Parameters: 
            subs (dcts.Substance): Substance instance indicating data to be binned
            xcoord (dcts.Coordinate): Coordinate to use for 1D-binning
            bin_attr (str): e.g. vmean/vstdv/rvstd, binned output to be averaged

            key xbsize (float): Bin-size along x-axis
            key bin_equi1d (bp.Bin_equi1d, bp.Bin_notequi1d): Binning structure
            key tps ([dcts.Coordinate]): specify tropopause definitions to display 
        """
        tps = self.tp_coords() if not kwargs.get('tps') else kwargs.get('tps')
        
        #TODO shared_indices
        fig, (ax_t, ax_label, ax_s) = plt.subplots(1, 3, dpi=200, 
                                         figsize=(9,4), sharey=True)
        
        bin_description = f'{self.grid_size}°' \
            + ('latitude' if xcoord.hcoord=='lat' else ('longitude' if xcoord.hcoord=='lon' else 'HUH')) \
                + ' bins'
        description_dict = dict(
            vmean = f'Average mixing ratio in {bin_description}',
            vstdv = f'Variability in {bin_description}',
            vcount ='Total number of datapoints',
            rvstd = 'Relative variability', )
        description = description_dict.get(bin_attr)
        if not description: 
            description = bin_attr
        
        fig.suptitle(f'{description} of {subs.label()}' if not bin_attr=='vcount' else description)
        fig.subplots_adjust(top=0.85)
        ax_t.set_title('Troposphere', fontsize=9, loc='right')
        ax_s.set_title('Stratosphere', fontsize=9, loc='left')
        
        tropo_bar_vals = []
        strato_bar_vals = []
        bar_labels = []
        
        self.create_df_sorted() # create df_sorted
        shared_indices = self.get_shared_indices(tps)

        for i, tp in enumerate(tps):
            tropo_data = self.sel_tropo(**tp.__dict__).df
            strato_data = self.sel_strato(**tp.__dict__).df
            
            if kwargs.get('shared_index'): 
                tropo_data[tropo_data.index.isin(shared_indices)]
                strato_data[strato_data.index.isin(shared_indices)]

            t_av = self.calc_bin_avs(tropo_data,
                                     subs, xcoord, bin_attr, **kwargs)
            s_av = self.calc_bin_avs(strato_data,
                                     subs, xcoord, bin_attr, **kwargs)

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
        minimum = np.nanmin(tropo_bar_vals+strato_bar_vals) if bin_attr != 'rvstd' else 0
        for decimal_place in [4,3,2,1,0]:
            if all(i>np.round(minimum, decimal_place) for i in tropo_bar_vals+strato_bar_vals): 
                minimum = np.round(minimum, decimal_place)
            else: 
                break
        padding = (maximum-minimum)/3 

        ax_t.set_xlim(maximum +padding , minimum-padding if not minimum-padding<0 else 0)
        ax_s.set_xlim(minimum-padding if not minimum-padding<0 else 0, maximum +padding)
        
        ax_t.grid('both', ls='dotted')
        ax_s.grid('both', ls='dotted')
        
        if not bin_attr=='vcount':
            ax_t.bar_label(t_bars, ['{0:.3f}'.format(t_val)
                                    for t_val in tropo_bar_vals], 
                           padding=2)
            ax_s.bar_label(s_bars, ['{0:.3f}'.format(s_val)
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

class BinPlotter2DMixin(BinPlotterBaseMixin): 
    """ Two-dimensional binning & plotting. 
    
    Methods: 
        calc_average(bin2d_inst, bin_attr)
        calc_ts_averages(bin2d_inst, bin_attr)
        yearly_maps(subs, bin_attr)
        seasonal_2d_plots(subs, xcoord, ycoord, bin_attr, cmap)
        plot_2d_mxr(subs, xcoord, ycoord)
        plot_2d_stdv(subs, xcoord, ycoord)
        plot_mixing_ratios()
        plot_stdv_subset()
        plot_total_2d(subs, xcoord, ycoord, bin_attr)
        plot_mxr_diff(params_1, params2)
        plot_differences()
        single_2d_plot(ax, bin2d_inst, bin_attr, xcoord, ycoord,
                       cmap, norm, xlims, ylims)
    """

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

    def yearly_maps(self, subs, bin_attr, **kwargs):
        #  subs, single_yr=None, c_pfx=None, years=None, detr=False):
        """ Create binned 2D plots for each available year on a grid. """
        
        nplots = len(self.years)
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
        
        vlims = kwargs.get('vlims')
        if vlims is None: vlims = self.get_vlimit(subs, bin_attr)
        norm = Normalize(*vlims)  # normalise color map to set limits
        
        for i, (ax, year) in enumerate(zip(grid, self.years)):
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
            df_year = self.sel_year(year).df
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
            cmap = plt.cm.viridis

        binned_seasonal = self.bin_2d_seasonal(subs, xcoord, ycoord, **kwargs)

        if not any(bin_attr in bin2d_inst.__dict__ for bin2d_inst in binned_seasonal.values()):
            raise KeyError(f'\'{bin_attr}\' is not a valid attribute of Bin2D objects.')

        vlims = kwargs.get('vlims')
        if vlims is None: vlims = self.get_vlimit(subs, bin_attr)
        xlims = self.get_coord_lims(xcoord, 'x')
        ylims = self.get_coord_lims(ycoord, 'y')
        
        # vlims, xlims, ylims = self.get_limits(subs, xcoord, ycoord, bin_attr)
        norm = Normalize(*vlims)
        fig, axs = plt.subplots(2, 2, dpi=100, figsize=(8,9),
                                sharey=True, sharex=True)

        fig.subplots_adjust(top = 1.1)

        data_title = 'Mixing ratio' if bin_attr=='vmean' else ('Varibility' + '(RSTD)' if bin_attr=='rvstd' else '')
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
        cbar.ax.set_xlabel(data_title +' of '+ subs.label(name_only=True) \
                           + f' [{subs.unit}]' if not bin_attr=='rvstd' else ' [\%]', 
                           # fontsize=13
                           )

        cbar_vals = cbar.get_ticks()
        cbar_vals = [vlims[0]] + cbar_vals[1:-1].tolist() + [vlims[1]]
        cbar.ax.tick_params(bottom=True, top=False)
        cbar.set_ticks(ticks = cbar_vals) #, labels=cbar_vals, ticklocation='bottom')

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
        
        vlims = kwargs.get('vlims')
        if vlims is None: vlims = self.get_vlimit(subs, bin_attr)
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

            bin_obj = binned_seasonal_1[season].binclassinstance

            img = ax.imshow(vmean.T, cmap = cmap, norm=norm,
                            aspect='auto', origin='lower',
                            extent=[bin_obj.xbmin, bin_obj.xbmax, bin_obj.ybmin, bin_obj.ybmax])
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
            f'{subs1.label()} {xcoord1.label()} {ycoord1.label()} \n vs.\n' + \
                f'{subs2.label()} {xcoord2.label()} {ycoord2.label()}')
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

        bin_obj = bin2d_inst.binclassinstance
        data = getattr(bin2d_inst, bin_attr) # atttribute: 'vmean', 'vstdv'

        img = ax.imshow(data.T,
                        cmap = cmap, norm=norm,
                        aspect='auto', origin='lower',
                        # if not ycoord.vcoord in ['p', 'mxr'] else 'upper',
                        extent=[bin_obj.xbmin, bin_obj.xbmax, bin_obj.ybmin, bin_obj.ybmax] 
                        # if not ycoord.vcoord in ['p', 'mxr'] else [bin_obj.xbmin, bin_obj.xbmax, bin_obj.ybmax, bin_obj.ybmin]
                        )

        ax.set_xlabel(xcoord.label())
        ax.set_ylabel(ycoord.label())

        ax.set_xlim(*xlims)
        #TODO with count_limit > 1, might not have data in all bins - get dynamic bins?
        ax.set_ylim(ylims[0] - bin_obj.ybsize*1.5, ylims[1] + bin_obj.ybsize*1.5)

        ax.set_xticks(np.arange(-90, 90+30, 30)) # stop+30 to include stop

        if bin_obj.ybmin < 0:
            # make sure 0 is included in ticks, evenly spaced away from 0
            ax.set_yticks(list(np.arange(0, abs(bin_obj.ybmin) + bin_obj.ybsize*3, bin_obj.ybsize*3) * -1)
                          + list(np.arange(0, bin_obj.ybmax + bin_obj.ybsize, bin_obj.ybsize*3)))
        else:
            ax.set_yticks(np.arange(bin_obj.ybmin, bin_obj.ybmax + bin_obj.ybsize*3, bin_obj.ybsize*3))
        ax.set_yticks(np.arange(bin_obj.ybmin, bin_obj.ybmax+bin_obj.ybsize, bin_obj.ybsize), minor=True)

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

    def three_sideplots_2d_binned(self, subs, zcoord, eql=False, 
                           bin_attr = 'vmean', **kwargs): 
        """ """
        # Create the figure outline 
        fig, axes =  self._three_sideplot_structure()
        
        fig.suptitle(zcoord.label(filter_label = True))
        fig.subplots_adjust(top = 0.8)
        
        ax_fig, ax_main, ax_upper, ax_right, ax_cube = axes

        # Define variables
        cmap = dcts.dict_colors()[bin_attr]
        norm = Normalize(*subs.vlims(bin_attr))
        
        xcoord = dcts.get_coord('geometry.x')
        ycoord = dcts.get_coord('geometry.y') if not eql else \
            self.get_coords(hcoord='eql', model='ERA5')[0]
        
        args = (subs, cmap, norm, bin_attr)
        
        if not eql:
            tools.add_world(ax_main) 

        def _make_right_plot(self, ax_right, ycoord, zcoord, subs, cmap, norm, bin_attr):
            """ Right-hand-side plot for 3D projections. 
            
            Axes: zcoord on the abscissa ('x-axis') and ycoord / latitude on the ordinate ('y-axis'). 

            """
            bin2d_inst = self.bin_2d(subs, zcoord, ycoord)
            bin_obj = bin2d_inst.binclassinstance
            data = getattr(bin2d_inst, bin_attr) # atttribute: 'vmean', 'vstdv'

            img = ax_right.imshow(data,
                            cmap = cmap, norm=norm,
                            aspect='auto', origin='lower',
                            extent=[bin_obj.xbmin, bin_obj.xbmax, 
                                    bin_obj.ybmin, bin_obj.ybmax] 
                            )
            ax_right.set_xlabel(zcoord.label(False, True))
            ax_right.set_ylabel(ycoord.label())
            ax_right.set_xlim(bin_obj.xbmin  - bin_obj.xbsize*1.5, bin_obj.xbmax + bin_obj.xbsize*1.5) # *self.get_coord_lims(zcoord))
            ax_right.set_ylim(-90, 90)

            ax_right.yaxis.set_label_position("right")
            ax_right.xaxis.set_label_position("bottom")

            ax_right.grid('both', lw=0.4, ls = '--')
            
            return img

        def _make_upper_plot(self, ax_upper, xcoord, zcoord, subs, cmap, norm, bin_attr): 
            """ Upper plot for 3D projections. 
            
            Axes: xcoord / longitude on the abscissa ('x-axis') and zcoord on the ordinate ('y-axis'). 
            """
            bin2d_inst = self.bin_2d(subs, xcoord, zcoord)
            bin_obj = bin2d_inst.binclassinstance
            data = getattr(bin2d_inst, bin_attr) # atttribute: 'vmean', 'vstdv'

            img = ax_upper.imshow(data.T,
                            cmap = cmap, norm=norm,
                            aspect='auto', origin='lower',
                            extent=[bin_obj.xbmin, bin_obj.xbmax, 
                                    bin_obj.ybmin, bin_obj.ybmax] 
                            )
            ax_upper.set_xlabel(xcoord.label())
            ax_upper.set_ylabel(zcoord.label(False, True))
            ax_upper.set_ylim(bin_obj.ybmin  - bin_obj.ybsize*1.5, bin_obj.ybmax + bin_obj.ybsize*1.5) # *self.get_coord_lims(zcoord))
            ax_upper.set_xlim(-180, 180)

            ax_upper.yaxis.set_label_position("left")
            ax_upper.xaxis.set_label_position("top")

            ax_upper.grid('both', lw=0.4, ls = '--')
            
            return img
        
        def _make_center_plot(self, ax_main, xcoord, ycoord, subs, cmap, norm, bin_attr): 
            """ Non-fancy Longitude-latitude binned 2D plot. 
            
            Note: This doesn't show anything related to tropopause coordinates, 
            so don't be tempted to use it for anything going forwards. Just sayin'. 
            """
            bin2d_inst = self.bin_2d(subs, xcoord, ycoord)
            bin_obj = bin2d_inst.binclassinstance
            data = getattr(bin2d_inst, bin_attr) # atttribute: 'vmean', 'vstdv'

            img = ax_main.imshow(data.T,
                            cmap = cmap, norm=norm,
                            aspect='auto', origin='lower',
                            extent=[bin_obj.xbmin, bin_obj.xbmax, 
                                    bin_obj.ybmin, bin_obj.ybmax] 
                            )
            ax_main.set_xlabel(xcoord.label())
            ax_main.set_ylabel(ycoord.label())
            ax_main.set_ylim(-90, 90)
            ax_main.set_xlim(-180, 180)

            ax_main.yaxis.set_label_position("left")
            ax_main.xaxis.set_label_position("bottom")

            ax_main.grid('both', lw=0.4, ls = '--')
            
            return img

        def add_cube(ax, 
                    abc_colors=('tab:blue', 'tab:orange', 'tab:green'), 
                    sides = (1,1,1)):
            """
            Plot a cube with individually colored edges.

            Parameters:
                ax (matplotlib 3D axis)
                abc_colors (List[str]): Colors of the 3 front edges of the cube (a,b,c).
                sides (List[float]): Length of a,b,c sides of the object
    
            """
            edge_colors = [
                'k', 'k', 'w', 'w', 
                abc_colors[0], # a
                abc_colors[1], # b
                'k', 'k', 'k', 
                abc_colors[2], # c
                'k', 'w']

            a, b, c = sides

            vertices = np.array([
                [0, 0, 0], [a, 0, 0], [a, b, 0], [0, b, 0],
                [0, 0, c], [a, 0, c], [a, b, c], [0, b, c]
                ])
            
            edges = [
                [vertices[j] for j in [0, 1]],   # Bottom edges
                [vertices[j] for j in [1, 2]],
                [vertices[j] for j in [2, 3]],
                [vertices[j] for j in [3, 0]],
                [vertices[j] for j in [4, 5]],   # Top edges
                [vertices[j] for j in [5, 6]],
                [vertices[j] for j in [6, 7]],
                [vertices[j] for j in [7, 4]],
                [vertices[j] for j in [0, 4]],   # Vertical edges
                [vertices[j] for j in [1, 5]],
                [vertices[j] for j in [2, 6]],
                [vertices[j] for j in [3, 7]]
            ]

            # Create a Line3DCollection for each edge with specified colors
            for i, edge in enumerate(edges):
                line = Line3DCollection([edge], colors=edge_colors[i], linewidths=2)
                ax.add_collection3d(line)
            ax.axis('off')

        img = _make_right_plot(self, ax_right, ycoord, zcoord, *args)
        _ = _make_upper_plot(self, ax_upper, xcoord, zcoord, *args)
        _ = _make_center_plot(self, ax_main, xcoord, ycoord, *args)
        _ = add_cube(ax_cube, sides = (1, 0.5, 0.4))

        # longitude / a
        for spine in (ax_main.spines['right'], 
                      ax_main.spines['left'], 
                      ax_right.spines['right'], 
                      ax_right.spines['left']):
            spine.set_color('tab:orange')
        
        # latitude / b
        for spine in (ax_main.spines['top'],
                      ax_main.spines['bottom'], 
                      ax_upper.spines['top'],
                      ax_upper.spines['bottom']): 
            spine.set_color('tab:blue')
        
        # zcoord / c
        for spine in (ax_upper.spines['right'], 
                      ax_upper.spines['left'], 
                      ax_right.spines['top'], 
                      ax_right.spines['bottom']): 
            spine.set_color('tab:green')

        [ax.spines[i].set_linewidth(1.5) for ax in (ax_main, ax_upper, ax_right) \
            for i in ax.spines]
        
        plt.colorbar(img, 
                     ax = (ax_main, ax_upper, ax_right, ax_cube),
                     fraction = 0.05,
                     orientation = 'horizontal', 
                     label = subs.label(bin_attr=bin_attr),
                     )
        plt.show()

class BinPlotter3DMixin(BinPlotterBaseMixin): 
    """ Three-dimensional binning & plotting. 
    
    Methods: 
        z_crossection(subs, tp, bin_attr, save_gif_path)
        stratosphere_map(subs, tp, bin_attr)
        matrix_plot_3d_stdev_subs(substance, note, tps, save_fig)
        matrix_plot_stdev(note, atm_layer, savefig)
    """

    def calc_Bin3DFitted_dict(self, *args, **kwargs): # NotImplemented
        return NotImplementedError

    def Bin3DFitted_dict(self, subs, zcoord, eql, **kwargs
                         ) -> tuple[dict[tools.Bin3DFitted]]:
        """ Get Bin3DFitted instances for tropospheric data sorted with each tps. 
        
        Parameters: 
            subs (dcts.Substance)
            zcoord (dcts.Coordinate): Vertical coordinate used for binning
            eql (bool): Use equivalent latitude instead of latitude
            
            key bci_3d (bp.Bin_equi3d, bp.Bin_notequi3d): 3D-Binning structure
            key *(xbsize, ybsize, zbsize) (float): Binsize for x/y/z dimensions. Optional

        Returns stratospheric, tropospheric 3D-bin dictionaries keyed by tropopause definition.
        """
        if hasattr(self, 'temp_tropo_3d') and hasattr(self, 'temp_strato_3d') and not kwargs.get('calc'):
            return self.temp_strato_3d, self.temp_tropo_3d
        
        tropo_Bin3D_dict, strato_Bin3D_dict = {}, {}
        for tp in self.tps:
            tropo_plotter = self.sel_tropo(tp)
            tropo_Bin3D_dict[tp.col_name] = tropo_plotter.bin_3d(
                subs, zcoord, eql=eql, **kwargs)
            
            strato_plotter = self.sel_strato(tp)
            strato_Bin3D_dict[tp.col_name] = strato_plotter.bin_3d(
                subs, zcoord, eql=eql, **kwargs)
            
        self.temp_strato_3d = strato_Bin3D_dict
        self.temp_tropo_3d = tropo_Bin3D_dict

        return strato_Bin3D_dict, tropo_Bin3D_dict

    def get_data_3d_dicts(self, subs, zcoord, eql, bin_attr,
                          **kwargs) -> tuple[dict, dict]: 
        """ Extract specific attributes from Bin3D dictionaries. 
        Returns data dictionaries such that {tp_col : np.ndarray} """
        strato_Bin3D_dict, tropo_Bin3D_dict = self.Bin3DFitted_dict(
            subs, zcoord, eql, **kwargs)

        strato_attr_dict = {k:getattr(v, bin_attr) for k,v in strato_Bin3D_dict.items()}
        tropo_attr_dict = {k:getattr(v, bin_attr) for k,v in tropo_Bin3D_dict.items()}
        
        if bin_attr == 'rvstd': 
            # Multiply everything by 100 to get spercentages
            strato_attr_dict = {k:v*100 for k,v in strato_attr_dict.items()}
            tropo_attr_dict = {k:v*100 for k,v in tropo_attr_dict.items()}
        
        return strato_attr_dict, tropo_attr_dict

    def get_lognorm_stats_df(self, Bin3D_dict: dict, lognorm_attr: str) -> pd.DataFrame: 
        """ Create combined lognorm-fit statistics dataframe for all tps. 
        
        Parameters: 
            Bin3D_dict (dict[tools.Bin3DFitted]): Binned data incl. lognorm fits
            lognorm_attr (str): vmean_fit / vsdtv_fit / rvstd_fit
        """
        return pd.DataFrame({k:getattr(v, lognorm_attr).stats() for k,v in Bin3D_dict.items()})

    def three_sideplots_3d_binned(self, subs, zcoord, eql=False, 
                           bin_attr = 'vmean', **kwargs): 
        """ Plot 3d-binned color-coded plots on 3 projections. """
        
        # Make the figure
        fig, axs = self._three_sideplot_structure()
        (ax_fig, ax_main, ax_upper, ax_right, ax_cube) = axs
        if not eql: 
            tools.add_world(ax_main)

        # Get the data
        binned_data = self.bin_3d(subs, zcoord, eql=eql, **kwargs)
        data3d = getattr(binned_data, bin_attr)

        cmap = dcts.dict_colors()[bin_attr]
        norm = Normalize(*subs.vlims(bin_attr))
        
        norm = Normalize(np.nanmin(data3d), np.nanmax(data3d))
              
        # --- xy mean (av. along z, 2) - main --- # 
        img = ax_main.imshow(
            np.nanmean(data3d, axis = 2).T,
            cmap = cmap, norm=norm,
            aspect='auto', origin='lower',
            # if not ycoord.vcoord in ['p', 'mxr'] else 'upper',
            extent=[binned_data.xbmin, binned_data.xbmax, 
                    binned_data.ybmin, binned_data.ybmax],
            zorder = 1)

        # --- yz mean (av. along x, 0) - right --- #
        ax_right.imshow(
            np.nanmean(data3d, axis = 0),
            cmap = cmap, norm=norm,
            aspect='auto', origin='lower',
            extent=[binned_data.zbmin, binned_data.zbmax, 
                    binned_data.ybmin, binned_data.ybmax],
            )

        # --- xz mean (av. along y, 1) - upper --- #
        ax_upper.imshow(
            np.nanmean(data3d, axis = 1).T,
            cmap = cmap, norm=norm,
            aspect='auto', origin='lower',
            extent=[binned_data.xbmin, binned_data.xbmax, 
                    binned_data.zbmin, binned_data.zbmax],
            )
        
        self._three_sideplot_labels(fig, axs, zcoord, eql)
        
        plt.colorbar(img,
            ax = (ax_main, ax_upper, ax_right, ax_cube),
            fraction = 0.05,
            orientation = 'horizontal', 
            label = subs.label(bin_attr = bin_attr),
            )

        plt.show()

    def lil_histogram_3d_helper(self, data_3d_dict, figaxs=None): 
        """ Create histograms for all keys. """
        if figaxs is None:
            fig, axs = self._make_two_column_axs(self.tps)
        else: 
            fig, axs = figaxs

        colors_20c = plt.cm.tab20c.colors
        colors = colors_20c[:2] + colors_20c[4:7] + colors_20c[8:9]

        fig.set_size_inches(7, 10)
        
        # Get bin limits
        lim_min, lim_max = np.nan, np.nan
        for data3d in data_3d_dict.values(): 
            lim_min = np.nanmin([lim_min] + list(data3d.flatten()))
            lim_max = np.nanmax([lim_max] + list(data3d.flatten()))
        
        for ax, tp_col, c in zip(axs.flatten(), 
                                 data_3d_dict, 
                                 colors):
            ax.set_title(dcts.get_coord(tp_col).label(filter_label=True))
            ax.set_xlabel('Frequency [#]')
            
            data3d = data_3d_dict[tp_col]

            data_flat = data3d.flatten()
            data_flat = data_flat[~np.isnan(data_flat)]
            # data_flat = data_flat[data_flat != 0.0]

            ax.hist(data_flat, 
                    bins = 30, range = (lim_min, lim_max), 
                    orientation = 'horizontal',
                    edgecolor = 'black', alpha=0.7, color=c)
            
            ax.set_xscale('log')
            
            ax.grid(axis='x', ls ='dashed', lw = 1, color='grey', zorder=0)
        
        fig.tight_layout()
        fig.subplots_adjust(top = 0.85)
        
        return fig, axs
        
    def histogram_for_3d_bins_single_atm_layer_sorted(self, subs, zcoord, 
                                                      eql=False, bin_attr = 'vstdv', 
                                                      strato_3d_dict = None, tropo_3d_dict = None, 
                                                      **kwargs): 
        """ Plotting basic histograms of bin_attr for Stratos & Tropos on separate figures. """
        if strato_3d_dict is None or tropo_3d_dict is None:
            strato_3d_dict, tropo_3d_dict = self.get_data_3d_dicts(
                subs, zcoord, eql, bin_attr, **kwargs)
        
        # Stratospheric           
        fig_s, axs_s = self.lil_histogram_3d_helper(strato_3d_dict)
        fig_s.subplots_adjust(top = 0.8)
        fig_s.suptitle(f'Stratospheric 3D-binned distribution in {zcoord.label(coord_only=True)}')
        for ax in axs_s.flatten(): 
            ax.set_ylabel(subs.label(bin_attr=bin_attr))
        
        # Tropospheric 
        fig_t, axs_t = self.lil_histogram_3d_helper(tropo_3d_dict)
        fig_t.subplots_adjust(top = 0.9)
        fig_t.suptitle(f'Tropospheric 3D-binned distribution in {zcoord.label(coord_only=True)}')
        for ax in axs_t.flatten(): 
            ax.set_ylabel(subs.label(bin_attr=bin_attr))
          
    def fancy_histogram_plots_nested(self, subs, zcoord, eql=False, bin_attr='vstdv', 
                                     tropo_3d_dict = None, strato_3d_dict = None, 
                                     xscale = 'linear',
                                     fig_kwargs = {}, **kwargs): 
        """ Comparison plot for tropopause definition substance histograms + lognorm fit. 

        Parameters: 
            subs (dcts.Substance): 
                Substance to be evaluated
            zcoord (dcts.Coordinate): 
                Vertical coordinate to use for binning

            eql (bool): Default False
                Use equivalent latitude instead of latitude. 
            bin_attr (str): Default 'vsdtv'
                Which bin-box quantity to plot. 
            tropo_3d_dict (dict[np.ndarray]): Default None
                Precalculated tropospheric data per tropopause. 
            strato_3d_dict (dct[np.ndarray]): Default None
                Precalculated stratospheric data per tropopause. 
            xscale (str): Default 'linear' 
                x-axis scaling (e.g. linear/log/symlog). 
                
            key show_stats (bool): 
                Adds mode and sigma values to the plot. 
        """
                      
        # Get the 3D binned data
        if strato_3d_dict is None or tropo_3d_dict is None:
            strato_3d_dict, tropo_3d_dict = self.get_data_3d_dicts(
                subs, zcoord, eql, bin_attr, **kwargs)
        
        # Create the figure 
        fig, main_axes, sub_ax_arr = self._nested_subplots_two_column_axs(self.tps, **fig_kwargs)
        self._adjust_labels_ticks(sub_ax_arr)
        
        gs = main_axes.flat[0].get_gridspec()
        gs.update(wspace = 0.3)

        tropo_axs = sub_ax_arr[:,:,0].flat
        strato_axs = sub_ax_arr[:,:,-1].flat

        colors_20c = plt.cm.tab20c.colors
        colors = colors_20c[:2] + colors_20c[4:7] + colors_20c[8:9]

        # Set axis titles and labels """
        pad = 12 if kwargs.get('show_stats') else 5
        
        for ax in sub_ax_arr[0,:,0].flat: # Top row inner left
            ax.set_title('Troposphere', style = 'oblique', pad = pad)
        for ax in sub_ax_arr[0,:,-1].flat: # Top row inner right
            ax.set_title('Stratosphere', style = 'oblique', pad = pad)

        for ax in sub_ax_arr.flat: 
            # All subplots
            ax.set_xlabel('Frequency [#]')
            ax.set_ylabel(subs.label(bin_attr=bin_attr), fontsize = 8)
            ax.grid(
                # axis='x', 
                ls ='dotted', lw = 1, color='grey', zorder=0)
            ax.set_xscale(xscale)
            
        # Add histograms and lognorm fits
        for axes, data_3d_dict in zip([tropo_axs, strato_axs],
                                      [tropo_3d_dict, strato_3d_dict]):     
            bin_lim_min, bin_lim_max = np.nan, np.nan
            for data3d in data_3d_dict.values(): 
                bin_lim_min = np.nanmin([bin_lim_min] + list(data3d.flatten()))
                bin_lim_max = np.nanmax([bin_lim_max] + list(data3d.flatten()))

            for ax, tp_col, c in zip(axes,
                                    data_3d_dict, 
                                    colors): 
                data3d = data_3d_dict[tp_col]
                data_flat = data3d.flatten()
                
                # Adding the histograms to the figure
                lognorm_inst = self._hist_lognorm_fitted(
                    data_flat, (bin_lim_min, bin_lim_max), ax, c,
                    hist_kwargs = dict(range = (bin_lim_min, bin_lim_max)))
                
                # Show values of mode and sigma
                if kwargs.get('show_stats'):
                    ax.text(x = 0, y = 1.015, 
                        s = 'Mode = {:.1f} / $\sigma$ = {:.2f}'.format(
                        lognorm_inst.mode,
                        lognorm_inst.sigma),
                        fontsize = 6,
                        transform = ax.transAxes,
                        style = 'italic'
                        )

        # Set xlims to maximum xlim for each subplot in tropos / stratos
        tropo_xmax = max([max(ax.get_xlim()) for ax in sub_ax_arr[:,:,0].flat])
        for ax in sub_ax_arr[:,:,0].flat: 
            ax.set_xlim(0 if xscale == 'linear' else 0.7, tropo_xmax)

        strato_xmax = max([max(ax.get_xlim()) for ax in sub_ax_arr[:,:,-1].flat])
        for ax in sub_ax_arr[:,:,-1].flat: 
            ax.set_xlim(0 if xscale == 'linear' else 0.7, strato_xmax)

        if bin_attr == 'rvstd': 
            # Equal y-limits for tropo / strato
            max_y = max([max(ax.get_ylim()) for ax in sub_ax_arr.flat])
            for ax in sub_ax_arr.flat: 
                ax.set_ylim(0, max_y)
            

        # Add tropopause definition text boxes and invert tropo x-axis
        for ax, tp_col in zip(sub_ax_arr[:,:,0].flat, tropo_3d_dict):
            ax.invert_xaxis()
            tp_title = dcts.get_coord(tp_col).label(filter_label=True).split("(")[0] # shorthand of tp label
            ax.text(**dcts.note_dict(ax, s = tp_title, x = 0.1, y = 0.85))
        
        return fig, main_axes, sub_ax_arr # strato_3d_dict, tropo_3d_dict

    def make_lognorm_fit_comparison(self, subs, zcoord, bin_attr = 'vstdv', eql=False, **kwargs): 
        """ Compare characteristic quantities for lognorm fits of strato/tropo data between tps. """
        strato_dict, tropo_dict = self.get_data_3d_dicts(subs, zcoord, eql, bin_attr, **kwargs)
        
        tropo_stats = self.get_lognorm_stats_df(tropo_dict, f'{bin_attr}_fit')
        strato_stats = self.get_lognorm_stats_df(strato_dict, f'{bin_attr}_fit')
        
        fig, axs = plt.subplots(1,2, figsize = (6,3.5), sharey = True)
        colors_20c = plt.cm.tab20c.colors
        colors = colors_20c[:2] + colors_20c[4:7] + colors_20c[8:9]

        axs[0].set_title('Troposphere',  size = 9, pad = 3)
        axs[1].set_title('Stratosphere', size = 9, pad = 3)

        marker_kw = dict(color = 'xkcd:dark grey',
                        path_effects = [self.outline])

        for df, ax in zip([tropo_stats, strato_stats], axs):
            ax.grid(axis='x', ls = 'dashed')
            for i, (tp, c) in enumerate(zip(df.columns, colors)): 
                y = dcts.get_coord(tp).label(filter_label = True).split('(')[0]

                # Lines
                line_kw = dict(color = c, lw = 7)
                ax.fill_betweenx([y]*2, *df[tp].int_68, **line_kw, alpha = 0.8)
                ax.fill_betweenx([y]*2, *df[tp].int_95, **line_kw, alpha = 0.5)

                # Markers
                kw_mode = dict(alpha = 0.7, zorder = 9, marker = 'D')
                kw_68   = dict(alpha = 0.7, zorder = 8, marker = 'd')
                kw_95   = dict(alpha = 1.0, zorder = 7, marker = '|')

                ax.scatter(df[tp].Mode,    y,    **kw_mode, **marker_kw)
                ax.scatter(df[tp].int_68, [y]*2, **kw_68,   **marker_kw)
                ax.scatter(df[tp].int_95, [y]*2, **kw_95,   **marker_kw)
            
                # Numeric values 
                ax.annotate(
                    text = f'{df[tp].Mode}', size = 8, 
                    xy = (df[tp].Mode, y),
                    xytext = (df[tp].Mode, i+0.35),
                    ha = 'center', va = 'center', )

        axs[0].invert_yaxis()
        axs[0].set_ylim(-0.5, len(df.columns)- 0.25)
        axs[0].tick_params(labelleft=True, left=False)
        axs[1].tick_params(left=False)

        m_Mode = mlines.Line2D([], [], ls = 'None', **kw_mode, **marker_kw)
        m_68 =   mlines.Line2D([], [], ls = 'None', **kw_68,   **marker_kw)
        m_95 =   mlines.Line2D([], [], ls = 'None', **kw_95,   **marker_kw)

        l = ['Mode', '68$\,$% Interval', '95$\,$% Interval']
        h = [m_Mode, m_68, m_95]

        fig.tight_layout()
        fig.subplots_adjust(bottom = 0.2, top = 0.85)
        fig.suptitle('Distribution of ' + subs.label(bin_attr=bin_attr))
        fig.legend(h, l, loc = 'lower center', ncols = len(h))
        
    def z_crossection(self, subs, tp, bin_attr, 
                      save_gif_path=None, **kwargs): 
        """ Create lat/lon gridded plots for all z-bins. 
        
        Args: 
            subs (dcts.Substance)
            tp (dcts.Coordinate)
            bin_attr (str): e.g. vmean, vsdtv, rvstd
            save_gif_path (str): Save all generated images as a gif to the given location
            
            key eql (bool): Use equivalent latitude for binning 
            key threshold (int): Minimum number of datapoints per plot
            key zbsize (float): Size of vertical bin 
            key zoom_factor (float): Use spline interpolation to zoom data by this factor)
        """

        eql = False if 'eql' not in kwargs else kwargs.get('eql')
        threshold = 3 if 'threshold' not in kwargs else kwargs.get('threshold')
        zbsize=None if 'zbsize' not in kwargs else kwargs.get('zbsize')
        zoom_factor = 1 if 'zoom_factor' not in kwargs else kwargs.get('zoom_factor')
        
        binned_data = self.bin_3d(subs, tp, zbsize=zbsize, eql=eql)
        data3d = getattr(binned_data, bin_attr)
        
        vlims = kwargs.get('vlims')
        if vlims is None: vlims = self.get_vlimit(subs, bin_attr)
        norm = Normalize(*vlims)
        cmap = dcts.dict_colors()[bin_attr]

        data_title = 'Mixing ratio' if bin_attr=='vmean' else 'Varibility'
        # fig.suptitle(f'{data_title} of {subs.label()}', y=0.95)

        if tp.rel_to_tp:
            title = f'Cross section binned relative to {tp.label(filter_label=True)} Tropopause'
        else: 
            title = '' # f' in {tp.label()}'

        images = []

        for iz in range(binned_data.nz):
            data2d = data3d[:,:,iz]
            if sum(~np.isnan(data2d.flatten())) > threshold: 
                fig, ax = plt.subplots(dpi=200)
                tools.add_world(ax)
                ax.set_title(title)
                ax.text(s = '{} to {} {}'.format(
                    binned_data.zbinlimits[iz], 
                    binned_data.zbinlimits[iz+1], 
                    tp.unit),
                        **dcts.note_dict(ax, x=0.025, y=0.05))

                img = ax.imshow(tools.nan_zoom(data2d, zoom_factor).T,
                                cmap = cmap, norm=norm,
                                aspect='auto', origin='lower',
                                # if not ycoord.vcoord in ['p', 'mxr'] else 'upper',
                                extent=[binned_data.xbmin, binned_data.xbmax, 
                                        binned_data.ybmin, binned_data.ybmax],
                                zorder = 1)

                cbar = fig.colorbar(img, ax = ax, aspect=30, pad=0.09, orientation='horizontal')
                cbar.ax.set_xlabel(f'{data_title} of {subs.label()}')
                
                if save_gif_path is not None:
                    # Save the figure to a BytesIO object
                    buf = io.BytesIO()
                    plt.savefig(buf, format='png')
                    plt.close(fig)
                    buf.seek(0)
                    
                    # Open the image from the BytesIO object
                    img = Image.open(buf)
                    images.append(img)
                else: 
                    plt.show()
    
        if save_gif_path is not None:
            if not save_gif_path.endswith('.gif'): 
                save_gif_path = save_gif_path + '.gif'
            tools.gif_from_images(images, save_gif_path)

        return binned_data

    def stratosphere_map(self, subs, tp, bin_attr, **kwargs): 
        """ Plot (first two ?) stratospheric bins on a lon-lat binned map. """
        df = self.sel_strato(**tp.__dict__).df
        # df = self.sel_tropo(**tp.__dict__).df
        
        #!!! df = self.sel_LMS(**tp.__dict__).df

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
        
        vlims = kwargs.get('vlims')
        if vlims is None: vlims = self.get_vlimit(subs, bin_attr)
        
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

    def matrix_plot_3d_stdev_subs(self, substance, note='', tps=None, savefig=False
                                  ) -> tuple[np.array, np.array]:
        """
        Create matrix plot showing variability per latitude bin per tropopause definition
    
        Parameters:
            (GlobalObject): Contains the data in self.df
            key short_name (str): Substance short name to show, e.g. 'n2o'

        Returns:
            tropospheric, stratospheric standard deviations within each bin as list for each tp coordinate
        """
        if not tps: 
            tps = [tp for tp in dcts.get_coordinates(tp_def='not_nan')
                   if 'tropo_'+tp.col_name in self.df_sorted.columns]
    
        lat_bmin, lat_bmax = -90, 90 # np.nanmin(lat), np.nanmax(lat)
        lat_bci = bp.Bin_equi1d(lat_bmin, lat_bmax, self.grid_size)
    
        tropo_stdevs = np.full((len(tps), lat_bci.nx), np.nan)
        tropo_av_stdevs = np.full(len(tps), np.nan)
        strato_stdevs = np.full((len(tps), lat_bci.nx), np.nan)
        strato_av_stdevs = np.full(len(tps), np.nan)
    
        tropo_out_list = []
        strato_out_list = []
    
        for i, tp in enumerate(tps):
            # troposphere
            tropo_data = self.sel_tropo(**tp.__dict__).df
            tropo_lat = np.array([tropo_data.geometry[i].y for i in range(len(tropo_data.index))]) # lat
            tropo_out_lat = bp.Simple_bin_1d(tropo_data[substance.col_name], tropo_lat, 
                                             lat_bci, count_limit = self.count_limit)
            tropo_out_list.append(tropo_out_lat)
            tropo_stdevs[i] = tropo_out_lat.vstdv if not all(np.isnan(tropo_out_lat.vstdv)) else tropo_stdevs[i]
            
            # weighted average stdv
            tropo_nonan_stdv = tropo_out_lat.vstdv[~ np.isnan(tropo_out_lat.vstdv)]
            tropo_nonan_vcount = tropo_out_lat.vcount[~ np.isnan(tropo_out_lat.vstdv)]
            tropo_weighted_average = np.average(tropo_nonan_stdv, weights = tropo_nonan_vcount)
            tropo_av_stdevs[i] = tropo_weighted_average 
            
            # stratosphere
            strato_data = self.sel_strato(**tp.__dict__).df
            strato_lat = np.array([strato_data.geometry[i].y for i in range(len(strato_data.index))]) # lat
            strato_out_lat = bp.Simple_bin_1d(strato_data[substance.col_name], strato_lat, 
                                              lat_bci, count_limit = self.count_limit)
            strato_out_list.append(strato_out_lat)
            strato_stdevs[i] = strato_out_lat.vstdv if not all(np.isnan(strato_out_lat.vstdv)) else strato_stdevs[i]
            
            # weighted average stdv
            strato_nonan_stdv = strato_out_lat.vstdv[~ np.isnan(strato_out_lat.vstdv)]
            strato_nonan_vcount = strato_out_lat.vcount[~ np.isnan(strato_out_lat.vstdv)]
            strato_weighted_average = np.average(strato_nonan_stdv, weights = strato_nonan_vcount)
            strato_av_stdevs[i] = strato_weighted_average 
    
        # Plotting
        # -------------------------------------------------------------------------
        pixels = self.grid_size # how many pixels per imshow square
        yticks = np.linspace(0, (len(tps)-1)*pixels, num=len(tps))[::-1] # order was reversed for some reason
        tp_labels = [tp.label(True)+'\n' for tp in tps]
        xticks = np.arange(lat_bmin, lat_bmax+self.grid_size, self.grid_size)
    
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
                    ax_strato1.text(x+0.5*self.grid_size,
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
                         extent = [0, self.grid_size,
                                   0, len(tps)*pixels],
                         cmap = strato_cmap, norm=norm)
        for i,y in enumerate(yticks): 
            value = strato_av_stdevs[i]
            if str(value) != 'nan':
                ax_strato2.text(0.5*self.grid_size,
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
                    ax_tropo1.text(x+0.5*self.grid_size,
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
                         extent = [0, self.grid_size,
                                   0, len(tps)*pixels],
                         cmap = tropo_cmap, norm=norm)
    
        for i,y in enumerate(yticks): 
            value = tropo_av_stdevs[i]
            if str(value) != 'nan':
                ax_tropo2.text(0.5*self.grid_size,
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
    
    def matrix_plot_stdev(self, note='', savefig=False):
        substances = [s for s in self.substances
                      if not s.col_name.startswith('d_')]
        for subs in substances:
            self.matrix_plot_stdev_subs(subs, note=note,savefig=savefig)

class BinPlotterMixin(BinPlotter1DMixin, BinPlotter2DMixin, BinPlotter3DMixin): 
    pass
