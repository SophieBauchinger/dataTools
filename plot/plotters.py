# -*- coding: utf-8 -*-
"""
@Author: Sophie Bauchinger, IAU
@Date: Thu May 11 13:22:38 2023

Class definitions for combining data structures and plotting possibilities.

"""
import math
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

from dataTools.plot.data import GlobalDataPlotterMixin
from dataTools.plot.binsubs import BinPlotterMixin
from dataTools.plot.tropopause import TropopausePlotterMixin

from dataTools.data.Caribic import Caribic
from dataTools.data.CampaignData import CampaignData
from dataTools import tools
import dataTools.dictionaries as dcts  

class PlotterMixin(GlobalDataPlotterMixin,
                   BinPlotterMixin, 
                   TropopausePlotterMixin):
    """ Combination of all available plotting modules for global datasets + figure structures. """
    def __repr__(self): 
        return f"""Plotter {self.base_class}
    years: {self.years}
    status: {self.status}"""
    
    def _three_sideplot_structure(self): 
        """ Create Figure with central + upper/right additional plots + space on top right. """
        fig,ax_fig = plt.subplots()
        ax_fig.axis('off')

        gs = gridspec.GridSpec(5, 5, # nrows, ncols 
                            figure = fig,
                            wspace=0, hspace=0,
                            width_ratios=[2,2,2,1,1], height_ratios=[1,1,1,1,1])

        ax_main = fig.add_subplot(gs[2:, :-2])
        ax_upper = fig.add_subplot(gs[:2, :-2], sharex = ax_main)
        ax_right = fig.add_subplot(gs[2:, -2:], sharey = ax_main)

        ax_upper.tick_params(bottom=False, labelbottom=False, top=True, labeltop=True)
        ax_right.tick_params(left=False, labelleft=False, right = True, labelright=True)
        
        ax_cube = fig.add_subplot(gs[:2, -2:], projection = '3d')
        ax_cube.axis('off')
        
        axs = (ax_fig, ax_main, ax_upper, ax_right, ax_cube)
        
        return fig, (axs)
    
    def _three_sideplot_labels(self, fig, axs, zcoord: dcts.Coordinate, eql: bool): 
        """ Finish doing axis labels, limits and scaling for the 3 sideplot structure thing. """
        (ax_fig, ax_main, ax_upper, ax_right, ax_cube) = axs
        
        fig.suptitle(zcoord.label(filter_label=True))
        fig.subplots_adjust(top = 0.85)
        
        xcoord = dcts.get_coord('geometry.x')
        ycoord = dcts.get_coord('geometry.y') if not eql else \
            self.get_coords(hcoord='eql', model='ERA5')[0]

        ax_right.invert_xaxis() # for a more intuitive represenation 
        
        # Homogenise intuitive understanding for different vertical coordinates
        if zcoord.vcoord == 'p': 
            ax_upper.invert_yaxis()
            ax_upper.set_yscale('log' if not zcoord.rel_to_tp else 'symlog')

            ax_right.invert_xaxis()
            ax_right.set_yscale('log' if not zcoord.rel_to_tp else 'symlog')

        # Add axis labels
        ax_main.set_xlabel(xcoord.label())
        ax_main.set_ylabel(ycoord.label())
        
        ax_upper.set_xlabel(xcoord.label())
        ax_upper.set_ylabel(zcoord.label(coord_only=True))
        ax_upper.yaxis.set_label_position("left")
        ax_upper.xaxis.set_label_position("top")
        ax_upper.grid(True, zorder=0, ls='dashed', alpha=0.5)
        
        ax_right.set_xlabel(zcoord.label(coord_only=True))
        ax_right.set_ylabel(ycoord.label())
        ax_right.yaxis.set_label_position("right")
        ax_right.xaxis.set_label_position("bottom")
        ax_right.grid(True, zorder=0, ls='dashed', alpha=0.5)
        
        # Match vertical coordinate limits 
        z_lims = ax_upper.get_ylim() + ax_right.get_xlim()
        z_lims = min(z_lims), max(z_lims)
        ax_right.set_xlim(*z_lims)
        ax_upper.set_ylim(*z_lims)

    def _make_two_column_axs(self, tps, extra_axes=0, sharex=True, sharey=True, 
                             **fig_kwargs): 
        """ Create the necessary nr of subplots and hide superfluous axes. """
        no_of_axs = len(tps) + extra_axes

        fig = plt.figure(figsize=(7, math.ceil(no_of_axs/2)*2), dpi=100, **fig_kwargs)
        
        gs = fig.add_gridspec(math.ceil(no_of_axs/2), 2)
        axs = gs.subplots(sharex=sharex, sharey=sharey)
        gs.tight_layout(fig)

        if no_of_axs%2: 
            axs.flatten()[len(tps)].axis('off')
            axs.flatten()[-1].axis('off')
        
        return fig, axs

    def _nested_subplots_two_column_axs(self, tps, nsubcol=2, **fig_kwargs
            ) -> tuple[plt.Figure, np.ndarray, np.ndarray]:
        """ Expand 2-column functionality for stacked subplots within the 2-column structure. 
        
        Parameters: 
            tps (list[dcts.Coordinate]): Tropopause definitions

        Returns
            fig: plt.Figure
            main_axs: array of shape(nrow, ncol)
            sub_ax_arr: array of shape(nrow, ncol, nsubcol)
                get subsets of this using the syntax `sub_ax_arr[Row, Column, InnerPos]`
            
            ncol, nsubcol default to 2
        
        Examples for accessing specific sub-structures of sub_ax_arr:
            sub_ax_arr[0,:,] # Top row
            sub_ax_arr[-1,:,] # Bottom row

            sub_ax_arr[:,:,0] # Left sub axes = Tropos
            sub_ax_arr[:,:,-1] # Right sub axes = Stratos

            sub_ax_arr[:,0,0] # Left column inner left
            sub_ax_arr[:,-1,-1] # Right column inner right
        
        """
        fig,_ = self._make_two_column_axs(tps, **fig_kwargs)
        outer_gs = fig.get_axes()[0].get_gridspec()

        for i, gs in enumerate(outer_gs): 
            inner_gs = gridspec.GridSpecFromSubplotSpec(
                1, nsubcol, 
                subplot_spec = gs, 
                wspace=0.1, hspace=0.1)

            for j in range(nsubcol): # 2 plots per subplot: Tropo, Strato
                ax = plt.Subplot(fig, inner_gs[j])
                fig.add_subplot(ax) 
                if i >= len(tps): 
                    # Add axis anyway to retain array shape, but turn off before adding to figure
                    ax.axis('off')

        outer_gs.tight_layout(fig)
        
        main_axes = [ax for ax in fig.axes if isinstance(ax.get_gridspec(), gridspec.GridSpec)]
        main_axes = np.array(main_axes).reshape(int(len(main_axes)/2), 2)
        for ax in main_axes.flat: 
            ax.axis('off')

        sub_axes = [ax for ax in fig.axes if isinstance(ax.get_gridspec(), gridspec.GridSpecFromSubplotSpec)]
        sub_ax_arr = np.array(sub_axes).reshape(int(len(sub_axes)/(2*nsubcol)), 2, nsubcol)
    
        return fig, main_axes, sub_ax_arr
    
    def _adjust_labels_ticks(self, sub_ax_arr): 
        """ Move axis labels and ticks to the outside of subsubplots and axes. """
        # --- Move tick marks to the outside 
        for ax in sub_ax_arr[:,:,0].flat:
            # Inner left plots
            ax.tick_params(axis = 'y',
                        right=False, labelright=False, 
                        left = True, labelleft = True)

        for ax in sub_ax_arr[:,:,-1].flat:
            # Inner right plots
            ax.tick_params(axis = 'y',
                        right=True, labelright=True,
                        left = False, labelleft = False)
        
        # --- Add outer axis descriptions and titles 
        for ax in sub_ax_arr[:,0,0].flat: 
            # Left column inner left
            ax.yaxis.set_label_position('left')

        for ax in sub_ax_arr[:,-1,-1].flat: 
            # Right column inner right
            ax.yaxis.set_label_position('right')
            
        # --- Hide x- and y-axis labels for inner subplots
        for ax in list(sub_ax_arr[:,0,-1].flat) + list(sub_ax_arr[:,-1,0].flat): 
            # Left column inner right & right column inner left
            ax.yaxis.label.set_visible(False)
        
        for ax in sub_ax_arr[:-1,:,].flat: 
            # All plots not on the bottom row
            ax.xaxis.label.set_visible(False)
        
        return sub_ax_arr

    def _hist_lognorm_fitted(self, x, range, ax, c, hist_kwargs = {}, bin_nr = 50,
                             **kwargs): 
        """ Adds a horizontal histogram with a lognorm best fit to the given axis. 
        
        x ((n,) array or sequence of (n,) arrays): 1D array with data to be histogrammed
        range (tuple): histogram range
        ax (Axis): axis to be plotted onto 
        c (str): color of the histogram bars

        """
        x = x[~np.isnan(x)]
        ax.hist(x, 
            bins = bin_nr, range = range, 
            orientation = 'horizontal',
            edgecolor = 'white', lw = 0.3, 
            color=c,
            )
        lognorm_inst = tools.LognormFit(x, bin_nr = bin_nr, hist_kwargs = hist_kwargs)
        lognorm_fit = lognorm_inst.lognorm_fit
        bin_center = lognorm_inst.bin_center
        ax.plot(lognorm_fit, bin_center,
                c = 'k', lw = 1)
        
        ax.hlines(lognorm_inst.mode, 0, max(lognorm_fit),
                  ls = 'dashed', 
                  color = 'k', 
                  lw = 1)

        return lognorm_inst

class CaribicPlotter(PlotterMixin,
                     Caribic): 
    def __init__(self, **kwargs): 
        super().__init__(**kwargs)
        self.base_class = Caribic

class CampaignPlotter(PlotterMixin,
                     CampaignData): 
    def __init__(self, **kwargs): 
        super().__init__(**kwargs)
        self.base_class = CampaignData
