# -*- coding: utf-8 -*-
""" Plotting Tropopause heights for different tropopauses different vertical coordinates

@Author: Sophie Bauchinger, IAU
@Date Mon Aug 14 14:06:26 2023

class TropopausePlotter

class CaribicTropopause(Caribic, TropopausePlotter)
 ->> can create ..TP classes for all GlobalData subclasses 

"""
import math
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm
from matplotlib.cm import ScalarMappable
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from statistics import median 

import toolpac.calc.binprocessor as bp # type: ignore
from toolpac.conv.times import datetime_to_fractionalyear as dt_to_fy # type: ignore

import dataTools.dictionaries as dcts
from dataTools import tools

# Coordinate vlimits (for e.g. tropopause height colormaps)
vlims = {'p':(100,500), 'pt':(300, 350), 'z':(6.5,14), 'mxr': (290, 330)}
rel_vlims = {'p':(-100,100), 'pt':(-30, 40), 'z':(-1,2.5)}

#%% Define TropopausePlotter
class TropopausePlotterMixin: 
    """ Class to hold plotting functionality for GlobalData objecs. 
    
    Needs to be used in multiple inheritance together with a subclass of GlobalData: 
        class GlobalSubclassTropopause(GlobalSubclass, TropopausePlotter)
    """
# --- Plot differences in tropopause heights for TP definitions / seasons / years ---       
    def tp_height_global_2D_overview(self, rel=False, tps=None):
        """ Show 2D-binned latitude-longitude maps of tropopause height for various definitions. """
        for tp in (self.tps if not tps else tps): 
            fig, ax = plt.subplots(dpi=150, figsize=(10,5))
            ax.set_title(tp.label(True))

            bci =  bp.Bin_equi2d(-180, 180, self.grid_size, 
                                 -90, 90, self.grid_size)

            out = bp.Simple_bin_2d(self.df[tp.col_name],
                                   self.df.geometry.x, 
                                   self.df.geometry.y, 
                                   bci, count_limit = self.count_limit)

            tools.add_world(ax)
            # ax.set_title(dcts.dict_season()[f'name_{s}'])
            cmap = 'viridis_r' if tp.vcoord=='p' else 'viridis'

            vlims = tp.get_lims()
            norm = Normalize(*vlims)

            # df_r.plot(tp.col_name, cmap=cmap, legend=True, ax=ax)
            img = ax.imshow(out.vmean.T, cmap = cmap, norm=norm, origin='lower',
                            extent = [bci.xbmin, bci.xbmax, bci.ybmin, bci.ybmax])
            cbar = plt.colorbar(img, ax=ax, pad=0.08,
                                orientation='vertical', extend='both') # colorbar
            cbar.ax.set_xlabel('{}{} [{}]'.format(
                '' if not rel else '$\Delta $',
                tp.vcoord if not tp.vcoord=='pt' else '$\Theta$',
                tp.unit))
            ax.set_ylim(-90, 90); ax.set_xlim(-180, 180)
            ax.set_ylabel('Longitude [°E]')
            ax.set_xlabel('Latitude [°N]')
            plt.show()

    def tp_height_seasonal_2D_overview(self, savefig=False, year=None, tps=None):
        """ Plot 2D-binned latitude-longitude map of tropopause heights for all seasons.

        Parameters:
            savefig (bool): save plot to pdir instead of plotting
            year (float): select single specific year to plot / save
            tps (list): tropopause definitions, use if specified 
        """
        pdir = r'C:\Users\sophie_bauchinger\sophie_bauchinger\Figures\tp_scatter_2d'
        
        for tp in (self.tps if not tps else tps): 
            fig, axs = plt.subplots(2,2, dpi=200, figsize=(10,5))
            fig.suptitle(tp.label(filter_label = True))

            [lon_coord] = self.get_coords(col_name = 'geometry.x')
            [lat_coord] = self.get_coords(col_name = 'geometry.y')
            
            bin2d_dict = self.bin_2d_seasonal(
                var = tp,
                xcoord = lon_coord, 
                ycoord = lat_coord, 
                xbsize = self.grid_size, 
                ybsize = self.grid_size)

            if year: 
                fig.text(0.9, 0.95, f'{year}',
                         bbox = dict(boxstyle='round', facecolor='white',
                                     edgecolor='grey', alpha=0.5, pad=0.25))
                bin2d_dict = self.sel_year(year).bin_2d_seasonal(
                    var = tp,
                    xcoord = lon_coord, 
                    ycoord = lat_coord, 
                    xbsize = self.grid_size, 
                    ybsize = self.grid_size)
            
            for s,ax in zip([1,2,3,4], axs.flatten()):
                bin2d = bin2d_dict[s]

                cmap = 'viridis_r' if tp.vcoord=='p' else 'viridis'
                norm = Normalize(*vlims[tp.vcoord]) # colormap normalisation

                img = ax.imshow(bin2d.vmean.T, cmap = cmap, 
                                origin='lower', norm=norm,
                            extent = [bin2d.xbmin, bin2d.xbmax, 
                                      bin2d.ybmin, bin2d.ybmax])

                cbar = plt.colorbar(img, ax=ax, pad=0.08,
                                    orientation='vertical') # colorbar
                cbar.ax.set_xlabel(f'{tp.vcoord} [{tp.unit}]')

                ax.set_title(dcts.dict_season()[f'name_{s}'])
                tools.add_world(ax)
                ax.set_ylim(-90, 90); ax.set_xlim(-180, 180)

            fig.tight_layout()
            if savefig:
                plt.savefig(pdir+'\{}{}.png'.format(
                    tp.col_name, '_'+str(year) if year else ''))

            plt.show()
            plt.close()

    def tp_height_seasonal_1D_binned(self, tp, **kwargs): 
        """ Plot the average tropopause height (or delta) per season binned over latitude. 
        
        Args: 
            key df (pd.DataFrame): Dataset from which to draw plot info. Defaults to self.df
            key xcoord (dcts.Coordinate): Coord used for 1D binning. Defaults to latitude
            key bsize (float): Bin size
        """
        df = kwargs.get('df', self.df)
        coord = kwargs.get('coord', dcts.get_coord(col_name = 'geometry.y'))
        bci = self.make_bci(coord, xbsize = kwargs.get('bsize', coord.get_bsize()))
        n2o_color = 'xkcd:violet'

        # Prepare the plot
        ax = kwargs.get('ax') if 'ax' in kwargs else plt.subplots()[1]
        ax.set_title(tp.label(filter_label=True))
        ax.set_ylabel(tp.label(coord_only=True) + f' [{tp.unit}]', 
                      color=n2o_color if tp.crit=='n2o' else 'k')
        ax.set_xlabel(coord.label())
        ax.grid(True, ls='dotted')

        # Add data for each season and the average 
        for s in ['av',1,2,3,4]:
            data = df if s=='av' else df.query(f'season == {s}')
            bin1d = self.bin_1d(tp, coord, df = data, bci_1d = bci)

            plot_kwargs = dict(lw=3, path_effects = [self.outline])
            if s=='av': 
                plot_kwargs.update(dict(
                    label = 'Average', 
                    color='dimgray', 
                    ls = 'dashed', zorder=5))
                # if average, want to plot the standard deviation in light grey
                ax.fill_between(bin1d.xintm,
                                bin1d.vmean - bin1d.vstdv, 
                                bin1d.vmean + bin1d.vstdv,
                                alpha=0.13, color=plot_kwargs['color'])

            else: 
                plot_kwargs.update(dict(
                    label = dcts.dict_season()[f'name_{s}'], 
                    color = dcts.dict_season()[f'color_{s}']))
            ax.plot(bin1d.xintm, bin1d.vmean,
                     path_effects = plot_kwargs.pop('path_effects'),
                     **plot_kwargs)

        if 'yscale' in kwargs:
            ax.set_yscale(kwargs.get('yscale'))
        if 'ylims' in kwargs: 
            ax.set_ylim(*kwargs.get('ylims'))
        if 'xlims' in kwargs: 
            ax.set_xlim(*kwargs.get('xlims'))
        if kwargs.get('invert_yaxis'):
            ax.invert_yaxis()
        if tp.rel_to_tp: 
            tools.add_zero_line(ax)

    def tps_height_comparison_seasonal_1D(self, **kwargs): 
        """ Default plot for comparing Tropopause heights in latitude bins. """ 
        tps = kwargs.pop('tps', self.tps)
        fig, axs = plt.subplots(3,2, figsize = (10, 10), sharex=True, 
                                dpi=kwargs.pop('dpi', 150))
        ylims = kwargs.pop('ylims', None)
        
        for tp, ax in zip(tps, axs.flat): 
            self.tp_height_seasonal_1D_binned(
                tp, ax = ax, 
                invert_yaxis = True if tp.vcoord =='mxr' else False,
                ylims = ylims if not (tp.vcoord=='mxr' or ylims is None) else tp.get_lims(), 
                **kwargs)
            if tp.crit == 'n2o': 
                n2o_color = 'xkcd:violet'
                ax.tick_params(axis='y', color=n2o_color, labelcolor=n2o_color)
                ax.spines['right'].set_color(n2o_color)
                ax.spines['left'].set_color(n2o_color)
        fig.suptitle('Vertical extent of tropopauses')
        fig.tight_layout()
        fig.subplots_adjust(top = 0.85)
        fig.legend(handles = self.season_legend_handles(av=True), 
                   ncol = 3, loc='upper center', 
                   bbox_to_anchor=[0.5, 0.95])

# --- Basic quantification of differences between tropopause definitions ---
    def show_strato_tropo_vcounts(self, **kwargs): 
        """ Bar plots of data point allocation for multiple tp definitions. """
        tps = kwargs.get('tps', self.tps)
        tropo_counts, _ = self.tropo_strato_ratios()
        tropo_counts = tropo_counts[[tp.col_name for tp in tps 
                                     if tp.col_name in tropo_counts.columns]]
        tp_labels = [tp.label(filter_label=True, no_vc = True) for tp in tps]
        
        tropo_bar_vals = [tropo_counts[i].loc[True] for i in tropo_counts.columns]
        strato_bar_vals = [tropo_counts[i].loc[False] for i in tropo_counts.columns]

        fig, (ax_t, ax_label, ax_s) = plt.subplots(1, 3, dpi=400, 
                                        figsize=(9,4), sharey=True)
        
        fig.subplots_adjust(top=0.85)
        ax_t.set_title('# Tropospheric points', fontsize=9, loc='right')
        ax_s.set_title('# Stratospheric points', fontsize=9, loc='left')

        bar_labels = tp_labels
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
        minimum = np.nanmin(tropo_bar_vals+strato_bar_vals)
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

        ax_t.bar_label(t_bars, ['{0:.0f}'.format(t_val) for t_val in tropo_bar_vals], 
                       padding=2)
        ax_s.bar_label(s_bars, ['{0:.0f}'.format(s_val) for s_val in strato_bar_vals], 
                       padding=2)

        for ax in [ax_t, ax_s]: 
            ax.yaxis.set_major_locator(ticker.NullLocator())
        if kwargs.get('note'): ax.text(s=kwargs.get('note'), **dcts.note_dict(ax, y=1.1))
        fig.subplots_adjust(wspace=0)
        return fig

    def seasonal_ratio_comparison(self, **kwargs): 
        """ Show tropospheric / stratospheric ratio of data points per season per TP def. 
        Parameters: 
            key dpi (int): Figure resolution
            key title (bool): Show figure title 

        """
        fig, axs = plt.subplots(2, 2, figsize = (7,6), dpi=kwargs.get('dpi', 250))
        for s, ax in zip(range(1,5), axs.flat): 
            self.sel_season(s).show_ratios(ax=ax, xlim = kwargs.pop('xlim', (0, 1.8)), **kwargs)
            ax.set_ylabel('')
            ax.yaxis.set_visible(False)
            ax.set_title('')
            # ax.set_title(dcts.dict_season()[f'name_{s}'])
            s = dcts.dict_season()[f'name_{s}'].split()[0] + '\n' + dcts.dict_season()[f'name_{s}'].split()[1]
            ax.text(s = s, y = 0.2, x = 0.85, 
                    transform = ax.transAxes, ha = 'center', va = 'center_baseline', 
                    bbox = dict(facecolor='white', edgecolor = 'grey'))
        fig.tight_layout()
        fig.subplots_adjust(top = 0.8)
        if kwargs.get('title'):
            fig.suptitle('Ratio of tropospheric / stratospheric data points per tropopause definition')
        fig.legend(handles = self.tp_legend_handles(lw = 5, no_vc=True), ncol = 3, 
                loc='upper center', bbox_to_anchor=[0.5, 0.93]);

    def show_ratios(self, **kwargs):
        """ Plot ratio of tropo / strato datapoints on a horizontal bar plot 
        
        Args: 
            key ax (Axis): Axis to draw the figure onto
            key tps (list[dcts.Coordinate]): Tropopause definitions to compare
            key filter (str)
        """
        if not 'ax' in kwargs: fig, ax = plt.subplots(figsize = (5,3))
        else: ax = kwargs.pop('ax')
        
        tropo_counts, ratio_df = self.tropo_strato_ratios(**kwargs)
        ratios = ratio_df.loc['ratios']

        ax.set_title(f'Ratio of tropospheric / stratospheric datapoints in {self.source}')
        ax.axvline(1, linestyle='--', color='k', alpha=0.3, zorder=0, lw=1) # vertical lines
        ax.set_axisbelow(True)

        for tp in kwargs.get('tps', self.tps)[::-1]:
            color = tp.get_color()
            ratio = ratios[tp.col_name]
            n_value = tropo_counts[tp.col_name].loc[True] + tropo_counts[tp.col_name].loc[False] 
            label = tp.label(filter_label = True, no_vc = True)
            
            bars = ax.barh(label, ratio, rasterized=True, color=color, alpha=0.9)
            if kwargs.get('n_values'):
                bar_labels = ['{:.2f} (n={:.0f})'.format(r,n) for r,n in zip([ratio], [n_value])]
            else: 
                bar_labels = ['{:.2f}'.format(ratio)]

            ax.bar_label(bars, bar_labels, fmt='%.3g', padding=1)
            ax.set_xlim(kwargs.get('xlim', (0,2)))

        if 'fig' in locals(): fig.tight_layout()

    def distributions_1d_binned(self, var, xcoords, density=True, 
                                note = None, **bci_kwargs): 
        """ Show distribution of var-values in bins of xcoords (tps). 
        
        Args: 
            var (dcts.Coordinate|dcts.Substance)
            xcoords (dcts.Coordinate or array thereof)
            density (bool): Normalise distributions
            
            key bci_1d (bp.binclassinstance)
            key v_bins (array): Bins for variable distribution histograms
            
        Returns v_bins, bin_dict - var. distr. bins / data binned using xcoords. 
        """

        # Calculate bins for variable distributions
        v = self.df[var.col_name]
        vbsize = var.get_bsize()
        v_bins= np.arange(np.floor(np.nanmin(v)),
                          np.nanmax(v) + vbsize,
                          step = vbsize)

        bin_dict = {}
        for xcoord in xcoords: 
            bci_1d = self.make_bci(xcoord, **bci_kwargs)
            x = self.df[xcoord.col_name]
            bp1d = bp.Simple_bin_1d(v, x, bci_1d)
            bin_dict[xcoord] = bp1d
            
        # Plot values 
        fig, axs_top_to_bottom = plt.subplots(len(bp1d.xintm), 
                                              len(xcoords)+1, 
                                figsize = (2*(len(xcoords)+1), # width
                                           0.8 * len(bp1d.xintm)), # height 
                                sharey=True, sharex=True,)
        
        axs = np.flip(axs_top_to_bottom, axis = 0) # highest on top 
        
        # Add bin description, grid, top ticks, xlabel
        for ax in axs_top_to_bottom[0,1:]: # top row 
            ax.tick_params(axis = 'x', labeltop = True, top = True)
            ax.xaxis.set_label_position('top')

        # Show xlabel on top / bottom. Only outer axes if label is too long
        label_axs = list(axs[0,1:]) + list(axs[-1, 1:])
        if len(var.label()) > 18: 
            label_axs = [axs[0,1], axs[0,-1], axs[-1, 1], axs[-1, -1]]
        for ax in label_axs:
            ax.set_xlabel(var.label())

        # Get secondary axis to plot normalised distribution in the background 
        flat_twin_axs = [ax.twinx() for ax in axs.flat]
        for ax in flat_twin_axs[1:]: 
            ax.sharey(flat_twin_axs[0])
        twin_axs = np.reshape(flat_twin_axs, axs.shape).copy()

        # Label bins in leftmost column
        # NB: left edge inside the bin, right edge outside
        for xi, ax in enumerate(axs[:,0]): 
            ax.axis('off')
            ax.text(s = f'[{bp1d.xbinlimits[xi]}, {bp1d.xbinlimits[xi+1]}) {xcoords[0].unit} ', 
                    x = 0.7 if density=='both' else 0.9, 
                    y = 0.5, transform = ax.transAxes, 
                    va = 'center', ha = 'right', fontsize = 12)
        for ax in axs.flat: # all 
            ax.grid(True, axis = 'x', ls = 'dashed', color = 'grey', alpha = 0.5)

        # Add histograms for var distribution within delta-xcoord bins
        for tp_i, (tp, bp1d) in enumerate(bin_dict.items()): 
            column = axs[:,tp_i + 1]
            for xi, vdata in enumerate(bp1d.vbindata): 
                if str(vdata) == 'nan': continue
                if len(vdata) > self.count_limit:
                    column[xi].hist(vdata, 
                                    bins = v_bins, 
                                    histtype='step',
                                    edgecolor=tp.get_color(), 
                                    density = density if not density=='both' else False, 
                                    lw = 2, 
                                    zorder = 3)
                    if density == 'both': 
                        twin_axs[xi,tp_i + 1].hist(vdata,
                                        bins = v_bins, 
                                        histtype='stepfilled',
                                        color=tp.get_color(), 
                                        density = True, 
                                        alpha = 0.25,
                                        zorder = 1)
                        twin_axs[xi,tp_i + 1].hist(vdata,
                                        bins = v_bins, 
                                        histtype='step',
                                        density = True, 
                                        alpha = 0.5,
                                        edgecolor=tp.get_color(), 
                                        ls = 'dashed',
                                        lw = 1)

        # Add vlines for median now that ylims are set for all histograms
        ymin = np.min([ax.get_ylim()[0] for ax in axs.flat])
        ymax = np.max([ax.get_ylim()[1] for ax in axs.flat])
        for tp_i, (tp, bp1d) in enumerate(bin_dict.items()): 
            for xi, vdata in enumerate(bp1d.vbindata):
                if str(vdata) == 'nan': continue
                if len(vdata) > self.count_limit: 
                    axs[xi, tp_i+1].vlines(median(vdata), ymin, ymax/2, 
                                            color='k', ls = 'dashed', 
                                            lw = 1.5, zorder = 5)

        # Adjust yticks and spines 
        for ax in axs[:,-1]: # rightmost column
            ax.tick_params(axis = 'y', labelright = True, right = True)
            if any(t%1 != 0 for t in ax.get_yticks()):
                ax.set_yticks([float('%.1g' % v) for v in 
                            [ymin, ymax*0.33, ymax*0.66]])
            else: 
                ax.set_yticks([float('%.3g' % v) for v in 
                            [ymin, ymax*0.33, ymax*0.66]])
        if density=='both':
            norm_ymax = np.max([ax.get_ylim()[1] for ax in twin_axs.flat])
            for ax in twin_axs.flat:
                ax.tick_params(labelright=False, labelleft=False, 
                            right = False, left = False)
            for ax in twin_axs[:, 1].flat: # left data column
                ax.tick_params(labelleft=True, left = True, 
                            color='xkcd:dark grey',
                            labelcolor='xkcd:dark grey')
                ax.spines['left'].set(color = 'xkcd:dark grey')
                ax.set_yticks([float('%.1g' % v) for v in 
                        [0, norm_ymax*0.33, norm_ymax*0.66]])
            for ax in twin_axs[:,0]:
                ax.axis('off')
            for ax in axs[:,1].flat:
                ax.tick_params(left=False)

        # Add tps description as legend
        legend = fig.legend(handles = self.tp_legend_handles(tps = xcoords), 
                loc = 'upper center', ncols = 1)
        
        # Make space for the legend dynamically 
        legend_bbox = legend.get_window_extent(renderer=fig.canvas.get_renderer())
        legend_height = legend_bbox.height / fig.dpi  # height in inches
        
        fig_bbox = fig.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        fig.set_size_inches(fig_bbox.width, fig_bbox.height + legend_height)
        fig_bbox = fig.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        new_top = 1 - (legend_height / fig_bbox.height) * 1.75
        fig.subplots_adjust(top = new_top,
                            wspace=0, hspace=0)
        
        # Add textbox anchored to the legend location 
        if note: 
            legend_bbox = legend.get_window_extent(renderer=fig.canvas.get_renderer())
            legend_loc = fig.transFigure.inverted().transform(legend_bbox.get_points())

            fig.text(legend_loc[1,0] + 0.1, 
                    (legend_loc[0, 1] + legend_loc[1, 1]) / 2, 
                    note, 
                    fontsize=10,
                    bbox=dict(facecolor='white', alpha=0.8, 
                                edgecolor='xkcd:dark grey', boxstyle='round,pad=0.5'),
                    ha='left', va='center')
        
        return v_bins, bin_dict

# --- Plot substance values sorted into stratosphere / troposphere
    def make_figures_per_vcoord(self, plotting_function, **kwargs): 
        """ Create suitable canvas to plot various plots per tp def and vcoord onto. 
        Parameters: 
            plotting function (): Function to pass ax and kwargs to
                must be given in the form of {ClassInstance.method} if class method, else as {method}
        
        Optional:
            key subs (dcts.Substance): substance to sort and plot
            key tps (list[dcst.Coordinates])
            key vcoords (list[dcts.Coordinates])
            popt0 / popt1 (tuple[float]): initial / filtered baseline fit parameters
        """
        tps = (self.tps if not kwargs.get('tps') else kwargs.get('tps'))
        vcoords = set(tp.vcoord for tp in tps) if not kwargs.get('vcoords') else kwargs.get('vcoords')

        for vcoord in vcoords: 
            tps_vc = [tp for tp in tps if tp.vcoord == vcoord]

            fig, axs = plt.subplots(math.ceil(len(tps)/2), 2, dpi=200,
                                    figsize=(7, math.ceil(len(tps)/2)*2),
                                    sharey=True, sharex=True)
            if len(tps_vc)==0: continue

            if len(tps_vc)%2: axs.flatten()[-1].axis('off')
            fig.suptitle(f'{kwargs.get("subs").label()}')

            for tp, ax in zip(tps_vc, axs.flatten()):
                plotting_function(ax=ax, tp=tp, **kwargs) 
            
            if vcoord=='p': 
                ax.invert_yaxis()

            if kwargs.get('xdate'): 
                fig.autofmt_xdate()
            fig.tight_layout()
            fig.subplots_adjust(top = 0.8 + math.ceil(len(tps_vc))/150)
            lines, labels = axs.flatten()[0].get_legend_handles_labels()
            fig.legend(lines, labels, loc='upper center', ncol=2,
                        bbox_to_anchor=[0.5, 0.94])
            plt.show()

    def timeseries_subs_STsorted(self, subs, tps = None):
        """ Plot timeseries of datapoints sorted into stratophere / troposphere for given tps

        Args:
            subs (dcts.Substance): Substance to be sorted & plotted
        
        Optional:
            key subs (dcts.Substance): substance to sort and plot
            key tps (list[dcst.Coordinates])
            key vcoords (list[dcts.Coordinates])
            popt0 / popt1 (tuple[float]): initial / filtered baseline fit parameters
        """

        tps = self.tps if tps is None else tps

        fig, axs = plt.subplots(math.ceil(len(tps)/2), 2, dpi=200,
                                figsize=(7, math.ceil(len(tps)/2)*2),
                                sharey=True, sharex=True)

        if len(tps)%2: axs.flatten()[-1].axis('off')
        fig.suptitle(subs.label())

        for tp, ax in zip(tps, axs.flatten()):
            self._ax_timeseries_subs_STsorted(ax=ax, subs=subs, tp=tp) 

        if tp.vcoord=='p': 
            ax.invert_yaxis()
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.subplots_adjust(top = 0.8 + math.ceil(len(tps))/150)
        lines, labels = axs.flatten()[0].get_legend_handles_labels()
        fig.legend(lines, labels, loc='upper center', ncol=2,
                    bbox_to_anchor=[0.5, 0.94])
        plt.show()

    def subs_coloring_ST_sorted(self, x_axis, y_axis, c_axis, **kwargs): 
        """ Plot x over y data with coloring based on substance mixing ratios 
        Red / Black dots indicate S/T sorting per tp. 
        
        Parameters: 
            x_axis (dcts.Coordinate or dcts.Substance)
            y_axis (dcts.Coordinate or dcts.Substance)
            c_axis (dcts.Substance): Values used to colour the datapoints according to mixing ratio 
        
            key tps (list[dcts.Coordinates]): TP defs for sorting
            key ylims (tuple[float]): Colormap limits
        """
        tps = kwargs.get('tps', self.tps)
        
        vlims = kwargs.get('vlims') or c_axis.vlims()
        norm = Normalize(*vlims)#np.nanmin(df[o3_subs.col_name]), np.nanmax(df[o3_subs.col_name]))
        cmap = plt.cm.viridis_r

        fig, axs = plt.subplots(math.ceil(len(tps)/2), 2, dpi=200,
                                figsize=(7, math.ceil(len(tps)/2)*2),
                                sharey=True, sharex=True)
        if len(tps)%2: axs.flatten()[-1].axis('off')
        # fig.suptitle(c_axis.label())

        for tp, ax in zip(tps, axs.flatten()):
            tp_tropo = self.df[self.df_sorted['tropo_'+tp.col_name] == True].copy()
            tp_strato = self.df[self.df_sorted['strato_'+tp.col_name] == True].copy()
            
            tp_tropo.dropna(subset = [c_axis.col_name], inplace=True)
            tp_strato.dropna(subset = [c_axis.col_name], inplace=True)
            
            c_tropo = cmap(norm(tp_tropo[c_axis.col_name]))
            c_strato = cmap(norm(tp_strato[c_axis.col_name]))
            
            ax.set_title(tp.label(filter_label=True), fontsize=8)
            ax.grid('both', ls = 'dashed', color = 'grey', lw = 0.5, zorder=0)

            ax.scatter(tp_strato.index if x_axis == 'time' else tp_strato[x_axis.col_name], 
                       tp_strato[y_axis.col_name],
                       color=c_strato, zorder = 0, 
                       )
            ax.scatter(tp_strato.index if x_axis == 'time' else tp_strato[x_axis.col_name], 
                       tp_strato[y_axis.col_name],
                       color = 'r', marker = 'x', s = 1, 
                       label = 'Stratosphere')

            ax.scatter(tp_tropo.index if x_axis == 'time' else tp_tropo[x_axis.col_name], 
                       tp_tropo[y_axis.col_name],
                       color=c_tropo, 
                       marker='o', zorder = 0,
                       )
            ax.scatter(tp_tropo.index if x_axis == 'time' else tp_tropo[x_axis.col_name], 
                       tp_tropo[y_axis.col_name],
                       color='k', marker='x', s = 1, 
                       label='Troposphere')
        
        ax.set_ylabel(y_axis.label())
        ax.set_xlabel('Time' if x_axis == 'time' else x_axis.label())

        if tp.vcoord=='p': 
            ax.invert_yaxis()
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.subplots_adjust(top = 0.8 + math.ceil(len(tps))/150, 
                            bottom = 0.15 + math.ceil(len(tps))/150)
        
        # fig.subplots_adjust(bottom = 0.2)
        cax = fig.add_axes([0.1, 0, 0.8, 0.1])
        cax.axis('off')
        
        fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), 
                     ax = cax, fraction = 0.6, aspect = 30,  
                     orientation = 'horizontal', 
                     label = c_axis.label())
        
        lines, labels = axs.flatten()[0].get_legend_handles_labels()
        fig.legend(lines, labels, loc='upper center', ncol=2,
                    bbox_to_anchor=[0.5, 0.94])
        plt.show()       

    def _ax_timeseries_subs_STsorted(self, ax, subs, tp): 
        """ Plot timeseries of subs mixing ratios with strato / tropo colours. 
        Parameters: 
            subs(dcts.Substance)
            vcoords (list[str]): vertical coordinates to plot
            tps (list[dcts.Coordinate]): tropopause definitions to use for sorting 
        """
        tp_tropo = self.df[self.df_sorted['tropo_'+tp.col_name] == True] #.dropna(axis=0, subset=[tp.col_name])
        tp_strato = self.df[self.df_sorted['strato_'+tp.col_name] == True] #.dropna(axis=0, subset=[tp.col_name])

        ax.set_title(tp.label(filter_label=True), fontsize=8)
        ax.scatter(tp_strato.index, tp_strato[subs.col_name],
                    c='grey',  marker='.', zorder=0, label='Stratosphere')
        ax.scatter(tp_tropo.index, tp_tropo[subs.col_name],
                    c='xkcd:kelly green',  marker='.', zorder=1, label='Troposphere')

    def subs_vs_ycoord_STsorted(self, subs, ycoord, tps = None, **kwargs): 
        """ Show distribution of tropospheric and stratospheric substance measurements. 
        
        Args: 
            subs (dcts.Substance): Substance to plot data fpor 
            ycoord (dcts.Coordinate): (Vertical) coordinate to plot on the y-axis
            tps (List[dcts.Coordinate]): Tropoause definitions to use for sorting 
            
            key joint (bool): Show datapoints that are always sorted into troposphere / stratosphere
            key heatmap (bool): Show distribution of always-tropo / always-strato points as heatmap
            key alpha (float): Marker transparency
            key marker (str): Marker type
            key s (int): Marker size
            key facecolors (str): Marker inner colour 
        """
        tps = self.tps if tps is None else tps
        if not ycoord: [ycoord] = self.get_coords(vcoord='pt', tp_def='nan', model='ERA5') # backup solution 

        no_of_axs = len(tps)
        if kwargs.get('joint'): no_of_axs += 1
        # if kwargs.get('heatmap'): no_of_axs += 2 # tropo / strato / colorbar

        fig, axs = plt.subplots(math.ceil(no_of_axs/2), 2, 
                                dpi=100,
                                figsize=(7, math.ceil(no_of_axs/2)*2),
                                sharey=True, sharex=True)

        if no_of_axs%2: 
            axs.flatten()[len(tps)].axis('off')
            axs.flatten()[-1].axis('off')
            
        fig.suptitle(subs.label())
        
        for tp, ax in zip(tps, axs.flatten()):
            df_tropo, df_strato = self._ax_subs_vs_ycoord_STsorted(
                ax=ax, subs=subs, ycoord=ycoord, tp=tp, **kwargs) 
        
        if kwargs.get('heatmap') or kwargs.get('joint'): 
            # indicate points that are always tropospheric no matter the TP definition 
            print('Joint tropospheric points : {}\nJoint stratospheric points: {}'.format(
                *[len(i) for i in self.unambiguously_sorted_indices(tps)] ) )
        
        if kwargs.get('joint'): 
            joint_ax = axs.flatten()[no_of_axs-1] # if not kwargs.get('heatmap') else no_of_axs-3]
            self.add_consistently_sorted(subs, ycoord, tps, joint_ax, **kwargs)
        
        if tp.vcoord=='p': 
            ax.invert_yaxis()
        if kwargs.get('xdate'): 
            fig.autofmt_xdate()

        fig.tight_layout()
        top = 0.8 + math.ceil(len(tps))/150
        lines, labels = axs.flatten()[0].get_legend_handles_labels()
        if kwargs.get('joint'): 
            joint_lines, joint_labels = joint_ax.get_legend_handles_labels()
            lines = lines + joint_lines
            labels = labels + joint_labels
            top = 0.75 + math.ceil(len(tps))/150
        fig.subplots_adjust(top = top)
        leg = fig.legend(lines, labels, loc='upper center', ncol=2,
                    bbox_to_anchor=[0.5, 0.94],
                    markerscale=2)
        for lh in leg.legend_handles: 
            lh.set_alpha(1)
        plt.show()
        
        if kwargs.get('heatmap'): 
            self.consistently_sorted_heatmap(subs, ycoord, tps,
                                             #axs.flatten()[-2:], 
                                             **kwargs)

    def add_consistently_sorted(self, subs, ycoord, tps, joint_ax, **kwargs): 
        """ Plot datapoints which are consistently sorted into tropo / strato onto given axis. """
        shared_tropo, shared_strato = self.unambiguously_sorted_indices(tps)
        joint_ax.axis('on')

        marker_params = dict(
            marker = '+',
            facecolors = 'none', 
            s = 2, alpha = 0.05)
        for key in marker_params: 
            if key in kwargs: 
                marker_params.update({key: kwargs[key]})
        
        joint_ax.scatter(self.df.loc[shared_tropo][subs.col_name], 
                    self.df.loc[shared_tropo][ycoord.col_name], 
                    c='xkcd:pink', # red
                    zorder=2, 
                    label='always Troposphere',
                    **marker_params)

        joint_ax.scatter(self.df.loc[shared_strato][subs.col_name], 
                    self.df.loc[shared_strato][ycoord.col_name], 
                    c='c', # cyan
                    zorder=1, 
                    label='always Stratosphere',
                    **marker_params)

        joint_ax.set_ylabel(ycoord.label())
        joint_ax.set_xlabel(subs.label())

        # Legend
        # joint_leg = joint_ax.legend(markerscale=4)
        # for lh in joint_leg.legend_handles: 
        #     lh.set_alpha(1)
        
        return joint_ax

    def consistently_sorted_heatmap(self, subs, ycoord, tps, axs = None, **kwargs):
        """ Create and add heatmap for distribution in given dataframe. 
        
        Args: 
            subs (dcts.Substance)
            ycoord (dcts.Coordinate)
            tps (List[dcts.Coordinate])
            axs (plt.Axes, plt.Axes): List of two axes for tropospheric and stratospheric plot
            
            key LogNorm (bool)
            key xbsize (float)
            key xbins (int)
            key ybsize (float)
            key ybins (int)
            key bins ( int or array_like or [int, int] or [array, array] )
        """
        if axs is None: 
            _, axs = plt.subplots(2, figsize = (8,5), sharex=True, sharey=True)
        
        H_tropo_ax, H_strato_ax = axs
        
        H_tropo_ax.set_title('Tropospheric')
        H_strato_ax.set_title('Stratospheric')

        cmap = dcts.dict_colors()['vcount'] # ['heatmap']
        cmap.set_bad('k', 0)
        
        
        data = self.df.dropna(subset = [subs.col_name])
        shared_tropo, shared_strato = self.unambiguously_sorted_indices(tps)
        
        shared_tropo = shared_tropo[shared_tropo.isin(data.index)]
        shared_strato = shared_strato[shared_strato.isin(data.index)]
        
        # define default bins
        xmin, xmax = data[subs.col_name].min(), data[subs.col_name].max()
        xbsize = 5 * ( np.ceil((xmax-xmin)/10) / 5 ) if 'xbsize' not in kwargs else kwargs.get('xbsize')
        xlims = (xmin - xbsize/2, xmax + xbsize/2)
        xbin_edges = np.linspace(*xlims, 100 if not 'xbins' in kwargs else kwargs.get('xbins'))
        
        ybsize = ycoord.get_bsize() if 'ybsize' not in kwargs else kwargs.get('ybsize')
        ylims = (data[ycoord.col_name].dropna().min() - ybsize, 
                data[ycoord.col_name].dropna().max() + 2*ybsize)
        ybin_edges = np.linspace(*ylims, 
                                 20 if not 'ybins' in kwargs else kwargs.get('ybins'))

        for ax, idx in zip([H_tropo_ax, H_strato_ax], 
                           [shared_tropo, shared_strato]): 
            heatmap, xedges, yedges = np.histogram2d(data[subs.col_name].loc[idx],
                                                     data[ycoord.col_name].loc[idx], 
                                                     bins=(xbin_edges, ybin_edges) if not 'bins' in kwargs else kwargs.get('bins'))
            heatmap[heatmap == 0] = np.nan # filter out bins without data
            img = ax.pcolormesh(xedges, yedges, 
                                heatmap.T, 
                                norm = None if not kwargs.get('LogNorm') else LogNorm(), 
                                cmap=cmap)
            
            ax.set_ylabel(ycoord.label())
        ax.set_xlabel(subs.label())

        cbar = plt.colorbar(img, ax = axs, orientation='vertical')
        cbar.set_label(f'Distribution of {subs.label(True)} msmts [#]')

    def _ax_subs_vs_ycoord_STsorted(self, ax, subs, ycoord, tp, popt0=None, popt1=None, **kwargs):
        """ Plot strat / trop sorted data """
        # only take data with index that is available in df_sorted
        data = self.df[self.df.index.isin(self.df_sorted.index)]
        data.sort_index(inplace=True)

        tropo_col = 'tropo_'+tp.col_name
        strato_col = 'strato_'+tp.col_name

        # take 'data' here because substances may not be available in df_sorted
        df_tropo = data[self.df_sorted[tropo_col] == True]
        df_strato = data[self.df_sorted[strato_col] == True]

        ax.set_title(tp.label(True))#' filter on {subs.label()} data')

        marker_params = dict(
            marker = '+',
            # facecolors = 'none', 
            s = 2,
            alpha = 0.05,
            )
        for key in marker_params: 
            if key in kwargs: 
                marker_params.update({key: kwargs[key]})

        tropos_c = dict(c = 'tab:red')
        stratos_c = dict(c = 'tab:orange')

        ax.scatter(df_tropo[subs.col_name], 
                df_tropo[ycoord.col_name], 
                label='Troposphere',
                zorder=1, 
                **tropos_c, 
                **marker_params)
        
        ax.scatter(df_strato[subs.col_name], 
                df_strato[ycoord.col_name], 
                label='Stratosphere',
                zorder=0,
                **stratos_c,
                **marker_params)

        if popt0 is not None and popt1 is not None and subs.short_name == tp.crit:
            # only plot baseline for chemical tropopause def and where crit is being plotted
            t_obs_tot = np.array(dt_to_fy(self.df_sorted.index, method='exact'))
            ls = 'solid' if tp.ID == subs.ID else 'dashed'

            func = dcts.get_subs(col_name=tp.col_name).function
            ax.plot(self.df_sorted.index, func(t_obs_tot-2005, *popt0),
                    c='r', lw=1, ls=ls, label='initial')
            ax.plot(self.df_sorted.index, func(t_obs_tot-2005, *popt1),
                    c='k', lw=1, ls=ls, label='filtered')

        ax.set_ylabel(ycoord.label())
        ax.set_xlabel(subs.label())
        
        return df_tropo, df_strato

# --- Tables of (seasonal & average) binned standard deviation ----
    def calculate_seasonal_stdv_dict(self, subs, tps, **kwargs): 
        """ Calculate seasonal standard deviations for the given substance and tropopause definitions

        Parameter:
            subs (Substance): Substance for which to calculate variability
            tps (List[Coordinate]): Tropopause definitions to calculate atmos.layer variability for
            
            key rel (bool): Calculate relative instead of absolute standard deviation
        """
        self.df['season'] = tools.make_season(self.df.index.month)
        stdv_df_dict = {}

        for season in set(self.df.season):
            stdv_df = self.sel_season(season).strato_tropo_stdv(subs, tps)
            stdv_df = stdv_df[[c for c in stdv_df.columns if 'stdv' in c]]

            if kwargs.get('rel'): 
                stdv_df = stdv_df[[c for c in stdv_df.columns if 'rel' in c]]
            else: 
                stdv_df = stdv_df[[c for c in stdv_df.columns if 'rel' not in c]]
            
            stdv_df_dict[season] = stdv_df.rename(columns = {c:c +f'_{season}' for c in stdv_df.columns})
        return stdv_df_dict

    def make_stdv_table(self, ax, subs, stdv_df, **kwargs):
        """ Create table for values of standard deviation in troposphere / stratosphere.

        Parameters:
            ax (matplotlib.axes.Axes): Axis to be plotted on
            subs (Substance): Substance for which standard deviation was calculated
            stdv_df (pd.DataFrame): Dataframe containing standard deviation data
            
            key rel (bool): Calculate relative instead of absolute standard deviation
            key prec (int): Precision of displayed values in table view
            key NoColor (bool): Toogle filling in of color on tables 

        Returns:
            edited matplotlib.axes.Axes object, matplotlib.axes.table instance 
        """
        if 'prec' in kwargs:
            prec = kwargs.get('prec')
        else:  
            prec = 1 if kwargs.get('rel') else 2

        stdv_df = stdv_df.astype(float).round(prec)
        stdv_df.sort_index()
        
        cellValues = [[ '{:.{prec}f}'.format(j, prec=prec) 
                for j in i] for i in stdv_df.values]
              
        rowLabels = [dcts.get_coord(col_name=c).label(True) for c in stdv_df.index]
        colLabels = [f'Troposphere [{subs.unit}]', f'Stratosphere [{subs.unit}]'
                     ] if not kwargs.get('rel') else [
                         'Troposphere [%]', 'Stratosphere [%]']

        norm_t = Normalize(np.min(stdv_df.values[:,0]) * 0.95, np.max(stdv_df.values[:,0]) * 1.15)
        cmap_t = dcts.dict_colors()['vstdv_tropo']

        norm_s = Normalize(np.min(stdv_df.values[:,1]) * 0.95, np.max(stdv_df.values[:,1]) * 1.15)
        cmap_s = dcts.dict_colors()['vstdv_strato']

        cellColors = pd.DataFrame([[cmap_t(norm_t(i)) for i in stdv_df.values[:,0]], 
                                [cmap_s(norm_s(i)) for i in stdv_df.values[:,1]]]
                                ).transpose().values

        table = ax.table(cellValues, 
                         rowLabels = rowLabels, 
                         # rowColours = ['xkcd:light grey']*len(rowLabels),
                         colLabels = colLabels,
                         # colColours = ['xkcd:light grey']*len(colLabels),
                         cellLoc = 'center',
                         loc='center',
                         cellColours = cellColors if not kwargs.get('NoColor') else None,
                         )
        table.set_fontsize(15)
        table.scale(1,3)
        ax.axis('off')

        return ax, table

    def total_stdv_table(self, subs, tps=None, **kwargs):
        """ Creates a table of the overall variability for stratos / tropos distribution using TP definitions. 
        Parameters: 
            subs (dcst.Substance)
            tps (list[dcts.Coordinate])

            key rel (bool): Calculate relative instead of absolute standard deviation
            key prec (int): Precision of displayed values in table view
            key NoColor (bool): Toogle filling in of color on tables 
        """
        tps = self.tps if not tps else tps
        stdv_df = self.strato_tropo_stdv(subs, tps)
        stdv_df = stdv_df[[c for c in stdv_df.columns if 'stdv' in c]]
        
        if kwargs.get('rel'): 
            stdv_df = stdv_df[[c for c in stdv_df.columns if 'rel' in c]]
        else: 
            stdv_df = stdv_df[[c for c in stdv_df.columns if 'rel' not in c]]
              
        fig, ax = plt.subplots(dpi=250)
        fig.suptitle(('Relative ' if kwargs.get('rel') else '') + 'Varibility of ' + subs.label())
        ax,_ = self.make_stdv_table(ax, subs, stdv_df, **kwargs)
        fig.show()

    def seasonal_stdv_tables(self, subs, tps=None, **kwargs):
        """ Calculate and display table of variability per season and RMS.
 
        Parameters: 
            subs (Substance): Substance for which to calculate variability
            tps (List[Coordinate]): Tropopause definitions to calculate atmos.layer variability for
            
            key rel (bool): Calculate relative instead of absolute standard deviation
            key prec (int): Precision of displayed values in table view
            key NoColor (bool): Toogle filling in of color on tables 
        """
        stdv_df_dict = self.calculate_seasonal_stdv_dict(subs, tps, **kwargs)
        
        for season in stdv_df_dict: 
            fig, ax = plt.subplots(dpi=250)
            stdv_df = stdv_df_dict[season]
            ax,_ = self.make_stdv_table(ax, subs, stdv_df, **kwargs)
            
            ax.set_title('Variability of {} in {}'.format(
                subs.label(),
                dcts.dict_season()[f'name_{season}'] ))
        
        # Show average and RMS seasonal averages 
        self.average_seasonal_stdv_table(subs, stdv_df_dict, **kwargs)
        self.rms_seasonal_stdv_table(subs, stdv_df_dict, **kwargs)

    def average_seasonal_stdv_table(self, subs, stdv_df_dict, **kwargs):
        """ Create table showing average seasonal standard deviation 

        Parameters:
            subs (dcts.Substance): Substance
            stdv_df_dict (dict): Binned seasonal standard deviation data
            
            key rel (bool): Calculate relative instead of absolute standard deviation
            key prec (int): Precision of displayed values in table view
            key NoColor (bool): Toogle filling in of color on tables 
        """
        df = pd.concat(stdv_df_dict.values(), axis=1)
        strato_cols = [c for c in df.columns if 'strato_stdv' in c]
        tropo_cols = [c for c in df.columns if 'tropo_stdv' in c]

        # Average of seasonal relative standard deviation
        df['strato_stdv_av'] = df[strato_cols].sum(axis=1) / len(stdv_df_dict)
        df['tropo_stdv_av'] = df[tropo_cols].sum(axis=1) / len(stdv_df_dict)
        
        df_av = df[['tropo_stdv_av', 'strato_stdv_av']]
        df_av = df_av.astype(float).round(3 if kwargs.get('rel') else 3)
        
        fig, ax = plt.subplots(dpi=250)
        ax,_ = self.make_stdv_table(ax, subs, df_av, **kwargs)
        ax.set_title('Average {}seasonal variability of {}'.format(
            'relative ' if kwargs.get('rel') else '',
            subs.label()))
        fig.show()

    def rms_seasonal_stdv_table(self, subs, stdv_df_dict, **kwargs):
        """ Create table showing Root-mean-square of seasonal (relative) standard deviation
        RMS = ( 1/n * (x_1**2 + x_2**2 + ... + x_n**2) )**0.5

        Parameters:
            subs (dcts.Substance): Substance 
            stdv_df_dict (dict): Binned seasonal standard deviation data 
            
            key rel (bool): Calculate relative instead of absolute standard deviation
            key prec (int): Precision of displayed values in table view
            key NoColor (bool): Toogle filling in of color on tables 
        """
        df = pd.concat(stdv_df_dict.values(), axis=1)
        strato_cols = [c for c in df.columns if 'strato_stdv' in c]
        tropo_cols = [c for c in df.columns if 'tropo_stdv' in c]
        
        df['strato_stdv_RMS'] = ((df[strato_cols] **2 ).sum(axis=1) / len(stdv_df_dict) )**0.5
        df['tropo_stdv_RMS'] = ((df[tropo_cols] **2 ).sum(axis=1) / len(stdv_df_dict) )**0.5
        
        df_RMS = df[['tropo_stdv_RMS', 'strato_stdv_RMS']]
        df_RMS = df_RMS.astype(float).round(3)
        
        fig, ax = plt.subplots(dpi=250)
        ax,_ = self.make_stdv_table(ax, subs, df_RMS, **kwargs)
        ax.set_title('RMS of {}seasonal variability of {}'.format(
            'relative ' if kwargs.get('rel') else '',
            subs.label()))
        fig.show()

    # --- Matrix plots per substance for tropopause definitions --- 
    def create_lat_binned_matrix_plot(self, subs, tps, atm_layer, bin_attr, **kwargs): 
        """ Find matrix values for the given substance to compare tropoause definitions. 
        
        Args:
            subs (dcts.Substance): Substance to be investigated
            tps (list[dcts.Coordinate]): Tropopause definitions 
            atm_layer (str): Atmospheric layer i.e. tropo / strato / LMS
            bin_attr (str): vstdv / rvstd / vmean
            
            key lat_bmin/lat_bmax: Minimum / Maximum latitude values for binning
            
        """
        shared_indices = self.get_shared_indices(tps=tps)
        
        lat_bmin = 30 if not 'lat_bmin' in kwargs else kwargs.get('lat_bmin')
        lat_bmax = 90 if not 'lat_bmax' in kwargs else kwargs.get('lat_bmax')
        lat_bci = bp.Bin_equi1d(lat_bmin, lat_bmax, self.grid_size)
        
        # initialise output variables 
        # out_dict = {}
        values = np.full((len(tps), lat_bci.nx), np.nan)
        av_values = np.full(len(tps), np.nan)
        
        [lat_coord] = self.get_coords(hcoord='lat')
        
        # 2. Separate into stratospheric / tropospheric for each tps
        for i, tp in enumerate(tps): 
            if atm_layer == 'tropo': 
                data = self.sel_tropo(**tp.__dict__).df
            elif atm_layer == 'strato': 
                data = self.sel_strato(**tp.__dict__).df
            elif atm_layer == 'LMS': 
                data = self.sel_LMS(**tp.__dict__).df

            data = data[data.index.isin(shared_indices)]
            
            # 3. Latitude binning for each tps and atm. layer 
            binned = self.bin_1d(subs, lat_coord, bci_1d=lat_bci, xbsize=self.grid_size, df=data)
            # out_dict[tp.col_name] = binned
            vals = getattr(binned, bin_attr)
            values[i] = vals
            
            # 4. Calcualte weighted average across latitudes for each tp and atm. layer
            weighted_average = np.average(vals[~ np.isnan(vals)], 
                                          weights = binned.vcount[~ np.isnan(vals)])
            av_values[i] = weighted_average
        
        return values, av_values

    def create_lon_binned_matrix_plot(self, subs, tps, atm_layer, bin_attr, **kwargs): 
        """ Find matrix values for the given substance to compare tropoause definitions. 
        
        Args:
            subs (dcts.Substance): Substance to be investigated
            tps (list[dcts.Coordinate]): Tropopause definitions 
            atm_layer (str): Atmospheric layer i.e. tropo / strato / LMS
            bin_attr (str): vstdv / rvstd / vmean
            
            key lon_bmin/lon_bmax: Minimum / Maximum longitude values for binning
            
        """
        shared_indices = self.get_shared_indices(tps=tps)
        
        lon_bmin = 30 if not 'lon_bmin' in kwargs else kwargs.get('lon_bmin')
        lon_bmax = 90 if not 'lon_bmax' in kwargs else kwargs.get('lon_bmax')
        lon_bci = bp.Bin_equi1d(lon_bmin, lon_bmax, self.grid_size)
        
        # initialise output variables 
        # out_dict = {}
        values = np.full((len(tps), lon_bci.nx), np.nan)
        av_values = np.full(len(tps), np.nan)
        
        [lon_coord] = self.get_coords(hcoord='lon')
        
        # 2. Separate into stratospheric / tropospheric for each tps
        for i, tp in enumerate(tps): 
            if atm_layer == 'tropo': 
                data = self.sel_tropo(**tp.__dict__).df
            elif atm_layer == 'strato': 
                data = self.sel_strato(**tp.__dict__).df
            elif atm_layer == 'LMS': 
                data = self.sel_LMS(**tp.__dict__).df

            data = data[data.index.isin(shared_indices)]
            
            # 3. Latitude binning for each tps and atm. layer 
            binned = self.bin_1d(subs, lon_coord, bci_1d=lon_bci, xbsize=self.grid_size, df=data)
            # out_dict[tp.col_name] = binned
            vals = getattr(binned, bin_attr)
            values[i] = vals
            
            # 4. Calcualte weighted average across latitudes for each tp and atm. layer
            weighted_average = np.average(vals[~ np.isnan(vals)], 
                                          weights = binned.vcount[~ np.isnan(vals)])
            av_values[i] = weighted_average
        
        return values, av_values

    def matrix_plot_stdev_subs(self, subs,  note='', tps=None, savefig=False, **kwargs):
        """
        Create matrix plot showing variability per latitude bin per tropopause definition

        Parameters:
            subs (dcts.Substance)
            tps (list[dcts.Coordinate])
            
            key prec (int): Number of decimals to display 
        """
        # 1. Prepare data and variables
        bin_attr = 'vstdv' if not 'bin_attr' in kwargs else kwargs.pop('bin_attr')
        tps = self.tps if not tps else tps
        if not 'df_sorted' in self.data: 
            self.df_sorted

        if 'prec' in kwargs:
            prec = kwargs.get('prec')
        else:  
            prec = 1 if bin_attr == 'rvstd' else 2
        
        lat_bmin = 30 if not 'lat_bmin' in kwargs else kwargs.get('lat_bmin')
        lat_bmax = 90 if not 'lat_bmax' in kwargs else kwargs.get('lat_bmax')
        lat_bci = bp.Bin_equi1d(lat_bmin, lat_bmax, self.grid_size)
        
        # Prepare the figure
        pixels = self.grid_size # how many pixels per imshow square
        yticks = np.linspace(0, (len(tps)-1)*pixels, num=len(tps))[::-1] # order was reversed for some reason
        tp_labels = [tp.label(True)+'\n' for tp in tps]
        xticks = np.arange(lat_bmin, lat_bmax+self.grid_size, self.grid_size)

        fig = plt.figure(dpi=200, 
                         figsize=(lat_bci.nx*0.825, len(tps)*2))

        gs = gridspec.GridSpec(5, 2, figure=fig,
                            height_ratios = [1, 0.1, 0.02, 1, 0.1],
                            width_ratios = [1, 0.09])
        axs = gs.subplots()

        [ax.remove() for ax in axs[2, 0:]]
        middle_ax = plt.subplot(gs[2, 0:])
        middle_ax.axis('off')
        
        # Plot matrices for stratospheric and tropospheric values
        for atm_layer in ['strato', 'tropo']:
            # Get data for the current atmospheric layer
            if atm_layer == 'strato': 
                ax1 = axs[0,0]
                ax2 = axs[0,1]
                [ax.remove() for ax in  axs[1, 0:]]
                cax = plt.subplot(gs[1, 0:])
            else: 
                ax1 = axs[3,0]
                ax2 = axs[3,1]
                [ax.remove() for ax in axs[4, 0:]]
                cax = plt.subplot(gs[4, 0:])
            
            values, av_values = self.create_lat_binned_matrix_plot(subs, tps, atm_layer, bin_attr, **kwargs)
            if bin_attr == '  ': 
                values, av_values = values *100, av_values*100 # make into percentages !! 
                            
            # Define variables for colormapping
            vmin, vmax = subs.vlims(bin_attr, atm_layer)
            norm = Normalize(vmin, vmax) 
            cmap = dcts.dict_colors()[f'{bin_attr}_{atm_layer}'] # create colormap
            
            value_type = 'variability' if bin_attr == 'vstdv' else 'relative variability'
            ax1.set_title(f'{atm_layer[0].upper()}{atm_layer[1:]}spheric {value_type} of {subs.label()}{note}', fontsize=14)

            img = ax1.matshow(values, alpha=0.75,
                              extent = [lat_bmin, lat_bmax,
                                        0, len(tps)*pixels],
                              cmap=cmap, norm=norm)
            
            # Add descriptions for tropopause definitions and latitude bins
            ax1.set_xlabel('Latitude [°N]')
            ax1.set_xticks(xticks)
            ax1.set_yticks(yticks, labels=tp_labels)
            ax1.tick_params(axis='x', top=False, labeltop=False, labelbottom=True)
            for label in ax1.get_yticklabels():
                label.set_verticalalignment('bottom')
            ax1.grid('both')

            # add numeric values
            for j,x in enumerate(xticks[:-1]):
                for i,y in enumerate(yticks):
                    value = values[i,j]
                    if str(value) != 'nan':
                        ax1.text(
                            x+0.5*self.grid_size,
                            y+0.5*pixels,
                            '{0:.{prec}f}'.format(value, prec=prec) if value>vmax/100 \
                                else '<{0:.{prec}f}'.format(vmax/100, prec=prec),
                            va='center', ha='center')

            # Create and format colorbar 
            cbar = plt.colorbar(img, cax=cax, orientation='horizontal')
            cbar.set_label(f'Standard deviation of {subs.label(name_only=True)} within bin [{subs.unit}]')
            # make sure vmin and vmax are shown as colorbar ticks
            cbar_vals = cbar.get_ticks()
            cbar_vals = [vmin] + cbar_vals[1:-1].tolist() + [vmax]
            cbar.set_ticks(cbar_vals)

            # Add values for average variability across latitude bins (weighted)
            img = ax2.matshow(np.array([av_values]).T, alpha=0.75,
                            extent = [0, self.grid_size,
                                    0, len(tps)*pixels],
                            cmap = cmap, norm=norm)
            for i,y in enumerate(yticks): 
                value = av_values[i]
                if str(value) != 'nan':
                    ax2.text(0.5*self.grid_size,
                            y+0.5*pixels,
                            '{0:.{prec}f}'.format(value, prec=prec) if value>vmax/100 \
                                else '<{0:.{prec}f}'.format(vmax/100, prec=prec),
                            va='center', ha='center')
            ax2.tick_params(axis='both', bottom=False, top=False, labeltop=False, left=False, labelleft=False)
            ax1.grid('both')
            ax2.set_xlabel('Average')

        fig.tight_layout()
        fig.subplots_adjust(top=0.8)

        if savefig:
            plt.savefig(f'E:/CARIBIC/Plots/variability_lat_binned/variability_{subs.col_name}.png', format='png')
        fig.show()

    def create_3d_matrix_plot(self, subs, tps, bin_attr, **kwargs): 
        """ Create matrix plot for 3D-binned data. """
        pass

    def matrix_3d_binned(self, subs, tps, bin_attr=False, **kwargs): 
        """ Create table for seasonally and inner-atmospheric layer binned substance data. 
        
        1. (if seasonal): Separate data into seasons
        2. data is binned into lon / (eq.) lat / tp cells
            x - lon, y - (eq.) lat, z - tp
        3. The bin_attr of those boxes is found
        4. Visualisation of the calculated quantities 
        
        Args: 
            subs (dcts.Substance)
            tps (List[dcts.Coordinate])
            bin_attr (str): e.g. vmean, vstdv, rvstd
            
            key eql (bool): Take equivalent latitude instead of latitude
            key seasonal (bool). Separate data into seasons before making calculations
        """

        for tp in self.tps if not tps else tps: 
            binned_3d = self.bin_3d(subs, tp, 
                                    eql = (True if kwargs.get('eql') else False) )
            data = getattr(binned_3d, bin_attr)
            
            for ix in range(binned_3d.nx):
                for iy in range(binned_3d.ny): 
                    for iz in range(binned_3d.nz):
                        datapoint = data[ix, iy, iz]
                        
        # what do I want? 
        # lat matrix but with data separated into potential temperature bins as well? 
        # or sth else? 
        
        # zonal mean 
        zonal_mean = np.nanmean(data, axis=1) 
        # get the zonal mean of the standard deviation of the 3D cells (not equivalant to 2D binning!)
        # show same as Millan23 or as boxenplots 
        
        for ix in range(binned_3d.nx): 
            for iz in range(binned_3d.nz):
                datapoint = zonal_mean[ix, iz]

# %%
