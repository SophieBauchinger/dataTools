# -*- coding: utf-8 -*-
""" Stats comparison for binned data - mean/mode, interval widths. 

@Author: Sophie Bauchinger, IAU
@Date Fri Jan 15 15:00:00 2025

"""
from copy import deepcopy
import math
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

import dataTools.dictionaries as dcts
import dataTools.data.BinnedData as bin_tools
import dataTools.plot.create_figure as cfig

def subs_ST_sorted(self, x_axis, y_axis, **kwargs):
    """ Plot x over y data
    Grey / Pink dots indicate S/T sorting per tp. 
    
    Parameters: 
        x_axis (dcts.Coordinate or dcts.Substance)
        y_axis (dcts.Coordinate or dcts.Substance)
    
        key tps (list[dcts.Coordinates]): TP defs for sorting
        key ylims (tuple[float]): Colormap limits
    """
    tps = kwargs.get('tps', self.tps)

    c_dict = dict(tropo = 'm', strato='xkcd:charcoal grey') # color of atm_layer indicators
    l_dict = dict(tropo = 'Troposphere', strato = 'Stratosphere') # Labels

    fig, axs = cfig.tp_comp_plot(tps, **kwargs)

    for tp, ax in zip(tps, axs.flatten()):
        ax.set_title(tp.label(filter_label=True))
        ax.grid('both', ls = 'dashed', color = 'grey', lw = 0.5, zorder=0)
        
        for atm_layer in ('strato', 'tropo'): 
            tp_df = self.sel_atm_layer(atm_layer, tp).df
            
            x = tp_df.index if x_axis == 'time' else tp_df[x_axis.col_name]
            y = tp_df[y_axis.col_name]

            ax.scatter(x, y, c = c_dict[atm_layer],
                       marker = '.', s = kwargs.get('s', 2), 
                       label = l_dict[atm_layer])

    for ax in [axs.flat[0], axs[-1,0]]: 
        ax.set_ylabel(y_axis.label())
    for ax in axs[-1, :]:
        ax.set_xlabel('Time' if x_axis == 'time' else x_axis.label())

    if tp.vcoord=='p': 
        ax.invert_yaxis()
    if x_axis == 'time': 
        fig.autofmt_xdate()
    fig.tight_layout()
    fig.subplots_adjust(top = 0.8 + math.ceil(len(tps))/150)
    
    lines, labels = axs.flat[0].get_legend_handles_labels()    
    fig.legend(lines[::-1], labels[::-1], 
               loc='upper center', ncol=2,
               bbox_to_anchor=[0.5, 0.94], 
               markerscale=6)
    return fig, axs

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
    
    c_size = kwargs.get('c_size', 100)
    c_dict = dict(tropo = 'm', strato='xkcd:charcoal grey') # color of atm_layer indicators
    l_dict = dict(tropo = 'Troposphere', strato = 'Stratosphere') # Labels

    fig, axs = plt.subplots(math.ceil(len(tps)/2), 2, dpi=200,
                            figsize=(7, math.ceil(len(tps)/2)*2),
                            sharey=True, sharex=True)
    if len(tps)%2: axs.flatten()[-1].axis('off')

    for tp, ax in zip(tps, axs.flatten()):
        ax.set_title(tp.label(filter_label=True), fontsize=8)
        ax.grid('both', ls = 'dashed', color = 'grey', lw = 0.5, zorder=0)
        
        for atm_layer in ('strato', 'tropo'): 
            tp_df = self.sel_atm_layer(atm_layer, tp).df
            tp_df.dropna(subset = [c_axis.col_name], inplace = True)
            
            x = tp_df.index if x_axis == 'time' else tp_df[x_axis.col_name]
            y = tp_df[y_axis.col_name]
            c = tp_df[c_axis.col_name]
            
            ax.scatter(x, y , c = c, 
                       marker = '.', alpha = 0.6,
                       cmap = cmap, norm = norm, 
                       s = c_size, lw = 0)
            
            ax.scatter(x, y, c = c_dict[atm_layer],
                       marker = '.', s = 2, 
                       label = l_dict[atm_layer])

    for ax in [axs.flat[0], axs[-1,0]]: 
        ax.set_ylabel(y_axis.label())
    axs[-1, 0].set_xlabel('Time' if x_axis == 'time' else x_axis.label())

    if tp.vcoord=='p': 
        ax.invert_yaxis()
    if x_axis == 'time': 
        fig.autofmt_xdate()
    fig.tight_layout()
    fig.subplots_adjust(top = 0.8 + math.ceil(len(tps))/150, 
                        bottom = 0.15 + math.ceil(len(tps))/150)
    cax = fig.add_axes([0.1, 0, 0.8, 0.1])
    cax.axis('off')
    
    fig.colorbar(
        ScalarMappable(norm=norm, cmap=cmap), 
        ax = cax, fraction = 0.6, aspect = 30,  
        orientation = 'horizontal', 
        label = c_axis.label())

    handles, labels = axs.flatten()[0].get_legend_handles_labels()
    fig.legend(handles[::-1], labels[::-1], loc='upper center', ncol=2,
                bbox_to_anchor=[0.5, 0.94], markerscale=4)
    plt.show()

#%% subs ST sorted with binned profiles shown on top 
def st_sorted_with_gradient(self, subs, coord, **kwargs): 
    """ Plot mixing ratios in background and gradient with vstdv on top. """
    tps = kwargs.get('tps', self.tps)
    
    c_dict = dict(tropo = 'm', strato='xkcd:charcoal grey') # color of atm_layer indicators
    c_scatter_dict = dict(tropo = 'xkcd:pink', strato = 'xkcd:grey')
    l_dict = dict(tropo = 'Troposphere', strato = 'Stratosphere') # Labels

    # Make figure
    fig, axs = plt.subplots(math.ceil(len(tps)/2), 2, dpi=500,
                            figsize=(6, math.ceil(len(tps)/2)*2),
                            sharey=True, sharex=True)
    if len(tps)%2: axs.flat[-1].axis('off')

    for tp, ax in zip(tps, axs.flatten()):
        ax.set_title(tp.label(filter_label=True))
        ax.grid('both', ls = 'dashed', color = 'grey', lw = 0.5, zorder=0)
        
        for atm_layer in ('strato', 'tropo'): 
            tp_subset =  self.sel_atm_layer(atm_layer, tp)
            tp_df = tp_subset.df
            
            # Plot background scatter
            ax.scatter(tp_df[subs.col_name], 
                       tp_df[coord.col_name], 
                       c = c_scatter_dict[atm_layer],
                       marker = '.', s = kwargs.get('bgd_s', 20), 
                    #    label = l_dict[atm_layer],
                       alpha = 0.8, zorder = 1)
            
            # Plot gradient
            bin_obj = bin_tools.binning(
                tp_subset.df, subs, coord, **kwargs)
            vmean = getattr(bin_obj, 'vmean')
            y = bin_obj.xintm
            
            color = c_dict[atm_layer]
            marker = 'd'

            # Errorbars
            (_, caps, _) = ax.errorbar(
                vmean, y, 
                xerr = bin_obj.vstdv, 
                c = color, lw = 0.5, 
                capsize = 1.5, zorder = 3)
            for cap in caps: 
                cap.set_markeredgewidth(1)
                cap.set(alpha=1, zorder=20)

            # Lines
            ax.plot(
                vmean, y, 
                marker=marker,  
                markersize = kwargs.get('diamond_s', 2), 
                c = color, # label = label,
                linewidth=1,
                label = l_dict[atm_layer],
                zorder = 2)
    
    # Set axis labels 
    for ax in [axs.flat[0], axs[-1,0]]: 
        ax.set_ylabel(coord.label())
    for ax in axs[-1, :]:
        ax.set_xlabel(subs.label())

    fig.tight_layout()
    fig.subplots_adjust(top = 0.8 + math.ceil(len(tps))/150)
    
    # Make legend
    lines, labels = axs.flat[0].get_legend_handles_labels()
    
    fig.legend(
        lines[::-1], labels[::-1], 
        loc='upper center', ncol=2,
        bbox_to_anchor=[0.5, 0.94])
    return fig, axs

#%% 
def plot_1d_gradient(ax, s, bin_obj,
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

#%% 2D binning warum auch immer ich das jetzt hier rein packe
def seasonal_2d_plots(self, subs, xcoord, ycoord, bin_attr, **kwargs):
    """
    Parameters:
        bin_attr (str): 'vmean', 'vstdv', 'vcount'
        key v/x/ylims (tuple[float])
    """
    
    try: 
        cmap = dcts.dict_colors()[bin_attr]
    except: 
        cmap = plt.cm.viridis

    binned_seasonal = bin_tools.seasonal_binning(self.df, subs, xcoord, ycoord, **kwargs)

    if not any(bin_attr in bin2d_inst.__dict__ for bin2d_inst in binned_seasonal.values()):
        raise KeyError(f'\'{bin_attr}\' is not a valid attribute of Bin2D objects.')

    vlims = kwargs.get('vlims', self.get_vlimit(subs, bin_attr))
    xlims = kwargs.pop('xlims', xcoord.get_lims())
    ylims = kwargs.pop('ylims', ycoord.get_lims())
    
    # vlims, xlims, ylims = self.get_limits(subs, xcoord, ycoord, bin_attr)
    norm = Normalize(*vlims)
    fig, axs = plt.subplots(2, 2, dpi=500, figsize=(8,9),
                            sharey=True, sharex=True)

    fig.subplots_adjust(top = 1.1)
    
    for season, ax in zip(binned_seasonal.keys(), axs.flatten()):
        bin2d_inst = binned_seasonal[season]
        ax.set_title(dcts.dict_season()[f'name_{season}'])
        
        img = self.single_2d_plot(ax, bin2d_inst, bin_attr, xcoord, ycoord, 
                                    cmap, norm, xlims, ylims, **kwargs)
        ax.set_xlim(*xlims)
        ax.set_ylim(*ylims)

    fig.subplots_adjust(right=0.9)
    fig.tight_layout(pad=2.5)

    cbar = fig.colorbar(img, ax = axs.ravel().tolist(), aspect=30, pad=0.08, 
                        orientation='horizontal', 
                        # location='top', ticklocation='bottom'
                        )
    cbar.ax.set_xlabel(subs.label(bin_attr=bin_attr))

    cbar_vals = cbar.get_ticks()
    cbar_vals = [vlims[0]] + cbar_vals[1:-1].tolist() + [vlims[1]]
    cbar.ax.tick_params(bottom=True, top=False)
    cbar.set_ticks(ticks = cbar_vals) #, labels=cbar_vals, ticklocation='bottom')

    plt.show()

