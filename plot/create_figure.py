# -*- coding: utf-8 -*-
"""
@Author: Sophie Bauchinger, IAU
@Date: Fri Dec 20 16:40:00 2024

Templates for complex figure layouts and legends. 
Includes helpers for large datasets and special plot components. 

"""
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import math
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.lines as mlines
from matplotlib.patches import Patch
import numpy as np
import plotly.graph_objs as go
from plotly_resampler import FigureResampler

import dataTools.dictionaries as dcts
from dataTools import tools

#%% Helpers
def save(fig, fname, path=None):
    """ Save figures as pdf with 300dpi and tight bbox_inches -> default path """
    if not path: 
        path = 'C:\\Users\\sophie_bauchinger\\Documents\\tropopause paper\\FIGURES_pdf\\'
    if not len(fname.split('.'))==2: 
        fname = fname+".pdf"
    fig.savefig(path+fname, dpi=300, bbox_inches='tight')
    print(f"Saved figure as {path+fname}")

#%% Big data figure resampling
def big_data_go_resample(df, x_cols, y_cols, fig=None): 
    """ Use plotly figure resampler to display big datasets. """
    # Prep data (x needs to increase monotonically)
    if type(y_cols)==str: 
        y_cols = [y_cols]
    if type(x_cols)==str:
        x_cols = [x_cols]
    if not fig: 
        fig = FigureResampler(go.Figure())
    for x_c in set(x_cols):
        df_sorted = df.dropna(subset=x_c).sort_values(x_c)
        if len(df_sorted.dropna(subset = [x_c]))==0: continue
        for y_c in set(y_cols).difference({x_c}):
            if len(df_sorted.dropna(subset = [y_c]))==0: continue
            note = '' if not 'STN' in x_c else f"/ {x_c.split('_STN')[1]}"
            fig.add_trace(
                go.Scattergl(name = f"{y_c.split('_S')[0]} / {x_c.split('_S')[0]} {note}",
                             mode='markers', showlegend=True),
                hf_x=df_sorted[x_c], 
                hf_y=df_sorted[y_c],
        )
    fig.update_yaxes(type="log")
    return fig

#%% Basic map / plotting helpers
def outline(lw=2): 
    """ Helper function to add outline to lines in plots. 
    lw (float): Linewidth of the line to outline. 
    """
    import matplotlib.patheffects as mpe
    return mpe.withStroke(linewidth=lw+1, foreground='white')

def add_zero_line(ax):
    """ Highlight the gridline at 0 for the chosen axis on the given Axes object.

    NB! Call when everything else has been plotted and adjusted already, otherwise the limits will be messy 
    (and it may highlight the wrong line).
    """
    zero_lines = np.delete(ax.get_ygridlines(), ax.get_yticks() != 0)
    for l in zero_lines:
        l.set_color('k')
        l.set_linestyle('-.')

    if len(zero_lines) == 0:
        xlims = ax.get_xlim()
        ax.hlines(0, *xlims, color='k', ls='-.', lw=.5)
        ax.set_xlim(*xlims)

def world_map():
    """ Basic PlateCarree map of the world with land and ocean. """
    fig = plt.figure(dpi=250)
    gs = gridspec.GridSpec(1, 1, hspace=0.5)
    ax = fig.add_subplot(gs[-1], projection=ccrs.PlateCarree())

    ax.add_feature(cfeature.COASTLINE, edgecolor = '#505050', lw = 0.5)
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue', alpha = 0.5)

    ax.set_ylim(-90, 90)
    ax.set_xlim(-180, 180)

    gridliner = ax.gridlines(draw_labels=True, ls='dotted')
    gridliner.right_labels = False
    gridliner.top_labels = False
    return fig, ax

# Figures and axes creation
def monthly_with_cbar(): 
    """ Create a 3x4 grid with a vertical colorbar on the right side. """
    fig = plt.figure()
    gs = fig.add_gridspec(3,4)
    axs = gs.subplots(sharex=True, sharey=True)
    def add_cbar(fig, axs, image): 
        return fig.colorbar(
            image,
            ax=axs,
            orientation='vertical')
    return fig, axs, add_cbar

def outer_tick_labels(axs, xlabel=None, ylabel=None): 
    """ Show tick labels on all outside edges. 
    axs (np.ndarray): 2D array of matplotlib axes """

    for ax in axs[:,0].flat: # Left column: y-label
        if ylabel: ax.set_ylabel(ylabel)

    for ax in axs[-1,:].flat: # Bottom row: x-label
        if xlabel: ax.set_xlabel(xlabel)

    for ax in axs[0,:].flat: # Top row: ticks + nrs.
        ax.tick_params(top=True, labeltop=True)
        
    for ax in axs[:,-1].flat: # Right column: ticks + nrs.
        ax.tick_params(right=True, labelright=True)

    return axs

def three_sideplot_structure() -> tuple[plt.Figure, tuple[plt.Axes]]: 
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

def three_sideplot_labels(fig, axs, zcoord: dcts.Coordinate, eql_coord: dcts.Coordinate
                          ) -> tuple[plt.Figure, tuple[plt.Axes]]: 
    """ Finish doing axis labels, limits and scaling for the 3 sideplot structure thing. """
    (ax_fig, ax_main, ax_upper, ax_right, ax_cube) = axs
    
    fig.suptitle(zcoord.label(filter_label=True))
    fig.subplots_adjust(top = 0.85)
    
    xcoord = dcts.get_coord('geometry.x')
    ycoord = dcts.get_coord('geometry.y') if not eql_coord else eql_coord

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

    axs = (ax_fig, ax_main, ax_upper, ax_right, ax_cube)
    
    return fig, (axs)

def tp_comp_plot(tps, **kwargs):
    """ Default structure for tropopause definition comparisons. """
    fig, axs = plt.subplots(
        math.ceil(len(tps)/2), 2, 
        dpi=kwargs.get('dpi', 500),
        figsize=kwargs.get('figsize', (6, math.ceil(len(tps)/2)*2)),
        sharex=kwargs.get('sharex', True), 
        sharey=kwargs.get('sharey', True),
        )
    if len(tps)%2: axs.flat[-1].axis('off')
    if not kwargs.get('sharey', True): 
        [ax.tick_params(labelright=True, right=True, 
                        labelleft=False, left=False)
         for ax in axs[:,-1].flat]
        [ax.yaxis.set_label_position("right") 
         for ax in axs[:,-1].flat]
    return fig, axs

def make_two_column_axs(tps, extra_axes=0, sharex=True, sharey=True, **fig_kwargs) -> tuple[plt.Figure, tuple[plt.Axes]]: 
    """ Create the necessary nr of subplots and hide superfluous axes. """
    no_of_axs = len(tps) + extra_axes

    figsize = fig_kwargs.pop('figsize', (8, math.ceil(no_of_axs/2)*2))
    dpi = fig_kwargs.pop('dpi', 100)
    fig = plt.figure(figsize=figsize, dpi=dpi, **fig_kwargs)
    
    gs = fig.add_gridspec(math.ceil(no_of_axs/2), 2)
    axs = gs.subplots(sharex=sharex, sharey=sharey)
    gs.tight_layout(fig)

    if no_of_axs%2: 
        axs.flatten()[len(tps)].axis('off')
        axs.flatten()[-1].axis('off')
    
    return fig, axs

def nested_subplots_two_column_axs(tps, nsubcol=2, **fig_kwargs
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
    fig,_ = make_two_column_axs(tps, **fig_kwargs)
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

def adjust_labels_ticks(sub_ax_arr) -> np.ndarray[plt.Axes]: 
    """ Additional helper for nested_subplots_two_column_axs;
    Move axis labels and ticks to the outside of subsubplots and axes. """
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
            
    # --- Hide x- and y-axis labels for inner subplots
    for ax in list(sub_ax_arr[:,0,-1].flat) + list(sub_ax_arr[:,-1,0].flat): 
        # Left column inner right & right column inner left
        ax.yaxis.label.set_visible(False)
    
    for ax in sub_ax_arr[:-1,:,].flat: 
        # All plots not on the bottom row
        ax.xaxis.label.set_visible(False)
        
    # --- Add outer axis descriptions and titles 
    for ax in sub_ax_arr[:,0,0].flat: 
        # Left column inner left
        ax.yaxis.set_label_position('left')

    for ax in sub_ax_arr[:,-1,-1].flat: 
        # Right column inner right
        ax.yaxis.set_label_position('right')
    
    return sub_ax_arr

def hist_lognorm_fitted(x, range, ax, c, hist_kwargs = {}, bin_nr = 50) -> tools.LognormFit: 
    """ Adds a horizontal histogram with a lognorm best fit to the given axis. 
    
    x ((n,) array or sequence of (n,) arrays): 1D array with data to be histogrammed
    range (tuple): histogram range
    ax (Axis): axis to be plotted onto 
    c (str): color of the histogram bars

    """
    print('Hier samma. Puntigamer.')
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

def highlight_axis(ax, color='g', axis='y'):
    """ Highlight an axis that does not conform to the rest of the plot. """
    ax.tick_params(axis=axis, color=color, labelcolor=color)
    if axis=='y':
        ax.spines['right'].set_color(color)
        ax.spines['left'].set_color(color)
        ax.yaxis.label.set_color(color)
    elif axis=='x': 
        ax.spines['top'].set_color(color)
        ax.spines['bottom'].set_color(color)
        ax.xaxis.label.set_color(color)
    return ax

#%% Legends 
def season_legend_handles(av = False, av_std=False, **kwargs) -> list[Line2D]:
    """ Create a legend for the default season-color scale. 
    
    Args: 
        av (bool): Include dashed grey line for average value
        
        key lw (float): linewidth
        key seasons (list[int]): seasons to include, must be in [1,2,3,4]
    """ 
    lw = kwargs.pop('lw', 3)
    seasons = kwargs.pop('seasons', range(1,5))
    lines = [Line2D([0], [0], 
                        label=dcts.dict_season()[f'name_{s}'], 
                        color=dcts.dict_season()[f'color_{s}'], 
                        path_effects = [outline()], 
                        lw = lw, **kwargs
                        )
                for s in seasons]
    
    if av: 
        lines.append(Line2D([0], [0], label='Average',
                            color='dimgrey', ls = 'dashed', lw = 3, 
                            path_effects = [outline()]))
    if av_std: 
        lines.append(Line2D([0], [0], label=r'$\sigma$ (Av.)',
                    color='dimgrey', lw = 6, 
                    alpha=0.3))
    return lines

linestyles = dict(
    int_CARIBIC2_H_rel_TP  = '-',
    int_ERA5_D_HEIGHT_1_5 = '-',
    int_ERA5_D_HEIGHT_2_0 = (0, (5,1)),
    int_ERA5_D_HEIGHT_3_5 = (0, (2.5,0.5)),
    int_ERA5_D_HEIGHT_thermTP = (0, (2.5,0.5,1,0.5)),
)

def tp_legend_handles(tps, filter_label=True, coord_only=False, 
                      no_vc=True, no_model=True, linestyle_dict=None, 
                      **kwargs) -> list[Line2D]: 
    """ Create a legend for all tropopause definitions in self.tps (or tps if given).         
    Args: 
        key tps (list[dcts.Coordinate]): Tropopause definition coordinates
        key lw (float): linewidth
        key ls (str): linestyle 
        key marker (str) 
        key markersize (float)
        key linestyle_dict()
    """ 
    lw = kwargs.pop('lw', 3)   
    lines = [Line2D([0], [0], 
                    label=tp.label(filter_label=filter_label, 
                                    coord_only=coord_only,
                                    no_vc=no_vc,
                                    no_model=no_model), 
                    color=tp.get_color(), 
                    ls = 'solid' if not linestyle_dict else linestyle_dict.get(tp.col_name),
                    lw = lw,
                    **kwargs)
                for tp in tps]
    return lines

def lognorm_legend_handles() -> tuple[list[str], list[Line2D, Patch, Patch]]: 
    """ Create a legend for Mode, 68% Interval and 95% Interval for lognorm stats. """
    # loc = 'lower center', ncols = len(h)
    
    h_Mode = mlines.Line2D([], [], ls = 'None', alpha = 1, zorder = 9, marker = 'o', color = 'k')
    h_68 = Patch(color = 'grey', alpha = 0.8)
    h_95 = Patch(color = 'grey', alpha = 0.5)
    
    l = ['Mode', r'68$\,$% Interval', r'95$\,$% Interval']
    h = [h_Mode, h_68, h_95]

    return h, l

def typical_profile_legend_handles(**kwargs): 
    """ Show season color as circles, binned ver. profiles as line and mean thermal TP as dashed line. """
    seasons = kwargs.pop('seasons', range(1,5))
    actual_lines = []
    lines = [Line2D([0], [0], marker = 'o',
                    markersize=6,
                    label=dcts.dict_season()[f'name_{s}'], 
                    color='w', 
                    markerfacecolor=dcts.dict_season()[f'color_{s}'], 
                    )
            for s in seasons]
    
    actual_lines.append(Line2D([0], [0], label='Binned profile',
                    color='dimgrey', lw = 2))
    
    actual_lines.append(Line2D([0], [0], label='Mean thermal TP',
                    color='dimgrey', lw = 2, ls='dashed'))
    actual_lines = actual_lines + lines
    return actual_lines

def month_legend_handles(months = range(1,13)): 
    """ Subsample colorblind colormap for monthly traces. """
    m_dict = dcts.dict_month()
    lines = [Line2D([0], [0], 
                    linestyle = '-' if m%2 else '--', 
                    color = m_dict[f'color_{m}'],
                    label = m_dict[f'name_{m}'],
                    )
            for m in months]
    return lines

#%% Testing functions

def rand_data(ax): 
    x, y, scale = np.random.randn(3, 100)
    ax.scatter(x=x, y=y, c=scale, s=np.abs(scale)*100, zorder=5)
    return ax