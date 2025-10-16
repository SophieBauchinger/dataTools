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

# TODO: fix get_Bin3D_dict calls

#%% Imports
import geopandas
import math
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines
import matplotlib.patheffects as mpe
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from mpl_toolkits.axes_grid1 import AxesGrid
import numpy as np
import pandas as pd
from PIL import Image
import io
import itertools
from scipy import stats
import warnings

import toolpac.calc.binprocessor as bp

import dataTools.dictionaries as dcts
from dataTools import tools
import dataTools.data.BinnedData as bin_tools
from dataTools.plot import cfig

warnings.filterwarnings(action='ignore', message='Mean of empty slice')

def get_vlimit(subs:dcts.Substance, bin_attr='vmean', vlims=None, df=None) -> tuple: 
    """ Get colormap limits for given substance and bin attribute. """
    if vlims:
        return vlims
    try:
        vlims = subs.vlims(bin_attr=bin_attr)
    except KeyError:
        if bin_attr=='vmean':
            vlims = (np.nanmin(df[subs.col_name]), np.nanmax(df[subs.col_name]))
        else:
            raise KeyError('Could not generate colormap limits.')
    except: 
        raise KeyError('Could not generate colormap limits.')
    return vlims

# TODO
def get_bsize(coord):
    pass

def plot_1d(simple_bin_1d, bin_attr='vmean'): 
    """ scatter plot of binned data. """
    data = getattr(simple_bin_1d, bin_attr)

    fig, ax = plt.subplots(dpi=150, figsize=(6,7))
    ax.plot(simple_bin_1d.xintm, data, label=bin_attr)
    ax.legend()
    plt.show()

def plot_2d(simple_bin_2d, bin_attr='vmean'): 
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

#%% --- Lognorm fitted histograms and things --- # 
def get_lognorm_stats_df(data_dict: dict, lognorm_attr: str, prec:int = 1, use_percentage = False) -> pd.DataFrame: 
    """ Create combined lognorm-fit statistics dataframe for all tps. 
    Relative standard deviations are multiplied by 100 to return percentages instead of fractions. 
    
    Parameters: 
        Bin3D_dict (dict[tools.Bin3DFitted]): Binned data incl. lognorm fits (all variables)
        lognorm_attr (str): vmean_fit / vsdtv_fit / rvstd_fit
    """
    
    if all(isinstance(v, np.ndarray) for v in data_dict.values()): 
        # then we have a lognorm_attr extracted thing already! 
        print('AAHHHHH')
    
    if not 'rvstd' in lognorm_attr and not use_percentage: 
        if isinstance(list(data_dict.values())[0], dict): # seasonal
            
            pass # more complicated ahhhh
            
        return pd.DataFrame({k:getattr(v, lognorm_attr).stats(prec=prec) for k,v in data_dict.items()})

    # else: need to multiple by 100 to get percentage 
    # (and initially return values with increased precision)
    stats_df = pd.DataFrame({k:getattr(v, lognorm_attr).stats(prec=prec+2) for k,v in data_dict.items()})
    
    
    
    df = stats_df.T.convert_dtypes()
    non_float_cols = [c for c in df.columns if c not in df.select_dtypes([int, float]).columns]
    for c in df.select_dtypes(float).columns: 
        df[c] = df[c].apply(lambda x: round(100*x, prec+2))
    for c in non_float_cols:
        df[c] =  df[c].apply(lambda x: tuple([round(100*i, prec+2) for i in x]))
    return df.T

def plot_lognorm_stats(ax, df, s = None, xlims = None): 
    for i, tp_col in enumerate(df.columns): 
        tp = dcts.get_coord(tp_col)
        if s is not None: 
            y = s*8+i
        else: 
            y = tp.label(filter_label = True).split('(')[0]
        # Lines
        line_kw = dict(color = tp.get_color(), lw = 5)
        ax.fill_betweenx([y]*2, *df[tp_col].int_68, **line_kw, alpha = 0.8) # avoids round edges
        ax.fill_betweenx([y]*2, *df[tp_col].int_95, **line_kw, alpha = 0.5)

        # Mode marker
        kw_mode = dict(alpha = 1, zorder = 9, marker = 'o')
        ax.scatter(df[tp_col].Mode, y, **kw_mode, color = 'k')

        # Numeric value of mode
        ax.annotate(
            text = f'{df[tp_col].Mode}  ', 
            xy = (df[tp_col].Mode, y),
            xytext = (0, y),
            ha = 'right', va = 'center', 
            size = 7, fontweight = 'medium',
            )

def seasonal_lognorm_stats(strato_Bin_seas_dict, tropo_Bin_seas_dict, 
                            var, bin_attr = 'vstdv', **kwargs): 
    """ Create plot of lognormal stats for each tropopause in troposphere / stratosphere. 
    Args: 
        axs (tuple[plt.Axes]): Tuple of tropos_axis, stratos_axis
        strato_stats (pd.DataFrame): Stratospheric lognorm fit statistics
        tropo_stats (pd.DataFrame): Tropospheric lognorm fit statistics
        
        key label (str): If 'off', do not show tropopause definition labels
    
    """
    if not 'axs' in kwargs: 
        fig, axs = plt.subplots(1,2, figsize = (8,5), sharey=True, dpi=250)
        axs[0].set_title(f'Troposphere', size = 10, pad = 3)
        axs[1].set_title(f'Stratosphere', size = 10, pad = 3)
    else: 
        axs = kwargs.get('axs')

    seasons = set(next(iter(strato_Bin_seas_dict.values())))
    xlims = dcts.vlim_dict_per_substance(var.short_name)[bin_attr]
    
    for s in seasons:
        strato_BinDict = {k:v[s] for k,v in strato_Bin_seas_dict.items()}
        tropo_BinDict = {k:v[s] for k,v in tropo_Bin_seas_dict.items()}

        strato_stats = get_lognorm_stats_df(strato_BinDict, 
                                                f'{bin_attr}_fit', 
                                                prec = kwargs.get('prec', 1), 
                                                use_percentage = var.detr)
        tropo_stats = get_lognorm_stats_df(tropo_BinDict, 
                                                f'{bin_attr}_fit', 
                                                prec = kwargs.get('prec', 1),
                                                use_percentage = var.detr)

        for df, ax in zip([tropo_stats, strato_stats], axs):
            plot_lognorm_stats(ax, df, s, xlims)
    
    y_arr =  [s*8+2.5 for s in seasons]# [i*8+s for i in range(len(df.columns)) for s in seasons]
    y_ticks = [dcts.dict_season()[f'name_{s}'] for s in seasons]# [dcts.get_coord(tp_col).label(filter_label = True).split('(')[0] for tp_col in df.columns]

    axs[0].set_yticks(y_arr, y_ticks)
    axs[0].tick_params(labelleft=True if not kwargs.get('label')=='off' else False, 
                        left=False)
    axs[1].tick_params(left=False, labelleft=False)
    
    for ax in axs:
        ax.set_xlim(ax.get_xlim()[0]-ax.get_xlim()[1]/8, ax.get_xlim()[1])
        ax.set_xlabel(var.label(bin_attr=bin_attr))
        ax.grid(axis='x', ls = 'dashed')

    axs[0].legend(handles = cfig.tp_legend_handles(filter_label=True, no_vc = True)[::-1], 
            prop=dict(size = 6));

    if 'fig' in locals(): 
        fig.subplots_adjust(bottom = 0.15, wspace = 0.05)
        fig.legend(*cfig.lognorm_legend_handles(), loc = 'lower center', ncols = 3)

    return axs

def plot_histogram_comparison(self, var, strato_dict, tropo_dict, bin_attr='vstdv', 
                                xscale = 'linear', show_stats = False, fig_kwargs = {}, **kwargs):
    """ Plot histogram with lognorm fit comparison between tropopauses. 
    
    Args: 
        tropo_BinDict, strato_BinDict (dict[tp:Bin*DFitted]): 
            Fitted binned data in x dimensions
        
        key season (int): If passed, add suptitle with current season
    """
    tps = [self.get_coords(col_name = k)[0] for k in tropo_dict.keys()]

    fig, main_axes, sub_ax_arr = cfig.nested_subplots_two_column_axs(tps, **fig_kwargs)
    cfig._adjust_labels_ticks(sub_ax_arr)

    if kwargs.get('season'): 
        fig.suptitle(dcts.dict_season()['name_{}'.format(kwargs.get('season'))])
        fig.subplots_adjust(top = 0.9)
    
    gs = main_axes.flat[0].get_gridspec()
    gs.update(wspace = 0.3)

    tropo_axs = sub_ax_arr[:,:,0].flat
    strato_axs = sub_ax_arr[:,:,-1].flat

    # Set axis titles and labels """
    pad = 12 if show_stats else 5
    
    for ax in sub_ax_arr[0,:,0].flat: # Top row inner left
        ax.set_title('Troposphere', style = 'oblique', pad = pad)
    for ax in sub_ax_arr[0,:,-1].flat: # Top row inner right
        ax.set_title('Stratosphere', style = 'oblique', pad = pad)

    for ax in sub_ax_arr.flat: 
        # All subplots
        ax.set_xlabel('Frequency [#]')
        ax.set_ylabel(var.label(bin_attr=bin_attr), fontsize = 8)
        ax.grid(ls ='dotted', lw = 1, color='grey', zorder=0)
        ax.set_xscale(xscale)
        
    # Add histograms and lognorm fits
    for axes, data_Bin_dict in zip([tropo_axs, strato_axs], [tropo_dict, strato_dict]): 
        
        # Extract bin_attr
        data_dict = bin_tools.extract_attr(data_Bin_dict, bin_attr)
            
        # Get overall tropo / strato bin limits
        hist_min, hist_max = np.nan, np.nan
        for data in data_dict.values(): 
            hist_min = np.nanmin([hist_min] + list(data.flatten()))
            hist_max = np.nanmax([hist_max] + list(data.flatten()))

        for ax, tp_col in zip(axes, data_dict): 
            data_flat = data_dict[tp_col].flatten() # fails if no bin_attr extracted

            # Adding the histograms to the figure
            # lognorm_inst = bin_fitted.plot_hist(ax,
            #     hist_range = (hist_min, hist_max),
            #     color = dcts.get_coord(tp_col).get_color(),
            #     bin_attr = bin_attr)

            lognorm_inst = cfig.hist_lognorm_fitted(data_flat, (hist_min, hist_max), ax, 
                                                        dcts.get_coord(tp_col).get_color(),
                                                        hist_kwargs = dict(range = (hist_min, hist_max)))
            
            # Show values of mode and sigma at the top of each subplot
            if show_stats:
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
    for ax, tp_col in zip(sub_ax_arr[:,:,0].flat, tropo_dict):
        ax.invert_xaxis()
        tp_title = dcts.get_coord(tp_col).label(filter_label=True).split("(")[0] # shorthand of tp label
        ax.text(**dcts.note_dict(ax, s = tp_title, x = 0.1, y = 0.85))
    return fig, main_axes, sub_ax_arr

#%% 1D
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

def flight_height_histogram(self, tp, alpha: float = 0.7, **hist_kwargs): 
    """ Make a histogram of the number of datapoints for each tp bin. """
    _, ax = plt.subplots(dpi=150, figsize=(6,4))
    ax.set_ylabel(tp.label())
    data = self.df[tp.col_name].dropna()
    ax.set_title(f'Distribution of {self.source} measurements')
    rlims = (-70, 70) if (tp.vcoord=='pt' and tp.rel_to_tp) else (data.min(), data.max())
    hist = ax.hist(data.values, 
                    bins=30, range=rlims, alpha=alpha, 
                    orientation='horizontal',
                    edgecolor = hist_kwargs.pop('edgecolor', 'white'), 
                    lw=hist_kwargs.pop('lw', 0.2),
                    **hist_kwargs)
    ax.grid(ls='dotted', zorder = 0)
    if (tp.rel_to_tp is True) and ax is not None: 
        ax.hlines(0, max(hist[0]), 0, color='k', ls='dashed')
    ax.set_xlabel('# Datapoints')
    
    if tp.crit == 'n2o': 
        ax.invert_yaxis()
        ax.hlines(320.459, max(hist[0]), 0, color='k', ls='dashed', lw=0.5)
    return hist

def plot_vertial_profile_variability_comparison(subs, tps: list, 
                                                rel: bool = True, 
                                                bin_attr: str = 'vstdv', 
                                                seasons: list[int] = [1,3],
                                                **kwargs): 
    """ Compare relative mixing ratio varibility between tropopause definitions. """
    
    
    
    fig, ax = plt.subplots(dpi=500, figsize=(5,6))
    outline = mpe.withStroke(linewidth=2, foreground='white')
    
    for i, tp in enumerate(tps):
        bin_dict = bin_tools.seasonal_binning(subs, tp)
        
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

def plot_1d_seasonal_gradient(df, subs, coord, 
                              bin_attr: str = 'vmean', 
                              add_stdv: bool = False, 
                              **kwargs):
    """ Plot gradient per season onto one plot. """
    big = kwargs.pop('big') if 'big' in kwargs else False
    
    bin_dict = bin_tools.seasonal_binning(df, subs, coord, **kwargs)

    fig, ax = plt.subplots(dpi=500, figsize= (6,4) if not big else (3,4))

    if coord.vcoord=='pt' and coord.rel_to_tp: 
        ax.set_yticks(np.arange(-60, 75, 20) + [0])

    for s in bin_dict.keys():
        plot_1d_gradient(ax, s, bin_dict[s], bin_attr, add_stdv)
        
    ax.set_title(coord.label(filter_label=True))
    ax.set_ylabel(coord.label(coord_only = True))

    if bin_attr=='vmean':
        ax.set_xlabel(subs.label())
    elif bin_attr=='vstdv': 
        ax.set_xlabel('Relative variability of '+subs.label(name_only=True))

    if (coord.vcoord in ['mxr', 'p'] and not coord.rel_to_tp) or coord.col_name == 'N2O_residual': 
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
        ax.legend(loc=kwargs.get('legend_loc', 'lower left'))
    ax.grid('both', ls='dashed', lw=0.5)
    ax.set_axisbelow(True)
    
    if coord.rel_to_tp: 
        tools.add_zero_line(ax)

    return bin_dict, fig 

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

#%% 2D
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

def calc_average(bin2d_inst, bin_attr='vstdv'):
    """ Calculate weighted overall average. """
    data = getattr(bin2d_inst, bin_attr)
    data = data[~np.isnan(data)]

    weights = bin2d_inst.vcount
    weights = bin2d_inst.vcount[[i!=0 for i in weights]]

    try: weighted_average = np.average(data, weights = weights)
    except ZeroDivisionError: weighted_average = np.nan
    return weighted_average

def calc_ts_averages(bin2d_inst, bin_attr = 'vstdv'):
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
    bin_equi2d = bin_tools.make_bci(xcoord, ycoord, 
                                    xbmin = -180, xbmax = 180,
                                    ybmin = -90, ybmax = 90,
                                    **kwargs)
    vlims = kwargs.get('vlims')
    if vlims is None: vlims = get_vlimit(subs, bin_attr)
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

def seasonal_2d_plots(df, subs, xcoord, ycoord, bin_attr, **kwargs):
    """
    Parameters:
        bin_attr (str): 'vmean', 'vstdv', 'vcount'
        key v/x/ylims (tuple[float])
    """
    
    try: 
        cmap = dcts.dict_colors()[bin_attr]
    except: 
        cmap = plt.cm.viridis

    binned_seasonal = bin_tools.seasonal_binning(df, subs, xcoord, ycoord, **kwargs)

    if not any(bin_attr in bin2d_inst.__dict__ for bin2d_inst in binned_seasonal.values()):
        raise KeyError(f'\'{bin_attr}\' is not a valid attribute of Bin2D objects.')

    vlims = kwargs.get('vlims', get_vlimit(subs, bin_attr))
    xlims = kwargs.pop('xlims', xcoord.get_lims())
    ylims = kwargs.pop('ylims', ycoord.get_lims())

    norm = Normalize(*vlims)
    fig, axs = plt.subplots(2, 2, dpi=100, figsize=(8,9),
                            sharey=True, sharex=True)

    fig.subplots_adjust(top = 1.1)
    
    for season, ax in zip(binned_seasonal.keys(), axs.flatten()):
        bin2d_inst = binned_seasonal[season]
        ax.set_title(dcts.dict_season()[f'name_{season}'])
        
        img = single_2d_plot(ax, bin2d_inst, bin_attr, xcoord, ycoord, 
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

def plot_total_2d(df, subs, xcoord, ycoord, bin_attr='vstdv', **kwargs):
    """ Single 2D plot of varibility of given substance. """

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

    xbsize = xcoord.get_bsize()
    ybsize = xcoord.get_bsize()

    # get bins as multiples of the bin size
    xbmax = ((np.nanmax(x) // xbsize) + 1) * xbsize
    xbmin = (np.nanmin(x) // xbsize) * xbsize

    ybmax = ((np.nanmax(y) // ybsize) + 1) * ybsize
    ybmin = (np.nanmin(y) // ybsize) * ybsize

    bin_equi2d = bp.Bin_equi2d(xbmin, xbmax, xbsize,
                                ybmin, ybmax, ybsize)

    bin2d_inst = bp.Simple_bin_2d(np.array(df[subs.col_name]), 
                                  x, y, bin_equi2d)
    
    vlims = kwargs.get('vlims', get_vlimit(subs, bin_attr))
    xlims = bin_tools.get_var_lims(xcoord, xbsize, df)
    ylims = bin_tools.get_var_lims(ycoord, ybsize, df)
    
    norm = Normalize(*vlims)
    fig, ax = plt.subplots(dpi=250, figsize=(8,9))
    fig.subplots_adjust(top = 1.1)

    data_title = 'Mixing ratio' if bin_attr=='vmean' else 'Varibility'
    # fig.suptitle(f'{data_title} of {subs.label()}', y=0.95)

    cmap = dcts.dict_colors()[bin_attr]

    img = single_2d_plot(ax, bin2d_inst, bin_attr, xcoord, ycoord, 
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

    xbsize = xcoord1.get_bsize(**kwargs)
    ybsize = ycoord1.get_bsize(**kwargs)
    
    bin_equi2d = bp.Bin_equi2d(np.nanmin(self.df[xcoord1.col_name]),
                                np.nanmax(self.df[xcoord1.col_name]),
                                xbsize,
                                np.nanmin(self.df[ycoord1.col_name]),
                                np.nanmax(self.df[ycoord1.col_name]),
                                ybsize)

    binned_seasonal_1 = bin_tools.seasonal_binning(self.df, *params_1, bci=bin_equi2d)
    binned_seasonal_2 = bin_tools.seasonal_binning(self.df, *params_2, bci=bin_equi2d)

    # vlims = get_vlimit(subs1, 'vmean')
    xlims = bin_tools.get_var_lims(xcoord1)
    ylims = bin_tools.get_var_lims(ycoord1)

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

def violin_boxplot_with_stats(self, var, xcoord, ycoord, 
                                bin_attr='vstdv', atm_layer = 'tropo',
                                **kwargs):
    """ Make two figures with violin etc plots - one tropo, one strato.
    Data is 2D binned (could theoretically extend to 3D binning)
    
    Parameters: 
        var (dcts.Substance|dcts.Coordinate)
        x/ycoord (dcts.Coordinate)
        bin_attr (str)
        
        key x/ybsize (float). Size of x/y bins
        key jitter (float). Extent of scatter jitter. Default 0.05
    """
    # Get data
    tps = kwargs.pop('tps', self.tps)

    COLOR_SCALE = [tp.get_color() for tp in tps]
    LABELS = [tp.label(filter_label = True) for tp in tps]
    POSITIONS = np.linspace(0, len(tps)-1, len(tps))
    jitter = kwargs.pop('jitter', 0.05)
    ylabel = var.label(bin_attr = bin_attr)
    
    # Boxplot parameters 
    medianprops = dict(
        linewidth=1.5, 
        color="xkcd:dark grey",
        solid_capstyle="butt")
    boxprops = dict(
        linewidth=1, 
        color="xkcd:dark")
    
    data_set = []
    for tp in tps:
        bin2d = self.sel_atm_layer(atm_layer, tp).bin_2d(
            var, xcoord, ycoord, **kwargs)
        data = getattr(bin2d, bin_attr).flatten()
        data = data[~np.isnan(data)]
        data_set.append(data)

    means = [y.mean() for y in data_set]
    stds = [np.std(y) for y in data_set]
    skews = [stats.skew(y) for y in data_set]
    
    # Create plot
    fig, ax = plt.subplots(figsize= (8, 5))
    ax.set_title({'tropo': 'Troposphere',
                    'strato' : 'Stratosphere'}.get(atm_layer))

    # Add violins 
    violins = ax.violinplot(
        data_set, 
        positions=POSITIONS,
        widths=0.6,
        bw_method="silverman",
        showmeans=False, 
        showmedians=False,
        showextrema=False)

    for pc in violins["bodies"]:
        pc.set_facecolor("none")
        pc.set_edgecolor("k")
        pc.set_linewidth(1.4)
        pc.set_alpha(1)

    # Add boxplot
    ax.boxplot(
        data_set,
        widths = 0.25,
        positions=POSITIONS, 
        showfliers = False, # Do not show the outliers beyond the caps.
        showcaps = False,   # Do not show the caps
        medianprops = medianprops,
        whiskerprops = boxprops,
        boxprops = boxprops)

    # Add data points
    x_data = [np.array([i] * len(d)) for i, d in enumerate(data_set)]
    x_jittered = [x + stats.t(df=6, scale=jitter).rvs(len(x)) for x in x_data]

    for x, y, color, label in zip(x_jittered, data_set, COLOR_SCALE, LABELS):
        ax.scatter(x, y, s = 80, color=color, alpha=0.3, label = label)

    # Add dot representing the mean
    for i, mean in enumerate(means):
        ax.scatter(i, mean, s=50, color="#850e00", edgecolor ='k', zorder=3)
    ax.legend(handles = self.tp_legend_handles(),
            ncols = 1 if atm_layer=='tropo' else 3,
            fontsize = 9,
            loc = 'upper left' if atm_layer=='tropo'
            else 'upper center')
    
    y_stats = [f'$\mu$ = {m:.2f}\n $\sigma$ = {s:.2f}\n $\gamma$ = {y:.2f}\n' \
        for m,s,y in zip(means, stds, skews)]

    for x,y,s in zip(POSITIONS, 
                    [max(y) for y in data_set], 
                    y_stats):
        ax.text(x = x, y = y, s = s, 
                ha = 'center', va = 'bottom')
    ax.set_ylim(0, ax.get_ylim()[1]+(20 if atm_layer=='tropo'
                                    else 150))
    ax.tick_params(labelbottom=False, bottom=False)
    ax.set_ylabel(ylabel)

    fig.tight_layout()
    plt.show()

# --- Lognorm Fits --- #
def seasonal_2d_lognorm_stats(self, var, bin_attr='vstdv', **kwargs): 
    """ docstring """
    [s_zcoord] = self.get_coords(vcoord = 'pt', model = 'ERA5', tp_def = 'nan')
    [t_zcoord] = self.get_coords(vcoord = 'z', model = 'ERA5', tp_def = 'nan', var='nan')

    [s_hcoord] = self.get_coords(hcoord = 'eql', model = 'ERA5')
    [t_hcoord] = self.get_coords(col_name = 'geometry.y') # latitude

    strato_Bin2Dseas_dict, tropo_Bin2Dseas_dict = {}, {}

    for tp in self.tps:
        strato_df = self.sel_strato(tp).df
        strato_Bin2Dseas_dict[tp.col_name] = bin_tools.seasonal_binning(strato_df, var, s_hcoord, s_zcoord)
        tropo_df = self.sel_tropo(tp).df
        tropo_Bin2Dseas_dict[tp.col_name] = bin_tools.seasonal_binning(tropo_df, var, t_hcoord, t_zcoord)
        
    fig, axs = plt.subplots(1,2, figsize = (8,5), sharey=True, dpi=250)
    axs[0].set_title(f'Troposphere ({t_hcoord.label(coord_only=True)}'+r' $\times$ '\
        +f'{t_zcoord.label(coord_only=True)})',
                        size = 10, pad = 3)
    axs[1].set_title(f'Stratosphere ({s_hcoord.label(coord_only=True)}'+r' $\times$ '\
        +f'{s_zcoord.label(coord_only=True)})',
                        size = 10, pad = 3)

    self.seasonal_lognorm_stats(strato_Bin2Dseas_dict, tropo_Bin2Dseas_dict, var, bin_attr, 
                                axs=axs, fig=fig, **kwargs)

def histogram_2d_comparison(self, var, bin_attr='vstdv', **kwargs): 
    """ Comparison plot for tropopause definition substance histograms + lognorm fit. """
    [s_hcoord] = self.get_coords(hcoord = 'eql', model = 'ERA5')
    [t_hcoord] = self.get_coords(col_name = 'geometry.y') # latitude

    [s_zcoord] = self.get_coords(vcoord = 'pt', model = 'ERA5', tp_def = 'nan')
    [t_zcoord] = self.get_coords(vcoord = 'z', model = 'ERA5', tp_def = 'nan', var='nan')

    strato_BinDict, tropo_BinDict = {}, {}

    for tp in kwargs.get('tps', self.tps):
        strato_BinDict[tp.col_name] = self.sel_strato(tp).bin_2d(var, s_hcoord, s_zcoord, **kwargs)
        tropo_BinDict[tp.col_name] = self.sel_tropo(tp).bin_2d(var, t_hcoord, t_zcoord, **kwargs)
    
    # Create the figure 
    self.plot_histogram_comparison(var, 
                            tropo_BinDict, strato_BinDict,
                            bin_attr=bin_attr, 
                            **kwargs)

    print('Troposphere: '+f'{t_hcoord.label(coord_only=True)}'+r' $\times$ '\
            +f'{t_zcoord.label(coord_only=True)}')

    print('Stratosphere: '+f'{s_hcoord.label(coord_only=True)}'+r' $\times$ '\
            +f'{s_zcoord.label(coord_only=True)}')

def seasonal_2d_histograms(self, var, bin_attr='vstdv', **kwargs): 
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

    [s_zcoord] = self.get_coords(vcoord = 'pt', model = 'ERA5', tp_def = 'nan')
    [t_zcoord] = self.get_coords(vcoord = 'z', model = 'ERA5', tp_def = 'nan', var='nan')

    [s_hcoord] = self.get_coords(hcoord = 'eql', model = 'ERA5')
    [t_hcoord] = self.get_coords(col_name = 'geometry.y') # latitude

    strato_Bin2Dseas_dict, tropo_Bin2Dseas_dict = {}, {}

    for tp in kwargs.get('tps', self.tps):
        strato_df = self.sel_strato(tp).df
        strato_Bin2Dseas_dict[tp.col_name] = bin_tools.seasonal_binning(strato_df, var, s_hcoord, s_zcoord)
        tropo_df = self.sel_tropo(tp).df
        tropo_Bin2Dseas_dict[tp.col_name] = bin_tools.seasonal_binning(tropo_df, var, t_hcoord, t_zcoord)
    
    # Create the figure 
    for s in set(self.df['season']):
        strato_BinDict = {k:v[s] for k,v in strato_Bin2Dseas_dict.items()}
        tropo_BinDict = {k:v[s] for k,v in tropo_Bin2Dseas_dict.items()}
        
        self.plot_histogram_comparison(var, 
                                tropo_BinDict, strato_BinDict,
                                bin_attr=bin_attr, 
                                **kwargs)

#%% 3D
""" Three-dimensional binning & plotting. 

Methods: 
    z_crossection(subs, tp, bin_attr, save_gif_path)
    stratosphere_map(subs, tp, bin_attr)
    matrix_plot_3d_stdev_subs(substance, note, tps, save_fig)
    matrix_plot_stdev(note, atm_layer, savefig)
"""

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

def histogram_3d_comparison(self, var, bin_attr='vstdv', **kwargs):
    """ Create lognorm fitted histogram comparison for 3D binned data. """
    [xcoord] = self.get_coords(col_name = 'geometry.x') # longitude
    
    [s_ycoord] = self.get_coords(hcoord = 'eql', model = 'ERA5')
    [t_ycoord] = self.get_coords(col_name = 'geometry.y') # latitude
    
    [s_zcoord] = self.get_coords(vcoord = 'pt', model = 'ERA5', tp_def = 'nan')
    [t_zcoord] = self.get_coords(vcoord = 'z', model = 'ERA5', tp_def = 'nan', var='nan')

    strato_BinDict, tropo_BinDict = {}, {}
    for tp in kwargs.get('tps', self.tps):
        strato_BinDict[tp.col_name] = self.sel_strato(tp).bin_3d(
            var, xcoord, s_ycoord, s_zcoord, **kwargs)
        tropo_BinDict[tp.col_name] = self.sel_tropo(tp).bin_3d(
            var, xcoord, t_ycoord, t_zcoord, **kwargs)
    
    # Create the figure 
    self.plot_histogram_comparison(var, tropo_BinDict, strato_BinDict,
                                    bin_attr=bin_attr, **kwargs)

def lognorm_stats_3d_seasonal(self, var, bin_attr='vstdv', **kwargs): 
    """ Lognorm-fit stats for histograms of 3D-binned variable data. """
    [xcoord] = self.get_coords(col_name = 'geometry.x') # longitude
    [s_ycoord] = self.get_coords(hcoord = 'eql', model = 'ERA5')
    [t_ycoord] = self.get_coords(col_name = 'geometry.y') # latitude
    [s_zcoord] = self.get_coords(vcoord = 'pt', model = 'ERA5', tp_def = 'nan')
    [t_zcoord] = self.get_coords(vcoord = 'z', model = 'ERA5', tp_def = 'nan', var='nan')

    strato_Bin3Dseas_dict, tropo_Bin3Dseas_dict = {}, {}
    for tp in kwargs.get('tps', self.tps):
        strato_df = self.sel_strato(tp).df
        strato_Bin3Dseas_dict[tp.col_name] = bin_tools.seasonal_binning(
            strato_df, var, xcoord, s_ycoord, s_zcoord, **kwargs)
        tropo_df = self.sel_tropo(tp).df
        tropo_Bin3Dseas_dict[tp.col_name] = bin_tools.seasonal_binning(
            tropo_df, var, xcoord, t_ycoord, t_zcoord, **kwargs)

    fig, axs = plt.subplots(1,2, figsize = (8,5), sharey=True, dpi=250)
    axs[0].set_title('Troposphere (Lat' + r' $\times$ ' + 'Lon' + r' $\times$ ' \
        +f'{t_zcoord.label(coord_only=True)})',
                        size = 10, pad = 3)
    axs[1].set_title('Stratosphere (Eq.Lat' + r' $\times$ ' + 'Lon' + r' $\times$ ' \
        +f'{s_zcoord.label(coord_only=True)})',
                        size = 10, pad = 3)

    self.seasonal_lognorm_stats(strato_Bin3Dseas_dict, tropo_Bin3Dseas_dict, var, bin_attr, 
                                axs=axs, fig=fig, **kwargs)

def lognorm_stats_3d_all(self, var, bin_attr='vstdv', **kwargs): 
    """ Lognorm-fit stats for histograms of 3D-binned variable data. """
    [xcoord] = self.get_coords(col_name = 'geometry.x') # longitude
    [s_ycoord] = self.get_coords(hcoord = 'eql', model = 'ERA5')
    [t_ycoord] = self.get_coords(col_name = 'geometry.y') # latitude
    [s_zcoord] = self.get_coords(vcoord = 'pt', model = 'ERA5', tp_def = 'nan')
    [t_zcoord] = self.get_coords(vcoord = 'z', model = 'ERA5', tp_def = 'nan', var='nan')

    strato_Bin3D_dict, tropo_Bin3D_dict = {}, {}
    for tp in kwargs.get('tps', self.tps):
        strato_Bin3D_dict[tp.col_name] = self.sel_strato(tp).bin_3d(
            var, xcoord, s_ycoord, s_zcoord, **kwargs)
        tropo_Bin3D_dict[tp.col_name] = self.sel_tropo(tp).bin_3d(
            var, xcoord, t_ycoord, t_zcoord, **kwargs)

    fig, axs = plt.subplots(1,2, figsize = (8,3), sharey=True, dpi=250)
    axs[0].set_title('Troposphere (Lat' + r' $\times$ ' + 'Lon' + r' $\times$ ' \
        +f'{t_zcoord.label(coord_only=True)})',
                        size = 10, pad = 3)
    axs[1].set_title('Stratosphere (Eq.Lat' + r' $\times$ ' + 'Lon' + r' $\times$ ' \
        +f'{s_zcoord.label(coord_only=True)})',
                        size = 10, pad = 3)

    xlims = dcts.vlim_dict_per_substance(var.short_name)[bin_attr]

    strato_stats = get_lognorm_stats_df(strato_Bin3D_dict, 
                                            f'{bin_attr}_fit', 
                                            prec = kwargs.get('prec', 1), 
                                            use_percentage = var.detr)
    tropo_stats = get_lognorm_stats_df(tropo_Bin3D_dict, 
                                            f'{bin_attr}_fit', 
                                            prec = kwargs.get('prec', 1),
                                            use_percentage = var.detr)

    for df, ax in zip([tropo_stats, strato_stats], axs):
        self.plot_lognorm_stats(ax, df, None, xlims)

    # y_arr =  [s*8+2.5 for s in seasons]# [i*8+s for i in range(len(df.columns)) for s in seasons]
    # y_ticks = [dcts.dict_season()[f'name_{s}'] for s in seasons]# [dcts.get_coord(tp_col).label(filter_label = True).split('(')[0] for tp_col in df.columns]
    # axs[0].set_yticks(y_arr, y_ticks)
    axs[0].set_ylim(-0.5, len(df.columns)- 0.25)
    axs[0].tick_params(labelleft=True if not kwargs.get('label')=='off' else False, 
                        left=False)
    axs[1].tick_params(left=False, labelleft=False)
    
    for ax in axs:
        ax.set_xlim(ax.get_xlim()[0]-ax.get_xlim()[1]/5, ax.get_xlim()[1])
        ax.set_xlabel(var.label(bin_attr=bin_attr))
        ax.grid(axis='x', ls = 'dashed')

    # axs[0].legend(handles = self.tp_legend_handles(filter_label=True, no_vc = True)[::-1], 
    #         prop=dict(size = 6));

    fig.subplots_adjust(bottom = 0.3, wspace = 0.05)
    fig.legend(*cfig.lognorm_legend_handles(), loc = 'lower center', ncols = 3)


def improved_fancy_histogram_plots_nested(self, subs, bin_attr='vstdv', 
                                    xscale = 'linear', show_stats = True, fig_kwargs = {}): 
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

    [s_zcoord] = self.get_coords(vcoord = 'pt', model = 'ERA5', tp_def = 'nan')
    [t_zcoord] = self.get_coords(vcoord = 'z', model = 'ERA5', tp_def = 'nan', var='nan')

    # Get the 3D binned data
    strato_3d_dict, _ = self.get_data_3d_dicts(subs, s_zcoord, eql = True, bin_attr = bin_attr)
    _, tropo_3d_dict = self.get_data_3d_dicts(subs, t_zcoord, eql = False, bin_attr = bin_attr)
    
    # Create the figure 
    fig, main_axes, sub_ax_arr = self._nested_subplots_two_column_axs(self.tps, **fig_kwargs)
    self._adjust_labels_ticks(sub_ax_arr)
    
    gs = main_axes.flat[0].get_gridspec()
    gs.update(wspace = 0.3)

    tropo_axs = sub_ax_arr[:,:,0].flat
    strato_axs = sub_ax_arr[:,:,-1].flat

    # Set axis titles and labels """
    pad = 12 if show_stats else 5
    
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

        for ax, tp_col in zip(axes,
                                data_3d_dict): 
            data3d = data_3d_dict[tp_col]
            data_flat = data3d.flatten()
            
            # Adding the histograms to the figure
            lognorm_inst = self._hist_lognorm_fitted(
                data_flat, (bin_lim_min, bin_lim_max), ax, dcts.get_coord(tp_col).get_color(),
                hist_kwargs = dict(range = (bin_lim_min, bin_lim_max)))
            
            # Show values of mode and sigma
            if show_stats:
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

def fancy_histogram_plots_nested(self, subs, zcoord, eql=False, bin_attr='vstdv', 
                                    xscale = 'linear', show_stats = True, fig_kwargs = {}): 
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
    strato_3d_dict, tropo_3d_dict = self.get_data_3d_dicts(subs, zcoord, eql, bin_attr)
    
    # Create the figure 
    fig, main_axes, sub_ax_arr = self._nested_subplots_two_column_axs(self.tps, **fig_kwargs)
    self._adjust_labels_ticks(sub_ax_arr)
    
    gs = main_axes.flat[0].get_gridspec()
    gs.update(wspace = 0.3)

    tropo_axs = sub_ax_arr[:,:,0].flat
    strato_axs = sub_ax_arr[:,:,-1].flat

    # Set axis titles and labels """
    pad = 12 if show_stats else 5
    
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

        for ax, tp_col in zip(axes,
                                data_3d_dict): 
            data3d = data_3d_dict[tp_col]
            data_flat = data3d.flatten()
            
            # Adding the histograms to the figure
            lognorm_inst = self._hist_lognorm_fitted(
                data_flat, (bin_lim_min, bin_lim_max), ax, dcts.get_coord(tp_col).get_color(),
                hist_kwargs = dict(range = (bin_lim_min, bin_lim_max)))
            
            # Show values of mode and sigma
            if show_stats:
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

def improved_coords_lognorm_fit_comparison(self, subs, bin_attr = 'vstdv', **kwargs): 
    """ 
    Compare characteristic quantities for lognorm fits of strato/tropo data between tps.
    BUT: Using theta / eqlat for the stratosphere but alt / lat for the troposphere. 
    """
    # TODO: fix get_Bin3D_dict calls
    [s_zcoord] = self.get_coords(vcoord = 'pt', model = 'ERA5', tp_def = 'nan')
    strato_BinDict,_ = self.get_Bin3D_dict(subs, s_zcoord, eql = True, **kwargs)

    [t_zcoord] = self.get_coords(vcoord = 'z', model = 'ERA5', tp_def = 'nan', var='nan')
    _, tropo_BinDict = self.get_Bin3D_dict(subs, t_zcoord, eql = False, **kwargs)

    strato_stats = get_lognorm_stats_df(strato_BinDict, 
                                                f'{bin_attr}_fit', 
                                                prec = kwargs.get('prec', 1), 
                                                use_percentage = subs.detr)
    tropo_stats = get_lognorm_stats_df(tropo_BinDict, 
                                            f'{bin_attr}_fit', 
                                            prec = kwargs.get('prec', 1),
                                            use_percentage = subs.detr)
    
    fig, axs = plt.subplots(1,2, figsize = (6,3.5), sharey = True, dpi = 250)

    axs[0].set_title('Troposphere',  size = 10, pad = 3)
    axs[1].set_title('Stratosphere', size = 10, pad = 3)

    marker_kw = dict(color = 'k')

    for df, ax in zip([tropo_stats, strato_stats], axs):
        ax.grid(axis='x', ls = 'dashed')
        for i, tp_col in enumerate(df.columns): 
            tp = dcts.get_coord(tp_col)
            y = tp.label(filter_label = True).split('(')[0]

            # Lines
            line_kw = dict(color = tp.get_color(), lw = 10)
            ax.fill_betweenx([y]*2, *df[tp_col].int_68, **line_kw, alpha = 0.8)
            ax.fill_betweenx([y]*2, *df[tp_col].int_95, **line_kw, alpha = 0.5)

            # Mode marker
            kw_mode = dict(alpha = 1, zorder = 9, marker = 'o')
            ax.scatter(df[tp_col].Mode, y, **kw_mode, **marker_kw)
        
            # Numeric value of mode
            ax.annotate(
                text = f'{df[tp_col].Mode}', size = 9, 
                xy = (df[tp_col].Mode, y),
                xytext = (df[tp_col].Mode, i+0.35),
                ha = 'center', va = 'center', 
                fontweight = 'medium',
                )

    # axs[0].invert_yaxis()
    axs[0].set_ylim(-0.5, len(df.columns)- 0.25)
    axs[0].tick_params(labelleft=True, left=False)
    axs[1].tick_params(left=False)

    h_Mode = mlines.Line2D([], [], ls = 'None', **kw_mode, **marker_kw)
    h_68 = Patch(color = 'grey', alpha = 0.8)
    h_95 = Patch(color = 'grey', alpha = 0.5)

    l = ['Mode', '68$\,$% Interval', '95$\,$% Interval']
    h = [h_Mode, h_68, h_95]

    fig.tight_layout()
    fig.subplots_adjust(bottom = 0.2, top = 0.85)
    if not subs.detr: 
        fig.suptitle('Distribution of ' + subs.label(bin_attr=bin_attr))
    else:
        fig.suptitle('Distribution of ' + subs.label(bin_attr=bin_attr).split('[')[0] + '[%]')
        
    fig.legend(h, l, loc = 'lower center', ncols = len(h))

def make_lognorm_fit_comparison(self, subs, zcoord, bin_attr = 'vstdv', eql=False, **kwargs): 
    """ Compare characteristic quantities for lognorm fits of strato/tropo data between tps. """
    # TODO: fix get_Bin3D_dict calls
    strato_BinDict, tropo_BinDict = self.get_Bin3D_dict(subs, zcoord, eql, **kwargs)
    
    strato_stats = get_lognorm_stats_df(strato_BinDict, 
                                                f'{bin_attr}_fit', 
                                                prec = kwargs.get('prec', 1))
    tropo_stats = get_lognorm_stats_df(tropo_BinDict, 
                                            f'{bin_attr}_fit', 
                                            prec = kwargs.get('prec', 1))
    
    fig, axs = plt.subplots(1,2, figsize = (6,3.5), sharey = True)

    axs[0].set_title('Troposphere',  size = 10, pad = 3)
    axs[1].set_title('Stratosphere', size = 10, pad = 3)

    marker_kw = dict(color = 'k')

    for df, ax in zip([tropo_stats, strato_stats], axs):
        ax.grid(axis='x', ls = 'dashed')
        for i, tp_col in enumerate(df.columns): 
            tp = dcts.get_coord(tp_col)
            y = tp.label(filter_label = True).split('(')[0]

            # Lines
            line_kw = dict(color = tp.get_color(), lw = 10)
            ax.fill_betweenx([y]*2, *df[tp_col].int_68, **line_kw, alpha = 0.8)
            ax.fill_betweenx([y]*2, *df[tp_col].int_95, **line_kw, alpha = 0.5)

            # Mode marker
            kw_mode = dict(alpha = 1, zorder = 9, marker = 'o')
            ax.scatter(df[tp_col].Mode, y, **kw_mode, **marker_kw)
        
            # Numeric value of mode
            ax.annotate(
                text = f'{df[tp_col].Mode}', size = 9, 
                xy = (df[tp_col].Mode, y),
                xytext = (df[tp_col].Mode, i+0.35),
                ha = 'center', va = 'center', 
                fontweight = 'medium',
                )

    # axs[0].invert_yaxis()
    axs[0].set_ylim(-0.5, len(df.columns)- 0.25)
    axs[0].tick_params(labelleft=True, left=False)
    axs[1].tick_params(left=False)

    h_Mode = mlines.Line2D([], [], ls = 'None', **kw_mode, **marker_kw)
    h_68 = Patch(color = 'grey', alpha = 0.8)
    h_95 = Patch(color = 'grey', alpha = 0.5)

    l = ['Mode', '68$\,$% Interval', '95$\,$% Interval']
    h = [h_Mode, h_68, h_95]

    fig.tight_layout()
    fig.subplots_adjust(bottom = 0.2, top = 0.85)
    fig.suptitle('Distribution of ' + subs.label(bin_attr=bin_attr))
    fig.legend(h, l, loc = 'lower center', ncols = len(h))

def lognorm_ridgeline_comps(self, subs, zcoord, bin_attr='vstdv', eql=False, **kwargs): 
    """ Same as lognorm_fit_comparison but displaying the fits on a ridgeline graph. """
    lognorm_attr = f'{bin_attr}_fit'

    def lognorm_fit_pdf(value, lognorm_attr = 'vstdv_fit'):
        """ From dictionary value make a new not-normed PDF of the lognorm fit. """ 
        from scipy import stats
        ln_fit = getattr(value, lognorm_attr)
        fit_params = ln_fit.fit_params
        bins = ln_fit.bins
        
        x = np.linspace(min(bins) - 5, max(bins) + 5, len(bins)*2 + 2)    
        normed_ln_fit = stats.lognorm.pdf(x, *fit_params)

        area = sum(np.diff(ln_fit.bins) * ln_fit.counts)
        return x, normed_ln_fit * area
    
    # Create figure and axis
    fig, (ax_t, ax_s) = plt.subplots(1, 2, figsize = (12, 6))
    norm_factor = 20

    # TODO: fix get_Bin3D_dict calls
    for i, (ax, bin_dict) in enumerate(zip([ax_s, ax_t],
            self.get_Bin3D_dict(subs, zcoord, eql, **kwargs))): 
        # Get data
        modes = pd.Series({k:getattr(v, lognorm_attr).mode 
                            for k,v in bin_dict.items()}, name = 'Mode')
        fits = pd.Series({k:lognorm_fit_pdf(v, lognorm_attr)[1]
                        for k,v in bin_dict.items()}, name = 'Fits')
        fit_x = pd.Series({k:lognorm_fit_pdf(v, lognorm_attr)[0] 
                        for k,v in bin_dict.items()}, name = 'x')
        
        # Plotting each tropopause definition
        for i, tp in enumerate(self.tps):
            delta_y = (len(self.tps) - i) * norm_factor
            tp_col = tp.col_name
            
            mode = modes[tp_col]
            y_fit = fits[tp_col]
            x_fit = fit_x[tp_col]
            
            # Plotting white line above the data to separate the tps visually
            ax.plot(x_fit, 
                    y_fit + delta_y + 0.8,
                    color='white', 
                    lw = 2)
            
            # Plotting strong data line
            ax.plot(x_fit, 
                    y_fit + delta_y, 
                    color=tp.get_color(), 
                    lw = 3)
            
            # Plotting filled area
            ax.fill_between(x_fit, 
                            y_fit + delta_y, 
                            color=tp.get_color(), 
                            label=tp.label(filter_label=True),
                            alpha = 0.75)
            
            # Block out everything below the offset / y-start 
            ax.fill_between([-1] + list(x_fit) + [300], 
                            delta_y-1, 
                            color='white')
            
            # Add vertical lines and annotations for the distribution's modes
            ax.vlines(mode, delta_y, delta_y + max(y_fit), 
                    color = 'k', ls = 'dashed', zorder = 1)
            ax.annotate(
                    text = f'{np.format_float_positional(mode, 1)}',
                    xy = (mode + (2 if i==0 else 0.5), delta_y + max(y_fit) / 2),
                    va = 'center', ha = 'left'
                    )

            ax.set_xlabel(subs.label(bin_attr = 'vstdv'))
            ax.get_yaxis().set_visible(False)  # Hides the y-axis tick labels

            # ax.set_xscale('symlog')
            # ax.set_xlim(0, 300)
            ax.set_ylim(15, 7.5*norm_factor)

    ax_s.set_xlim(0, 150) 
    ax_t.set_xlim(0, 50) 

    ax_s.set_title('Stratosphere')
    ax_t.set_title('Troposphere')

    # Display the plot
    fig.tight_layout()
    fig.subplots_adjust(right = 0.75)
    fig.legend(*ax_s.get_legend_handles_labels(), 
            loc = 'right', fontsize = 11)


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
    if vlims is None: vlims = get_vlimit(subs, bin_attr)
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
    
    raise NotImplementedError('sel_LMS works, calling bin_3d needs to be updated, ...')
    
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
    
    bin_equi3d = bp.Bin_equi3d(-180, 180, xbsize, 
                                -90, 90, ybsize, 
                                0, zbsize*2, zbsize)
    
    vlims = kwargs.get('vlims')
    if vlims is None: vlims = get_vlimit(subs, bin_attr)
    
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

# %%