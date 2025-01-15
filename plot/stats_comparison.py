# -*- coding: utf-8 -*-
""" Stats comparison for binned data - mean/mode, interval widths. 

@Author: Sophie Bauchinger, IAU
@Date Fri Jan 10 12:05:00 2025

"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import dataTools.dictionaries as dcts
from dataTools import tools
import dataTools.plot.create_figure as cfig
from dataTools.data.BinnedData import extract_attr

#%% Lognorm stats as lines with dots

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

def plot_lognorm_stats(ax, df, s = None): 
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

def seasonal_lognorm_stats(self, strato_Bin_seas_dict, tropo_Bin_seas_dict, 
                           strato_var, tropo_var = None, bin_attr = 'vstdv', **kwargs): 
    """ Create plot of lognormal stats for each tropopause in troposphere / stratosphere. 
    Args: 
        axs (tuple[plt.Axes]): Tuple of tropos_axis, stratos_axis
        strato_stats (pd.DataFrame): Stratospheric lognorm fit statistics
        tropo_stats (pd.DataFrame): Tropospheric lognorm fit statistics
        
        strato_var (dcts.Substance|dcts.Coordinate)
        tropo_var: Optional. Defaults to strato_var
    
    """
    if not all(i in kwargs for i in ['axs', 'fig']): 
        fig, axs = plt.subplots(1,2, figsize = (8,5), sharey=True, dpi=250)
        axs[0].set_title(f'Troposphere', size = 10, pad = 3)
        axs[1].set_title(f'Stratosphere', size = 10, pad = 3)
    else: 
        fig = kwargs.get('fig')
        axs = kwargs.get('axs')

    if tropo_var is None:
        tropo_var = strato_var

    seasons = set(next(iter(strato_Bin_seas_dict.values())))
    
    for s in seasons:
        strato_BinDict = {k:v[s] for k,v in strato_Bin_seas_dict.items()}
        tropo_BinDict = {k:v[s] for k,v in tropo_Bin_seas_dict.items()}

        strato_stats = get_lognorm_stats_df(strato_BinDict, 
                                                f'{bin_attr}_fit', 
                                                prec = kwargs.get('prec', 1), 
                                                use_percentage = strato_var.detr)
        tropo_stats = get_lognorm_stats_df(tropo_BinDict, 
                                                f'{bin_attr}_fit', 
                                                prec = kwargs.get('prec', 1),
                                                use_percentage = tropo_var.detr)

        for df, ax in zip([tropo_stats, strato_stats], axs):
            plot_lognorm_stats(ax, df, s)
    
    y_arr =  [s*8+2.5 for s in seasons]# [i*8+s for i in range(len(df.columns)) for s in seasons]
    y_ticks = [dcts.dict_season()[f'name_{s}'] for s in seasons]# [dcts.get_coord(tp_col).label(filter_label = True).split('(')[0] for tp_col in df.columns]

    axs[0].set_yticks(y_arr, y_ticks)
    axs[0].tick_params(labelleft=True if not kwargs.get('label')=='off' else False, 
                        left=False)
    axs[1].tick_params(left=False, labelleft=False)
    
    for ax in axs:
        ax.set_xlim(ax.get_xlim()[0]-ax.get_xlim()[1]/8, ax.get_xlim()[1])
        ax.grid(axis='x', ls = 'dashed')
    axs[0].set_xlabel(tropo_var.label(bin_attr=bin_attr))
    axs[1].set_xlabel(strato_var.label(bin_attr=bin_attr))

    axs[0].legend(handles = self.tp_legend_handles(filter_label=True, no_vc = True), 
            prop=dict(size = 6));

    fig.subplots_adjust(bottom = 0.15, wspace = 0.05)
    fig.legend(*self.lognorm_legend_handles(), loc = 'lower center', ncols = 3)

    return axs

def seasonal_2d_lognorm_stats(self, tropo_params, strato_params,
                              bin_attr='vstdv', **kwargs): 
    """ 
    Show seasonal statistics of lognorm fit on bin_attr of binned data. 
    
    Parameters: 
        tropo/strato_params (dict): Required var, xcoord, ycoord.
            Optional keys: xbsize, ybsize, ...
        bin_attr (str): Bin attribute to base statistics on
           
    """
    
    strato_var = strato_params.pop('var')
    strato_xcoord = strato_params.pop('xcoord')
    strato_ycoord = strato_params.pop('ycoord')
    
    tropo_var = tropo_params.pop('var')
    tropo_xcoord = tropo_params.pop('xcoord')
    tropo_ycoord = tropo_params.pop('ycoord')

    strato_Bin2Dseas_dict, tropo_Bin2Dseas_dict = {}, {}
    for tp in self.tps[::-1]:
        strato_Bin2Dseas_dict[tp.col_name] = self.sel_strato(
            tp).bin_2d_seasonal(
                strato_var, strato_xcoord, strato_ycoord, **strato_params
                )
        tropo_Bin2Dseas_dict[tp.col_name] = self.sel_tropo(
            tp).bin_2d_seasonal(
                tropo_var, tropo_xcoord, tropo_ycoord, **tropo_params,
                )
        
    fig, axs = plt.subplots(1,2, figsize = (8,5), sharey=True, dpi=250)
    
    # Labels 
    txcbs = tropo_params.get('xbsize', tropo_xcoord.get_bsize())
    tycbs = tropo_params.get('ybsize', tropo_ycoord.get_bsize())
    
    txc_label = tropo_xcoord.label(coord_only=True) + f' [{txcbs}]'
    tyc_label = tropo_ycoord.label(coord_only=True) + f' [{tycbs}]'
    
    sxcbs = strato_params.get('xbsize', strato_xcoord.get_bsize())
    sycbs = strato_params.get('ybsize', strato_ycoord.get_bsize())
    
    sxc_label = strato_xcoord.label() +f' - {sxcbs}'
    syc_label = strato_ycoord.label() +f' - {sycbs}'

    # axs[0].set_title('Troposphere (' + txc_label + r' $\times$ ' + tyc_label + ')',
    #                  size = 10, pad = 3)
    # axs[1].set_title('Stratosphere (' + sxc_label + r' $\times$ ' + syc_label + ')',
    #                  size = 10, pad = 3)

    axs[0].set_title('Troposphere')
    axs[1].set_title('Stratosphere')
    

    seasonal_lognorm_stats(self, strato_Bin2Dseas_dict, tropo_Bin2Dseas_dict, 
                           strato_var, tropo_var, 
                           bin_attr, axs=axs, fig=fig, **kwargs)


#%% Lognorm fitted histograms 

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
                lw = 1)- m 

    return lognorm_inst

def plot_histogram_comparison(self, tropo_var, strato_var, tropo_dict, strato_dict, bin_attr='vstdv', 
                                xscale = 'linear', show_stats = False, fig_kwargs = {}, **kwargs):
    """ Plot histogram with lognorm fit comparison between tropopauses. 
    
    Args: 
        tropo_BinDict, strato_BinDict (dict[tp:Bin*DFitted]): 
            Fitted binned data in x dimensions
        
        key season (int): If passed, add suptitle with current season
    """
    tps = [self.get_coords(col_name = k)[0] for k in tropo_dict.keys()]

    fig, main_axes, sub_ax_arr = cfig.nested_subplots_two_column_axs(tps, **fig_kwargs)
    cfig.adjust_labels_ticks(sub_ax_arr)

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
        ax.grid(ls ='dotted', lw = 1, color='grey', zorder=0)
        ax.set_xscale(xscale)
        
    for ax in tropo_axs:
        ax.set_ylabel(tropo_var.label(bin_attr=bin_attr), fontsize = 8)
    for ax in strato_axs:
        ax.set_ylabel(strato_var.label(bin_attr=bin_attr), fontsize = 8)

    # Add histograms and lognorm fits
    for axes, data_Bin_dict in zip([tropo_axs, strato_axs], [tropo_dict, strato_dict]): 
        # Extract bin_attr
        data_dict = extract_attr(self, data_Bin_dict, bin_attr)
        print(data_dict)
            
        # Get overall tropo / strato bin limits
        hist_min, hist_max = np.nan, np.nan
        for data in data_dict.values(): 
            hist_min = np.nanmin([hist_min] + list(data.flatten()))
            hist_max = np.nanmax([hist_max] + list(data.flatten()))

        for ax, tp_col in zip(axes, data_dict): 
            data_flat = data_dict[tp_col].flatten() # fails if no bin_attr extracted
            print(data_flat)
            # Adding the histograms to the figure
            lognorm_inst = hist_lognorm_fitted(data_flat, (hist_min, hist_max), ax, 
                                               dcts.get_coord(tp_col).get_color(),
                                               hist_kwargs = dict(range = (hist_min, hist_max)))
            print(lognorm_inst)
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

def histogram_2d_comparison(self, tropo_params, strato_params, bin_attr='vstdv', **kwargs):
    """ 2D-binned data lognorm-fitted histograms. 

    Parameters: 
        var (dcts.Substance|dcts.Coordinate)
        bin_attr (str)
    """

    tropo_var = tropo_params.pop('var')
    tropo_xcoord = tropo_params.pop('xcoord')
    tropo_ycoord = tropo_params.pop('ycoord')

    strato_var = strato_params.pop('var')
    strato_xcoord = strato_params.pop('xcoord')
    strato_ycoord = strato_params.pop('ycoord')

    tropo_BinDict, strato_BinDict = {}, {}

    for tp in kwargs.get('tps', self.tps):
        tropo_BinDict[tp.col_name] = self.sel_tropo(tp).bin_2d(
            tropo_var, tropo_xcoord, tropo_ycoord, **tropo_params)
        strato_BinDict[tp.col_name] = self.sel_strato(tp).bin_2d(
            strato_var, strato_xcoord, strato_ycoord, **strato_params)

    plot_histogram_comparison(self, tropo_var, strato_var,
                              tropo_BinDict, strato_BinDict,
                              bin_attr=bin_attr, **kwargs)
