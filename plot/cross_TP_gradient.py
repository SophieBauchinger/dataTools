# -*- coding: utf-8 -*-
""" Cross-tropopause gradient statistics and plotting

@Author: Sophie Bauchinger, IAU
@Date: Mon Mar 10 15:30:00 2025
"""

import cmasher as cmr
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.patheffects as mpe
import numpy as np
import pandas as pd
import seaborn as sns

import dataTools.dictionaries as dcts
from dataTools import tools
import dataTools.data.tropopause as tp_tools
import dataTools.data.BinnedData as bin_tools


def seasonal_gradient_comparison(GlobalObject, subs, tps, **kwargs):
    """ Calculate and show normalised seasonal cross-tropopause gradients. 

    Args:
        GlobalObject (dataTools.data.GlobalData)
        subs (dcts.Substance|dcts.Coordinate)
        tps (list[dcts.Coordinate]]): TP-relative coordinates 
    """
    seas_tropo_avs = tp_tools.seasonal_tropospheric_average(GlobalObject, subs, tps)
    
    xbsize = kwargs.pop('xbsize', 0.5)
    gradient_seas_norm_df = pd.DataFrame(columns = [tp.col_name for tp in tps])
    for tp in tps:
        bin_dict = bin_tools.seasonal_binning(
            GlobalObject.df, subs, tp, xbsize=xbsize, **kwargs)
        for s in bin_dict.keys():
            i_pos = next(x for x, val in enumerate(bin_dict[s].xintm) if val > 0) 
            continue
        norm_gradients = dict()
        for s in bin_dict.keys(): 
            gradient = (bin_dict[s].vmean[i_pos] - bin_dict[s].vmean[i_pos-1]) / bin_dict[s].xbsize
            norm_gradients[s] = gradient / seas_tropo_avs[tp.col_name][s]
        
        gradient_seas_norm_df[tp.col_name] = norm_gradients
        
    # gradient_seas_norm_df.drop(columns = 'N2O_residual', inplace=True)
    gradient_seas_norm_df.loc['Average'] = gradient_seas_norm_df.mean(axis=0)

    seasons = [dcts.dict_season()[f'name_{s}'] for s in bin_dict.keys()]

    # Show the gradient fractions as heatmap
    plt.figure(figsize=(6, 3), dpi=200)
    ax = sns.heatmap(
        gradient_seas_norm_df.T, 
        annot=True, cmap='YlGnBu', #cmr.get_sub_cmap('PRGn', 0.1, 0.9), # 'PRGn',# 'YlGnBu', 
        cbar_kws={'label': 'Gradient [fraction/km]'}, 
        yticklabels=[dcts.get_coord(c).label(filter_label=True) 
                     for c in gradient_seas_norm_df.columns], 
        
        xticklabels=seasons + ['Average'],
        
        # xticklabels=['Spring', 'Summer', 'Autumn', 'Winter', 'Average'],
        fmt='.2f',
        linewidths=1, linecolor='white',
        )
    df = gradient_seas_norm_df.T
    # Highlight the "Average" row with a separate color (light gray)    
    av_values = df['Average']
    norm = Normalize(av_values.min(), av_values.max())
    cmap = cmr.get_sub_cmap('binary', 0.2, 0.65)
    facecolors = cmap([norm(v) for v in av_values])
    for i in range(len(df)):
        ax.add_patch(plt.Rectangle((len(df.columns)-1, i), 1, 1, # xy, width, height
                                   facecolor = facecolors[i],# 'lightgrey', 
                                   lw=2, edgecolor='white'))
    plt.show()
    return gradient_seas_norm_df.T


def plot_1d_gradient(ax, s, bin_obj,
                     bin_attr: str = 'vmean', 
                     add_stdv: bool = False, 
                     **kwargs):
    """ Create scatter/line plot for the given binned parameter for a single season. """
    
    outline = mpe.withStroke(linewidth=2, foreground='white')
    
    color = kwargs.get('color', dcts.dict_season()[f'color_{s}'])
    label = kwargs.get('label', dcts.dict_season()[f'name_{s}'])

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
                color = color, label = label,
                linewidth=2,# if not kwargs.get('big') else 3,
                path_effects = [outline], zorder = 2)

    ax.scatter(vdata, y, 
                marker=marker,
                color = color, zorder = 3)


def plot_1d_seasonal_gradient(GlobalObject, subs, coord, 
                                bin_attr: str = 'vmean', 
                                add_stdv: bool = False, 
                                **kwargs):
    """ Plot gradient per season onto one plot. """
    big = kwargs.pop('big') if 'big' in kwargs else False
    
    bin_dict = bin_tools.seasonal_binning(GlobalObject.df, subs, coord, **kwargs)
    
    if 'figax' in kwargs: 
        fig, ax = kwargs.get('figax')
    else: 
        fig, ax = plt.subplots(dpi=500, figsize= (6,4) if not big else (3,4))
    # fig, ax = plt.subplots(dpi=500, figsize= (6,4) if not big else (3,4))

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

    if not big: 
        ax.legend(loc=kwargs.get('legend_loc', 'lower left'))
    ax.grid('both', ls='dashed', lw=0.5)
    ax.set_axisbelow(True)
    
    if coord.rel_to_tp: 
        tools.add_zero_line(ax)

    return bin_dict, fig 