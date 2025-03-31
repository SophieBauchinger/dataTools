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
    xbsize = kwargs.pop('xbsize', 0.5)
    
    param_dict = {}
    gradient_df = pd.DataFrame(columns = [tp.col_name for tp in tps])
    for tp in tps:
        bin_dict = bin_tools.seasonal_binning(
            GlobalObject.df, subs, tp, xbsize=xbsize, **kwargs)
        for s in bin_dict.keys():
            i_pos = next(x for x, val in enumerate(bin_dict[s].xintm) if val > 0) 
            continue
        norm_gradients, gradients = {}, {}
        param_dict[tp.col_name] = {}
        for s in bin_dict.keys(): 
            i_pos_gradient = i_pos if not 'offset' in kwargs else i_pos+kwargs.get('offset')
            pos_val = bin_dict[s].vmean[i_pos_gradient]
            neg_val = bin_dict[s].vmean[i_pos_gradient-1]
            gradient = (pos_val - neg_val) / bin_dict[s].xbsize
            TP_average = np.mean(list(bin_dict[s].vbindata[i_pos]) + list(bin_dict[s].vbindata[i_pos-1]))
            norm_gradients[s] = gradient / TP_average
            gradients[s] = gradient
        
            param_dict[tp.col_name][s] = pd.Series(dict(
                    neg_val = neg_val,
                    TP_average = TP_average,
                    pos_val = pos_val, 
                    gradient = gradient,
                    norm_gradient = gradient / TP_average
                ))
        gradient_df[tp.col_name] = norm_gradients if kwargs.get('norm', True) else gradients
        
    gradient_df.loc['Average'] = gradient_df.mean(axis=0)
    seasons = [dcts.dict_season()[f'name_{s}'].split()[0] for s in bin_dict.keys()]

    param_df = pd.DataFrame.from_dict(
        {(i,j): param_dict[i][j] 
        for i in param_dict.keys() 
        for j in param_dict[i].keys()},
        orient='index')

    # Show the gradient fractions as heatmap
    if 'figax' in kwargs: 
        fig, ax = kwargs.get('figax')
    else: 
        fig, ax = plt.subplots(figsize=(6, 3), dpi=500)
    sns.heatmap(
        gradient_df.T, 
        annot=True, cmap='YlGnBu', #cmr.get_sub_cmap('PRGn', 0.1, 0.9), # 'PRGn',# 'YlGnBu', 
        cbar_kws={'label': 'Gradient [fraction/km]'}, 
        yticklabels=[dcts.get_coord(c).label(filter_label=True) 
                     for c in gradient_df.columns], 
        
        xticklabels=seasons + ['Average'],
        
        # xticklabels=['Spring', 'Summer', 'Autumn', 'Winter', 'Average'],
        fmt='.2f', ax=ax,
        linewidths=1.5, linecolor='white',
        )
    df = gradient_df.T
    # Highlight the "Average" row with a separate color (light gray)    
    av_values = df['Average']
    norm = Normalize(av_values.min(), av_values.max())
    cmap = cmr.get_sub_cmap('binary', 0.35, 0.7)
    facecolors = cmap([norm(v) for v in av_values])
    for i in range(len(df)):
        ax.add_patch(plt.Rectangle((len(df.columns)-1, i), 1, 1, # xy, width, height
                                   facecolor = facecolors[i],# 'lightgrey', 
                                   lw=2, edgecolor='white'))
    if 'figax' in kwargs:
        return fig, ax
    else:
        plt.show()
    return gradient_df.T, param_df

def format_gradient_params_LATEX(gradient_params): 
    """ Returns very specific formatting of the gradient parameter dataframe to copy into .tex file """
    seasons = gradient_params.reset_index()['level_1'].apply(lambda x: '& ' + dcts.dict_season()[f'name_{x}'].split(' ')[0])
    param_df = gradient_params.reset_index().drop(columns = ['level_0'])
    param_df['level_1']  = seasons
    param_df.set_index('level_1', inplace=True)
    print(param_df.to_latex(header=False, float_format="%.2f"))

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