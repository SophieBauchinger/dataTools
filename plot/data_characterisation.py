# -*- coding: utf-8 -*-
""" Data characterisation for CARIBIC 

@Author: Sophie Bauchinger, IAU
@Date Fri Jan 22 12:23:00 2025
"""

import matplotlib.pyplot as plt
import matplotlib.patheffects as mpe
import numpy as np

import dataTools.dictionaries as dcts
from dataTools import tools
import dataTools.data.BinnedData as bin_data
import dataTools.plot.create_figure as cfig

def tp_height_seasonal_1D_binned(self, tp, **kwargs): 
    """ Plot the average tropopause height (or delta) per season binned over latitude. 
    
    Args: 
        key df (pd.DataFrame): Dataset from which to draw plot info. Defaults to self.df
        key xcoord (dcts.Coordinate): Coord used for 1D binning. Defaults to latitude
        key bsize (float): Bin size
    """
    df = kwargs.get('df', self.df)
    coord = kwargs.get('coord', dcts.get_coord(col_name = 'geometry.y'))
    xbsize = kwargs.get('bsize', coord.get_bsize())
    bci = bin_data.make_bci(coord, xbsize=xbsize, gdf = self.df)
    n2o_color = 'g'

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
        bin1d = bin_data.binning(data, tp, xcoord = coord, bci = bci, 
                                 count_limit = self.count_limit)

        plot_kwargs = dict(lw=3, path_effects = [cfig.outline()])
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
    fig, axs = plt.subplots(3,2, figsize = (8, 8), sharex=True, 
                            dpi=kwargs.pop('dpi', 150))
    ylims = kwargs.pop('ylims', None)
    
    for tp, ax in zip(tps, axs.flat): 
        tp_height_seasonal_1D_binned(self,
            tp, ax = ax, 
            invert_yaxis = True if tp.vcoord =='mxr' else False,
            ylims = ylims if not (tp.vcoord=='mxr' or ylims is None) else tp.get_lims(), 
            **kwargs)
        if tp.crit == 'n2o': 
            n2o_color = 'g'
            ax.tick_params(axis='y', color=n2o_color, labelcolor=n2o_color)
            ax.spines['right'].set_color(n2o_color)
            ax.spines['left'].set_color(n2o_color)
    fig.suptitle('Vertical extent of tropopauses')
    fig.tight_layout()
    fig.subplots_adjust(top = 0.85)
    fig.legend(handles = self.season_legend_handles(av=True), 
                ncol = 3, loc='upper center', 
                bbox_to_anchor=[0.5, 0.95])


#%% 1D seasonal gradients

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
        
def plot_1d_seasonal_gradient(self, subs, coord, 
                                bin_attr: str = 'vmean', 
                                add_stdv: bool = False, 
                                **kwargs):
    """ Plot gradient per season onto one plot. """
    big = kwargs.pop('big') if 'big' in kwargs else False
    bin_dict = bin_data.binning_seasonal(subs, xcoord = coord,
                                         count_limit = self.count_limit,
                                         **kwargs)
    if 'figax' in kwargs: 
        _, ax = kwargs.get('figax')
    else: 
        _, ax = plt.subplots(dpi=500, figsize= kwargs.get(
            'figsize', (6,4)) if not big else (3,4))

    if coord.vcoord=='pt' and coord.rel_to_tp: 
        ax.set_yticks(np.arange(-60, 75, 20) + [0])

    for s in bin_dict.keys():
        plot_1d_gradient(ax, s, bin_dict[s], bin_attr, add_stdv)
        
    # ax.set_title(coord.label(filter_label=True))
    # ax.set_ylabel(coord.label())

    if bin_attr=='vmean':
        ax.set_xlabel(subs.label())
    elif bin_attr=='vstdv': 
        ax.set_xlabel('Relative variability of '+subs.label(name_only=True))

    if (coord.vcoord in ['mxr', 'p'] and not coord.rel_to_tp) or coord.col_name == 'N2O_residual': 
        ax.invert_yaxis()
    if coord.vcoord=='p': 
        ax.set_yscale('symlog' if coord.rel_to_tp else 'log')

    if not big: 
        ax.legend(loc=kwargs.get('legend_loc', 'lower left'))
    ax.grid('both', ls='dashed', lw=0.5)
    ax.set_axisbelow(True)
    
    if coord.rel_to_tp is True: 
        tools.add_zero_line(ax)

    return bin_dict
