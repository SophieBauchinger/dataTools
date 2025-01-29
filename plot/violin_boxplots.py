# -*- coding: utf-8 -*-
""" Voilin and boxplots for binned data. 

@Author: Sophie Bauchinger, IAU
@Date Fri Jan 10 11:45:00 2025

"""

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 12})

def violin_boxplot(self, var, xcoord, ycoord, atm_layer, **kwargs):
    """ 
    Create violin and boxplots. 

    Parameters: 
        var (dcts.Coordinates)
        xcoord (dcts.Coordinates)
        ycoord (dcts.Coordinates)

    """
    # Prep the data -------------------------------------------------
    filtered_data = []
    tps = kwargs.get('tps', self.tps)
    for tp in tps:
        bin2d = self.sel_atm_layer(
            atm_layer, tp).bin_2d(
                var, xcoord, ycoord, 
                xbsize = kwargs.pop('xbsize', 5), 
                ybsize = kwargs.pop('ybsize', 0.5),
                **kwargs)
        
        data = getattr(bin2d, kwargs.get('bin_attr', 'vstdv')).flatten()    
        # data = bin2d.vstdv.flatten()
        data = data[~np.isnan(data)]
        
        filtered_data.append(data)
    ax_label = var.label(bin_attr = kwargs.get('bin_attr', 'vstdv'))

    # Prep stats and helpers 
    means = [y.mean() for y in filtered_data]
    stds = [np.std(y) for y in filtered_data]
    skews = [stats.skew(y) for y in filtered_data]

    COLOR_SCALE = [tp.get_color() for tp in tps]
    LABELS = [tp.label(filter_label = True) for tp in tps]
    POSITIONS = np.linspace(0, len(tps)-1, len(tps))


    # Make the figure 
    if 'figax' in kwargs: 
        fig, ax = kwargs.get('figax')
    else: plt.subplots(figsize= (8, 5), dpi = 300)

    # Add violins ----------------------------------------------------
    # bw_method="silverman" is the bandwidth of the kernel density
    violins = ax.violinplot(
        filtered_data, 
        positions=POSITIONS,
        widths=0.6,
        bw_method="silverman",
        showmeans=False, 
        showmedians=False,
        showextrema=False,
    )

    # Customize violins (remove fill, customize line, etc.)
    for pc, c in zip(violins["bodies"], COLOR_SCALE):
        pc.set_facecolor(c)
        pc.set_edgecolor('k')
        pc.set_linewidth(0.8)
        pc.set_alpha(0.9)
        
    # Add boxplots 
    medianprops = dict(
        linewidth=1.5, 
        color="xkcd:dark grey",
        solid_capstyle="butt"
    )
    boxprops = dict(
        linewidth=1, 
        color="xkcd:dark"
    )
    ax.boxplot(
        filtered_data,
        widths = 0.25,
        positions=POSITIONS, 
        showfliers = True, # Do not show the outliers beyond the caps.
        showcaps = False,   # Do not show the caps
        medianprops = medianprops,
        whiskerprops = boxprops,
        boxprops = boxprops
    )

    # Add jittered dots ----------------------------------------------
    # jitter = 0.05
    # x_data = [np.array([i] * len(d)) for i, d in enumerate(filtered_data)]
    # x_jittered = [x + stats.t(df=6, scale=jitter).rvs(len(x)) for x in x_data]

    # for x, y, color, label in zip(x_jittered, filtered_data, COLOR_SCALE, LABELS):
    #     ax.scatter(x, y, s = 80, color=color, alpha=0.3, label = label)
        
    # Add statistics labels ------------------------------------------
    for i, mean in enumerate(means):
        # Add dot representing the mean
        ax.scatter(i, mean, s=50, 
                   color="#850e00", 
                   edgecolor ='k', zorder=3,
                   label = 'Mean' if i==0 else '')

    y_stats = [
    f'$\mu$ = {m:.1f}\n\
$\sigma$ = {s:.1f}\n\
$\gamma$ = {y:.1f}\n\
    ' for m,s,y in zip(means, stds, skews)]

    for x,y,s in zip(POSITIONS, 
                    [max(y) for y in filtered_data], 
                    y_stats):
        ax.text(x = x, y = y, s = s, 
                ha = 'center', va = 'bottom')

    # ax.set_ylim(0, ax.get_ylim()[1]+23)
    ax.set_ylim(max([0, ax.get_ylim()[0]]), 
                ax.get_ylim()[1]*kwargs.get('vspace', 1.2))

    ax.tick_params(labelbottom=False, bottom=False)
    ax.set_ylabel(ax_label)
    fig.tight_layout()

    # Add legend
    fig.subplots_adjust(bottom = 0.15)
    
    ax.legend(loc = 'upper left')
    
    fig.legend(handles = self.tp_legend_handles(no_vc = True), 
              ncols = kwargs.get('legend_ncols', 3),
              loc = kwargs.get('legend_loc', 'lower center'))
    ax.set_title('Troposphere' if atm_layer =='tropo' else 'Stratosphere')
    

def rel_binning_violin_boxplot(self, var, rel_coords, ycoord, atm_layer, **kwargs):
    """ 
    Create violin and boxplots. 

    Parameters: 
        var (dcts.Coordinates)
        xcoord (dcts.Coordinates)
        ycoord (dcts.Coordinates)

    """
    tps = kwargs.get('tps', self.tps)
    # Prep the data -------------------------------------------------
    filtered_data = []
    for tp, rel_c in zip(tps, rel_coords):
        bin2d = self.sel_atm_layer(
            atm_layer, tp).bin_2d(
                var, rel_c, ycoord, 
                xbsize = kwargs.get('xbsize', 5), 
                ybsize = kwargs.get('ybsize', 0.5))
        data = bin2d.vstdv.flatten()
        data = data[~np.isnan(data)]
        
        filtered_data.append(data)
    ax_label = var.label(bin_attr = 'vstdv')

    # Prep stats and helpers 
    means = [y.mean() for y in filtered_data]
    stds = [np.std(y) for y in filtered_data]
    skews = [stats.skew(y) for y in filtered_data]

    COLOR_SCALE = [tp.get_color() for tp in tps]
    LABELS = [tp.label(filter_label = True) for tp in tps]
    POSITIONS = np.linspace(0, len(tps)-1, len(tps))


    # Make the figure 
    if 'figax' in kwargs: 
        fig, ax = kwargs.get('figax')
    else: plt.subplots(figsize= (8, 5), dpi = 300)
    
    # Add violins ----------------------------------------------------
    # bw_method="silverman" is the bandwidth of the kernel density
    violins = ax.violinplot(
        filtered_data, 
        positions=POSITIONS,
        widths=0.6,
        bw_method="silverman",
        showmeans=False, 
        showmedians=False,
        showextrema=False,
    )

    # Customize violins (remove fill, customize line, etc.)
    for pc, c in zip(violins["bodies"], COLOR_SCALE):
        pc.set_facecolor(c)
        pc.set_edgecolor('k')
        pc.set_linewidth(0.8)
        pc.set_alpha(0.9)
        
    # Add boxplots 
    medianprops = dict(
        linewidth=1.5, 
        color="xkcd:dark grey",
        solid_capstyle="butt"
    )
    boxprops = dict(
        linewidth=1, 
        color="xkcd:dark"
    )
    ax.boxplot(
        filtered_data,
        widths = 0.25,
        positions=POSITIONS, 
        showfliers = True, # Do not show the outliers beyond the caps.
        showcaps = False,   # Do not show the caps
        medianprops = medianprops,
        whiskerprops = boxprops,
        boxprops = boxprops
    )

    # Add jittered dots ----------------------------------------------
    # jitter = 0.05
    # x_data = [np.array([i] * len(d)) for i, d in enumerate(filtered_data)]
    # x_jittered = [x + stats.t(df=6, scale=jitter).rvs(len(x)) for x in x_data]

    # for x, y, color, label in zip(x_jittered, filtered_data, COLOR_SCALE, LABELS):
    #     ax.scatter(x, y, s = 80, color=color, alpha=0.3, label = label)
        
    # Add statistics labels ------------------------------------------
    for i, mean in enumerate(means):
        # Add dot representing the mean
        ax.scatter(i, mean, s=50, 
                   color="#850e00", 
                   edgecolor ='k', zorder=3,
                   label = 'Mean' if i==0 else '')

    y_stats = [
    f'$\mu$ = {m:.1f}\n\
$\sigma$ = {s:.1f}\n\
$\gamma$ = {y:.1f}\n\
    ' for m,s,y in zip(means, stds, skews)]

    for x,y,s in zip(POSITIONS, 
                    [max(y) for y in filtered_data], 
                    y_stats):
        ax.text(x = x, y = y, s = s, 
                ha = 'center', va = 'bottom')

    # ax.set_ylim(0, ax.get_ylim()[1]+23)
    ax.set_ylim(max([0, ax.get_ylim()[0]]), 
                ax.get_ylim()[1]*kwargs.get('vspace', 1.2))

    ax.tick_params(labelbottom=False, bottom=False)
    ax.set_ylabel(ax_label)
    fig.tight_layout()

    # Add legend
    fig.subplots_adjust(bottom = 0.15)
    
    ax.legend(loc = 'upper left') # mean
    # fig.legend(handles = self.tp_legend_handles(no_vc = True), 
    #           ncols = kwargs.get('legend_ncols', 3),
    #           loc = kwargs.get('legend_loc', 'lower center'))
    # ax.set_title('Troposphere' if atm_layer =='tropo' else 'Stratosphere')
    
    if 'legend_loc' in kwargs: 
        ax.legend(handles = self.tp_legend_handles(tps=tps, no_vc = True), 
                ncols = kwargs.get('legend_ncols', 3),
                loc = kwargs.get('legend_loc', 'lower center'))
    else: 
        fig.legend(handles = self.tp_legend_handles(tps = tps, no_vc = True), 
                   ncols = kwargs.get('legend_ncols', 3),
                   loc = kwargs.get('legend_loc', 'lower center')) 
    ax.set_title('Troposphere' if atm_layer =='tropo' else 'Stratosphere')
