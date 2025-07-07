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
from scipy.optimize import curve_fit

import toolpac.outliers.ol_fit_functions as fctns

import dataTools.dictionaries as dcts
from dataTools import tools
import dataTools.data.BinnedData as bin_tools
import dataTools.plot.create_figure as cfig

# Heatmap for plotting gradients per tropopause defintion and season
def gradient_heatmap(df, **kwargs):
    """ Heatmap for plotting gradients per tropopause defintion and season.
    Parameters: 
        df (pd.DataFrame): 
            (Normalised) gradients per season (& av.) per TP def.
            
        key cbar_label (str): Label for the colorbar. Default: Gradient [fraction/km]
        key figax ((plt.Figure, plt.Axes))
        key vlims (float,float): Normalisation limits for the colorbar
        
    """
    if 'figax' in kwargs: 
        fig, ax = kwargs.get('figax')
    else: 
        fig, ax = plt.subplots(figsize=(6, 3), dpi=200)
    if not 'Average' in df.index:
         df.loc['Average'] = df.mean(axis=0)

    sns.heatmap(
        df.T, annot=True, cmap=kwargs.get('cmap', 'YlGnBu'), #cmr.get_sub_cmap('PRGn', 0.1, 0.9), # 'PRGn',# 'YlGnBu', 
        norm = Normalize(*kwargs.get('vlims')) if 'vlims' in kwargs else None,
        cbar_kws={'label': kwargs.get('cbar_label', ('Gradient [fraction/km]')), 
                  'pad' : 0.05, # space between plot and cbar
        },
                #   'labelpad' : 15}, 
        yticklabels=[dcts.get_coord(c).label(filter_label=True, no_vc=True, no_model=True) for c in df.columns], 
        xticklabels=['Spring', 'Summer', 'Autumn', 'Winter', 'Average'],
        fmt='.2f', ax = ax,
        linewidths=1, linecolor='white',
        )

    # Highlight the "Average" row with a separate color (light gray)
    av_values = df.loc['Average']
    norm = Normalize(av_values.min(), av_values.max())
    cmap = cmr.get_sub_cmap('binary', 0.35, 0.7)
    facecolors = cmap([norm(v) for v in av_values])
    for i in range(len(df.T)):
        ax.add_patch(plt.Rectangle((len(df.index)-1, i), 1, 1, 
                                    facecolor = facecolors[i], 
                                    lw=2, edgecolor='white'))
    
    for text in ax.texts:
        if (int(text._x) == len(df.columns) - 1) and (norm(float(text.get_text())) > 0.4):
            text.set_color('white')
    return fig, ax

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
        # This is an insane way to do it and I know it
        bin_dict_for_TP_av = bin_tools.seasonal_binning(
            GlobalObject.df, subs, tp, xbsize=xbsize/2, **kwargs)
        for s in bin_dict.keys():
            i_pos = next(x for x, val in enumerate(bin_dict[s].xintm) if val > 0) 
            i_pos_TP_av = next(x for x, val in enumerate(bin_dict_for_TP_av[s].xintm) if val > 0) 
            continue
        norm_gradients, gradients = {}, {}
        param_dict[tp.col_name] = {}
        for s in bin_dict.keys(): 
            i_pos_gradient = i_pos if not 'offset' in kwargs else i_pos+kwargs.get('offset')
            pos_val = bin_dict[s].vmean[i_pos_gradient]
            neg_val = bin_dict[s].vmean[i_pos_gradient-1]
            gradient = (pos_val - neg_val) / bin_dict[s].xbsize
            TP_average = np.mean( # pls forgive me
                list(bin_dict_for_TP_av[s].vbindata[i_pos_TP_av]) 
                + list(bin_dict_for_TP_av[s].vbindata[i_pos_TP_av-1]))
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

    fig, ax = gradient_heatmap(gradient_df, **kwargs)
    
    if 'figax' in kwargs:
        return fig, ax
    else:
        plt.show()
    return fig, gradient_df.T, param_df

def format_gradient_params_LATEX(gradient_params): 
    """ Returns very specific formatting of the gradient parameter dataframe to copy into .tex file """
    seasons = gradient_params.reset_index()['level_1'].apply(lambda x: '& ' + dcts.dict_season()[f'name_{x}'].split(' ')[0])
    param_df = gradient_params.reset_index().drop(columns = ['level_0'])
    param_df['level_1']  = seasons
    param_df.set_index('level_1', inplace=True)
    print(param_df.to_latex(header=False, float_format="%.2f"))

def get_r_squared(v_data, x_data, function, popt): 
    """ Find a value for R^2 goodness of fit. """
    ss_res = np.sum((v_data - function(x_data, *popt)) ** 2) # residual sum of squares
    ss_tot = np.sum((v_data - np.mean(v_data)) ** 2) # total sum of squares
    r2 = 1 - (ss_res / ss_tot)
    return r2


# --- FUNCTIONS ---
def tanh_func(x, a, b, c, d): 
    """ Hyperbolic tangent function. """
    return a + b * np.tanh(c * x + d)

def ddx_tanh(x, b, c, d):
    """ First derivative of the hyperbolic tangent function. """
    return (b*c)/(np.cosh(d + c*x)**2)

def d2dx2_tanh(x, b, c, d):
    """ Second derivative of the hyperbolic tangent function. """
    return -(2*b*c**2 * np.sinh(d + c*x))/(np.cosh(d + c*x)**3)

def d2dx2_tanh_at_zero(b, c, d):
    """ Second derivative of the hyperbolid tanget function at x = 0 """
    nom = 2*b*c**2 * np.sinh(d)
    denom = np.cosh(d)**3
    return - nom/denom


# --- CURVATURE ---
def get_curvature(bin1d, function, plot=False, lim=2.5, p0=None):
    """ Find the curvature at the tropopause from fitted data ±2km around the TP. """
    df = pd.DataFrame({'x' : bin1d.x, 'v' : bin1d.v})
    if plot: 
        plt.scatter(df.v, df.x, color='grey')
    df = df.sort_values('x')
    df = df[abs(df.x) < lim]
    
    popt, pcov = curve_fit(function, df.x,  df.v, p0=p0)
    
    x_smooth = np.linspace(-lim, lim, 50)

    # Get curvature
    if str(function) in [str(fctns.quadratic), str(fctns.poly)]:
        curvature = 2*popt[2]
    elif str(function) == str(tanh_func): 
        gradient = ddx_tanh(0, popt[1], popt[2], popt[3])
        curvature = d2dx2_tanh(0, popt[1], popt[2], popt[3])
    else:
        raise Warning(f'Cannot evaluate curvature for {function}')

    if plot: 
        plt.scatter(df.v, df.x)
        plt.plot(function(x_smooth, *popt), 
                 x_smooth, color='k', lw=3,
                 label = f'{lim} : {curvature:.2f}')
        plt.legend(loc = 'lower right')
        
    r2_df = df[abs(df.x) < 1.5]
    r2 = get_r_squared(r2_df.v, r2_df.x, function, popt)

    return curvature, popt, r2

def get_curvature_binned(bin1d, function, plot=False, lim=2.5, p0=None, offset = 0, normalise=False):
    """ Find the curvature at the tropopause from fitted data ±2km around the TP. 
    Args: 
        bin1d ()
    
    """
    df = pd.DataFrame({'x' : bin1d.xintm, 'v' : bin1d.vmean})
    if plot: 
        plt.figure(dpi=150)
        plt.scatter(df.v, df.x, color='grey')
    df = df.sort_values('x')
    df = df[abs(df.x) < lim]
    df.dropna(inplace = True)
    popt, pcov = curve_fit(function, df.x,  df.v, p0=p0, nan_policy='raise')

    x_smooth = np.linspace(-lim, lim, 50)

    # Get curvature
    if str(function) in [str(fctns.quadratic), str(fctns.poly)]:
        gradient = popt[1]
        curvature = 2*popt[2]
        # curve_err = 2*np.sqrt(np.diag(pcov))[2]
    elif str(function) == str(tanh_func): 
        gradient = ddx_tanh(offset, popt[1], popt[2], popt[3])
        curvature = d2dx2_tanh(offset, popt[1], popt[2], popt[3])
    else: 
        raise Warning(f'Cannot evaluate curvature for {function}')

    if normalise: 
        gradient = gradient/function(0, *popt)
        curvature = curvature/function(0, *popt)

    if plot: 
        plt.scatter(df.v, df.x)
        
        # New fit with curvature and gradient
        plt.plot(function(x_smooth, *popt), 
                x_smooth, color='k', lw=3,
                label = f'Curvature @TP : {curvature:.2f}')
        
        plt.plot([gradient * (offset-0.5) + function(0, *popt), 
                    gradient*(offset+0.5) + function(0, *popt)],
                    [offset-0.5, 
                    offset+0.5],
                    label = f'Gradient @TP: {gradient:.2f}', 
                    color = 'xkcd:green', lw=1)

        plt.legend(
            title = f'Range ±{lim} km',
            loc = 'lower right')
        
    r2_df = df[abs(df.x) < lim*0.75]
    r2 = get_r_squared(r2_df.v, r2_df.x, function, popt)    

    return curvature, popt, r2, gradient, df

def seasonal_binned_curvature(GlobalObject, subs, tps, function=tanh_func, **kwargs): 
    """ Seaonsal binned curvature based on best fit to 'function' """
    fit_popt = {tp.col_name:{} for tp in tps}
    fit_curvature = {tp.col_name:{} for tp in tps}
    fit_gradient = {tp.col_name:{} for tp in tps}
    fit_r2 = {tp.col_name:{} for tp in tps}

    xbsize=kwargs.get('xbsize', 0.1)
    for tp in tps: 
        bin_dict = bin_tools.seasonal_binning(
            GlobalObject.df, subs, tp, xbsize=xbsize)
        if kwargs.get('plot_old'): 
            bin_dict_old = bin_tools.seasonal_binning(
                GlobalObject.df, subs, tp, xbsize=0.5)
        for s in bin_dict.keys():
            x_smooth = np.linspace(-kwargs.get('lim', 2), kwargs.get('lim', 2), 50)
            offset=kwargs.get('offset', 0)
            lim = kwargs.get('lim', 2)

            curvature, popt, r2, gradient, df = get_curvature_binned(
                bin_dict[s], 
                function = function, 
                lim=lim, 
                offset=offset,
                p0=kwargs.get('p0', [100, 100, 1, -1]),
                normalise = kwargs.get("normalise", False)
                )

            if kwargs.get('plot', False): 
                fig, ax = plt.subplots(dpi = kwargs.get('dpi', 150))
                
                curvature_str = f"f\'\'({offset}): {curvature:.1f} {subs.unit}/{tp.unit}$^2$"
                gradient_str = f"f\'({offset}): {gradient:.1f} {subs.unit}/{tp.unit}"
                r2_str = f"R$^2$: {r2:.2f}"
                func_str = "Hyperbolic tangent"

                plt.scatter(df.v, df.x, label = f"Bins with $\Delta$x = {xbsize}",
                            alpha = 1 if not kwargs.get('plot_old') else 0)
                
                xlims = ax.get_xlim(); ylims = ax.get_ylim()
                
                plt.scatter(bin_dict[s].v, 
                            bin_dict[s].x,
                            color = "#dbdbdb",
                            zorder = 0)
                plt.ylim(*ylims)
                plt.xlim(*xlims)
                
                
                # Best fit in ±lim range
                plt.plot(function(x_smooth, *popt), 
                        x_smooth, color='k', lw=2,
                        label = f'tanh(x) ({r2_str})')

                # Gradient                
                plt.plot([gradient * (offset-xbsize*5) + function(0, *popt), 
                        gradient*(offset+xbsize*5) + function(0, *popt)],
                        [offset-xbsize*5, 
                        offset+xbsize*5],
                        label = f"Gradient, f\'({offset})", 
                        color = "#3ABD28", lw=2, ls = (0, (3, 1, 1, 1)),
                        alpha = 1 if not kwargs.get('plot_old') else 0)

                # Old averaging calculation method
                if kwargs.get('plot_old'): 
                    plt.axline( # gradient as long thing line (for better comparison)
                        (gradient * (offset) + function(0, *popt), offset),
                        (gradient*(offset+xbsize) + function(0, *popt), offset+xbsize),
                        color = "#3ABD28", lw=1)
                    
                    def get_old_gradient(bin_dict, s):
                        """ Calculate the gradient from the bins above and below the tropopause+offset. """
                        i_pos = next(x for x, val in enumerate(bin_dict[s].xintm) if val > 0)              
                        i_pos_gradient = i_pos+kwargs.get('offset', 0)
                        pos_val = bin_dict[s].vmean[i_pos_gradient]
                        neg_val = bin_dict[s].vmean[i_pos_gradient-1]
                        gradient = (pos_val - neg_val) / (bin_dict[s].xbsize)
                        return pos_val, neg_val, gradient, i_pos_gradient
                    pos_val, neg_val, old_gradient, i_pos_old = get_old_gradient(bin_dict_old, s)
                    
                    plt.axline((neg_val, bin_dict_old[s].xintm[i_pos_old-1]), 
                               (pos_val, bin_dict_old[s].xintm[i_pos_old]),
                               label = f'Old G: {old_gradient:.2f}', 
                               color = 'r', lw = 1)
                    
                    plt.scatter(
                        bin_dict_old[s].vmean,
                        bin_dict_old[s].xintm,
                        color = 'r')

                plt.xlabel(subs.label())
                plt.ylabel(tp.label(coord_only=True) + f' [{tp.unit}]')
                plt.legend(
                    title = f"{func_str}\n{gradient_str}\n{curvature_str}",
                    loc = 'lower right')
            
            fit_popt[tp.col_name][s] = popt
            fit_curvature[tp.col_name][s] = curvature
            fit_gradient[tp.col_name][s] = gradient
            fit_r2[tp.col_name][s] = r2
            # except: 
            #     print('ERROR for '+ tp.col_name +f' in {s}'+ dcts.dict_season()[f'name_{s}'])
            if kwargs.get('plot'):
                plt.title(tp.label(filter_label=True, no_vc=True, no_model=True) +' in '+ dcts.dict_season()[f'name_{s}'])
                plt.grid(True, lw=0.5, ls='dotted', color='xkcd:grey')
                cfig.add_zero_line(ax)
                plt.show()
    return fit_popt, fit_curvature, fit_r2, fit_gradient

def seasonal_fits_r2(GlobalObject, subs, tps, function=tanh_func, **kwargs): 
    """ Show only the best fits and R2-labels for tp-rel coords on subs. """
    lim = kwargs.get('lim', 2)
    
    fit_popt, _, fit_r2, _ = seasonal_binned_curvature(
        GlobalObject, subs, tps, plot=False, function=function, **kwargs)
    
    fig, axs = cfig.tp_comp_plot(tps)
    x_smooth = np.linspace(-lim,lim,50)

    axs.flat[-1].legend(handles = cfig.season_legend_handles(), loc = 'center')
    for ax, (tp_col, popt_dict) in zip(axs.flat, fit_popt.items()):
        [tp] = GlobalObject.get_coords(col_name = tp_col)
        for s in popt_dict.keys():
            ax.plot(tanh_func(x_smooth, *popt_dict[s]), x_smooth,
                    color = dcts.dict_season()[f'color_{s}'],
                    label = f'R$^2$: {fit_r2[tp_col][s]:.2f}')
            ax.legend(fontsize=8, loc='lower right')

        ax.set_title(tp.label(filter_label=True, no_vc=True))
        ax.grid(True, lw=0.5, ls='dotted')
        ax.set_ylabel(tp.label(coord_only=True))

    ax.set_xlabel(subs.label(no_ID=True))
    fig.tight_layout()
    for ax in axs.flat:
        cfig.add_zero_line(ax)
    return fig, axs


# BASIC AF STUFF FROM AGES AGO
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