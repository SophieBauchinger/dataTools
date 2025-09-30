# -*- coding: utf-8 -*-
""" Cross-tropopause gradient statistics and plotting

@Author: Sophie Bauchinger, IAU
@Date: Mon Mar 10 15:30:00 2025
"""

import cmasher as cmr
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.gridspec as gridspec
import matplotlib.patheffects as mpe
from matplotlib.ticker import FixedLocator
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.optimize import curve_fit

import toolpac.outliers.ol_fit_functions as fctns

import dataTools.dictionaries as dcts
from dataTools import tools
import dataTools.data.BinnedData as bin_tools
import dataTools.plot.create_figure as cfig

# Heatmap, gradient and other useful things
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

    cbar_kws={'label': kwargs.get('cbar_label', ('Gradient [fraction/km]')), 
                  'pad' : 0.02, # space between plot and cbar
                  }
    cbar_kws.update(kwargs.get("cbar_kws", {}))
                #   'labelpad' : 15}, 

    sns.heatmap(
        df.T, annot=True, cmap=kwargs.get('cmap', 'YlGnBu'), #cmr.get_sub_cmap('PRGn', 0.1, 0.9), # 'PRGn',# 'YlGnBu', 
        norm = Normalize(*kwargs.get('vlims')) if 'vlims' in kwargs else None,
        cbar_kws=cbar_kws,
        yticklabels=[dcts.get_coord(c).label(filter_label=True, no_vc=True, no_model=True) for c in df.columns], 
        xticklabels=['Spring', 'Summer', 'Autumn', 'Winter', 'Average'],
        fmt='.2f', ax = ax,
        linewidths=1, linecolor='white',
        )
    print(cbar_kws)

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
    # print(param_df.to_latex(header=False, float_format="%.2f"))
    print(param_df.to_latex(header=False))

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

def get_curvature_profile(GlobalObject, subs, tps, 
                          popt_dict=None, lim=2, normalise=True, **kwargs): 
    """ Given the parameters for tanh fit, plot curvature profile within lims. """
    # Find fit parameters per season and per TP def, calc curvature of region ±2km
    if not popt_dict:
        popt_dict, _, _, _ = seasonal_binned_curvature(
            GlobalObject, subs, tps, plot=False, xbsize=0.1,
            p0 = [100, 100, 1, 1], lim=lim, normalise=normalise)
    
    x_vals = kwargs.get('x_vals', np.linspace(-2, 2, 100))
    offset_curv = {s : dict() for s in [1,2,3,4]}
    curv_zero = {s : dict() for s in [1,2,3,4]}
    param_dict = dict()
    
    for tp_col in popt_dict.keys():
        season_dict = popt_dict[tp_col]
        param_dict[tp_col] = dict()
        
        for s in season_dict.keys(): 
            popt = season_dict[s]
            curvatures = d2dx2_tanh(x_vals, popt[1], popt[2], popt[3])
            curv_at_zero = d2dx2_tanh(0, popt[1], popt[2], popt[3])
            
            offset_curv[s][tp_col] = curvatures
            if normalise: 
                offset_curv[s][tp_col] = curvatures / tanh_func(0, *popt)
                curv_zero[s][tp_col] = curv_at_zero / tanh_func(0, *popt)
    
            param_dict[tp_col][s] = pd.Series(dict(
                val0 = tanh_func(0, *popt),
                curv0 = curv_at_zero,
                norm_curv0 = curv_at_zero / tanh_func(0, *popt),
                norm_max_c = curvatures.max() / tanh_func(0, *popt),
                max_c_height = pd.Series(curvatures, index = x_vals).idxmax()*1000,
                # max_c = curvatures.max(),
            ))

    param_df = pd.DataFrame.from_dict(
        {(i,j): param_dict[i][j] 
        for i in param_dict.keys() 
        for j in param_dict[i].keys()},
        orient='index')
    for sig_fig, c in zip(kwargs.get('sig_figs', [1,1,2,2,0]), param_df.columns):
        param_df[c] = param_df[c].map(f"{{:.{sig_fig}f}}".format)

    if kwargs.get('plot', True):
        tps = [dcts.get_coord(col_name=tp_col) for tp_col in popt_dict.keys()]
        fig, axs = plt.subplots(2,2, sharex=True, sharey=True, dpi=300, 
                            figsize=(7,6))
        for s, ax in zip(offset_curv.keys(), axs.flat):
            for tp in offset_curv[s].keys(): 
                ax.plot(offset_curv[s][tp], x_vals, color = dcts.get_coord(col_name=tp).get_color(), 
                        ls='-' if not tp=="N2O_residual" else '--', 
                        path_effects = [cfig.outline()])
                
            ax.set_title(dcts.dict_season()[f"name_{s}"])

        for ax in axs.flat: 
            ax.grid(True, ls='dotted', c='grey')
            cfig.add_zero_line(ax)
        for ax in axs[:,0]:
            ax.set_ylabel(tps[-1].label(coord_only=True)+f" [{tps[-1].unit}]")
        for ax in axs[1,:]:
            ax.set_xlabel("Normalised curvature [km$^{-2}$]")

        fig.tight_layout()
        fig.subplots_adjust(top=0.85)
        fig.legend(handles = cfig.tp_legend_handles(tps), ncols=3, loc='upper center')

    return offset_curv, curv_zero, param_df

def seasonal_binned_curvature(GlobalObject, subs, tps, function=tanh_func, **kwargs): 
    """ Seaonsal binned curvature based on best fit to 'function' """
    fit_popt = {tp.col_name:{} for tp in tps}
    fit_curvature = {tp.col_name:{} for tp in tps}
    fit_gradient = {tp.col_name:{} for tp in tps}
    fit_r2 = {tp.col_name:{} for tp in tps}

    xbsize=kwargs.get('xbsize', 0.1)
    offset=kwargs.get('offset', 0)
    lim = kwargs.get('lim', 2)
    x_smooth = np.linspace(-lim, lim, 50)

    for tp in tps: 
        bin_dict = bin_tools.seasonal_binning(
            GlobalObject.df, subs, tp, xbsize=xbsize)
        if kwargs.get('plot_old'): 
            bin_dict_old = bin_tools.seasonal_binning(
                GlobalObject.df, subs, tp, xbsize=0.5)
        for s in bin_dict.keys():
            curvature, popt, r2, gradient, _ = get_curvature_binned(
                bin_dict[s], 
                function = function, 
                lim=lim, 
                offset=offset,
                p0=kwargs.get('p0', [100, 100, 1, -1]),
                normalise = kwargs.get("normalise", False)
                )

            df = pd.DataFrame({'x' : bin_dict[s].x, 'v' : bin_dict[s].v})
            df = df.sort_values('x')
            df = df[abs(df.x) < lim]

            fit_popt[tp.col_name][s] = popt
            fit_curvature[tp.col_name][s] = curvature
            fit_gradient[tp.col_name][s] = gradient
            fit_r2[tp.col_name][s] = r2

            if kwargs.get('plot', False): 
                fig, ax = plot_fits(subs, tp, function, popt, 
                               bin_dict[s], x_smooth, offset, curvature, gradient, r2,
                               bin_dict_old[s] if kwargs.get('plot_old') else None, df, **kwargs)
                plt.ylim(-lim*1.05, lim*1.05)
                plt.title(tp.label(filter_label=True, no_vc=True, no_model=True) +' in '+ dcts.dict_season()[f'name_{s}'])
                plt.grid(True, lw=0.5, ls='dotted', color='xkcd:grey')
                cfig.add_zero_line(ax)
                plt.show()

    return fit_popt, fit_curvature, fit_r2, fit_gradient

def plot_fits(
    subs, tp, function, popt, 
    bin_s, x_smooth, offset, curvature, gradient, r2, 
    bin_old, df, **kwargs):
    """ Plot the fitted curve and describe fit in legend """
    fig, ax = plt.subplots(dpi = kwargs.get('dpi', 150))
                
    curvature_str = f"f\'\'({offset}): {curvature:.1f} {subs.unit}/{tp.unit}$^2$"
    gradient_str = f"f\'({offset}): {gradient:.1f} {subs.unit}/{tp.unit}"
    r2_str = f"R$^2$: {r2:.2f}"
    func_str = "Hyperbolic tangent"

    plt.scatter(bin_s.vmean, bin_s.xintm, label = f"Bins with $\Delta$x = {bin_s.xbsize}",
                alpha = 1 if not kwargs.get('plot_old') else 0, color = 'tab:blue')
    xlims = ax.get_xlim(); ylims = ax.get_ylim()
    plt.scatter(df.v, df.x,
                color = "#dbdbdb", zorder = 0)

    plt.xlim(*xlims)
    plt.ylim(*ylims)

    # Best fit in ±lim range
    plt.plot(function(x_smooth, *popt), 
             x_smooth, color='k', lw=2,
             label = f'tanh(x) ({r2_str})')

    # Gradient
    plt.plot([gradient * (offset-bin_s.xbsize*10) + function(0, *popt), 
             gradient*(offset+bin_s.xbsize*10) + function(0, *popt)],
             [offset-bin_s.xbsize*10, 
             offset+bin_s.xbsize*10],
             label = f"Gradient, f\'({offset})", 
             color = "#3ABD28", lw=2, ls = (0, (3, 1, 1, 1)),
             alpha = 1 if not kwargs.get('plot_old') else 0)


    # Old averaging calculation method
    if kwargs.get('plot_old', False): 
        show_old_gradient(bin_old, **kwargs)
        
        plt.axline((function(offset, *popt), offset), slope = 1/gradient,
                   color = "#3ABD28", lw=1)

    plt.xlabel(subs.label())
    plt.ylabel(tp.label(coord_only=True) + f' [{tp.unit}]')
    plt.legend(
        title = f"{func_str}\n{gradient_str}\n{curvature_str}",
        loc = 'lower right')
        
    return fig, ax

def show_old_gradient(bin_old, **kwargs):
    """ Calculate and plot the gradient from the bins above and below the tropopause+offset. """
    i_pos = next(x for x, val in enumerate(bin_old.xintm) if val > 0)              
    i_pos_gradient = i_pos+kwargs.get('offset', 0)
    pos_val = bin_old.vmean[i_pos_gradient]
    neg_val = bin_old.vmean[i_pos_gradient-1]
    gradient = (pos_val - neg_val) / (bin_old.xbsize)
    plt.axline(
        (neg_val, bin_old.xintm[i_pos-1]),
        slope=1/gradient, 
        # (pos_val, bin_old.xintm[i_pos]),
        label = f'Old G: {gradient:.2f}', 
        color = 'tab:orange', lw = 1)
    plt.scatter(
        bin_old.vmean,
        bin_old.xintm,
        color = 'tab:orange')

def seasonal_fits_r2(GlobalObject, subs, tps, function=tanh_func, **kwargs): 
    """ Show only the best fits and R2-labels for tp-rel coords on subs. """
    lim = kwargs.get('lim', 2)

    fit_popt, _, fit_r2, _ = seasonal_binned_curvature(
        GlobalObject, subs, tps, plot=False, function=function, **kwargs)

    # if any(tp.crit=='n2o' for tp in tps): 
    #     n2o_popt_dict, _, n2o_r2_dict, _ = seasonal_binned_curvature(
    #         GlobalObject, subs, GlobalObject.get_coords(col_name="N2O_residual"), 
    #         plot=False, function=tanh_func, lim = 5, xbsize=0.5)
    #     n2o_x_smooth = np.linspace(-5,5,50)
    #     n2o_popt = n2o_popt_dict["N2O_residual"]
    #     n2o_r2 = n2o_r2_dict["N2O_residual"]
    
    fig, axs = cfig.tp_comp_plot(tps, figsize=(5.6,6.8))
    x_smooth = np.linspace(-lim,lim,50)

    for ax, (tp_col, popt_dict) in zip(axs.flat, fit_popt.items()):
        [tp] = GlobalObject.get_coords(col_name = tp_col)
        for s in popt_dict.keys():
            ax.plot(tanh_func(x_smooth, *popt_dict[s]), x_smooth,
                    color = dcts.dict_season()[f'color_{s}'],
                    label = f'R$^2$: {fit_r2[tp_col][s]:.2f}')
            ax.legend(fontsize=8, loc='lower right')

        ax.set_title(tp.label(filter_label=True, no_vc=True))
        # ax.grid(True, lw=0.5, ls='dotted')
        ax.set_ylabel(tp.label(coord_only=True))

    # fig.legend(handles = cfig.season_legend_handles(), loc = 'center')
    for ax in axs[-1,:]:
        ax.set_xlabel(subs.label(no_ID=True))
    fig.tight_layout()

    # for ax in axs.flat:
    #     cfig.add_zero_line(ax)
    return fig, axs

def get_max_curvature(offset_curv, offset_vals=None): 
    """ Get value and location of the maximum curvature in the fitted vertical profile. 
    Args: 
        offset_curv_dict: output of vertical_curvature_profile
        
    Returns offset in km, diaplcement height in metres 
    """
    if not offset_vals: 
        offset_vals = np.linspace(-2, 2, 100)

    max_c_dict = {s: {k : v.max() 
                    for k,v in offset_curv[s].items()} 
                for s in offset_curv.keys()}

    max_c_height = {s : {tp_col : pd.Series(
        offset_curv[s][tp_col], index = offset_vals).idxmax()*1000 
                         for tp_col in offset_curv[s]}
                    for s in offset_curv.keys()}
    
    gradient_heatmap(
        pd.DataFrame(max_c_height).T, 
        cbar_label = 'Normalised curvature [km$^{-2}$]', 
        cmap='RdBu_r', vlims = (-700, 700));

# Very specific: Two-panel plot for paper 
def plot_curvature_profile_heatmap(GlobalObject, subs, tps, linestyles, lim=2): 
    """ For the paper: Two-panel nicely formatted and adjusted showing curvature over Δz and at TP. """
    _, fit_curvature, _, _ = seasonal_binned_curvature(
        GlobalObject, subs, tps, plot=False, lim=lim, normalise=True)
    
    offset_curv, _, _ = get_curvature_profile(
        GlobalObject, subs, tps, plot=False, lim=lim, normalise=True)
    
    x_vals = np.linspace(-lim, lim, 100)
    
    # Plot the results
    fig = plt.figure(figsize=(7,9), dpi=300)
    # Outer grid: 2 rows — top = curvature, bottom = heatmap
    outer = gridspec.GridSpec(2, 1, height_ratios=[2.5, 1.2], hspace=0.325)
    # Top 2x2 curvature grid
    top_gs = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=outer[0], hspace = 0.4)

    # Create subplots for curvature profiles
    axs = [fig.add_subplot(top_gs[0, 0]), fig.add_subplot(top_gs[0, 1]),
        fig.add_subplot(top_gs[1, 0]), fig.add_subplot(top_gs[1, 1])]

    for s, ax, id in zip(offset_curv.keys(), axs, 'abcd'):
        for tp in offset_curv[s].keys(): 
            ax.plot(offset_curv[s][tp], x_vals, 
                    color = dcts.get_coord(col_name=tp).get_color(), 
                    ls = linestyles[tp],
                    # ls='-' if not tp=="N2O_residual" else '--', 
                    path_effects = [cfig.outline()])
            
        ax.set_title(f"({id}) "+dcts.dict_season()[f"name_{s}"])
        ax.grid(True, ls='dotted', c='grey')
        cfig.add_zero_line(ax)

    xmax = max([item for ax in axs for item in ax.get_xlim()])
    for ax in axs: 
        ax.set_xlim(-xmax, xmax)


    for ax in [axs[0], axs[2]]:
        ax.set_ylabel(tps[-1].label(coord_only=True)+f" [{tps[-1].unit}]")
    for ax in [axs[2], axs[3]]:
        ax.set_xlabel("Normalised curvature [km$^{-2}$]")

    bottom_gs = gridspec.GridSpecFromSubplotSpec(1, 2, width_ratios=[0.1, 1], wspace=0.05, subplot_spec=outer[1])
    pad_ax = fig.add_subplot(bottom_gs[0])
    pad_ax.text(0.2, 1.1, '(e)', transform=pad_ax.transAxes,
                fontsize=12, va='top', ha='right')
    pad_ax.axis('off') # spacer

    heatmap_ax = fig.add_subplot(bottom_gs[1])
    gradient_heatmap(
        pd.DataFrame(fit_curvature), 
        cbar_label = 'Normalised curvature [km$^{-2}$]', 
        cmap='BuGn', figax = (fig, heatmap_ax),
        cbar_kws={'pad': 0.02, 'shrink': 0.7, 'location': 'right'})


    fig.legend(handles = cfig.tp_legend_handles(tps, linestyle_dict = linestyles, lw=2), 
            ncols=3, loc='upper center')
    fig.subplots_adjust(top=0.9)
    
    return fig, axs

    # fig.savefig(path+"curvature_everything.pdf", dpi=300, bbox_inches = "tight")

# Very specific: seasonal_fits_r2 with highlighted n2o axis 
def seasonal_fits_with_n2o(GlobalObject, subs, tps): 
    """ Show tanh fit and R^2 for tps including N2O baseline coordinate. """
    [n2o_coord] = GlobalObject.get_coords(col_name = 'N2O_residual')
    
    fig, axs = seasonal_fits_r2(GlobalObject, subs, tps)
    for ax in axs.flat: 
        ax.set_xticks([0, 100, 200, 300, 400, 500])
        ax.yaxis.set_major_locator(FixedLocator([-2, -1, 0, 1, 2]))
        # ax.set_yticks([-2, -1, 0, 1, 2])
        ax.grid(True, lw=0.5, ls='dotted', which='major')

    # Hide original axis underneath N2O panel (cannot just turn off because we want the grid)
    axs[0,1].clear()
    for spine in ['left', 'right', 'top']:
        axs[0,1].spines[spine].set_visible(False)    # Hide unwanted spines
    axs[0,1].grid(True, lw=0.5, ls='dotted')
    n2o_ax = axs[0,1].twinx()
    new_axs = list([axs[0,0]]) + list([n2o_ax]) + list(axs.flat[2:])

    # Recalculate N2O stuff with new limits and everything
    n2o_popt, _, n2o_r2, _ = seasonal_binned_curvature(
            GlobalObject, subs, GlobalObject.get_coords(col_name="N2O_residual"), 
            plot=False, function=tanh_func, lim = 5, xbsize=0.5)
    x_smooth = np.linspace(-5,5,50)

    for s in n2o_popt["N2O_residual"].keys():
        n2o_ax.plot(tanh_func(x_smooth, *n2o_popt["N2O_residual"][s]), x_smooth,
                color = dcts.dict_season()[f'color_{s}'],
                label = f'R$^2$: {n2o_r2["N2O_residual"][s]:.2f}')
        n2o_ax.legend(fontsize=8, loc='lower right')

    n2o_ax.set_title(n2o_coord.label(filter_label=True, no_vc=True))
    n2o_ax.set_ylabel(n2o_coord.label(coord_only=True) + f" [{n2o_coord.unit}]")
    n2o_ax.set_xlabel(subs.label(no_ID=True))
    n2o_ax.invert_yaxis()

    # Adjust tick params etc. 
    for ax in list(axs[:,1]) + [n2o_ax]:
        ax.tick_params(
            labelleft=False, left=False, 
            labelright = True, right = True,
            top=False, labeltop=False, 
            bottom=True)
        ax.yaxis.set_label_position('right')

    axs[0,1].tick_params(right=False, labelright=False)

    cfig.highlight_axis(n2o_ax)

    fig.subplots_adjust(top=0.87)
    fig.legend(handles = cfig.season_legend_handles(), ncols=2, loc='upper center')

    for ax, id in zip(new_axs, 'abcdef'):
        ax.set_title(f"({id}) "+ ax.get_title())
        cfig.add_zero_line(ax)
    cfig.add_zero_line(axs[0,1])
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