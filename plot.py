# -*- coding: utf-8 -*-
"""
@Author: Sophie Bauchimger, IAU
@Date: Thu May 11 13:22:38 2023

Defines different plotting possibilities for objects of the type GlobalData 
or LocalData as defined in data_classes

"""

import datetime as dt
import geopandas
import numpy as np
from calendar import monthrange

import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable as sm
from matplotlib.colors import ListedColormap as lcm
from matplotlib.patches import Patch

from aux_fctns import monthly_mean
from dictionaries import get_col_name, get_vlims, get_default_unit

# supress a gui backend userwarning, not really advisible
import warnings; warnings.filterwarnings("ignore", category=UserWarning, module='matplotlib')

#%% GlobalData
def plot_scatter_global(global_data, substance=None, single_yr=None):
    if global_data.source=='Caribic':
        if substance is None: substance = global_data.substance
        if single_yr is not None: 
            df = global_data.select_year(single_yr)
            df_mm = monthly_mean(df).notna()
        else: df = global_data.df; df_mm = global_data.df_monthly_mean

        # Plot mixing ratio msmts and monthly mean
        fig, ax = plt.subplots(dpi=250)
        plt.title(f'{global_data.source} {global_data.substance_short.upper()} measurements')
        ymin = np.nanmin(df[substance])
        ymax = np.nanmax(df[substance])

        cmap = plt.cm.viridis_r
        extend = 'neither'
        if global_data.v_limits: vmin, vmax = global_data.v_limits# ; extend = 'both'
        else: vmin = ymin; vmax = ymax
        norm = Normalize(vmin, vmax)

        plt.scatter(df.index, df[substance],
                    label=f'{global_data.substance_short.upper()} {global_data.years}', marker='x', zorder=1,
                    c = df[substance],
                    cmap = cmap, norm = norm)
        for i, mean in enumerate(df_mm[substance]):
            y,m = df_mm.index[i].year, df_mm.index[i].month
            xmin, xmax = dt.datetime(y, m, 1), dt.datetime(y, m, monthrange(y, m)[1])
            ax.hlines(mean, xmin, xmax, color='black',
                      linestyle='dashed', zorder=2)
        plt.colorbar(sm(norm=norm, cmap=cmap), aspect=50, ax = ax, extend=extend)
        plt.ylabel(f'{global_data.substance_short.upper()} mixing ratio [ppt]')
        plt.ylim(ymin-0.15, ymax+0.15)
        fig.autofmt_xdate()
        if hasattr(global_data, 'pfxs'): ax.text(0,0, f'{global_data.pfxs}', verticalalignment='top')
        plt.show() # for some reason there's a matplotlib user warning here: converting a masked element to nan. xys = np.asarray(xys)

    elif global_data.source=='Mozart':
        global_data.plot_1d(substance, single_yr)

def plot_global_binned_1d(global_data, substance=None, single_yr=None, plot_mean=False, single_graph=False):
    """
    Plots 1D averaged values over latitude / longitude including colormap 
    Parameters:
        substance (str): if None, plots default substance for the object
        single_yr (int): if specified, plots only data for that year [default=None]
        plot_mean (bool): choose whether to plot the overall average over all years
        single_graph (bool): choose whether to plot all years on one graph
    """
    if substance is None: substance = global_data.substance
    if single_yr is not None: years = [int(single_yr)]
    else: years = global_data.years

    out_x_list, out_y_list = global_data.binned_1d(substance, single_yr)

    if not single_graph:
        # Plot mixing ratios averages over lats / lons for each year separately
        for out_x, out_y, year in zip(out_x_list, out_y_list, years):
            fig, ax = plt.subplots(dpi=300, ncols=2, sharey=True, figsize=(8,3.5))
            fig.suptitle('{} {} modeled SF$_6$ concentration. Gridsize={}'.format(
                global_data.source, year, global_data.grid_size))

            cmap = plt.cm.viridis_r
            if global_data.v_limits: vmin, vmax = global_data.v_limits
            else:
                vmin = min([np.nanmin(out_x.vmean), np.nanmin(out_y.vmean)])
                vmax = max([np.nanmin(out_x.vmean), np.nanmin(out_y.vmean)])
            norm = Normalize(vmin, vmax) # allows mapping colormap onto available values

            ax[0].plot(out_x.xintm, out_x.vmean, zorder=1, color='black', lw = 0.5)
            ax[0].scatter(out_x.xintm, out_x.vmean, # plot across latitude
                          c = out_x.vmean, cmap = cmap, norm = norm, zorder=2)
            ax[0].set_xlabel('Latitude [deg]'); plt.xlim(out_x.xbmin, out_x.xbmax)
            ax[0].set_ylabel('Mean SF$_6$ mixing ratio [ppt]')

            ax[1].plot(out_y.xintm, out_y.vmean, zorder=1, color='black', lw = 0.5)
            ax[1].scatter(out_y.xintm, out_y.vmean, # plot across longitude
                          c = out_y.vmean, cmap = cmap, norm = norm, zorder=2)
            ax[1].set_xlabel('Longitude [deg]'); plt.xlim(out_y.xbmin, out_y.xbmax)
            ax[1].set_ylabel('Mean SF$_6$ mixing ratio [ppt]')

            fig.colorbar(sm(norm=norm, cmap=cmap), aspect=50, ax = ax[1])
            plt.show()

    if single_graph:
        # Plot averaged mixing ratios for all years on one graph
        fig, ax = plt.subplots(dpi=300, ncols=2, sharey=True, figsize=(8,3.5))
        fig.suptitle(f'{global_data.source} {global_data.years[0]} - {global_data.years[-1]} modeled {substance} mixing ratio. Gridsize={global_data.grid_size}')

        cmap = cm.get_cmap('plasma_r')
        vmin, vmax = global_data.years[0], global_data.years[-1]
        norm = Normalize(vmin, vmax)

        for out_x, out_y, year in zip(out_x_list, out_y_list, global_data.years): # add each year to plot
            ax[0].plot(out_x.xintm, out_x.vmean, label=year)#, c = cmap(norm(year)))
            ax[0].set_xlabel('Latitude [deg]'); plt.xlim(out_x.xbmin, out_x.xbmax)
            ax[0].set_ylabel(f'Mean {substance} mixing ratio [ppt]')

            ax[1].plot(out_y.xintm, out_y.vmean, label=year)# , c = cmap(norm(year)))
            ax[1].set_xlabel('Longitude [deg]'); plt.xlim(out_y.xbmin, out_y.xbmax)
            ax[1].set_ylabel(f'Mean {substance} mixing ratio [ppt]')

        if plot_mean: # add average over available years to plot
            total_x_vmean = np.mean([i.vmean for i in out_x_list], axis=0)
            total_y_vmean = np.mean([i.vmean for i in out_y_list], axis=0)
            ax[0].plot(out_x.xintm, total_x_vmean, label='Mean', c = 'k', ls ='dashed')
            ax[1].plot(out_y.xintm, total_y_vmean, label='Mean', c = 'k', ls ='dashed')

        handles, labels = ax[0].get_legend_handles_labels()
        plt.legend(reversed(handles), reversed(labels), # reversed so that legend aligns with graph
                   bbox_to_anchor=(1,1), loc='upper left')
        plt.show()
    return

def plot_global_binned_2d(global_data, substance=None, single_yr=None):
    """
    Create a 2D plot of binned mixing ratios for each available year.
    Parameters:
        substance (str): if None, plots default substance for the object
        single_yr (int): if specified, plots only data for that year [default=None]
    """
    if substance is None: substance = global_data.substance
    if single_yr is not None: years = [int(single_yr)]
    else: years = global_data.years
    out_list = global_data.binned_2d(substance, single_yr)

    for out, yr in zip(out_list, years):
        plt.figure(dpi=300, figsize=(8,3.5))
        plt.gca().set_aspect('equal')

        cmap = plt.cm.viridis_r # create colormap
        if global_data.v_limits: vmin, vmax = global_data.v_limits # set colormap limits
        else: vmin = np.nanmin(out.vmin); vmax = np.nanmax(out.vmax)
        norm = Normalize(vmin, vmax) # normalise color map to set limits

        world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
        world.boundary.plot(ax=plt.gca(), color='black', linewidth=0.3)

        plt.imshow(out.vmean, cmap = cmap, norm=norm, origin='lower',  # plot values
                   extent=[out.ybmin, out.ybmax, out.xbmin, out.xbmax])
        cbar = plt.colorbar(ax=plt.gca(), pad=0.08, orientation='vertical') # colorbar
        cbar.ax.set_xlabel('Mean SF$_6$ [ppt]')

        plt.title('{} {} SF$_6$ concentration measurements. Gridsize={}'.format(
            global_data.source, yr, global_data.grid_size))
        plt.xlabel('Longitude  [degrees east]'); plt.xlim(-180,180)
        plt.ylabel('Latitude [degrees north]'); plt.ylim(-60,100)
        plt.show()
    return

# Mozart
def plot_1d_LonLat(mzt_obj, lon_values = [10, 60, 120, 180],
                   lat_values = [70, 30, 0, -30, -70],
                   single_yr=None, substance=None):
    """ 
    Plots mixing ratio with fixed lon/lat over lats/lons side-by-side 
    Parameters:
        lon_values (list of ints): longitude values to average over
        lat_values (list of ints): latitude values to average over
        substance (str): if None, plots default substance for the object
        single_yr (int): if specified, plots only data for that year [default=None]
    """
    if substance is None: substance = mzt_obj.substance
    if single_yr is not None: years = [int(single_yr)]
    else: years = mzt_obj.years

    out_x_list, out_y_list = mzt_obj.binned_1d(substance, single_yr)

    for out_x, out_y, year in zip(out_x_list, out_y_list, years):
        fig, (ax1, ax2) = plt.subplots(dpi=250, ncols=2, figsize=(9,5), sharey=True)
        fig.suptitle(f'MOZART {year} SF$_6$ at fixed longitudes / latitudes', size=17)
        mzt_obj.ds.SF6.sel(time = year, longitude=lon_values,
                        method='nearest').plot.line(x = 'latitude', ax=ax1)
        ax1.plot(out_x.xintm, out_x.vmean, c='k', ls='dashed', label='average')

        mzt_obj.ds.SF6.sel(time = year, latitude=lat_values,
                        method="nearest").plot.line(x = 'longitude', ax=ax2)
        ax2.plot(out_y.xintm, out_y.vmean, c='k', ls='dashed', label='average')
        
        ax1.set_title(''); ax2.set_title('')
        ax2.set_ylabel('')
        plt.show()
    
    return

#%% LocalData

def plot_local(local_data_obj, substance=None, greyscale=True, v_limits = (6,9)):
    """ 
    Plot all available data as timeseries 
    Parameters:
        substance (str): specify substance (optional)
        greyscale (bool): toggle plotting in greyscale or viridis colormap
        v_limits (tuple(int, int)): change limits for colormap
    """
    if greyscale: colors = {'day':lcm(['grey']), 'msmts': lcm(['silver'])} # defining monoscale colormap for greyscale plots
    else: colors = {'msmts':plt.cm.viridis_r, 'day': plt.cm.viridis_r} 

    if not substance: substance = local_data_obj.substance
    col_name = get_col_name(substance, local_data_obj.source)
    vmin, vmax = get_vlims(substance)
    norm = Normalize(vmin, vmax)
    dflt_unit = get_default_unit(substance)

    # Plot all available info on one graph
    fig, ax = plt.subplots(figsize = (5,3.5), dpi=250)
    # Measurement data
    plt.scatter(local_data_obj.df.index, local_data_obj.df[col_name], c=local_data_obj.df[col_name], zorder=0,
                    cmap=colors['msmts'], norm=norm, marker='+',
                    label=f'{local_data_obj.source_print} {substance.upper()}')
    
    # Daily mean
    if not local_data_obj.df_Day.empty: # check if there is data in the daily df
        plt.scatter(local_data_obj.df_Day.index, local_data_obj.df_Day[col_name], c = local_data_obj.df_Day[col_name], 
                    cmap=colors['day'], norm=norm, marker='+', zorder=2,
                    label=f'{local_data_obj.source_print} {substance.upper()} (D)')

    # Monthly mean
    if not local_data_obj.df_monthly_mean.empty: # check for data in the monthly df
        for i, mean in enumerate(local_data_obj.df_monthly_mean[col_name]): # plot monthly mean
            y, m = local_data_obj.df_monthly_mean.index[i].year, local_data_obj.df_monthly_mean.index[i].month
            xmin = dt.datetime(y, m, 1)
            xmax = dt.datetime(y, m, monthrange(y, m)[1])
            ax.hlines(mean, xmin, xmax, color='black', linestyle='dashed', zorder=2)
        ax.hlines(mean, xmin, xmax, color='black', ls='dashed', 
                  label=f'{local_data_obj.source_print} {substance.upper()} (M)') # needed for legend, just plots on top

    plt.ylabel(f'{local_data_obj.substance.upper()} mixing ratio [{dflt_unit}]')
    plt.xlim(min(local_data_obj.df.index), max(local_data_obj.df.index))
    plt.xlabel('Time')
    
    if not greyscale: 
        plt.colorbar(sm(norm=norm, cmap=colors['day']), aspect=50, ax=ax, extend='neither')
    
    # Slightly weird code to create a legend showing the range of the colormap)
    handles, labels = ax.get_legend_handles_labels()
    step = 0.2
    pa = [ Patch(fc=colors['msmts'](norm(v))) for v in np.arange(vmin, vmax, step)]
    pb = [ Patch(fc=colors['day'](norm(v))) for v in np.arange(vmin, vmax, step)]
    pc = [ Patch(fc='black') for v in np.arange(vmin, vmax, step)]
    
    h = [] # list of handles
    for a, b, c in zip(pa, pb, pc): # need to do this to have them in the right order
        h.append(a); h.append(b); h.append(c)            
    l = [''] * (len(h) - len(labels)) + labels # needed to have multiple color patches for one proper label 
    ax.legend(handles=h, labels=l, ncol=len(h)/3, handletextpad=1/(len(h)/2)+0.2, handlelength=0.15, columnspacing=-0.3)

    fig.autofmt_xdate()
    plt.show()

#%% 
if __name__=='__main__':
    from data_classes import Caribic
    caribic = Caribic([2005])
    plot_scatter_global(caribic)
    plot_global_binned_1d(caribic)
    plot_global_binned_2d(caribic)