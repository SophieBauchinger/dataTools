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

# import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable as sm
from matplotlib.colors import ListedColormap as lcm
from matplotlib.patches import Patch

from tools import monthly_mean
from dictionaries import get_col_name, get_vlims, get_default_unit

# supress a gui backend userwarning, not really advisible
import warnings; warnings.filterwarnings("ignore", category=UserWarning,
                                         module='matplotlib')
# ignore warning for np.nanmin / np.nanmax for all-nan sclice
warnings.filterwarnings(action='ignore', message='All-NaN slice encountered')

#%% GlobalData
def scatter_global(glob_obj, subs, single_yr=None, verbose=False,
                   c_pfx=None, as_subplot=False, ax=None):
    """
    Default plotting of scatter values for global data
    
    If as_subplot, plots scatterplot onto given axis (ax)
    """
    if glob_obj.source=='Caribic':
        if c_pfx: pfxs = [c_pfx]
        else: pfxs = glob_obj.pfxs
        for pfx in pfxs:
            df = glob_obj.data[pfx]

            substance = get_col_name(subs, glob_obj.source, pfx)
            if substance not in df.columns:
                if verbose: print(f'No {substance} values to plot in {pfx}')
                continue

            if single_yr is not None: df = df[df.index.year == single_yr]
            df_mm = monthly_mean(df).notna()

            # Plot mixing ratio msmts and monthly mean
            if ax is None or 'fig' in locals():
                fig, ax = plt.subplots(dpi=250)
                plt.title(f'{glob_obj.source} {pfx} {substance} measurements')

            ymin = np.nanmin(df[substance])
            ymax = np.nanmax(df[substance])

            cmap = plt.cm.viridis_r
            extend = 'neither'
            vmin = np.nanmin([ymin, get_vlims(subs)[0]])
            vmax = np.nanmax([ymax, get_vlims(subs)[1]])
            norm = Normalize(vmin, vmax)

            ax.scatter(df.index, df[substance],
                        label=f'{substance.upper()} \
                                {min(df.index.year), max(df.index.year)}',
                        marker='x', zorder=1, c = df[substance],
                        cmap = cmap, norm = norm)

            for i, mean in enumerate(df_mm[substance]):
                y,m = df_mm.index[i].year, df_mm.index[i].month
                xmin = dt.datetime(y, m, 1),
                xmax = dt.datetime(y, m, monthrange(y, m)[1])
                ax.hlines(mean, xmin, xmax, color='black',
                          linestyle='dashed', zorder=2)

            plt.colorbar(sm(norm=norm, cmap=cmap), aspect=50, ax = ax, extend=extend)
            ax.set_ylabel(f'{substance}')
            ax.set_ylim(ymin-0.15, ymax+0.15)
            if 'fig' in locals(): fig.autofmt_xdate(); plt.show()

    elif glob_obj.source=='Mozart':
        substance = get_col_name(subs, glob_obj.source)
        pl = binned_1d(glob_obj, subs, single_yr)
        return pl

def binned_1d(glob_obj, subs, single_yr=None, plot_mean=False,
                          single_graph=False, c_pfx=None, ax=None):
    """
    Plots 1D averaged values over latitude / longitude including colormap
    Parameters:
        substance (str): if None, plots default substance for the object
        single_yr (int): if specified, plots only data for that year
        plot_mean (bool): plot the overall average over all years
        single_graph (bool): choose whether to plot all years on one graph
    """
    # substance = subs # get_col_name(subs, global_data.source, c_pfx)

    if single_yr is not None: years = [int(single_yr)]
    else: years = glob_obj.years

    out_x_list, out_y_list = glob_obj.binned_1d(subs, c_pfx = c_pfx, 
                                                single_yr = single_yr)

    if not single_graph:
        # Plot mixing ratios averages over lats / lons for each year separately
        for out_x, out_y, year in zip(out_x_list, out_y_list, years):
            if not ax: 
                fig, ax = plt.subplots(dpi=300, ncols=2, sharey=True, figsize=(8,3.5))
                fig.suptitle(f'{glob_obj.source} ({c_pfx}) {subs.upper()} for \
                             {year}. Gridsize={glob_obj.grid_size}')

            cmap = plt.cm.viridis_r
            vmin = np.nanmin([np.nanmin(out_x.vmean), np.nanmin(out_y.vmean),
                              get_vlims(subs)[0]])
            vmax = np.nanmax([np.nanmin(out_x.vmean), np.nanmin(out_y.vmean),
                              get_vlims(subs)[1]])
            norm = Normalize(vmin, vmax) # colormap normalisation for chosen vlims

            ax[0].plot(out_x.xintm, out_x.vmean, zorder=1, color='black', lw = 0.5)
            ax[0].scatter(out_x.xintm, out_x.vmean, # plot across latitude
                          c = out_x.vmean, cmap = cmap, norm = norm, zorder=2)
            ax[0].set_xlabel('Latitude [deg]'); plt.xlim(out_x.xbmin, out_x.xbmax)
            ax[0].set_ylabel(f'{get_col_name(subs, glob_obj.source, c_pfx)}')

            ax[1].plot(out_y.xintm, out_y.vmean, zorder=1, color='black', lw = 0.5)
            ax[1].scatter(out_y.xintm, out_y.vmean, # plot across longitude
                          c = out_y.vmean, cmap = cmap, norm = norm, zorder=2)
            ax[1].set_xlabel('Longitude [deg]'); plt.xlim(out_y.xbmin, out_y.xbmax)
            ax[1].set_ylabel(f'{get_col_name(subs, glob_obj.source, c_pfx)}')

            fig.colorbar(sm(norm=norm, cmap=cmap), aspect=50, ax = ax[1])
            plt.show()

    if single_graph:
        # Plot averaged mixing ratios for all years on one graph
        fig, ax = plt.subplots(dpi=300, ncols=2, sharey=True, figsize=(8,3.5))
        fig.suptitle(f'{glob_obj.source} ({c_pfx}) {subs.upper()} for \
                     {glob_obj.years[0]} - {glob_obj.years[-1]}. \
                         Gridsize={glob_obj.grid_size}')

        for out_x, out_y, year in zip(out_x_list, out_y_list, glob_obj.years):
            ax[0].plot(out_x.xintm, out_x.vmean, label=year)#, c = cmap(norm(year)))
            ax[0].set_xlabel('Latitude [deg]'); plt.xlim(out_x.xbmin, out_x.xbmax)
            ax[0].set_ylabel(f'{get_col_name(subs, glob_obj.source, c_pfx)}')

            ax[1].plot(out_y.xintm, out_y.vmean, label=year)# , c = cmap(norm(year)))
            ax[1].set_xlabel('Longitude [deg]'); plt.xlim(out_y.xbmin, out_y.xbmax)
            # ax[1].set_ylabel(f'Mean {subs.upper()} mixing ratio [ppt]')

        if plot_mean: # add average over available years to plot
            total_x_vmean = np.mean([i.vmean for i in out_x_list], axis=0)
            total_y_vmean = np.mean([i.vmean for i in out_y_list], axis=0)
            ax[0].plot(out_x.xintm, total_x_vmean, label='Mean', c = 'k', ls ='dashed')
            ax[1].plot(out_y.xintm, total_y_vmean, label='Mean', c = 'k', ls ='dashed')

        handles, labels = ax[0].get_legend_handles_labels()
        # reversed so that legend aligns with graph
        plt.legend(reversed(handles), reversed(labels),
                   bbox_to_anchor=(1,1), loc='upper left')
        plt.show()
    return

def binned_2d(glob_obj, subs, single_yr=None, c_pfx='GHG', years=None):
    """
    Create a 2D plot of binned mixing ratios for each available year on a grid.
    Parameters:
        substance (str)
        single_yr (int): if specified, plots only data for the chosen year
    """

    if single_yr is not None:
        years = [int(single_yr)]
        # squeeze=False so that axs is still an array
        fig, axs = plt.subplots(dpi=300, figsize=(10,5), squeeze=False)
    else:
        if not years: years = glob_obj.years
        if len(years) <= 3:
            fig, axs = plt.subplots(len(years), 1, dpi=300,
                                    figsize=(8, 3*len(years)))
        elif len(years) > 3:
            fig, axs = plt.subplots(int(len(years)/3), 3, dpi=300,
                                    figsize=(20, max(len(years), 5)))

    out_list = glob_obj.binned_2d(subs, single_yr=single_yr, c_pfx=c_pfx)

    plt.suptitle(f'{glob_obj.source} ({c_pfx}) {subs.upper()} mixing ratio \
                 measurements. Gridsize={glob_obj.grid_size}')
    cmap = plt.cm.viridis_r # create colormap
    vmin = np.nanmin([np.nanmin(out.vmin) for out in out_list])
    vmax = np.nanmax([np.nanmax(out.vmax) for out in out_list])

    if not vmin: vmin = get_vlims(subs)[0]
    if not vmax: vmax = get_vlims(subs)[1]

    # vmin = min([np.nanmin([out.vmin for out in out_list]), [get_vlims(subs)[0]]])
    # vmax = max([np.nanmin([out.vmax for out in out_list]), [get_vlims(subs)[1]]])

    norm = Normalize(vmin, vmax) # normalise color map to set limits
    world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))

    for out, ax, yr in zip(out_list, axs.flatten(), years):
        world.boundary.plot(ax=ax, color='black', linewidth=0.3)
        img = ax.imshow(out.vmean, cmap = cmap, norm=norm, origin='lower',
                   extent=[out.ybmin, out.ybmax, out.xbmin, out.xbmax])

        cbar = plt.colorbar(img, ax=ax, pad=0.08, orientation='vertical') # colorbar
        cbar.ax.set_xlabel(f'{get_col_name(subs, glob_obj.source, c_pfx)}')

        ax.set_title(f'{yr}')
        ax.set_xlabel('Longitude [°E]'); ax.set_xlim(-180,180)
        ax.set_ylabel('Latitude [°N]'); ax.set_ylim(-60,100)

    plt.tight_layout()
    plt.show()
    return

# Mozart
def lonlat_1d(mzt_obj, subs='sf6',
                   lon_values = [10, 60, 120, 180],
                   lat_values = [70, 30, 0, -30, -70],
                   single_yr=None):
    """
    Plots mixing ratio with fixed lon/lat over lats/lons side-by-side
    Parameters:
        lon_values (list of ints): longitude values to average over
        lat_values (list of ints): latitude values to average over
        substance (str): e.g. 'sf6'
        single_yr (int): if specified, plots only data for that year
    """
    # substance = get_col_name(subs, mzt_obj.source)
    if single_yr is not None: years = [int(single_yr)]
    else: years = mzt_obj.years

    out_x_list, out_y_list = mzt_obj.binned_1d(subs, single_yr=single_yr)

    for out_x, out_y, year in zip(out_x_list, out_y_list, years):
        fig, (ax1, ax2) = plt.subplots(dpi=250, ncols=2, figsize=(9,5), sharey=True)
        fig.suptitle(f'MOZART {year} {subs.upper()} at fixed \
                     longitudes / latitudes', size=17)
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

# Caribic
def caribic_2d(c_obj, c_pfx, subs):
    df = c_obj.data[c_pfx]
    substance = get_col_name(subs, c_obj.source, c_pfx)
    df_mm = monthly_mean(df).notna()

    # Plot mixing ratio msmts and monthly mean
    fig, ax = plt.subplots(dpi=250)
    plt.title(f'{c_obj.source} {substance} measurements')
    ymin = np.nanmin(df[substance])
    ymax = np.nanmax(df[substance])

    cmap = plt.cm.viridis_r
    extend = 'neither'
    vmin = np.nanmin([ymin, get_vlims(subs)[0]])
    vmax = np.nanmax([ymax, get_vlims(subs)[1]])
    norm = Normalize(vmin, vmax)

    plt.scatter(df.index, df[substance],
                label=f'{substance.upper()} \
                    {min(df.index.year), max(df.index.year)}',
                marker='x', zorder=1, c = df[substance],
                cmap = cmap, norm = norm)

    for i, mean in enumerate(df_mm[substance]):
        y,m = df_mm.index[i].year, df_mm.index[i].month
        xmin, xmax = dt.datetime(y, m, 1), dt.datetime(y, m, monthrange(y, m)[1])
        ax.hlines(mean, xmin, xmax, color='black',
                  linestyle='dashed', zorder=2)

    plt.colorbar(sm(norm=norm, cmap=cmap), aspect=50, ax = ax, extend=extend)
    plt.ylabel(f'{substance}')
    plt.ylim(ymin-0.15, ymax+0.15)
    fig.autofmt_xdate()
    plt.show() # ignore matplotlib warning that comes up here
#%% LocalData

def local(loc_obj, substance=None, greyscale=False, v_limits = (None,None)):
    """
    Plot all available data as timeseries
    Parameters:
        substance (str): specify substance (optional)
        greyscale (bool): toggle plotting in greyscale or viridis colormap
        v_limits (tuple(int, int)): change limits for colormap
    """
    if greyscale:  # defining monoscale colormap for greyscale plots
        colors = {'day':lcm(['silver']), 'msmts': lcm(['grey'])}
    else: colors = {'msmts':plt.cm.viridis_r, 'day': plt.cm.viridis_r}

    if not substance: substance = loc_obj.substance
    col_name = get_col_name(substance, loc_obj.source)
    if all(isinstance(i, (int, float)) for i in v_limits):
        vmin, vmax = v_limits
    else: vmin, vmax = get_vlims(substance) # default values 
    norm = Normalize(vmin, vmax)
    dflt_unit = get_default_unit(substance)

    # Plot all available info on one graph
    fig, ax = plt.subplots(figsize = (5,3.5), dpi=250)
    # Measurement data (monthly)
    plt.scatter(loc_obj.df.index, loc_obj.df[col_name], c=loc_obj.df[col_name],
                    cmap=colors['msmts'], norm=norm, marker='+', zorder=1,
                    label=f'{loc_obj.source} {substance.upper()}')
    # Daily mean
    if hasattr(loc_obj, 'df_Day'):
        if not loc_obj.df_Day.empty: # check if there is data in the daily df
            plt.scatter(loc_obj.df_Day.index, loc_obj.df_Day[col_name],
                        # c = loc_obj.df_Day[col_name],
                        color = 'silver', marker='+', zorder=0,
                        label=f'{loc_obj.source} {substance.upper()} (D)')
    # Monthly mean
    if hasattr(loc_obj, 'df_monthly_mean'):
        if not loc_obj.df_monthly_mean.empty: # check for data in the monthly df
            for i, mean in enumerate(loc_obj.df_monthly_mean[col_name]):
                # plot monthly mean
                y = loc_obj.df_monthly_mean.index[i].year
                m = loc_obj.df_monthly_mean.index[i].month
                xmin = dt.datetime(y, m, 1)
                xmax = dt.datetime(y, m, monthrange(y, m)[1])
                ax.hlines(mean, xmin, xmax, color='black',
                          linestyle='dashed', zorder=2)
            # avoid multiple labels by replotting single hline with label
            ax.hlines(mean, xmin, xmax, color='black', ls='dashed',
                      label=f'{loc_obj.source} {substance.upper()} (M)')

    plt.ylabel(f'{loc_obj.substance.upper()} mixing ratio [{dflt_unit}]')
    plt.xlim(min(loc_obj.df.index), max(loc_obj.df.index))
    plt.xlabel('Time')

    handles, labels = ax.get_legend_handles_labels()
    if not greyscale:
        plt.colorbar(sm(norm=norm, cmap=colors['day']), aspect=50,
                     ax=ax, extend='neither')

        # Slightly convoluted code to create a legend showing the cmap spectrum
        if len(labels) > 1:
            step = 10
            pa = [ Patch(fc=colors['msmts'](norm(v))) for v
                  in np.linspace(vmin, vmax, step)] # monthly data
            pb = [ Patch(fc='silver') for v
                  in np.linspace(vmin, vmax, step)] # daily data
            pc = [ Patch(fc='black') for v
                  in np.linspace(vmin, vmax, step)] # monthly averages

            h = [] # list of handles
            for a, b, c in zip(pa, pb, pc):
                h.append(a)
                if hasattr(loc_obj, 'df_Day'): h.append(b)
                if hasattr(loc_obj, 'df_monthly_mean'): h.append(c)
            # needed to have multiple color patches for one proper label
            l = [''] * (len(h) - len(labels)) + labels

            ax.legend(handles=h, labels=l, ncol=len(h)/3, columnspacing=-0.3,
                      handletextpad=1/(len(h)/2)+0.2, handlelength=0.12)
    # only show greyscale legend if more than one trace
    elif len(labels) > 1: ax.legend()

    fig.autofmt_xdate()
    plt.show()

#%% Fctn calls - data.plot
# if __name__=='__main__':
#     from dictionaries import substance_list
#     calc_caribic = False
#     if calc_caribic:
#         from data import Caribic, Mauna_Loa, Mace_Head, Mozart
#         year_range = range(2000, 2018)
#         mlo_data = {subs : Mauna_Loa(year_range, substance=subs) for subs
#                     in substance_list('MLO')}
#         caribic = Caribic(year_range, pfxs = ['GHG', 'INT', 'INT2']) # 2005-2020
#         mhd = Mace_Head() # only 2012 data available
#         mzt = Mozart(year_range) # only available up to 2008

#     for pfx in caribic.pfxs: # scatter plots of all caribic data
#         substs = [x for x in substance_list(pfx)
#                   if x not in ['f11', 'f12', 'no', 'noy', 'o3', 'h2o']]
#         f, axs = plt.subplots(int(len(substs)/2), 2,
#                               figsize=(10,len(substs)*1.5), dpi=200)
#         plt.suptitle(f'Caribic {(pfx)}')
#         for subs, ax in zip(substs, axs.flatten()):
#             scatter_global(caribic, subs, c_pfx=pfx,
#                                 as_subplot=True, ax=ax)
#         f.autofmt_xdate()
#         plt.tight_layout()
#         plt.show()

#     for pfx in caribic.pfxs: # lon/lat plots of all caribic data
#         substs = [x for x in substance_list(pfx)
#                   if x not in ['f11', 'f12', 'no', 'noy', 'o3', 'h2o']]
#         for subs in substs:
#             binned_1d(caribic, subs=subs, c_pfx=pfx,
#                                   single_graph=True)

#     for pfx in caribic.pfxs: # maps of all caribic data
#         substs = [x for x in substance_list(pfx)
#                   if x not in ['f11', 'f12', 'no', 'noy', 'o3', 'h2o']]
#         for subs in substs:
#             binned_2d(caribic, subs=subs, c_pfx=pfx)

#     binned_1d(mzt, 'sf6', single_graph=True)
#     yr_ranges = [mzt.years[i:i + 9] for i
#                  in range(0, len(mzt.years), 9)] # create 6-year bundles
#     for yr_range in yr_ranges:
#         binned_2d(mzt, 'sf6', years=yr_range)

#     lonlat_1d(mzt, 'sf6', single_yr = 2005)

#     for subs, mlo_obj in mlo_data.items():
#         local(mlo_obj, subs)

#     local(mhd, 'sf6', v_limits=(7, 8))
