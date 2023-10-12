# -*- coding: utf-8 -*-
"""
@Author: Sophie Bauchinger, IAU
@Date: Thu May 11 13:22:38 2023

Defines different plotting possibilities for objects of the type GlobalData
or LocalData as defined in data_classes

"""
import sys
import warnings
from calendar import monthrange
import datetime as dt
import math
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patheffects as mpe
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable as sm
from matplotlib.colors import ListedColormap as lcm
from matplotlib.patches import Patch
import numpy as np

import geopandas
from mpl_toolkits.axes_grid1 import AxesGrid

if '..' not in sys.path:
    sys.path.append('..')
sys.path = sys.path[1:]
import tools
import dictionaries as dcts

# supress a gui backend userwarning, not really advisable
warnings.filterwarnings("ignore", category=UserWarning, module='matplotlib')
# ignore warning for np.nanmin / np.nanmax for all-nan sclice
warnings.filterwarnings(action='ignore', message='All-NaN slice encountered')

vlims = {  # optimised for Caribic measurements from 2005 to 2020
    'co': (15, 250),
    'o3': (0.0, 1000),
    'h2o': (0.0, 1000),
    'no': (0.0, 0.6),
    'noy': (0.0, 6),
    'co2': (370, 420),
    'ch4': (1650, 1970),
    'f11': (130, 250),
    'f12': (400, 540),
    'n2o': (290, 330),
    'sf6': (5.5, 10),
}


# %% GlobalData plotting
def timeseries_global(glob_obj, detr=False, colorful=True, note='', **subs_kwargs):
    """ Scatter plots of timeseries data incl. monthly averages of chosen substances. """
    data = glob_obj.df
    df_mm = tools.monthly_mean(data)
    substances = [dcts.get_subs(col_name=c) for c in data.columns
                  if c in [s.col_name for s in dcts.get_substances(**subs_kwargs)]
                  and not c.startswith('d_')]

    for substance in substances:
        col = substance.col_name

        if detr and 'detr' + col not in data.columns:
            try:
                glob_obj.detrend_substance(substance.short_name)
                data = glob_obj.df
                df_mm = tools.monthly_mean(data)
                if 'detr_' + col in data.columns:
                    col = 'detr_' + col
            finally:
                print(f'Could not detrend {substance.short_name}')

        fig, ax = plt.subplots(dpi=250, figsize=(8, 4))
        if note:
            ax.text(**dcts.note_dict(ax, s=note, y=0.075))

        # Plot mixing ratios x
        vmin, vmax = vlims.get(substance.short_name)
        if 'mol' in substance.unit:
            ref_unit = dcts.get_substances(short_name=substance.short_name,
                                           source='Caribic')[0].unit
            vmin = tools.conv_PartsPer_molarity(vmin, ref_unit)
            vmax = tools.conv_PartsPer_molarity(vmax, ref_unit)
        cmap = plt.cm.viridis_r if colorful else None
        extend = 'neither'
        norm = Normalize(vmin, vmax) if colorful else None

        ax.scatter(data.index, data[col],
                   # label='Mixing ratio',
                   marker='.', s=4, zorder=1,
                   c=data[col] if colorful else 'grey',
                   alpha=1 if colorful else 0.4,
                   cmap=cmap, norm=norm)

        if colorful:
            plt.colorbar(sm(norm=norm, cmap=cmap), aspect=30, ax=ax,
                         extend=extend, orientation='vertical',
                         label=f'{substance.short_name.upper()} [{substance.unit}]')
            # cbar.ax.set_xlabel(substance.unit, fontsize=8)

        ax.set_ylabel(dcts.make_subs_label(substance))
        ax.set_xlabel('Time')

        if len(glob_obj.years) > 3:
            ax.xaxis.set_major_locator(mdates.YearLocator())
        elif len(glob_obj.years) == 1:
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        else:
            ax.xaxis.set_minor_locator(mdates.MonthLocator(interval=1))

        # Plot monthly means on top of the data
        outline = mpe.withStroke(linewidth=2, foreground='white')
        ax.plot(df_mm.index + dt.timedelta(days=15), df_mm[col],
                path_effects=[outline],
                label='Monthly mean',
                color='k' if colorful else 'g',
                lw=0.7, zorder=2)

        for i, mean in enumerate(df_mm[col]):
            year, month = df_mm.index[i].year, df_mm.index[i].month
            xmin = dt.datetime(year, month, 1)
            xmax = dt.datetime(year, month, monthrange(year, month)[1])
            ax.hlines(mean, xmin, xmax, color='k' if colorful else 'g',
                      linestyle='-', zorder=2,
                      path_effects=[outline])

        ax.legend()
        fig.tight_layout(pad=3.0)
        fig.autofmt_xdate()
        plt.show()


def scatter_lat_lon_binned(glob_obj, detr=False, **subs_kwargs):
    """ Plots 1D averaged values over latitude / longitude including colormap

    Parameters:
        glob_obj (GlobalData): instance with global dataset
        detr (bool): Plot detrended data
    """
    data = glob_obj.df
    substances = [dcts.get_subs(col_name=c) for c in data.columns
                  if c in [s.col_name for s in dcts.get_substances(**subs_kwargs)]
                  and not c.startswith('d_')]

    nplots = len(glob_obj.years)
    nrows = nplots if nplots <= 4 else math.ceil(nplots / 3)
    ncols = 1 if nplots <= 4 else 3

    for substance in substances:
        # TODO: detrended data
        # col = substance.col_name
        # if detr and 'detr' + col not in data.columns:
        #     try:
        #         glob_obj.detrend_substance(substance.short_name)
        #         data = glob_obj.df
        #         if 'detr_' + col in data.columns:
        #             col = 'detr_' + col
        #     finally:
        #         print(f'Could not detrend {substance.short_name}')

        out_x_list, out_y_list = glob_obj.binned_1d(substance)

        values = [item for out_x, out_y in zip(out_x_list, out_y_list) for item in [*out_x.vmean, *out_y.vmean]]
        ymin = np.nanmin(values) * 1.2
        ymax = np.nanmax(values) * 0.9

        fig = plt.figure(dpi=200, figsize=(6 * ncols, 3 * nrows))
        outer_grid = fig.add_gridspec(nrows + 1, ncols,
                                      wspace=0.05, hspace=0.2,
                                      left=0.03, right=0.98,
                                      bottom=0.03, top=0.98,
                                      )
        if nplots >= 4:
            data_type = 'measured' if substance.model == 'MSMT' else 'modeled'
            fig.suptitle(f'Lat/Lon binned {data_type} mixing ratios of {dcts.make_subs_label(substance)}',
                         fontsize=25)
            plt.subplots_adjust(top=0.96)

        # Colormapping
        cmap = plt.cm.viridis_r
        vmin, vmax = vlims.get(substance.short_name)
        if 'mol' in substance.unit:
            ref_unit = dcts.get_substances(short_name=substance.short_name,
                                           source='Caribic')[0].unit
            vmin = tools.conv_PartsPer_molarity(vmin, ref_unit)
            vmax = tools.conv_PartsPer_molarity(vmax, ref_unit)
        norm = Normalize(vmin, vmax)  # colormap normalisation for chosen vlims

        # fig.colorbar(sm(norm=norm, cmap=cmap), aspect=50, ax = axs[1])

        for out_x, out_y, grid, year in zip(out_x_list, out_y_list, outer_grid, glob_obj.years):
            inner_grid = grid.subgridspec(1, 2)
            axs = inner_grid.subplots(sharey=True)
            axs[0].text(**dcts.note_dict(axs[0], x=0.25, y=0.9, s=year))

            if all(np.isnan(out_x.vmean)) and all(np.isnan(out_y.vmean)):
                continue

            axs[0].plot(out_x.xintm, out_x.vmean, zorder=1, color='black', lw=0.5)
            axs[0].scatter(out_x.xintm, out_x.vmean,  # plot across longitude
                           c=out_x.vmean, cmap=cmap, norm=norm)
            axs[0].set_xlabel('Longitude [deg]')  # ; plt.xlim(out_x.xbmin, out_x.xbmax)
            axs[0].set_ylabel(dcts.make_subs_label(substance))
            axs[0].set_ylim(ymax=ymax, ymin=ymin)

            axs[1].plot(out_y.xintm, out_y.vmean, zorder=1, color='black', lw=0.5)
            axs[1].scatter(out_y.xintm, out_y.vmean,  # plot across latitude
                           c=out_y.vmean, cmap=cmap, norm=norm, zorder=2)
            axs[1].set_xlabel('Latitude [deg]')  # ; plt.xlim(out_y.xbmin, out_y.xbmax)

        cax = fig.add_subplot(outer_grid[-1, :])
        cax.axis('off')
        fig.colorbar(sm(norm=norm, cmap=cmap), aspect=50, ax=cax, orientation='horizontal')

        outer_grid.tight_layout(fig, pad=3)

        fig.show()

        break

def plot_binned_2d(glob_obj, detr=False, **subs_kwargs):
    # glob_obj, subs, single_yr=None, c_pfx=None, years=None, detr=False):
    """ Create a 2D plot of binned mixing ratios for each available year on a grid.

    Parameters:
        glob_obj (GlobalData): instance with global dataset
        detr (bool): Plot detrended data
    """
    data = glob_obj.df
    substances = [dcts.get_subs(col_name=c) for c in data.columns
                  if c in [s.col_name for s in dcts.get_substances(**subs_kwargs)]
                  and not c.startswith('d_')]

    for substance in substances:  # for year in glob_obj.years
        # col = substance.col_name
        # if detr and 'detr' + col not in data.columns:
        #     try:
        #         glob_obj.detrend_substance(substance.short_name)
        #         data = glob_obj.df
        #         if 'detr_' + col in data.columns:
        #             col = 'detr_' + col
        #     finally:
        #         print(f'Could not detrend {substance.short_name}')

        nplots = len(glob_obj.years)
        nrows = nplots if nplots <= 4 else math.ceil(nplots / 3)
        ncols = 1 if nplots <= 4 else 3

        fig = plt.figure(dpi=10, figsize=(6 * ncols, 3 * nrows))

        grid = AxesGrid(fig, 111,  # similar to subplot(142)
                        nrows_ncols=(nrows, ncols),
                        axes_pad=0.4,
                        share_all=True,
                        label_mode="all",
                        cbar_location="bottom",
                        cbar_mode="single")

        if nplots >= 4:
            data_type = 'measured' if substance.model == 'MSMT' else 'modeled'
            fig.suptitle(f'Global {data_type} mixing ratios of {dcts.make_subs_label(substance)}',
                         fontsize=25)
            plt.subplots_adjust(top=0.96)

        out_list = glob_obj.binned_2d(substance)

        cmap = plt.cm.viridis_r  # create colormap
        vmin, vmax = vlims.get(substance.short_name)
        if 'mol' in substance.unit:
            ref_unit = dcts.get_substances(short_name=substance.short_name,
                                           source='Caribic')[0].unit
            vmin = tools.conv_PartsPer_molarity(vmin, ref_unit)
            vmax = tools.conv_PartsPer_molarity(vmax, ref_unit)
        norm = Normalize(vmin, vmax)  # normalise color map to set limits
        world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))

        for i, (out, ax, year) in enumerate(zip(out_list, grid, glob_obj.years)):
            world.boundary.plot(ax=ax, color='black', linewidth=0.3)
            img = ax.imshow(out.vmean.T, cmap=cmap, norm=norm, origin='lower',
                            extent=[out.xbmin, out.xbmax, out.ybmin, out.ybmax])

            # ax.set_title(f'{yr}')#, weight='bold')
            ax.text(**dcts.note_dict(ax, 0.13, 0.1, f'{year}'), weight='bold')
            ax.set_xlim(-180, 180)
            ax.set_ylim(-60, 100)

            # label outer plot axes
            if grid._get_col_row(i)[0] == 0:
                ax.set_ylabel('Latitude [°N]')
            if grid._get_col_row(i)[0] == ncols:
                ax.set_xlabel('Longitude [°E]')

            if i==0:
                cbar = grid.cbar_axes[0].colorbar(img)
                cbar.ax.tick_params(labelsize=15, which='both')
                # cbar.ax.minorticks_on()
                cbar.ax.set_xlabel(dcts.make_subs_label(substance), fontsize=15)

        for i, ax in enumerate(grid):  # hide extra plots
            if i >= nplots:
                ax.axis('off')

        # cbar = plt.colorbar(img, fig.add_axes([0.05, 0.98, 0.95, 0.05], box_aspect=0.05),
        #                     aspect=10, pad=0.1, orientation='horizontal') # colorbar


        # fig.tight_layout()
        plt.show()


# Mozart
def lonlat_1d(mzt_obj, subs='sf6',
              lon_values=(10, 60, 120, 180),
              lat_values=(70, 30, 0, -30, -70),
              single_yr=None):
    """
    Plots mixing ratio with fixed lon/lat over lats/lons side-by-side
    Parameters:
        mzt_obj (Mozart): Instance of Mozart data
        subs (str): short_name of substance to plot
        lon_values (List[int]): longitude values to average over
        lat_values (List[int]): latitude values to average over
        single_yr (int): if specified, plots only data for that year
    """
    if single_yr is not None:
        years = [int(single_yr)]
    else:
        years = mzt_obj.years

    out_x_list, out_y_list = mzt_obj.binned_1d(subs, single_yr=single_yr)

    for out_x, out_y, year in zip(out_x_list, out_y_list, years):
        fig, (ax1, ax2) = plt.subplots(dpi=250, ncols=2, figsize=(9, 5), sharey=True)
        fig.suptitle(f'MOZART {year} {subs.upper()} at fixed \
                     longitudes / latitudes', size=17)
        mzt_obj.ds.SF6.sel(time=year, longitude=lon_values,
                           method='nearest').plot.line(x='latitude', ax=ax1)
        ax1.plot(out_x.xintm, out_x.vmean, c='k', ls='dashed', label='average')

        mzt_obj.ds.SF6.sel(time=year, latitude=lat_values,
                           method="nearest").plot.line(x='longitude', ax=ax2)
        ax2.plot(out_y.xintm, out_y.vmean, c='k', ls='dashed', label='average')

        ax1.set_title('')
        ax2.set_title('')
        ax2.set_ylabel('')
        plt.show()


# %% LocalData

def local(loc_obj, greyscale=False, v_limits=(None, None), **subs_kwargs):
    """ Plot all available data as timeseries

    Parameters:
        loc_obj (LocalData): Instance of LocalData storing ground-based measurements
        greyscale (bool): toggle plotting in greyscale or viridis colormap
        v_limits (tuple(int, int)): change limits for colormap

        key short_name (str): specify substance (optional)
    """
    if greyscale:  # defining monoscale colormap for greyscale plots
        colors = {'day': lcm(['silver']), 'msmts': lcm(['grey'])}
    else:
        colors = {'msmts': plt.cm.viridis_r, 'day': plt.cm.viridis_r}

    substances = [s for s in dcts.get_substances(source=loc_obj.source, **subs_kwargs)
                  if s.col_name in loc_obj.df.columns]

    for subs in substances:
        if all(isinstance(i, (int, float)) for i in v_limits):
            vmin, vmax = v_limits
        else:
            vmin, vmax = vlims.get(subs.short_name)  # dcts.get_vlims(substance) # default values
        norm = Normalize(vmin, vmax)

        # Plot all available info on one graph
        fig, ax = plt.subplots(figsize=(5, 3.5), dpi=250)
        # Measurement data (monthly)
        plt.scatter(loc_obj.df.index, loc_obj.df[subs.col_name], c=loc_obj.df[subs.col_name],
                    cmap=colors['msmts'], norm=norm, marker='+', zorder=1,
                    label=dcts.make_subs_label(subs, name_only=True))
        # Daily mean
        if hasattr(loc_obj, 'df_Day'):
            if not loc_obj.df_Day.empty:  # check if there is data in the daily df
                plt.scatter(loc_obj.df_Day.index, loc_obj.df_Day[subs.col_name],
                            color='silver', marker='+', zorder=0,
                            label=f'{dcts.make_subs_label(subs, name_only=True)} (D)')
        # Monthly mean
        if hasattr(loc_obj, 'df_monthly_mean'):
            if not loc_obj.df_monthly_mean.empty:  # check for data in the monthly df
                for i, mean in enumerate(loc_obj.df_monthly_mean[subs.col_name]):
                    # plot monthly mean
                    y = loc_obj.df_monthly_mean.index[i].year
                    m = loc_obj.df_monthly_mean.index[i].month
                    xmin = dt.datetime(y, m, 1)
                    xmax = dt.datetime(y, m, monthrange(y, m)[1])
                    ax.hlines(mean, xmin, xmax, color='black',
                              linestyle='dashed', zorder=2)
                    if i == 0:  # avoid multiple labels by replotting single hline with label
                        ax.hlines(mean, xmin, xmax, color='black', ls='dashed',
                                  label=f'{dcts.make_subs_label(subs, name_only=True)} (M)')

        plt.ylabel(dcts.make_subs_label(subs))
        plt.xlim(min(loc_obj.df.index), max(loc_obj.df.index))
        plt.xlabel('Time')

        labels = ax.get_legend_handles_labels()[1]
        if not greyscale:
            plt.colorbar(sm(norm=norm, cmap=colors['day']), aspect=50,
                         ax=ax, extend='neither')

            # Slightly convoluted code to create a legend showing the cmap spectrum
            if len(labels) > 1:
                step = 10
                patches_msmt = [Patch(fc=colors['msmts'](norm(v))) for v
                                in np.linspace(vmin, vmax, step)]
                patches_day = [Patch(fc='silver')]*step
                patches_mon = [Patch(fc='black')]*step
                # ... for v in np.linspace(vmin, vmax, step)]

                handles = []  # list of handles
                for handles_msmt, handles_day, handles_mon in (
                        zip(patches_msmt, patches_day, patches_mon)):
                    handles.append(handles_msmt)
                    if hasattr(loc_obj, 'df_Day'):
                        handles.append(handles_day)
                    if hasattr(loc_obj, 'df_monthly_mean'):
                        handles.append(handles_mon)
                # needed to have multiple color patches for one proper label
                labels = [''] * (len(handles) - len(labels)) + labels

                ax.legend(handles=handles, labels=labels, ncol=len(handles) / 3, columnspacing=-0.3,
                          handletextpad=1 / (len(handles) / 2) + 0.2, handlelength=0.12)
        # only show greyscale legend if more than one trace
        elif len(labels) > 1:
            ax.legend()

        fig.autofmt_xdate()
        plt.show()

# %% Fctn calls - data.plot
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
#%%
sys.path.append('..')
from data import Caribic
caribic = Caribic()

scatter_lat_lon_binned(caribic, short_name='sf6')