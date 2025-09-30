# -*- coding: utf-8 -*-
"""
@Author: Sophie Bauchinger, IAU
@Date: Thu May 11 13:22:38 2023

Defines different plotting possibilities for objects of the type GlobalData
or LocalData as defined in data_classes

"""
from calendar import monthrange
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import datetime as dt
import geopandas
import math
import matplotlib.dates as mdates
from matplotlib.colors import Normalize, BoundaryNorm
from matplotlib.cm import ScalarMappable as sm
from matplotlib.colors import ListedColormap as lcm
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch, Rectangle
import matplotlib.patheffects as mpe
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid
import numpy as np
import plotly.graph_objs as go
import plotly.express as px

import toolpac.calc.binprocessor as bp # type: ignore

from dataTools import tools
import dataTools.dictionaries as dcts
import dataTools.plot.create_figure as cfig
import dataTools.data.BinnedData as bin_tools

def timeseries_global(GlobalObject, substances=None, colorful=True, note=''):
    """ Scatter plots of timeseries data incl. monthly averages of chosen substances. """
    data = GlobalObject.df
    df_mm = tools.time_mean(data, 'M')

    for subs in GlobalObject.substances if substances is None else substances:
        fig, ax = plt.subplots(dpi=250, figsize=(8, 4))
        if note:
            ax.text(**dcts.note_dict(ax, s=note, y=0.075))

        # Plot mixing ratios x
        vmin, vmax = subs.vlims()
        if 'mol' in subs.unit:
            ref_unit = dcts.get_substances(short_name=subs.short_name,
                                        source='Caribic')[0].unit
            vmin = tools.conv_PartsPer_molarity(vmin, ref_unit)
            vmax = tools.conv_PartsPer_molarity(vmax, ref_unit)
        cmap = plt.cm.viridis_r if colorful else None
        extend = 'neither'
        norm = Normalize(vmin, vmax) if colorful else None

        ax.scatter(data.index, data[subs.col_name],
                # label='Mixing ratio',
                marker='.', s=4, zorder=1,
                c=data[subs.col_name] if colorful else 'grey',
                alpha=1 if colorful else 0.4,
                cmap=cmap, norm=norm)

        if colorful:
            plt.colorbar(sm(norm=norm, cmap=cmap), aspect=30, ax=ax,
                        extend=extend, orientation='vertical',
                        label=f'{subs.short_name.upper()} [{subs.unit}]')
            # cbar.ax.set_xlabel(substance.unit, fontsize=8)

        ax.set_ylabel(subs.label())
        ax.set_xlabel('Time')

        if len(GlobalObject.years) > 3:
            ax.xaxis.set_major_locator(mdates.YearLocator())
        elif len(GlobalObject.years) == 1:
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        else:
            ax.xaxis.set_minor_locator(mdates.MonthLocator(interval=1))

        # Plot monthly means on top of the data
        outline = mpe.withStroke(linewidth=2, foreground='white')
        ax.plot(df_mm.index, 
                df_mm[subs.col_name],
                path_effects=[outline],
                label='Monthly mean',
                color='k' if colorful else 'g',
                lw=0.7, zorder=2)

        for i, mean in enumerate(df_mm[subs.col_name]):
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

def scatter_lat_lon_binned(GlobalObject, substances=None, bin_attr='vmean'):
    """ Plots 1D averaged values over latitude / longitude including colormap

    Parameters:
        self (GlobalData): instance with global dataset
        detr (bool): Plot detrended data
    """

    nplots = len(GlobalObject.years)
    nrows = nplots if nplots <= 4 else math.ceil(nplots / 3)
    ncols = 1 if nplots <= 4 else 3

    for subs in GlobalObject.substances if substances is None else substances:
        out_x_list, out_y_list = GlobalObject.binned_1d(subs)

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
            data_type = 'measured' if subs.model == 'MSMT' else 'modeled'
            fig.suptitle(f'Lat/Lon binned {data_type} mixing ratios of {subs.label()}',
                        fontsize=25)
            plt.subplots_adjust(top=0.96)

        # Colormapping
        cmap = plt.cm.viridis_r
        vmin, vmax = subs.vlims()
        if 'mol' in subs.unit:
            ref_unit = dcts.get_subss(short_name=subs.short_name,
                                        source='Caribic')[0].unit
            vmin = tools.conv_PartsPer_molarity(vmin, ref_unit)
            vmax = tools.conv_PartsPer_molarity(vmax, ref_unit)
        norm = Normalize(vmin, vmax)  # colormap normalisation for chosen vlims

        # fig.colorbar(sm(norm=norm, cmap=cmap), aspect=50, ax = axs[1])

        for out_x, out_y, grid, year in zip(out_x_list, out_y_list, outer_grid, GlobalObject.years):
            inner_grid = grid.subgridspec(1, 2)
            axs = inner_grid.subplots(sharey=True)
            axs[0].text(**dcts.note_dict(axs[0], x=0.25, y=0.9, s=year))

            if all(np.isnan(out_x.vmean)) and all(np.isnan(out_y.vmean)):
                continue
            
            data_x = getattr(out_x, bin_attr)
            data_y = getattr(out_y, bin_attr)

            axs[0].plot(out_x.xintm, data_x, zorder=1, color='black', lw=0.5)
            axs[0].scatter(out_x.xintm, data_x,  # plot across longitude
                        c=data_x, cmap=cmap, norm=norm)
            axs[0].set_xlabel('Longitude [deg]')  # ; plt.xlim(out_x.xbmin, out_x.xbmax)
            axs[0].set_ylabel(subs.label())
            axs[0].set_ylim(ymax=ymax, ymin=ymin)

            axs[1].plot(out_y.xintm, data_y, zorder=1, color='black', lw=0.5)
            axs[1].scatter(out_y.xintm, data_y,  # plot across latitude
                        c=data_y, cmap=cmap, norm=norm, zorder=2)
            axs[1].set_xlabel('Latitude [deg]')  # ; plt.xlim(out_y.xbmin, out_y.xbmax)

        cax = fig.add_subplot(outer_grid[-1, :])
        cax.axis('off')
        fig.colorbar(sm(norm=norm, cmap=cmap), aspect=50, ax=cax, orientation='horizontal')

        outer_grid.tight_layout(fig, pad=3)

        fig.show()

        break

def plot_binned_2d(GlobalObject, bin_attr='vmean', hide_lats=False, substances=None,
                    projection='moll'): 
    """ 2D plot of binned substance data for all years at once. """
    data = GlobalObject.df
    substances = GlobalObject.substances if substances is None else substances

    for subs in (substances if not bin_attr=='vcount' else [dcts.get_subs(col_name='N2O')]): 
        fig, ax = plt.subplots(dpi=300, figsize=(9, 3.5))
        tools.world().boundary.plot(ax=ax, color='black', linewidth=0.3)
        
        ax.set_title(subs.label() if bin_attr != 'vcount' \
                        else 'Distribution of greenhouse gas flask measurements')
        
        cmap = dcts.dict_colors()[bin_attr]
        vmin, vmax = subs.vlims(bin_attr=bin_attr)
        if 'mol' in subs.unit:
            ref_unit = dcts.get_substances(short_name=subs.short_name,
                                        source='Caribic')[0].unit
            vmin = tools.conv_PartsPer_molarity(vmin, ref_unit)
            vmax = tools.conv_PartsPer_molarity(vmax, ref_unit)
        norm = Normalize(vmin, vmax)  # normalise color map to set limits

        if bin_attr=='vcount': 
            cmap.set_under('white')
            bounds=[1, 5, 10, 15, 30, 45, 60]
            norm = BoundaryNorm(bounds, cmap.N)

        bci = bp.Bin_equi2d(-90, 90, GlobalObject.grid_size,
                            -180, 180, GlobalObject.grid_size)
        lat = data.geometry.y 
        lon = data.geometry.x
        out = bp.Simple_bin_2d(data[subs.col_name], lat, lon, bci)
        img_data = getattr(out, bin_attr)
        img = ax.imshow(img_data, cmap=cmap, norm=norm, origin='lower',
                        extent=[out.ybmin, out.ybmax, out.xbmin, out.xbmax])
        
        if hide_lats:
            ax.hlines(30, -180, 180, color='k', lw=1, ls='dashed')
            ax.add_patch(Rectangle((-180, -90), 180*2, 90+30, facecolor="grey", alpha=0.25))

        # label outer plot axes
        ax.set_ylabel('Latitude [°N]')
        ax.set_xlabel('Longitude [°E]')
        
        cbar = plt.colorbar(img, ax=ax, extend='neither' if not bin_attr=='vcount' else 'max')
        cbar.ax.set_xlabel(f'[{subs.unit}]' if bin_attr!='vcount' else '[# Points]')
    return fig, ax

def yearly_plot_binned_2d(GlobalObject, bin_attr='vmean', **subs_kwargs):
    # GlobalObject, subs, single_yr=None, c_pfx=None, years=None, detr=False):
    """ Create a 2D plot of binned mixing ratios for each available year on a grid.

    Parameters:
        GlobalObject (GlobalData): instance with global dataset
        detr (bool): Plot detrended data
    """
    data = GlobalObject.df
    substances = [dcts.get_subs(col_name=c) for c in data.columns
                if c in [s.col_name for s in dcts.get_substances(**subs_kwargs)]
                and not (c.startswith('d_') or '_d_' in c)]

    for subs in substances:  
        nplots = len(GlobalObject.years)
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
            data_type = 'measured' if subs.model == 'MSMT' else 'modeled'
            fig.suptitle(f'Global {data_type} mixing ratios of {subs.label()}',
                        fontsize=25)
            plt.subplots_adjust(top=0.96)

        out_list = GlobalObject.binned_2d(subs)

        cmap = plt.cm.viridis_r  # create colormap
        vmin, vmax = subs.vlims()# vlims.get(substance.short_name)
        if 'mol' in subs.unit:
            ref_unit = dcts.get_substances(short_name=subs.short_name,
                                        source='Caribic')[0].unit
            vmin = tools.conv_PartsPer_molarity(vmin, ref_unit)
            vmax = tools.conv_PartsPer_molarity(vmax, ref_unit)
        norm = Normalize(vmin, vmax)  # normalise color map to set limits
        world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))

        for i, (out, ax, year) in enumerate(zip(out_list, grid, GlobalObject.years)):
            world.boundary.plot(ax=ax, color='black', linewidth=0.3)
            
            img_data = getattr(out, bin_attr)
            img = ax.imshow(img_data.T, cmap=cmap, norm=norm, origin='lower',
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
                cbar.ax.set_xlabel(subs.label(), fontsize=15)

        for i, ax in enumerate(grid):  # hide extra plots
            if i >= nplots:
                ax.axis('off')

        # cbar = plt.colorbar(img, fig.add_axes([0.05, 0.98, 0.95, 0.05], box_aspect=0.05),
        #                     aspect=10, pad=0.1, orientation='horizontal') # colorbar


        # fig.tight_layout()
        plt.show()

def mxr_vs_vcoord(GlobalObject, subs, vcoord, tick_params = {}, ax=None, note=None, **kwargs): 
    """ Plot datapoints on a simple vcoord vs. substance mixing ratio plot. """
    GlobalObject.data['df']['season'] = tools.make_season(GlobalObject.data['df'].index.month)
    if ax is None:
        _, ax = plt.subplots(figsize = (6,3), dpi=kwargs.pop('dpi', None))
    for s in set(GlobalObject.df['season'].tolist()):
        df = GlobalObject.df[GlobalObject.df['season'] == s].dropna(subset=[vcoord.col_name, subs.col_name], how='any')
        if len(df) == 0: 
            continue

        x = df[subs.col_name]
        y = df[vcoord.col_name]

        ax.scatter(x, y, 
                    marker='.',
                    label = dcts.dict_season()[f'name_{s}'],
                    c=dcts.dict_season()[f'color_{s}'], 
                    zorder=2, 
                    lw=0.1)

    ax.set_ylabel(vcoord.label())
    if not tick_params.get('bottom') is False: 
        ax.set_xlabel(subs.label())
    
    ax.tick_params(**tick_params)
    if note: 
        ax.text(**dcts.note_dict(ax, y = 0.05, s= note))
        
    ax.grid(True, zorder=0, ls='dashed', alpha=0.5)    
    
def plot_sf6_detrend_reltp_progression(GlobalObject):    
    _, ((ax11, ax12), (ax21, ax22)) = plt.subplots(2, 2, dpi=300, figsize=(9,6))
    
    [s11] = GlobalObject.get_substs(short_name='sf6', detr=False) # dcts.get_subs(col_name='SF6')
    [c11] = GlobalObject.get_coords(vcoord='pt', tp_def='nan', model='ERA5')
    
    [s12] = GlobalObject.get_substs(short_name='detr_sf6', detr=True)
    c12 = c11
    
    s22 = s12
    [c22] = GlobalObject.get_coords(vcoord='pt', rel_to_tp = True, tp_def = 'therm', model='ERA5') 
    # dcts.get_coord(col_name='int_ERA5_D_TROP1_THETA')
    
    GlobalObject.mxr_vs_vcoord(s11, c11, 
                        ax = ax11, 
                        tick_params = dict(top=True, labeltop=True, bottom=True, labelbottom=True),
                        )
    ax11.set_xlabel(s11.label(), loc='center')
    ax11.set_ylabel(c11.label(), loc='center')
    ax11.set_xlim(5,11.6)
    
    GlobalObject.mxr_vs_vcoord(s12, c12, 
                ax=ax12,
                tick_params  = dict(top=True, labeltop=True, bottom=False, labelbottom=False), 
                )
    ax12.set_xlabel(s12.label(), loc='center')
    ax12.set_ylabel(c12.label(), loc='center')
    
    
    GlobalObject.mxr_vs_vcoord(s22, c22, 
                ax=ax22,
                tick_params  = dict(top=False, labeltop=False, bottom=True, labelbottom=True),
                )
    ax22.set_xlabel(s22.label(), loc='center')
    ax22.set_ylabel(c22.label(), loc='center')
    
    plt.subplots_adjust(wspace=0, hspace=0)
    
    ax12.yaxis.tick_right()
    ax12.yaxis.set_label_position("right")
    ax22.yaxis.tick_right()
    ax22.yaxis.set_label_position("right")
    
    ax21.set_visible(False)
    
    plt.show()

def make_3d_scatter(GlobalObject, vcoord, color_var, eql=False, **plot_kwargs): 
    """ Plot the given substance on a 3D plot of lon-lat-vcoord. """
            
    z_name = vcoord.col_name 
    c_name = color_var.col_name
    
    df = GlobalObject.df.dropna(subset = [c_name])
    
    c_label = '$' + ''.join([i for i in color_var.label() if i != '$'])  +'$'
    z_label = '$' + ''.join([i for i in vcoord.label() if i != '$']) + '$'
    
    c_label = fr'{c_label}'
    z_label = fr'{z_label}'
    
    labels = {
        'x' : dcts.get_coord('geometry.x').label(),
        'y' : dcts.get_coord('geometry.y').label(),
        z_name : z_label,
        c_name : c_label,
        }
    print(labels)

    y = df.geometry.y
    x = df.geometry.x
    if eql:
        [eql_coord] = GlobalObject.get_coords(hcoord = 'eql')
        x = df[eql_coord.col_name]
        
    fig = px.scatter_3d(df, x=x, y=y,
                        z=z_name,
                        color = c_name,
                        # labels = labels,
                        **plot_kwargs,
                        )
    fig.update_traces(marker_size=2.5)

    # Add the world outline
    map_traces = []
    for _, row in tools.world().iterrows():
        if row.geometry.geom_type == 'Polygon':
            x, y = row.geometry.exterior.xy
            x = np.array(x)
            y = np.array(y) 
            z = [df[vcoord.col_name].min()] * len(x)
            map_traces.append(go.Scatter3d(x=x, y=y, z=z, mode='lines', line=dict(color='black', width=1)))
        elif row.geometry.geom_type == 'MultiPolygon':
            for poly in row.geometry.geoms:
                x,y = poly.exterior.xy
                x = np.array(x)
                y = np.array(y)
                z = [df[vcoord.col_name].min()] * len(x)
                map_traces.append(go.Scatter3d(x=x, y=y, z=z, mode='lines', line=dict(color='black', width=1)))

    for trace in map_traces:
        fig.add_trace(trace)
        
    fig.update_scenes(
        aspectmode='data',
        camera=dict(
            eye = dict(x=0, y=-0.75, z=1.25))
    )
    fig.update_layout(showlegend=False) 
    fig.show()

# Some more processed stuff for publications
def typical_profiles_n2o_o3(GlobalObject, subs1, subs2, vcoord, tp_col): 
    """ Show seasonal average msmts and mean tropopause height on raw data background. """
    # Get seasonal mean tropopause height
    mean_tp_height = dict()
    for s in [1,2,3,4]:
        mean_tp_height[s] = GlobalObject.sel_season(s).df[tp_col].mean()
        
    # Plot seasonal mean vertical profiles
    fig, axs = plt.subplots(1,2, figsize=(6,4), dpi=300, sharey=True)
    for subs, ax in zip([subs1, subs2], axs):
        bin_dict = bin_tools.seasonal_binning(
            GlobalObject.df, subs, vcoord, xbsize=0.5)
        ax.set_xlabel(subs.label())
        ax.scatter(GlobalObject.get_var_data(subs), GlobalObject.df[vcoord.col_name], color='xkcd:light grey')
        for s in bin_dict.keys():
            ax.errorbar(bin_dict[s].vmean, bin_dict[s].xintm, xerr = bin_dict[s].vstdv, 
                        color = dcts.dict_season()[f"color_{s}"])
        xlims = ax.get_xlim()
        
        for s in bin_dict.keys():
            ax.hlines(mean_tp_height[s], *xlims,
                    color = dcts.dict_season()[f"color_{s}"],
                    ls='dashed', zorder=0,
                    label = dcts.dict_season()[f"name_{s}"].split(' ')[0])

            ax.hlines(mean_tp_height[s], *xlims,
                    color = dcts.dict_season()[f"color_{s}"],
                    zorder=1, lw=0.5, ls=(0, (5, 10)))
        ax.set_xlim(*xlims)
        ax.grid(axis='x', ls='dotted', c='xkcd:light gray')

    axs[0].set_ylabel(vcoord.label(coord_only=True) + f" [{vcoord.unit}]")
    axs[1].tick_params(left=False, labelleft=False, right=True, labelright=True)

    fig.tight_layout()
    fig.subplots_adjust(top=0.85)

    handles = cfig.typical_profile_legend_handles()
    axs[1].legend(handles = handles[:2], 
                  bbox_to_anchor = (0,0,1,0.35), 
                  loc = 'center left')
    axs[0].legend(handles = handles[2:], 
                  bbox_to_anchor = (0,0,1,1))
    
    return None

# Functions for plotting curtain and flight track
def curtain_plot(fig, ax, curtain_ds, var='pv'):
    """ Flight track curtain data on pressure after lon / lat interpolation. """
    # Variables
    data = curtain_ds[var]
    times = curtain_ds.time.values
    pressure = curtain_ds.isobaricInhPa.values
    
    # --- Upper: Curtain plot ---
    cf = ax.contourf(times, pressure, data.T, levels=20, cmap="coolwarm")
    ax.invert_yaxis()
    ax.set_title(f"{var.upper()} Curtain")
    ax.set_ylabel("Pressure (hPa)")
    ax.set_xlabel("Time")

    cbar = fig.colorbar(cf, ax=ax, label=f"{var.upper()} ({data.units})")
    cbar.set_ticks(np.linspace(cf.norm.vmin, cf.norm.vmax, 5))
    return fig, ax

def flight_track_map(fig, ax, gdf, fl_ID=''):
    """ Add geographical context: Map of the flight track with colour for times. """
    times = gdf.index
    lon = gdf.geometry.x
    lat = gdf.geometry.y
    
    ax.set_title(f"Flight Path {fl_ID}")
    ax.add_feature(cfeature.COASTLINE, edgecolor = '#505050')
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue', alpha = 0.5)
    ax.set_ylim(10, 90)
    ax.set_xlim(-180, 40)
    gridliner = ax.gridlines(draw_labels=True)
    gridliner.right_labels = False
    gridliner.top_labels = False

    sc = ax.scatter(lon, lat, c=np.arange(len(times)), cmap="plasma", s=10, transform=ccrs.PlateCarree())
    ax.plot(lon, lat, 'k-', transform=ccrs.PlateCarree(), linewidth=0.8)
    cbar = plt.colorbar(sc, ax=ax, pad=0.05, label="Time Index")
    ax.tick_params(labelright=False)
    return fig, ax

def curtain_overview(curtain_ds, flight_gdf, variables= ['pv', 't', 'o3', 'r', 'z', 'theta']):
    """ Create curtain plots with flight track map for the given variables """
    fig = plt.figure(figsize=(8, (len(vars)+1)*4))
    gs = gridspec.GridSpec(len(variables)+1, 1, hspace=0.5)

    for i, var in enumerate(variables): 
        ax = fig.add_subplot(gs[i])
        curtain_plot(fig, ax, curtain_ds, var)
        fig.autofmt_xdate()

        # Add flight pressure
        ax.plot(flight_gdf.index, flight_gdf['ERA5_PRESS'], 'k--', label='Flight level')
        ax.legend(loc='lower right')

    ax0 = fig.add_subplot(gs[-1], projection=ccrs.PlateCarree())
    flight_track_map(fig, ax0, flight_gdf)

    plt.show()

# %% GlobalData plotting
class GlobalDataPlotterMixin: 
    """ Simple plotting capabilities for GlobalData objects. """

    def timeseries_global(self, substances=None, colorful=True, note=''):
        """ Scatter plots of timeseries data incl. monthly averages of chosen substances. """
        data = self.df
        df_mm = tools.time_mean(data, 'M')

        for subs in self.substances if substances is None else substances:
            fig, ax = plt.subplots(dpi=250, figsize=(8, 4))
            if note:
                ax.text(**dcts.note_dict(ax, s=note, y=0.075))

            # Plot mixing ratios x
            vmin, vmax = subs.vlims()
            if 'mol' in subs.unit:
                ref_unit = dcts.get_substances(short_name=subs.short_name,
                                            source='Caribic')[0].unit
                vmin = tools.conv_PartsPer_molarity(vmin, ref_unit)
                vmax = tools.conv_PartsPer_molarity(vmax, ref_unit)
            cmap = plt.cm.viridis_r if colorful else None
            extend = 'neither'
            norm = Normalize(vmin, vmax) if colorful else None

            ax.scatter(data.index, data[subs.col_name],
                    # label='Mixing ratio',
                    marker='.', s=4, zorder=1,
                    c=data[subs.col_name] if colorful else 'grey',
                    alpha=1 if colorful else 0.4,
                    cmap=cmap, norm=norm)

            if colorful:
                plt.colorbar(sm(norm=norm, cmap=cmap), aspect=30, ax=ax,
                            extend=extend, orientation='vertical',
                            label=f'{subs.short_name.upper()} [{subs.unit}]')
                # cbar.ax.set_xlabel(substance.unit, fontsize=8)

            ax.set_ylabel(subs.label())
            ax.set_xlabel('Time')

            if len(self.years) > 3:
                ax.xaxis.set_major_locator(mdates.YearLocator())
            elif len(self.years) == 1:
                ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
            else:
                ax.xaxis.set_minor_locator(mdates.MonthLocator(interval=1))

            # Plot monthly means on top of the data
            outline = mpe.withStroke(linewidth=2, foreground='white')
            ax.plot(df_mm.index, 
                    df_mm[subs.col_name],
                    path_effects=[outline],
                    label='Monthly mean',
                    color='k' if colorful else 'g',
                    lw=0.7, zorder=2)

            for i, mean in enumerate(df_mm[subs.col_name]):
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

    def scatter_lat_lon_binned(self, substances=None, bin_attr='vmean'):
        """ Plots 1D averaged values over latitude / longitude including colormap

        Parameters:
            self (GlobalData): instance with global dataset
            detr (bool): Plot detrended data
        """

        nplots = len(self.years)
        nrows = nplots if nplots <= 4 else math.ceil(nplots / 3)
        ncols = 1 if nplots <= 4 else 3

        for subs in self.substances if substances is None else substances:
            out_x_list, out_y_list = self.binned_1d(subs)

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
                data_type = 'measured' if subs.model == 'MSMT' else 'modeled'
                fig.suptitle(f'Lat/Lon binned {data_type} mixing ratios of {subs.label()}',
                            fontsize=25)
                plt.subplots_adjust(top=0.96)

            # Colormapping
            cmap = plt.cm.viridis_r
            vmin, vmax = subs.vlims()
            if 'mol' in subs.unit:
                ref_unit = dcts.get_subss(short_name=subs.short_name,
                                            source='Caribic')[0].unit
                vmin = tools.conv_PartsPer_molarity(vmin, ref_unit)
                vmax = tools.conv_PartsPer_molarity(vmax, ref_unit)
            norm = Normalize(vmin, vmax)  # colormap normalisation for chosen vlims

            # fig.colorbar(sm(norm=norm, cmap=cmap), aspect=50, ax = axs[1])

            for out_x, out_y, grid, year in zip(out_x_list, out_y_list, outer_grid, self.years):
                inner_grid = grid.subgridspec(1, 2)
                axs = inner_grid.subplots(sharey=True)
                axs[0].text(**dcts.note_dict(axs[0], x=0.25, y=0.9, s=year))

                if all(np.isnan(out_x.vmean)) and all(np.isnan(out_y.vmean)):
                    continue
                
                data_x = getattr(out_x, bin_attr)
                data_y = getattr(out_y, bin_attr)

                axs[0].plot(out_x.xintm, data_x, zorder=1, color='black', lw=0.5)
                axs[0].scatter(out_x.xintm, data_x,  # plot across longitude
                            c=data_x, cmap=cmap, norm=norm)
                axs[0].set_xlabel('Longitude [deg]')  # ; plt.xlim(out_x.xbmin, out_x.xbmax)
                axs[0].set_ylabel(subs.label())
                axs[0].set_ylim(ymax=ymax, ymin=ymin)

                axs[1].plot(out_y.xintm, data_y, zorder=1, color='black', lw=0.5)
                axs[1].scatter(out_y.xintm, data_y,  # plot across latitude
                            c=data_y, cmap=cmap, norm=norm, zorder=2)
                axs[1].set_xlabel('Latitude [deg]')  # ; plt.xlim(out_y.xbmin, out_y.xbmax)

            cax = fig.add_subplot(outer_grid[-1, :])
            cax.axis('off')
            fig.colorbar(sm(norm=norm, cmap=cmap), aspect=50, ax=cax, orientation='horizontal')

            outer_grid.tight_layout(fig, pad=3)

            fig.show()

            break

    def plot_binned_2d(self, bin_attr='vmean', hide_lats=False, substances=None,
                       projection='moll'): 
        """ 2D plot of binned substance data for all years at once. """
        data = self.df
        substances = self.substances if substances is None else substances

        for subs in (substances if not bin_attr=='vcount' else [dcts.get_subs(col_name='N2O')]): 
            fig, ax = plt.subplots(dpi=300, figsize=(9, 3.5))
            tools.world().boundary.plot(ax=ax, color='black', linewidth=0.3)
            
            ax.set_title(subs.label() if bin_attr != 'vcount' \
                         else 'Distribution of greenhouse gas flask measurements')
            
            cmap = dcts.dict_colors()[bin_attr]
            vmin, vmax = subs.vlims(bin_attr=bin_attr)
            if 'mol' in subs.unit:
                ref_unit = dcts.get_substances(short_name=subs.short_name,
                                            source='Caribic')[0].unit
                vmin = tools.conv_PartsPer_molarity(vmin, ref_unit)
                vmax = tools.conv_PartsPer_molarity(vmax, ref_unit)
            norm = Normalize(vmin, vmax)  # normalise color map to set limits

            if bin_attr=='vcount': 
                cmap.set_under('white')
                bounds=[1, 5, 10, 15, 30, 45, 60]
                norm = BoundaryNorm(bounds, cmap.N)

            bci = bp.Bin_equi2d(-90, 90, self.grid_size,
                                -180, 180, self.grid_size)
            lat = data.geometry.y 
            lon = data.geometry.x
            out = bp.Simple_bin_2d(data[subs.col_name], lat, lon, bci)
            img_data = getattr(out, bin_attr)
            img = ax.imshow(img_data, cmap=cmap, norm=norm, origin='lower',
                            extent=[out.ybmin, out.ybmax, out.xbmin, out.xbmax])
            
            if hide_lats:
                ax.hlines(30, -180, 180, color='k', lw=1, ls='dashed')
                ax.add_patch(Rectangle((-180, -90), 180*2, 90+30, facecolor="grey", alpha=0.25))

            # label outer plot axes
            ax.set_ylabel('Latitude [°N]')
            ax.set_xlabel('Longitude [°E]')
            
            cbar = plt.colorbar(img, ax=ax, extend='neither' if not bin_attr=='vcount' else 'max')
            cbar.ax.set_xlabel(f'[{subs.unit}]' if bin_attr!='vcount' else '[# Points]')

    def yearly_plot_binned_2d(self, bin_attr='vmean', **subs_kwargs):
        # self, subs, single_yr=None, c_pfx=None, years=None, detr=False):
        """ Create a 2D plot of binned mixing ratios for each available year on a grid.

        Parameters:
            self (GlobalData): instance with global dataset
            detr (bool): Plot detrended data
        """
        data = self.df
        substances = [dcts.get_subs(col_name=c) for c in data.columns
                    if c in [s.col_name for s in dcts.get_substances(**subs_kwargs)]
                    and not (c.startswith('d_') or '_d_' in c)]

        for subs in substances:  
            nplots = len(self.years)
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
                data_type = 'measured' if subs.model == 'MSMT' else 'modeled'
                fig.suptitle(f'Global {data_type} mixing ratios of {subs.label()}',
                            fontsize=25)
                plt.subplots_adjust(top=0.96)

            out_list = self.binned_2d(subs)

            cmap = plt.cm.viridis_r  # create colormap
            vmin, vmax = subs.vlims()# vlims.get(substance.short_name)
            if 'mol' in subs.unit:
                ref_unit = dcts.get_substances(short_name=subs.short_name,
                                            source='Caribic')[0].unit
                vmin = tools.conv_PartsPer_molarity(vmin, ref_unit)
                vmax = tools.conv_PartsPer_molarity(vmax, ref_unit)
            norm = Normalize(vmin, vmax)  # normalise color map to set limits
            world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))

            for i, (out, ax, year) in enumerate(zip(out_list, grid, self.years)):
                world.boundary.plot(ax=ax, color='black', linewidth=0.3)
                
                img_data = getattr(out, bin_attr)
                img = ax.imshow(img_data.T, cmap=cmap, norm=norm, origin='lower',
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
                    cbar.ax.set_xlabel(subs.label(), fontsize=15)

            for i, ax in enumerate(grid):  # hide extra plots
                if i >= nplots:
                    ax.axis('off')

            # cbar = plt.colorbar(img, fig.add_axes([0.05, 0.98, 0.95, 0.05], box_aspect=0.05),
            #                     aspect=10, pad=0.1, orientation='horizontal') # colorbar


            # fig.tight_layout()
            plt.show()

    def mxr_vs_vcoord(self, subs, vcoord, tick_params = {}, ax=None, note=None, **kwargs): 
        """ Plot datapoints on a simple vcoord vs. substance mixing ratio plot. """
        self.data['df']['season'] = tools.make_season(self.data['df'].index.month)
        if ax is None:
            _, ax = plt.subplots(figsize = (6,3), dpi=kwargs.pop('dpi', None))
        for s in set(self.df['season'].tolist()):
            df = self.df[self.df['season'] == s].dropna(subset=[vcoord.col_name, subs.col_name], how='any')
            if len(df) == 0: 
                continue

            x = df[subs.col_name]
            y = df[vcoord.col_name]

            ax.scatter(x, y, 
                        marker='.',
                        label = dcts.dict_season()[f'name_{s}'],
                        c=dcts.dict_season()[f'color_{s}'], 
                        zorder=2, 
                        lw=0.1)

        ax.set_ylabel(vcoord.label())
        if not tick_params.get('bottom') is False: 
            ax.set_xlabel(subs.label())
        
        ax.tick_params(**tick_params)
        if note: 
            ax.text(**dcts.note_dict(ax, y = 0.05, s= note))
            
        ax.grid(True, zorder=0, ls='dashed', alpha=0.5)    
        
    def plot_sf6_detrend_reltp_progression(self):    
        _, ((ax11, ax12), (ax21, ax22)) = plt.subplots(2, 2, dpi=300, figsize=(9,6))
        
        [s11] = self.get_substs(short_name='sf6', detr=False) # dcts.get_subs(col_name='SF6')
        [c11] = self.get_coords(vcoord='pt', tp_def='nan', model='ERA5')
        
        [s12] = self.get_substs(short_name='detr_sf6', detr=True)
        c12 = c11
        
        s22 = s12
        [c22] = self.get_coords(vcoord='pt', rel_to_tp = True, tp_def = 'therm', model='ERA5') 
        # dcts.get_coord(col_name='int_ERA5_D_TROP1_THETA')
        
        self.mxr_vs_vcoord(s11, c11, 
                           ax = ax11, 
                           tick_params = dict(top=True, labeltop=True, bottom=True, labelbottom=True),
                           )
        ax11.set_xlabel(s11.label(), loc='center')
        ax11.set_ylabel(c11.label(), loc='center')
        ax11.set_xlim(5,11.6)
        
        self.mxr_vs_vcoord(s12, c12, 
                    ax=ax12,
                    tick_params  = dict(top=True, labeltop=True, bottom=False, labelbottom=False), 
                    )
        ax12.set_xlabel(s12.label(), loc='center')
        ax12.set_ylabel(c12.label(), loc='center')
        
        
        self.mxr_vs_vcoord(s22, c22, 
                    ax=ax22,
                    tick_params  = dict(top=False, labeltop=False, bottom=True, labelbottom=True),
                    )
        ax22.set_xlabel(s22.label(), loc='center')
        ax22.set_ylabel(c22.label(), loc='center')
        
        plt.subplots_adjust(wspace=0, hspace=0)
        
        ax12.yaxis.tick_right()
        ax12.yaxis.set_label_position("right")
        ax22.yaxis.tick_right()
        ax22.yaxis.set_label_position("right")
        
        ax21.set_visible(False)
        
        plt.show()

    def make_3d_scatter(self, vcoord, color_var, eql=False, **plot_kwargs): 
        """ Plot the given substance on a 3D plot of lon-lat-vcoord. """
               
        z_name = vcoord.col_name 
        c_name = color_var.col_name
        
        df = self.df.dropna(subset = [c_name])
        
        c_label = '$' + ''.join([i for i in color_var.label() if i != '$'])  +'$'
        z_label = '$' + ''.join([i for i in vcoord.label() if i != '$']) + '$'
        
        c_label = fr'{c_label}'
        z_label = fr'{z_label}'
        
        labels = {
            'x' : dcts.get_coord('geometry.x').label(),
            'y' : dcts.get_coord('geometry.y').label(),
            z_name : z_label,
            c_name : c_label,
            }
        print(labels)

        y = df.geometry.y
        x = df.geometry.x
        if eql:
            [eql_coord] = self.get_coords(hcoord = 'eql')
            x = df[eql_coord.col_name]
            
        fig = px.scatter_3d(df, x=x, y=y,
                            z=z_name,
                            color = c_name,
                            # labels = labels,
                            **plot_kwargs,
                            )
        fig.update_traces(marker_size=2.5)

        # Add the world outline
        map_traces = []
        for _, row in tools.world().iterrows():
            if row.geometry.geom_type == 'Polygon':
                x, y = row.geometry.exterior.xy
                x = np.array(x)
                y = np.array(y) 
                z = [df[vcoord.col_name].min()] * len(x)
                map_traces.append(go.Scatter3d(x=x, y=y, z=z, mode='lines', line=dict(color='black', width=1)))
            elif row.geometry.geom_type == 'MultiPolygon':
                for poly in row.geometry.geoms:
                    x,y = poly.exterior.xy
                    x = np.array(x)
                    y = np.array(y)
                    z = [df[vcoord.col_name].min()] * len(x)
                    map_traces.append(go.Scatter3d(x=x, y=y, z=z, mode='lines', line=dict(color='black', width=1)))

        for trace in map_traces:
            fig.add_trace(trace)
            
        fig.update_scenes(
            aspectmode='data',
            camera=dict(
                eye = dict(x=0, y=-0.75, z=1.25))
        )
        fig.update_layout(showlegend=False) 
        # fig.show()
        return fig

# Mozart
def mozart_lonlat_1d(mzt_obj, subs='sf6',
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

    # plot_sf6_detrend_reltp_progression(caribic, bp1)

    # =============================================================================
    # notes = ['$\Theta$ vs. SF$_6$',
    #          '$\Theta$ vs. SF$_6$',
    #          '$\Delta\Theta_{TP}$ vs. detrended SF$_6$']
    # for note in notes: 
    #     fig = plt.figure()
    #     plt.text(**dcts.note_dict(fig, y = 0.05, s= note))
    # =============================================================================

# %% LocalData

def local_timeseries(loc_obj, greyscale=False, v_limits=(None, None), **subs_kwargs):
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
                  if s.col_name in loc_obj.df.columns and not s.short_name.startswith('d_')]

    for subs in substances:
        if all(isinstance(i, (int, float)) for i in v_limits):
            vmin, vmax = v_limits
        else:
            vmin, vmax = subs.vlims() # default values
        norm = Normalize(vmin, vmax)

        # Plot all available info on one graph
        fig, ax = plt.subplots(figsize=(5, 3.5), dpi=250)
        # Measurement data (monthly)
        plt.scatter(loc_obj.df.index, loc_obj.df[subs.col_name], c=loc_obj.df[subs.col_name],
                    cmap=colors['msmts'], norm=norm, marker='+', zorder=1,
                    label=subs.label(name_only=True))
        # Daily mean
        if hasattr(loc_obj, 'df_Day'):
            if not loc_obj.df_Day.empty:  # check if there is data in the daily df
                plt.scatter(loc_obj.df_Day.index, loc_obj.df_Day[subs.col_name],
                            color='silver', marker='+', zorder=0,
                            label=f'{subs.label(name_only=True)} (D)')
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
                                  label=f'{subs.label(name_only=True)} (M)')

        plt.ylabel(subs.label())
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
