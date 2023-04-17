# -*- coding: utf-8 -*-

# from mpl_toolkits.basemap import basemap
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import cartopy.feature as cfeature
import pandas as pd
import numpy as np
import colorcet as cc
import matplotlib
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap
from pathlib import Path

import C_tools

from toolpac.calc import bin_1d_2d

matplotlib.rcParams['backend'] = "qt5agg"


# %%
def plot_timeseries(yvar, Fdata, df_flights, df_route_colors, flight_numbers, color='flight'):
    plot_scatter_samples(df_flights, Fdata, 'year_frac', yvar, df_route_colors, flight_numbers, color=color)


# %%
def plot_trajs(TrajData, sample_no=None, z_col=None):
    if sample_no is None:
        sample_no_str = list(TrajData.keys())
        sample_no_str.remove('headers')
    else:
        if type(sample_no) is int:
            sample_no = [sample_no]
        sample_no_str = [str(x).zfill(2) for x in sample_no]

    lonmin, lonmax, latmin, latmax = (-150, 150, -40, 80)
    fig, ax = create_map(lonmin, lonmax, latmin, latmax)
    ax.set_aspect('equal')
    ytickloc, labels = plt.yticks()
    xtickloc, labels = plt.xticks()

    minx, maxx, miny, maxy=(180, -180, 90, -90)

    colscale=False
    col_no=10
    colors = plt.cm.rainbow(np.linspace(0, 1, col_no))
    cmap = plt.cm.rainbow
    for i, x in enumerate(sample_no_str):
        tmp = TrajData[x]
        for y in tmp:
            if np.min(y['lon']) < minx:
                minx = np.min(y['lon'])
            if np.max(y['lon']) > maxx:
                maxx = np.max(y['lon'])
            if np.min(y['lat']) < miny:
                miny = np.min(y['lat'])
            if np.max(y['lat']) > maxy:
                maxy = np.max(y['lat'])

            if z_col is None:
                ax.plot(y['lon'], y['lat'], linewidth=0.3, color=colors[i % col_no])
            else:
                if z_col not in y.keys():
                    print('Column for color code is not in dataframe. Using default.')
                    ax.plot(y['lon'], y['lat'], linewidth=0.3, color=colors[i % col_no])
                else:
                    if z_col == 'p':
                        im = ax.scatter(y['lon'], y['lat'], c=y[z_col], marker=',', s=0.3, cmap='jet', vmin=150,
                                        vmax=850, linestyle='solid', transform=ccrs.PlateCarree(), zorder=3)
                        colscale = True
                    # for logarithmic colorscale use option norm=matplotlib.colors.LogNorm()
                    else:
                        im = ax.scatter(y['lon'], y['lat'], s=0.5, c=y[z_col], cmap=cmap, zorder=3,
                                        linestyle='solid', linewidth=0.3,  marker=",", transform=ccrs.PlateCarree())
                        colscale = True

        ax.scatter((TrajData['headers'][i]['lon']).tolist()[0]/10., (TrajData['headers'][i]['lat']).tolist()[0]/10.,
                   marker='o', c='black', s=8, zorder=3)
    # colorbar only for last sample
    if colscale:
        cb = plt.colorbar(im, shrink=0.45, orientation='horizontal')
        cb.set_label('pressure [hPa]')

    ax.set_extent([np.floor(minx-2), np.ceil(maxx+2), np.floor(miny-2), np.ceil(maxy+2)], ccrs.PlateCarree())
    fig.tight_layout()
    ax.set_aspect('auto')


# %%
def plot_scatter_samples(df_flights, Fdata, xvar, yvar, df_route_colors, flight, route=None,
                         # to plot map use
                         # # plot_scatter_samples(df_flights, Fdata, 'lon' ,'lat', df_route_colors, flight_numbers)
                         # flight can be an integer or list of integers
                         # route can be a string or list of strings
                         select_var=None, select_value=None, select_cf=None,
                         # select data according to column select by select value
                         # select_cf can be LT, LE, GT, GE, EQ, case insensitive
                         # selection with 'EQ' works for boolean variables
                         # strato and tropo are True /False, select_cf does not work
                         color=None, cmap=None, df_return=False, plot_return=False, add_all=False,
                         line_11=False):

    plot_map = False
    if xvar == 'lon' and yvar == 'lat':
        plot_map = True

    check_res = C_tools.check_select(select_var, select_value, select_cf)
    if check_res[0] is False:
        return None
  
    select_var = check_res[1]
    select_value = check_res[2]
    select_cf = check_res[3]

    if color is None:
        color_in = color
    else:
        color_in = color.lower()  # value of color might be changed later
        
    flight_list = C_tools.make_flight_list(df_flights, flight, route)

    if plot_map:
        lonmin, lonmax, latmin, latmax = (-150, 150, -40, 80)
        fig, ax = create_map(lonmin, lonmax, latmin, latmax)
        ax.set_aspect('equal')
        ytickloc, labels = plt.yticks()
        xtickloc, labels = plt.xticks()
    else:
        fig = plt.figure(figsize=(9, 5))
        ax = plt.subplot(111)

    df_all = pd.DataFrame(columns=[xvar, yvar])
    if add_all:
        df_all = C_tools.extract_data(df_flights, Fdata, [xvar, yvar], flight=flight_list)

    print('Coloring by ', color_in, '- default is by flight route.')
    # extract plot data
    if color_in in [None, 'route', 'flight']:
        df_plot = pd.DataFrame(columns=[xvar, yvar])
        collist = []
        for i in range(len(flight_list)):  # loop needed because array length per flight not known beforehand
            df = C_tools.extract_data(df_flights, Fdata, [xvar, yvar], flight=flight_list[i],
                                      select_var=select_var, select_value=select_value, select_cf=select_cf)
            if color_in == 'flight':
                color = flight_list[i]  # (df_flights.index)[i]
            else:
                color = df_route_colors['color'][df_flights['route'][flight_list[i]]]
                
            if df is not None:
                collist.extend([color] * len(df))
                df_plot = df_plot.append(df)
        
        df_plot['color'] = collist
    else:
        if color_in == xvar or color_in == yvar:
            df_plot = C_tools.extract_data(df_flights, Fdata, [xvar, yvar], flight=flight_list,
                                           select_var=select_var, select_value=select_value, select_cf=select_cf)
            df_plot['color'] = df_plot[color_in]
        else:
            df_plot = C_tools.extract_data(df_flights, Fdata, [xvar, yvar, color_in], flight=flight_list,
                                           select_var=select_var, select_value=select_value, select_cf=select_cf)
            df_plot.rename(columns={color_in: 'color'}, inplace=True)

    print(len(df_plot), 'samples')

    # plot data 
    if not cmap: # not prescribed in function call
        cmap = cc.cm.rainbow  # 'rainbow'
        if color_in == 'season':
            cmap = ListedColormap(['orange', 'green', 'red', 'blue'])
        elif color_in == 'month':
            cmap = plt.get_cmap('jet', 12)
        # elif color_in in []:  # reverse colormap
        #   cmap = 'rainbow_r'

    vmin = None
    vmax = None

    if color_in == 'season':
        vmin = 1
        vmax = 4
    elif color_in == 'month':
        vmin = 1
        vmax = 12
    elif color_in == 'int_pv':
        vmin = 0
    elif color_in == 'flight':
        vmin = min(flight)
        vmax = max(flight)

    if plot_map:
        if add_all:
            ax.scatter(df_all['lon'], df_all['lat'], s=5., marker='o', c='darkgrey',
                       transform=ccrs.PlateCarree(), zorder=3)
        im = ax.scatter(df_plot['lon'], df_plot['lat'],
                        c=df_plot['color'], cmap=cmap, vmin=vmin, vmax=vmax,
                        transform=ccrs.PlateCarree(), zorder=3)
    else:
        im = ax.scatter(df_plot[xvar], df_plot[yvar], zorder=3,
                        c=df_plot['color'], cmap=cmap, vmin=vmin, vmax=vmax)

    if select_var:
            ax.set_title(select_var+select_value+select_cf, fontsize=10)
    if plot_map:
        plt.subplots_adjust(left=0.05, bottom=0.01, right=0.99, top=0.95, wspace=0, hspace=0)
    else:
        plt.subplots_adjust(left=0.08, bottom=0.1, right=0.98, top=0.95, wspace=0, hspace=0)
        plt.xlabel(xvar)
        plt.ylabel(yvar)

    # minimize space around axes, does not work if colorbar is included
    # plt.tight_layout(pad=0)
    
    # add legend for coloring by flight number
    if color_in in ['flight', 'route']:
        plt.subplots_adjust(left=0.1, bottom=0.15, right=0.85, top=0.95, wspace=0, hspace=0)
        create_legend_for_flight_coloring(df_plot, cmap, df_route_colors)

    # if color_in not in [None, 'route']:
    if color_in not in [None, 'route', 'flight']:
        if color_in == 'season':
            cbar = fig.colorbar(im, shrink=0.5, ticks=[1.4, 2.125, 2.9, 3.625], orientation='horizontal')
            cbar.ax.set_xticklabels(['Spring', 'Summer', 'Fall', 'Winter'])
        elif color_in == 'month':
            cbar = fig.colorbar(im, shrink=0.45, orientation='horizontal')
            cbar.ax.set_xlabel('month')

        else:
            # cbar = fig.colorbar(im, shrink=0.45, orientation='horizontal')
            if color_in == 'int_pv':
                cbar = fig.colorbar(im, shrink=0.55, orientation='vertical')
            else:
                cbar = fig.colorbar(im, shrink=0.45, orientation='horizontal')
            # cbar_ax = fig_map.add_axes([0.85, 0.15, 0.05, 0.7])
            # cbar = plt.colorbar(im, cax = cbar_ax, location='bottom')
            if color_in.startswith('cfc') or color_in.startswith('hfc') or color_in.startswith('hcfc'):
                color_in2label = (color_in.replace('_', '-')).upper()
                cbar.ax.set_xlabel(f'{color_in2label} [ppt]')
            elif color_in.startswith('int'):
                color_in2label = (color_in.replace('int_', '')).upper()
                if color_in == 'int_pv':
                    cbar.ax.set_xlabel(f'{color_in2label} [PVU]')
                else:
                    cbar.ax.set_xlabel(f'{color_in2label} [ppt]')
            else:
                color_in2label = (color_in.replace('_', ' ')).capitalize()
                cbar.ax.set_xlabel(f'{color_in2label} [ppt]')

    if yvar in ['p']:
        plt.gca().invert_yaxis()    # reverse vertical axis

    if line_11:
        xy_min = min(ax.get_xlim()[0], ax.get_ylim()[0])
        xy_max = max(ax.get_xlim()[1], ax.get_ylim()[1])
        print(xy_min,xy_max)
        ax.plot((xy_min, xy_max), (xy_min, xy_max), color='gray', linewidth=2, linestyle='dashed')

    plt.show()
    if df_return:
        return df_plot
    if plot_return:
        return fig


# %%
def flight_overview(Fdata, flight, df_flights, alt_var='pressure', alt_var_MS=None, alt_var_WAS=None,
                    yvar_MS='Ozone',
                    yvar_INT='int_o3',
                    yvar1_WAS='Halon_1211', yvar2_WAS='Dichloromethane',
                    fig_return=False):

    print(yvar1_WAS, yvar2_WAS)

    # name of column time in data
    time_var = 'timecref' 
    
    if alt_var is not None:
        if alt_var_MS is not None:
            alt_var = alt_var_MS
        if alt_var == 'pressure':
            alt_var_MS = 'pstatic'
            alt_var_WAS = 'p'
        elif alt_var == 'tpot':
            alt_var_MS = 'tpot'
            alt_var_WAS = 'int_tpot'
        elif alt_var == 'h_rel_tp':
            alt_var_MS = 'h_rel_tp'
            alt_var_WAS = 'int_h_rel_tp'
        elif alt_var_MS is None:
            print('Altitude variable should be pressure, tpot or h_rel_tp.')
            print('For other options define alt_var_MS and alt_var_WAS manually.')
            return None
        
    fig = plt.figure(figsize=(11, 5))
    ax_l = plt.subplot(111)
    ax_l.set_xlabel('UTC [sec]')

    fig.subplots_adjust(left=0.15, right=0.85)
    
    color1 = 'grey'
    color1_2 = 'darkgrey'

    color2 = 'lightgreen'
    color2_2 = 'green'

    color3 = 'red'
    color3_2 = 'darkred'

    color4 = 'deepskyblue'
    color4_2 = 'blue'
    
    # create 4 axes
    
    if alt_var_MS is not None:
        alt_var_MS = alt_var_MS.lower()    
        legend_labels = [alt_var]
        handles = [(Line2D([0], [0], color=color1, marker='o', linestyle='-',
                    markerfacecolor=color1_2, markeredgecolor=color1))]
        
        ax_l.spines['left'].set_position(('axes', 0))
        ax_l.spines['left'].set_color(color1)   
        ax_l.set_ylabel(alt_var_MS, color=color1)
        ax_l.tick_params(colors=color1, direction='in')
        
        if alt_var == 'pressure':
            plt.gca().invert_yaxis()

    if alt_var_WAS is not None:
        alt_var_WAS = alt_var_WAS.lower()

    if yvar_MS is not None:
        yvar_MS = yvar_MS.lower()
        legend_labels.append(yvar_MS)
        handles.append(Line2D([0], [0], color=color2, marker='o', linestyle='-',
                       markerfacecolor=color2_2, markeredgecolor=color2))

        ax_r = ax_l.twinx()
        ax_r.spines['right'].set_position(('axes', 1.0))
        ax_r.set_ylabel(yvar_MS, color=color2_2)
        ax_r.spines['right'].set_color(color2_2)   
        ax_r.tick_params(colors=color2_2, direction='in')
    else:
        if yvar_INT is not None:
            yvar_INT = None
            print('yvar_INT set to none because argument yvar_MS was empty.')

    if yvar_INT is not None:
        yvar_INT = yvar_INT.lower()

    if yvar1_WAS is not None:
        yvar1_WAS = yvar1_WAS.lower()        
        legend_labels.append(yvar1_WAS)
        handles.append(Line2D([0], [0], color=color3, marker='D', linestyle=':',
                              markerfacecolor=color3, markeredgecolor=color3_2))

        ax_r2 = ax_l.twinx()
        ax_r2.spines['right'].set_position(('axes', 1.08))
        ax_r2.spines['right'].set_color(color3)   
        ax_r2.set_ylabel(yvar1_WAS, color=color3)
        ax_r2.tick_params(colors=color3, direction='in')

    if yvar2_WAS is not None:
        yvar2_WAS = yvar2_WAS.lower()
        legend_labels.append(yvar2_WAS)
        handles.append(Line2D([0], [0], color=color4, marker='D', linestyle=':',
                              markerfacecolor=color4, markeredgecolor=color4_2))

        ax_l2 = ax_l.twinx()
        ax_l2.spines['left'].set_position(('axes', -0.1))
        ax_l2.spines['left'].set_color(color4)   
        ax_l2.yaxis.set_label_position('left')
        ax_l2.yaxis.set_ticks_position('left')
        ax_l2.set_ylabel(yvar2_WAS, color=color4)
        ax_l2.tick_params(colors=color4, direction='in')
    # end of axes definition
    
    # check data content and plot
    df_MS = Fdata['F'+str(flight)+'_MS']
    df_MS.columns = map(str.lower, df_MS.columns)
    MS_time = df_MS[time_var].values.tolist()
    ms_cols = df_MS.columns.tolist()

    df_INT = Fdata['F'+str(flight)+'_INT']
    df_INT.columns = map(str.lower, df_INT.columns)
    int_cols = df_INT.columns.tolist()
    WAS_time = df_INT[time_var].values.tolist()

    df_GHG = Fdata['F' + str(flight) + '_GHG']
    if df_GHG is not None:
        df_GHG.columns = map(str.lower, df_GHG.columns)
        ghg_cols = df_GHG.columns.tolist()
    else:
        ghg_cols = [None]

    df_HCF = Fdata['F'+str(flight)+'_HCF']
    if df_HCF is not None:
        df_HCF.columns = map(str.lower, df_HCF.columns)
        hcf_cols = df_HCF.columns.tolist()
    else:
        hcf_cols = [None]

    df_HFO = Fdata['F'+str(flight)+'_HFO']
    if df_HFO is not None:
        df_HFO.columns = map(str.lower, df_HFO.columns)
        hfo_cols = df_HFO.columns.tolist()
    else:
        hfo_cols = [None]

    if any(alt_var_MS == s for s in ms_cols):
        MS_alt = df_MS[alt_var_MS].values.tolist()
    else:
        print('alt_var_MS not found in df_MS.')
        print(ms_cols)
        return None
    
    ax_l.plot(MS_time, MS_alt, color=color1, zorder=-1)

    if any(alt_var_WAS == s for s in int_cols):
        WAS_alt = df_INT[alt_var_WAS].values.tolist()
        ax_l.scatter(WAS_time, WAS_alt, color=color1_2, edgecolors=color1)
    else:
        print('skipping alt_var_WAS ', alt_var_WAS, ' not found in df_INT.')
        print(int_cols)

    if yvar_MS is not None:
        if any(yvar_MS == s for s in ms_cols):
            MS_yvar = df_MS[yvar_MS].values.tolist()
            ax_r.plot(MS_time, MS_yvar, color=color2, zorder=-1)
        else:
            print('yvar_MS not found in df_MS.')
            print(ms_cols)
            return None

    if yvar_INT is not None:
        if any(yvar_INT == s for s in int_cols):
            WAS_yvar = df_INT[yvar_INT].values.tolist()
            ax_r.scatter(WAS_time, WAS_yvar, color=color2, edgecolors=color2_2)
        else:
            print('skipping yvar_INT ', yvar_INT, ' not found in df_INT.')
            print(int_cols)
    
    if yvar1_WAS is not None:
        if any(yvar1_WAS == s for s in int_cols):
            WAS_yvar1 = df_INT[yvar1_WAS].values.tolist()
        elif any(yvar1_WAS == s for s in ghg_cols):
            WAS_yvar1 = df_GHG[yvar1_WAS].values.tolist()
        elif any(yvar1_WAS == s for s in hcf_cols):
            WAS_yvar1 = df_HCF[yvar1_WAS].values.tolist()        
        elif any(yvar1_WAS == s for s in hfo_cols):
            WAS_yvar1 = df_HFO[yvar1_WAS].values.tolist()
        else:
            print(yvar1_WAS, 'not found in any dataframe (yvar1_WAS).')

        ax_r2.scatter(WAS_time, WAS_yvar1, color=color3, marker="D", edgecolors=color3_2)
        ax_r2.plot(WAS_time, WAS_yvar1, color=color3, linestyle=':', zorder=-1)
        
    if yvar2_WAS is not None:
        if any(yvar2_WAS == s for s in int_cols):
            WAS_yvar2 = df_INT[yvar2_WAS].values.tolist()
        elif any(yvar2_WAS == s for s in ghg_cols):
            WAS_yvar2 = df_GHG[yvar2_WAS].values.tolist()
        elif any(yvar2_WAS == s for s in hcf_cols):
            print()
            WAS_yvar2 = df_HCF[yvar2_WAS].values.tolist()
        elif any(yvar2_WAS == s for s in hfo_cols):
            WAS_yvar2 = df_HFO[yvar2_WAS].values.tolist()
        else:
            print(yvar2_WAS, 'not found in any dataframe (yvar2_WAS).')

        ax_l2.scatter(WAS_time, WAS_yvar2, color=color4, marker="D", edgecolors=color4_2)
        ax_l2.plot(WAS_time, WAS_yvar2, color=color4, linestyle=':', zorder=-1)

    # some plot fine tuning
    ax_l.xaxis.label.set_color('black')
    ax_l.tick_params(axis='x', colors='black')
    ax_l.xaxis.label.set_color('black')

    plt.title('Flight '+str(flight)+','+str(df_flights.date[flight]), x=0)

    plt.legend(handles, legend_labels,
               loc='center', bbox_to_anchor=(.5, 1.05), ncol=len(legend_labels))
    
    plt.show()
    if fig_return:
        return fig, ax_l, ax_l2, ax_r, ax_r2, handles, legend_labels


# %%
def plot_histogram(df_flights, Fdata, var, binwidth, label='', swap=False, y_flip=False, flight=None, route=None):
    # swap switches x,y axis of plot
    var = var.lower()
    if label == '':
        label = var.title()

    flight_list = C_tools.make_flight_list(df_flights, flight, route)

    df_data = C_tools.extract_data(df_flights, Fdata, var, flight=flight_list)

    data_max = np.ceil(df_data[var].max())
    data_min = np.floor(df_data[var].min())
    bins = np.ceil((data_max-data_min)/binwidth)
    # hist = np.histogram(df_data[var], bins=int(bins))

    fig, ax = plt.subplots(figsize=(6, 4))
    if swap:
        n, bin_var, patches = ax.hist(df_data[var], bins=int(bins),
                                      color='#0504aa', rwidth=0.9, orientation=u'horizontal')
        plt.xlabel('Frequency')
        plt.ylabel(label)
    else:
        n, bin_var, patches = ax.hist(df_data[var], bins=int(bins), color='#0504aa', rwidth=0.9)
        plt.xlabel(label)
        plt.ylabel('Frequency')
    if y_flip:
        ax.invert_yaxis()

    # maxfreq = n.max()
    # Set a clean upper y-axis limit.
    # plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)


# %%
def pl_gradient_by_season(subst, Fdata, df_flights, tp='therm',
                          # tp can be 'therm' or 'dyn'
                          ref_path=Path(r'D:\Daten_andere\NOAA'),
                          ycoord='pt',
                          # ycoord can be 'z','pt','dp'
                          min_y=-50, max_y=80, bsize=10, pointsmin=5,
                          select_var=None, select_value=None, select_cf=None,
                          # select data according to column select by select value
                          # select_cf can be LT,LE, GT,GE, EQ, case insensitive)
                          flight=None, route=None):
    check_res = C_tools.check_select(select_var, select_value, select_cf)
    if check_res[0] is False:
        return None

    # y variable name for chosen tropopause definition
    if tp == 'therm':
        tpdef = 's'
    elif tp == 'dyn':
        tpdef = 'd'

    if ycoord == 'pt':
        unit = 'k'
        yvar = f'int_{ycoord}_rel_{tpdef}tp_{unit}'
    elif ycoord == 'dp':
        unit = 'hpa'
        yvar = f'int_{ycoord}_{tpdef}trop_{unit}'
    elif ycoord == 'z':
        unit = 'km'
        yvar = f'int_{ycoord}_rel_{tpdef}tp_{unit}'

    select_var = check_res[1]
    select_value = check_res[2]
    select_cf = check_res[3]

    flight_list = C_tools.make_flight_list(df_flights, flight, route)
    df_data = C_tools.extract_data(df_flights, Fdata, [subst, yvar, 'season', 'year_frac'], flight_list,
                                       select_var=select_var, select_value=select_value, select_cf=select_cf)

    df_nonan = df_data[df_data[subst].notnull()].copy()
    df_nonan.reset_index(drop=True, inplace=True)

    c_obs = df_nonan[subst].values
    t_obs = df_nonan['year_frac'].values

    # will only work for few substances,
    # e.g. N2O, SF6, Halon-1211, maybe CFC-11 and CFC-12
    df_nonan[f'detr_{subst}'] = C_tools.detrend_subst(t_obs, c_obs, subst, ref_path)

    nbins = (max_y - min_y) / bsize
    y_array = min_y + np.arange(nbins) * bsize + bsize * 0.5

    dict_season = {'name_1': 'spring', 'name_2': 'summer', 'name_3': 'autumn', 'name_4': 'winter',
                   'color_1': 'blue', 'color_2': 'orange', 'color_3': 'green', 'color_4': 'red'
                   }

    # iterate through seasons present in data and do 1d binning
    for s in set(df_nonan['season'].tolist()):
        df_sub = df_nonan.loc[df_nonan['season'] == s]
        y_values = df_sub[yvar].values
        x_values = df_sub[f'detr_{subst}'].values
        dict_season[f'bin1d_{s}'] = bin_1d_2d.bin_1d(x_values, y_values, min_y, max_y, bsize)

    # plot gradients in all seasons
    fig = plt.figure()

    x_min = np.nan
    x_max = np.nan
    for s in set(df_nonan['season'].tolist()):
        vmean = (dict_season[f'bin1d_{s}']).vmean
        vcount = (dict_season[f'bin1d_{s}']).vcount
        vmean = np.array([vmean[i] if vcount[i] >= pointsmin else np.nan for i in range(len(vmean))])
        vstdv = (dict_season[f'bin1d_{s}']).vstdv

        # find value range for axis limits
        all_vmin = np.nanmin((dict_season[f'bin1d_{s}']).vmin)
        all_vmax = np.nanmax((dict_season[f'bin1d_{s}']).vmax)
        x_min = np.nanmin((x_min, all_vmin))
        x_max = np.nanmax((x_min, all_vmax))

        plt.plot(vmean, y_array, '-',
                 marker='o', c=dict_season[f'color_{s}'], label=dict_season[f'name_{s}'])

#        plt.errorbar(vmean, y_array, None, vstdv, c=dict_season[f'color_{s}'])

    plt.tick_params(direction='in', top=True, right=True)

    plt.ylim([min_y, max_y])
    if ycoord == 'pt':
        plt.ylabel('$\Delta$$\Theta$ [K]')
    elif ycoord == 'dp':
        plt.ylabel('$\Delta$p [hpa]')
        plt.gca().invert_yaxis()
    elif ycoord == 'z':
        plt.ylabel('$\Delta$z [km]')

    x_min = np.floor(x_min)
    x_max = np.ceil(x_max)
    plt.xlim([x_min, x_max])
    # plt.xlabel(f'$\Delta${subst.upper()} [ppt]')
    plt.xlabel(f'{subst.upper()} [ppt] detrended')
    if subst == 'n2o' or subst == 'ch4':
        plt.xlabel(f'{subst.upper()} [ppb] detrended')
    elif subst == 'co2':
        plt.xlabel(f'{subst.upper()} [ppm] detrended')

    plt.legend()
    plt.show()

    return


# %%
def plot_1d(df_flights, Fdata, xvar,
            xbmin, xbmax, xbin,
            yvar=None,
            what='mean',
            flight=None, route=None,
            # flight can be an integer or list of integers
            # route can be a string or list of strings
            select_var=None, select_value=None, select_cf=None,
            # select data according to column select by select value
            # select_cf can be LT,LE, GT,GE, EQ, case insensitive)
            add_data=False):

    check_res = C_tools.check_select(select_var, select_value, select_cf)
    if check_res[0] is False:
        return None
  
    select_var = check_res[1]
    select_value = check_res[2]
    select_cf = check_res[3]
    
    flight_list = C_tools.make_flight_list(df_flights, flight, route)
        
    if yvar is None:
        df_data = C_tools.extract_data(df_flights, Fdata, [xvar], flight=flight_list,
                                       select_var=select_var, select_value=select_value, select_cf=select_cf)
        to_bin_1d = df_data[[xvar, xvar]]
    else:
        df_data = C_tools.extract_data(df_flights, Fdata, [xvar, yvar], flight=flight_list,
                                       select_var=select_var, select_value=select_value, select_cf=select_cf)
        to_bin_1d = df_data[[xvar, yvar]]

    bin1d = C_tools.rebin_data(to_bin_1d, xbmin, xbmax, xbin, bin2d=False)

    plt.figure(figsize=(9, 5))
    ax = plt.subplot(111)

    plt.subplots_adjust(left=0.1, bottom=0.15, right=0.95, top=0.95, wspace=0, hspace=0)
    plt.xlabel(xvar)

    if yvar is None:
        plt.ylabel('number of samples') 
        yerr = None
        V = bin1d.vcount 
        im = ax.plot(bin1d.xmean, V, '-o')
    else:
        plt.ylabel(yvar + ', ' + what)

        if what == 'mean':
            V = bin1d.vmean
            yerr = bin1d.vstdv
        elif what == 'median':
            V = bin1d.vmedian
            yerr = None
        elif what == 'min':
            V = bin1d.vmin
            yerr = None
        elif what == 'max':
            V = bin1d.vmax
            yerr = None
        elif what == 'stdv':
            yerr = None
            V = bin1d.vstdv
        elif what == 'count':
            yerr = None
            V = bin1d.vcount

        if what == 'mean':
            ax.errorbar(bin1d.xmean, V, xerr=bin1d.xstdv, yerr=yerr, marker='o')
        else:
            ax.plot(bin1d.xmean, V, '-o')

        if add_data:
            ax.scatter(df_data[xvar], df_data[yvar], color='lightgrey')

    plt.show()
    return bin1d


# %%
def plot_box_2d(Fdata, df_flights, xvar, yvar,
                xbmin, xbmax, xbin, ybmin, ybmax, ybin,
                zvar=None,
                what='mean',
                flight=None, route=None,
                # flight can be an integer or list of integers
                # route can be a string or list of strings
                select_var=None, select_value=None, select_cf=None):
                # select data according to column select by select value
                # select_cf can be LT,LE, GT,GE, EQ, case insensitive):

    plot_map = False
    if xvar == 'lon' and yvar == 'lat':
        plot_map = True

    check_res = C_tools.check_select(select_var, select_value, select_cf)
    if check_res[0] is False:
        return None
  
    select_var = check_res[1]
    select_value = check_res[2]
    select_cf = check_res[3]
    
    flight_list = C_tools.make_flight_list(df_flights, flight, route)
    print(flight_list)    
    if zvar is None:
        df_data = C_tools.extract_data(df_flights, Fdata, [xvar, yvar], flight=flight_list,
                                       select_var=select_var, select_value=select_value, select_cf=select_cf)
        to_bin_2d = df_data[[xvar, yvar]]
    else:
        df_data = C_tools.extract_data(df_flights, Fdata, [xvar, yvar, zvar], flight=flight_list,
                                       select_var=select_var, select_value=select_value, select_cf=select_cf)
        to_bin_2d = df_data[[xvar, yvar, zvar]]

    print(len(df_data), 'samples')

    bin2d = C_tools.rebin_data(to_bin_2d, xbmin, xbmax, xbin,
                               ybmin=ybmin, ybmax=ybmax, ybin=ybin, bin2d=True)
        
    if plot_map:
        lonmin, lonmax, latmin, latmax = (-180, 180, -90, 90)
        fig, ax = create_map(lonmin, lonmax, latmin, latmax)
        ax.set_aspect('equal')
        ytickloc, labels = plt.yticks()
        xtickloc, labels = plt.xticks()
    else:
        fig = plt.figure(figsize=(6, 5))
        ax = plt.subplot(111)

    if plot_map:
        plt.subplots_adjust(left=0.05, bottom=0.01, right=0.98, top=0.98, wspace=0, hspace=0)
    else:
        plt.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.95, wspace=0, hspace=0)
        plt.xlabel(xvar)
        plt.ylabel(yvar)

    xedges = np.append(bin2d.xint, bin2d.xbmax)
    yedges = np.append(bin2d.yint, bin2d.ybmax)
    X, Y = np.meshgrid(xedges, yedges)
    V = None
        
    if zvar is None:
        cmap = cc.cm.rainbow  # 'rainbow'
        V = bin2d.vcount.T     # T = transpose
        V[V == 0] = None
    else:
        # add keyword for what to plot, min, max, mean, median, stdv
        cmap = cc.cm.rainbow  # 'viridis'
        if what == 'mean':
            V = bin2d.vmean.T
        elif what == 'median':
            V = bin2d.vmedian.T
        elif what == 'min':
            V = bin2d.vmin.T
        elif what == 'max':
            V = bin2d.vmax.T
        elif what == 'stdv':
            V = bin2d.vstdv.T
        elif what == 'count':
            V = bin2d.vcount.T
        
    vmin = None
    vmax = None    
    im = ax.pcolormesh(X, Y, V, cmap=cmap, vmin=vmin, vmax=vmax)

    cbar = fig.colorbar(im, shrink=0.45, orientation='horizontal')

    if zvar is None:
        cbar.ax.set_xlabel('number of samples') 
    else:
        cbar.ax.set_xlabel(zvar+', '+what + ' value') 

    plt.show()
    return fig, ax


# %%
def cf_1d_two_routes(df_flights, Fdata, route1, route2, xvar, xbmin, xbmax, xbin, yvar,
                     select_var=None, select_value=None, select_cf=None):

    bin1d_route1 = plot_1d(df_flights, Fdata, xvar,
                           xbmin, xbmax, xbin,
                           yvar=yvar, what='mean', route=route1, add_data=True,
                           select_var=select_var, select_value=select_value, select_cf=select_cf)
    bin1d_route2 = plot_1d(df_flights, Fdata, xvar,
                           xbmin, xbmax, xbin,
                           yvar=yvar, what='mean', route=route2, add_data=True,
                           select_var=select_var, select_value=select_value, select_cf=select_cf)

    plt.figure(figsize=(9, 5))
    ax = plt.subplot(111)
    plt.subplots_adjust(left=0.1, bottom=0.15, right=0.95, top=0.95, wspace=0, hspace=0)
    plt.xlabel(xvar)
    plt.ylabel(yvar)

    ax.errorbar(bin1d_route1.xmean, bin1d_route1.vmean,
                xerr=bin1d_route1.xstdv, yerr=bin1d_route1.vstdv, marker='o', label=route1)
    ax.errorbar(bin1d_route2.xmean, bin1d_route2.vmean,
                xerr=bin1d_route2.xstdv, yerr=bin1d_route2.vstdv, marker='o', label=route2)
    ax.legend()

    # plt.legend(lines, labels, loc='center right', bbox_to_anchor=(1.15, 0.5))
    plt.show()


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# plotting sub routines start here
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %%
def create_map(lonmin, lonmax, latmin, latmax):
    # mapextent = [lonmin, lonmax, latmin, latmax]
    mapxticks = [-180, -120, -60, 0, 60, 120, 180]
    mapyticks = [-90, -60, -30, 0, 30, 60, 90]
    
    # plot basic map with coastlines
    fig_map = plt.figure(figsize=(9, 5))
    ax = plt.subplot(111, projection=ccrs.PlateCarree())

    # ax.stock_img()
    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.OCEAN)
    # ax.coastlines()
    ax.add_feature(cfeature.COASTLINE)

    size = fig_map.get_size_inches()
    ratio = size[0]/size[1]
    ax.set_aspect(ratio, anchor='N')

    ax.set_xticks(mapxticks, crs=ccrs.PlateCarree())
    ax.set_yticks(mapyticks, crs=ccrs.PlateCarree())
    ax.set_extent([lonmin, lonmax, latmin, latmax], ccrs.PlateCarree())
    # has to be set after xyticks

    return fig_map, ax


# %%
def create_legend_for_flight_coloring(df_plot, colmap, df_route_colors):
    # usable only for scatter plots with color coding by flight number

    # check if coloured by a numeric parameter (flight number)
    if np.issubdtype(df_plot['color'].dtype, np.number):
   
        flights = df_plot.color.unique().tolist()
        print(flights)
        labels = [str(x) for x in flights]
    
        col_min = min(flights)
        col_max = max(flights)  
        
        col_frac = [(x-flights[0])/(col_max - col_min) for x in flights]
        cmap = plt.cm.get_cmap(colmap)
        
        rgba = [cmap(x) for x in col_frac]
        # each element has 4 entries, last one is tranparency and should always be 1 here
        colors = [(x[0], x[1], x[2]) for x in rgba]
    else:  # should be routes
        routes = df_plot.color.unique().tolist()
        colors = [x for x in routes]
        labels = [df_route_colors.index[df_route_colors.color == x][0] for x in colors]

    lines = [Line2D([0], [0], color=c, linewidth=3, linestyle='', marker='o') for c in colors]
    
    # df_route_colors
    
    plt.legend(lines, labels, loc='center right', bbox_to_anchor=(1.15, 0.5))
    plt.show()
