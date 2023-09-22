# -*- coding: utf-8 -*-
"""
Line plots of Caribic, Mauna Loa, Mace Head datapoints. Data and monthly mean.

@Author: Sophie Bauchinger, IAU
@Date: Mon Feb 13 11:51:02 2023

"""
import numpy as np
from calendar import monthrange
import datetime as dt
import geopandas
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable as sm

from tpt_data import caribic_gdf, mlo_data, mhd_data, mozart_data, monthly_mean

#%% Mauna Loa & Mace Head SF6
def plot_mlo_mhd_2012():
    mlo_2012 = mlo_data(year = 2012)
    mhd_2012 = mhd_data()

    # Include monthly mean
    mlo_monthly = monthly_mean(mlo_2012)
    mhd_monthly = monthly_mean(mhd_2012)

    mlo_file_MM = r'C:\Users\sophie_bauchinger\toolpac tutorial\mlo_SF6_MM.dat'
    mlo_2012_MM = mlo_data(mlo_file_MM, 2012)

    fig, ax = plt.subplots(dpi=250)
    # plt.scatter(mhd_2012.index, mhd_2012['SF6[ppt]'],
    #             color='grey', label='Mace Head', marker='+')
    plt.scatter(mlo_2012.index, mlo_2012['SF6catsMLOm'],
                color='silver', label='Mauna Lao', marker='+')

    plt.plot(mlo_2012_MM.index, mlo_2012_MM['SF6catsMLOm'], label='line plot MM')
    plt.plot(mlo_monthly.index, mlo_monthly['SF6catsMLOm'], label='line calc MM')

    for i, mean in enumerate(mlo_monthly['SF6catsMLOm']): # plot MLO mean
        y, m = mlo_monthly.index[i].year, mlo_monthly.index[i].month
        xmin = dt.datetime(y, m, 1)
        xmax = dt.datetime(y, m, monthrange(y, m)[1])
        ax.hlines(mean, xmin, xmax, color='black', linestyle='dashed', zorder=2)

    # for i, mean in enumerate(mhd_monthly['SF6[ppt]']): # plot MHD mean
    #     y, m = mhd_monthly.index[i].year, mhd_monthly.index[i].month
    #     xmin = dt.datetime(y, m, 1)
    #     xmax = dt.datetime(y, m, monthrange(y, m)[1])
    #     ax.hlines(mean, xmin, xmax, color='black', linestyle='dashed', zorder=2)

    # plt.plot(mlo_2012_MM.index, mlo_2012_MM['SF6catsMLOm'], 'red', zorder=1,
    #          linestyle='dashed',  label='Mauna Lao, MM')
    plt.title('Ground-based SF$_6$ measurements 2012')
    plt.ylabel('Measured SF$_6$ mixing ratio [ppt]')
    plt.xlabel('Measurement time')
    plt.legend()
    plt.show()

mlo_mhd_2012 = plot_mlo_mhd_2012()

#%% Caribic SF6 1d
def plot_caribic(years, v_limits=None):
    """ Plot msmts and monthly mean for specified years [list] """
    gdfs = []
    for y in years:
        gdfs.append(caribic_gdf(y))

    # Plot SF6 mixing ratio msmts and monthly mean
    fig, ax = plt.subplots(dpi=250)
    plt.title('CARIBIC SF$_6$ measurements')
    ymin = min([gdf['SF6; SF6 mixing ratio; [ppt]\n'].min() for gdf in gdfs])
    ymax = max([gdf['SF6; SF6 mixing ratio; [ppt]\n'].max() for gdf in gdfs])

    cmap = plt.cm.viridis_r
    extend = 'neither'
    if v_limits: vmin = v_limits[0]; vmax = v_limits[1]; extend = 'both'
    else: vmin = ymin; vmax = ymax
    norm = Normalize(vmin, vmax)

    for gdf, y in zip(gdfs, years):
        gdf_MM = monthly_mean(gdf)
        plt.scatter(gdf.index, gdf['SF6; SF6 mixing ratio; [ppt]\n'],
                    label=f'SF$_6$ {y}', marker='x', zorder=1,
                    c = gdf['SF6; SF6 mixing ratio; [ppt]\n'],
                    cmap = cmap, norm = norm)

        for i, mean in enumerate(gdf_MM['SF6; SF6 mixing ratio; [ppt]\n']):
            y, m = gdf_MM.index[i].year, gdf_MM.index[i].month
            xmin = dt.datetime(y, m, 1)
            xmax = dt.datetime(y, m, monthrange(y, m)[1])
            ax.hlines(mean, xmin, xmax, color='black',
                      linestyle='dashed', zorder=2)
    plt.colorbar(sm(norm=norm, cmap=cmap), aspect=50, ax = ax, extend=extend)

    plt.ylabel('SF$_6$ mixing ratio [ppt]')
    plt.ylim(ymin-0.15, ymax+0.15)
    fig.autofmt_xdate()
    plt.show()

c_2008 = plot_caribic([2008])
c_2012 = plot_caribic([2012])
c_2008_2015 = plot_caribic([i for i in range(2011, 2017)])

#%% MOZART
def plot_mozart(year=2007, level=27):
    ds, df = mozart_data(year, level)

    lon_values = [10, 60, 120, 180]
    lat_values = [70, 30, 0, -30, -70]

    # plot change over lat, lon with fixed lon / lat
    fig, (ax1, ax2) = plt.subplots(dpi=250, ncols=2, figsize=(9,5), sharey=True)
    fig.suptitle('MOZART SF$_6$ at fixed longitudes / latitudes', size=17)
    ds.SF6.sel(longitude=lon_values, method='nearest').plot.line(x = 'latitude', ax=ax1)
    ds.SF6.sel(latitude=lat_values, method="nearest").plot.line(x = 'longitude', ax=ax2) # ax = ax
    ax1.set_title(''); ax2.set_title('')
    ax2.set_ylabel('')
    plt.show()

    # remap longitude range from 0->356 to -178->178
    x = np.array(ds.latitude)
    y = np.array([i for i in ds.longitude if i<=178] + [i - 356 for i in ds.longitude if i>178])
    ds['longitude'] = y
    ds = ds.sortby(ds.longitude)
    data = ds.SF6

    ybmin, ybmax, xbmin, xbmax = min(y), max(y), min(x), max(x)

    # Show
    cmap = plt.cm.viridis_r
    fig, ax = plt.subplots(dpi=250, figsize=(8,4), sharex=True)
    ax.set_aspect('equal')
    world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
    world.boundary.plot(ax=ax, color='black', linewidth=0.3)
    data.plot(cmap=cmap, ax=ax)
    cmap = ax.imshow(data, cmap = 'viridis_r', interpolation='nearest',
                      origin='lower', extent=[ybmin, ybmax, xbmin, xbmax])
    ax.set_title('MOZART SF$_6$ concentration {}, level: {:.2}'.format(year, float(ds.level)))
    plt.xlabel('Longitude  [deg]'); plt.xlim(-180,180)
    plt.ylabel('Latitude [deg]'); plt.ylim(-60,100)
    plt.show()

plot_mozart()
