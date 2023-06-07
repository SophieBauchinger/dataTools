# -*- coding: utf-8 -*-
"""
@Author: Sophie Bauchimger, IAU
@Date: Tue Jun  6 13:59:31 2023

Showing mixing ratios per season on a plot of coordinate relative to the 
tropopause (in km or K) versus equivalent latitude (in deg N)
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import pandas as pd

from toolpac.calc import bin_1d_2d
import C_tools
from dictionaries import get_col_name

def plot_eqlat_deltheta(c_obj, c_pfx='INT2', subs='n2o', tp = 'therm', pvu=2.0, x_bin=None, y_bin=None):
    """ 
    Creates plots of equivalent latitude versus potential temperature or height
    difference relative to tropopause (depends on tropopause definition).
    Plots each season separately on a 2x2 grid.

    c_obj (Caribic)
    c_pfx (str): 'INT', 'INT2'
    subs (str): substance to plot. 'n2o', 'ch4', ...
    tp (str): tropopause definition. 'therm', 'dyn', 'z' or 'pvu'
    pvu (float): potential vorticity set as tropopause definition. 1.5, 2.0 or 3.5
    """
    try: data = c_obj.data[c_pfx]
    except: 
        if c_obj.source == 'Mozart': data = c_obj.df
    substance = get_col_name(subs, c_obj.source, c_pfx)

    data['season'] = C_tools.make_season(data.index.month) # 1 = spring etc
    dict_season = {'name_1': 'Spring (MAM)', 'name_2': 'Summer (JJA)', 'name_3': 'Autumn (SON)', 'name_4': 'Winter (DJF)',
                   'color_1': 'blue', 'color_2': 'orange', 'color_3': 'green', 'color_4': 'red'}

    # get coordinates according to fctn variables
    if c_pfx == 'INT2': # co, co2, ch4, n2o
        x_coord = 'int_ERA5_EQLAT [deg N]' # Equivalent latitude (ERA5)
        if tp == 'z':
            y_coord = 'int_CARIBIC2_H_rel_TP [km]' # height relative to the tropopause in km: H_rel_TP; replacement for H_rel_TP
            y_label = '$\Delta$z [km]'
            if not y_bin: y_bin = 0.25 # km
        if tp == 'pvu':
            y_coord = 'int_ERA5_D_{}_{}PVU_BOT [K]'.format(str(pvu)[0], str(pvu)[2]) # pot temp difference to potential vorticity surface
            y_label = f'$\Delta\Theta$ ({pvu} PVU) [K]'
            if not y_bin: y_bin = 5 # K

    elif c_pfx =='INT': # co, co2, ch4
        x_coord = 'int_eqlat [deg]' # equivalent latitude in degrees north from ECMWF
        if tp =='therm': 
            y_coord = 'int_pt_rel_sTP_K [K]' #  potential temperature difference relative to thermal tropopause from ECMWF
            if not y_bin: y_bin = 5 # K
        elif tp == 'dyn': 
            y_coord = 'int_pt_rel_dTP_K [K]' #  potential temperature difference relative to  dynamical (PV=3.5PVU) tropopause from ECMWF
            if not y_bin: y_bin = 10 # K
        y_label = f'$\Delta\Theta$ ({tp}) [K]'

    # set bin sizes and plot limits 
    x_label = 'Eq. lat [°N]'
    if not x_bin: x_bin = 10 # °N

    if c_obj.source == 'Mozart':y_coord = 'PS'; y_label='Surface Pressure [Pa]'; y_bin = 5e3

    y_lims = np.nanmin(data[y_coord]) -y_bin, np.nanmax(data[y_coord])+y_bin
    x_lims = (-90-x_bin, 90+x_bin)

    vmin = np.nanmin(data[substance])
    vmax = np.nanmax(data[substance])

    # create parent plot
    f, axs = plt.subplots(2, 2, dpi=250, figsize=(9,7))
    for s, ax in zip(set(data['season'].tolist()), axs.flat): # flatten axs array
        ax.set_title(dict_season[f'name_{s}'])
        df = data[data['season'] == s]

    # f, axs = plt.subplots(6, 5, dpi=250, figsize=(18,14))
    # for y, ax in zip(c_obj.years, axs.flat): # flatten axs array
        # ax.set_title(y)
        # df = data[data.index.year == y]

        try: x = np.array(df[x_coord]) # °, eq. latitude 
        except: #!!! 
            if c_obj.source == 'Mozart': x = np.array(df.geometry.x); x_label='latitude'
        y = np.array(df[y_coord]) # K, pot temp

        xbmin, xbmax, xbsize = np.nanmin(x), np.nanmax(x), x_bin # °, eq. latitude 
        ybmin, ybmax, ybsize = np.nanmin(y), np.nanmax(y), y_bin # km, height relative to therm. tp

        out = bin_1d_2d.bin_2d(np.array(df[substance]), x, y,
                               xbmin, xbmax, xbsize, ybmin, ybmax, ybsize)

        ax.plot([out.xbmin, out.xbmax], [out.ybmin,out.ybmin])
        ax.plot([out.xbmin, out.xbmin], [out.ybmin, out.ybmax])
        ax.plot([out.xbmin, out.xbmax], [out.ybmax,out.ybmax])
        ax.plot([out.xbmax, out.xbmax], [out.ybmin, out.ybmax])
        print(y, out.xbmin, out.xbmax, out.ybmin, out.ybmax)

        cmap = plt.cm.viridis_r # create colormap
        norm = Normalize(vmin, vmax) # normalise color map to set limits

        # calculate aspect ratio so that pixels on plot are always square:
        # aspect = ((xbmax - xbmin) / out.ny) / ((ybmax - ybmin) / out.nx)
        # aspect = ((90+90)/out.ny) / ((np.nanmax(data[y_coord]) - np.nanmin(data[y_coord])) / out.nx)
        
        # origin='lower' bc otherwise the lower mxrs are at the bottom. .T bc otherwise not layered with delta theta? 
        img = ax.imshow(out.vmean.T, cmap = cmap, norm=norm, aspect='auto', origin='lower', #!!! check origin
                        extent=[out.xbmin, out.xbmax, out.ybmin, out.ybmax]) 

        ax.set_xlabel(x_label); ax.set_xlim(*x_lims)
        ax.set_ylabel(y_label); ax.set_ylim(*y_lims)

    f.suptitle('{} from {} {} between {} - {}. Grid: {}{} x {}{}'.format(
        substance, c_obj.source, c_pfx, c_obj.years[0], c_obj.years[-1], 
        x_bin, x_label[x_label.find('[')+1 : x_label.find(']')],
        y_bin, y_label[y_label.find('[')+1 : y_label.find(']')]))

    f.subplots_adjust(right=0.9)
    plt.tight_layout(pad=2)
    cbar = f.colorbar(img, ax = axs.ravel().tolist(), aspect=40)#, pad=0.2, orientation='vertical') # colorbar
    cbar.ax.set_xlabel('{}'.format(subs.upper() + ' ' + substance[substance.find('[') : substance.find(']')+1]))

    plt.show()

    return 



if __name__=='__main__':
    calc_caribic = False
    if calc_caribic: 
        from data_classes import Caribic
        caribic = Caribic(range(2000, 2018), pfxs = ['GHG', 'INT', 'INT2'])

    # plot_eqlat_deltheta(mzt, subs = 'sf6', y_bin =5e3) 

    plot_eqlat_deltheta(caribic, c_pfx='INT', subs='ch4', tp = 'dyn', pvu=3.5)

    # for subs in ['ch4', 'co', 'co2', 'ch4']:
    #     for tp in ['therm', 'dyn']:
    #         plot_eqlat_deltheta(caribic, c_pfx='INT', subs=subs, tp = tp)

    # for subs in ['ch4', 'co', 'co2', 'ch4', 'n2o']:
    #     for tp in ['z', 'pvu']: 
    #         for pvu in [1.5, 2.0, 3.5]:
    #             plot_eqlat_deltheta(caribic, c_pfx='INT2', subs=subs, tp=tp, pvu=pvu)
