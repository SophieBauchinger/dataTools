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

from toolpac.calc import bin_1d_2d
import C_tools
from dictionaries import get_col_name, get_vlims

def plot_eqlat_deltheta(c_obj, c_pfx='INT2', subs='n2o', tp = 'therm', pvu=2.0):
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
    data = c_obj.data[c_pfx]
    substance = get_col_name(subs, c_obj.source, c_pfx)

    data['season'] = C_tools.make_season(data.index.month) # 1 = spring etc
    dict_season = {'name_1': 'spring (MAM)', 'name_2': 'summer (JJA)', 'name_3': 'autumn (SON)', 'name_4': 'winter (DJF)',
                   'color_1': 'blue', 'color_2': 'orange', 'color_3': 'green', 'color_4': 'red'}

    # get coordinates according to fctn variables
    if c_pfx == 'INT2': # co, co2, ch4, n2o
        x_coord = 'int_ERA5_EQLAT [deg N]' # Equivalent latitude (ERA5)
        if tp == 'z':
            y_coord = 'int_CARIBIC2_H_rel_TP [km]' # height relative to the tropopause in km: H_rel_TP; replacement for H_rel_TP
            y_label = '$\Delta$z [km]'
        if tp == 'pvu':
            y_coord = 'int_ERA5_D_{}_{}PVU_BOT [K]'.format(str(pvu)[0], str(pvu)[2]) # pot temp difference to potential vorticity surface
            y_label = f'$\Delta$T wrt. {pvu} PV surface [K]'

    elif c_pfx =='INT': # co, co2, ch4
        x_coord = 'int_eqlat [deg]' # equivalent latitude in degrees north from ECMWF
        if tp =='therm': y_coord = 'int_pt_rel_sTP_K [K]' #  potential temperature difference relative to thermal tropopause from ECMWF
        elif tp == 'dyn': y_coord = 'int_pt_rel_dTP_K [K]' #  potential temperature difference relative to  dynamical (PV=3.5PVU) tropopause from ECMWF
        y_label = f'$\Delta$T [K] ({tp})'

    # set bin sizes and plot limits 
    x_label = 'Eq. lat [°N]'
    x_bin = 10 # °N

    if tp in ['therm', 'dyn', 'pvu']:   y_bin = 5 # Delta Theta
    elif tp in ['z']:                   y_bin = 0.25 # km 
    y_lims = np.nanmin(data[y_coord]), np.nanmax(data[y_coord])

    lat = True

    # create overall plot
    f, axs = plt.subplots(2, 2, dpi=250, figsize=(9,7))
    for s, ax in zip(set(data['season'].tolist()), axs.flatten()):
        df = data.loc[data['season'] == s]

        x = np.array(df[x_coord]) # °, eq. latitude 
        if lat: x = np.array(df.geometry.x); x_label = 'Latitude [°N]'
        y = np.array(df[y_coord]) # K, pot temp

        xbmin, xbmax, xbsize = np.nanmin(x), np.nanmax(x), x_bin # °, eq. latitude 
        ybmin, ybmax, ybsize = np.nanmin(y), np.nanmax(y), y_bin # km, height relative to therm. tp

        out = bin_1d_2d.bin_2d(np.array(df[substance]), x, y,
                               xbmin, xbmax, xbsize, ybmin, ybmax, ybsize)

        ax.set_title(dict_season[f'name_{s}'])

        cmap = plt.cm.viridis_r # create colormap
        vmin = np.nanmin([np.nanmin(out.vmean), get_vlims(subs)[0]])
        vmax = np.nanmax([np.nanmin(out.vmean), get_vlims(subs)[1]])
        norm = Normalize(vmin, vmax) # normalise color map to set limits

        img = ax.imshow(out.vmean, cmap = cmap, norm=norm, origin='lower', aspect='auto',  # plot values
                        extent=[out.xbmin, out.xbmax, out.ybmin, out.ybmax])

        cbar = plt.colorbar(img, ax=ax, pad=0.2, orientation='horizontal') # colorbar
        cbar.ax.set_xlabel(f'{substance}')

        ax.set_xlabel(x_label)
        ax.set_xlim(-90, 90)
        ax.set_ylabel(y_label)
        ax.set_ylim(*y_lims)

    plt.tight_layout()
    plt.show()

    return 

if __name__=='__main__':
    calc_caribic = False
    if calc_caribic: 
        from data_classes import Caribic
        caribic = Caribic(range(2000, 2018), pfxs = ['GHG', 'INT', 'INT2'])

    for subs in ['ch4', 'co', 'co2', 'ch4']:
        for tp in ['therm', 'dyn']:
            plot_eqlat_deltheta(caribic, c_pfx='INT', subs=subs, tp = tp)

    for subs in ['ch4', 'co', 'co2', 'ch4', 'n2o']:
        for tp in ['z', 'pvu']: 
            plot_eqlat_deltheta(caribic, c_pfx='INT2', subs=subs, tp=tp)
