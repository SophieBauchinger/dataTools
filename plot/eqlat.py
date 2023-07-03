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

from toolpac.calc import bin_1d_2d #!!!
from dictionaries import get_col_name
from detrend import detrend_substance
from tools import subs_merge, make_season
from data import Mauna_Loa

def plot_eqlat_deltheta(c_obj, subs='n2o', c_pfx='INT2', tp = 'therm', pvu=2.0,
                        x_bin=None, y_bin=None, x_source='ERA5', vlims=None,
                        detr=True, note=None):
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

    #!!! detrended data ?

    if not f'{subs}_data' in c_obj.data.keys():
        detrend_substance(c_obj, subs, Mauna_Loa(c_obj.years, subs))
        subs_merge(c_obj, subs, save=True, detr=True)

    try:
        data = c_obj.data[f'{subs}_data']
        substance = get_col_name(subs, c_obj.source, 'GHG')
        if detr and 'delta_'+substance in data.columns:
            substance = 'delta_'+substance
    except:
        if c_obj.source == 'Caribic':
            data = c_obj.data[c_pfx]
            substance = get_col_name(subs, c_obj.source, c_pfx)
        elif c_obj.source == 'Mozart': data = c_obj.df
        else: raise('Could not find the specified data set. Check your input')

    data['season'] = make_season(data.index.month) # 1 = spring etc
    dict_season = {'name_1': 'Spring (MAM)', 'name_2': 'Summer (JJA)',
                   'name_3': 'Autumn (SON)', 'name_4': 'Winter (DJF)',
                   'color_1': 'blue', 'color_2': 'orange',
                   'color_3': 'green', 'color_4': 'red'}

    # Get column name for y axis depending on function parameter
    if tp == 'z':
        # height relative to the tropopause in km: H_rel_TP
        y_coord = 'int_CARIBIC2_H_rel_TP [km]'
        y_label = '$\Delta$z [km]'
        if not y_bin: y_bin = 0.25 # km
    elif tp == 'pvu':
        # pot temp difference to potential vorticity surface
        y_coord = 'int_ERA5_D_{}_{}PVU_BOT [K]'.format(str(pvu)[0], str(pvu)[2])
        if not y_bin: y_bin = 5 # K
        y_label = f'$\Delta\Theta$ ({pvu} PVU - ERA5) [K]'
    elif tp =='therm':
        #  potential T. difference relative to thermal tropopause from ECMWF
        y_coord = 'int_pt_rel_sTP_K [K]'
        if not y_bin: y_bin = 5 # K
        y_label = f'$\Delta\Theta$ ({tp} - ECMWF) [K]'
    elif tp == 'dyn':
        #  potential T difference rel. to dynamical (PV=3.5PVU) tp from ECMWF
        y_coord = 'int_pt_rel_dTP_K [K]'
        if not y_bin: y_bin = 10 # K
        y_label = f'$\Delta\Theta$ ({tp} - ECMWF) [K]'
    else:
        y_coord = 'p [mbar]'
        if not y_bin: y_bin = 40 # mbar
        y_label = 'Pressure [mbar]'

    if x_source=='ERA5':
        x_coord = 'int_ERA5_EQLAT [deg N]' # Equivalent latitude (ERA5)
    elif x_source=='ECMWF':
        x_coord = 'int_eqlat [deg]' # equivalent latitude in °N from ECMWF
    else: raise('No valid source for equivalent latitude given. ')

    # set bin sizes and plot limits
    x_label = f'Eq. latitude ({x_source}) [°N]'
    if not x_bin: x_bin = 10 # °N

    if c_obj.source == 'Mozart':
        y_coord = 'PS'; y_label='Surface Pressure [Pa]'; y_bin = 5e3

    y_lims = np.nanmin(data[y_coord])-y_bin, np.nanmax(data[y_coord])+y_bin
    x_lims = (-90-x_bin, 90+x_bin)

    vmin_list, vmax_list = [], []; out_dict = {}

    # calculate binned output per season
    for s in set(data['season'].tolist()):
        df = data[data['season'] == s]

        if not c_obj.source == 'Mozart': x = np.array(df[x_coord])
        else: x = np.array(df.geometry.x); x_label='latitude'
        y = np.array(df[y_coord])
        xbmin, xbmax, xbsize = np.nanmin(x), np.nanmax(x), x_bin
        ybmin, ybmax, ybsize = np.nanmin(y), np.nanmax(y), y_bin

        out = bin_1d_2d.bin_2d(np.array(df[substance]), x, y,
                               xbmin, xbmax, xbsize, ybmin, ybmax, ybsize)
        out_dict[s] = out
        vmin_list.append(np.nanmin(out.vmean))
        vmax_list.append(np.nanmax(out.vmean))

    if vlims: vmin, vmax = vlims[0], vlims[1]
    else: vmin = np.nanmin(vmin_list); vmax = np.nanmax(vmax_list)

    # create parent plot
    f, axs = plt.subplots(2, 2, dpi=250, figsize=(9,7))

    for s, ax in zip(set(data['season'].tolist()), axs.flat): # flatten axs array
        out = out_dict[s] # take binned data for current season
        ax.set_title(dict_season[f'name_{s}'])
        if note: ax.text(x_lims[0]*0.9, y_lims[1]*0.85, note, style='italic',
                         bbox={'facecolor':'white'})
        cmap = plt.cm.viridis # create colormap
        norm = Normalize(vmin, vmax) # normalise color map to set limits
        img = ax.imshow(out.vmean.T, cmap = cmap, norm=norm,
                        aspect='auto', origin='lower',
                        extent=[out.xbmin, out.xbmax, out.ybmin, out.ybmax])
        ax.set_xlabel(x_label); ax.set_xlim(*x_lims)
        ax.set_ylabel(y_label); ax.set_ylim(*y_lims)

    # f.suptitle('{} from {} {} between {} - {}. Grid: {}{} x {}{}'.format(
    #     substance, c_obj.source, c_pfx, c_obj.years[0], c_obj.years[-1],
    #     x_bin, x_label[x_label.find('[')+1 : x_label.find(']')],
    #     y_bin, y_label[y_label.find('[')+1 : y_label.find(']')]))

    f.subplots_adjust(right=0.9)
    plt.tight_layout(pad=2.5)
    cbar = f.colorbar(img, ax = axs.ravel().tolist(), aspect=30, pad=0.09)
    xlabel = subs.upper() + ' ' + substance[substance.find('['):substance.find(']')+1]
    if detr: xlabel = '$\Delta $'+xlabel
    cbar.ax.set_xlabel(xlabel)
    plt.show()

    return

#%%
if __name__=='__main__':
    calc_caribic = False
    if calc_caribic:
        from data import Caribic
        caribic = Caribic(range(2000, 2018), pfxs = ['GHG', 'INT', 'INT2'])

    # for the meeting
    for subs in ['ch4', 'co2', 'n2o', 'sf6']:
        plot_eqlat_deltheta(caribic, subs=subs, tp = 'pvu',
                            x_source = 'ERA5', pvu=2.0)

    for subs in ['ch4', 'co2', 'n2o', 'sf6']:
        for tp, xs in zip(['therm', 'dyn', 'pvu'],
                          ['ECMWF', 'ECMWF', 'ERA5']):
            plot_eqlat_deltheta(caribic, subs=subs, c_pfx='INT2',
                                tp = tp, x_source = xs, pvu=2.0)

    # plot_eqlat_deltheta(mzt, subs = 'sf6', y_bin =5e3)

    # for subs in ['ch4', 'co', 'co2', 'ch4']:
    #     for tp in ['therm', 'dyn']:
    #         plot_eqlat_deltheta(caribic, c_pfx='INT', subs=subs, tp = tp)

    # for subs in ['ch4', 'co', 'co2', 'ch4', 'n2o']:
    #     for tp in ['z', 'pvu']:
    #         for pvu in [1.5, 2.0, 3.5]:
    #             plot_eqlat_deltheta(caribic, c_pfx='INT2', subs=subs, tp=tp, pvu=pvu)
