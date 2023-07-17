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

import toolpac.calc.binprocessor as bp

from dictionaries import get_col_name, get_h_coord, get_v_coord, dict_season, substance_list
from detrend import detrend_substance
from tools import make_season, coordinate_tools
from data import Mauna_Loa

#%% 2D plotting
def get_right_data(c_obj, subs='n2o', c_pfx='INT2', detr=True):
    substance = get_col_name(subs, c_obj.source, c_pfx)
    if substance is None: 
        pfxs_avail = [pfx for pfx in c_obj.pfxs if subs in substance_list(pfx)]
        if len(pfxs_avail)==0: print('No {subs} data available')
        else: c_pfx = input(f'No {c_pfx} data found for binning. Choose from {pfxs_avail}')
        if c_pfx not in c_obj.data.keys(): return
        substance = get_col_name(subs, c_obj.source, c_pfx)
    
    if c_obj.source == 'Caribic': data = c_obj.data[c_pfx]
    else: data = c_obj.df
    data['season'] = make_season(data.index.month) # 1 = spring etc

    if detr and not f'detr_{substance}' in data.columns:
        try: 
            c_obj.detrend_substance(subs, Mauna_Loa(c_obj.years, subs), save=True)
            data = c_obj.data[c_pfx]
            substance = f'detr_{substance}'
        except: print('Detrending not successful, proceeding with original data.')
    else: substance = f'detr_{substance}'
    
    return data, substance, c_pfx

def seasonal_binning(data, substance, y_bin, y_coord, x_coord, x_bin, vlims):
    vmin_list, vmax_list = [], []; out_dict = {}

    # calculate binned output per season
    for s in set(data['season'].tolist()):
        df = data[data['season'] == s]
        x = np.array(df[x_coord])
        y = np.array(df[y_coord])
        xbmin, xbmax, xbsize = np.nanmin(x), np.nanmax(x), x_bin
        ybmin, ybmax, ybsize = np.nanmin(y), np.nanmax(y), y_bin

        bin_equi2d = bp.Bin_equi2d(xbmin, xbmax, xbsize, ybmin, ybmax, ybsize)
        out = bp.Simple_bin_2d(np.array(df[substance]), x, y, bin_equi2d)
        out_dict[s] = out
        vmin_list.append(np.nanmin(out.vmean))
        vmax_list.append(np.nanmax(out.vmean))

    if not vlims: vmin, vmax = (np.nanmin(vmin_list), np.nanmax(vmax_list))
    else: vmin, vmax = vlims[0], vlims[1]

    return out_dict, vmin, vmax

def plot_parameters():
    """ x_coord, y_coord, x_lims, y_lims, x_label, y_label """

def plot_2d(bin2d_inst, x_coord, y_coord, ax, season, note,
            x_lims, y_lims, vlims, x_label, y_label):
    """ Plot bin2d output onto given axis """
    ax.set_title(dict_season()[f'name_{season}'])
    if note: ax.text(x_lims[0]*0.9, y_lims[1]*0.85, note, style='italic',
                     bbox={'facecolor':'white'})
    cmap = plt.cm.viridis # create colormap
    norm = Normalize(*vlims) # normalise color map to set limits

    # =============================================================================
    vmean_tot = np.nanmean(bin2d_inst.vmean)
    norm = Normalize(*vlims/ vmean_tot)
    bin2d_inst.vmean = bin2d_inst.vmean / vmean_tot
    # =============================================================================


    img = ax.imshow(bin2d_inst.vmean.T, cmap = cmap, norm=norm,
                    aspect='auto', origin='lower',
                    extent=[bin2d_inst.binclassinstance.xbmin, 
                            bin2d_inst.binclassinstance.xbmax, 
                            bin2d_inst.binclassinstance.ybmin, 
                            bin2d_inst.binclassinstance.ybmax])
    ax.set_xlabel(x_label); ax.set_xlim(*x_lims)
    ax.set_ylabel(y_label); ax.set_ylim(*y_lims)
    return img

def plot_variability(bin2d_inst, x_coord, y_coord, ax, season, note,
            x_lims, y_lims, vlims, x_label, y_label):
    ax.set_title(dict_season()[f'name_{season}'])
    if note: ax.text(x_lims[0]*0.9, y_lims[1]*0.85, note, style='italic',
                     bbox={'facecolor':'white'})
    cmap = plt.cm.viridis # create colormap
    norm = Normalize(*vlims) # normalise color map to set limits
    
    vmean_tot = np.nanmean(bin2d_inst.vmean)
    bin2d_inst.vmean.T - vmean_tot
    
    print(vmean_tot)
    
    img = ax.imshow(bin2d_inst.vmean.T- vmean_tot, cmap = cmap, norm=norm,
                    aspect='auto', origin='lower',
                    extent=[bin2d_inst.binclassinstance.xbmin, 
                            bin2d_inst.binclassinstance.xbmax, 
                            bin2d_inst.binclassinstance.ybmin, 
                            bin2d_inst.binclassinstance.ybmax])
    ax.set_xlabel(x_label); ax.set_xlim(*x_lims)
    ax.set_ylabel(y_label); ax.set_ylim(*y_lims)
    return img

def plot_difference(bin2d_inst1, bin2d_inst2, x_coord, y_coord, ax, season, note,
            x_lims, y_lims, vlims, x_label, y_label):
    """ Plot difference between plots """
    pass

def plot_2d_binned(c_obj, subs='n2o', c_pfx='INT2', ycoord = 'pt', xcoord = 'eql',
                   tp_def = 'dyn', pvu=3.5,
                   x_bin=10, y_bin=None, vlims=None, detr=True, note=None):
    """ Plot binned mxr on EqL vs. pot.T or 
    
    Creates plots of equivalent latitude versus potential temperature or height
    difference relative to tropopause (depends on tropopause definition).
    Plots each season separately on a 2x2 grid.

        c_obj (Caribic)
        c_pfx (str): 'INT', 'INT2'
        subs (str): substance to plot. 'n2o', 'ch4', ...
        tp (str): tropopause definition. 'therm', 'dyn', 'z' or 'pvu'
        pvu (float): potential vorticity set as tropopause definition. 1.5, 2.0 or 3.5
    """
    data, substance, c_pfx = get_right_data(c_obj, subs, c_pfx, detr)
    y_coord, y_label, x_coord, x_label = coordinate_tools(
        tp_def=tp_def, c_pfx=c_pfx, ycoord=ycoord, pvu=pvu, xcoord=xcoord)

    y_bins = {'z' : 0.25, 'pt' : 10, 'p' : 40}
    if not y_bin: y_bin = y_bins[ycoord]

    y_lims = np.nanmin(data[y_coord])-y_bin, np.nanmax(data[y_coord])+y_bin
    x_lims = (-90-x_bin, 90+x_bin)

    out_dict, vmin, vmax = seasonal_binning(data, substance, 
                                            y_bin, y_coord, x_coord, x_bin, vlims)

    # Create plots for all seasons separately
    f, axs = plt.subplots(2, 2, dpi=250, figsize=(9,7))
    for s, ax in zip(set(data['season'].tolist()), axs.flat): # flatten axs array
        out = out_dict[s] # take binned data for current season
        img = plot_2d(out, x_coord, y_coord, ax, s, note, 
                      x_lims, y_lims, (vmin, vmax), x_label, y_label)
        
        plot_variability(out, x_coord, y_coord, ax, s, note, 
                      x_lims, y_lims, (vmin, vmax), x_label, y_label)
        
    f.subplots_adjust(right=0.9)
    plt.tight_layout(pad=2.5)
    cbar = f.colorbar(img, ax = axs.ravel().tolist(), aspect=30, pad=0.09)
    xlabel = subs.upper() + ' ' + substance[substance.find('['):substance.find(']')+1]
    if detr: xlabel = '$\Delta$ '+xlabel
    cbar.ax.set_xlabel(xlabel)
    plt.show()

    return

#%% Fctn calls - eqlat
# if __name__=='__main__':
#     calc_caribic = False
#     if calc_caribic:
#         from data import Caribic
#         caribic = Caribic(range(2000, 2018), pfxs = ['GHG', 'INT', 'INT2'])

#     # for the meeting
#     for subs in ['ch4', 'co2', 'n2o', 'sf6']:
#         plot_eqlat_deltheta(caribic, subs=subs, tp = 'pvu',
#                             x_source = 'ERA5', pvu=2.0)

#     for subs in ['ch4', 'co2', 'n2o', 'sf6']:
#         for tp, xs in zip(['therm', 'dyn', 'pvu'],
#                           ['ECMWF', 'ECMWF', 'ERA5']):
#             plot_eqlat_deltheta(caribic, subs=subs, c_pfx='INT2',
#                                 tp = tp, x_source = xs, pvu=2.0)

    # plot_eqlat_deltheta(mzt, subs = 'sf6', y_bin =5e3)

    # for subs in ['ch4', 'co', 'co2', 'ch4']:
    #     for tp in ['therm', 'dyn']:
    #         plot_eqlat_deltheta(caribic, c_pfx='INT', subs=subs, tp = tp)

    # for subs in ['ch4', 'co', 'co2', 'ch4', 'n2o']:
    #     for tp in ['z', 'pvu']:
    #         for pvu in [1.5, 2.0, 3.5]:
    #             plot_eqlat_deltheta(caribic, c_pfx='INT2', subs=subs, tp=tp, pvu=pvu)
