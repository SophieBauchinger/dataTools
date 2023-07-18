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
from tools import make_season, coordinate_tools

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
            c_obj.detrend(subs, save=True)
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

def make_2d_plot(bin2d_inst, x_coord, y_coord, ax, season, note,
            x_lims, y_lims, vlims, x_label, y_label, percent=False):
    """ Plot percentage bin2d output onto given axis """
    ax.set_title(dict_season()[f'name_{season}'])

    if percent: 
        vmean_tot = np.nanmean(bin2d_inst.vmean)
        msg = f'% of {vmean_tot:.4}'
    else: vmean_tot = 1.0; msg=None
    norm = Normalize(vlims[0]/vmean_tot, vlims[1]/vmean_tot)
    bin2d_inst.vmean = bin2d_inst.vmean / vmean_tot

    img = ax.imshow(bin2d_inst.vmean.T, cmap = plt.cm.viridis, norm=norm,
                    aspect='auto', origin='lower',
                    extent=[bin2d_inst.binclassinstance.xbmin, 
                            bin2d_inst.binclassinstance.xbmax, 
                            bin2d_inst.binclassinstance.ybmin, 
                            bin2d_inst.binclassinstance.ybmax])
    ax.set_xlabel(x_label); ax.set_xlim(*x_lims)
    ax.set_ylabel(y_label); ax.set_ylim(*y_lims)

    if msg is not None: 
        if note: msg += note
        ax.legend([], [], title=msg, loc='upper left')
    elif note: ax.legend([], [], title=note)    

    return img

def make_diff_plot(
        bin2d_inst1, bin2d_inst2, x_coord, y_coord, ax, season, note,
            x_lims, y_lims, vlims, x_label, y_label):
    """ Plot difference between plots """

    #!!! want to make sure that the simplebin1d is the same for both

    ax.set_title(dict_season()[f'name_{season}'])

    vmean = bin2d_inst1.vmean - bin2d_inst2.vmean
    

    if percent: 
        vmean_tot = np.nanmean(bin2d_inst.vmean)
        msg = f'% of {vmean_tot:.4}'
    else: vmean_tot = 1.0; msg=None
    norm = Normalize(vlims[0]/vmean_tot, vlims[1]/vmean_tot)
    bin2d_inst.vmean = bin2d_inst.vmean / vmean_tot

    img = ax.imshow(bin2d_inst.vmean.T, cmap = plt.cm.viridis, norm=norm,
                    aspect='auto', origin='lower',
                    extent=[bin2d_inst.binclassinstance.xbmin, 
                            bin2d_inst.binclassinstance.xbmax, 
                            bin2d_inst.binclassinstance.ybmin, 
                            bin2d_inst.binclassinstance.ybmax])
    ax.set_xlabel(x_label); ax.set_xlim(*x_lims)
    ax.set_ylabel(y_label); ax.set_ylim(*y_lims)

    if msg is not None: 
        if note: msg += note
        ax.legend([], [], title=msg, loc='upper left')
    elif note: ax.legend([], [], title=note)    

    return img

def plot_2d_binned(c_obj, subs='n2o', v_pfx='GHG', y_params={}, x_params={}, 
                   detr=True, vlims=None, note=None, percent=False):
    """ Plot binned mxr on EqL vs. pot.T or 
    
    Creates plots of equivalent latitude versus potential temperature or height
    difference relative to tropopause (depends on tropopause definition).
    Plots each season separately on a 2x2 grid.

        c_obj (Caribic)
        v_pfx (str): data source
        subs (str): substance to plot. 'n2o', 'ch4', ...

        x_params (dict): 
            keys: x_pfx, xcoord
        y_params (dict):
            keys: ycoord, tp_def, y_pfx, pvu

        tp (str): tropopause definition. 'therm', 'dyn', 'z' or 'pvu'
        pvu (float): potential vorticity set as tropopause definition. 1.5, 2.0 or 3.5
    """
    if (not all(i in x_params.keys() for i in  ['x_pfx', 'xcoord']) 
    or not all(i in y_params.keys() for i in ['ycoord', 'y_pfx', 'tp_def'])): 
        raise KeyError('Please supply all necessary parameters: x_pfx, xcoord / ycoord, tp_def, y_pfx, (pvu)')
    
    if not subs in c_obj.data.keys(): c_obj.create_substance_df(subs)
    data = c_obj.data[subs]
    df, substance, v_pfx = get_right_data(c_obj, subs, v_pfx, detr)
    y_coord, y_label, x_coord, x_label = coordinate_tools(**y_params, **x_params)

    y_bins = {'z' : 0.25, 'pt' : 10, 'p' : 40}
    if not 'y_bin' in y_params.keys(): y_bin = y_bins[y_params['ycoord']]
    else: y_bin = y_params['y_bin']
    y_lims = np.nanmin(data[y_coord])-y_bin, np.nanmax(data[y_coord])+y_bin

    if not 'x_bin' in x_params.keys(): x_bin = 10
    else: x_bin = x_params['x_bin']
    x_lims = (-90, 90)

    out_dict, vmin, vmax = seasonal_binning(data, substance, 
                                            y_bin, y_coord, x_coord, x_bin, vlims)

    # Create plots for all seasons separately
    f, axs = plt.subplots(2, 2, dpi=250, figsize=(9,7))
    for s, ax in zip(set(data['season'].tolist()), axs.flat): # flatten axs array
        out = out_dict[s] # take binned data for current season
        img = make_2d_plot(out, x_coord, y_coord, ax, s, note, 
                           x_lims, y_lims, (vmin, vmax), x_label, y_label, percent=percent)
        
        # plot_variability(out, x_coord, y_coord, ax, s, note, 
        #               x_lims, y_lims, (vmin, vmax), x_label, y_label)
        
    f.subplots_adjust(right=0.9)
    plt.tight_layout(pad=2.5)
    cbar = f.colorbar(img, ax = axs.ravel().tolist(), aspect=30, pad=0.09)
    xlabel = subs.upper() + ' ' + substance[substance.find('['):substance.find(']')+1]
    if detr: xlabel = '$\Delta$ '+ xlabel + '\n wrt. 2005'
    cbar.ax.set_xlabel(xlabel)
    plt.show()

    return

def plot_2d_diff(c_obj, subs='n2o', v_pfx='GHG', 
                 y1_params={}, x1_params={},
                 y2_params={}, x2_params={},
                 detr=True, vlims=None, note=None, percent=False):
    """ Plot binned mxr on EqL vs. pot.T or 
    
    Creates plots of equivalent latitude versus potential temperature or height
    difference relative to tropopause (depends on tropopause definition).
    Plots each season separately on a 2x2 grid.

        c_obj (Caribic)
        v_pfx (str): data source
        subs (str): substance to plot. 'n2o', 'ch4', ...

        x_params (dict): 
            keys: x_pfx, xcoord
        y_params (dict):
            keys: ycoord, tp_def, y_pfx, pvu

        tp (str): tropopause definition. 'therm', 'dyn', 'z' or 'pvu'
        pvu (float): potential vorticity set as tropopause definition. 1.5, 2.0 or 3.5
    """
    if (not all(i in x1_params.keys() for i in  ['x_pfx', 'xcoord']) 
    or not all(i in y1_params.keys() for i in ['ycoord', 'y_pfx', 'tp_def'])
    or not all(i in x2_params.keys() for i in  ['x_pfx', 'xcoord']) 
    or not all(i in y2_params.keys() for i in ['ycoord', 'y_pfx', 'tp_def'])): 
        raise KeyError('Please supply all necessary parameters: x_pfx, xcoord / ycoord, tp_def, y_pfx, (pvu)')
    
    if not subs in c_obj.data.keys(): c_obj.create_substance_df(subs)
    data = c_obj.data[subs]
    df, substance, v_pfx = get_right_data(c_obj, subs, v_pfx, detr)
    y1_coord, y1_label, x1_coord, x1_label = coordinate_tools(**y1_params, **x1_params)
    y2_coord, y2_label, x2_coord, x2_label = coordinate_tools(**y2_params, **x2_params)

    y_bins = {'z' : 0.25, 'pt' : 10, 'p' : 40}
    if not 'y_bin' in y1_params.keys(): y_bin = y_bins[y1_params['ycoord']]
    else: y_bin = y1_params['y_bin']
    y_lims = np.nanmin(data[y1_coord])-y_bin, np.nanmax(data[y1_coord])+y_bin

    if not 'x_bin' in x1_params.keys(): x_bin = 10
    else: x_bin = x1_params['x_bin']
    x_lims = (-90, 90)


    #!!! THERM ERA5 PT is not correct

    # calculate binned output per season
    # for s in set(data['season'].tolist()):
    #     df = data[data['season'] == s]
    #     x = np.array(df[x_coord])
    #     y = np.array(df[y_coord])
    #     xbmin, xbmax, xbsize = np.nanmin(x), np.nanmax(x), x_bin
    #     ybmin, ybmax, ybsize = np.nanmin(y), np.nanmax(y), y_bin

    #     bin_equi2d = bp.Bin_equi2d(xbmin, xbmax, xbsize, ybmin, ybmax, ybsize)
    #     out = bp.Simple_bin_2d(np.array(df[substance]), x, y, bin_equi2d)
    #     out_dict[s] = out
    #     vmin_list.append(np.nanmin(out.vmean))
    #     vmax_list.append(np.nanmax(out.vmean))

    # if not vlims: vmin, vmax = (np.nanmin(vmin_list), np.nanmax(vmax_list))
    # else: vmin, vmax = vlims[0], vlims[1]

    # return out_dict, vmin, vmax

    out_dict = {}
    for s in set(data['season'].tolist()):
        df = data[data['season'] == s]

        # !!! this doesn't work - need to have bin and data in the same shape...
        
        bin_equi2d = bp.Bin_equi2d(*x_lims, x_bin, *y_lims, y_bin)
        
        x1 = np.array(df[x1_coord])
        y1 = np.array(df[y1_coord])
        out1 = bp.Simple_bin_2d(np.array(data[substance]), x1, y1, bin_equi2d)
        
        x2 = np.array(df[x2_coord])
        y2 = np.array(df[y2_coord])
        out2 = bp.Simple_bin_2d(np.array(data[substance]), x2, y2, bin_equi2d)
        
        out_dict[s] = (out1, out2)
    
    return out_dict
    # if not vlims: vmin, vmax = (np.nanmin(vmin_list), np.nanmax(vmax_list))


    # out1_dict, vmin1, vmax1 = seasonal_binning(data, substance, 
    #                                         y_bin, y1_coord, x1_coord, x_bin, vlims)

    # out2_dict, vmin2, vmax2 = seasonal_binning(data, substance, 
    #                                         y_bin, y2_coord, x2_coord, x_bin, vlims)

    # vmin = min(vmin1, vmin2); vmax = max(vmax1, vmax2)

    # # Create plots for all seasons separately
    # f, axs = plt.subplots(2, 2, dpi=250, figsize=(9,7))
    # for s, ax in zip(set(data['season'].tolist()), axs.flat): # flatten axs array
    #     out1 = out1_dict[s] # take binned data for current season
    #     out2 = out2_dict[s]
    #     img = make_diff_plot(out1, out2, x_coord, y_coord, ax, s, note, 
    #                        x_lims, y_lims, (vmin, vmax), x_label, y_label, percent=percent)
        
    #     # plot_variability(out, x_coord, y_coord, ax, s, note, 
    #     #               x_lims, y_lims, (vmin, vmax), x_label, y_label)
        
    # f.subplots_adjust(right=0.9)
    # plt.tight_layout(pad=2.5)
    # cbar = f.colorbar(img, ax = axs.ravel().tolist(), aspect=30, pad=0.09)
    # xlabel = subs.upper() + ' ' + substance[substance.find('['):substance.find(']')+1]
    # if detr: xlabel = '$\Delta$ '+ xlabel + '\n wrt. 2005'
    # cbar.ax.set_xlabel(xlabel)
    # plt.show()

    return


#%% Fctn calls - eqlat
if False: caribic = True # BS to avoid error
# --- chem ---
yp_c1 = {'tp_def' : 'chem', 
       'y_pfx' : 'INT2', # same data in INT
       'ycoord' : 'z'}

# --- therm ---
yp_t1 = {'tp_def' : 'therm', 
       'y_pfx' : 'INT2', 
       'ycoord' : 'pt'}
yp_t2 = {'tp_def' : 'therm', 
       'y_pfx' : 'INT', 
       'ycoord' : 'pt'}

# --- dyn ---
yp_d1 = {'tp_def' : 'dyn', 
       'y_pfx' : 'INT', 
       'ycoord' : 'pt', 
       'pvu' : 3.5}
yp_d2 = {'tp_def' : 'dyn', 
       'y_pfx' : 'INT', 
       'ycoord' : 'pt', 
       'pvu' : 3.5}

# --- x params ----
xp1 = {'x_pfx' : 'INT', # ECMWF
      'xcoord' : 'eql'}

xp2 = {'x_pfx' : 'INT2', # ERA5
      'xcoord' : 'eql'}

if __name__ == '__main__':
    for subs in ['sf6', 'n2o', 'co2', 'ch4']:
        for yp in [yp_c1, yp_t1, yp_t2, yp_d1, yp_d2]:
            plot_2d_binned(caribic, subs, y_params=yp, x_params=xp1)

        for y1, y2 in [(yp_d1, yp_d2)]:
            pass
