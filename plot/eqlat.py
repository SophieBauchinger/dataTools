# -*- coding: utf-8 -*-
"""
@Author: Sophie Bauchinger, IAU
@Date: Tue Jun  6 13:59:31 2023

Showing mixing ratios per season on a plot of coordinate relative to the
tropopause (in km or K) versus equivalent latitude (in deg N)
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import pandas as pd
import itertools

import toolpac.calc.binprocessor as bp

import dictionaries as dcts
import tools
# from dictionaries import get_col_name, get_h_coord, get_v_coord, dict_season, substance_list
# from tools import make_season, coordinate_tools

class Bin2dPlotter():
    """ """ 
    def __init__(self, df, detr=False, 
                 subs_params=dict(), 
                 x_params=dict(hcoord='eql'), 
                 y_params=dict(vcoord='pt', tp_def='dyn', rel_to_tp=True),
                 **kwargs):
        """ Initialise class instances. """
        df['season'] = tools.make_season(df.index.month)
        self.data = {'df' : df} # dataframe
        self.detr = detr
        self.substances = [s for s in dcts.get_substances(**subs_params)
                           if s.col_name in df.columns]
        self.x_coordinates = [s for s in dcts.get_coordinates(**x_params)
                              if s.col_name in df.columns]
        self.y_coordinates = [s for s in dcts.get_coordinates(**y_params)
                              if s.col_name in df.columns]
        self.kwargs = kwargs
        self.check_input()

    def __repr__(self):
        return '<Bin2dPlotter>\n - subs: {}\n - xcoords: {}\n - ycoords: {}'.format(
            [f'{s.short_name} ({s.ID})' for s in self.substances],
            self.x_coordinates, self.y_coordinates)

    @property
    def df(self) -> pd.DataFrame:
        return self.data['df']

    def check_input(self) -> bool: 
        """ Check if any subs, x, y columns are in data. """
        if not any([(s.col_name in self.df.columns) for s in self.substances]): 
            raise KeyError('No substances found in data that fit the parameters. ')
        if not any([(c.col_name in self.df.columns) for c in self.x_coordinates]): 
            raise KeyError('No x-coordinate found in data that fit the parameters. ')
        if not any([(c.col_name in self.df.columns) for c in self.y_coordinates]): 
            raise KeyError('No y-coordinate found in data that fit the parameters. ')
        return True

    def get_limits(self, subs, xcoord, ycoord) -> (tuple, tuple, tuple):
        """ Check kwargs for limits, otherwise set default values """
        if 'vlims' in self.kwargs:
            vlims = self.kwargs.get('vlims')
        else: 
            try: vlims = dcts.get_default_vlims(subs.short_name, detr=self.detr)
            except KeyError: vlims = (np.nanmin(self.df[subs.col_name]), np.nanmax(self.df[subs.col_name]))
        
        if 'xlims' in self.kwargs:
            xlims = self.kwargs.get('xlims')
        else: 
            xlims = (-90, 90)
            # xlims = (np.nanmin(self.df[xcoord.col_name]), np.nanmax(self.df[xcoord.col_name]))

        if 'ylims' in self.kwargs:
            ylims = self.kwargs.get('ylims')
        else: 
            ylims = (np.nanmin(self.df[ycoord.col_name]), np.nanmax(self.df[ycoord.col_name]))
        
        return vlims, xlims, ylims

    def get_bins(self, subs, xcoord, ycoord) -> (float, float):
        """ Get bin size for x- and y-coordinates. """
        vlims, xlims, ylims = self.get_limits(subs, xcoord, ycoord)
        
        x_bin = 10 if not self.kwargs.get('x_bin') else self.kwargs.get('x_bin')
        y_bin = max(ylims)/10 if not self.kwargs.get('y_bin') else self.kwargs.get('y_bin')

        return x_bin, y_bin

    def bin_seasonal(self, subs, xcoord, ycoord, 
                     bin_equi2d = None, 
                     x_bin=None, y_bin=None) -> dict:
        """ Bin the dataframe per season. """
        out_dict = {}
        if not x_bin: 
            x_bin = self.get_bins(subs, xcoord, ycoord)[0]
        if not y_bin: 
            y_bin = self.get_bins(subs, xcoord, ycoord)[1]

        # calculate binned output per season
        for s in set(self.df['season'].tolist()):
            df = self.df[self.df['season'] == s]
            x = np.array(df[xcoord.col_name])
            y = np.array(df[ycoord.col_name])

            xbmin, xbmax, xbsize = np.nanmin(x), np.nanmax(x), x_bin
            ybmin, ybmax, ybsize = np.nanmin(y), np.nanmax(y), y_bin

            if not bin_equi2d: 
                bin_equi2d = bp.Bin_equi2d(xbmin, xbmax, xbsize, ybmin, ybmax, ybsize)

            if self.detr: 
                out = bp.Simple_bin_2d(np.array(df['detr_'+subs.col_name]), x, y, bin_equi2d)
            else: out = bp.Simple_bin_2d(np.array(df[subs.col_name]), x, y, bin_equi2d)
            
            out = bp.Simple_bin_2d(np.array(df[subs.col_name]), x, y, bin_equi2d)

            out_dict[s] = out

        return out_dict

    def plot_2d_mxr(self, subs, xcoord, ycoord, **kwargs):
        """ Plot percentage bin2d output onto given axis. """
        binned_seasonal = self.bin_seasonal(subs, xcoord, ycoord)
        vlims, xlims, ylims = self.get_limits(subs, xcoord, ycoord)
        
        norm = Normalize(*vlims)
        
        fig, axs = plt.subplots(2, 2, dpi=250, figsize=(8,9), 
                                sharey=True, sharex=True)

        for season, ax in zip([1,2,3,4], axs.flatten()):
            bin2d_inst = binned_seasonal[season]
            ax.set_title(dcts.dict_season()[f'name_{season}'])

            bci = bin2d_inst.binclassinstance
            img = ax.imshow(bin2d_inst.vmean.T, 
                            cmap = plt.cm.viridis, norm=norm,
                            aspect='auto', origin='lower',
                            extent=[bci.xbmin, bci.xbmax, bci.ybmin, bci.ybmax])
            ax.set_xlim(xlims[0]*0.95, xlims[1]*1.05)
            ax.set_ylim(ylims[0]*0.95, ylims[1]*1.05)

            ax.set_xlabel(dcts.make_coord_label(xcoord))
            ax.set_ylabel(dcts.make_coord_label(ycoord))
            
            if kwargs.get('note'): 
                ax.text(**dcts.note_dict(ax, s=kwargs.get('note')))
        
        fig.subplots_adjust(right=0.9)
        fig.tight_layout(pad=2.5)
        
        cbar = fig.colorbar(img, ax = axs.ravel().tolist(), aspect=30, pad=0.09, orientation='horizontal')
        cbar.ax.set_xlabel(dcts.make_subs_label(subs))

        plt.show()

    def plot_mxr_diff(self, params_1, params_2, **kwargs): 
        """ Plot difference between two plots. """
        subs1, xcoord1, ycoord1 = params_1
        subs2, xcoord2, ycoord2 = params_2

        x_bin, y_bin = self.get_bins(*params_1)
        bin_equi2d = bp.Bin_equi2d(np.nanmin(self.df[xcoord1.col_name]), 
                                   np.nanmax(self.df[xcoord1.col_name]), 
                                   x_bin, 
                                   np.nanmin(self.df[ycoord1.col_name]), 
                                   np.nanmax(self.df[ycoord1.col_name]),
                                   y_bin)

        binned_seasonal_1 = self.bin_seasonal(*params_1, bin_equi2d=bin_equi2d)
        binned_seasonal_2 = self.bin_seasonal(*params_2, bin_equi2d=bin_equi2d)
        
        vlims, xlims, ylims = self.get_limits(*params_1)
        cmap = plt.cm.PiYG

        fig, axs = plt.subplots(2, 2, dpi=250, figsize=(8,9), 
                                sharey=True, sharex=True)

        for season, ax in zip([1,2,3,4], axs.flatten()):
            ax.set_title(dcts.dict_season()[f'name_{season}'])
            ax.set_facecolor('lightgrey')

            # note simple substraction filters out everything where either is nan
            vmean = binned_seasonal_1[season].vmean - binned_seasonal_2[season].vmean
            vmax_abs = max(abs(np.nanmin(vmean)), abs(np.nanmax(vmean)))
            norm = Normalize(-vmax_abs, vmax_abs) #TODO meh
            
            bci = binned_seasonal_1[season].binclassinstance

            img = ax.imshow(vmean.T, cmap = cmap, norm=norm,
                            aspect='auto', origin='lower', 
                            extent=[bci.xbmin, bci.xbmax, bci.ybmin, bci.ybmax])
            ax.set_xlim(xlims[0]*0.95, xlims[1]*1.05)
            ax.set_ylim(ylims[0]*0.95, ylims[1]*1.05)

            ax.set_xlabel(dcts.make_coord_label(xcoord1))
            ax.set_ylabel(dcts.make_coord_label(ycoord1))
            
            if kwargs.get('note'): 
                ax.text(**dcts.note_dict(ax, s=kwargs.get('note')))
        
        fig.subplots_adjust(right=0.9)
        fig.tight_layout(pad=2.5)
        
        cbar = fig.colorbar(img, ax = axs.ravel().tolist(), aspect=30, 
                            pad=0.09, orientation='horizontal')
        cbar.ax.set_xlabel(
            f'{dcts.make_subs_label(subs1)} {dcts.make_coord_label(xcoord1)} {dcts.make_coord_label(ycoord1)} \n vs.\n\
{dcts.make_subs_label(subs2)} {dcts.make_coord_label(xcoord2)} {dcts.make_coord_label(ycoord2)}')
        plt.show()


    def plot_mixing_ratios(self, **kwargs):
        """ Plot all possible permutations of subs, xcoord, ycoord. """
        permutations = list(itertools.product(self.substances,
                                              self.x_coordinates, 
                                              self.y_coordinates))
        for perm in permutations:
            self.plot_2d_mxr(*perm, **kwargs)

    def plot_differences(self, **kwargs):
        """ Plot the mixing ratio difference between different substance cols and coordinates. """
        permutations = list(itertools.product(self.substances,
                                              self.x_coordinates, 
                                              self.y_coordinates))

        for params_1, params_2 in itertools.combinations(permutations, 2): 
            if (params_1[0].short_name == params_2[0].short_name
                and params_1[1].hcoord == params_2[1].hcoord
                and params_1[2].vcoord == params_2[2].vcoord): 
                # only compare same substance in same coordinate system
                self.plot_mxr_diff(params_1, params_2, **kwargs)
        

#%% 2D plotting
def get_right_data(c_obj, subs='n2o', c_pfx='INT2', detr=True):
    substance = dcts.get_col_name(subs, c_obj.source, c_pfx)
    if substance is None: 
        pfxs_avail = [pfx for pfx in c_obj.pfxs if subs in dcts.substance_list(pfx)]
        if len(pfxs_avail)==0: print('No {subs} data available')
        else: c_pfx = input(f'No {c_pfx} data found for binning. Choose from {pfxs_avail}')
        if c_pfx not in c_obj.data.keys(): return
        substance = dcts.get_col_name(subs, c_obj.source, c_pfx)
    
    if c_obj.source == 'Caribic': 
        try: data = c_obj.data[subs]
        except: data = c_obj.data[c_pfx]
    else: data = c_obj.df
    data['season'] = tools.make_season(data.index.month) # 1 = spring etc

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


def bin_seasonal(glob_obj, subs, subs_params={}, y_params={}, x_params={}, **kwargs) -> (dict, float, float):
    """ Find substance, x, y, coordinates and calculate seasonal bins. """
    
    data = glob_obj.df
    subs = dcts.get_subs(short_name=subs, source = glob_obj.source, **subs_params)

    subs_col = subs.col_name
    subs_label = dcts.make_subs_label(subs)
    
    if kwargs.get('detr') and 'detr_'+subs.col_name in data.columns:
        subs_col = 'detr_' + subs_col
        subs_label = f'{subs_label} detrended wrt. MLO 2005. '

    y_coord = dcts.get_coord(**y_params)
    y_label = dcts.make_coord_label(y_coord)
    x_coord = dcts.get_coord(**x_params)
    x_label = dcts.make_coord_label(x_coord)
    
    y_lim = np.nanmax(abs(data[y_params.col_name])) if not y_params.get('y_lim') else kwargs.get('y_lim')
    y_bin = y_lim/10 if not y_params.get('y_bin') else y_params.get('y_bin')
    x_lims = (-90, 90)
    x_bin = 10 if not x_params.get('x_bin') else x_params.get('x_bin')

    # seasonal_binning
    vmin_list, vmax_list = [], []; out_dict = {}

    # calculate binned output per season
    for s in set(data['season'].tolist()):
        df = data[data['season'] == s]
        x = np.array(df[x_coord])
        y = np.array(df[y_coord])
        xbmin, xbmax, xbsize = np.nanmin(x), np.nanmax(x), x_bin
        ybmin, ybmax, ybsize = np.nanmin(y), np.nanmax(y), y_bin

        bin_equi2d = bp.Bin_equi2d(xbmin, xbmax, xbsize, ybmin, ybmax, ybsize)
        out = bp.Simple_bin_2d(np.array(df[subs_col]), x, y, bin_equi2d)
        out_dict[s] = out
        vmin_list.append(np.nanmin(out.vmean))
        vmax_list.append(np.nanmax(out.vmean))

    if not vlims: vmin, vmax = (np.nanmin(vmin_list), np.nanmax(vmax_list))
    else: vmin, vmax = vlims[0], vlims[1]

    return out_dict, vmin, vmax


def make_2d_plot(bin2d_inst, x_coord, y_coord, ax, season, note,
            x_lims, y_lims, vlims, x_label, y_label, percent=False):
    """ Plot percentage bin2d output onto given axis """
    ax.set_title(dcts.dict_season()[f'name_{season}'])

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

def make_2d_MAD(bin2d_inst, x_coord, y_coord, ax, season, note,
            x_lims, y_lims, vlims, x_label, y_label, percent=False):
    """ mean absolute deviation - 
        sum(abs(x - mu)) / N 
        
        x - data point value
        mu - mean (total)
        N - sample size
        """
    ax.set_title(dcts.dict_season()[f'name_{season}'])

    if percent: 
        vmean_tot = np.nanmean(bin2d_inst.vmean)
        msg = f'% of {vmean_tot:.4}'
    else: vmean_tot = 1.0; msg=None
    # norm = Normalize(vlims[0]/vmean_tot, vlims[1]/vmean_tot)

    # bin2d_inst.vmean = bin2d_inst.vmean / vmean_tot

    # values = # per bin??
    norm = Normalize(vlims[0], vlims[1]) # ? 

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

def plot_2d_binned(glob_obj, subs, subs_params={}, y_params={}, x_params={}, **kwargs):
                   # subs='n2o', subs_ID='GHG', y_params={}, x_params={}, subs_params={},
                   # detr=True, vlims=None, note=None, percent=False, ylim_plt=None):
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
    out_dict, vmin, vmax = bin_seasonal(glob_obj, subs, subs_params, y_params, x_params, **kwargs)

    # Create plots for all seasons separately
    fig, axs = plt.subplots(2, 2, dpi=250, figsize=(9,7))
    for s, ax in zip(set(data['season'].tolist()), axs.flat): # flatten axs array
        out = out_dict[s] # take binned data for current season
        img = make_2d_plot(out, x_coord, y_coord, ax, s, note, 
                           x_lims, (-y_lim, y_lim), (vmin, vmax), x_label, y_label, percent=percent)

        # plot_variability(out, x_coord, y_coord, ax, s, note, 
        #               x_lims, y_lims, (vmin, vmax), x_label, y_label)
        if kwargs.get('note'): 
            ax.text(**dcts.note_dict(ax, s=note))
        
    fig.subplots_adjust(right=0.9)
    plt.tight_layout(pad=2.5)

    cbar = fig.colorbar(img, ax = axs.ravel().tolist(), aspect=30, pad=0.09)
    cbar.ax.set_xlabel(subs_label)

    plt.show()

    return out_dict

def make_diff_plot(bin2d_inst1, bin2d_inst2, 
                   x_coord, y_coord, ax, season, note, 
                   x_lims, y_lims, vlims, x_label, y_label, percent,
                   mismatch_indic=False):
    """ Plot difference between plots """

    ax.set_title(dcts.dict_season()[f'name_{season}'])

    # NB simple substraction filters out everything where either is nan
    vmean = bin2d_inst1.vmean - bin2d_inst2.vmean
    cmap = plt.cm.PiYG
    if vlims is not None: norm = Normalize(*vlims)
    elif percent: #!!! implement percent change...
        
    # Maximaldifferenz 
    
        norm = Normalize(np.nanmin(vmean), np.nanmax(vmean))
    else: norm = Normalize(np.nanmin(vmean), np.nanmax(vmean))
    extent = [bin2d_inst1.binclassinstance.xbmin, 
              bin2d_inst1.binclassinstance.xbmax, 
              bin2d_inst1.binclassinstance.ybmin, 
              bin2d_inst1.binclassinstance.ybmax]

    # indicating on the plot where a single one of them is nan
    if mismatch_indic:
        single_nan_bool = np.isnan(bin2d_inst1.vmean) ^ np.isnan(bin2d_inst2.vmean) # bool multidim arr, True where only one is nan
        single_nan = np.where(np.where(~single_nan_bool, 0, 1), 1, np.nan) # 1 where True, nan where False
        vmean = np.where(~single_nan_bool, vmean, -9999)
        cmap.set_under(color='blue', alpha=0.08)
        ax.contourf(single_nan.T, True, norm=norm, origin='lower', hatches=['....'],
                    alpha=0, extent=extent, extend='max')
        # ax.contour(single_nan_bool.T, True, levels=2, origin='lower',
        #             alpha=0.5, extent=extent, antialiased=True)

    img = ax.imshow(vmean.T, cmap = cmap, norm=norm,
                    aspect='auto', origin='lower', extent=extent)

    ax.set_xlabel(x_label); ax.set_xlim(*x_lims)
    ax.set_ylabel(y_label); ax.set_ylim(*y_lims)

    if note: ax.legend([], [], title=note, loc='upper left')

    return img

def plot_2d_diff(c_obj, subs='n2o', v_pfx='GHG', 
                 y1_params={}, x1_params={},
                 y2_params={}, x2_params={},
                 detr=True, vlims=None, note=None, percent=False, 
                 mismatch_indic = True, ylim_plt = None):
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
    data, substance, v_pfx = get_right_data(c_obj, subs, v_pfx, detr)
    y1_coord, y1_label, x1_coord, x1_label = coordinate_tools(**y1_params, **x1_params)
    y2_coord, y2_label, x2_coord, x2_label = coordinate_tools(**y2_params, **x2_params)

    y_bins = {'z' : 0.25, 'pt' : 10, 'p' : 40}
    if not 'y_bin' in y1_params.keys(): y_bin = y_bins[y1_params['ycoord']]
    else: y_bin = y1_params['y_bin']
    
    ylim = np.nanmax([abs(data[y1_coord]), abs(data[y2_coord])]) # maximum extent of x-coordinate
    # y_lims = np.nanmin(data[y1_coord])-y_bin, np.nanmax(data[y1_coord])+y_bin

    if not 'x_bin' in x1_params.keys(): x_bin = 10
    else: x_bin = x1_params['x_bin']
    x_lims = (-90, 90)

    vmin_list = vmax_list = []
    out_dict = {}
    for s in set(data['season'].tolist()):
        df = data[data['season'] == s]

        # !!! this doesn't work - need to have bin and data in the same shape...
        
        bin_equi2d = bp.Bin_equi2d(*x_lims, x_bin, -ylim, ylim, y_bin)
        
        x1 = np.array(df[x1_coord])
        y1 = np.array(df[y1_coord])
        out1 = bp.Simple_bin_2d(np.array(df[substance]), x1, y1, bin_equi2d)
        
        x2 = np.array(df[x2_coord])
        y2 = np.array(df[y2_coord])
        out2 = bp.Simple_bin_2d(np.array(df[substance]), x2, y2, bin_equi2d)
        
        out_dict[s] = (out1, out2)
        vmin_list.append([np.nanmin(out1.vmean), np.nanmin(out2.vmean)])
        vmax_list.append([np.nanmax(out1.vmean), np.nanmax(out2.vmean)])

    if not vlims: vmin, vmax = (np.nanmin(vmin_list), np.nanmax(vmax_list))
    else: vmin, vmax = vlims[0], vlims[1]

    # Create plots for all seasons separately
    
    if ylim_plt is not None: ylims = (-ylim_plt, ylim_plt)
    else: ylims = (-ylim, ylim)
    
    f, axs = plt.subplots(2, 2, dpi=250, figsize=(9,7))
    for s, ax in zip(set(data['season'].tolist()), axs.flat): # flatten axs array
        out1 = out_dict[s][0] # take binned data for current season
        out2 = out_dict[s][1]

        img = make_diff_plot(out1, out2, x1_coord, y1_coord, ax, s, note, 
                            x_lims, ylims, (vmin, vmax), 
                            x1_label, y1_label, percent=percent, 
                            mismatch_indic = mismatch_indic)
        
        # plot_variability(out, x_coord, y_coord, ax, s, note, 
        #               x_lims, y_lims, (vmin, vmax), x_label, y_label)
        
    f.subplots_adjust(right=0.9)
    plt.tight_layout(pad=2.5)
    cbar = f.colorbar(img, ax = axs.ravel().tolist(), aspect=30, pad=0.09)
    xlabel = subs.upper() + ' ' + substance[substance.find('['):substance.find(']')+1]
    if detr: xlabel = '$\Delta$ '+ xlabel + '\n therm - chem'
    cbar.ax.set_xlabel(xlabel)
    plt.show()

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
       'y_pfx' : 'INT2', 
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
            # pass
        for y1, y2 in [(yp_d1, yp_d2)]:
            plot_2d_diff(caribic, subs, 'GHG', y1, xp1, y2, xp2)

    for subs, vlims in zip(['sf6', 'n2o', 'co2', 'ch4'],
                           [(-0.1, 0.1), (-1, 1), (-1, 1), (-10, 10)]):
        ylim = 150
        plot_2d_binned(caribic, subs, 'GHG', yp_t1, xp2, ylim_plt=ylim)
        plot_2d_binned(caribic, subs, 'GHG', yp_d2, xp2, ylim_plt=ylim)
        plot_2d_diff(caribic, subs, 'GHG', yp_t1, xp2, yp_d2, xp2, vlims=vlims, ylim_plt=ylim, percent=True)
