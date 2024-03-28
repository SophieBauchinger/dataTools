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
import matplotlib.patheffects as mpe
import pandas as pd
import itertools

import cmasher as cmr
import toolpac.calc.dev_binprocessor as bp

import dataTools.dictionaries as dcts
from dataTools import tools

import warnings
warnings.filterwarnings("ignore", message="Boolean Series key will be reindexed to match DataFrame index. result = super().__getitem__(key)")

tps = tools.minimise_tps(dcts.get_coordinates(tp_def='not_nan'))
eql = dcts.get_coord(hcoord='eql', model='ERA5')

sf6 = dcts.get_subs(short_name='detr_sf6', ID='GHG')
n2o = dcts.get_subs(short_name='detr_n2o', ID='GHG')

#TODO might want to implement a logarithmic scale for pressure at some point

class BinPlotter():
    """
    Plotting class to facilitate creating binned 2D plots for any choice of x and y.

    Attributes:
        data (pd.DataFrame): Input data
        detr (bool): Use detrended data for all plots
        substances (List[Substance]): All substances to consider when plotting
        x_coordinates (List[Coordinate]): All horizontal coordinates to consider when plotting
        y_coordinates (List[Coordinate]): All vertical coordinates to consider when plotting

    Methods:
        get_limits(subs, xcoord, ycoord)
        get_bins(subs. xcoord, ycoord)
        bin_2d_seasonal(subs, xcoord, ycoord, bin_equi2d, x_bin, y_bin)
        plot_2d_mxr(subs, xcoord, ycoord)
        plot_mixing_ratios()
        plot_mxr_diff(params_1, params2)
        plot_differences()
        plot_2d_stdv(subs, xcoord, ycoord)
        plot_stdv_subset()

    """
    def __init__(self, glob_obj, detr=True,
                 subs_params=dict(),
                 x_params=dict(hcoord='eql'),
                 y_params=dict(vcoord='pt', tp_def='dyn', rel_to_tp=True),
                 **kwargs):
        """ Initialise class instances. """
        self.glob_obj = glob_obj

        self.data = {'df' : glob_obj.df} # dataframe

        filter_tps = kwargs.pop('filter_tps') if 'filter_tps' in kwargs else True
        if filter_tps: 
            tps = tools.minimise_tps(dcts.get_coordinates(tp_def='not_nan'))
            self.filter_non_shared_indices(tps)
            glob_obj.data['df'] = self.df.copy()

        self.count_limit = glob_obj.count_limit

        self.data['df']['season'] = tools.make_season(self.data['df'].index.month)
        
        self.kwargs = kwargs

    def __repr__(self):
        return f'<class eqlat.BinPlotter> - based on \n {self.glob_obj.__repr__()}'
        # return '<Bin2dPlotter>\n - subs: {}\n - xcoords: {}\n - ycoords: {}'.format(
        #     [f'{s.short_name} ({s.ID})' for s in self.substances],
        #     self.x_coordinates, self.y_coordinates)

    @property
    def df(self) -> pd.DataFrame:
        return self.data['df']

    def get_limits(self, subs, xcoord, ycoord=None, bin_attr='vmean'):
        """ Check kwargs for limits, otherwise set default values """

        if 'vlims' in self.kwargs:
            vlims = self.kwargs.get('vlims')
        else:
            try:
                vlims = subs.vlims(bin_attr=bin_attr) # dcts.get_vlims(subs.short_name, bin_attr=bin_attr)
            except KeyError:
                if bin_attr=='vmean':
                    vlims = (np.nanmin(self.df[subs.col_name]), np.nanmax(self.df[subs.col_name]))
                else:
                    raise KeyError('Could not generate colormap limits.')
            except: 
                raise KeyError('Could not generate colormap limits.')

        if 'xlims' in self.kwargs:
            xlims = self.kwargs.get('xlims')
        else:
            xlims = (-90, 90)
            # xlims = (np.nanmin(self.df[xcoord.col_name]), np.nanmax(self.df[xcoord.col_name]))
        
        if ycoord: 
            if 'ylims' in self.kwargs:
                ylims = self.kwargs.get('ylims')
            else:
                ylims = (np.floor(np.nanmin(self.df[ycoord.col_name])),
                         np.ceil(np.nanmax(self.df[ycoord.col_name])))
            return vlims, xlims, ylims
        else: 
            return vlims, xlims

    def get_bsize(self, subs, xcoord, ycoord=None):
        """ Get bin size for x- and y-coordinates. """
        if ycoord: 
            vlims, xlims, ylims = self.get_limits(subs, xcoord, ycoord)
        else: 
            vlims, xlims =  self.get_limits(subs, xcoord, ycoord)

        x_bin = self.kwargs.get('x_bin')
        if not x_bin:
            x_bin = xcoord.get_bsize()
        if not x_bin:
            x_bin = 10

        if ycoord: 
            y_bin = self.kwargs.get('y_bin')
            if not y_bin:
                y_bin = ycoord.get_bsize() # dcts.get_default_bsize(ycoord.vcoord)
            if not y_bin:
                y_bin = 5 * ( np.ceil((ylims[1]-ylims[0])/10) / 5 )
                if (ylims[1]-ylims[0])/10<1: y_bin=0.5
            return x_bin, y_bin
        else: 
            return x_bin

    def filter_non_shared_indices(self, tps):
        """ Filter dataframe for datapoints that don't exist for all tps. """
        cols = [tp.col_name for tp in tps]
        self.data['df'].dropna(subset=cols, how='any', inplace=True)
        return self.data['df']

class BinPlotter1D(BinPlotter):
    """ Single dimensional binning & plotting. """
    def __init__(self, glob_obj, **kwargs):
        super().__init__(glob_obj, **kwargs)

    def bin_1d_seasonal(self, subs, coord, bin_equi1d=None, xbsize=None) -> dict:
        """ Bin the data onto coord for each season. """
        out_dict = {}
        if not xbsize:
            xbsize = self.get_bsize(subs, coord)

        for s in set(self.df['season'].tolist()):
            df = self.df[self.df['season'] == s]

            if coord.col_name == 'geometry.y': # latitude
                x = df.geometry.y
            elif coord.col_name == 'geometry.x':
                x = df.geometry.x
            else:
                x = np.array(df[coord.col_name])

            # get bins as multiples of the bin size
            xbmax = ((np.nanmax(x) // xbsize) + 1) * xbsize
            xbmin = (np.nanmin(x) // xbsize) * xbsize

            if not bin_equi1d:
                bin_equi1d = bp.Bin_equi1d(xbmin, xbmax, xbsize)

            out = bp.Simple_bin_1d(np.array(df[subs.col_name]), x,
                                   bin_equi1d, count_limit=self.count_limit)
            out_dict[s] = out

        return out_dict

    def plot_1d_gradient(self, subs, coord, bin_attr='vmean', **kwargs):
        """ Plot gradient per season onto one plot. """
        bin_dict = self.bin_1d_seasonal(subs, coord)
        
        fig, ax = plt.subplots(dpi=200, figsize=(6,5))
        outline = mpe.withStroke(linewidth=2, foreground='white')

        for s in set(self.df['season'].tolist()):
            vmean = bin_dict[s].vmean
            y = bin_dict[s].xintm
            ax.plot(vmean, y, '-', marker='o', c=dcts.dict_season()[f'color_{s}'],
                      label=dcts.dict_season()[f'name_{s}'],
                      path_effects=[outline])
        
        ax.set_ylabel(coord.label())
        ax.set_xlabel(subs.label())

        xmin = np.nanmin([np.nanmin(bin_inst.vmean) for bin_inst in bin_dict.values()])
        xmax = np.nanmax([np.nanmax(bin_inst.vmean) for bin_inst in bin_dict.values()])

        if coord.rel_to_tp: 
            ax.hlines(0, xmin, xmax, ls='dashed', color='k', lw=1)

        ax.grid('both')
        ax.legend()

class BinPlotter2D(BinPlotter): 
    """ Two-dimensional binning & plotting. """
    def __init__(self, glob_obj, **kwargs): 
        """ Initialise bin plotter. """
        super().__init__(glob_obj, **kwargs)

    def bin_2d_seasonal(self, subs, xcoord, ycoord,
                     bin_equi2d = None,
                     xbsize=None, ybsize=None) -> dict:
        """ Bin the dataframe per season. """
        out_dict = {}
        if not xbsize:
            xbsize = self.get_bsize(subs, xcoord, ycoord)[0]
        if not ybsize:
            ybsize = self.get_bsize(subs, xcoord, ycoord)[1]

        # calculate binned output per season
        for s in set(self.df['season'].tolist()):
            df = self.df[self.df['season'] == s]

            if xcoord.col_name == 'geometry.y': # latitude
                x = df.geometry.y
            else:
                x = np.array(df[xcoord.col_name])

            y = np.array(df[ycoord.col_name])

            # get bins as multiples of the bin size
            xbmax = ((np.nanmax(x) // xbsize) + 1) * xbsize
            xbmin = (np.nanmin(x) // xbsize) * xbsize

            ybmax = ((np.nanmax(y) // ybsize) + 1) * ybsize
            ybmin = (np.nanmin(y) // ybsize) * ybsize

            if not bin_equi2d:
                bin_equi2d = bp.Bin_equi2d(xbmin, xbmax, xbsize,
                                           ybmin, ybmax, ybsize)

            out = bp.Simple_bin_2d(np.array(df[subs.col_name]), x, y,
                                   bin_equi2d, count_limit=self.count_limit)
            out_dict[s] = out

        return out_dict

    def single_2d_plot(self, ax, bin2d_inst, bin_attr, xcoord, ycoord, 
                       cmap, norm, xlims, ylims, **kwargs):
        """ Plot binned data with imshow. """

        bci = bin2d_inst.binclassinstance
        data = getattr(bin2d_inst, bin_attr) # atttribute: 'vmean', 'vstdv'

        img = ax.imshow(data.T,
                        cmap = cmap, norm=norm,
                        aspect='auto', origin='lower',
                        # if not ycoord.vcoord in ['p', 'mxr'] else 'upper',
                        extent=[bci.xbmin, bci.xbmax, bci.ybmin, bci.ybmax] 
                        # if not ycoord.vcoord in ['p', 'mxr'] else [bci.xbmin, bci.xbmax, bci.ybmax, bci.ybmin]
                        )

        ax.set_xlabel(xcoord.label())
        ax.set_ylabel(ycoord.label())

        ax.set_xlim(*xlims)
        #TODO with count_limit > 1, might not have data in all bins - get dynamic bins?
        ax.set_ylim(ylims[0] - bci.ybsize*1.5, ylims[1] + bci.ybsize*1.5)

        ax.set_xticks(np.arange(-90, 90+30, 30)) # stop+30 to include stop

        if bci.ybmin < 0:
            # make sure 0 is included in ticks, evenly spaced away from 0
            ax.set_yticks(list(np.arange(0, abs(bci.ybmin) + bci.ybsize*3, bci.ybsize*3) * -1)
                          + list(np.arange(0, bci.ybmax + bci.ybsize, bci.ybsize*3)))
        else:
            ax.set_yticks(np.arange(bci.ybmin, bci.ybmax + bci.ybsize*3, bci.ybsize*3))
        ax.set_yticks(np.arange(bci.ybmin, bci.ybmax+bci.ybsize, bci.ybsize), minor=True)

        if ycoord.rel_to_tp:
            ax.hlines(0, *xlims, color='k', ls='dashed', zorder=1, lw=1)

        if kwargs.get('averages'):
            if ycoord.rel_to_tp:
                tropo_av, strato_av = self.calc_ts_averages(bin2d_inst, bin_attr)
                ax.text(**dcts.note_dict(ax, x=0.275, y = 0.9,
                                             s=str('S-Av: {0:.2f}'.format(strato_av)
                                                   + '\n' + 'T-Av: {0:.2f}'.format(tropo_av))))
            else: 
                average = self.calc_average(bin2d_inst, bin_attr)
                ax.text(**dcts.note_dict(ax, x=0.225, y = 0.9,
                                             s=str('Av: {0:.2f}'.format(average))))

        if kwargs.get('note'):
            ax.text(**dcts.note_dict(ax, s=kwargs.get('note')))

        ax.grid('both', lw=0.4)
        
        return img

    def seasonal_2d_plots(self, subs, xcoord, ycoord, bin_attr, cmap, **kwargs):
        """
        Parameters:
            bin_attr (str): 'vmean', 'vstdv'
            cmap (plt.colormap)
        """
        # if bin_attr not in ['vmean', 'vstdv']:
        #     raise KeyError(f'Please pass one of vmean, vstdv as bin_attr, not {bin_attr}. ')

        binned_seasonal = self.bin_2d_seasonal(subs, xcoord, ycoord)
        vlims, xlims, ylims = self.get_limits(subs, xcoord, ycoord, bin_attr)
        norm = Normalize(*vlims)
        fig, axs = plt.subplots(2, 2, dpi=250, figsize=(8,9),
                                sharey=True, sharex=True)

        fig.subplots_adjust(top = 1.1)

        data_title = 'Mixing ratio' if bin_attr=='vmean' else 'Varibility'
        # fig.suptitle(f'{data_title} of {subs.label()}', y=0.95)

        for season, ax in zip([1,2,3,4], axs.flatten()):
            bin2d_inst = binned_seasonal[season]
            ax.set_title(dcts.dict_season()[f'name_{season}'])
            
            img = self.single_2d_plot(ax, bin2d_inst, bin_attr, xcoord, ycoord, 
                               cmap, norm, xlims, ylims, **kwargs)

        fig.subplots_adjust(right=0.9)
        fig.tight_layout(pad=2.5)

        cbar = fig.colorbar(img, ax = axs.ravel().tolist(), aspect=30, pad=0.08, 
                            orientation='horizontal', 
                            # location='top', ticklocation='bottom'
                            )
        cbar.ax.set_xlabel(data_title+' of '+subs.label(), 
                           # fontsize=13
                           )

        cbar_vals = cbar.get_ticks()
        cbar_vals = [vlims[0]] + cbar_vals[1:-1].tolist() + [vlims[1]]
        cbar.set_ticks(cbar_vals, ticklocation='bottom')
        
        # cbar.ax.xaxis.set_ticks_position('bottom')

        plt.show()

    def calc_average(self, bin2d_inst, bin_attr='vstdv'):
        """ Calculate weighted overall average. """
        data = getattr(bin2d_inst, bin_attr)
        data = data[~np.isnan(data)]

        weights = bin2d_inst.vcount
        weights = bin2d_inst.vcount[[i!=0 for i in weights]]

        try: weighted_average = np.average(data, weights = weights)
        except ZeroDivisionError: weighted_average = np.nan
        return weighted_average

    def calc_ts_averages(self, bin2d_inst, bin_attr = 'vstdv'):
        """ Calculate tropospheric and stratospheric weighted averages for rel_to_tp coord data. """
        data = getattr(bin2d_inst, bin_attr)

        tropo_mask = bin2d_inst.yintm < 0
        tropo_data = data[[tropo_mask]*bin2d_inst.nx]
        tropo_data = tropo_data[~ np.isnan(tropo_data)]

        tropo_weights = bin2d_inst.vcount[[tropo_mask]*bin2d_inst.nx]
        tropo_weights = tropo_weights[[i!=0 for i in tropo_weights]]

        try: tropo_weighted_average =  np.average(tropo_data, weights = tropo_weights)
        except ZeroDivisionError: tropo_weighted_average = np.nan

        # stratosphere
        strato_mask = bin2d_inst.yintm > 0
        strato_data = data[[strato_mask]*bin2d_inst.nx]
        strato_data = strato_data[~ np.isnan(strato_data)]

        strato_weights = bin2d_inst.vcount[[strato_mask]*bin2d_inst.nx]
        strato_weights = strato_weights[[i!=0 for i in strato_weights]]

        try: strato_weighted_average = np.average(strato_data, weights = strato_weights)
        except ZeroDivisionError: strato_weighted_average = np.nan

        return tropo_weighted_average, strato_weighted_average

    def plot_2d_mxr(self, subs, xcoord, ycoord,**kwargs):
        bin_attr = 'vmean'
        cmap = plt.cm.viridis # blue-green-yellow
        self.seasonal_2d_plots(subs, xcoord, ycoord, bin_attr, cmap, **kwargs)

    def plot_2d_stdv(self, subs, xcoord, ycoord, averages=True, **kwargs):
        bin_attr = 'vstdv'
        # cmap = cmr.get_sub_cmap('autumn_r', 0.1, 0.9) # yellow-orange-red
        cmap = cmr.get_sub_cmap('summer_r', 0.1, 1) # yellow-green
        self.seasonal_2d_plots(subs, xcoord, ycoord, bin_attr, cmap, averages=averages, **kwargs)

    def plot_mixing_ratios(self, **kwargs):
        """ Plot all possible permutations of subs, xcoord, ycoord. """
        permutations = list(itertools.product(self.substances,
                                              # self.x_coordinates,
                                              [dcts.get_coord(col_name='int_ERA5_EQLAT')],
                                              # self.y_coordinates
                                              tools.minimise_tps(dcts.get_coordinates(tp_def='not_nan'))
                                              ))
        for perm in permutations:
            self.plot_2d_mxr(*perm, **kwargs)

    def plot_stdv_subset(self, **subs_kwargs):
        """ Plot a small subset of standard deviation plots. """
        tps = tools.minimise_tps(dcts.get_coordinates(tp_def='not_nan'))
        # xcoords = [dcts.get_coord(col_name='geometry.y'), dcts.get_coord(col_name='int_ERA5_EQLAT')]
        xcoords = [dcts.get_coord(col_name='int_ERA5_EQLAT')]
        substances = [s for s in dcts.get_substances(ID='GHG', **subs_kwargs) if s.short_name.startswith('detr_')]

        for subs in substances:
            print(subs)
            for tp in tps:
                for xcoord in xcoords:
                    self.plot_2d_stdv(subs, xcoord, tp)

    def plot_total_2d(self, subs, xcoord, ycoord, bin_attr='vstdv', **kwargs):
        """ Single 2D plot of varibility of given substance. """
    
        df = self.df

        if xcoord.col_name == 'geometry.y': # latitude
            x = df.geometry.y
        else:
            x = np.array(df[xcoord.col_name])
        y = np.array(df[ycoord.col_name])

        xbsize = self.get_bsize(subs, xcoord, ycoord)[0]
        ybsize = self.get_bsize(subs, xcoord, ycoord)[1]

        # get bins as multiples of the bin size
        xbmax = ((np.nanmax(x) // xbsize) + 1) * xbsize
        xbmin = (np.nanmin(x) // xbsize) * xbsize

        ybmax = ((np.nanmax(y) // ybsize) + 1) * ybsize
        ybmin = (np.nanmin(y) // ybsize) * ybsize

        bin_equi2d = bp.Bin_equi2d(xbmin, xbmax, xbsize,
                                   ybmin, ybmax, ybsize)

        bin2d_inst = bp.Simple_bin_2d(np.array(df[subs.col_name]), x, y,
                               bin_equi2d, count_limit=self.count_limit)
        
        vlims, xlims, ylims = self.get_limits(subs, xcoord, ycoord, bin_attr=bin_attr)
        norm = Normalize(*vlims)
        fig, ax = plt.subplots(dpi=250, figsize=(8,9))
        fig.subplots_adjust(top = 1.1)

        data_title = 'Mixing ratio' if bin_attr=='vmean' else 'Varibility'
        # fig.suptitle(f'{data_title} of {subs.label()}', y=0.95)

        cmap = plt.cm.viridis if bin_attr=='vmean' else cmr.get_sub_cmap('summer_r', 0.1, 1)

        img = self.single_2d_plot(ax, bin2d_inst, bin_attr, xcoord, ycoord, 
                           cmap, norm, xlims, ylims, **kwargs)

        fig.subplots_adjust(right=0.9)
        fig.tight_layout(pad=2.5)

        cbar = fig.colorbar(img, ax = ax, aspect=30, pad=0.09, orientation='horizontal')
        cbar.ax.set_xlabel(data_title+' of '+subs.label())

        cbar_vals = cbar.get_ticks()
        cbar_vals = [vlims[0]] + cbar_vals[1:-1].tolist() + [vlims[1]]
        cbar.set_ticks(cbar_vals)

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

        binned_seasonal_1 = self.bin_2d_seasonal(*params_1, bin_equi2d=bin_equi2d)
        binned_seasonal_2 = self.bin_2d_seasonal(*params_2, bin_equi2d=bin_equi2d)

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

            ax.set_xlabel(xcoord1.label())
            ax.set_ylabel(ycoord1.label())

            if kwargs.get('note'):
                ax.text(**dcts.note_dict(ax, s=kwargs.get('note')))

        fig.subplots_adjust(right=0.9)
        fig.tight_layout(pad=2.5)

        cbar = fig.colorbar(img, ax = axs.ravel().tolist(), aspect=30,
                            pad=0.09, orientation='horizontal')
        cbar.ax.set_xlabel(
            f'{subs1.label()} {xcoord1.label()} {ycoord1.label()} \n vs.\n\
{subs2.label()} {xcoord2.label()} {ycoord2.label()}')
        plt.show()

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
