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

import cmasher as cmr
import toolpac.calc.dev_binprocessor as bp

import dictionaries as dcts
import tools

# Bin2dPlotter(caribic.df).plot_2d_mxr(dcts.get_subs(short_name='sf6', ID='GHG'),
#                                      dcts.get_coord(hcoord='lat', source='Caribic'),
#                                      dcts.get_coord(vcoord='pt', source='Caribic',
#                                                     tp_def='nan', col_name='int_Theta'))

# Bin2dPlotter(tpause).plot_2d_stdv(
# dcts.get_subs(short_name='detr_sf6', ID='GHG'),
# dcts.get_coord(source='Caribic', hcoord='lat'),
# dcts.get_coord(tp_def='dyn', model='ERA5', pvu=3.5, rel_to_tp=True)
# )

#TODO make sure x and y are actually correct when binning and plotting -> Done?
#TODO might want to implement a logarithmic scale for pressure at some point
#TODO average standard deviations is not weighted and therefore depends on bins
    # -> that will be the case no matter what, since count_limit exclusion also depends on bin sizes

class Bin2dPlotter():
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
        bin_seasonal(subs, xcoord, ycoord, bin_equi2d, x_bin, y_bin)
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
        df = glob_obj.df
        self.count_limit = glob_obj.count_limit

        df['season'] = tools.make_season(df.index.month)
        self.data = {'df' : df} # dataframe
        self.detr = detr
        if not 'detr' in subs_params and detr:
            subs_params.update(dict(detr=detr))
            if 'short_name' in subs_params and not subs_params.get('short_name').startswith('detr_'):
                subs_params.update({'short_name' : 'detr_'+subs_params.get('short_name')})
        self.substances = [s for s in dcts.get_substances(**subs_params)
                           if s.col_name in df.columns and not s.col_name.startswith('d_')]
        self.x_coordinates = [s for s in dcts.get_coordinates(**x_params)
                              if s.col_name in df.columns]
        self.y_coordinates = [s for s in dcts.get_coordinates(**y_params)
                              if s.col_name in df.columns]
        self.kwargs = kwargs
        self._check_input()

    def __repr__(self):
        return '<Bin2dPlotter>\n - subs: {}\n - xcoords: {}\n - ycoords: {}'.format(
            [f'{s.short_name} ({s.ID})' for s in self.substances],
            self.x_coordinates, self.y_coordinates)

    @property
    def df(self) -> pd.DataFrame:
        return self.data['df']

    def _check_input(self) -> bool:
        """ Check if any subs, x, y columns are in data. """
        if not any([(s.col_name in self.df.columns) for s in self.substances]):
            raise KeyError('No substances found in data that fit the parameters. ')
        if not any([(c.col_name in self.df.columns) for c in self.x_coordinates]):
            raise KeyError('No x-coordinate found in data that fit the parameters. ')
        if not any([(c.col_name in self.df.columns) for c in self.y_coordinates]):
            raise KeyError('No y-coordinate found in data that fit the parameters. ')
        return True

    def get_limits(self, subs, xcoord, ycoord, bin_attr='vmean') -> (tuple, tuple, tuple):
        """ Check kwargs for limits, otherwise set default values """
       
        if 'vlims' in self.kwargs:
            vlims = self.kwargs.get('vlims')
        else:
            try: 
                vlims = dcts.get_vlims(subs.short_name, bin_attr=bin_attr)
            except KeyError: 
                if bin_attr=='vmean': 
                    vlims = (np.nanmin(self.df[subs.col_name]), np.nanmax(self.df[subs.col_name]))
                else: 
                    raise KeyError('Could not generate colormap limits.')

        if 'xlims' in self.kwargs:
            xlims = self.kwargs.get('xlims')
        else:
            xlims = (-90, 90)
            # xlims = (np.nanmin(self.df[xcoord.col_name]), np.nanmax(self.df[xcoord.col_name]))

        if 'ylims' in self.kwargs:
            ylims = self.kwargs.get('ylims')
        else:
            ylims = (np.floor(np.nanmin(self.df[ycoord.col_name])),
                     np.ceil(np.nanmax(self.df[ycoord.col_name])))

        return vlims, xlims, ylims

    def get_bsize(self, subs, xcoord, ycoord) -> (float, float):
        """ Get bin size for x- and y-coordinates. """
        vlims, xlims, ylims = self.get_limits(subs, xcoord, ycoord)

        x_bin = self.kwargs.get('x_bin')
        if not x_bin:
            try:
                x_bin = dcts.get_default_bsize(xcoord.hcoord)
            except KeyError:
                x_bin = 10

        y_bin = self.kwargs.get('y_bin')
        if not y_bin:
            try:
                y_bin = dcts.get_default_bsize(ycoord.vcoord)
            except KeyError:
                y_bin = 5 * ( np.ceil((ylims[1]-ylims[0])/10) / 5 )
                if (ylims[1]-ylims[0])/10<1: y_bin=0.5

        return x_bin, y_bin

    def bin_seasonal(self, subs, xcoord, ycoord,
                     bin_equi2d = None,
                     xbsize=None, y_bin=None) -> dict:
        """ Bin the dataframe per season. """
        out_dict = {}
        if not xbsize:
            xbsize = self.get_bsize(subs, xcoord, ycoord)[0]
        if not y_bin:
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

    def make_2d_plot(self, subs, xcoord, ycoord, bin_attr, cmap, **kwargs):
        """ 
        Parameters: 
            bin_attr (str): 'vmean', 'vstdv'
            cmap (plt.colormap)
        """
        if bin_attr not in ['vmean', 'vstdv']: 
            raise KeyError(f'Please pass one of vmean, vstdv as bin_attr, not {bin_attr}. ')
        
        binned_seasonal = self.bin_seasonal(subs, xcoord, ycoord)
        vlims, xlims, ylims = self.get_limits(subs, xcoord, ycoord, bin_attr)
        norm = Normalize(*vlims)
        fig, axs = plt.subplots(2, 2, dpi=250, figsize=(8,9),
                                sharey=True, sharex=True)
        fig.subplots_adjust(top = 1.1)
        
        data_title = 'Mixing ratio' if bin_attr=='vmean' else 'Varibility'
        # fig.suptitle(f'{data_title} of {dcts.make_subs_label(subs)}', y=0.95)

        for season, ax in zip([1,2,3,4], axs.flatten()):
            bin2d_inst = binned_seasonal[season]
            ax.set_title(dcts.dict_season()[f'name_{season}'])

            bci = bin2d_inst.binclassinstance
            
            data = getattr(bin2d_inst, bin_attr) # atttribute: 'vmean', 'vstdv'

            img = ax.imshow(data.T,
                            cmap = cmap, norm=norm,
                            aspect='auto', origin='lower',
                            extent=[bci.xbmin, bci.xbmax, bci.ybmin, bci.ybmax])

            ax.set_xlabel(dcts.make_coord_label(xcoord))
            ax.set_ylabel(dcts.make_coord_label(ycoord))

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
                ax.hlines(0, -90, 90, color='k', ls='dashed', zorder=1, lw=1)

            if kwargs.get('averages') and ycoord.rel_to_tp: 
                tropo_av, strato_av = self.calc_st_averages(bin2d_inst, bin_attr)
                ax.text(**dcts.note_dict(ax, x=0.275, y = 0.9,
                                         s=str('S-Av: {0:.2f}'.format(strato_av)
                                               + '\n' + 'T-Av: {0:.2f}'.format(tropo_av))))

            if kwargs.get('note'):
                ax.text(**dcts.note_dict(ax, s=kwargs.get('note')))

            ax.grid('both', lw=0.4)

        fig.subplots_adjust(right=0.9)
        fig.tight_layout(pad=2.5)

        cbar = fig.colorbar(img, ax = axs.ravel().tolist(), aspect=30, pad=0.09, orientation='horizontal')
        cbar.ax.set_xlabel(data_title+' of '+dcts.make_subs_label(subs))
        
        cbar_vals = cbar.get_ticks()
        cbar_vals = [vlims[0]] + cbar_vals[1:-1].tolist() + [vlims[1]]
        cbar.set_ticks(cbar_vals)

        plt.show()

    def calc_st_averages(self, bin2d_inst, bin_attr = 'vstdv'): 
        """ Calculate tropospheric and stratospheric weighted averages. """
        tropo_mask = bin2d_inst.yintm < 0

        tropo_stdv = bin2d_inst.vstdv[[tropo_mask]*bin2d_inst.nx]
        tropo_stdv = tropo_stdv[~ np.isnan(tropo_stdv)]

        tropo_weights = bin2d_inst.vcount[[tropo_mask]*bin2d_inst.nx]
        tropo_weights = tropo_weights[[i!=0 for i in tropo_weights]]

        try: tropo_weighted_average =  np.average(tropo_stdv, weights = tropo_weights)
        except ZeroDivisionError: tropo_weighted_average = np.nan

        # stratosphere
        strato_mask = bin2d_inst.yintm > 0

        strato_stdv = bin2d_inst.vstdv[[strato_mask]*bin2d_inst.nx]
        strato_stdv = strato_stdv[~ np.isnan(strato_stdv)]

        strato_weights = bin2d_inst.vcount[[strato_mask]*bin2d_inst.nx]
        strato_weights = strato_weights[[i!=0 for i in strato_weights]]

        try: strato_weighted_average = np.average(strato_stdv, weights = strato_weights)
        except ZeroDivisionError: strato_weighted_average = np.nan
        
        return tropo_weighted_average, strato_weighted_average
        
    def plot_2d_mxr(self, subs, xcoord, ycoord,**kwargs):
        bin_attr = 'vmean'
        cmap = plt.cm.viridis # blue-green-yellow
        self.make_2d_plot(subs, xcoord, ycoord, bin_attr, cmap, **kwargs)

    def plot_2d_stdv(self, subs, xcoord, ycoord, averages=True, **kwargs):
        bin_attr = 'vstdv'
        # cmap = cmr.get_sub_cmap('autumn_r', 0.1, 0.9) # yellow-orange-red
        cmap = cmr.get_sub_cmap('summer_r', 0.1, 1) # yellow-green
        self.make_2d_plot(subs, xcoord, ycoord, bin_attr, cmap, averages=averages, **kwargs)

    def plot_mixing_ratios(self, **kwargs):
        """ Plot all possible permutations of subs, xcoord, ycoord. """
        permutations = list(itertools.product(self.substances,
                                              self.x_coordinates,
                                              self.y_coordinates))
        for perm in permutations:
            self.plot_2d_mxr(*perm, **kwargs)

    def plot_stdv_subset(self):
        """ Plot a small subset of standard deviation plots. """
        tps = tools.minimise_tps(dcts.get_coordinates(tp_def='not_nan'))
        xcoords = [dcts.get_coord(col_name='geometry.y'), dcts.get_coord(col_name='int_ERA5_EQLAT')]
        substances = [s for s in dcts.get_substances(ID='GHG') if s.short_name.startswith('detr_')]

        for subs in substances:
            print(subs)
            for tp in tps:
                for xcoord in xcoords:
                    self.plot_2d_stdv(subs, xcoord, tp)

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
