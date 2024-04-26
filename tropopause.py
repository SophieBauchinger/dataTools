# -*- coding: utf-8 -*-
"""
@Author: Sophie Bauchinger, IAU
@Date Mon Aug 14 14:06:26 2023

Plotting Tropopause heights for different tropopauses different vertical coordinates
"""
import matplotlib.pyplot as plt
import numpy as np
import geopandas
import math
from shapely.geometry import Point
from matplotlib.colors import Normalize
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
import matplotlib.patheffects as mpe
import pandas as pd
from PIL import Image
import glob
import cmasher as cmr

import toolpac.calc.binprocessor as bp # type: ignore
from toolpac.conv.times import datetime_to_fractionalyear as dt_to_fy # type: ignore

import dataTools.dictionaries as dcts
from dataTools.data import TropopauseData, Caribic, GlobalData
from dataTools import tools

world = geopandas.read_file(r'c:\Users\sophie_bauchinger\Documents\GitHub\110m_cultural_511\ne_110m_admin_0_map_units.shp')
# geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
vlims = {'p':(100,500), 'pt':(300, 350), 'z':(6.5,14), 'mxr': (290, 330)}
rel_vlims = {'p':(-100,100), 'pt':(-30, 40), 'z':(-1,2.5)}
count_limit = 5

#TODO Add disclaimer to dyn and cpt to show reduced latitude ranges 

def substance_strato_tropo(DataObj, tp, subs):
    fig, (ax1, ax2) = plt.subplots(2, dpi=250, figsize=(15, 9), sharex=True)
    # tp = dcts.get_coord(col_name='int_ERA5_D_TROP1_PRESS')
    fig.suptitle(tp.label(filter_label=True) + ' tropopause', fontsize=15, y=0.95)
    
    subs_data = DataObj.df[subs.col_name]
    df_sorted = DataObj.df_sorted
    
    if 'tropo_'+tp.col_name not in df_sorted.columns: 
        raise KeyError(f'{tp} not found in df_sorted')
    
    t_subs = subs_data[df_sorted['tropo_' + tp.col_name]].dropna()
    s_subs = subs_data[df_sorted['strato_' + tp.col_name]].dropna()
    
    ax1.set_title('Stratospheric')
    ax1.scatter(subs_data.index, subs_data, color='grey', label='background')
    ax1.scatter(s_subs.index, s_subs, color='blue', label='strato')
    ax1.set_ylabel(subs.label())
    
    ax2.set_title('Tropospheric')
    ax2.scatter(subs_data.index, subs_data, color='grey', label='background')
    ax2.scatter(t_subs.index, t_subs, color='orange', label='tropo')
    ax2.set_ylabel(subs.label())
    
    # ax1.legend(); ax2.legend()
    plt.show()

#%% Define TropopausePlotter
class TropopausePlotter(GlobalData): 
    """ Define plotting functionality for GlobalData object. """
    def set_tps(self, **kwargs): 
        """ Set default tropopause definitions to plot. """
        coords = self.coordinates
        self.tps = [tp for tp in coords if (
            str(tp.tp_def)!='nan' and 
            tp.col_name in [c.col_name for c in dcts.get_coordinates(**kwargs)])]
        self.tps.sort(key=lambda x: x.tp_def)
        return self.tps

    def tp_height_global_scatter(self, rel=False):
        """ """
        
        tps = tools.minimise_tps(self.coordinates)

        for tp in tps: 
            fig, ax = plt.subplots(dpi=150, figsize=(10,5))
            ax.set_title(tp.label(True))

            bci =  bp.Bin_equi2d(-180, 180, self.grid_size, 
                                 -90, 90, self.grid_size)

            out = bp.Simple_bin_2d(self.df[tp.col_name],
                                   self.df.geometry.x, 
                                   self.df.geometry.y, 
                                   bci, count_limit = self.count_limit)

            world.boundary.plot(ax=ax, color='black', linewidth=0.3)
            # ax.set_title(dcts.dict_season()[f'name_{s}'])
            cmap = 'viridis_r' if tp.vcoord=='p' else 'viridis'

            v_lims = vlims[tp.vcoord] if not rel else rel_vlims[tp.vcoord]
            norm = Normalize(*v_lims)

            # df_r.plot(tp.col_name, cmap=cmap, legend=True, ax=ax)
            img = ax.imshow(out.vmean.T, cmap = cmap, norm=norm, origin='lower',
                            extent = [bci.xbmin, bci.xbmax, bci.ybmin, bci.ybmax])
            cbar = plt.colorbar(img, ax=ax, pad=0.08,
                                orientation='vertical', extend='both') # colorbar
            cbar.ax.set_xlabel('{}{} [{}]'.format(
                '' if not rel else '$\Delta $',
                tp.vcoord if not tp.vcoord=='pt' else '$\Theta$',
                tp.unit))
            ax.set_ylim(-90, 90); ax.set_xlim(-180, 180)
            ax.set_ylabel('Longitude [°E]')
            ax.set_xlabel('Latitude [°N]')
            plt.show()

    def tp_height_seasonal_global_scatter(self, savefig=False, year=None, rel = False, 
                   minimise_tps = True):
        """ 2D global scatter of tropopause height.
        Parameters:
            save (bool): save plot to pdir instead of plotting
            year (float): select single specific year to plot / save
        """
        pdir = r'C:\Users\sophie_bauchinger\sophie_bauchinger\Figures\tp_scatter_2d'

        tps = dcts.get_coordinates(tp_def='not_nan', rel_to_tp=rel)
        if minimise_tps: 
            tps = tools.minimise_tps(tps)
        
        for tp in tps:
            if tp.tp_def=='cpt' or tp.vcoord not in ['pt', 'p', 'z']: 
                continue
            fig, axs = plt.subplots(2,2,dpi=150, figsize=(10,5))
            # fig.suptitle(f'{tp.col_name} - {tp.long_name}')
            fig.suptitle(tp.label(True))
            if year: fig.text(0.9, 0.95, f'{year}',
                              bbox = dict(boxstyle='round', facecolor='white',
                                          edgecolor='grey', alpha=0.5, pad=0.25))
            for s,ax in zip([1,2,3,4], axs.flatten()):
                if year: df_r = self.sel_year(year).sel_season(s).df
                else: df_r = self.sel_season(s).df
                if df_r.empty: continue
                df_r.geometry = [Point(pt.y,pt.x) for pt in df_r.geometry]

                lon = np.array([df_r.geometry[i].x for i in range(len(df_r.index))])
                lat = np.array([df_r.geometry[i].y for i in range(len(df_r.index))])
                bci = bp.Bin_equi2d(np.nanmin(lon), np.nanmax(lon), self.grid_size,
                                    np.nanmin(lat), np.nanmax(lat), self.grid_size)
                out = bp.Simple_bin_2d(df_r[tp.col_name], lon, lat, bci,
                                       count_limit = self.count_limit)

                world.boundary.plot(ax=ax, color='black', linewidth=0.3)
                ax.set_title(dcts.dict_season()[f'name_{s}'])
                cmap = 'viridis_r' if tp.vcoord=='p' else 'viridis'

                norm = Normalize(*vlims[tp.vcoord]) # colormap normalisation
                # df_r.plot(tp.col_name, cmap=cmap, legend=True, ax=ax)
                img = ax.imshow(out.vmean, cmap = cmap, origin='lower', norm=norm,
                            extent = [bci.ybmin, bci.ybmax, bci.xbmin, bci.xbmax])
                cbar = plt.colorbar(img, ax=ax, pad=0.08,
                                    orientation='vertical', extend='both') # colorbar
                cbar.ax.set_xlabel(f'{tp.vcoord} [{tp.unit}]')
                ax.set_ylim(-90, 90); ax.set_xlim(-180, 180)
            fig.tight_layout()
            if savefig:
                plt.savefig(pdir+'\{}{}.png'.format(
                    tp.col_name, '_'+str(year) if year else ''))

            plt.show()
            plt.close()

    def tp_height_latitude_multiple_definitions(self, tps=None, rel=False): 
        """ """
        bci = bp.Bin_equi1d(-90, 90, self.grid_size)
        outline = mpe.withStroke(linewidth=4, foreground='white')
        
        if tps is None: 
            tps = tools.minimise_tps(self.coordinates, rel=rel) # CANNOT ask for this 

        fig, ax = plt.subplots(dpi=250, figsize=(7,4))

        data = self.df
        data = data[data.index.isin(self.get_shared_indices(tps, df=True))]

        for tp in tps:
            label = tp.label(True)

            bin1d = bp.Simple_bin_1d(data[tp.col_name],
                                  np.array(data.geometry.y), bci,
                                  count_limit = self.count_limit)
            plot_kwargs = dict(
                lw=2.5, path_effects = [outline])
            
            plot_kwargs.update(dict(
                label = label, 
                # color='dimgray', 
                # ls = 'dashed', 
                zorder=5))

            ax.plot(bin1d.xintm, bin1d.vmean, **plot_kwargs)

            if tp.vcoord in ['mxr', 'p']: 
                ax.invert_yaxis()
                if tp.vcoord=='p': 
                    ax.set_yscale('log' if not tp.rel_to_tp else 'symlog')


            # ax.set_ylabel(tp.label())# if not tp.rel_to_tp else f'$\Delta\,${vc_label[tp.vcoord]} [{tp.unit}] - ' + ylabel)
            ax.set_ylabel(tp.label(coord_only=True))
            ax.set_xlabel(dcts.axis_label('lat'))

            ax.set_xticks(np.arange(-90, 85, 5), minor=True)

            if tp.vcoord=='pt' and tp.rel_to_tp: 
                ax.set_yticks(np.arange(-30, 60, 15))
                ax.set_ylim(-35, 60)
            if tp.vcoord=='z' and tp.rel_to_tp: 
                ax.set_ylim(-2.5,4.1)
            ax.grid(True, ls='dotted')
            ax.legend(loc='upper left' if tp.rel_to_tp else 'upper right', 
                      fontsize=9)
            
            if tp.rel_to_tp: 
                zero_lines = np.delete(ax.get_ygridlines(), ax.get_yticks()!=0)
                for l in zero_lines: 
                    l.set_color('xkcd:dark grey')
                    l.set_linestyle('-.')

    def tp_height_latitude_binned_hlines(self, rel=False, note=''): 
        """ Plot latitude-binned tropoause heights with horizontal lines over bin. """
        bci = bp.Bin_equi1d(-90, 90, self.grid_size)
        outline = mpe.withStroke(linewidth=4, foreground='white')

        for tp in tools.minimise_tps(dcts.get_coordinates(tp_def='not_nan', rel_to_tp=rel)):
            fig, ax = plt.subplots(dpi=250)
            # ax.set_title('Vertical extent of'+tp.label(True)+' Tropopause')
            ax.set_xlim(30, 90)

            for s in ['av', 1,2,3,4]:
                data = self.sel_season(s).df if not s=='av' else self.df
                bin1d = bp.Simple_bin_1d(data[tp.col_name],
                                      np.array(data.geometry.y), bci,
                                      count_limit = self.count_limit)
                if s!='av': 
                    plot_kwargs = dict(
                        color=dcts.dict_season()[f'color_{s}'], 
                        lw=3, path_effects = [outline])
                else: 
                    plot_kwargs = dict(
                        color='dimgray', 
                        lw=2, ls = 'dashed')

                for i in range(len(bin1d.xmean)): 
                    if bin1d.vmean[i] == np.nan: 
                        continue
                    trace = ax.hlines(bin1d.vmean[i], 
                                      bin1d.xint[i], bin1d.xint[i]+bin1d.xbsize,
                                      **plot_kwargs)
                    if not s=='av': 
                        ax.fill_between([bin1d.xint[i], bin1d.xint[i]+bin1d.xbsize], 
                                        [bin1d.vmean[i] - bin1d.vstdv[i]]*2, 
                                        [bin1d.vmean[i] + bin1d.vstdv[i]]*2,
                                        alpha=0.2, color=plot_kwargs['color'])
                ax.set_xticks(np.arange(30, 95, 5), minor=True)
                ax.grid(True, ls='dotted')
                ax.legend(loc='lower right' if tp.rel_to_tp else 'upper right')

                if s=='av' and tp.vcoord in ['mxr', 'p']: 
                    ax.invert_yaxis()
                    if tp.vcoord=='p': 
                        ax.set_yscale('log' if not tp.rel_to_tp else 'symlog')

                # ax.text(0.05, 0.05,
                #         tp.label(True),
                #         transform=ax.transAxes, verticalalignment='bottom',
                #         bbox = dict(boxstyle='round', facecolor='white',
                #                     edgecolor='grey', alpha=0.5, pad=0.25))
            
                # get last hline and set label for the legend
                trace.set_label(dcts.dict_season()[f'name_{s}'] if not s=='av' else 'Average')

            ylabel = tp.label(True) # dcts.axis_label(tp.vcoord) +' of '+ tp.label(True)
            vc_label = {'pt': '$\Theta$', 'z':'z', 'mxr':'mxr'}
            ax.set_ylabel(ylabel if not tp.rel_to_tp else f'$\Delta\,${vc_label[tp.vcoord]} [{tp.unit}] - ' + ylabel)
            ax.set_xlabel(dcts.axis_label('lat'))

            plt.show()

    def tp_height_latitude_binned_yearly(self, rel=False, note=''): 
        """ Plot average tropopause height over available years. """
        bci = bp.Bin_equi1d(-90, 90, self.grid_size)
        outline = mpe.withStroke(linewidth=2, foreground='white')

        # ncols = max(4, len(self.years)) 
        # nrows = math.ceil(len(self.years)/ncols)

        cmap = plt.cm.cool
        norm = Normalize(min(self.years), max(self.years))
        
        for tp in tools.minimise_tps(dcts.get_coordinates(tp_def='not_nan', rel_to_tp=rel)):
            fig, ax = plt.subplots(dpi=250)#nrows, ncols)

            ax.set_xlim(30, 90)
            if tp.vcoord=='pt' and not rel: 
                ax.set_ylim(290, 380)
            elif tp.vcoord=='z' and not rel: 
                ax.set_ylim(6, 14)
            
            for year in self.years: 
                data = self.sel_year(year).df
                if data.empty: 
                    continue
                bin1d = bp.Simple_bin_1d(data[tp.col_name],
                                         np.array(data.geometry.y), bci,
                                         count_limit = self.count_limit)
                ax.plot(bin1d.xintm, bin1d.vmean, 
                        color = cmap(norm(year)),
                        label=str(year),
                        lw=1, path_effects = [outline])
            ax.set_title(tp.label(True))

            ylabel = tp.label(True) # dcts.axis_label(tp.vcoord) +' of '+ tp.label(True)
            vc_label = {'pt': '$\Theta$', 'z':'z', 'mxr':'mxr'}
            ax.set_ylabel(ylabel if not tp.rel_to_tp else f'$\Delta\,${vc_label[tp.vcoord]} [{tp.unit}] - ' + ylabel)
            ax.set_xlabel(dcts.axis_label('lat'))

            ax.grid(True, ls='dotted')
            ax.legend()
            plt.show()
            
    def tp_height_seasonal_latitude_binned(self, rel=False, note='', 
                                          seasonal_stdvs=False): 
        """ Plot average and seasonal tropopause heights, optionally with standard deviation. """
        bci = bp.Bin_equi1d(-90, 90, self.grid_size)
        outline = mpe.withStroke(linewidth=4, foreground='white')
        
        tps = self.tps # tools.minimise_tps(dcts.get_coordinates(tp_def='not_nan', rel_to_tp=rel, source='Caribic'))

        for tp in tps:
            fig, ax = plt.subplots(dpi=250, figsize=(7*0.8,4*0.8))
            ax.set_title(tp.label(True))
            # ax.set_title('Vertical extent of'+tp.label(True)+' Tropopause')
            
            if tp.vcoord=='pt' and not rel: 
                ax.set_ylim(290, 380)
            elif tp.vcoord=='z' and not rel: 
                ax.set_ylim(6, 14)

            for s in ['av', 1,2,3,4]:
                data = self.sel_season(s).df if not s=='av' else self.df
                data = data[data.index.isin(self.get_shared_indices(tps, df=True))]
                
                bin1d = bp.Simple_bin_1d(data[tp.col_name],
                                      np.array(data.geometry.y), bci,
                                      count_limit = self.count_limit)
                plot_kwargs = dict(
                    lw=3, path_effects = [outline])
                
                if s!='av': 
                    plot_kwargs.update(dict(
                        label = dcts.dict_season()[f'name_{s}'], 
                        color=dcts.dict_season()[f'color_{s}']))
                    
                    if seasonal_stdvs: 
                        ax.fill_between(bin1d.xintm, 
                                        bin1d.vmean - bin1d.vstdv, 
                                        bin1d.vmean + bin1d.vstdv,
                                        alpha=0.2, color=plot_kwargs['color'])
                else: 
                    plot_kwargs.update(dict(
                        label = 'Average', 
                        color='dimgray', 
                        ls = 'dashed', zorder=5))
                    if not seasonal_stdvs: 
                        ax.fill_between(bin1d.xintm, 
                                        bin1d.vmean - bin1d.vstdv, 
                                        bin1d.vmean + bin1d.vstdv,
                                        alpha=0.13, color=plot_kwargs['color'])

                ax.plot(bin1d.xintm, bin1d.vmean, **plot_kwargs)

                if s=='av' and tp.vcoord in ['mxr', 'p']: 
                    ax.invert_yaxis()
                    if tp.vcoord=='p': 
                        ax.set_yscale('log' if not tp.rel_to_tp else 'symlog')

            ax.set_xlim(30, 80)

            if note: 
                ax.text(0.05, 0.05, note,
                        transform=ax.transAxes, verticalalignment='bottom',
                        bbox = dict(boxstyle='round', facecolor='white',
                                    edgecolor='grey', alpha=0.5, pad=0.25))

            # ax.set_ylabel(tp.label())# if not tp.rel_to_tp else f'$\Delta\,${vc_label[tp.vcoord]} [{tp.unit}] - ' + ylabel)
            ax.set_ylabel(tp.label(coord_only=True))
            ax.set_xlabel(dcts.axis_label('lat'))

            ax.set_xticks(np.arange(30, 85, 5), minor=True)

            if tp.vcoord=='pt' and tp.rel_to_tp: 
                ax.set_yticks(np.arange(-30, 60, 15))
                ax.set_ylim(-35, 60)
            if tp.vcoord=='z' and tp.rel_to_tp: 
                ax.set_ylim(-2.5,4.1)
            ax.grid(True, ls='dotted')
            ax.legend(loc='upper left' if tp.rel_to_tp else 'upper right', 
                      fontsize=9)
            
            if tp.rel_to_tp: 
                zero_lines = np.delete(ax.get_ygridlines(), ax.get_yticks()!=0)
                for l in zero_lines: 
                    l.set_color('xkcd:dark grey')
                    l.set_linestyle('-.')

    def plot_subs_sorted(self, substances, vcoords=['p', 'pt', 'z', 'mxr'],
                         detr = False, minimise_tps = True):
        """ Plot timeseries of subs mixing ratios with strato / tropo colours. """
        if not substances:
            substances = dcts.get_substances(source='EMAC') + dcts.get_substances(source='Caribic')
            substances = [s for s in substances if not s.col_name.startswith('d_')]
        elif isinstance(substances, dcts.Substance): substances = [substances]
        if not isinstance(substances, list): 
            raise Warning('Cannot work like this. Pls supply substances as list.')

        if not 'df_sorted' in self.data: self.create_df_sorted(save=True)
        df_sorted = self.df_sorted.copy()

        cols = [c[6:] for c in df_sorted.columns if c.startswith('tropo_')]
        tps = [tp for tp in dcts.get_coordinates(tp_def='not_nan') if tp.col_name in cols]
        if minimise_tps: tps = tools.minimise_tps(tps)

        if detr: 
            for subs_short in set([s.short_name for s in substances if not s.startswith(('d', 'detr'))]):
                self.detrend_substance(subs_short)
        else: 
            substances = [s for s in substances if not s.short_name.startswith('detr_')]

        for subs in substances: # new plot for each substance 
            if not subs.col_name in self.df.columns:
                print(f'{subs.col_name} not found in data')
                continue

            if minimise_tps: 
                if len(tps)==0: continue
                fig, axs = plt.subplots(math.ceil(len(tps)/2), 2, dpi=200,
                                        figsize=(7, math.ceil(len(tps)/2)*2),
                                        sharey=True, sharex=True)
                if len(tps)%2: axs.flatten()[-1].axis('off')
                fig.suptitle(f'{subs.col_name}')

                for tp, ax in zip(tps, axs.flatten()):
                    tp_tropo = self.df[df_sorted['tropo_'+tp.col_name] == True] #.dropna(axis=0, subset=[tp.col_name])
                    tp_strato = self.df[df_sorted['strato_'+tp.col_name] == True] #.dropna(axis=0, subset=[tp.col_name])

                    ax.set_title(tp.label(filter_label=True), fontsize=8)
                    ax.scatter(tp_strato.index, tp_strato[subs.col_name],
                                c='grey',  marker='.', zorder=0, label='strato')
                    ax.scatter(tp_tropo.index, tp_tropo[subs.col_name],
                                c='xkcd:kelly green',  marker='.', zorder=1, label='tropo')

                fig.autofmt_xdate()
                fig.tight_layout()
                fig.subplots_adjust(top = 0.8 + math.ceil(len(tps))/150)
                lines, labels = axs.flatten()[0].get_legend_handles_labels()
                fig.legend(lines, labels, loc='upper center', ncol=2,
                           bbox_to_anchor=[0.5, 0.94])
                plt.show()
            
            # new figure for each vcoord, otherwise overloading the plots
            if not minimise_tps: 
                for vcoord in vcoords:
                    tps_vc = [tp for tp in tps if tp.vcoord==vcoord]
                    if len(tps_vc)==0: continue
                    fig, axs = plt.subplots(math.ceil(len(tps_vc)/2), 2, dpi=200,
                                            figsize=(7, math.ceil(len(tps_vc)/2)*2),
                                            sharey=True, sharex=True)
                    if len(tps_vc)%2: axs.flatten()[-1].axis('off')
                    fig.suptitle(f'{subs.col_name}')
        
                    for tp, ax in zip(tps_vc, axs.flatten()):
                        tp_tropo = self.df[df_sorted['tropo_'+tp.col_name] == True] #.dropna(axis=0, subset=[tp.col_name])
                        tp_strato = self.df[df_sorted['strato_'+tp.col_name] == True] #.dropna(axis=0, subset=[tp.col_name])
    
                        ax.set_title(tp.label(filter_label=True), fontsize=8)
                        ax.scatter(tp_strato.index, tp_strato[subs.col_name],
                                    c='grey',  marker='.', zorder=0, label='strato')
                        ax.scatter(tp_tropo.index, tp_tropo[subs.col_name],
                                    c='xkcd:kelly green',  marker='.', zorder=1, label='tropo')
    
                    fig.autofmt_xdate()
                    fig.tight_layout()
                    fig.subplots_adjust(top = 0.8 + math.ceil(len(tps_vc))/150)
                    lines, labels = axs.flatten()[0].get_legend_handles_labels()
                    fig.legend(lines, labels, loc='upper center', ncol=2,
                               bbox_to_anchor=[0.5, 0.94])
                    plt.show()

    def show_strato_tropo_vcounts(self, tps=None, shared=True, note='', **tp_kwargs): 
        """ Bar plots of data point allocation for multiple tp definitions. """
        if not tps: 
            tps = self.tps
        
        if shared is False: 
            tropo_counts = self.calc_ratios(ratios=False) # dataframe
            # tropo_counts = tropo_counts[[tp.col_name for tp in tps 
            #                              if tp.col_name in tropo_counts.columns]]
            note += ' all'
        elif shared is True: 
            tropo_counts = self.calc_shared_ratios(tps = tps, ratios=False)
            # note += ' shared'
        elif shared == 'No': 
            tropo_counts = self.calc_non_shared_ratios(tps = tps, ratios=False)
            note += ' non-shared'
        else: 
            raise KeyError(f'Invalid value for shared: {shared}')

        tropo_counts = tropo_counts[[tp.col_name for tp in tps 
                                     if tp.col_name in tropo_counts.columns]]
        tps = [tp for tp in tps if tp.col_name in tropo_counts.columns] # not all tps have ratios (chekkkk)
        tp_labels = [tp.label(filter_label=True) for tp in tps]
        
        tropo_bar_vals = [tropo_counts[i].loc[True] for i in tropo_counts.columns]
        strato_bar_vals = [tropo_counts[i].loc[False] for i in tropo_counts.columns]

        # cols, tp_labels = map(list, zip(*[('tropo_'+tp.col_name, tp.label(filter_label=True)) 
        #                                 for tp in tps]))

        fig, (ax_t, ax_label, ax_s) = plt.subplots(1, 3, dpi=400, 
                                        figsize=(9,4), sharey=True)
        
        # fig.suptitle('Number of tropospheric / stratospheric datapoints')
        fig.subplots_adjust(top=0.85)
        ax_t.set_title('# Tropospheric points', fontsize=9, loc='right')
        ax_s.set_title('# Stratospheric points', fontsize=9, loc='left')

        bar_labels = tp_labels
        
        bar_params = dict(align='center', edgecolor='k',  rasterized=True,
                          alpha=0.65, zorder=10)
        nums = range(len(bar_labels))

        t_bars = ax_t.barh(nums, tropo_bar_vals, **bar_params)
        s_bars = ax_s.barh(nums, strato_bar_vals, **bar_params)
        for i in nums: 
            ax_label.text(0.5, i, bar_labels[i], 
                          horizontalalignment='center', verticalalignment='center')
        ax_label.axis('off')
        
        maximum = np.nanmax(tropo_bar_vals+strato_bar_vals)
        minimum = np.nanmin(tropo_bar_vals+strato_bar_vals)
        for decimal_place in [4,3,2,1,0]:
            if all(i>np.round(minimum, decimal_place) for i in tropo_bar_vals+strato_bar_vals): 
                minimum = np.round(minimum, decimal_place)
            else: 
                break
        padding = (maximum-minimum)/3
        
        ax_t.set_xlim(maximum +padding , minimum-padding if not minimum-padding<0 else 0)
        ax_s.set_xlim(minimum-padding if not minimum-padding<0 else 0, maximum +padding)
        
        ax_t.grid('both', ls='dotted')
        ax_s.grid('both', ls='dotted')
        # ax_t.axis('off')
        # ax_s.axis('off')
        
        ax_t.bar_label(t_bars, ['{0:.0f}'.format(t_val) for t_val in tropo_bar_vals], 
                       padding=2)
        ax_s.bar_label(s_bars, ['{0:.0f}'.format(s_val) for s_val in strato_bar_vals], 
                       padding=2)

        for ax in [ax_t, ax_s]: 
            ax.yaxis.set_major_locator(ticker.NullLocator())
        if note: ax.text(s=note, **dcts.note_dict(ax, y=1.1))
        fig.subplots_adjust(wspace=0)

    def show_ratios(self, tps, shared=False, note='', **tp_kwargs):
    # , as_subplot=False, ax=None, single_tp_def=None, group_vc=False,
    #                 unity_line=True, minimise_tps=True, note='', **tp_kwargs):
        """ Plot ratio of tropo / strato datapoints on a horizontal bar plot """
        tropo_counts = self.calc_ratios() # dataframe
        if shared: 
            tropo_counts = self.calc_shared_ratios()
        if shared == 'No': 
            tropo_counts = self.calc_non_shared_ratios()

        ratios = tropo_counts.loc['ratios']
        tropo_counts.drop(index='ratios', inplace=True)
        n_values = [tropo_counts[i].loc[True] + tropo_counts[i].loc[False] 
                    for i in tropo_counts.columns]

        bar_labels = ['{:.2f} (n={:.2f}'.format(r, n) for r,n in zip(ratios, n_values)]
        # bar_labels = ['{:.2f} (n={})'.format(r,nr) for r, nr in zip(ratios, n_values)]

        # find coordinates for each column
        
        if not tps: 
            tps = [dcts.get_coord(col_name = c) for c in tropo_counts.columns 
                   if c in [c.col_name for c in dcts.get_coordinates() 
                            if c.tp_def not in ['combo', 'cpt']]]
            tps = tools.minimise_tps(tps)
        if len(tps)==0: raise Exception('No tps found that fit the criteria.')

        # make sure cols and labels are related 
        cols, tp_labels = map(list, zip(*[('tropo_'+tp.col_name, tp.label(filter_label=True)) 
                                        for tp in tps]))

        fig, ax = plt.subplots(dpi=240, figsize=(8,6))
        ax.set_title(f'Ratio of tropospheric / stratospheric datapoints in {self.ID}')

        # tp_defs = set([tp.tp_def for tp in tps])
        # ax.grid(True, axis='x', c='k', alpha=0.3)
        ax.axvline(1, linestyle='--', color='k', alpha=0.3, zorder=0, lw=1) # vertical lines
        ax.set_axisbelow(True)

        for tp in tps:
            color = dcts.dict_tps()[f'color_{tp.tp_def}']
            ratio = ratios[tp.col_name]
            n_value = tropo_counts[tp.col_name].loc[True] + tropo_counts[tp.col_name].loc[False] 
            label = tp.label(True)
            
            bars = ax.barh(label, ratio, rasterized=True, color=color, alpha=0.9)
            bar_labels = ['{:.2f} (n={:.0f})'.format(r,n) for r,n in zip([ratio], [n_value])]

            ax.bar_label(bars, bar_labels, fmt='%.3g', padding=1)
            ax.set_xlim(0,3) # np.nanmax(ratios)*1.2) #4.5)

        if note: ax.text(s=note, **dcts.note_dict(ax))

        fig.tight_layout()
        plt.show()

    def show_ratios_seasonal(self, note='', **tp_kwargs):
        """ Create 2x2 plot for all trop/strat ratios per season """
        fig, axs = plt.subplots(2,2, figsize=(12,12), dpi=350, sharey=True, sharex=True)
        fig.suptitle('Ratio of tropospheric / stratospheric datapoints', fontsize=20)
        for season, ax in zip([1,2,3,4], axs.flatten()):
            self.sel_season(season).show_ratios(as_subplot=True, ax=ax, note=note, **tp_kwargs)
            ax.set_title(dcts.dict_season()[f'name_{season}'])
        fig.tight_layout()
        plt.show()

    def make_stdv_table(self, ax, subs, stdv_df, **kwargs):
        """ Create table for values of standard deviation in troposphere / stratosphere.

        Parameters:
            ax (matplotlib.axes.Axes): Axis to be plotted on
            subs (Substance): Substance for which standard deviation was calculated
            stdv_df (pd.DataFrame): Dataframe containing standard deviation data

        Returns:
            edited matplotlib.axes.Axes object, matplotlib.axes.table instance 
        """
        if 'prec' in kwargs:
            prec = kwargs.get('prec')
        else:  
            prec = 1 if kwargs.get('rel') else 2

        stdv_df = stdv_df.astype(float).round(prec)
        stdv_df.sort_index()
        
        cellValues = [[ '{:.{prec}f}'.format(j, prec=prec) 
                for j in i] for i in stdv_df.values]
              
        rowLabels = [dcts.get_coord(col_name=c).label(True) for c in stdv_df.index]
        colLabels = [f'Troposphere [{subs.unit}]', f'Stratosphere [{subs.unit}]'
                     ] if not kwargs.get('rel') else [
                         'Troposphere [%]', 'Stratosphere [%]']

        norm_t = Normalize(np.min(stdv_df.values[:,0]) * 0.95, np.max(stdv_df.values[:,0]) * 1.15)
        cmap_t = dcts.dict_colors()['vstdv_tropo']

        norm_s = Normalize(np.min(stdv_df.values[:,1]) * 0.95, np.max(stdv_df.values[:,1]) * 1.15)
        cmap_s = dcts.dict_colors()['vstdv_strato']

        cellColors = pd.DataFrame([[cmap_t(norm_t(i)) for i in stdv_df.values[:,0]], 
                                [cmap_s(norm_s(i)) for i in stdv_df.values[:,1]]]
                                ).transpose().values

        table = ax.table(cellValues, 
                         rowLabels = rowLabels, 
                         # rowColours = ['xkcd:light grey']*len(rowLabels),
                         colLabels = colLabels,
                         # colColours = ['xkcd:light grey']*len(colLabels),
                         cellLoc = 'center',
                         loc='center',
                         cellColours = cellColors if not kwargs.get('NoColor') else None,
                         )
        table.set_fontsize(15)
        table.scale(1,3)
        ax.axis('off')

        return ax, table

    def strato_tropo_stdv_table(self, subs, tps=None, **kwargs):
        """ Creates a table with the """
        stdv_df = self.strato_tropo_stdv(subs, tps)
        stdv_df = stdv_df[[c for c in stdv_df.columns if 'stdv' in c]]
        
        if kwargs.get('rel'): 
            stdv_df = stdv_df[[c for c in stdv_df.columns if 'rel' in c]]
        else: 
            stdv_df = stdv_df[[c for c in stdv_df.columns if 'rel' not in c]]
              
        fig, ax = plt.subplots(dpi=250)
        ax, table = self.make_stdv_table(ax, subs, stdv_df, **kwargs)
        fig.show()

    def strato_tropo_stdv_mean_seasonal_table(self, subs, tps=None, **kwargs):
        """ Calculate and display table of variability per season and RMS.
 
        Parameters: 
            subs (Substance): Substance for which to calculate variability
            tps (List[Coordinate]): Tropopause definitions to calculate atmos.layer variability for
        """
        self.df['season'] = tools.make_season(self.df.index.month)
        stdv_df_dict = {}

        for season in set(self.df.season):
            stdv_df = self.sel_season(season).strato_tropo_stdv(subs, tps)
            stdv_df = stdv_df[[c for c in stdv_df.columns if 'stdv' in c]]

            if kwargs.get('rel'): 
                stdv_df = stdv_df[[c for c in stdv_df.columns if 'rel' in c]]
            else: 
                stdv_df = stdv_df[[c for c in stdv_df.columns if 'rel' not in c]]
            
            stdv_df_dict[season] = stdv_df.rename(columns = {c:c +f'_{season}' for c in stdv_df.columns})
                
            fig, ax = plt.subplots(dpi=250)
            ax, table = self.make_stdv_table(ax, subs, stdv_df, **kwargs)
            
            ax.set_title('Variability of {} in {}'.format(
                subs.label(),
                dcts.dict_season()[f'name_{season}'] ))
            fig.show()
        
        # Calculate average seasonal averages 
        df = pd.concat(stdv_df_dict.values(), axis=1)
        strato_cols = [c for c in df.columns if 'strato_stdv' in c]
        tropo_cols = [c for c in df.columns if 'tropo_stdv' in c]

        # Average of seasonal relative standard deviation
        df['strato_stdv_av'] = df[strato_cols].sum(axis=1) / len(stdv_df_dict)
        df['tropo_stdv_av'] = df[tropo_cols].sum(axis=1) / len(stdv_df_dict)
        
        df_av = df[['tropo_stdv_av', 'strato_stdv_av']]
        df_av = df_av.astype(float).round(3 if kwargs.get('rel') else 3)
        
        fig, ax = plt.subplots(dpi=250)
        ax, table = self.make_stdv_table(ax, subs, df_av, **kwargs)
        ax.set_title('Average {}seasonal variability of {}'.format(
            'relative ' if kwargs.get('rel') else '',
            subs.label()))
        fig.show()

        # Root-mean-square of seasonal relative standard deviation
        # RMS = ( 1/n * (x_1**2 + x_2**2 + ... + x_n**2) )**0.5
        df['strato_stdv_RMS'] = ((df[strato_cols] **2 ).sum(axis=1) / len(stdv_df_dict) )**0.5
        df['tropo_stdv_RMS'] = ((df[tropo_cols] **2 ).sum(axis=1) / len(stdv_df_dict) )**0.5
        
        df_RMS = df[['tropo_stdv_RMS', 'strato_stdv_RMS']]
        df_RMS = df_RMS.astype(float).round(3)
        
        fig, ax = plt.subplots(dpi=250)
        ax, table = self.make_stdv_table(ax, subs, df_RMS, **kwargs)
        ax.set_title('RMS of {}seasonal variability of {}'.format(
            'relative ' if kwargs.get('rel') else '',
            subs.label()))
        fig.show()
        
        return stdv_df_dict

#%% N2O filter
def plot_sorted(glob_obj, tp, subs, popt0=None, popt1=None, **kwargs):
    """ Plot strat / trop sorted data """
    # only take data with index that is available in df_sorted
    data = glob_obj.df[glob_obj.df.index.isin(glob_obj.df_sorted.index)]
    data.sort_index(inplace=True)
    
    tropo_col = 'tropo_'+tp.col_name
    strato_col = 'strato_'+tp.col_name

    # take 'data' here because substances may not be available in df_sorted
    df_tropo = data[glob_obj.df_sorted[tropo_col] == True]
    df_strato = data[glob_obj.df_sorted[strato_col] == True]

    fig, ax = plt.subplots(dpi=200)
    plt.title(f'{tp.label(True)}')#' filter on {subs.label()} data')
    # ax.scatter(df_strato.index, df_strato[subs.col_name],
    #             c='grey',  marker='.', zorder=0, label='strato')
    # ax.scatter(df_tropo.index, df_tropo[subs.col_name],
    #             c='xkcd:kelly green',  marker='.', zorder=1, label='tropo')

    ax.scatter(df_strato[subs.col_name], df_strato['p'], 
                c='grey',  marker='.', zorder=0, label='strato')
    ax.scatter(df_tropo[subs.col_name], df_tropo['p'], 
                c='xkcd:kelly green',  marker='.', zorder=1, label='tropo')


    if popt0 is not None and popt1 is not None and subs.short_name == tp.crit:
        # only plot baseline for chemical tropopause def and where crit is being plotted
        t_obs_tot = np.array(dt_to_fy(glob_obj.df_sorted.index, method='exact'))
        ls = 'solid' if tp.ID == subs.ID else 'dashed'

        func = dcts.get_subs(col_name=tp.col_name).function
        ax.plot(glob_obj.df_sorted.index, func(t_obs_tot-2005, *popt0),
                c='r', lw=1, ls=ls, label='initial')
        ax.plot(glob_obj.df_sorted.index, func(t_obs_tot-2005, *popt1),
                c='k', lw=1, ls=ls, label='filtered')

    plt.ylabel(subs.label())
    plt.legend()
    plt.show()

#%% Variability
def matrix_plot_stdev_subs(glob_obj, substance,  note='', tps=None,
                           atm_layer='both', savefig=False):
    """
    Create matrix plot showing variability per latitude bin per tropopause definition

    Parameters:
        glob_obj (GlobalObject): Contains the data in self.df

        key short_name (str): Substance short name to show, e.g. 'n2o'

    Returns:
        tropospheric, stratospheric standard deviations within each bin as list for each tp coordinate
    """
    if not tps: 
        tps = glob_obj.tp_coords()

    lat_bmin, lat_bmax = 30, 90 # np.nanmin(lat), np.nanmax(lat)
    lat_bci = bp.Bin_equi1d(lat_bmin, lat_bmax, glob_obj.grid_size)

    tropo_stdevs = np.full((len(tps), lat_bci.nx), np.nan)
    tropo_av_stdevs = np.full(len(tps), np.nan)
    strato_stdevs = np.full((len(tps), lat_bci.nx), np.nan)
    strato_av_stdevs = np.full(len(tps), np.nan)

    tropo_out_list = []
    strato_out_list = []

    for i, tp in enumerate(tps):
        # troposphere
        tropo_data = glob_obj.sel_tropo(**tp.__dict__).df
        shared_indices = glob_obj.sel_tropo(**tp.__dict__).get_shared_indices()
        tropo_data = tropo_data.loc[shared_indices]
        
        tropo_lat = np.array([tropo_data.geometry[i].y for i in range(len(tropo_data.index))]) # lat
        tropo_out_lat = bp.Simple_bin_1d(tropo_data[substance.col_name], tropo_lat, 
                                         lat_bci, count_limit = glob_obj.count_limit)
        tropo_out_list.append(tropo_out_lat)
        tropo_stdevs[i] = tropo_out_lat.vstdv if not all(np.isnan(tropo_out_lat.vstdv)) else tropo_stdevs[i]
        
        # weighted average stdv
        tropo_nonan_stdv = tropo_out_lat.vstdv[~ np.isnan(tropo_out_lat.vstdv)]
        tropo_nonan_vcount = tropo_out_lat.vcount[~ np.isnan(tropo_out_lat.vstdv)]
        tropo_weighted_average = np.average(tropo_nonan_stdv, weights = tropo_nonan_vcount)
        tropo_av_stdevs[i] = tropo_weighted_average 
        
        # stratosphere
        strato_data = glob_obj.sel_strato(**tp.__dict__).df
        shared_indices = glob_obj.sel_strato(**tp.__dict__).get_shared_indices()
        tropo_data = strato_data.loc[shared_indices]
        
        strato_lat = np.array([strato_data.geometry[i].y for i in range(len(strato_data.index))]) # lat
        strato_out_lat = bp.Simple_bin_1d(strato_data[substance.col_name], strato_lat, 
                                          lat_bci, count_limit = glob_obj.count_limit)
        strato_out_list.append(strato_out_lat)
        strato_stdevs[i] = strato_out_lat.vstdv if not all(np.isnan(strato_out_lat.vstdv)) else strato_stdevs[i]
        
        # weighted average stdv
        strato_nonan_stdv = strato_out_lat.vstdv[~ np.isnan(strato_out_lat.vstdv)]
        strato_nonan_vcount = strato_out_lat.vcount[~ np.isnan(strato_out_lat.vstdv)]
        strato_weighted_average = np.average(strato_nonan_stdv, weights = strato_nonan_vcount)
        strato_av_stdevs[i] = strato_weighted_average 

    # Plotting
    # -------------------------------------------------------------------------
    pixels = glob_obj.grid_size # how many pixels per imshow square
    yticks = np.linspace(0, (len(tps)-1)*pixels, num=len(tps))[::-1] # order was reversed for some reason
    tp_labels = [tp.label(True)+'\n' for tp in tps]
    xticks = np.arange(lat_bmin, lat_bmax+glob_obj.grid_size, glob_obj.grid_size)

    fig = plt.figure(dpi=200, figsize=(lat_bci.nx*0.825, len(tps)*2))

    gs = gridspec.GridSpec(5, 2, figure=fig,
                           height_ratios = [1, 0.1, 0.02, 1, 0.1],
                           width_ratios = [1, 0.09])
    axs = gs.subplots()

    [ax.remove() for ax in axs[2, 0:]]
    middle_ax = plt.subplot(gs[2, 0:])
    middle_ax.axis('off')

    ax_strato1 = axs[0,0]
    ax_strato2 = axs[0,1]
    [ax.remove() for ax in  axs[1, 0:]]
    cax_s = plt.subplot(gs[1, 0:])
    
    ax_tropo1 = axs[3,0]
    ax_tropo2 = axs[3,1]
    [ax.remove() for ax in axs[4, 0:]]
    cax_t = plt.subplot(gs[4, 0:])

    # Plot STRATOSPHERE
    # -------------------------------------------------------------------------
    try: 
        vmin, vmax = substance.vlims('vstdv', 'strato')
    except KeyError: 
        vmin, vmax = np.nanmin(strato_stdevs), np.nanmax(strato_stdevs)
        
    norm = Normalize(vmin, vmax) 
    strato_cmap = plt.cm.BuPu  # create colormap
    ax_strato1.set_title(f'Stratospheric variability of {substance.label()}{note}', fontsize=14)

    img = ax_strato1.matshow(strato_stdevs, alpha=0.75,
                     extent = [lat_bmin, lat_bmax,
                               0, len(tps)*pixels],
                     cmap = strato_cmap, norm=norm)
    ax_strato1.set_yticks(yticks, labels=tp_labels)
    ax_strato1.set_xticks(xticks, loc='bottom')
    ax_strato1.tick_params(axis='x', top=False, labeltop=False, labelbottom=True)

    for label in ax_strato1.get_yticklabels():
        label.set_verticalalignment('bottom')

    ax_strato1.grid('both')
    ax_strato1.set_xlabel('Latitude [°N]')

    # add numeric values
    for j,x in enumerate(xticks[:-1]):
        for i,y in enumerate(yticks):
            value = strato_stdevs[i,j]
            if str(value) != 'nan':
                ax_strato1.text(x+0.5*glob_obj.grid_size,
                        y+0.5*pixels,
                        '{0:.2f}'.format(value) if value>vmax/100 else '<{0:.2f}'.format(vmax/100),
                        va='center', ha='center')
    cbar = plt.colorbar(img, cax=cax_s, orientation='horizontal')
    cbar.set_label(f'Standard deviation of {substance.label(name_only=True)} within bin [{substance.unit}]')
    # make sure vmin and vmax are shown as colorbar ticks
    cbar_vals = cbar.get_ticks()
    cbar_vals = [vmin] + cbar_vals[1:-1].tolist() + [vmax]
    cbar.set_ticks(cbar_vals)

    # Stratosphere average variability
    img = ax_strato2.matshow(np.array([strato_av_stdevs]).T, alpha=0.75,
                     extent = [0, glob_obj.grid_size,
                               0, len(tps)*pixels],
                     cmap = strato_cmap, norm=norm)
    for i,y in enumerate(yticks): 
        value = strato_av_stdevs[i]
        if str(value) != 'nan':
            ax_strato2.text(0.5*glob_obj.grid_size,
                    y+0.5*pixels,
                    '{0:.2f}'.format(value) if value>vmax/100 else '<{0:.2f}'.format(vmax/100),
                    va='center', ha='center')
    ax_strato2.tick_params(axis='both', bottom=False, top=False, labeltop=False, left=False, labelleft=False)
    ax_strato2.set_xlabel('Average')

    # Plot TROPOSPHERE
    # -------------------------------------------------------------------------
    try: 
        vmin, vmax = substance.vlims('vstdv', 'tropo')
    except KeyError: 
        vmin, vmax = np.nanmin(strato_stdevs), np.nanmax(strato_stdevs)
    norm = Normalize(vmin, vmax)  # normalise color map to set limits
    tropo_cmap = cmr.get_sub_cmap('YlOrBr', 0, 0.75) # create colormap
    ax_tropo1.set_title(f'Tropospheric variability of {substance.label()}{note}', fontsize=14)

    img = ax_tropo1.matshow(tropo_stdevs, alpha=0.75,
                     extent = [lat_bmin, lat_bmax,
                               0, len(tps)*pixels],
                     cmap = tropo_cmap, norm=norm)
    ax_tropo1.set_yticks(yticks, labels=tp_labels)
    ax_tropo1.set_xticks(xticks, loc='bottom')
    ax_tropo1.tick_params(axis='x', top=False, labeltop=False, labelbottom=True)
    ax_tropo1.set_xlabel('Latitude [°N]')

    for label in ax_tropo1.get_yticklabels():
        label.set_verticalalignment('bottom')

    ax_tropo1.grid('both')
    # ax1.set_xlim(-40, 90)

    # add numeric values
    for j,x in enumerate(xticks[:-1]):
        for i,y in enumerate(yticks):
            value = tropo_stdevs[i,j]
            if str(value) != 'nan':
                ax_tropo1.text(x+0.5*glob_obj.grid_size,
                        y+0.5*pixels,
                        '{0:.2f}'.format(value) if value>vmax/100 else '<{0:.2f}'.format(np.ceil(vmax/100)),
                        va='center', ha='center')
    cbar = plt.colorbar(img, cax=cax_t, orientation='horizontal')
    cbar.set_label(f'Standard deviation of {substance.label(name_only=True)} within bin [{substance.unit}]')
    # make sure vmin and vmax are shown as colorbar ticks
    cbar_vals = cbar.get_ticks()
    cbar_vals = [vmin] + cbar_vals[1:-1].tolist() + [vmax]
    cbar.set_ticks(cbar_vals)
    
    # Tropopsphere average variability
    img = ax_tropo2.matshow(np.array([tropo_av_stdevs]).T, alpha=0.75,
                     extent = [0, glob_obj.grid_size,
                               0, len(tps)*pixels],
                     cmap = tropo_cmap, norm=norm)

    for i,y in enumerate(yticks): 
        value = tropo_av_stdevs[i]
        if str(value) != 'nan':
            ax_tropo2.text(0.5*glob_obj.grid_size,
                    y+0.5*pixels,
                    '{0:.2f}'.format(value) if value>vmax/100 else '<{0:.2f}'.format(np.ceil(vmax/100)),
                    va='center', ha='center')


    ax_tropo2.set_xlabel('Average')
    ax_tropo2.tick_params(axis='both', bottom=False, top=False, labeltop=False, left=False, labelleft=False)

    # -------------------------------------------------------------------------
    fig.tight_layout()
    fig.subplots_adjust(top=0.8)

    if savefig:
        plt.savefig(f'E:/CARIBIC/Plots/variability_lat_binned/variability_{substance.col_name}.png', format='png')
    fig.show()

    return tropo_out_list, strato_out_list

def matrix_plot_stdev(glob_obj, note='', atm_layer='both', savefig=False,
                      minimise_tps=True, **subs_kwargs):
    substances = [s for s in dcts.get_substances(**subs_kwargs)
                  if (s.col_name in glob_obj.df.columns
                      and not s.col_name.startswith('d_'))]

    for subs in substances:
        matrix_plot_stdev_subs(glob_obj, subs,  note=note, minimise_tps=minimise_tps,
                                   atm_layer=atm_layer, savefig=savefig)


#%% Combine functionality of TropopausePlotter with specific GlobalData sub-classes 
class CaribicTropopause(Caribic, TropopausePlotter): 
    """ Add functionality of TropopausePlotter to Caribic objects. """
    def __init__(self, **kwargs): 
        super().__init__(**kwargs) # Caribic

#%% Deprecated
# class TropopausePlotter(TropopauseData):
#     """ Add plotting functionality to tropopause data objects """
#     def __init__(self, tp_data = None, years=range(2005, 2020), 
#                  interp=True, method='n', df_sorted=True):
#         if not tp_data is None: #isinstance(tp_data, TropopauseData):
#             self.__dict__ = tp_data.__dict__.copy()
#         else: 
#             super().__init__(years, interp, method, df_sorted)
#         self.detrend_all()
#         self.tps = self.set_tps()