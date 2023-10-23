# -*- coding: utf-8 -*-
"""
@Author: Sophie Bauchinger, IAU
@Date Mon Aug 14 14:06:26 2023

Plotting Tropopause heights for different tropopauses different vertical coordinates
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import geopandas
import math
from shapely.geometry import Point
from matplotlib.colors import Normalize
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import AxesGrid
from PIL import Image
import glob
import cmasher as cmr

import toolpac.calc.dev_binprocessor as bp
# from toolpac.calc.binprocessor import Bin_equi1d, Simple_bin_1d, Bin_equi2d, Simple_bin_2d
from toolpac.conv.times import datetime_to_fractionalyear as dt_to_fy

import dictionaries as dcts
from data import TropopauseData
import tools

world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
vlims = {'p':(100,500), 'pt':(300, 400), 'z':(7.5,17.5)}
count_limit = 5

#TODO Add disclaimer to dyn and cpt to show reduced latitude ranges 

#%% Class definition
class TropopausePlotter(TropopauseData):
    """ Add plotting functionality to tropopause data objects """
    def __init__(self, years=range(2005, 2020), interp=True, method='n', df_sorted=True,
                 tp_data = None):
        if isinstance(tp_data, TropopauseData):
            self.__dict__ = tp_data.__dict__.copy()
        else: 
            super().__init__(years, interp, method, df_sorted)
        self.detrend_all()

    def test_scatter(self, year=None):
        vc = 'pt'
        tp = dcts.get_coordinates(vcoord=vc, tp_def='not_nan', rel_to_tp=False)[0]
        fig, axs = plt.subplots(2,2,dpi=150, figsize=(10,5))
        fig.suptitle(f'{tp.col_name} - {tp.long_name}')
        if year: fig.text(0.9, 0.95, f'{year}',
                          bbox = dict(boxstyle='round', facecolor='white',
                                      edgecolor='grey', alpha=0.5, pad=0.25))
        for s,ax in zip([1,2,3,4], axs.flatten()):
            if year: df_r = self.sel_year(year).sel_season(s).df
            else: df_r = self.sel_season(s).df
            if df_r.empty: continue
            df_r.geometry = [Point(pt.y, pt.x) for pt in df_r.geometry]

            lon = np.array([df_r.geometry[i].x for i in range(len(df_r.index))])
            lat = np.array([df_r.geometry[i].y for i in range(len(df_r.index))])

            bci = bp.Bin_equi2d(np.nanmin(lon), np.nanmax(lon), self.grid_size,
                                np.nanmin(lat), np.nanmax(lat), self.grid_size)

            out = bp.Simple_bin_2d(df_r[tp.col_name], lon, lat, bci, 
                                   count_limit = self.count_limit)

            world.boundary.plot(ax=ax, color='black', linewidth=0.3)
            ax.set_title(dcts.dict_season()[f'name_{s}'])
            cmap = 'viridis_r' if tp.vcoord=='p' else 'viridis'

            norm = Normalize(*vlims[vc]) # colormap normalisation
            # df_r.plot(tp.col_name, cmap=cmap, legend=True, ax=ax)
            img = ax.imshow(out.vmean, cmap = cmap, norm=norm, origin='lower',
                            extent = [bci.ybmin, bci.ybmax, bci.xbmin, bci.xbmax])
            cbar = plt.colorbar(img, ax=ax, pad=0.08,
                                orientation='vertical', extend='both') # colorbar
            cbar.ax.set_xlabel(f'{vc} [{tp.unit}]')
            ax.set_ylim(-90, 90); ax.set_xlim(-180, 180)
        fig.tight_layout()
        plt.show()

    def tp_height_2d(self, savefig=False, year=None, rel = False, 
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
            fig.suptitle(f'{tp.col_name} - {tp.long_name}')
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

    def tp_height_1d(self, vcoord='pt', rel=False, seasonal=False, note='', 
                     minimise_tps = True):
        """ Plots tropopause height over latitude. 

        Parameters:
            vcoord (str): vertical coordinate indicating tropopause extent
            rel (bool): vertical coordinate relative to CARIBIC flight track
            seasonal (bool): separate data into seasons
        """
        tps = dcts.get_coordinates(vcoord=vcoord, tp_def='not_nan', rel_to_tp=rel)

        if minimise_tps: tps = tools.minimise_tps(tps)
        else: 
            for tp in [tp for tp in tps if tp.pvu in [1.5, 2.0]]: # rmv 1.5 and 2.0 PVU TPs
                tps.remove(tp)
            tps.sort(key=lambda x: x.col_name)

        nrows = math.ceil((len(tps)+1)/4)
        fig, axs = plt.subplots(nrows, 4, dpi=150,
                                figsize=(10,nrows*2.5), sharey=True, sharex=True)

        for i, ax in enumerate(axs.flatten()): # hide extra plots
            if i > len(tps): ax.axis('off')

        vcoord_names = {'p': 'Pressure',
                        'z' : 'Geopotential Height',
                        'pt' : 'Potential Temperature'}

        fig.suptitle('{} {}'.format(
            'Vertical extent of troposphere in' if not rel
            else 'Distance between flight track and troposphere in', vcoord_names[vcoord]))
        
        if note: 
            fig.subplots_adjust(top=0.85)
            fig.text(s=note, **dcts.note_dict(fig, x=0.98, y=0.94))

        bci = bp.Bin_equi1d(-90, 90, self.grid_size)
        vmeans = pd.DataFrame(index = bci.xintm) # overall average
        vmeans_std = pd.DataFrame(index = bci.xintm) # overall average

        for s in ([None] if not seasonal else [1,2,3,4]):
            for tp, ax in zip(tps, axs.flatten()[:len(tps)]):
                # get data
                data = self.df.copy()
                if seasonal:
                    data = self.sel_season(s).df
                # prep data: only take tps in ranges they make sense in
                if not tp.col_name in data.columns: pass
                if tp.tp_def == 'dyn': # dynamic TP only outside the tropics
                    data = data[np.array([(i>30 or i<-30) for i in np.array(data.geometry.y) ])]
                if tp.tp_def == 'cpt': # cold point TP only in the tropics
                    data = data[np.array([(i<30 and i>-30) for i in np.array(data.geometry.y) ])]

                # bin using same binclassinstance as all other tropopauses
                bin1d = bp.Simple_bin_1d(data[tp.col_name],
                                      np.array(data.geometry.y), bci,
                                      count_limit = self.count_limit)
                if not seasonal:
                    vmeans[tp.col_name] = bin1d.vmean
                    vmeans_std[tp.col_name+'_std'] = bin1d.vstdv
                    label = '{}_{}{}'.format(tp.model, tp.tp_def, '_'+str(tp.pvu) if tp.tp_def=='dyn' else '')
                    ax.plot(bin1d.xmean, bin1d.vmean, color='#1f77b4', label=label)
                    ax.fill_between(bin1d.xmean, bin1d.vmean-bin1d.vstdv, bin1d.vmean+bin1d.vstdv,
                                    alpha=0.3, color='#1f77b4')
                    ax.legend(loc='lower left')

                else: # seasonal. separate vmeans by season to calc av later
                    vmeans[tp.col_name+f'_{s}'] = bin1d.vmean
                    color = dcts.dict_season()[f'color_{s}']
                    ax.plot(bin1d.xmean, bin1d.vmean, color=color, label=dcts.dict_season()[f'name_{s}'])
                    ax.text(0.05, 0.05,
                            '{}_{}{}'.format(tp.model, tp.tp_def,
                                             '_'+str(tp.pvu) if tp.tp_def=='dyn' else ''),
                            transform=ax.transAxes, verticalalignment='bottom',
                            bbox = dict(boxstyle='round', facecolor='white',
                                        edgecolor='grey', alpha=0.5, pad=0.25))

            ax.set_xticks(np.arange(-30, 90, 30))

        if not seasonal:
            # indicate average tropopause height on all plots & add xaxis label to extra plot
            average = vmeans.mean(axis=1).values
            vmeans['av_std'] = np.nan
            for i in range(len(vmeans.index)):
                sqrt_of_sum_of_squares = np.sqrt(np.nansum([unc**2 for unc in vmeans_std.iloc[i]])) / 2
                vmeans['av_std'].iloc[i] = sqrt_of_sum_of_squares

            axAv = axs.flatten()[len(tps)]
            axAv.plot(bci.xintm, average, ls='dashed', c='k', alpha=0.5,
                    zorder=1, label='Average')
            axAv.fill_between(bci.xintm,
                              average-vmeans['av_std'], average+vmeans['av_std'],
                              alpha=0.3, color='k')
            axAv.legend(loc='lower left')

        else:
            for s in [1,2,3,4]:
                vmeans_s = vmeans[[c for c in vmeans.columns if c.endswith(f'_{s}')]]
                average = vmeans_s.mean(axis=1).values
                axAv = axs.flatten()[len(tps)]
                axAv.plot(bci.xintm, average, ls='dashed',
                          c = dcts.dict_season()[f'color_{s}'], # alpha=0.5,
                          zorder=1, label=dcts.dict_season()[f'name_{s}'])

        # go through axes, (add average), set label
        for ax in axs.flatten()[:len(tps)+1]:
            if not seasonal:
                ax.plot(bci.xintm, average, ls='dashed', c='k', alpha=0.3, zorder=1)
            if rel: ax.hlines(0, min(bci.xintm), max(bci.xintm),
                              color='grey', zorder=2, lw=0.5, ls='dotted')
            ax.set_xlabel('Latitude [°N]')

        if vcoord == 'p': # because sharey=True, applied for all axes
            axAv.invert_yaxis()
            axAv.set_yscale('{}'.format('symlog' if rel else 'log'))

        
        if nrows>1: 
            for ax in [axs[0,0], axs[0,1]]: # left most
                ax.set_ylabel('{}{} [{}]'.format('$\Delta$' if tp.rel_to_tp else '',
                                                 vcoord_names[tp.vcoord], tp.unit))
        else: 
            axs[0].set_ylabel('{}{} [{}]'.format('$\Delta$' if tp.rel_to_tp else '',
                                                 vcoord_names[tp.vcoord], tp.unit))

        fig.tight_layout()
        if seasonal: # add horizontal figure legend for seasons at the top
            fig.subplots_adjust(top=0.85) # add space for fig legend
            lines, labels = axs.flatten()[0].get_legend_handles_labels()
            fig.legend(lines, labels, loc='upper center', ncol=4,
                bbox_to_anchor=[0.5, 0.94])
        plt.show()
        return
    
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

                    ax.set_title(dcts.make_coord_label(tp, filter_label=True), fontsize=8)
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
    
                        ax.set_title(dcts.make_coord_label(tp, filter_label=True), fontsize=8)
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
        return self

    def show_ratios(self, as_subplot=False, ax=None, single_tp_def=None, group_vc=False,
                    unity_line=True, minimise_tps=True, note='', **tp_kwargs):
        """ Plot ratio of tropo / strato datapoints on a horizontal bar plot """
        tropo_counts = self.calc_ratios(group_vc=group_vc) # dataframe

        ratios = tropo_counts.loc['ratios']
        tropo_counts.drop(index='ratios', inplace=True)
        n_values = [tropo_counts[i].loc[True] + tropo_counts[i].loc[False] 
                    for i in tropo_counts.columns]

        bar_labels = ['{:.2f}'.format(r) for r in ratios]
        # bar_labels = ['{:.2f} (n={})'.format(r,nr) for r, nr in zip(ratios, n_values)]

        # find coordinates for each column
        tps = [dcts.get_coord(col_name = c) for c in tropo_counts.columns 
               if c in [c.col_name for c in dcts.get_coordinates() 
                        if c.tp_def not in ['combo', 'cpt']]]

        if minimise_tps:
            tps = tools.minimise_tps(tps)

        if len(tps)==0: raise Exception('No tps found that fit the criteria.')

        # make sure cols and labels are related 
        cols, tp_labels = map(list, zip(*[('tropo_'+tp.col_name, dcts.make_coord_label(tp, filter_label=True)) 
                                        for tp in tps]))

        if not as_subplot: 
            fig, ax = plt.subplots(dpi=240, figsize=(8,6))
            ax.set_title('Ratio of tropospheric / stratospheric datapoints in Caribic-2')

        tp_defs = set([tp.tp_def for tp in tps]) if single_tp_def is None else [single_tp_def]
        # ax.grid(True, axis='x', c='k', alpha=0.3)
        if unity_line: 
            ax.axvline(1, linestyle='--', color='k', alpha=0.3, zorder=0, lw=1) # vertical lines
            ax.set_axisbelow(True)

        for tp_def in tp_defs:
            color = dcts.dict_tps()[f'color_{tp_def}']
            current_tps = [tp for tp in tps if tp.tp_def==tp_def]
            # make sure cols and labels are related 
            cols, labels = map(list, zip(*[(tp.col_name, dcts.make_coord_label(tp, filter_label=True)) 
                                            for tp in current_tps]))
            current_ratios = ratios[cols]
            
            if 'vcoord' in tp_kwargs: 
                labels = [l[l.find('(')+1 : l.find(')')] for l in labels]
                if not as_subplot:
                    ax.set_title('Ratio of tropospheric / stratospheric datapoints in Caribic-2\n' 
                                 + f'Vertical coordinate: {current_tps[0].vcoord} [{current_tps[0].unit}]')

            bars = ax.barh(labels, current_ratios, rasterized=True, color=color,
                           alpha=0.9 if 'vcoord' in tp_kwargs else 0.9)#, hatch=hatches)
            bar_labels = ['{:.2f}'.format(r) for r in current_ratios]
            
            ax.bar_label(bars, bar_labels, fmt='%.3g', padding=1)
            ax.set_xlim(0,3) # np.nanmax(ratios)*1.2) #4.5)

        if note: ax.text(s=note, **dcts.note_dict(ax))

        if not as_subplot: 
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

#%% N2O filter
def plot_sorted(glob_obj, df_sorted, crit, ID, popt0=None, popt1=None,
                subs=None, subs_col=None, detr=True, **kwargs):
    """ Plot strat / trop sorted data """
    # only take data with index that is available in df_sorted
    if subs in glob_obj.data.keys(): df = glob_obj.data[subs]
    elif glob_obj.source=='Caribic': df = glob_obj.data[ID]
    else: df = glob_obj.df
    data = df[df.index.isin(df_sorted.index)]

    # data = glob_obj.data[ID][glob_obj.data[ID].index.isin(df_sorted.index)]
    data.sort_index(inplace=True)

    # separate trop/strat data for any criterion
    tropo_col = [col for col in df_sorted.columns if col.startswith('tropo')][0]
    strato_col = [col for col in df_sorted.columns if col.startswith('strato')][0]

    # take 'data' here because substances may not be available in df_sorted
    df_tropo = data[df_sorted[tropo_col] == True]
    df_strato = data[df_sorted[strato_col] == True]

    if crit in ['o3', 'n2o'] and not subs: subs = crit

    if 'subs_pfx' in kwargs.keys():
        subs_pfx = kwargs['subs_pfx']
        substance = dcts.get_col_name(subs, glob_obj.source, kwargs['subs_pfx'])
    else:
        if subs_col is None and subs is not None:
            for subs_pfx in (ID, 'GHG', 'INT', 'INT2'):
                try: substance = dcts.get_col_name(subs, glob_obj.source, subs_pfx); break
                except: substance = None; continue
        else: substance = subs_col; subs_pfx = ID
    if substance is None:
        print(f'Cannot plot {subs}, not available in {ID}.'); return
    if 'detr_'+substance in data.columns: substance = 'detr_'+substance

    fig, ax = plt.subplots(dpi=200)
    plt.title(f'{crit} filter on {ID} data')
    ax.scatter(df_strato.index, df_strato[substance],
                c='grey',  marker='.', zorder=0, label='strato')
    ax.scatter(df_tropo.index, df_tropo[substance],
                c='xkcd:kelly green',  marker='.', zorder=1, label='tropo')

    if popt0 is not None and popt1 is not None and (subs==crit or subs is None):
        # only plot baseline for chemical tropopause def and where crit is being plotted
        t_obs_tot = np.array(dt_to_fy(df_sorted.index, method='exact'))
        ls = 'solid'
        if not subs_pfx == ID: ls = 'dashed'
        ax.plot(df_sorted.index, dcts.get_fct_substance(crit)(t_obs_tot-2005, *popt0),
                c='r', lw=1, ls=ls, label='initial')
        ax.plot(df_sorted.index, dcts.get_fct_substance(crit)(t_obs_tot-2005, *popt1),
                c='k', lw=1, ls=ls, label='filtered')

    # plt.ylim(220, 340)

    plt.ylabel(substance)
    plt.legend()
    plt.show()

#%% Variability

variability_tropo = {
    'detr_sf6' : (0.05, 0.17),
    'detr_n2o' : (0.8, 1.8),
    'detr_co'  : (16, 30),
    'detr_co2' : (2.0, 3.0),
    'detr_ch4' : (16, 26),
    }

variability_strato = {
    'detr_sf6' : (0.05, 0.3),
    'detr_n2o' : (5.1, 13),
    'detr_co'  : (10, 26),
    'detr_co2' : (1.2, 1.8),
    'detr_ch4' : (30, 60),
    }

def plot_stdev_2d(glob_obj, detr=False, 
                  grid_size = 5, count_limit = 5, 
                  **subs_kwargs):
    # glob_obj, subs, single_yr=None, c_pfx=None, years=None, detr=False):
    """ Create a 2D plot of binned mixing ratios for each available year on a grid.

    Parameters:
        glob_obj (GlobalData): instance with global dataset
        detr (bool): Plot detrended data
    """
    data = glob_obj.df
    substances = [dcts.get_subs(col_name=c) for c in data.columns
                  if c in [s.col_name for s in dcts.get_substances(**subs_kwargs)]
                  and not c.startswith(('d_', 'detr_'))]

    for substance in substances:

        nplots = len(glob_obj.years)
        nrows = nplots if nplots <= 4 else math.ceil(nplots / 3)
        ncols = 1 if nplots <= 4 else 3

        fig = plt.figure(dpi=100, figsize=(6 * ncols, 3 * nrows))

        grid = AxesGrid(fig, 111, # similar to subplot(142)
                        nrows_ncols=(nrows, ncols),
                        axes_pad=0.4,
                        share_all=True,
                        label_mode="all",
                        cbar_location="bottom",
                        cbar_mode="single")

        if nplots >= 4:
            data_type = 'measured' if substance.model == 'MSMT' else 'modeled'
            fig.suptitle(f'Standard deviation of binned global{data_type} mixing ratios of {dcts.make_subs_label(substance)}',
                         fontsize=25)
            plt.subplots_adjust(top=0.96)

        out_list = glob_obj.binned_2d(substance)

        world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))

        for i, (out, ax, year) in enumerate(zip(out_list, grid, glob_obj.years)):
            world.boundary.plot(ax=ax, color='grey', linewidth=0.3)

            cmap = plt.cm.Purples
            vmin = -10 # np.nanmin(out.vstdv)
            vmax = np.nanmax(out.vstdv)*0.5
            norm = Normalize(vmin, vmax)  # normalise color map to set limits

            img = ax.imshow(out.vstdv.T, cmap=cmap, norm=norm, origin='lower',
                            extent=[out.xbmin, out.xbmax, out.ybmin, out.ybmax])

            ax.text(**dcts.note_dict(ax, 0.13, 0.1, f'{year}'), weight='bold')
            ax.set_xlim(-180, 180)
            ax.set_ylim(-60, 100)

            # label outer plot axes
            if grid._get_col_row(i)[0] == 0:
                ax.set_ylabel('Latitude [°N]')
            if grid._get_col_row(i)[0] == ncols:
                ax.set_xlabel('Longitude [°E]')

            if i==0:
                cbar = grid.cbar_axes[0].colorbar(img)
                cbar.ax.tick_params(labelsize=15, which='both')
                # cbar.ax.minorticks_on()
                cbar.ax.set_xlabel(dcts.make_subs_label(substance), fontsize=15)

        for i, ax in enumerate(grid):  # hide extra plots
            if i >= nplots:
                ax.axis('off')

        # cbar = plt.colorbar(img, fig.add_axes([0.05, 0.98, 0.95, 0.05], box_aspect=0.05),
        #                     aspect=10, pad=0.1, orientation='horizontal') # colorbar


        # fig.tight_layout()
        plt.show()

def matrix_plot_stdev_subs(glob_obj, substance,  note='', minimise_tps=True,
                           atm_layer='both', savefig=False) -> (np.array, np.array):
    """
    Create matrix plot showing variability per latitude bin per tropopause definition

    Parameters:
        glob_obj (GlobalObject): Contains the data in self.df

        key short_name (str): Substance short name to show, e.g. 'n2o'

    Returns:
        tropospheric, stratospheric standard deviations within each bin as list for each tp coordinate
    """
    tps = [tp for tp in dcts.get_coordinates(tp_def='not_nan')
           if 'tropo_'+tp.col_name in glob_obj.df_sorted.columns]

    if minimise_tps:
        tps = tools.minimise_tps(tps)

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
    tp_labels = [dcts.make_coord_label(tp)+'\n' for tp in tps]
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
    if substance.short_name in variability_strato: 
        vmin = variability_strato[substance.short_name][0]
        vmax = variability_strato[substance.short_name][1]
    else: 
        vmin, vmax = np.nanmin(strato_stdevs), np.nanmax(strato_stdevs)
    norm = Normalize(vmin, vmax)  # normalise color map to set limits
    strato_cmap = plt.cm.BuPu  # create colormap
    ax_strato1.set_title(f'Stratospheric variability of {dcts.make_subs_label(substance)}{note}', fontsize=14)

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
    cbar.set_label(f'Standard deviation of {dcts.make_subs_label(substance, name_only=True)} within bin [{substance.unit}]')
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
    if substance.short_name in variability_tropo: 
        vmin = variability_tropo[substance.short_name][0]
        vmax = variability_tropo[substance.short_name][1]
    else: 
        vmin, vmax = np.nanmin(tropo_stdevs), np.nanmax(tropo_stdevs)
    norm = Normalize(vmin, vmax)  # normalise color map to set limits
    tropo_cmap = cmr.get_sub_cmap('YlOrBr', 0, 0.75) # create colormap
    ax_tropo1.set_title(f'Tropospheric variability of {dcts.make_subs_label(substance)}{note}', fontsize=14)

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
    cbar.set_label(f'Standard deviation of {dcts.make_subs_label(substance, name_only=True)} within bin [{substance.unit}]')
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

#%% Plotting function calls
# if __name__=='__main__':
#     tpause = TropopauseData()
#     tpp_extratropics = TropopausePlotter(tp_inst=tpause).sel_latitude(30, 90)
#     tpp_extratropics.show_ratios()

    # # Global 2D scatter of tropopause heights for all definitions
    # for year in tp.years:
    #     tp.scatter_2d(save=False, year=year)

    # #  Binned versus latitude
    # for vc in ['p', 'pt', 'z']:
    #     tp.tropopause_vs_latitude(vc, rel=False)
    #     # tropopause_vs_latitude(vc, rel=True)
    #     tp.tropopause_vs_latitude(vc, rel=False, seasonal=True)

    # # Substances in tropopsphere / stratosphere
    # substances = get_substances(source='EMAC') + get_substances(source='Caribic')
    # substances = [s for s in substances if not s.col_name.startswith('d_')]
    # df_sorted = tp.sort_tropo_strato(substances)
    # tp.trop_strat_ratios()

#%% Animate changes over years
def make_gif():
    pdir = r'C:\Users\sophie_bauchinger\sophie_bauchinger\Figures\tp_scatter_2d'
    for vc in ['p', 'pt', 'z']:
        tps = dcts.get_coordinates(vcoord=vc, tp_def='not_nan', rel_to_tp=False)
        for tp in tps:
            # fn = pdir+.format(, '_'+str(year) if year else ''))
            frames = [Image.open(image) for image in glob.glob(f'{pdir}/{tp.col_name}*_*.png')]
            if len(frames)==0: frames = [Image.open(image) for image in glob.glob(f'{pdir}/{tp.col_name[:-1]}*_*.png')]

            # frames = [Image.open(image) for image in glob.glob(f"{pdir}/*.JPG")]
            frame_one = frames[0]
            frame_one.save(f'C:/Users/sophie_bauchinger/sophie_bauchinger/Figures/tp_scatter_2d_GIFs/{tp.col_name}.gif',
                           format="GIF", append_images=frames,
                           save_all=True, duration=200, loop=0)
# if __name__ == "__main__":
#     make_gif()
