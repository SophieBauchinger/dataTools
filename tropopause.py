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
from PIL import Image
import glob

from toolpac.calc.binprocessor import Bin_equi1d, Simple_bin_1d, Bin_equi2d, Simple_bin_2d

from dictionaries import get_coordinates, dict_season, get_substances
from data import TropopauseData

world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
vlims = {'p':(100,500), 'pt':(250, 350), 'z':(5,20)}

#%% Import data
class TropopausePlotter(TropopauseData):
    """ Add plotting functionality to tropopause data objects """
    def __init__(self, years=range(2005, 2020), interp=True, method='n'):
        super().__init__(years, interp, method)

    def scatter_2d(self, save=False, year=None):
        """ 2D global scatter of tropopause height.
        Parameters: 
            save (bool): save plot to pdir instead of plotting
            year (float): select single specific year to plot / save 
        """
        pdir = r'C:\Users\sophie_bauchinger\sophie_bauchinger\Figures\tp_scatter_2d'
        for vc in ['p', 'pt', 'z']:
            tps = get_coordinates(vcoord=vc, tp_def='not_nan', rel_to_tp=False)
            for tp in tps:
                fig, axs = plt.subplots(2,2,dpi=150, figsize=(10,5))
                fig.suptitle(f'{tp.col_name} - {tp.long_name}')
                if year: fig.text(0.9, 0.95, f'{year}',
                                  bbox = dict(boxstyle='round', facecolor='white',
                                              edgecolor='grey', alpha=0.5, pad=0.25))
                for s,ax in zip([1,2,3,4], axs.flatten()):
                    if year: df_r = self.sel_season(s).df
                    else: df_r = self.sel_season(s).df
                    if df_r.empty: pass
                    df_r.geometry = [Point(pt.y,pt.x) for pt in df_r.geometry]
    
                    x = np.array([df_r.geometry[i].x for i in range(len(df_r.index))])
                    y = np.array([df_r.geometry[i].y for i in range(len(df_r.index))])
                    binclassinstance = Bin_equi2d(np.nanmin(x), np.nanmax(x), 5,
                                                  np.nanmin(y), np.nanmax(y), 5)
                    out = Simple_bin_2d(df_r[tp.col_name], x, y, binclassinstance)
    
                    world.boundary.plot(ax=ax, color='black', linewidth=0.3)
                    ax.set_title(dict_season()[f'name_{s}'])
                    cmap = 'viridis_r' if tp.vcoord=='p' else 'viridis'
    
                    norm = Normalize(*vlims[vc]) # colormap normalisation
                    # df_r.plot(tp.col_name, cmap=cmap, legend=True, ax=ax)
                    img = ax.imshow(out.vmean.T, cmap = cmap, origin='lower', norm=norm,
                                extent=[np.nanmin(x), np.nanmax(x), np.nanmin(y), np.nanmax(y)])
                    cbar = plt.colorbar(img, ax=ax, pad=0.08,
                                        orientation='vertical', extend='both') # colorbar
                    cbar.ax.set_xlabel(f'{vc} [{tp.unit}]')
                    ax.set_ylim(-90, 90); ax.set_xlim(-180, 180)
                fig.tight_layout()
                if save: 
                    plt.savefig(pdir+'\{}{}.png'.format(
                        tp.col_name, '_'+str(year) if year else ''))
                    
                plt.show()
                plt.close()

    def tropopause_vs_latitude(self, vcoord, rel, seasonal=False):
        """ Plots tropopause height over latitude
        Parameters:
            vcoord (str): vertical coordinate indicating tropopause extent
            rel (bool): vertical coordinate relative to CARIBIC flight track
            seasonal (bool): separate data into seasons
        """
        tps = get_coordinates(vcoord=vcoord, tp_def='not_nan', rel_to_tp=rel)
        for tp in [tp for tp in tps if tp.pvu in [1.5, 2.0]]: # rmv 1.5 and 2.0 PVU TPs
            tps.remove(tp)
        tps.sort(key=lambda x: x.col_name)
        nrows = math.ceil((len(tps)+1)/4)
        fig, axs = plt.subplots(nrows, 4, dpi=150,
                                figsize=(10,nrows*2.5), sharey=True, sharex=True)
    
        for i, ax in enumerate(axs.flatten()): # hide extra plots
            if i > len(tps): ax.axis('off')
    
        vcs = {'p': 'Pressure',
               'z' : 'geopotential height',
               'pt' : 'Potential Temperature'}
    
        fig.suptitle('{} {} coordinates'.format(
            'Vertical extent of troposphere in' if not rel
            else 'Distance between flight track and troposphere in', vcs[vcoord]))
    
        xbmin, xbmax, xbsize = -90, 90, 5
        bci = Bin_equi1d(xbmin, xbmax, xbsize)
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
                    data = data[np.array([(i>30 or i<-30) for i in np.array(data.geometry.x) ])]
                if tp.tp_def == 'cpt': # cold point TP only in the tropics
                    data = data[np.array([(i<30 and i>-30) for i in np.array(data.geometry.x) ])]
    
                # bin using same binclassinstance as all other tropopauses
                bin1d = Simple_bin_1d(data[tp.col_name],
                                      np.array(data.geometry.x), bci)
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
                    color = dict_season()[f'color_{s}']
                    ax.plot(bin1d.xmean, bin1d.vmean, color=color, label=dict_season()[f'name_{s}'])
                    ax.text(0.05, 0.05,
                            '{}_{}{}'.format(tp.model, tp.tp_def,
                                             '_'+str(tp.pvu) if tp.tp_def=='dyn' else ''),
                            transform=ax.transAxes, verticalalignment='bottom',
                            bbox = dict(boxstyle='round', facecolor='white',
                                        edgecolor='grey', alpha=0.5, pad=0.25))
            ax.set_xticks([-30, 0, 30, 60, 90])
    
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
                          c = dict_season()[f'color_{s}'], # alpha=0.5,
                          zorder=1, label=dict_season()[f'name_{s}'])
    
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
    
        for ax in [axs[0,0], axs[0,1]]: # left most
            ax.set_ylabel('{}{} [{}]'.format('$\Delta$' if tp.rel_to_tp else '',
                                             tp.vcoord, tp.unit))
    
        fig.tight_layout()
        if seasonal: # add horizontal figure legend for seasons at the top
            fig.subplots_adjust(top=0.85) # add space for fig legend
            lines, labels = axs.flatten()[0].get_legend_handles_labels()
            fig.legend(lines, labels, loc='upper center', ncol=4,
                       bbox_to_anchor=[0.5, 0.94])
        plt.show()
        return

    def sort_tropo_strato(self, substances, vcoords=['p', 'z', 'pt'], plot=True):
        """ Returns dataframe with bool strat / trop columns """
        data = TropopauseData().df.copy()
        df_sorted = pd.DataFrame(index=data.index)
    
        for subs in substances:
            if not subs.col_name in data.columns:
                print(f'{subs.col_name} not found in data'); pass
            for vcoord in vcoords:
                tps = get_coordinates(vcoord=vcoord, tp_def='not_nan', rel_to_tp=True)
                for tp in [tp for tp in tps if tp.pvu in [1.5, 2.0]]: # rmv 1.5 and 2.0 PVU TPs
                    tps.remove(tp)
    
                if plot: # initialise figure
                    fig, axs = plt.subplots(math.ceil(len(tps)/2), 2, dpi=200,
                                            figsize=(7, math.ceil(len(tps)/2)*2))
                    if len(tps)%2: axs.flatten()[-1].axis('off')
                    fig.suptitle(f'{subs.col_name} in sorted with TP in {vcoord}')
    
                for tp, ax in zip(tps, axs.flatten() if plot else [None]*len(tps)):
                    tp_df = data.dropna(axis=0, subset=[tp.col_name])
    
                    if tp.tp_def == 'dyn': # dynamic TP only outside the tropics
                        tp_df = tp_df[np.array([(i>30 or i<-30) for i in np.array(tp_df.geometry.x) ])]
                    if tp.tp_def == 'cpt': # cold point TP only in the tropics
                        tp_df = tp_df[np.array([(i<30 and i>-30) for i in np.array(tp_df.geometry.x) ])]
    
                    # col names
                    tropo = 'tropo_'+tp.col_name# 'tropo_%s%s_%s' % (tp.tp_def, '_'+f'{tp.pvu}' if tp.tp_def == 'dyn' else '', tp.vcoord)
                    strato ='strato_'+tp.col_name # 'strato_%s%s_%s' % (tp.tp_def, '_'+f'{tp.pvu}' if tp.tp_def == 'dyn' else '', tp.vcoord)
    
                    tp_sorted = pd.DataFrame({strato:pd.Series(np.nan, dtype='float'),
                                              tropo:pd.Series(np.nan, dtype='float')},
                                             index=tp_df.index)
    
                    # tropo: high p (gt 0), low everything else (lt 0)
                    tp_sorted.loc[tp_df[tp.col_name].gt(0) if tp.vcoord=='p' else tp_df[tp.col_name].lt(0),
                                (strato, tropo)] = (False, True)
    
                    # strato: low p (lt 0), high everything else (gt 0)
                    tp_sorted.loc[tp_df[tp.col_name].lt(0) if tp.vcoord=='p' else tp_df[tp.col_name].gt(0),
                                (strato, tropo)] = (True, False)
    
                    df_tropo = tp_df[tp_sorted[tropo] == True]
                    df_strato = tp_df[tp_sorted[strato] == True]
    
                    if plot: # plot data
                        # df_tropo, df_strato = plot_sorted_TP(TropopauseData(), df_sorted, vcoord, subs.col_name, ax=ax)
                        ax.set_title(tp.col_name, fontsize=8)
                        ax.scatter(df_strato.index, df_strato[subs.col_name],
                                    c='grey',  marker='.', zorder=0, label='strato')
                        ax.scatter(df_tropo.index, df_tropo[subs.col_name],
                                    c='xkcd:kelly green',  marker='.', zorder=1, label='tropo')
    
                    df_sorted[tropo] = tp_sorted[tropo]
                    df_sorted[strato] = tp_sorted[strato]
    
                if plot: # add legend, format axes, ...
                    fig.autofmt_xdate()
                    fig.tight_layout()
                    fig.subplots_adjust(top=0.85)
                    lines, labels = axs.flatten()[0].get_legend_handles_labels()
                    fig.legend(lines, labels, loc='upper center', ncol=2,
                               bbox_to_anchor=[0.5, 0.94])
                    plt.show()
        return df_sorted

    def trop_strat_ratios(self):
        """ Plot ratio of tropo / strato datapoints for each troposphere definition """
        substances = get_substances(source='EMAC') + get_substances(source='Caribic')
        substances = [s for s in substances if not s.col_name.startswith('d_')]
    
        for vcoord in ['p', 'z', 'pt']:
            out = self.sort_tropo_strato(substances, vcoords=[vcoord], plot=False)
            val_count = out[[c for c in out.columns if c.startswith('strato')]].apply(pd.value_counts)
            plt.figure(figsize=(3,3), dpi=240)
            plt.title(f'Ratio of tropospheric / stratospheric datapoints in {vcoord}')
            for ratio, l in zip([val_count[c][0] / val_count[c][1] for c in val_count.columns], list(val_count.columns)):
                plt.hlines(l, 0, ratio, lw=12, alpha=0.7)
            plt.axvspan(0.995, 1.005, facecolor='k', alpha=0.7)
            plt.show()

#%% Plotting function calls
if __name__=='__main__':
    tp = TropopausePlotter().sel_flight(270)
    
    # Global 2D scatter of tropopause heights for all definitions
    for year in range(2005, 2020):
        tp.scatter_2d(save=False, year=year)
    
    #  Binned versus latitude
    for vc in ['p', 'pt', 'z']:
        tp.tropopause_vs_latitude(vc, rel=False)
        # tropopause_vs_latitude(vc, rel=True)
        tp.tropopause_vs_latitude(vc, rel=False, seasonal=True)

    # Substances in tropopsphere / stratosphere
    substances = get_substances(source='EMAC') + get_substances(source='Caribic')
    substances = [s for s in substances if not s.col_name.startswith('d_')]
    # df_sorted = sort_tropo_strato(substances)
    tp.trop_strat_ratios()

#%% Animate changes over years
def make_gif():
    pdir = r'C:\Users\sophie_bauchinger\sophie_bauchinger\Figures\tp_scatter_2d'
    for vc in ['p', 'pt', 'z']:
        tps = get_coordinates(vcoord=vc, tp_def='not_nan', rel_to_tp=False)
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

#%% Nope
# =============================================================================
# def plot_sorted_TP(self, df_sorted, vcoord, subs_col, ax):
#     """ Plot strat / trop sorted data """
#     # only take data with index that is available in df_sorted
#     df_sorted = self.sort_tropo_strato(vcoord)
#     
#     df = self.df
#     data = df[df.index.isin(df_sorted.index)]
#     data.sort_index(inplace=True)
# 
#     # separate trop/strat data for any criterion
#     tropo_col = [col for col in df_sorted.columns if col.startswith('tropo')][0]
#     strato_col = [col for col in df_sorted.columns if col.startswith('strato')][0]
# 
#     # take 'data' here because substances may not be available in df_sorted
#     df_tropo = data[df_sorted[tropo_col] == True]
#     df_strato = data[df_sorted[strato_col] == True]
# 
#     substance = subs_col
#     if 'detr_'+substance in data.columns: substance = 'detr_'+substance
# 
#     # fig, ax = plt.subplots(dpi=200)
#     # plt.title(f'{subs_col} in {vcoord}')
#     ax.scatter(df_strato.index, df_strato[substance],
#                 c='grey',  marker='.', zorder=0, label='strato')
#     ax.scatter(df_tropo.index, df_tropo[substance],
#                 c='xkcd:kelly green',  marker='.', zorder=1, label='tropo')
# 
#     # if popt0 is not None and popt1 is not None and (subs==crit or subs is None):
#     #     # only plot baseline for chemical tropopause def and where crit is being plotted
#     #     t_obs_tot = np.array(dt_to_fy(df_sorted.index, method='exact'))
#     #     ls = 'solid'
#     #     if not subs_pfx == c_pfx: ls = 'dashed'
#     #     ax.plot(df_sorted.index, get_fct_substance(crit)(t_obs_tot-2005, *popt0),
#     #             c='r', lw=1, ls=ls, label='initial')
#     #     ax.plot(df_sorted.index, get_fct_substance(crit)(t_obs_tot-2005, *popt1),
#     #             c='k', lw=1, ls=ls, label='filtered')
# 
#     # plt.ylim(220, 340)
# 
#     ax.set_title(substance)
#     ax.legend()
#     return df_tropo, df_strato
# =============================================================================
