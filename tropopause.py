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

from toolpac.calc.binprocessor import Bin_equi1d, Simple_bin_1d, Bin_equi2d, Simple_bin_2d

from dictionaries import get_coordinates, dict_season, get_substances
from data import GlobalData, Caribic, EMACData
from tools import assign_t_s

#%% Import data
class TropopauseData(GlobalData):
    """ Holds Caribic data and Caribic-specific EMAC Model output """
    def __init__(self, years=range(2005, 2020)):
        if isinstance(years, int): years = [years]
        super().__init__([yr for yr in years if yr >= 2000 and yr <= 2019])
        self.source = 'TP'
        self.data = {}
        self.get_data(years)

    def __repr__(self):
        self.years.sort() 
        return f'TropopauseData object\n\
            years: {self.years}\n\
            status: {self.status}'
            
    def get_data(self, years=range(2005,2020)):
        """ Return merged dataframe with interpolated EMAC / Caribic data """
        caribic = Caribic(years=years)
        emac = EMACData(years=years)
        df_caribic = caribic.df
        df_emac = emac.df
        df = pd.merge( df_caribic, df_emac, how='outer', sort=True,
                      left_index=True, right_index=True)
        df.geometry = df_caribic.geometry.combine_first(df_emac.geometry)
        df = df.drop(columns=['geometry_x', 'geometry_y'])
        df['Flight number'].interpolate(method='nearest', inplace=True) #TODO add other variables here
        # for c in ['Flight number']:
        #     if c in df.columns:
        #         df[c].interpolate(method='nearest', inplace=True)
        self.data['df'] = df
        return df
    
    @property
    def df(self):
        return self.data['df']

tropopause_data = TropopauseData()

#%% Global 2D scatter of tropopause heights for all definitions 
world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
vlims = {'p':(100,500), 'pt':(250, 350), 'z':(5,20)}

def scatter_2d():
    for vc in ['p', 'pt', 'z']:
        tps = get_coordinates(vcoord=vc, tp_def='not_nan', rel_to_tp=False)
        for tp in tps:
            fig, axs = plt.subplots(2,2,dpi=150, figsize=(10,5))
            fig.suptitle(f'{tp.col_name} - {tp.long_name}')
            for s,ax in zip([1,2,3,4], axs.flatten()):
                df_r = tropopause_data.sel_season(s).df
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
            plt.show()

scatter_2d()

#%% Binned versus latitude
def tropopause_vs_latitude(vcoord, rel, seasonal=False):
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
            data = tropopause_data.df.copy()
            if seasonal: 
                data = tropopause_data.sel_season(s).df 
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
                                    edgecolor='grey', alpha=0.5, pad=0.1))
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

        # add horizontal legend for seasons 
        lines, labels = axs.flatten()[0].get_legend_handles_labels()
        fig.legend(lines, labels, loc='lower center', ncol=4)

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
    plt.show()

if __name__=='__main__':
    for vc in ['p', 'pt', 'z']:
        tropopause_vs_latitude(vc, rel=False)
        # tropopause_vs_latitude(vc, rel=True)
        tropopause_vs_latitude(vc, rel=False, seasonal=True)

#%% Copy plot_sorted from tropFilter
# Plotting sorted data
def plot_sorted_TP(glob_obj, df_sorted, vcoord, subs_col, ax):
    """ Plot strat / trop sorted data """
    # only take data with index that is available in df_sorted
    df = glob_obj.df
    data = df[df.index.isin(df_sorted.index)]
    data.sort_index(inplace=True)

    # separate trop/strat data for any criterion
    tropo_col = [col for col in df_sorted.columns if col.startswith('tropo')][0]
    strato_col = [col for col in df_sorted.columns if col.startswith('strato')][0]

    # take 'data' here because substances may not be available in df_sorted
    df_tropo = data[df_sorted[tropo_col] == True]
    df_strato = data[df_sorted[strato_col] == True]

    substance = subs_col
    if 'detr_'+substance in data.columns: substance = 'detr_'+substance

    # fig, ax = plt.subplots(dpi=200)
    # plt.title(f'{subs_col} in {vcoord}')
    ax.scatter(df_strato.index, df_strato[substance],
                c='grey',  marker='.', zorder=0, label='strato')
    ax.scatter(df_tropo.index, df_tropo[substance],
                c='xkcd:kelly green',  marker='.', zorder=1, label='tropo')

    # if popt0 is not None and popt1 is not None and (subs==crit or subs is None):
    #     # only plot baseline for chemical tropopause def and where crit is being plotted
    #     t_obs_tot = np.array(dt_to_fy(df_sorted.index, method='exact'))
    #     ls = 'solid'
    #     if not subs_pfx == c_pfx: ls = 'dashed'
    #     ax.plot(df_sorted.index, get_fct_substance(crit)(t_obs_tot-2005, *popt0),
    #             c='r', lw=1, ls=ls, label='initial')
    #     ax.plot(df_sorted.index, get_fct_substance(crit)(t_obs_tot-2005, *popt1),
    #             c='k', lw=1, ls=ls, label='filtered')

    # plt.ylim(220, 340)

    ax.set_title(substance)
    ax.legend()
    return df_tropo, df_strato

#%% Substances in tropopsphere / stratosphere
substances = get_substances(source='EMAC') + get_substances(source='Caribic')
substances = [s for s in substances if not s.col_name.startswith('d_')]

def sort_tropo_strato(substances, vcoords=['p', 'z', 'pt'], plot=True):
    data = tropopause_data.df.copy()
    out = pd.DataFrame(index=data.index)

    for subs in substances: 
        if not subs.col_name in data.columns:
            print(f'{subs.col_name} not found in data'); pass
        for vcoord in vcoords:
            tps = get_coordinates(vcoord=vcoord, tp_def='not_nan', rel_to_tp=True)
            for tp in [tp for tp in tps if tp.pvu in [1.5, 2.0]]: # rmv 1.5 and 2.0 PVU TPs
                tps.remove(tp)
            
            if plot: # initialise figure
                fig, axs = plt.subplots(math.ceil(len(tps)/2), 2, dpi=200)
                if len(tps)%2: axs.flatten()[-1].axis('off')
                fig.suptitle(f'{subs.col_name} in sorted with TP in {vcoord}')
    
            for tp, ax in zip(tps, axs.flatten()): 
                tp_df = data.dropna(axis=0, subset=[tp.col_name])

                # col names
                tropo = 'tropo_'+tp.col_name# 'tropo_%s%s_%s' % (tp.tp_def, '_'+f'{tp.pvu}' if tp.tp_def == 'dyn' else '', tp.vcoord)
                strato ='strato_'+tp.col_name # 'strato_%s%s_%s' % (tp.tp_def, '_'+f'{tp.pvu}' if tp.tp_def == 'dyn' else '', tp.vcoord)
                
                df_sorted = pd.DataFrame({strato:pd.Series(np.nan, dtype='float'),
                                          tropo:pd.Series(np.nan, dtype='float')},
                                         index=tp_df.index)
            
                # tropo: high p (gt 0), low everything else (lt 0)
                df_sorted.loc[tp_df[tp.col_name].gt(0) if tp.vcoord=='p' else tp_df[tp.col_name].lt(0),
                            (strato, tropo)] = (False, True)
    
                # strato: low p (lt 0), high everything else (gt 0)
                df_sorted.loc[tp_df[tp.col_name].lt(0) if tp.vcoord=='p' else tp_df[tp.col_name].gt(0),
                            (strato, tropo)] = (True, False)
            
                df_tropo = tp_df[df_sorted[tropo] == True]
                df_strato = tp_df[df_sorted[strato] == True]
    
                if plot: # plot data
                    # df_tropo, df_strato = plot_sorted_TP(tropopause_data, df_sorted, vcoord, subs.col_name, ax=ax)
                    ax.set_title(tp.col_name, fontsize=8)
                    ax.scatter(df_strato.index, df_strato[subs.col_name],
                                c='grey',  marker='.', zorder=0, label='strato')
                    ax.scatter(df_tropo.index, df_tropo[subs.col_name],
                                c='xkcd:kelly green',  marker='.', zorder=1, label='tropo')
                
                out[tropo] = df_sorted[tropo]
                out[strato] = df_sorted[strato]
            
            if plot: # add legend, format axes, ...
                lines, labels = axs.flatten()[0].get_legend_handles_labels()
                fig.legend(lines, labels, loc='lower center', ncol=2)
                    
                fig.autofmt_xdate()
                fig.tight_layout()
                plt.show()
    return out

#%% Old scatter plots
# =============================================================================
# def tp_vs_latitude(ax, obj, plot_params, **tp_params):
#     """ """
#     data = obj.df.copy()
#     if not tp_params['col_name'] in data.columns: 
#         print('{} not in columns'.format(tp_params['col_name']))
#         return
# 
#     if tp_params['tp_def'] == 'dyn': # dynamic TP only outside the tropics
#         data = data[np.array([(i>30 or i<-30) for i in np.array(data.geometry.x) ])]
#     if tp_params['tp_def'] == 'cpt': # cold point TP only in the tropics 
#         data = data[np.array([(i<30 and i>-30) for i in np.array(data.geometry.x) ])]
# 
#     x = np.array(data.geometry.x)
#     v = data[tp_params['col_name']]
# 
#     ax.scatter(x, v, label = '{}_{}{}'.format(
#         tp_params['model'], tp_params['tp_def'], 
#         '_'+str(tp_params['pvu']) if tp_params['tp_def']=='dyn' else ''))
# 
#     # if tp_params['var1'] in data.columns and not tp_params['rel_to_tp']:
#     #     ax.scatter(x, data[tp_params['var1']], c='k')
# 
#     ax.set_xlabel('Latitude [°N]')
#     ax.set_ylabel('{}{} [{}]'.format('$\Delta$' if tp_params['rel_to_tp'] else '', 
#                                      tp_params['vcoord'], tp_params['unit']))
#     # ax.set_ylim(plot_params['ylim'])
#     # ax.set_xlim(plot_params['xlim'])
#     return
# 
# def av_tp_vs_latitude(ax, obj, plot_params, **tp_params):
#     """ """
#     data = obj.df.copy()
#     if not tp_params['col_name'] in data.columns: return
# 
#     if tp_params['tp_def'] == 'dyn': # dynamic TP only outside the tropics
#         data = data[np.array([(i>30 or i<-30) for i in np.array(data.geometry.x) ])]
#     if tp_params['tp_def'] == 'cpt': # cold point TP only in the tropics 
#         data = data[np.array([(i<30 and i>-30) for i in np.array(data.geometry.x) ])]
# 
#     x = np.array(data.geometry.x)
#     v = data[tp_params['col_name']]
# 
#     xbmin, xbmax, xbsize = -90, 90, 10
#     bci = Bin_equi1d(xbmin, xbmax, xbsize)
#     bin1d = Simple_bin_1d(v,x,bci)
#     
#     # ax.plot(bin1d.xmean, bin1d.vmean, label = ID)
#     #colors = {'clim':'grey', 'cpt':'blue', 'dyn':'green', 'therm':'red', 'combo':'grey'}
#     ax.plot(bin1d.xmean, bin1d.vmean, #c=colors[tp_params['tp_def']],
#                label = '{}_{}{}'.format(tp_params['model'], tp_params['tp_def'], 
#                                           '_'+str(tp_params['pvu']) if tp_params['tp_def']=='dyn' else ''))
#     ax.fill_between(bin1d.xmean, bin1d.vmean-bin1d.vstdv, bin1d.vmean+bin1d.vstdv, alpha=0.3)
#     # ax.scatter(bin1d.xmean, bin1d.vmean, label = tp_params['col_name'])   
#     # ax.errorbar(bin1d.xmean, bin1d.vmean, bin1d.vstdv, capsize=2)
# 
#     ax.set_xlabel('Latitude [°N]')
#     ax.set_ylabel('{}{} [{}]'.format('$\Delta$' if tp_params['rel_to_tp'] else '', 
#                                      tp_params['vcoord'], tp_params['unit']))
#     # ax.set_ylim(plot_params['ylim'])
#     # ax.set_xlim(plot_params['xlim'])
#     return
# 
# def tp_vs_lat(vcoord, rel, av=True):
#     tps = get_coordinates(vcoord=vcoord, tp_def='not_nan', rel_to_tp=rel)
#     fig, axs = plt.subplots(round(len(tps)/2), 2, dpi=150, figsize=(10,7), sharey=True, sharex=True)
#     fig.suptitle('{} {}'.format('TP in' if not rel else 'TP wrt flight in', vcoord))
#     for tp, ax in zip(tps, axs.flatten()):
#         if av: 
#             av_tp_vs_latitude(ax, tropopause_data, 
#                             plot_params = {},#'ylim':(50, 500), 'xlim':(-40, 90)},
#                             **tp.__dict__)
#         else: 
#             tp_vs_latitude(ax, tropopause_data, 
#                             plot_params = {},#'ylim':(50, 500), 'xlim':(-40, 90)},
#                             **tp.__dict__)
#         # if vcoord == 'p': 
#         #     ax.invert_yaxis()
#         #     ax.set_yscale('log')
#     # plt.legend()
#     # lines, labels = axs.flatten()[0].get_legend_handles_labels()    
#     # fig.legend(lines, labels, loc='center right')
#     fig.tight_layout()
#     plt.show()
# =============================================================================

#%% Old Seasonal plots
# =============================================================================
# def seasonal_av_tp_vs_latitude(axs, obj, plot_params, **tp_params):
#     """ """
#     data = obj.df.copy()
#     if not tp_params['col_name'] in data.columns: return
# 
#     data['season'] = make_season(data.index.month) # 1 = spring etc
# 
#     if tp_params['tp_def'] == 'dyn': # dynamic TP only outside the tropics
#         data = data[np.array([(i>30 or i<-30) for i in np.array(data.geometry.x) ])]
#     if tp_params['tp_def'] == 'cpt': # cold point TP only in the tropics 
#         data = data[np.array([(i<30 and i>-30) for i in np.array(data.geometry.x) ])]
# 
#     for s,ax in zip(set(data['season'].tolist()), axs.flatten()):
#         df = data.loc[data['season'] == s]
#         x = np.array(df.geometry.x)
#         v = df[tp_params['col_name']]
#     
#         xbmin, xbmax, xbsize = -90, 90, 10
#         bci = Bin_equi1d(xbmin, xbmax, xbsize)
#         bin1d = Simple_bin_1d(v,x,bci)
# 
#         ax.scatter(bin1d.xmean, bin1d.vmean, # c=colors[tp_params['tp_def']],
#                label = '{}_{}{}'.format(tp_params['model'], tp_params['tp_def'], 
#                                           '_'+str(tp_params['pvu']) if tp_params['tp_def']=='dyn' else ''))
#                    # c = dict_season()[f'color_{s}'],
#                    # label = dict_season()[f'name_{s}'])
#         ax.errorbar(bin1d.xmean, bin1d.vmean, bin1d.vstdv, capsize=2)
# 
#         ax.set_xlabel('Latitude [°N]')
#         ax.set_ylabel('{}{} [{}]'.format('$\Delta$' if tp_params['rel_to_tp'] else '', 
#                                          tp_params['vcoord'], tp_params['unit']))
#         ax.set_title(dict_season()[f'name_{s}'])
#         # ax.set_ylim(plot_params['ylim'])
#         # ax.set_xlim(plot_params['xlim'])
#     return
# 
# def seasonal_tp_scatter_vs_latitude(vcoord, rel):
#     tps = get_coordinates(vcoord=vcoord, tp_def='not_nan', rel_to_tp=rel)
#     fig, axs = plt.subplots(2,2, dpi=150, figsize=(9,5))
#     plt.suptitle(f'TP in {vcoord}')
#     for tp in tps:
#         seasonal_av_tp_vs_latitude(axs, tropopause_data, 
#                         plot_params = {},#'ylim':(50, 500), 'xlim':(-40, 90)},
#                         **tp.__dict__)
# 
#     if vcoord == 'p':
#         for ax in axs.flatten(): 
#             ax.invert_yaxis()
#             ax.set_yscale('log')
#     fig.tight_layout()
#     
#     lines, labels = axs.flatten()[0].get_legend_handles_labels()
#     # lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
#     fig.legend(lines, labels, loc='center right')
#     plt.subplots_adjust(right=0.8)
#     
#     plt.show()
#     return
# 
# =============================================================================
