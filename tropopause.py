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

#%% Import data
class TropopauseData(GlobalData):
    """ Holds Caribic data and Caribic-specific EMAC Model output """
    def __init__(self, years=range(2005, 2020), interp=True, method='n'):
        if isinstance(years, int): years = [years]
        super().__init__([yr for yr in years if yr >= 2000 and yr <= 2019])
        self.source = 'TP'
        self.data = {}
        self.get_data()
        # select year on this object afterwards bc otherwise interpolation is missing surrounding values
        if interp: self.interpolate_emac(method)
        self.data = self.sel_year(*years).data

    def __repr__(self):
        self.years.sort()
        return f'TropopauseData object\n\
            years: {self.years}\n\
            status: {self.status}'

    def get_data(self):
        """ Return merged dataframe with interpolated EMAC / Caribic data """
        caribic = Caribic() #.sel_year(*years)
        emac = EMACData() #.sel_year(*years)
        df_caribic = caribic.df
        df_emac = emac.df
        df = pd.merge( df_caribic, df_emac, how='outer', sort=True,
                      left_index=True, right_index=True)
        df.geometry = df_caribic.geometry.combine_first(df_emac.geometry)
        df = df.drop(columns=['geometry_x', 'geometry_y'])
        df['Flight number'].interpolate(method='nearest', inplace=True) #TODO add other variables here
        df['Flight number'].interpolate(inplace=True, limit_direction='both') # fill in first two timestamps too
        df['Flight number'] = df['Flight number'].astype(int)
        self.data['df'] = df
        return df

    @property
    def df(self):
        """ Allow accessing df as class attribute. """
        return self.data['df']

    def interpolate_emac(self, method, verbose=True):
        """ Add interpolated EMAC data to joint df to match caribic timestamps.

        Parameters:
            method (str): interpolation method. Limit is set to 2 consecutive NaN values
                'n' - nearest neighbour, 'b' - bilinear

        Note: Residial NaN values in nearest because EMAC only goes to 2019.
        Explanation on methods see at https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html
        """
        data = self.df.copy()
        tps_emac = [i.col_name for i in get_coordinates(source='EMAC') if i.col_name in self.df.columns] + [
            i for i in ['ECHAM5_tm1_at_fl', 'ECHAM5_tpoteq_at_fl', 'ECHAM5_press_at_fl'] if i in self.df.columns]
        subs_emac = [i.col_name for i in get_substances(source='EMAC') if i.col_name in self.df.columns]

        nan_count_i = data[tps_emac[0]].isna().value_counts().loc[True]
        for c in tps_emac+subs_emac:
            if method=='b': data[c].interpolate(method='linear', inplace=True, limit=2)
            elif method=='n': data[c].interpolate(method='nearest', inplace=True, limit=2)
            else: raise KeyError('Please choose either b-linear or n-nearest neighbour interpolation.')
            data[c] = data[c].astype(float)
        nan_count_f = data[tps_emac[0]].isna().value_counts().loc[True]

        if verbose: print('{} NaNs in EMAC data filled using {} interpolation'.format(
                nan_count_i-nan_count_f, 'nearest neighbour' if method=='n' else 'linear'))

        self.data['df'] = data
        return data

tropopause_data = TropopauseData()

#%% Global 2D scatter of tropopause heights for all definitions
world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
vlims = {'p':(100,500), 'pt':(250, 350), 'z':(5,20)}

def scatter_2d(save=False):
    pdir = r'C:\Users\sophie_bauchinger\sophie_bauchinger\Figures\tp_scatter_2d'
    for vc in ['p', 'pt', 'z']:
        tps = get_coordinates(vcoord=vc, tp_def='not_nan', rel_to_tp=False)
        for tp in tps:
            fig, axs = plt.subplots(2,2,dpi=150, figsize=(10,5))
            fig.suptitle(f'{tp.col_name} - {tp.long_name}')
            for s,ax in zip([1,2,3,4], axs.flatten()):
                df_r = TropopauseData().sel_season(s).df
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
            if save: plt.savefig(pdir+f'\{tp.col_name}.png')
            else: plt.show()

scatter_2d(save=True)

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
def sort_tropo_strato(substances, vcoords=['p', 'z', 'pt'], plot=True):
    """ Returns """
    data = tropopause_data.df.copy()
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
                    # df_tropo, df_strato = plot_sorted_TP(tropopause_data, df_sorted, vcoord, subs.col_name, ax=ax)
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

def trop_strat_ratios():
    """ Plot ratio of tropo / strato datapoints for each troposphere definition """
    substances = get_substances(source='EMAC') + get_substances(source='Caribic')
    substances = [s for s in substances if not s.col_name.startswith('d_')]

    for vcoord in ['p', 'z', 'pt']:
        out = sort_tropo_strato(substances, vcoords=[vcoord], plot=False)
        val_count = out[[c for c in out.columns if c.startswith('strato')]].apply(pd.value_counts)
        plt.figure(figsize=(3,3), dpi=240)
        plt.title(f'Ratio of tropospheric / stratospheric datapoints in {vcoord}')
        for ratio, l in zip([val_count[c][0] / val_count[c][1] for c in val_count.columns], list(val_count.columns)):
            plt.hlines(l, 0, ratio, lw=12, alpha=0.7)
        plt.axvspan(0.995, 1.005, facecolor='k', alpha=0.7)
        plt.show()

if __name__=='__main__':
    substances = get_substances(source='EMAC') + get_substances(source='Caribic')
    substances = [s for s in substances if not s.col_name.startswith('d_')]
    # df_sorted = sort_tropo_strato(substances)
    trop_strat_ratios()

#%% Animate changes over years



# save the images to create a gif from as png
fig = scatter_2d(save=True)


# images = []
# for filename in filenames:
#     images.append(imageio.imread(filename))
# imageio.mimsave('/path/to/movie.gif', images)