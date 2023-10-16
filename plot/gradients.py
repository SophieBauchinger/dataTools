# -*- coding: utf-8 -*-
"""
@Author: Sophie Bauchinger, IAU
@Date: Fri Apr 28 14:13:28 2023

Plot vertical profiles of Caribic data split into seasons. 

"""
import numpy as np
import matplotlib.pyplot as plt

import toolpac.calc.binprocessor as bp

import tools
import dictionaries as dcts

#%% Plotting Gradient by season
def plot_gradient_by_season(c_obj, subs_params = {}, y_params = {},
                            detr=True, note=None, errorbars=False):
    """
    Plotting gradient by season using 1D binned data. Detrended data used by default.
    (Based on C_plot.pl_gradient_by_season)

    subs_params (dict):
        keys: short_name, ID
    y_params (dict):
        keys: vcoord, tp_def, ID, (pvu)
    """
    substance = dcts.get_subs(**subs_params)
    if not substance.short_name in c_obj.data.keys(): 
        c_obj.create_substance_df(substance.short_name)
    data = c_obj.data[substance.short_name]

    subs_col = substance.col_name # subs, c_obj.source, x_params['ID'])
    if detr: # detrended data for x-axis
        subs_col = f'detr_{subs_col}'
        if not subs_col in data.columns:
            raise ValueError(f'Detrended data not available for {subs.upper()}')

    # y-axis
    y_coord = dcts.get_coordinates(**y_params)[0]
    y_label = dcts.make_coord_label(y_coord)
    y_bins = {'z' : 0.5, 'pt' : 10, 'p' : 40}
    if not 'y_bin' in y_params.keys():
        y_bin = y_bins[y_params['vcoord']]
    else: y_bin = y_params['y_bin']

    min_y, max_y = np.nanmin(data[y_coord.col_name].values), np.nanmax(data[y_coord.col_name].values)
    nbins = (max_y - min_y) / y_bin
    y_array = min_y + np.arange(nbins) * y_bin + y_bin * 0.5

    data['season'] = tools.make_season(data.index.month) # 1 = spring etc
    out_dict = {}
    fig, ax = plt.subplots(dpi=200)
    for s in set(data['season'].tolist()):
        df = data.loc[data['season'] == s]
        y_values = df[y_coord.col_name].values # df[eq_lat_col].values #
        x_values = df[subs_col].values

        out_dict[f'bin1d_{s}'] = bp.Simple_bin_1d(x_values, y_values,
                                                  bp.Bin_equi1d(min_y, max_y, y_bin))
        vmean = (out_dict[f'bin1d_{s}']).vmean
        vcount = (out_dict[f'bin1d_{s}']).vcount
        vmean = np.array([vmean[i] if vcount[i] >= 5
                          else np.nan for i in range(len(vmean))])

        plt.plot(vmean, y_array, '-', marker='o', c=dcts.dict_season()[f'color_{s}'],
                 label=dcts.dict_season()[f'name_{s}'])

        if errorbars: # add error bars
            vstdv = (out_dict[f'bin1d_{s}']).vstdv
            plt.errorbar(vmean, y_array, None, vstdv,
                         c=dcts.dict_season()[f'color_{s}'], elinewidth=0.5)

    plt.tick_params(direction='in', top=True, right=True)
    if note: 
        ax.text(**dcts.note_dict(ax, x=0, y=1.08, s=note))

    plt.ylim([min_y, max_y])
    plt.ylabel(y_label)
    plt.xlabel(dcts.make_subs_label(substance, detr=detr))
    if detr: # remove the 'delta_' and replace with symbol
        plt.xlabel(subs_col.split("_")[-1] + ' detrended wrt. 2005')

    plt.legend()
    plt.show()

#%% Fct calls - gradients

if __name__ == '__main__':
    from data import Caribic
    caribic = Caribic() # BS to avoid error

    # yp1 = {'tp_def' : 'chem',
    #        'ID' : 'INT2',
    #        'vcoord' : 'z'}

    yp2 = {'tp_def' : 'therm',
           'ID' : 'INT',
           'vcoord' : 'pt'}

    yp3 = {'tp_def' : 'therm',
           'ID' : 'INT2',
           'vcoord' : 'pt'}

    yp4 = {'tp_def' : 'dyn',
           'ID' : 'INT',
           'vcoord' : 'pt',
           'pvu' : 3.5}

    for subs in ['sf6', 'n2o', 'co2', 'ch4']:
        for yp in [yp2, yp3, yp4]:
            subs_params = {'short_name': subs, 'ID' : 'GHG'}
            plot_gradient_by_season(caribic.sel_latitude(30, 90), subs_params, yp, note='lat>30°N', detr=False)
            try: plot_gradient_by_season(caribic.sel_latitude(30, 90), subs_params, yp, note='lat>30°N', detr=True)
            except ValueError: pass
