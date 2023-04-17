

import matplotlib.pyplot as plt
import colorcet as cc
import numpy as np

import C_tools



# bla = plot_scatter_samples(df_flights, Fdata, 'lon', 'lat', df_route_colors, flight_numbers,
#                          color='month', df_return=True, add_all=True, select_var=['ol_ch4','ol_co2'], select_value=[0.,0.], select_cf=['GT','LT'])

# bla = plot_scatter_samples(df_flights, Fdata, 'lon', 'lat', df_route_colors, flight_numbers,
  #                        color='month', df_return=True, add_all=True, select_var=['ol_ch4','ol_n2o'], select_value=[0.,0.], select_cf=['GT','GT'])

# bla = plot_scatter_samples(df_flights, Fdata, 'lon', 'lat', df_route_colors, flight_numbers,
 #                         color='month', df_return=True, add_all=True, select_var=['fl_ch4','fl_sf6', 'fl_n2o'], select_value=[0,0,0], select_cf=['GT','GT', 'GT'])

"""
for flight in bla.flight.unique():

    # get sample number from Fdata and the read WAS traj data
    TrajData = read_WAS_trajs(path, flight, sample_no)

"""


def plot_ol_rel_corr(Fdata, flight_numbers, subst1, subst2, col='month', rel=False,
                     flagged1=False, flagged2=False):
    df_merge = C_tools.do_data_merge(Fdata, flight_numbers, prefixes=['INT', 'GHG', 'HCF', 'HFO', 'FLG'], verbose=True)

    x_array = df_merge[f'ol_{subst1}'].tolist()
    y_array = df_merge[f'ol_{subst2}'].tolist()
    # print(len(x_array), len(y_array))
    if rel:
        x_array = [100*x for x in df_merge[f'ol_rel_{subst1}'].tolist()]
        y_array = [100*y for y in df_merge[f'ol_rel_{subst2}'].tolist()]

    if flagged1:
        x_flag = df_merge[f'fl_{subst1}']
        x_array = [np.nan if x_flag[i] == 0 else x for i, x in enumerate(x_array)]
    if flagged2:
        y_flag = df_merge[f'fl_{subst2}']
        y_array = [np.nan if y_flag[i] == 0 else y for i, y in enumerate(y_array)]
    # print(len(x_array), len(y_array))

    if col == 'month':
        color = df_merge[col]
        cmin = 1
        cmax = 12
        col_label = col
    else:
        color = df_merge[col]
        cmin = df_merge[col].min()
        cmax = df_merge[col].max()
        col_label = col

    cmap = cc.cm.rainbow  # 'rainbow'

    plt.figure(figsize=(6, 5))
    plt.scatter(x_array, y_array, marker='o', c=color, cmap=cmap,
                vmin=cmin, vmax=cmax)

    # add zero lines
#    plt.plot([0, 0], plt.xlim(), color='black')
#    plt.plot(plt.xlim(), [0, 0], color='black')

    plt.hlines(0, plt.xlim()[0], plt.xlim()[1], color='black')
    plt.vlines(0, plt.ylim()[0], plt.ylim()[1], color='black')

    if rel:
        plt.xlabel(f'{subst1} enhancement [%]')
        plt.ylabel(f'{subst2} enhancement [%]')
    else:
        plt.xlabel(f'{subst1} enhancement (abs)')
        plt.ylabel(f'{subst2} enhancement (abs)')

    plt.colorbar(label=col_label)
    plt.clim(cmin, cmax)

    plt.show()
