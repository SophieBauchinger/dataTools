# -*- coding: utf-8 -*-
"""
@Author: Sophie Bauchinger, IAU
@Date Wed Mar 13 09:41:40 2024

"""
import pandas as pd
import matplotlib.pyplot as plt

from plot.binsubs import BinPlotter1D
from data import Caribic
import dictionaries as dcts

caribic = Caribic()
bp1 = BinPlotter1D(caribic)

caribic.detrend_substance(dcts.get_subs(col_name='SF6'))
subs = dcts.get_subs(col_name='detr_SF6')

n2o = dcts.get_coord(col_name='N2O')

Tcaribic = caribic.sel_tropo(vcoord='mxr', crit='n2o')
Scaribic = caribic.sel_strato(vcoord='mxr', crit='n2o')
bp_t = BinPlotter1D(Tcaribic)
bp_s = BinPlotter1D(Scaribic)

tps = caribic.tp_coords(rel_to_tp=True, model='ERA5', vcoord='pt')
tps.pop(0)

rms_rvstd = {}

for tp in tps: 
    df = bp1.rms_seasonal_vstdv(subs, tp)
    # rms_rvstd[tp.col_name + '_all'] = df['rms_rvstd']
    rms_rvstd[tp.label(True) + '_s'] = df['rms_rvstd'][df.index > 0]
    rms_rvstd[tp.label(True) + '_t'] = df['rms_rvstd'][df.index < 0]
    
theta = dcts.get_coord(vcoord='pt', model='ERA5', tp_def='nan', col_name='int_ERA5_THETA')

for tp in [theta]: # [dcts.get_coord(vcoord='mxr', crit='n2o', ID='GHG')]:
    df_t = bp_t.rms_seasonal_vstdv(subs, tp)
    df_s = bp_s.rms_seasonal_vstdv(subs, tp)
    
    rms_rvstd['N2O_THETA_t'] = df_t['rms_rvstd']
    rms_rvstd['N2O_THETA_s'] = df_s['rms_rvstd']
    
for tp in [dcts.get_coord(col_name='int_h_rel_TP')]:
    df = bp1.rms_seasonal_vstdv(subs, tp)
    # rms_rvstd[tp.col_name + '_all'] = df['rms_rvstd']
    rms_rvstd[tp.label(True) + '_s'] = df['rms_rvstd'][df.index > 0]
    rms_rvstd[tp.label(True) + '_t'] = df['rms_rvstd'][df.index < 0]

print([f'{k} : {i.mean()*100:.3} %' for k,i in rms_rvstd.items()])


def strato_tropo_stdv_table(self, subs, tps=None, **kwargs):
    """ Creates a table with the """
    stdv_df = self.strato_tropo_stdv(subs, tps)
    
    if kwargs.get('rel'): 
        stdv_df = stdv_df[[c for c in stdv_df.columns if 'rel' in c]]
    else: 
        stdv_df = stdv_df[[c for c in stdv_df.columns if 'rel' not in c]]
    
    stdv_df = stdv_df.astype(float).round(2 if kwargs.get('rel') else 3)
    stdv_df.sort_index()
    
    fig, ax = plt.subplots(dpi=250)

    rowLabels = [dcts.get_coord(col_name=c).label(True) for c in stdv_df.index]
    colLabels = [f'Troposphere [{subs.unit}]', f'Stratosphere [{subs.unit}]'] if not kwargs.get('rel') else ['Troposphere [%]', 'Stratosphere [%]']
    table = ax.table(stdv_df.values, 
                      rowLabels=rowLabels, 
                      colLabels = colLabels,
                     loc='center')
    table.set_fontsize(14)
    table.scale(1,4)
    ax.axis('off')
