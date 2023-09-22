# # -*- coding: utf-8 -*-
# """
# @Author: Sophie Bauchinger, IAU
# @Date: Fri Apr 28 09:58:15 2023

# Remove linear trend of substances using free troposphere as reference.

# """
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import datetime as dt

# from toolpac.conv.times import datetime_to_fractionalyear

# from dictionaries import get_col_name, substance_list
# from groundbased import Mauna_Loa

# def detrend_substance(c_obj, subs, loc_obj=None, degree=2, save=True, plot=False,
#                       as_subplot=False, ax=None, c_pfx=None, note=''):
#     """ Remove linear trend of substances using free troposphere as reference.
#     (redefined from C_tools.detrend_subs)

#     Parameters:
#         c_obj (GlobalData/Caribic)
#         subs (str): substance to detrend e.g. 'sf6'
#         loc_obj (LocalData): free troposphere data, defaults to Mauna_Loa
#     """
#     if loc_obj is None: 
#         try: loc_obj = Mauna_Loa(c_obj.years, subs)
#         except: raise ValueError(f'Cannot detrend as ref. data could not be found for {subs.upper()}') 
#     out_dict = {}

#     if c_pfx: pfxs = [c_pfx]
#     else: pfxs = [pfx for pfx in c_obj.pfxs if subs in substance_list(pfx)]

#     if plot:
#         if not as_subplot:
#             fig, axs = plt.subplots(len(pfxs), dpi=250, figsize=(6,4*len(pfxs)))
#             if len(pfxs)==1: axs = [axs]
#         elif ax is None:
#             ax = plt.gca()

#     for c_pfx, i in zip(pfxs, range(len(pfxs))):
#         df = c_obj.data[c_pfx]
#         substance = get_col_name(subs, c_obj.source, c_pfx)
#         if substance is None: continue

#         c_obs = df[substance].values
#         t_obs =  np.array(datetime_to_fractionalyear(df.index, method='exact'))

#         ref_df = loc_obj.df
#         ref_subs = get_col_name(subs, loc_obj.source)
#         if ref_subs is None: raise ValueError(f'No reference data found for {subs}')
#         # ignore reference data earlier and later than two years before/after msmts
#         ref_df = ref_df[min(df.index)-dt.timedelta(356*2)
#                         : max(df.index)+dt.timedelta(356*2)]
#         ref_df.dropna(how='any', subset=ref_subs, inplace=True) # remove NaN rows
#         c_ref = ref_df[ref_subs].values
#         t_ref = np.array(datetime_to_fractionalyear(ref_df.index, method='exact'))

#         popt = np.polyfit(t_ref, c_ref, degree)
#         c_fit = np.poly1d(popt) # get popt, then make into fct

#         detrend_correction = c_fit(t_obs) - c_fit(min(t_obs))
#         c_obs_detr = c_obs - detrend_correction
#         # get variance (?) by substracting offset from 0
#         c_obs_delta = c_obs_detr - c_fit(min(t_obs))

#         df_detr = pd.DataFrame({f'detr_{substance}' : c_obs_detr,
#                                  f'delta_{substance}' : c_obs_delta,
#                                  f'detrFit_{substance}' : c_fit(t_obs)}, 
#                                 index = df.index)
#         # maintain relationship between detr and fit columns
#         df_detr[f'detrFit_{substance}'] = df_detr[f'detrFit_{substance}'].where(
#             ~df_detr[f'detr_{substance}'].isnull(), np.nan)

#         out_dict[f'detr_{c_pfx}_{subs}'] = df_detr
#         out_dict[f'popt_{c_pfx}_{subs}'] = popt

#         if save:
#             columns = [f'detr_{substance}', 
#                        f'delta_{substance}', 
#                        f'detrFit_{substance}']
            
#             c_obj.data[c_pfx][columns] = df_detr[columns]
#             # move geometry column to the end again
#             c_obj.data[c_pfx]['geometry'] =  c_obj.data[c_pfx].pop('geometry')

#         if plot:
#             if not as_subplot: ax = axs[i]
#             # ax.annotate(f'{c_pfx} {note}', xy=(0.025, 0.925), xycoords='axes fraction',
#             #                       bbox=dict(boxstyle="round", fc="w"))
#             ax.scatter(df_detr.index, c_obs, color='orange', label='Flight data', marker='.')
#             ax.scatter(df_detr.index, c_obs_detr, color='green', label='trend removed', marker='.')
#             ax.scatter(ref_df.index, c_ref, color='gray', label='MLO data', alpha=0.4, marker='.')
#             ax.plot(df_detr.index, c_fit(t_obs), color='black', ls='dashed',
#                       label='trendline')
#             ax.set_ylabel(f'{substance}') # ; ax.set_xlabel('Time')
#             handles, labels = ax.get_legend_handles_labels()
#             leg = ax.legend(title=f'{c_pfx} {note}')
#             leg._legend_box.align = "left"

#     if plot and not as_subplot:
#         fig.tight_layout()
#         fig.autofmt_xdate()
#         plt.show()

#     return out_dict

# #%% Fctn calls - detrend
# # if __name__=='__main__':
# #     from data import Caribic, Mauna_Loa
# #     from dictionaries import substance_list
# #     year_range = (2000, 2020)

# #     calc_caribic = False
# #     if calc_caribic:
# #         caribic = Caribic(year_range, pfxs = ['GHG', 'INT', 'INT2'])

# #     calc_mlo = False
# #     if calc_mlo:
# #         year_range = range(1980, 2021)
# #         mlo_data = {subs : Mauna_Loa(year_range, substance=subs) for subs
# #                     in substance_list('MLO')}

# #     for c_pfx in caribic.pfxs:
# #         substs = [x for x in substance_list(c_pfx)
# #                   if x in ['ch4', 'co2', 'n2o', 'sf6', 'co']]
# #         f, axs = plt.subplots(int(len(substs)/2), 2,
# #                               figsize=(10,len(substs)*1.5), dpi=200)
# #         plt.suptitle(f'Caribic {c_pfx} detrended wrt Mauna Loa')
# #         for subs, ax in zip(substs, axs.flatten()):
# #             output = detrend_substance(caribic, subs, mlo_data[subs], save=True,
# #                               as_subplot=True, ax=ax, c_pfx=c_pfx)
# #         f.autofmt_xdate()
# #         plt.tight_layout()
# #         plt.show()


# #%% reset detr datasets 
# # if __name__=='__main__':
# #     del caribic.data['detr_GHG']
# #     del caribic.data['detr_INT']
# #     del caribic.data['detr_INT2']
