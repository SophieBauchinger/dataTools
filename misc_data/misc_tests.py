# -*- coding: utf-8 -*-
"""
@Author: Sophie Bauchinger, IAU
@Date Thu Aug 10 16:30:17 2023

Test functions on miscellaneous data 
"""
import numpy as np

from toolpac.readwrite import find
from toolpac.readwrite.FFI1001_reader import FFI1001DataReader

#%% Test changes to FFI1001ReadHeader, FFI1001DataReader
fnames = [
    r'C:/Users/sophie_bauchinger/Documents/GitHub/iau-caribic/misc_data/b47_cryosampler_GCMS_ECD_PIC_w_cat.csv',
    r'C:/Users/sophie_bauchinger/Documents/GitHub/iau-caribic/misc_data/ACinst_GUF003_202108122119_RA.ict',
    r'E:\CARIBIC\Caribic2data\Flight148_20060428\GHG_20060428_148_MNL_CAN_V11.txt',
    ]

cryo = FFI1001DataReader(fnames[0], sep_header=';', df=True, flatten_vnames=False)
ac = FFI1001DataReader(fnames[1], sep_header=',', df=True)
ac_flat = FFI1001DataReader(fnames[1], sep_header=',', df=True, flatten_vnames=False)
c = FFI1001DataReader(fnames[2], sep_variables=';', df=True)

print(cryo.df.shape, cryo.df.columns[:4])
print(ac.df.shape, ac.df.columns[:4])
print(ac_flat.df.shape, ac_flat.df.columns[:4])
print(c.df.shape, c.df.columns[:4])
