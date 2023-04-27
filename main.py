# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 15:56:59 2023

@author: sophie_bauchinger
"""
import sys

from local_data import Mauna_Loa, Mace_Head
from global_data import Caribic, Mozart


from toolpac.calc import bin_1d_2d
from toolpac.outliers import outliers
from toolpac.outliers import ol_fit_functions as fct
from toolpac.outliers.outliers import get_no_nan, fit_data
from toolpac.age import calculate_lag as cl
from toolpac.conv.times import datetime_to_fractionalyear, fractionalyear_to_datetime

sys.path.insert(0, r'C:\Users\sophie_bauchinger\sophie_bauchinger\Caribic_data_handling')
from C_filter import filter_outliers
import C_SF6_age
import C_tools

