# -*- coding: utf-8 -*-
"""
@Author: Sophie Bauchinger, IAU
@Date: Wed Aug  9 16:58:43 2023

Plot tropopause heights n stuff from different data sources
"""
import geopandas
import numpy as np

import matplotlib.pyplot as plt

# import matplotlib.cm as cm
from tools import monthly_mean
from dictionaries import get_col_name, get_vlims, get_default_unit, substance_list

def tropopause_height_latitude_averaged(glob_obj):
    plt.plot()