# -*- coding: utf-8 -*-
"""
@Author: Sophie Bauchinger, IAU
@Date: Thu May 11 13:22:38 2023

Class definitions for combining data structures and plotting possibilities.

"""
from dataTools.plot.data import GlobalDataPlotterMixin
from dataTools.plot.binsubs import BinPlotterMixin

from dataTools.data._global import GlobalData
from dataTools.data.Caribic import Caribic

class PlotterMixin(GlobalDataPlotterMixin, 
              BinPlotterMixin, 
              GlobalData):
    """ Combination of all available plotting modules for global datasets. """
    def __repr__(self): 
        return f"""Plotter for GlobalData based on {self.base_class}
    years: {self.years}
    status: {self.status}"""
    

class CaribicPlotter(GlobalDataPlotterMixin, 
                     BinPlotterMixin, 
                     Caribic): 
    def __init__(self): 
        super().__init__()
        self.base_class = Caribic
