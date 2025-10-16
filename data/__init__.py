from ._caribic import Caribic
from ._campaigns import CampaignData, CampaignSQLData, DataCollection
from ._model import Era5ModelData, EMAC, Mozart
from ._local import MaunaLoa, MaceHead

from .tropopause import n2o_baseline_filter
from .BinnedData import binning, seasonal_binning, monthly_binning

if not (
    all(isinstance(var, type) for var in 
        [Caribic, CampaignData, CampaignSQLData, 
        Era5ModelData, EMAC, Mozart,
        MaunaLoa, MaceHead, DataCollection]) 
    and all(callable(var) for var in 
        [n2o_baseline_filter, binning, 
        seasonal_binning, monthly_binning])): 
    print("Some of the imported data types do not match the expected value. ")
