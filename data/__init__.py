from ._caribic import Caribic
from ._campaigns import CampaignData, CampaignSQLData
from .local import LocalData, MaunaLoa, MaceHead
from .tropopause import n2o_baseline_filter

from .BinnedData import binning, seasonal_binning, monthly_binning

if not (
    all(isinstance(var, type) for var in 
        [Caribic, CampaignData, CampaignSQLData, 
        LocalData, MaunaLoa, MaceHead]) 
    and all(callable(var) for var in 
        [n2o_baseline_filter, binning, 
        seasonal_binning, monthly_binning])): 
    print(
        'Caribic: ', type(Caribic), '\n',
        'CampaignData: ', type(CampaignData),'\n',
        'CampaignSQLData: ', type(CampaignSQLData),'\n',
        'LocalData: ', type(LocalData),'\n',
        'MaunaLoa: ', type(MaunaLoa),'\n',
        'MaceHead: ', type(MaceHead),'\n',
        'n2o_baseline_filter: ', type(n2o_baseline_filter),'\n',
        'binning: ', type(binning),'\n',
        'seasonal_binning: ', type(seasonal_binning),'\n',
        'monthly_binning: ', type(monthly_binning),'\n',
        )
