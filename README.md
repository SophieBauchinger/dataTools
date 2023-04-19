# iau-caribic
All tools etc that I need to analyse Caribic data wrt ground-based measurement stations and model data 
## toolpac_tutorial
General data structure is based on Pandas GeoDataFrame
- Index is datetime format
- Global datasets have a geometry column (comprised of Points) rather than lon / lat data

Namespace of toolpac tutorial: 
- class Caribic
  - __init__(self, years, grid_size=5, v_limits=None, flight_nr = None, subst='sf6', pfxs=['GHG']):
  - caribic_data(self, pfxs):
  - get_col_name(self, substance):
  - plot_scatter(self):
  - try_plot_2d(self):
- class Mozart
  - __init__(self, years, grid_size=5, v_limits=None):
  - mozart_data(self, year, level = 27, remap = True, file = r'C:\Users\sophie_bauchinger\sophie_bauchinger\toolpac_tutorial\RIGBY_2010_SF6_MOLE_FRACTION_1970_2008.nc'):
  - plot_scatter(self, total=False):
- class Mauna_Loa
  - __init__(self, years, path = None, path_MM = None, substance='sf6')
  - mlo_data(self, yr, path):
  - plot_MM(self):
- class Mace_Head
  - __init__(self, path = None):
  - mhd_data(self, path):
  - plot_mhd(self):
  - plot_1d_LonLat(self, lon_values = [10, 60, 120, 180], lat_values = [70, 30, 0, -30, -70]):

- fct ds_to_gdf(ds)

- fct monthly_mean(df, first_of_month=True)

## strat_filter_on_caribic_data
- fct cal_time_lags(c_data, ref_data, ref_subs = 'SF6catsMLOm'):
- fct plot_time_lags(c_data, lags, ref_lims):
- fct get_mlo_fit(mlo_df, substance='N2OcatsMLOm'):
- fct pre_flag(data, n2o_col, t_obs_tot, mlo_fit):
- fct filter_strat_trop(data, ref_data, crit, mlo_fit):
