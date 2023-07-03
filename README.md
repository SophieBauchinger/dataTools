# iau-caribic
Data extraction and analysis for Caribic GHG measurements & other data sources

### data
class GlobalData
- get_data(self, c_pfxs, remap_lon, mozart_file, verbose)
- binned_1d(self, subs, **kwargs)
- binned_2d(self, subs, **kwargs)
- sel_year(self, *yr_list)
- sel_latitude(self, lat_min, lat_max)

class Caribic(GlobalData)
  - self_flight(self, *flights_list)
class Mozart(GlobalData)

class LocalData
  - get_data(self, path)
class Mauna_Loa(LocalData)
class Mace_Head(LocalData)

### plot
#### .data
- caribic_plots(c_obj, key, subs)
- plot_1d_LonLat(mzt_obj, subs='sf6', lon_values=[10, 60, 120, 180], lat_values=[70, 30, 0, -30, -70], single_yr=None)
- plot_global_binned_1d(glob_obj, subs, single_yr=None, plot_mean=False, single_graph=False, c_pfx=None)
- plot_global_binned_2d(glob_obj, subs, single_yr=None, c_pfx='GHG', years=None)
- plot_local(loc_obj, substance=None, greyscale=False, v_limits=(None, None))
- plot_scatter_global(glob_obj, subs, single_yr=None, verbose=False, dataframe=None, c_pfx=None, as_subplot=False, ax=None)
#### .eqlat
- plot_eqlat_deltheta(c_obj, subs='n2o', c_pfx='INT2', tp='therm', pvu=2.0, x_bin=None, y_bin=None, x_source='ERA5', vlims=None, detr=True, note=None)
#### .gradients
- plot_gradient_by_season(c_obj, subs, tp='therm', pvu=2.0, errorbars=False, bsize=None, use_detr=True, note=None)

### dictionaries
- choose_column(df, var='subs')
- coord_dict()
- get_col_name(substance, source, c_pfx='', CLaMS=False)
- get_coord_name(coord, source, c_pfx=None, CLaMS=True)
- get_default_unit(substance)
- get_fct_substance(substance)
- get_vlims(substance)
- substance_list(ID)
- validated_input(prompt, choices)

### tools
- bin_1d(glob_obj, subs, **kwargs)
- bin_2d(glob_obj, subs, **kwargs)
- bin_prep(glob_obj, subs, **kwargs)
- coord_combo(c_obj, save=True)
- daily_mean(df)
- ds_to_gdf(ds)
- get_lin_fit(df, substance='N2OcatsMLOm', degree=2)
- make_season(month)
- monthly_mean(df, first_of_month=True)
- rename_columns(columns)
- subs_merge(c_obj, subs, save=True, detr=True)

### lags
- calc_time_lags(c_obj, ref_obj, years, substance='sf6', pfx='GHG', ref_min=2003, ref_max=2020, plot_yr=False, plot_all=True, save=True, verbose=False)
- plot_time_lags(df, lags, years, ref_min=2003, ref_max=2020, subs='sf6')

### outliers
- filter_strat_trop(glob_obj, ref_obj, crit, pfx='GHG', save=True, verbose=False, plot=True, limit=0.97)
- filter_trop_outliers(glob_obj, subs, pfx, crit=None, ref_obj=None, save=True)
- pre_flag(glob_obj, ref_obj, crit, limit=0.97, pfx=None, verbose=False)

### detrend
- detrend_substance(c_obj, subs, loc_obj, degree=2, plot=True)
