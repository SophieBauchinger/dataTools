# -*- coding: utf-8 -*-
"""
@Author: Sophie Bauchinger, IAU
@Date: Mon Feb 13 14:46:39 2023

"""
import geopandas
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable as sm

from toolpac.calc import bin_1d_2d
from tpt_data import caribic_gdf, mozart_data

#%% Binned Object 
class binned_object(object):
    def __init__(self, year, grid_size, v_limits):
        self.year = year
        self.grid_size = grid_size
        self.v_limits = v_limits

    def plot_1d(obj):
        xbmin, xbmax = min(obj.x), max(obj.x)
        ybmin, ybmax = min(obj.y), max(obj.y)
    
        fig, ax = plt.subplots(dpi=300, ncols=2, sharey=True, figsize=(8,3.5))
        fig.suptitle('{} {} modeled SF$_6$ concentration. Gridsize={}'.format(
            obj.source, obj.year, obj.grid_size))
    
        cmap = plt.cm.viridis_r
        if obj.v_limits: vmin, vmax = obj.v_limits
        else:
            vmin = min([np.nanmin(obj.outx.vmean), np.nanmin(obj.outy.vmean)])
            vmax = max([np.nanmin(obj.outx.vmean), np.nanmin(obj.outy.vmean)])
        norm = Normalize(vmin, vmax)
    
        ax[0].plot(obj.outx.xintm, obj.outx.vmean, zorder=1, color='black', lw = 0.5)
        ax[0].scatter(obj.outx.xintm, obj.outx.vmean, 
                      c = obj.outx.vmean, cmap = cmap, norm = norm, zorder=2)
        ax[0].set_xlabel('Latitude [deg]'); plt.xlim(xbmin, xbmax)
        ax[0].set_ylabel('Mean SF$_6$ mixing ratio [ppt]')
    
        ax[1].plot(obj.outy.xintm, obj.outy.vmean, zorder=1, color='black', lw = 0.5)
        ax[1].scatter(obj.outy.xintm, obj.outy.vmean, 
                      c = obj.outy.vmean, cmap = cmap, norm = norm, zorder=2)
        ax[1].set_xlabel('Longitude [deg]'); plt.xlim(ybmin, ybmax)
        ax[1].set_ylabel('Mean SF$_6$ mixing ratio [ppt]')
    
        #ax[0].set_ylim(6,7)
    
        fig.colorbar(sm(norm=norm, cmap=cmap), aspect=50, ax = ax[1])
    
        plt.show()
        
    def plot_2d(obj):
        
        xbmin, xbmax, xbsize = min(obj.x), max(obj.x), obj.grid_size
        ybmin, ybmax, ybsize = min(obj.y), max(obj.y), obj.grid_size
    
        out = bin_1d_2d.bin_2d(np.array(obj.gdf[obj.substance]), obj.x, obj.y,
                               xbmin, xbmax, xbsize, ybmin, ybmax, ybsize)
    
        world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
    
        fig, ax = plt.subplots(figsize=(10,10), dpi=300)
        ax.set_title('{} {} SF$_6$ concentration measurements. Gridsize={}'.format(
            obj.source, obj.year, obj.grid_size))
        ax.set_aspect('equal')
        if obj.v_limits: vmin, vmax = obj.v_limits
        else: vmin = np.nanmin(out.vmin); vmax = np.nanmax(out.vmax)
        world.boundary.plot(ax=ax, color='black', linewidth=0.3)
        cmap = ax.imshow(out.vmean, cmap = 'viridis_r', interpolation='nearest',
                         origin='lower', extent=[ybmin, ybmax, xbmin, xbmax],
                         vmin=vmin, vmax = vmax)
        cbar = fig.colorbar(cmap, ax=ax, aspect=50, pad=0.08, orientation='horizontal')
        cbar.ax.set_xlabel('Mean $SF_6$ [ppt]')
        plt.xlabel('Longitude  [deg]'); plt.xlim(-180,180)
        plt.ylabel('Latitude [deg]'); plt.ylim(-60,100)
        plt.show()
    
        return out
    
    def out_x(obj):
        xbmin, xbmax, xbsize = float(min(obj.x)), float(max(obj.x)), obj.grid_size
        out_x = bin_1d_2d.bin_1d(obj.SF6, obj.x, xbmin, xbmax, xbsize)
        return out_x
    
    def out_y(obj):
        ybmin, ybmax, ybsize = float(min(obj.y)), float(max(obj.y)), obj.grid_size
        out_y = bin_1d_2d.bin_1d(obj.SF6, obj.y, ybmin, ybmax, ybsize)
        return out_y

class caribic_binned(binned_object):
    """ docstring """
    def __init__(self, year=2007, grid_size=5, v_limits=None, flight_nr = None, 
               substance='SF6; SF6 mixing ratio; [ppt]\n'):
        binned_object.__init__(self, year, grid_size, v_limits)
        self.gdf = caribic_gdf(year) # pandas GeoDataFrame
        if flight_nr:
            self.gdf = self.gdf[self.gdf.values == flight_nr]

        self.x = np.array([self.gdf.geometry[i].x for i in range(0, len(self.gdf.index))]) # lat
        self.y = np.array([self.gdf.geometry[i].y for i in range(0, len(self.gdf.index))]) # lon
        self.substance = substance
        
        self.SF6 = np.array(self.gdf[self.substance])

        #self.SF6_x = np.array(self.gdf[self.substance])
        #self.SF6_y = np.array(self.gdf[self.substance])

        self.outx = self.out_x()
        self.outy = self.out_y()
        
        self.source = 'Caribic'

class mozart_binned(binned_object):
    """ docstring """
    def __init__(self, year=2007, grid_size=5, v_limits=None, level=0, substance='SF6'):
        binned_object.__init__(self, year, grid_size, v_limits)
        self.ds, self.gdf = mozart_data(year=year, level=level)
        self.x = np.array([self.gdf.geometry[i].x for i in range(0, len(self.gdf.index))]) # lat
        self.y = np.array([self.gdf.geometry[i].y for i in range(0, len(self.gdf.index))]) # lon
        self.y = np.array([i for i in self.y if i<=178] + 
                          [i - 364 for i in self.y if i>178])
                          
        self.substance = substance

        self.SF6 = self.gdf['SF6']

        #self.SF6_x = self.ds.SF6.mean(dim='longitude')
        #self.SF6_y = self.ds.SF6.mean(dim='latitude')

        self.outx = self.out_x()
        self.outy = self.out_y()
        
        self.source = 'Mozart'

def plot_mozart_caribic(year, grid_size, mozart_level):
    m_obj = mozart_binned(year, grid_size, level=mozart_level)
    c_obj = caribic_binned(year, grid_size)

    xbmin, xbmax = min(min(m_obj.x), min(c_obj.x)), max(max(m_obj.x), max(c_obj.x))
    ybmin, ybmax = min(min(m_obj.y), min(c_obj.y)), max(max(m_obj.y), max(c_obj.y))

    fig, ax = plt.subplots(dpi=300, ncols=2, sharey=True, figsize=(8,3.5))
    fig.suptitle('{} {} modeled SF$_6$ concentration. Gridsize={}'.format(
        m_obj.source, m_obj.year, m_obj.grid_size))

    for i, obj in enumerate([c_obj, m_obj]):
        if i ==0: label = 'Caribic'
        elif i == 1: label = 'Mozart'
        ax[0].plot(obj.outx.xintm, obj.outx.vmean, zorder=1, color='black', lw = 0.5)
        ax[0].scatter(obj.outx.xintm, obj.outx.vmean, label=label)#, 
                      #c = obj.outx.vmean, cmap = cmap, norm = norm, zorder=2)
        ax[0].set_xlabel('Latitude [deg]'); plt.xlim(xbmin, xbmax)
        ax[0].set_ylabel('Mean SF$_6$ mixing ratio [ppt]')
    
        ax[1].plot(obj.outy.xintm, obj.outy.vmean, zorder=1, color='black', lw = 0.5)
        ax[1].scatter(obj.outy.xintm, obj.outy.vmean, label=label)#, 
                      #c = obj.outy.vmean, cmap = cmap, norm = norm, zorder=2)
        ax[1].set_xlabel('Longitude [deg]'); plt.xlim(ybmin, ybmax)
        ax[1].set_ylabel('Mean SF$_6$ mixing ratio [ppt]')
        ax[0].legend()
    plt.show()

if __name__=='__main__':
    v_lims = (6,7)
    mozart_level = 27
    yr = 2007
    grid_size = 10
    
    mozart = mozart_binned(yr, grid_size, level=mozart_level, v_limits=v_lims)
    caribic = caribic_binned(yr, grid_size, v_limits=v_lims)

    mozart.plot_1d()
    caribic.plot_1d()

    mozart.plot_2d()
    caribic.plot_2d()

    plot_mozart_caribic(2008, grid_size, mozart_level=mozart_level)

#%% Caribic 1D and 2D binning & plotting

# =============================================================================
# class caribic_binned(object):
#     """ docstring """
#     def __init__(self, year=2008, grid_size=5, v_limits=None, flight_nr = None, 
#                substance='SF6; SF6 mixing ratio; [ppt]\n'):
#         self.year = year
#         self.grid_size = grid_size
#         self.v_limits = v_limits
# 
#         self.gdf = caribic_gdf(year) # pandas GeoDataFrame
#         if flight_nr:
#             self.gdf = self.gdf[self.gdf.values == flight_nr]
# 
#         self.x = np.array([self.gdf.geometry[i].x for i in range(0, len(self.gdf.index))]) # lat
#         self.y = np.array([self.gdf.geometry[i].y for i in range(0, len(self.gdf.index))]) # lon
#         self.substance = substance
# 
#         self.outx = out_x(self)
#         self.outy = out_y(self)
# 
#         self.SF6_x = np.array(self.gdf[self.substance])
#         self.SF6_y = np.array(self.gdf[self.substance])
# 
#     
#     def plot_1d(self):
#         # out_x, out_y = self.out_x, self.out_y
#         xbmin, xbmax = min(self.x), max(self.x)
#         ybmin, ybmax = min(self.y), max(self.y)
#         
#         fig, ax = plt.subplots(dpi=300, ncols=2, sharey=True)
#         fig.suptitle('CARIBIC {} {} concentration measurements. Gridsize={}'.format(
#             self.year, self.substance[5:-8], self.grid_size))
#     
#         cmap = plt.cm.viridis_r
#         if self.v_limits: vmin =self.v_limits[0]; vmax = self.v_limits[1]
#         else:
#             vmin = min([np.nanmin(self.outx.vmean), np.nanmin(self.outy.vmean)])
#             vmax = max([np.nanmin(self.outx.vmean), np.nanmin(self.outy.vmean)])
#         norm = Normalize(vmin, vmax)
#     
#         ax[0].plot(self.outx.xintm, self.outx.vmean, zorder=1, color='black', lw = 0.5)
#         ax[0].scatter(self.outx.xintm, self.outx.vmean, 
#                       c = self.outx.vmean, cmap = cmap, norm = norm, zorder=2)
#         ax[0].set_xlabel('Latitude [deg]'); plt.xlim(xbmin, xbmax)
#         ax[0].set_ylabel('Mean SF$_6$ mixing ratio [ppt]')
#     
#         ax[1].plot(self.outy.xintm, self.outy.vmean, zorder=1, color='black', lw = 0.5)
#         ax[1].scatter(self.outy.xintm, self.outy.vmean, 
#                       c = self.outy.vmean, cmap = cmap, norm = norm, zorder=2)
#         ax[1].set_xlabel('Longitude [deg]'); plt.xlim(ybmin, ybmax)
#         ax[1].set_ylabel('Mean SF$_6$ mixing ratio [ppt]')
#     
#         ax[0].set_ylim(6,7); ax[1].set_ylim(6,8)
#     
#         fig.colorbar(sm(norm=norm, cmap=cmap), aspect=50, ax = ax[1])
#     
#         plt.show()
# 
#     def plot_2d(self):
#         xbmin, xbmax, xbsize = min(self.x), max(self.x), self.grid_size
#         ybmin, ybmax, ybsize = min(self.y), max(self.y), self.grid_size
# 
#         out = bin_1d_2d.bin_2d(np.array(self.gdf[self.substance]), self.x, self.y,
#                                xbmin, xbmax, xbsize, ybmin, ybmax, ybsize)
# 
#         world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
# 
#         fig, ax = plt.subplots(figsize=(10,10), dpi=300)
#         ax.set_title('CARIBIC {} {} concentration measurements. Gridsize={}'.format(
#             self.year, self.substance[5:-8], self.grid_size))
#         ax.set_aspect('equal')
#         if v_limits: vmin = v_limits[0]; vmax = v_limits[1]
#         else: vmin = np.nanmin(out.vmin); vmax = np.nanmax(out.vmax)
#         world.boundary.plot(ax=ax, color='black', linewidth=0.3)
#         cmap = ax.imshow(out.vmean, cmap = 'viridis_r', interpolation='nearest',
#                          origin='lower', extent=[ybmin, ybmax, xbmin, xbmax],
#                          vmin=vmin, vmax = vmax)
#         cbar = fig.colorbar(cmap, ax=ax, aspect=50, pad=0.08, orientation='horizontal')
#         cbar.ax.set_xlabel('Mean {} {}'.format(self.substance[5:-8], self.substance[-6:-1]))
#         plt.xlabel('Longitude  [deg]'); plt.xlim(-180,180)
#         plt.ylabel('Latitude [deg]'); plt.ylim(-60,100)
#         plt.show()
# 
#         return out
#         
# 
# 
# if __name__=='__main__':
#     # c1d_2008 = caribic_1d_bin_plot(2008, 5, v_limits = (6,8))
#     # c1d_2012 = caribic_1d_bin_plot(2012, 5, v_limits = (6,8))
# 
#     caribic_binned(2007, 5, v_limits = (6,8)).plot_1d()
#     caribic_binned(2012, 5, v_limits = (6,8)).plot_1d()
#     caribic_binned(2007, 5, v_limits = (6,8)).plot_2d()
#     caribic_binned(2012, 5, v_limits = (6,8)).plot_2d()
# =============================================================================


#%% Mozart 1D and 2D binning & plotting

# =============================================================================
# class mozart_binned(object):
#     """ docstring """
#     def __init__(self, year=2007, grid_size=5, v_limits=None, level=0, substance='SF6'):
#         self.year = year
#         self.grid_size = grid_size
#         self.v_limits = v_limits
# 
#         self.ds, self.gdf = mozart_data(year=year, level=level)
#         self.x = np.array([self.gdf.geometry[i].x for i in range(0, len(self.gdf.index))]) # lat
#         self.y = np.array([self.gdf.geometry[i].y for i in range(0, len(self.gdf.index))]) # lon
#         self.substance = substance
# 
#         self.outx = self.out_x()
#         self.outy = self.out_y()
#     
#         self.SF6_x = self.ds.SF6.mean(dim='longitude')
#         self.SF6_y = self.ds.SF6.mean(dim='latitude')
# 
# 
# def mozart_1d_bin_plot(year, grid_size=5, v_limits=None, level=0):
#     ds, df = mozart_data(year=year, level=level) # ds, df
#     x,y = ds.latitude, ds.longitude # len = 36, 72
# 
#     # average over all heights: .mean(dim='level')
#     SF6_x = ds.SF6.mean(dim='longitude')
#     SF6_y = ds.SF6.mean(dim='latitude')
# 
#     xbmin, xbmax, xbsize = float(min(x)), float(max(x)), grid_size
#     out_x = bin_1d_2d.bin_1d(SF6_x, x, xbmin, xbmax, xbsize)
# 
#     ybmin, ybmax, ybsize = float(min(y)), float(max(y)), grid_size
#     out_y = bin_1d_2d.bin_1d(SF6_y, y, ybmin, ybmax, ybsize)
# 
#     fig, ax = plt.subplots(dpi=300, ncols=2, sharey=True)
#     fig.suptitle('MOZART {} modeled SF$_6$ concentration. Gridsize={}'.format(
#         year, grid_size))
# 
#     cmap = plt.cm.viridis_r
#     if v_limits: vmin = v_limits[0]; vmax = v_limits[1]
#     else:
#         vmin = min([np.nanmin(out_x.vmean), np.nanmin(out_y.vmean)])
#         vmax = max([np.nanmin(out_x.vmean), np.nanmin(out_y.vmean)])
#     norm = Normalize(vmin, vmax)
# 
#     ax[0].plot(out_x.xintm, out_x.vmean, zorder=1, color='black', lw = 0.5)
#     ax[0].scatter(out_x.xintm, out_x.vmean, 
#                   c = out_x.vmean, cmap = cmap, norm = norm, zorder=2)
#     ax[0].set_xlabel('Latitude [deg]'); plt.xlim(xbmin, xbmax)
#     ax[0].set_ylabel('Mean SF$_6$ mixing ratio [ppt]')
# 
#     ax[1].plot(out_y.xintm, out_y.vmean, zorder=1, color='black', lw = 0.5)
#     ax[1].scatter(out_y.xintm, out_y.vmean, 
#                   c = out_y.vmean, cmap = cmap, norm = norm, zorder=2)
#     ax[1].set_xlabel('Longitude [deg]'); plt.xlim(ybmin, ybmax)
#     ax[1].set_ylabel('Mean SF$_6$ mixing ratio [ppt]')
# 
#     #ax[0].set_ylim(6,7)
# 
#     fig.colorbar(sm(norm=norm, cmap=cmap), aspect=50, ax = ax[1])
# 
#     plt.show()
# 
#     return out_x
# 
# if __name__=='__main__':
#     m1d_2008 = mozart_1d_bin_plot(2000, 10, level=0)
# =============================================================================

#%% 2D bin & plotting
# =============================================================================
# def caribic_2d_bin_plot(year, grid_size=5, v_limits=None, 
#                         substance='SF6; SF6 mixing ratio; [ppt]\n', 
#                         flight_nr=None):
#     """
#     Creates a map of the average mixing ratio of a substance on a grid of
#     specified grid resolution.
# 
#     path (str): location of the parent directory containing data files
#     year (int): measurement year
#     grid_size (int): number of lat / lon degrees to average over. Default=5
#     substance (str): name[unit] of measurement series from flight data. Default='SF6[ppt]'
#     flight_nr (int): number of the CARIBIC flight. Default=None
#     """
#     data = caribic_gdf(year) # pandas GeoDataFrame
#     if flight_nr:
#         data = data[data.values == flight_nr]
#     lon = np.array([data.geometry[i].x for i in range(0, len(data.index))])
#     lat = np.array([data.geometry[i].y for i in range(0, len(data.index))])
# 
#     xbmin, xbmax, xbsize = min(lon), max(lon), grid_size
#     ybmin, ybmax, ybsize = min(lat), max(lat), grid_size
# 
#     out = bin_1d_2d.bin_2d(np.array(data[substance]), lon, lat,
#                            xbmin, xbmax, xbsize, ybmin, ybmax, ybsize)
# 
#     world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
# 
#     fig, ax = plt.subplots(figsize=(10,10), dpi=300)
#     ax.set_title('CARIBIC {} {} concentration measurements. Gridsize={}'.format(
#         year, substance[5:-8], grid_size))
#     ax.set_aspect('equal')
#     if v_limits: vmin = v_limits[0]; vmax = v_limits[1]
#     else: vmin = np.nanmin(out.vmin); vmax = np.nanmax(out.vmax)
#     world.boundary.plot(ax=ax, color='black', linewidth=0.3)
#     cmap = ax.imshow(out.vmean, cmap = 'viridis_r', interpolation='nearest',
#                      origin='lower', extent=[ybmin, ybmax, xbmin, xbmax],
#                      vmin=vmin, vmax = vmax)
#     cbar = fig.colorbar(cmap, ax=ax, aspect=50, pad=0.08, orientation='horizontal')
#     cbar.ax.set_xlabel('Mean {} {}'.format(substance[5:-8], substance[-6:-1]))
#     plt.xlabel('Longitude  [deg]'); plt.xlim(-180,180)
#     plt.ylabel('Latitude [deg]'); plt.ylim(-60,100)
#     plt.show()
# 
#     return out
# 
# if __name__=='__main__':
#     c2d_2008 = caribic_2d_bin_plot(2008, 5)
#     c2d_2012 = caribic_2d_bin_plot(2012, 5)
#     c2d_2008_380 = caribic_2d_bin_plot(2012, 5, flight_nr=390)
# =============================================================================

#%%
# =============================================================================
# def mozart_2d_bin_plot(year, grid_size=5, v_limits=None, level = 0):
#     """
#     Creates a map of the average mixing ratio of a substance on a grid of
#     specified grid resolution.
# 
#     path (str): location of the parent directory containing data files
#     year (int): measurement year
#     grid_size (int): number of lat / lon degrees to average over. Default=5
#     substance (str): name[unit] of measurement series from flight data. Default='SF6[ppt]'
#     flight_nr (int): number of the CARIBIC flight. Default=None
#     """
# 
#     data = caribic_gdf(year) # pandas GeoDataFrame
#     if flight_nr:
#         data = data[data.values == flight_nr]
#     lon = np.array([data.geometry[i].x for i in range(0, len(data.index))])
#     lat = np.array([data.geometry[i].y for i in range(0, len(data.index))])
# 
#     xbmin, xbmax, xbsize = min(lon), max(lon), grid_size
#     ybmin, ybmax, ybsize = min(lat), max(lat), grid_size
# 
#     out = bin_1d_2d.bin_2d(np.array(data[substance]), lon, lat,
#                            xbmin, xbmax, xbsize, ybmin, ybmax, ybsize)
#     
#     ds, gdf = mozart_data(year=year, level=level) # ds, gdf
#     #x,y = np.array(ds.latitude), np.array(ds.longitude) # len = 36, 72
#     
#     x = np.array([gdf.geometry[i].x for i in range(0, len(gdf.index))])
#     y = np.array([gdf.geometry[i].y for i in range(0, len(gdf.index))])
# 
#     # average over all heights: .mean(dim='level')
# # =============================================================================
# #     SF6_x = ds.SF6.mean(dim='longitude')
# #     SF6_y = ds.SF6.mean(dim='latitude')
# # =============================================================================
# 
#     xbmin, xbmax, xbsize = float(min(x)), float(max(x)), grid_size
#     ybmin, ybmax, ybsize = float(min(y)), float(max(y)), grid_size
# 
#     out = bin_1d_2d.bin_2d(gdf.SF6, x, y, 
#                            xbmin, xbmax, xbsize, ybmin, ybmax, ybsize)
# 
#     world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
# 
#     fig, ax = plt.subplots(figsize=(10,10), dpi=300)
#     ax.set_title('MOZART {} simulated SF$_6$ concentration. Gridsize={}'.format(
#         year, grid_size))
#     ax.set_aspect('equal')
#     if v_limits: vmin = v_limits[0]; vmax = v_limits[1]
#     else: vmin = np.nanmin(out.vmin); vmax = np.nanmax(out.vmax)
#     world.boundary.plot(ax=ax, color='black', linewidth=0.3)
#     cmap = ax.imshow(out.vmean, cmap = 'viridis_r', interpolation='nearest',
#                      origin='lower', extent=[ybmin, ybmax, xbmin, xbmax],
#                      vmin=vmin, vmax = vmax)
#     fig.colorbar(cmap, ax=ax, aspect=50, pad=0.08, orientation='horizontal')
#     plt.xlabel('Longitude  [deg]')#; plt.xlim(-180,180)
#     plt.ylabel('Latitude [deg]')#; plt.ylim(-60,100)
#     plt.show()
# 
#     return out
# 
# if __name__=='__main__':
#     m2d_2008 = mozart_2d_bin_plot(2008, 5)
# =============================================================================

#%% CARIBIC on MOZART background

# =============================================================================
# v_limits = (6,7)
# c2d_2008 = caribic_2d_bin_plot(2008, 5, v_limits)
# m2d_2008 = mozart_2d_bin_plot(2008, 5, v_limits)
# 
# =============================================================================
# TODO: Caribic and Mozart on one graph - correlation ? 
# TODO: N20 Werte als Filterkriterium, ...

#%% Archive
# def caribic_1d_bin_plot(year, grid_size=5, v_limits = None,
#                         substance='SF6; SF6 mixing ratio; [ppt]\n'):
#     """
#     Creates a plot of the average mixing ratio over bins on a latitudinal and
#     longitudianal 1d representation

#     path (str): location of the parent directory containing data files
#     year (int): measurement year
#     grid_size (int): number of lat / lon degrees to average over
#     substance (str): name[unit] of measurement series from flight data
#     """
#     data = caribic_gdf(year) # pandas GeoDataFrame
#     x = np.array([data.geometry[i].x for i in range(0, len(data.index))]) # lat
#     y = np.array([data.geometry[i].y for i in range(0, len(data.index))]) # lon

#     xbmin, xbmax, xbsize = min(x), max(x), grid_size
#     out_x = bin_1d_2d.bin_1d(np.array(data[substance]), x, xbmin, xbmax, xbsize)

#     ybmin, ybmax, ybsize = min(y), max(y), grid_size
#     out_y = bin_1d_2d.bin_1d(np.array(data[substance]), y, ybmin, ybmax, ybsize)

#     fig, ax = plt.subplots(dpi=300, ncols=2, sharey=True)
#     fig.suptitle('CARIBIC {} {} concentration measurements. Gridsize={}'.format(
#         year, substance[5:-8], grid_size))

#     cmap = plt.cm.viridis_r
#     if v_limits: vmin = v_limits[0]; vmax = v_limits[1]
#     else:
#         vmin = min([np.nanmin(out_x.vmean), np.nanmin(out_y.vmean)])
#         vmax = max([np.nanmin(out_x.vmean), np.nanmin(out_y.vmean)])
#     norm = Normalize(vmin, vmax)

#     ax[0].plot(out_x.xintm, out_x.vmean, zorder=1, color='black', lw = 0.5)
#     ax[0].scatter(out_x.xintm, out_x.vmean, 
#                   c = out_x.vmean, cmap = cmap, norm = norm, zorder=2)
#     ax[0].set_xlabel('Latitude [deg]'); plt.xlim(xbmin, xbmax)
#     ax[0].set_ylabel('Mean SF$_6$ mixing ratio [ppt]')

#     ax[1].plot(out_y.xintm, out_y.vmean, zorder=1, color='black', lw = 0.5)
#     ax[1].scatter(out_y.xintm, out_y.vmean, 
#                   c = out_y.vmean, cmap = cmap, norm = norm, zorder=2)
#     ax[1].set_xlabel('Longitude [deg]'); plt.xlim(ybmin, ybmax)
#     ax[1].set_ylabel('Mean SF$_6$ mixing ratio [ppt]')

#     ax[0].set_ylim(6,7); ax[1].set_ylim(6,8)

#     fig.colorbar(sm(norm=norm, cmap=cmap), aspect=50, ax = ax[1])

#     plt.show()

#     return out_x, out_y