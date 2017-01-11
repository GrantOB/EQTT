# -*- coding: utf-8 -*-
"""
Created on Sun Sep  4 09:57:49 2016

@author: grantobrien

plan:
for each eq location eq(xyz), above the plate interface p(xyz),
calculate the travel time EQ_owt(xyz) to every point in the velocity model v(xyz)
using the eq source point (zero contour) --> the time to reciever (geonet station)
is first arrival.
Then to get twt...
for each reciever location r(xyz) calculate the travel time R_owt(xyz) to every
point in the velocity model v(xyz) using the reciever location as source point
(zero contour), then add: EQ_owt(xyz) + R_owt(xyz) = EQ_R_TWT 
which should be twt from eq(xyz) to r(xyz)
then for each point on the plate interface p(xyz) find the twt by slicing the 
twt 3d grid (EQ_R_TWT) by the p(xyz) surface (as done before)
 ---> may need to reduce vel model size as this needs to be done for each 
reciever (but only once?)

TODO: crop vel model by land dem?
only uses selected stations in section plots

NOTE: to make eq location match zero contour use and even X,Y grid
if the vel model x, y are used there is a missmatch because of the 
shape of the globe etc. even thought itis correct the eq location is not in 
same projection.
"""

import numpy as np
import pandas as PD
import skfmm
from scipy.interpolate import griddata
from scipy.spatial import KDTree, cKDTree
import gdal
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
import shapefile
#from mpl_toolkits.basemap import pyproj
import tqdm
import time
from obspy.geodetics import gps2dist_azimuth


import sys, os
mylibpath = os.path.abspath(u'/Users/marine/OBrienG/')
if not mylibpath in sys.path:
    sys.path.append(mylibpath)
else:
    #print "re-appending", mylibpath
    sys.path.remove(mylibpath)
    sys.path.append(mylibpath)

from GOBEQ import earthquake_tools as ET



class reflections(object):
    """
    calculate twt for earthquakes and reflections off subduction interface.
    """
    def __init__(self, model, stations):
        self.model = model
        self.stations = stations
        #self.load_model()
        #self.calculate_travel_time()
        
    def load_model(self, dim=None):
        """
        model = 'models/nzvel_even_grid_cropped_NZ_100j_ds1.npy' # fast
        model = 'models/nzvel_even_grid_cropped_NZ_500j_ds10.npy' # slow
        dim = 100j
        """
        self.dim = dim
        self.x, self.y, self.z, self.v, self.mask = np.load(self.model)
        #self.z = self.z + 15000.
        #if dim is None:
        #    self.dim = self.x.shape
        self.xscaler = np.nanmax(self.x)
        self.yscaler = np.nanmax(self.y)
        self.zscaler = np.nanmax(self.z)*1000.
        self.xmin = np.nanmin(self.x)
        self.xmax = np.nanmax(self.x)
        self.ymin = np.nanmin(self.y)
        self.ymax = np.nanmax(self.y)
        self.zmin = np.nanmin(self.z)*1000.
        self.zmax = np.nanmax(self.z)*1000.
        self.x[self.mask != 0.0] = np.nan
        self.y[self.mask != 0.0] = np.nan
        self.z[self.mask != 0.0] = np.nan
        self.v = self.v*1000.
        self.v[self.mask != 0.0] = np.nan


    def add_reciever_locations(self, network='NZ'):
        """
        stations = "../GIS_data/GEONET.csv"
        """
        self.stations = PD.read_csv(self.stations, low_memory=False, skiprows=1)
        if network is not 'ALL':
            self.stations = self.stations.groupby(self.stations.Network)
            self.stations = self.stations.get_group(network)
        
        self.stations_xy = np.vstack([self.stations['Longitude'].values,
                                   self.stations['Latitude'].values]).T
        self.stations_xy = ET.CT_nztm_to_wgs84(self.stations_xy, islist=True,
                                         inverse=True)

   
    def create_phi(self, x, y, z):
        """
        need input so one can specify if receiver or earthquake
        """
        self.xloc = x
        self.yloc = y
        self.zloc = z
        #print self.xloc, self.yloc, self.zloc
        #--- make phi ---#
        X, Y, Z = np.mgrid[self.xmin:self.xmax:self.dim,
                           self.ymin:self.ymax:self.dim,
                           self.zmin:self.zmax:self.dim]
        self.phi = X**2+Y**2+Z**2
        #--- locate x,y,z within phi ---#
        X, Y, Z = np.ogrid[self.xmin:self.xmax:self.dim,
                           self.ymin:self.ymax:self.dim,
                           self.zmin:self.zmax:self.dim]
        zo = np.zeros_like(self.phi)    
        Ix = ET.find_nearest(X, self.xloc)
        Iy = ET.find_nearest(Y, self.yloc)
        Iz = ET.find_nearest(Z, self.zloc)
        I = np.where((X == Ix) & (Y == Iy) & (Z == Iz))
        zo[I] = 1.
        zero_mask = np.ma.make_mask(zo)
        self.phi[zero_mask != False] = 0.
        self.phi[zero_mask != True] = 1.
        self.X, self.Y, self.Z = np.mgrid[self.xmin:self.xmax:self.dim,
                                          self.ymin:self.ymax:self.dim,
                                          self.zmin:self.zmax:self.dim]
        self.phi_copy = self.phi.copy()
        self.phi[self.mask != 0.0] = np.nan
        #--- end make phi ---#
        #--- get grid size/spacing ---#
        self.xspacing = int(self.X[1,1,1]-self.X[0,0,0])
        self.yspacing = int(self.Y[1,1,1]-self.Y[0,0,0])
        self.zspacing = int(self.Z[1,1,1]-self.Z[0,0,0])


    def name_match(self, dataDict, searchin, search4):
        """
        for searching in a pandas dataframe
        """
        #I = np.searchsorted(searchin, search4)
        I = np.where(searchin == search4)[0]
        found = dataDict[int(I[0]):int(I[0])+1]
        
        return found
        

    def add_earthquake_location(self, catalogue=None, event_id=None,
                                lon=None, lat=None, depth=None):
        """
        will come in as lat,long,depth(km),
        maybe from a catalogue and eventid
        - quakeml
        - csv
        """
        self.event_id = event_id
        if catalogue is not None:
            print "selecting an earthquake from a catalogue is work in progress"
            self.cat = PD.read_csv(catalogue, sep=',')
            self.eq = self.name_match(self.cat, searchin=self.cat['event_id'],
                                      search4=self.event_id)
            self.lon, self.lat, self.depth = self.eq['longitude'].values, \
            self.eq['latitude'].values, self.eq['depth'].values
            #return
        elif lon is not None and lat is not None and depth is not None:
            self.lon, self.lat, self.depth = lon, lat, depth
        self.eqx, self.eqy = ET.CT_nztm_to_wgs84([self.lon, self.lat],
                                                 islist=False, inverse=True)
        self.eqz = self.depth*1000.
        
        
    def calculate_travel_time(self, cell_size=None):
        """
        Uses scikits fmm (in C) to calculate the travel time from the 
        zero contour (earthquake or receiver) to everywhere in the 
        3D grid/volume.
        cell_size is to be a list or tuple specifying the 
        cell dimensions [x, y, z] \n
        TODO: check x and y spacing is right (it's the major control)
        ---> the order of dx=[tt.zspacing, tt.xspacing, tt.yspacing] looks best?
        make even in x, y and z if needed = regrid
        """
        #--- define cell size ---#
        if cell_size is None:
            cellx = self.xspacing
            celly = self.yspacing
            cellz = self.zspacing
        else:
            cellx, celly, cellz = cell_size
        #--- calculate the travel times ---#
        time1 = time.time()
        self.t = skfmm.travel_time(self.phi, self.v,
                                   dx=[cellx,
                                       celly,
                                       cellz],)
        time2 = time.time()
        print "calculated TT", self.t.shape, "in ",(time2-time1), "seconds"
        

    def plot_slice(self, direction='z', slice_no=-1, style='p'):
        """
        TODO: slice direction
        """
        nz = r"../GIS_data/NZ_COAST"
        nz = shapefile.Reader(nz)
        shapes = nz.shapes()
        #--- north and south islands ---#
        nxy = np.array(shapes[462].points)[::5]
        sxy = np.array(shapes[18].points)[::5]

        hik_chull = r"../GIS_data/hik_sub_convex"
        hik = shapefile.Reader(hik_chull)
        shapes = hik.shapes()
        hxy = np.array(shapes[0].points)

        nlon, nlat = nxy[:,0], nxy[:,1]
        slon, slat = sxy[:,0], sxy[:,1]
        hlon, hlat = hxy[:,0], hxy[:,1]
        hlon, hlat = ET.transCoords_WGS84_NZTM(hlon, hlat)

        self.slice_no = slice_no
        plt.figure(figsize=(15,8))
        depth = str(int(np.nanmin(self.z[:,:,int(self.slice_no)])))
        eqloc = str([self.xloc, self.yloc, self.zloc]) # nztm
        #eqloc = str([self.lon, self.lat, self.zloc]) # lon/lat
        plt.title("slice at %s"%depth+"km, n for EQ at: \n %s"%eqloc)
        if style == 'c':
            cont = plt.contourf(self.X[:,:,int(self.slice_no)],
                                self.Y[:,:,int(self.slice_no)],
                                self.t[:,:,int(self.slice_no)],
                                50, cmap=plt.cm.jet_r)
        elif style == 'p':
            cont = plt.pcolormesh(self.X[:,:,int(self.slice_no)],
                                  self.Y[:,:,int(self.slice_no)],
                                  self.t[:,:,int(self.slice_no)],
                                  cmap=plt.cm.jet_r)
        cb = plt.colorbar(cont)
        cb.set_label('OWT (model)')
        #--- eq loc ---#
        plt.contour(self.X[:,:,int(self.slice_no)], 
                    self.Y[:,:,int(self.slice_no)],
                    self.phi_copy[:,:,int(0)], 1,
                    colors='k')
        plt.scatter(self.xloc, self.yloc, s=50, marker='o',
                    edgecolor='k', facecolor='w')
        plt.plot(self.stations_xy[0], self.stations_xy[1], 'k+')
        #name = "graphics/vel_slices/slice_"+str(s)+".png"
        #name = "graphics/tt_slices/slice_"+str(s)+".png"
        plt.plot(hlon, hlat, 'k-', lw=0.5)
        plt.plot(nlon, nlat, 'k.', markersize=0.5)
        plt.plot(slon, slat, 'k.', markersize=0.5)
        plt.xlim(500000, 2700000)
        plt.ylim(4500000, 6500000)
        #plt.xlim(np.nanmin(x), np.nanmax(x))
        #plt.ylim(np.nanmin(y), np.nanmax(y))
        #----------------#


    def extract_values_on_surface_via_grid(self, cube=None, surface=None,
                                  vds=10, sds=20):
        """
        downsample for tests as it will be slow
        ---> figure out how to just get the indices
        """
        grd = gdal.Open(r'../GIS_data/SRL-Hikurangi_interface.grd')
        gt = grd.GetGeoTransform()
        gextent = (gt[0], gt[0]+grd.RasterXSize*gt[1],
                   gt[3]+grd.RasterYSize*gt[5], gt[3])
        self.surf_z = -grd.ReadAsArray()
        hiklon = np.linspace(gextent[0], gextent[1], grd.RasterXSize)
        hiklat = np.linspace(gextent[3], gextent[2], grd.RasterYSize)
        hiklong, hiklatg = np.meshgrid(hiklon, hiklat)
        self.surf_x, self.surf_y = ET.transCoords_WGS84_NZTM(hiklong, hiklatg)
        
        t = time.time()
        print "starting extraction (gridding) at:", t
        self.vos = griddata(np.vstack([(self.X.ravel()[::vds],
                                        self.Y.ravel()[::vds],
                                        self.Z.ravel()[::vds])]).T,
                            self.t.ravel()[::vds], 
                            (self.surf_x[::sds,::sds],
                             self.surf_y[::sds,::sds],
                             self.surf_z[::sds,::sds]),
                            method='linear')
                 
        nt = time.time()
        print "extraction time:", (nt-t)/60, "mins"


    def select_xyz_nearest_surface(self, limit=10e3):
        """
        tt_xyzt as vstack(n.ravel()).T
        try cKDTree
        """
        xyz = np.loadtxt(r'../GIS_data/SRL-Hikurangi_interface.xyz',
                         delimiter='\t')
        x, y, z = xyz[:,0], xyz[:,1], -xyz[:,2]
        x = x[~np.isnan(z)]
        y = y[~np.isnan(z)]
        sz = z[~np.isnan(z)]
        sx, sy = ET.transCoords_WGS84_NZTM(x, y)
        #tt_xyz = np.vstack([self.x.ravel(), self.y.ravel(),
        #                     self.z.ravel()*1000.]).transpose()
        tt_xyz = np.vstack([self.X.ravel(), self.Y.ravel(),
                            self.Z.ravel()]).transpose()                     
        surf = np.vstack([sx, sy, sz]).transpose()
    
        t = time.time()
        print "starting extraction (KDTree search)"
        tree = KDTree(surf)#, leafsize=)
        distances, ndx = tree.query([tt_xyz], k=1, distance_upper_bound=limit,
                                    n_jobs=2)
        print distances.shape
        d = distances[0]
        mask = np.isfinite(d)
        I = np.where(mask == True)
        tt_xyzt = np.vstack([self.x.ravel(), self.y.ravel(),
                             self.z.ravel(), self.t.ravel()]).transpose()[I]                    
        nt = time.time()
        print "extraction time:", (nt-t)/60, "mins"
        return tt_xyzt #[ex, ey, ez, ev]
    

    def save_H5(self, h5file=None, array_name=None):
        """
        save each tt array from the recievers into one h5 file
        or a series of earthquake tt arrays. \n
        example use: 
        h5f = h5py.File('recievers.h5', 'w')
        h5f.create_dataset('array1'), data=array1)
        h5f.create_dataset('array2'), data=array2)
        h5f.close()
        and to retrieve
        h5f = h5py.File('recievers.h5', 'r')
        array1 = h5f['array1'][:]
        """
        import h5py
        self.h5file = h5file
        if self.h5file is not None:
            if not os.path.exists(self.h5file): 
                h5f = h5py.File(self.h5file, 'w')
                h5f.create_dataset(array_name, data=self.t)
                h5f.close()
            else:
                h5f = h5py.File(self.h5file, 'a')
                h5f.create_dataset(array_name, data=self.t)
                h5f.close()
                

    def time_at_recievers(self, elev):
        """
        get the t value at each receiver location (grid cell)
        keep in pd df so can get names etc.
        TODO: auto pick up receiver elev based on slice no 
        NB: I recently changed it to search on XYz - seems ok so far
        TODO: for receiver locs get upper slice first to save time?
        """
        self.stations['NZTM_E'] = self.stations_xy[0]
        self.stations['NZTM_N'] = self.stations_xy[1]
        self.relev = elev
        self.stations['ELEVATION'] = np.linspace(self.relev, self.relev,
                                                 len(tt.stations_xy[0]))
        #self.xi = np.searchsorted(self.X.ravel(), self.stations.NZTM_E, side='right')
        #self.yi = np.searchsorted(self.Y.ravel(), self.stations.NZTM_N, side='right')
        t = time.time()
        print "retrieving reciever owt (cKDTree search)"
        #--- if use xyz instead of XYZ it takes longer time 9mins ---#
        
        tree = cKDTree(np.vstack([self.X.ravel(),
                                  self.Y.ravel(),
                                  self.Z.ravel()]).transpose())
        self.td, self.ndx = tree.query(np.vstack([self.stations['NZTM_E'],
                                      self.stations['NZTM_N'],
                                      self.stations['ELEVATION']]).transpose(),
                                      k=1, n_jobs=2)
        
        self.receiver_IP = np.vstack([self.X.ravel(), 
                                      self.Y.ravel(), 
                                      self.Z.ravel(),
                                      self.t.ravel()]).transpose()[self.ndx]        
        #plt.scatter(tt.receiver_IP[:,0], tt.receiver_IP[:,1], c=-tt.receiver_IP[:,3], linewidths=0)
        nt = time.time()
        print "took:", (nt-t)/60, "mins"
        self.rlon, self.rlat = ET.CT_nztm_to_wgs84([self.receiver_IP[:,0],
                                                    self.receiver_IP[:,1]],
                                                    islist=False,
                                                    inverse=False)
        distAzi = []
        for i, v in enumerate(self.rlat):
            d = gps2dist_azimuth(self.rlat[i], self.rlon[i],
                                 self.lat, self.lon)
            distAzi.append(d)
        self.distAzi = np.array(distAzi)
        self.receiver_IP = np.hstack([self.receiver_IP, self.distAzi])

        #--- how do i know order is right??? ---#        
        self.stations['out_x'] = self.receiver_IP[:,0]
        self.stations['out_y'] = self.receiver_IP[:,1]
        self.stations['OWT'] = self.receiver_IP[:,3]
        self.stations['eq2receiver_dist_m'] = self.distAzi[:,0]
        self.stations['eq2receiver_azi'] = self.distAzi[:,1]
        self.stations['eq2receiver_bazi'] = self.distAzi[:,2]


    def assign_negative_distances(self, ac):
        """
        ac is azimuth you want to split on
        only works >270 to 90 i.e upper part of circle
        """
        al = np.rad2deg(np.deg2rad(ac) - np.deg2rad(90))%360
        ar = np.rad2deg(np.deg2rad(ac) + np.deg2rad(90))%360
        #print al, ar
            
        for i, v in enumerate(self.distAzi[:,2]):
            if self.distAzi[:,2][i] > al and self.distAzi[:,2][i] < ar:
                self.distAzi[:,0][i] = self.distAzi[:,0][i]
                
            elif self.distAzi[:,2][i] < al and self.distAzi[:,2][i] > ar:
                self.distAzi[:,0][i] = -self.distAzi[:,0][i]
        self.stations['eq2receiver_dist_m'] = self.distAzi[:,0]         

    def IP_plot(self):    
        plt.figure()
        plt.plot(self.distAzi[:,0], self.receiver_IP[:,3], '.')
        

    
if __name__ == "__main__":
    #model = r'models/nzvel_even_grid_cropped_NZ_500j_ds10.npy'
    model = r'models/nzvel_even_grid_cropped_NZ_100j_ds1.npy'
    stations = r"../GIS_data/GEONET.csv"
    tt = reflections(model, stations)
    tt.load_model(dim=100j)
    tt.add_reciever_locations(network='NZ')#'ALL', 'NZ'
    #tt.add_earthquake_location(catalogue=None, event_id='2016p390950',
    #                           lon=176.88225, lat=-39.441883, depth=13.)
    tt.add_earthquake_location(catalogue='basic_eq_catalogue.csv',
                               event_id='2016p390950')

    #r = 0
    #tt.create_phi(tt.stations_xy[0][r], tt.stations_xy[1][r], 0.0)
    tt.create_phi(tt.eqx, tt.eqy, tt.eqz)
    tt.calculate_travel_time()#cell_size=[40e3,40e3,40e3])
    ##tt.save_H5('results/receivers.h5', tt.stations.Code[r:r+1].values[0])
    slice_no = -3 # (-3 is 0km, 0 is 750km, -1 is -15km)
    tt.plot_slice(direction='z', slice_no=slice_no, style='p')

    tt.time_at_recievers(elev=0.) #-15000.
    #o = tt.select_xyz_nearest_surface(limit=1000)
    #tt.extract_values_on_surface_via_grid(vds=20, sds=20)
    tt.assign_negative_distances(ac=25)
    
    scat = plt.scatter(tt.receiver_IP[:,0], tt.receiver_IP[:,1],
                       c=tt.receiver_IP[:,3],
                       cmap=plt.cm.jet_r, marker='s', s=30,
                       linewidths=0.2,
                       vmin=tt.t[:,:,int(slice_no)].min(),
                       vmax=tt.t[:,:,int(slice_no)].max())
    plt.colorbar(scat, label='OWT (receivers)')
    #plt.close('all')
    
    tt.IP_plot()
    
    p = plt.imread('graphics/no_time_match_tt2fast_crop.png')
    plt.imshow(p, extent=[-1500000,1500000,0,200], aspect=10000, alpha=0.3)


#from mpl_toolkits.mplot3d import Axes3D
#fig = plt.figure()
#ax = Axes3D(fig)
#ax.scatter(o[:,0], o[:,1], o[:,2], c=o[:,3], linewidths=0)

#fig = plt.figure()
#ax = Axes3D(fig)
#t = tt.vos.ravel()
#t[t == np.nan] = 0.
#t.reshape(tt.surf_x[::20,::20].shape)
#ax.scatter(tt.surf_x[::20,::20], tt.surf_y[::20,::20], tt.surf_z[::20,::20],
#           c=-t,
#           linewidths=0)


#-----------------------------------------------------------------------------#
#    #--- calculate the OWT for each receiver in the NZ network ---#
#    for i, receiver in enumerate(tt.stations.Code.values):
#        tt.create_phi(tt.stations_xy[0][i], tt.stations_xy[1][i], 0.0)
#        tt.calculate_travel_time()
#        tt.save_H5('results/receivers.h5', receiver)
#-----------------------------------------------------------------------------#


"""
TODO: 1) select earthquake from catalogue with event_id
      2) is earthquake above plate interface? as we want reflections
      3) get first arrival to reciever = time at surface, to test on section plot
      4) cut tt cubes for reciever and eq
      maybe a fast way ---> only cut once ans then get the indices of 
      the cut plane which should be the same for each.
      5) add receiver and eq tt values on plate interface together to get 
      twt reflection of plate interface, plot on section plot
      6) check with geonet where depth is measured from, SL or GL etc.
      
extras:
      slices through volume through x, y or z
      parallel processes when in loop

animations:
ffmpeg -framerate 7 -i '%03d.png' -c:v libx264 -pix_fmt yuv420p slices.mp4      
"""
