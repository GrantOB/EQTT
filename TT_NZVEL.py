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
import h5py
import obspy


import sys, os
#mylibpath = os.path.abspath(u'/Users/marine/OBrienG/')
#if not mylibpath in sys.path:
#    sys.path.append(mylibpath)
#else:
#    #print "re-appending", mylibpath
#    sys.path.remove(mylibpath)
#    sys.path.append(mylibpath)

import earthquake_tools as ET



class reflections(object):
    """
    calculate earthquake travel time (P, S)  using an input 3D velocity model.
    The earthquake focus needs to originate from somewhere within the \
    velocity model bounds.
    """
    def __init__(self, model, stations):
        self.model = model
        self.stations = stations
        #self.load_model()
        #self.calculate_travel_time()
        
    def load_model(self, dim=None, velocity_multiplier=1.):
        """
        model = 'models/nzvel_even_grid_cropped_NZ_100j_ds1.npy' # fast
        model = 'models/nzvel_even_grid_cropped_NZ_500j_ds10.npy' # slow
        dim = 100j
        """
        self.dim = dim
        self.x, self.y, self.z, self.vp, self.vs, self.mask = np.load(self.model)
        #self.z = 15000. + self.z
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
        self.vp = (self.vp*1000.) * velocity_multiplier
        self.vp[self.mask != 0.0] = np.nan
        self.vs = (self.vs*1000.) * velocity_multiplier
        self.vs[self.mask != 0.0] = np.nan


    def add_reciever_locations(self, network='NZ'):
        """
        stations = "GIS_data/GEONET.csv"
        """
        self.stations = PD.read_csv(self.stations, low_memory=False, skiprows=1)
        self.stations = tt.stations.drop_duplicates(subset={'Code'})
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
        print self.xloc, self.yloc, self.zloc
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


    def add_earthquake_location_old(self, catalogue=None, event_id=None,
                                lon=None, lat=None, depth=None):
        """
        will come in as lat,long,depth(km),
        maybe from a catalogue and eventid
        """
        if catalogue is not None:
            print "selecting an earthquake from a catalogue is not initiated yet"
            return
        self.lon, self.lat, self.depth = lon, lat, depth
        self.eqx, self.eqy = ET.CT_nztm_to_wgs84([self.lon, self.lat],
                                                 islist=False, inverse=True)
        self.eqz = self.depth*1000.
        

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
            print "NB: The earthquake needs to be in the catalogue"
            self.cat = PD.read_csv(catalogue, sep=',')
            self.eq = self.name_match(self.cat, searchin=self.cat['publicid'],
                                      search4=self.event_id)
            self.lon, self.lat, self.depth = self.eq['longitude'].values, \
            self.eq['latitude'].values, self.eq['depth'].values
            
        elif lon is not None and lat is not None and depth is not None:
            self.lon, self.lat, self.depth = lon, lat, depth
            self.eq = event_id
        self.eqx, self.eqy = ET.CT_nztm_to_wgs84([self.lon, self.lat],
                                                 islist=False, inverse=True)
        self.eqz = self.depth*1000.
        
        
    def calculate_P_travel_time(self, dxyz=None, iprint=False):
        """
        Uses scikits fmm (in C) to calculate the travel time from the 
        zero contour (earthquake or receiver) to everywhere in the 
        3D grid/volume.
        TODO: check x and y spacing is right (it changes everything!!!!!!!!)
        """
        #--- define cell sizes ---#
        if dxyz is None:
            dxyz = [self.xspacing, self.yspacing, self.zspacing]
        try:
            #--- calculate the travel times ---#
            time1 = time.time()
            self.t = skfmm.travel_time(self.phi, self.vp,
                                   dx=[dxyz[0], dxyz[1], dxyz[2]])
            time2 = time.time()
            if iprint is True:
                print "calculated eq Vp OWT", self.t.shape, "in ",(time2-time1), "seconds"
        except:
            self.t = np.zeros(self.phi.shape)
            pass

    def calculate_S_travel_time(self, dxyz=None):
        """
        Uses scikits fmm (in C) to calculate the travel time from the 
        zero contour (earthquake or receiver) to everywhere in the 
        3D grid/volume.
        """
        #--- define cell sizes ---#
        if dxyz is None:
            dxyz = [self.xspacing, self.yspacing, self.zspacing]
        #--- calculate the travel times ---#
        time1 = time.time()
        self.ts = skfmm.travel_time(self.phi, self.vs,
                                   dx=[dxyz[0], dxyz[1], dxyz[2]])
        time2 = time.time()
        print "calculated eq Vs OWT", self.ts.shape, "in ",(time2-time1), "seconds"
        

    def calculate_distances(self, dxyz=None):
        """
        Uses scikits fmm (in C) to calculate the distances from the 
        zero contour (earthquake or receiver) to every cell in the 
        3D grid/volume.
        this needs checking/updating as of 02/02/2017
        """
        if dxyz is None:
            dxyz = [self.xspacing, self.yspacing, self.zspacing]
        #--- claculate the distances ---#
        time1 = time.time()
        self.d = skfmm.distance(self.phi_copy, dx=[dxyz[0], dxyz[1], dxyz[2]])
        time2 = time.time()
        print "calculated distances", self.d.shape, "in ",(time2-time1), "seconds"


    def plot_slice(self, direction='z', slice_no=0, style='p', dists=False,
                   cmap=plt.cm.bwr_r):
        """
        TODO: slice direction
        """
        nz = r"GIS_data/NZ_COAST"
        nz = shapefile.Reader(nz)
        shapes = nz.shapes()
        #--- north and south islands ---#
        nxy = np.array(shapes[462].points)[::5]
        sxy = np.array(shapes[18].points)[::5]

        hik_chull = r"GIS_data/hik_sub_convex"
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
        eqloc = str([int(self.xloc), int(self.yloc), int(self.zloc)])
        plt.title("slice at %s"%depth+"km,  EQ at: %s"%eqloc \
        +',  model:'+self.model.split('/')[-1])
        if style is None:
            print ''
        if style == 'c':
            cont = plt.contourf(self.X[:,:,int(self.slice_no)],
                                self.Y[:,:,int(self.slice_no)],
                                self.t[:,:,int(self.slice_no)],
                                50, cmap=cmap)
            cb = plt.colorbar(cont)
            cb.set_label('OWT (model slice)')                    
        elif style == 'p':
            cont = plt.pcolormesh(self.X[:,:,int(self.slice_no)],
                                  self.Y[:,:,int(self.slice_no)],
                                  self.t[:,:,int(self.slice_no)],
                                  cmap=cmap)
            cb = plt.colorbar(cont)
            cb.set_label('OWT (model slice)')
        #--- distances ---#
        if dists is not False:
            self.calculate_distances()
            d = plt.contour(self.X[:,:,int(self.slice_no)],
                            self.Y[:,:,int(self.slice_no)],
                            self.d[:,:,int(self.slice_no)]/1000.,
                            10,
                            colors='k')
            plt.clabel(d, inline=1, fontsize=8)                    
        #--- eq loc ---#
        plt.contour(self.X[:,:,int(self.slice_no)], 
                    self.Y[:,:,int(self.slice_no)],
                    self.phi_copy[:,:,int(0)], 1,
                    colors='k')
        plt.scatter(self.xloc, self.yloc, s=100, marker='*',
                    edgecolor='k', facecolor='w', zorder=1000)
        plt.plot(self.stations_xy[0], self.stations_xy[1], 'k+')
        #name = "graphics/vel_slices/slice_"+str(s)+".png"
        #name = "graphics/tt_slices/slice_"+str(s)+".png"
        plt.plot(hlon, hlat, 'k-', lw=0.5)
        plt.plot(nlon, nlat, 'k.', markersize=0.5)
        plt.plot(slon, slat, 'k.', markersize=0.5)
        plt.xlim(500000, 2700000)
        plt.ylim(4500000, 6500000)
        plt.grid(True)
        #plt.xlim(np.nanmin(x), np.nanmax(x))
        #plt.ylim(np.nanmin(y), np.nanmax(y))
        #----------------#


    def extract_values_on_surface_via_grid(self, cube=None, surface=None,
                                  vds=10, sds=20):
        """
        downsample for tests as it will be slow
        ---> figure out how to just get the indices
        """
        grd = gdal.Open(r'GIS_data/SRL-Hikurangi_interface.grd')
        gt = grd.GetGeoTransform()
        gextent = (gt[0], gt[0]+grd.RasterXSize*gt[1],
                   gt[3]+grd.RasterYSize*gt[5], gt[3])
        self.surf_z = grd.ReadAsArray()
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
        xyz = np.loadtxt(r'GIS_data/SRL-Hikurangi_interface.xyz',
                         delimiter='\t')
        x, y, z = xyz[:,0], xyz[:,1], -xyz[:,2]*1000.
        x = x[~np.isnan(z)]
        y = y[~np.isnan(z)]
        sz = z[~np.isnan(z)]
        sx, sy = ET.transCoords_WGS84_NZTM(x, y)
        tt_xyz = np.vstack([self.X.ravel(), self.Y.ravel(),
                            self.Z.ravel()]).transpose()                     
        surf = np.vstack([sx, sy, sz]).transpose()
        t = time.time()
        print "starting extraction (cKDTree search) at:", t
        tree = cKDTree(surf)#, leafsize=)
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
    

    def save_cubes_H5(self, h5file=None, array_name=None):
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


    def distance_at_receivers(self, depth=0.):
        """
        should be integrated with time at recs, but this is a test
        on 2D space (not 3D)
        """
        self.stations['NZTM_E'] = self.stations_xy[0]
        self.stations['NZTM_N'] = self.stations_xy[1]
        self.stations['ELEVATION'] = np.linspace(depth, depth,
                                                 len(tt.stations_xy[0]))
        t = time.time()
        print "retrieving eq --> reciever distances (cKDTree search) at:", t
        tree = cKDTree(np.vstack([self.X.ravel(),
                                  self.Y.ravel(),
                                  self.Z.ravel()]).transpose())
        td, self.d_ndx = tree.query(np.vstack([self.stations['NZTM_E'],
                                               self.stations['NZTM_N'],
                                               self.stations['ELEVATION']]).transpose(),
                                               k=1)
        self.od = self.d.ravel()[self.d_ndx] 
               

    def find_nearest(self, array, value):
        idx = (np.abs(array-value)).argmin()
        return array[idx]


    def time_at_recievers(self, depth=0., gridOrReceiverLoc='grid'):
        """
        get the t value at each receiver location (each grid cell) \n
        uses pd df so can get names etc.
        specify depth so in future can use station elevation etc.
        """
        #self.X[:,:,int(self.slice_no)]
        #self.Y[:,:,int(self.slice_no)]
        #self.Z[:,:,int(self.slice_no)]
        #self.t[:,:,int(self.slice_no)]
        self.stations['NZTM_E'] = self.stations_xy[0]
        self.stations['NZTM_N'] = self.stations_xy[1]
        self.stations['ELEVATION'] = np.linspace(depth, depth,
                                                 len(tt.stations_xy[0]))
        t = time.time()
        print "Extracting eq --> reciever OWT (cKDTree search) at:", t
        #i = np.where(self.Z.ravel() == self.find_nearest(self.Z.ravel(), 0.0))
        #i = np.where(self.z.ravel() <= 0.)
#        i = self.slice_no
#        tree = cKDTree(np.vstack([self.X[:,:,i].ravel(),
#                                  self.Y[:,:,i].ravel(),
#                                  self.Z[:,:,i].ravel()]).transpose())
#                                  
#        td, self.t_ndx = tree.query(np.vstack([self.stations['NZTM_E'],
#                                               self.stations['NZTM_N'],
#                                               self.stations['ELEVATION']]).transpose(),
#                                               k=1, p=1, n_jobs=2)
#        self.receiver_IP = np.vstack([self.X[:,:,i].ravel(),
#                                      self.Y[:,:,i].ravel(), 
#                                      self.Z[:,:,i].ravel(),
#                                      self.t[:,:,i].ravel()]).transpose()[self.t_ndx]        
        i = self.slice_no
        tree = cKDTree(np.vstack([self.X.ravel(),
                                  self.Y.ravel(),
                                  self.Z.ravel()]).transpose())
                                  
        td, self.t_ndx = tree.query(np.vstack([self.stations['NZTM_E'],
                                    self.stations['NZTM_N'],
                                    self.stations['ELEVATION']]).transpose(),
                                    k=1, p=1, n_jobs=2)
        self.receiver_IP = np.vstack([self.X.ravel(),
                                      self.Y.ravel(), 
                                      self.Z.ravel(),
                                      self.t.ravel()]).transpose()[self.t_ndx]
        self.receiver_IP[self.receiver_IP == 0.] = np.nan
        nt = time.time()
        print "took:", (nt-t)/60, "mins"
        self.rlon, self.rlat = ET.CT_nztm_to_wgs84([self.receiver_IP[:,0],
                                                    self.receiver_IP[:,1]],
                                                    islist=False,
                                                    inverse=False)
        distAzi = []
        if gridOrReceiverLoc == 'receiver':
            for i, v in enumerate(self.rlat):
                d = gps2dist_azimuth(self.stations['Latitude'].values[i],
                                     self.stations['Longitude'].values[i],
                                     self.lat, self.lon)                      
                distAzi.append(d)
        elif gridOrReceiverLoc == 'grid':
            for i, v in enumerate(self.rlat):
                d = gps2dist_azimuth(self.rlat[i], self.rlon[i],
                                     self.lat, self.lon)
                distAzi.append(d)        
        self.distAzi = np.array(distAzi)
        self.receiver_IP = np.hstack([self.receiver_IP, self.distAzi])
        #--- add to df/dict ---#
        self.stations['out_x'] = self.receiver_IP[:,0]
        self.stations['out_y'] = self.receiver_IP[:,1]
        self.stations['eq2receiver_OWT'] = self.receiver_IP[:,3]
        self.stations['epi_%s_d'%gridOrReceiverLoc] = self.distAzi[:,0]
        self.stations['eq2receiver_azi'] = self.distAzi[:,1]
        self.stations['eq2receiver_bazi'] = self.distAzi[:,2]
        
      
    def assign_negative_distances(self, ac=None):
        """
        ac is azimuth you want to split 90 either side of 
        """
        al = np.rad2deg(np.deg2rad(ac) - np.deg2rad(90))%360
        ar = np.rad2deg(np.deg2rad(ac) + np.deg2rad(90))%360
        #print al, ar
        for i, v in enumerate(self.distAzi[:,2]):
            if self.distAzi[:,2][i] > al and self.distAzi[:,2][i] < ar:
                self.distAzi[:,0][i] = self.distAzi[:,0][i]
            elif self.distAzi[:,2][i] < al and self.distAzi[:,2][i] > ar:
                self.distAzi[:,0][i] = -self.distAzi[:,0][i]
        self.stations['epi_grid_d'] = self.distAzi[:,0]


    def receivers_on_map(self, cmap):
        plt.gca()
        scat = plt.scatter(self.receiver_IP[:,0], self.receiver_IP[:,1],
                           c=self.receiver_IP[:,3], cmap=cmap,
                           linewidths=1, marker='s', s=25,
                           vmin=self.t[:,:,self.slice_no].min(),
                           vmax=self.t[:,:,self.slice_no].max())
        plt.colorbar(scat, label='OWT (at receivers)')
        

    def create_stream_station_list(self):
        stream_station_names = []
        msl = []
        epi_d = []
        for tr in self.stream.sort(keys=['station']):
            stream_station_names.append(tr.stats.station)
            try:
                match = self.name_match(self.stations, self.stations.Code,
                                      tr.stats.station)
                msl.append(PD.DataFrame(match))
                epi_d.append(tr.stats.distance)
                      
            except:
                pass
                                 
        self.stream_station_names = np.array(stream_station_names)
        self.matched_stream_stations = PD.concat(msl)
        self.matched_stream_stations['epi_d'] = np.array(epi_d)


#    def match_steam_stations_to_stations(self):
#        self.msl = PD.DataFrame()
#        for name in self.stream_station_names:
#            try:
#                self.msl.append(self.name_match(self.stations,
#                                            self.stations.Code, name))
#            except:
#                print 'searching...'
#                                
#            #i = np.where(self.stream_station_names == name)[0]
#            #if len(i) > 0:
#            #    self.msl.append(self.stream_station_names[int(i)])
#        #self.msl = np.array(self.msl)    

        
    def tt_section_plot(self, event_id=None, picked=False, TWT=None,
                        log_distances=False, scale=5, vred=None,
                        pos_neg=None,
                        wffolder=r'D:/earthquakes/picked/'):
        if event_id is not None:
            st = obspy.read(wffolder+EVENT+'.h5')
            self.stream = st
            self.create_stream_station_list()
            
            #--- remove traces in stream not in chosen network ---#
            self.nstream = obspy.Stream()
            self.nstations = PD.DataFrame()
                        
            for name in self.stations.Code.values:
                #trname = tr.stats.station
                i = np.where(self.stream_station_names == name)[0]
                if len(i) > 0:
                    self.nstream.append(self.stream[int(i)])
#"""
#chnage: move station distances on to grid distances where names match and 
#leave rest as grid distances
#"""
                    
            names = []
            for tr in self.nstream:
                names.append(tr.stats.station)
            names = np.array(names)
            #--- remove stations not in stream ---#
            i = self.stations['Code'].isin(names)
            self.nstations = self.stations[i]

            #st.trim(endtime=st[0].stats.starttime+250.)
            if log_distances is True:
                for tr in st:
                    tr.stats.distance = np.log(tr.stats.distance)
            if pos_neg is False:
                for tr in st:
                    tr.stats.distance = abs(tr.stats.distance)
                    
            fig = plt.figure(figsize=(15, 8))
            ax = fig.add_axes([0.05,0.3,0.58,0.45])
            #gwf.stream.sort()
            self.nstream.plot(type='section', handle=True,
                    fig=fig, time_down=True, linewidth=0.25,
                    grid_linewidth=0.25, vred=vred, scale=scale)
            if picked is True:
                d = []
                t = []
                for tr in self.nstream:
                    d.append(tr.stats.distance)
                    t.append(tr.stats.ip_onset)
                self.epid = np.array(d)
                self.dmax = self.epid.max()
                
                #if log_distances is True:
                #    d = np.log(d)
                #    self.dmax = np.log(d.max())
                #    self.distAzi[:,0] = np.log(self.distAzi[:,0])
                    
                t = np.array(t)
                ax.plot(self.epid/self.dmax, t,
                        color='r', marker='x', linestyle='')    
                #i = np.argsort(self.distAzi[:,0]/self.dmax)
                #plt.plot((self.distAzi[:,0]/self.dmax)[i],
                #          self.receiver_IP[:,3][i], 'b+')
                i = np.argsort(self.epid/self.dmax)
                plt.plot((self.epid/self.dmax)[i],
                          self.receiver_IP[:,3][i], 'g.')
            #--------------------------------------------#
            # here              
            else:
                d = []
                for tr in self.nstream:
                    d.append(tr.stats.distance)
                self.epid = np.array(d)
                self.dmax = self.epid.max()
            #--------------------------------------------#    
                if log_distances is True:
                    d = np.log(self.epid)
                    self.dmax = np.log(d.max())
                    self.distAzi[:,0] = np.log(self.distAzi[:,0])
                    i = np.argsort(self.distAzi[:,0]/self.dmax)
                    plt.plot(self.distAzi[:,0][i], self.receiver_IP[:,3][i], 'm.')
                #--------------------------------------------#
                # here    
                else:    
                    #i = np.argsort(self.distAzi[:,0]/self.dmax)
                    #plt.plot((self.distAzi[:,0]/self.dmax)[i],
                    #          self.receiver_IP[:,3][i], 'b+')
                    i = np.argsort(self.matched_stream_stations['epi_d'].values/self.dmax)
                    plt.plot((self.matched_stream_stations['epi_d']/self.dmax),
                              self.matched_stream_stations.eq2receiver_OWT.values, 'b.')
                   
                   #I = np.argsort((tt.matched_stream_stations['epi_d'].values/tt.dmax))    
                   #plt.plot((tt.matched_stream_stations['epi_d'].values/tt.dmax)[I],
                   #np.array(twt_mins)[I], 'r.')          
                #--------------------------------------------#

#    def tt_section_plot(self, event_id=None, picked=False, TWT=None,
#                        log_distances=False, scale=5, vred=None,
#                        pos_neg=None,
#                        wffolder=r'D:/earthquakes/picked/'):
#        #import obspy
#        if event_id is not None:
#            st = obspy.read(wffolder+EVENT+'.h5')
#            #st.trim(endtime=st[0].stats.starttime+250.)
#            fig = plt.figure(figsize=(15, 8))
#            ax = fig.add_axes([0.05,0.3,0.58,0.45])
#            #gwf.stream.sort()
#            st.plot(type='section', handle=True,
#                    fig=fig, time_down=True, linewidth=0.25,
#                    grid_linewidth=0.25, vred=None, scale=scale)
#            if picked is True:
#                d = []
#                t = []
#                for tr in st:
#                    d.append(tr.stats.distance)
#                    t.append(tr.stats.ip_onset)
#                d = np.array(d)
#                self.dmax = d.max()
#                t = np.array(t)
#                ax.plot(d/d.max(), t,
#                        color='r', marker='x', linestyle='')    
#                i = np.argsort(self.distAzi[:,0]/d.max())
#                plt.plot((self.distAzi[:,0]/d.max())[i],
#                          self.receiver_IP[:,3][i], 'b+')
#            else:
#                d = []
#                for tr in st:
#                    d.append(tr.stats.distance)
#                d = np.array(d)
#                self.dmax = d.max()
#                i = np.argsort(self.distAzi[:,0]/d.max())
#                plt.plot((self.distAzi[:,0]/d.max())[i],
#                          self.receiver_IP[:,3][i], 'b+')
#                #plt.plot((self.distAzi[:,0]/d.max())[i],
#                #          self.r_twt[i], 'g.')
                          
                          
    def get_indices_of_cutting_surface(self, TTcube=None, surface=None):
        """
        TTcube is the cube of values you want to cut
        surface is surface to cut with, which the indices of/in TTcube
        will be stored
        """
        xyz = np.loadtxt(surface, delimiter='\t')
        x, y, z = xyz[:,0], xyz[:,1], -xyz[:,2]*1000
        x = x[~np.isnan(z)]
        y = y[~np.isnan(z)]
        sz = z[~np.isnan(z)]
        sx, sy = ET.transCoords_WGS84_NZTM(x, y)

        tt_xyz = np.vstack([self.X.ravel(), self.Y.ravel(),
                                self.Z.ravel()]).transpose()

        surf = np.vstack([sx, sy, sz]).transpose()
        
        t = time.time()
        print "starting surface cut (cKDTree search) at:", t
        tree = cKDTree(tt_xyz)
        ds, self.surface_indx = tree.query(surf,# distance_upper_bound=1e3,
                                    k=1, p=1, n_jobs=2)
        nt = time.time()
        print "cut time:", (nt-t)/60, "mins"
        #print self.surface_indx.shape
        tt_xyzv = np.vstack([self.X.ravel(), self.Y.ravel(),
                            self.Z.ravel(), self.t.ravel()]).transpose() 
        self.cut = tt_xyzv[self.surface_indx]
        self.cut[self.cut == 0.0] = np.nan

      
    def calculate_TWT_from_surface(self, receiverOWTfile, receivers='ALL',
                                   surface_indx=None):
        """
        NB: h5 files seem to sort by alphabet so h5.keys() does not 
        preserve the order of receivers.
        
        First off used the min of both eq and receiver tt,
        but think the receiver can't be the min as != eq min reflection point
        what is value of receiver tt at eq tt.min, get indx...
        
        """
        h5 = h5py.File(receiverOWTfile, 'r')
        if not receivers == 'ALL':
            self.receivers = receivers  
        else:
            self.receivers = self.stations.Code.values
        self.rs_owt_array = []
        self.rs_owt = []
        self.eqsr_twt_array = []
        self.eqsr_twt = []
        print "calculating receiver to surface OWT ..."
        for receiver in tqdm.tqdm(self.receivers):
            #TWTcube = h5[receiver].value + self.t
            ##self.r_twt.append(TWTcube.ravel()[surface_indx])
            #self.r_twt.append(TWTcube.ravel()[self.surface_indx])
            self.rs_owt_array.append(np.ravel(h5[receiver].value)[self.surface_indx])
            self.rs_owt.append(np.min(np.ravel(h5[receiver].value)[self.surface_indx]))
            self.eqsr_twt_array.append(self.cut[:,3] +\
                 np.ravel(h5[receiver].value)[self.surface_indx])
            self.eqsr_twt.append(np.nanmin(self.cut[:,3] +\
                 np.ravel(h5[receiver].value)[self.surface_indx]))
        #self.r_twt = np.array(self.r_twt)
        self.stations['eqsr_twt'] = self.eqsr_twt
        #if len(self.r_twt) == len(self.stations):
        #    self.stations['TWT'] = self.r_twt
        
        #I = np.where(self.cut[:,3] == self.cut[:,3].min())[0]
        
        
        
#    def travel_time_table(self, h5file=None, receiver_name=None):
#        """
#        begin saving tt's to tables
#        receivers first which only need to be done once.
#        example use: 
#        'w' = write/create
#        'a' = open and append to
#        'r' = read
#        
#        h5f = h5py.File('recievers.h5', 'w')
#        h5f.create_dataset('array1'), data=array1)
#        h5f.create_dataset('array2'), data=array2)
#        h5f.close()
#        and to retrieve
#        h5f = h5py.File('recievers.h5', 'r')
#        array1 = h5f['array1'][:]
#        """
#        import h5py
#        self.h5file = h5file
#        if h5file is not None:
#            if not os.path.exists(self.h5file): 
#                h5f = h5py.File(self.h5file, 'w')
#                h5f.create_dataset(receiver_name, data=self.t)
#                h5f.close()
#            else:
#                h5f = h5py.File(self.h5file, 'a')
#                h5f.create_dataset(receiver_name, data=self.t)
#                h5f.close()

       
        
#-----------------------------------------------------------------------------#        



if __name__ == "__main__":
    #--- Event and network selection ---#
    #EVENT, lon, lat, depth = '2014p429900', None, None, None
    NETWORK = 'ALL' #'ALL','WA','EA'
    
    #--- candidates ---#
    EVENT, lon, lat, depth = '2017p512943', 175.09854, -41.165409, 5.0
    # EVENT, lon, lat, depth = '2017p421929', 174.48, -40.21, 17.0 
    # EVENT, lon, lat, depth = '2017p292246', 177.69107, -38.512371, 38.0
    # EVENT, lon, lat, depth = '2017p293719', 173.15588, -41.635399, 7.0
    # EVENT, lon, lat, depth = '2017p323493', 174.96199, -40.16156, 17.0
    # EVENT, lon, lat, depth = '2017p138806', 175.70, -38.93, 10.
    # EVENT, lon, lat, depth = '2017p237835', 175.67, -38.93, 6.0
    # EVENT, lon, lat, depth = '2017p266160', 174.45, -41.42, 13.0
    # EVENT, lon, lat, depth = '2017p140052', 173.59, -42.24, 32.
    # EVENT, lon, lat, depth = '2017p135795', 175.14, -39.93, 23.
    # 2015p523441
    # EVENT, lon, lat, depth = '2016p842451', 173.76, -39.32, 14.0 #retry
    # 3303572
    # EVENT, lon, lat, depth = '2017p136022', 179.03, -37.29, 5.
    # 2014p429900
    # 2016p198476 n
    # 2016p105478
    # 2016p892721 n
    # 2016p913211 n
    # 2016p761676 n
    # 2016p390950 n
    # EVENT, lon, lat, depth = '2013p507880', 176.88, -39.44, 13.
    # EVENT, lon, lat, depth = '2016p826546', 176.92244, -37.305275, 177.
    #--- outside hik hull ---#
    # EVENT, lon, lat, depth = '2016p355041', 172.70204, -43.589134, 7.
    
    #--- Velocity model and seismometer stations ---#
    dims = 200j
    #model = r'models/nzvel_even_grid_cropped_NZ_%s_ds5.npy'%dims
    model = r'models/nzvel_vpvs_gc_%s_ds5.npy'%dims
    #model = r'models/iasp91.npy'
    stations = r"GIS_data/GEONET_HIK_EC.csv"
    
    #--- Initiate program, with useful options ---#
    tt = reflections(model, stations)
    tt.load_model(dim=dims, velocity_multiplier=1.)
    tt.add_reciever_locations(network=NETWORK)
    
    #--- choose an earthquake ---#
    #--- if event NOT in catalogue ---#
    tt.add_earthquake_location(catalogue=None, event_id=EVENT, 
                               lon=lon, lat=lat, depth=depth)
    #--- if event in catalogue ---#
    #tt.add_earthquake_location(catalogue='catalogues/earthquakes_Mw4_HIK_extent.csv',
    #                           event_id=EVENT)

#-----------------------------------------------------------------------------#
#    #--- calculate the OWT for each receiver in a network ---#
#    #--- only needs to be done once maybe for each velocity model---#
#    print "Calculating receiver OWT"
#    for i, receiver in enumerate(tqdm.tqdm(tt.stations.Code.values)):
#        receiver_elevation = 1.0
#        tt.create_phi(tt.stations_xy[0][i], tt.stations_xy[1][i], receiver_elevation)
#        tt.calculate_P_travel_time(iprint=False)
#        tt.save_cubes_H5('receiver_OWT/200/%s_receivers_gc_OWT.h5'%NETWORK, receiver)
#-----------------------------------------------------------------------------#

    #--- create zero contour in 3D model, use EQ or other (i.e. receiver) ---#
    tt.create_phi(tt.eqx, tt.eqy, tt.eqz)
    
    #--- calculate distances as a check ---#
    #tt.calculate_distances()
    #tt.distance_at_receivers(depth=1.)

    #--- Calculate travel-times, one can change the cell sizes ---#
    tt.calculate_P_travel_time(iprint=True)#dxyz=[tt.xspacing, tt.yspacing, tt.zspacing])
    
    #--- saving, plotting and data extraction options ---#
    ##tt.save_H5('results/receivers.h5', tt.stations.Code[r:r+1].values[0])
    slice_no = 2
    cmap = plt.cm.bwr_r
    tt.plot_slice(direction='z', slice_no=slice_no, style=None, dists=False,
                  cmap=cmap)
    tt.time_at_recievers(depth=0., gridOrReceiverLoc='receiver')#'grid' or 'receiver')
    #
    
    #o = tt.select_xyz_nearest_surface(limit=1000)
    #tt.extract_values_on_surface_via_grid(vds=20, sds=20)
    
    tt.receivers_on_map(cmap=cmap)

    surface = r'GIS_data/SRL-Hikurangi_interface.xyz'
    tt.get_indices_of_cutting_surface(TTcube=None, surface=surface)

    tt.calculate_TWT_from_surface(receiverOWTfile='receiver_OWT/%s/ALL_receivers_gc_OWT.h5'%str(dims)[0:-1],#%NETWORK,
                receivers='ALL', surface_indx=tt.surface_indx)

    rtest = tt.receivers.searchsorted('BFZ')
    scat = plt.scatter(tt.cut[:,0], tt.cut[:,1], c=tt.eqsr_twt_array[rtest],
                       marker='s', linewidths=0,
                       s=np.sqrt(tt.eqsr_twt_array[rtest])+50, cmap=cmap,
                       #vmin=tt.t[:,:,tt.slice_no].min(),
                       #vmax=tt.t[:,:,tt.slice_no].max()
                       alpha=0.5)
    plt.colorbar(scat, label='TWT (at hik)')
    
    fastest = np.argmin(tt.rs_owt_array[rtest])
    plt.scatter(tt.cut[:,0][fastest], tt.cut[:,1][fastest], edgecolor='k',
                facecolor='w', marker='o', s=50,)
        
    #plt.close()
#    I = np.where(tt.cut[:,3] == np.nanmin(tt.cut[:,3]))[0][0]
#    
#    tt.calculate_TWT_from_surface(receiverOWTfile='receiver_OWT/%s_receivers_OWT.h5'%NETWORK,
#                     receivers='ALL', surface_indx=tt.surface_indx[I])
#

    #tt.assign_negative_distances(ac=20)
    #tt.stream.filter('bandpass', freqmin=12., freqmax=14.)
    tt.tt_section_plot(event_id=EVENT, picked=False, scale=1, vred=None, 
                       pos_neg=True, log_distances=False,
                       wffolder=r'D:/earthquakes/picked/')
    #s = np.argsort(tt.distAzi[:,0])
    #plt.plot((tt.distAzi[:,0]/tt.dmax)[s], tt.stations['TWT'].values[s], 'go')
    #plt.plot((tt.distAzi[:,0]/tt.dmax)[s], tt.r_twt[s], 'r.')
    twt_mins = []
    for i, v in enumerate(tt.matched_stream_stations.eqsr_twt):
        twt_mins.append(tt.matched_stream_stations.eqsr_twt.values[i])
        #plt.plot((tt.distAzi[:,0]/tt.dmax)[i], np.nanmin(tt.eqsr_twt_array[i]), 'g.')
    #I = np.argsort((tt.distAzi[:,0]/tt.dmax))    
    #plt.plot((tt.distAzi[:,0]/tt.dmax)[I], np.array(twt_mins)[I], 'g.')
    I = np.argsort((tt.matched_stream_stations['epi_d'].values/tt.dmax))    
    #plt.plot((tt.matched_stream_stations['epi_d'].values/tt.dmax)[I],
    #         np.array(twt_mins)[I], 
    #         markeredgecolor='r', marker='o', markerfacecolor=None, linestyle='')
    plt.scatter((tt.matched_stream_stations['epi_d'].values/tt.dmax)[I],
             np.array(twt_mins)[I], facecolors='none', edgecolors='r')

    plt.scatter((tt.stations['epi_receiver_d'].values/tt.dmax),
             tt.stations['eqsr_twt'].values, facecolors='none', 
             edgecolors='y', marker='*')



#-----------------------------------------------------------------------------#

#-----------------------------------------------------------------------------#

#
#from mpl_toolkits.mplot3d import Axes3D
#fig = plt.figure()
#ax = Axes3D(fig)
#ax.scatter(tt.matched_stream_stations['NZTM_E'],
#           tt.matched_stream_stations['NZTM_N'],
#           tt.matched_stream_stations['eq2receiver_OWT'],
#           c='g', alpha=1,
#           linewidths=0)
#
#ax.scatter(tt.matched_stream_stations['NZTM_E'],
#           tt.matched_stream_stations['NZTM_N'],
#           tt.matched_stream_stations['eqsr_twt'],
#           c='b', alpha=1,
#           linewidths=0)
#
#ax.scatter(tt.matched_stream_stations['NZTM_E'],
#           tt.matched_stream_stations['NZTM_N'],
#           tt.matched_stream_stations['eqsr_twt']-\
#           tt.matched_stream_stations['eq2receiver_OWT'],
#           c='k', alpha=1,
#           linewidths=0)
#plt.show()



#import obspy
#st = obspy.read(r'D:/earthquakes/picked/'+EVENT+'.h5')
#
##st.trim(endtime=st[0].stats.starttime+250.)
#
#fig = plt.figure(figsize=(15, 8))                
#ax = fig.add_axes([0.05,0.3,0.58,0.45])
##gwf.stream.sort()
#st.plot(type='section', handle=True,
#        fig=fig, time_down=True, linewidth=0.25,
#        grid_linewidth=0.25, vred=None, scale=5)
#
#d = []
#t = []
#for tr in st:
#    d.append(tr.stats.distance)
#    t.append(tr.stats.ip_onset)
#
#d = np.array(d)
#t = np.array(t)
#ax.plot(d/d.max(), t,
#            color='r', marker='.', linestyle='')
#
#i = np.argsort(tt.distAzi[:,0]/d.max())
#plt.plot((tt.distAzi[:,0]/d.max())[i], tt.receiver_IP[:,3][i], 'b.')
##plt.plot(tt.od/242833.7268669467, tt.receiver_IP[:,3], 'g.')
#
##for i, v in enumerate(tt.t[:,0,0]):
##    plt.plot(tt.distAzi[:,0]/d.max(), tt.t[:,:,i].ravel()[tt.t_ndx], '+')
##    #i += 20

#-----------------------------------------------------------------------------#
#--- example quake search ---#
#http://quakesearch.geonet.org.nz/csv?bbox=163.60840,-49.18170,182.98828,-32.28713&minmag=4&startdate=2010-1-1T0:00:00&enddate=2017-2-7T3:00:00


#from matplotlib.table import Table
#def checkerboard_table(data, fmt='{:.2f}', bkg_colors=['yellow', 'white']):
#    fig, ax = plt.subplots()
#    ax.set_axis_off()
#    tb = Table(ax, bbox=[0,0,1,1])
#
#    nrows, ncols = data.shape
#    width, height = 1.0 / ncols, 1.0 / nrows
#
#    # Add cells
#    for (i,j), val in np.ndenumerate(data):
#        # Index either the first or second item of bkg_colors based on
#        # a checker board pattern
#        idx = [j % 2, (j + 1) % 2][i % 2]
#        color = bkg_colors[idx]
#
#        tb.add_cell(i, j, width, height, text=fmt.format(val), 
#                    loc='center', facecolor=color)
#
#    # Row Labels...
#    for i, label in enumerate(data.index):
#        tb.add_cell(i, -1, width, height, text=label, loc='right', 
#                    edgecolor='none', facecolor='none')
#    # Column Labels...
#    for j, label in enumerate(data.columns):
#        tb.add_cell(-1, j, width, height/2, text=label, loc='center', 
#                           edgecolor='none', facecolor='none')
#    ax.add_table(tb)
#    return fig



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
      3) get first arrival to reciever = time at surface, to test on section plot - done
      4) cut tt cubes for reciever and eq
      maybe a fast way ---> only cut once and then get the indices of 
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
