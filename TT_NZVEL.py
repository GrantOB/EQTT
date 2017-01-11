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
    calculate twt for earthquakes...
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
        
        #self.v = np.flipud(self.v)


    def add_reciever_locations(self, network='NZ'):
        """
        stations = "GIS_data/GEONET.csv"
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
        print self.xloc, self.yloc, self.zloc
        #--- make phi ---#
        X, Y, Z = np.mgrid[self.xmin:self.xmax:self.dim,
                           self.ymin:self.ymax:self.dim,
                           self.zmin:self.zmax:self.dim]
        self.phi = X**2+Y**2+Z**2
        #self.phi = np.flipud(self.phi)
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


    def add_earthquake_location(self, catalogue=None, event_id=None,
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
        
        
    def calculate_travel_time(self, dxyz=None):
        """
        Uses scikits fmm (in C) to calculate the travel time from the 
        zero contour (earthquake or receiver) to everywhere in the 
        3D grid/volume.
        TODO: check x and y spacing is right (it changes everything!!!!!!!!)
        ---> the order of dx=[tt.zspacing, tt.xspacing, tt.yspacing] looks best?
        make even in x, y and z if needed = regrid
        """
        #self.v = np.flipud(self.v)
        
        
        if dxyz is None:
            dxyz = [self.xspacing, self.yspacing, self.zspacing]
        #--- calculate the travel times ---#
        time1 = time.time()
        self.t = skfmm.travel_time(self.phi, self.v,
                                   dx=[dxyz[0], dxyz[1], dxyz[2]])
        time2 = time.time()
        print "calculated TT", self.t.shape, "in ",(time2-time1), "seconds"
        

    def calculate_distances(self, dxyz=None):
        """
        Uses scikits fmm (in C) to calculate the distances from the 
        zero contour (earthquake or receiver) to every cell in the 
        3D grid/volume.

        """
        if dxyz is None:
            dxyz = [self.xspacing, self.yspacing, self.zspacing]
        #--- claculate the distances ---#
        time1 = time.time()
        self.d = skfmm.distance(self.phi_copy, dx=[dxyz[0], dxyz[1], dxyz[2]])
        time2 = time.time()
        print "calculated distances", self.d.shape, "in ",(time2-time1), "seconds"


    def plot_slice(self, direction='z', slice_no=-1, dists=None):
        nz = r"GIS_data/NZ_COAST"
        nz = shapefile.Reader(nz)
        shapes = nz.shapes()
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
        plt.figure()
        depth = str(int(np.nanmin(self.z[:,:,int(self.slice_no)])))
        eqloc = str([self.xloc, self.yloc, self.zloc])
        plt.title("slice at %s"%depth+"km, for EQ at: %s"%eqloc)
        #cont = plt.contourf(self.X[:,:,int(self.slice_no)],
        #                    self.Y[:,:,int(self.slice_no)],
        #                    self.t[:,:,int(self.slice_no)],
        #                    50, cmap=plt.cm.jet_r)
        cont = plt.pcolormesh(self.X[:,:,int(self.slice_no)],
                              self.Y[:,:,int(self.slice_no)],
                              self.t[:,:,int(self.slice_no)],
                              cmap=plt.cm.jet_r)
        cb = plt.colorbar(cont)
        cb.set_label('OWT')
        #--- distances ---#
        if dists is not None:
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
        grd = gdal.Open(r'GIS_data/SRL-Hikurangi_interface.grd')
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
        xyz = np.loadtxt(r'GIS_data/SRL-Hikurangi_interface.xyz',
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
        print "starting extraction (KDTree search) at:", t
        tree = KDTree(surf)#, leafsize=)
        distances, ndx = tree.query([tt_xyz], k=1, distance_upper_bound=limit)
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


    def distance_at_receivers(self):
        """
        should be integrated with time at recs, but this is a test
        on 2D space (not 3D)
        """
        self.stations['NZTM_E'] = self.stations_xy[0]
        self.stations['NZTM_N'] = self.stations_xy[1]
        self.stations['ELEVATION'] = np.linspace(-15000.,-15000.,
                                                 len(tt.stations_xy[0]))
        t = time.time()
        print "retrieving reciever distances (cKDTree search) at:", t
        tree = cKDTree(np.vstack([self.X.ravel(),
                                  self.Y.ravel(),
                                  self.Z.ravel()]).transpose())
        td, self.d_ndx = tree.query(np.vstack([self.stations['NZTM_E'],
                                               self.stations['NZTM_N'],
                                               self.stations['ELEVATION']]).transpose(),
                                               k=1)
        self.od = self.d.ravel()[self.d_ndx] 
               

    def time_at_recievers(self):
        """
        get the t value at each receiver location (each grid cell) \n
        uses pd df so can get names etc.
        """
        self.stations['NZTM_E'] = self.stations_xy[0]
        self.stations['NZTM_N'] = self.stations_xy[1]
        self.stations['ELEVATION'] = np.linspace(-15000.,-15000.,
                                                 len(tt.stations_xy[0]))
        #self.xi = np.searchsorted(self.X.ravel(), self.stations.NZTM_E, side='right')
        #self.yi = np.searchsorted(self.Y.ravel(), self.stations.NZTM_N, side='right')
        t = time.time()
        print "retrieving reciever owt (cKDTree search) at:", t
        tree = cKDTree(np.vstack([self.X.ravel(),
                                  self.Y.ravel(),
                                  self.Z.ravel()]).transpose())
        td, self.t_ndx = tree.query(np.vstack([self.stations['NZTM_E'],
                                               self.stations['NZTM_N'],
                                               self.stations['ELEVATION']]).transpose(),
                                               k=1)
        self.receiver_IP = np.vstack([self.X.ravel(), self.Y.ravel(), 
                                       self.Z.ravel(),
                                       self.t.ravel()]).transpose()[self.t_ndx]        
        #plt.scatter(tt.receiver_IP[:,0], tt.receiver_IP[:,1], c=-tt.receiver_IP[:,3], linewidths=0)
        nt = time.time()
        print "took:", (nt-t)/60, "mins"

        self.rlon, self.rlat = ET.CT_nztm_to_wgs84([self.receiver_IP[:,0],
                                                    self.receiver_IP[:,1]],
                                                    islist=False,
                                                    inverse=False)
        distAzi = []
        for i, v in enumerate(self.rlat):
            d = gps2dist_azimuth(self.rlat[i], self.rlon[i], self.lat, self.lon)
            distAzi.append(d)
        self.distAzi = np.array(distAzi)
        self.receiver_IP = np.hstack([self.receiver_IP, self.distAzi])

        #--- is order correct ? ---#
        self.stations['out_x'] = self.receiver_IP[:,0]
        self.stations['out_y'] = self.receiver_IP[:,1]
        self.stations['OWT'] = self.receiver_IP[:,3]
        self.stations['eq2receiver_dist_m'] = self.distAzi[:,0]
        self.stations['eq2receiver_azi'] = self.distAzi[:,1]
        self.stations['eq2receiver_bazi'] = self.distAzi[:,2]
        
        
    def assign_negative_distances(self, ac):
        """
        ac is azimuth you want to split 90 eiath side of 
        """
        al = np.rad2deg(np.deg2rad(ac) - np.deg2rad(90))%360
        ar = np.rad2deg(np.deg2rad(ac) + np.deg2rad(90))%360
        print al, ar
            
        for i, v in enumerate(self.distAzi[:,2]):
            if self.distAzi[:,2][i] > al and self.distAzi[:,2][i] < ar:
                self.distAzi[:,0][i] = self.distAzi[:,0][i]
                
            elif self.distAzi[:,2][i] < al and self.distAzi[:,2][i] > ar:
                self.distAzi[:,0][i] = -self.distAzi[:,0][i]
            
        self.stations['eq2receiver_dist_m'] = self.distAzi[:,0][i]
        
        
    def IP_plot(self):
        plt.figure()
        plt.plot(self.distAzi[:,0], self.receiver_IP[:,3], '.')
        
    
if __name__ == "__main__":
    model = r'models/nzvel_even_grid_cropped_NZ_100j_ds1.npy'
    stations = r"GIS_data/GEONET.csv"
    tt = reflections(model, stations)
    tt.load_model(dim=100j)
    tt.add_reciever_locations(network='NZ')#'ALL'
    tt.add_earthquake_location(catalogue=None, event_id=None,
                               lon=176.88, lat=-39.44, depth=13.)
    #r = 0
    #tt.create_phi(tt.stations_xy[0][r], tt.stations_xy[1][r], 0.0)
    tt.create_phi(tt.eqx, tt.eqy, tt.eqz)
    
    #--- calculate distances as a check ---#
    tt.calculate_distances()
    tt.distance_at_receivers()

    tt.calculate_travel_time()#dxyz=[40000, 30000, 25000])
    ##tt.save_H5('results/receivers.h5', tt.stations.Code[r:r+1].values[0])
    tt.plot_slice(direction='z', slice_no=-1, dists=True)
    tt.time_at_recievers()
    tt.assign_negative_distances(ac=20)
    
    #o = tt.select_xyz_nearest_surface(limit=1000)
    #tt.extract_values_on_surface_via_grid(vds=20, sds=20)

    scat = plt.scatter(tt.receiver_IP[:,0], tt.receiver_IP[:,1], c=tt.receiver_IP[:,3],
                   cmap=plt.cm.jet_r, linewidths=0, marker='s', s=25)
                   #vmin=tt.t[:,:,tt.slice_no].min(),
                   #vmax=tt.t[:,:,tt.slice_no].max())
    plt.colorbar(scat)
    #tt.IP_plot()
    #plt.close('all')



import obspy
st = obspy.read(r'D:/earthquakes/out_stream.h5')

st.trim(endtime=st[0].stats.starttime+250.)

fig = plt.figure(figsize=(15, 8))                
ax = fig.add_axes([0.05,0.3,0.58,0.45])
#gwf.stream.sort()
st.plot(type='section', handle=True,
        fig=fig, time_down=True, linewidth=0.25,
        grid_linewidth=0.25, vred=None, scale=5)

d = []
t = []
for tr in st:
    d.append(tr.stats.distance)
    t.append(tr.stats.ip_onset)
    ax.plot(tr.stats.distance/242833.7268669467, tr.stats.ip_onset,
            color='r', marker='.', linestyle='')

d = np.array(d)

wtf = 1.3
plt.plot(tt.distAzi[:,0]/242833.7268669467, tt.receiver_IP[:,3], 'b.')
#plt.plot(tt.od/242833.7268669467, tt.receiver_IP[:,3], 'g.')


#--------#


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
