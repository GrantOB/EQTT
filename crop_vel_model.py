# -*- coding: utf-8 -*-
"""
Created on Fri Sep 16 15:46:04 2016

@author: obrien
"""
import time
import sys
import os
import numpy as np
from scipy.interpolate import griddata#, RegularGridInterpolator
import gdal
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import shapefile
from mpl_toolkits.basemap import pyproj
from matplotlib import path

mylibpath = os.path.abspath(u'/Users/marine/OBrienG/')
if not mylibpath in sys.path:
    sys.path.append(mylibpath)
else:
    #print "re-appending", mylibpath
    sys.path.remove(mylibpath)
    sys.path.append(mylibpath)

from GOBEQ import earthquake_tools as ET


###############################################################################
#--- cut regridded velocity model by the huk subduction surface ---#
#--- the thing: the vel model will be cut to the xy extent of the sub as well ---#
#--- reload saved model ---#

#model = 'models/nzvel_even_grid500j_ds10.npy'
model = 'models/nzvel_even_grid100j_ds1.npy'
indim = 100j
inds = 1

x,y,z,v = np.load(model)
model_shape = x.shape

#--- convert lon, lat to meters, z: km to m ---#
x, y = ET.transCoords_WGS84_NZTM(x, y)
#z = z*1000. # this actually broke it!!!

#cell_sizex = x[1,1,1] - x[0,0,0]
#cell_sizey = y[1,1,1] - y[0,0,0]


zmin = z.min()
#z = z-zmin
#--- chop off land ---#
#--- TODO: don't chop off land ---#
#z[z < 0.] = 0.#np.nan

xyzs = np.vstack([x.ravel(), y.ravel(), z.ravel(), v.ravel()]).transpose()

grd = gdal.Open('../GIS_data/SRL-Hikurangi_interface.grd')
gt = grd.GetGeoTransform()
gextent = (gt[0], gt[0]+grd.RasterXSize*gt[1],
          gt[3]+grd.RasterYSize*gt[5], gt[3])

hik = -grd.ReadAsArray()
#hik = -(grd.ReadAsArray()+zmin)

hikmax = hik[~np.isnan(hik)].max()
hikmin = hik[~np.isnan(hik)].min()

hiklon = np.linspace(gextent[0], gextent[1], grd.RasterXSize)
hiklat = np.linspace(gextent[3], gextent[2], grd.RasterYSize)

### this to make -180 to 180 ###
#hiklon[hiklon > 180.] = (-360.+hiklon[hiklon > 180.])

hiklong, hiklatg = np.meshgrid(hiklon, hiklat)

hiklong, hiklatg = ET.transCoords_WGS84_NZTM(hiklong, hiklatg)

#hpoints = np.vstack([hiklong.ravel()[::5], hiklatg.ravel()[::5],
#                     hik.ravel()[::5]]).T

#--- may have to first crop the model by the extent/chull of the huk sub ---#
def ngrid_below_surf(surf_x, surf_y, surf_z, nx, ny, nz, ns, ds=1, mask=None):
    #--- get vol xyz where surf would be ---#
    Z = griddata(np.vstack([(surf_x[::ds,::ds].ravel(),
                             surf_y[::ds,::ds].ravel(),
                             surf_z[::ds,::ds].ravel())]).T,
                 surf_z[::ds,::ds].ravel(), 
                 (nx, ny, nz),
                 method='nearest')

    if mask is None:
        I = np.where(Z <= nz)
        nx[I] = np.nan
        ny[I] = np.nan
        nz[I] = np.nan
        ns[I] = np.nan        
        return nx, ny, nz, ns
    
    elif mask is True:
        I = np.where(Z >= nz)
        nx.mask[I] = False
        ny.mask[I] = False
        nz.mask[I] = False
        ns.mask[I] = False
        return nx, ny, nz, ns


#--- run and time function ---#
t = time.time()
print "start time:", t

#--- ds=1 took about 3 hours ---#
#x, y, z, v = ngrid_below_surf(hiklong, hiklatg, hik, x,y,z,v, ds=20, mask=None)

nt = time.time()
print "time taken =", (nt-t)/60, "mins"
#--- end function ---#


##--- mask outside hik chull ---#
#hik_chull = "../GIS_data/hik_sub_convex"
#hik_chull = shapefile.Reader(hik_chull)
#shapes = hik_chull.shapes()
#hxy = np.array(shapes[0].points)
#hxy = np.vstack([ET.transCoords_WGS84_NZTM(hxy[:,0], hxy[:,1])]).transpose()
#hxy_path = path.Path(hxy)
#vinside = hxy_path.contains_points(np.vstack([x.ravel(),y.ravel()]).T)
#mask = vinside.reshape(model_shape)

#--- mask outside useful boundary ---#
chull = "models/vel_cropper"
chull = shapefile.Reader(chull)
shapes = chull.shapes()
hxy = np.array(shapes[0].points)
#hxy = np.vstack([ET.transCoords_WGS84_NZTM(hxy[:,0], hxy[:,1])]).transpose()
hxy_path = path.Path(hxy)
vinside = hxy_path.contains_points(np.vstack([x.ravel(),y.ravel()]).T)
mask = vinside.reshape(model_shape)


x = np.ma.masked_array(x, ~mask)
y = np.ma.masked_array(y, ~mask)
z = np.ma.masked_array(z, ~mask)
v = np.ma.masked_array(v, ~mask)


#fig = plt.figure()
#s = -1
#plt.pcolormesh(x[:,:,int(s)], y[:,:,int(s)], v[:,:,int(s)],
#                      cmap=plt.cm.jet_r)
#
#plt.plot(hxy[:,0], hxy[:,1])


#fig = plt.figure()
#ax = Axes3D(fig)
#ds = 5
#ax.scatter(x[::ds,::ds,::ds], y[::ds,::ds,::ds], z[::ds,::ds,::ds],
#           c=v[::ds,::ds,::ds], linewidths=0, alpha=0.5)
#
#ax.plot_wireframe(hiklong, hiklatg, hik, color='k', 
#                  rstride=50, cstride=50)
#
#ax.invert_zaxis()


np.save('models/nzvel_even_grid_cropped_NZ_%s'%indim+'_ds%s.npy'%inds,
        [x, y, z, v, v.mask])


#---
#I = np.where(Z <= nz)
#v = nz
#v[I] = np.nan
#---
