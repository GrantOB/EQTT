# -*- coding: utf-8 -*-
"""
Created on Fri Aug 12 14:55:34 2016

@author: marine==obrien

make regularly spaced grid from Donna's latest NZVEL model (3D)
"""
import time
import sys
import os
import numpy as np
from scipy.interpolate import griddata#, RegularGridInterpolator
import gdal
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


mylibpath = os.path.abspath(u'/Users/marine/OBrienG/')
if not mylibpath in sys.path:
    sys.path.append(mylibpath)
else:
    #print "re-appending", mylibpath
    sys.path.remove(mylibpath)
    sys.path.append(mylibpath)

from GOBEQ import earthquake_tools as ET


table = "models/Table_S1.txt"

vel = np.genfromtxt(table, names=True)
#lon = vel['Longitude']

lon, lat, depth = vel['Longitude'],vel['Latitude'],vel['Depthkm_BSL']

points = np.vstack([lon.ravel(), lat.ravel(), depth.ravel()]).T
values = vel['Vp'].ravel()

#--- choose dims ---#
dim = 100j #500j
nx,ny,nz = np.mgrid[lon.min():lon.max():dim, lat.min():lat.max():dim,
                  depth.max():depth.min():dim]
#cell_size = ?
newp = np.vstack([nx.ravel(), ny.ravel(), nz.ravel()]).T

###############################################################################
#--- regrid velocity model so it is evenly spaced cells ---#
t = time.time()
print "start time:", t

#--- downsample ---#
#--- no ds takes about 40 mins ---#

ds = 10 # 10 5 2
new = griddata(points[::ds], values[::ds], newp, method='linear',
               fill_value=np.nan)#0. for plotly

np.save('models/nzvel_even_grid%s'%dim+'_ds%s.npy'%ds,
        [nx, ny, nz, new.reshape(nz.shape)])

nt = time.time()
print "time taken =", (nt-t)/60, "mins"
###############################################################################

#x = m[:,0].reshape(model_shape)
#y = m[:,1].reshape(model_shape)
#z = m[:,2].reshape(model_shape)
#v = m[:,3].reshape(model_shape)
#
##newp[i[:,0]] = np.nan
##newp = newp[i[:,0]]
#
#
##--- vel model values at surface ---#
##new = griddata(points[::ds], values[::ds], hpoints[::ds], method='linear')
#
#
#fig = plt.figure()
#ax = Axes3D(fig)
#ds = 5
#ax.scatter(xyzs[m][:,0][::ds], xyzs[m][:,1][::ds], xyzs[m][:,2][::ds],
#           c='k', linewidths=0)
#
#ax.invert_zaxis()
#
#
#

###############################################################################
#--- visualisations ---#

#from mpl_toolkits.mplot3d import Axes3D
#import matplotlib.pyplot as plt
#
##nz[nz < 0.] = np.nan
##new = np.sqrt(new)
##new[new == 'nan'] = 0
#
#nx = nx.ravel()
#ny = ny.ravel()
#nz = nz.ravel()
#
#I = np.where(new != 0.)[0]
#nx = nx[I]
#ny = ny[I]
#nz = nz[I]
#new = new[I]
#
#I = np.where(nz < 300.)
#nx = nx[I]
#ny = ny[I]
#nz = nz[I]
#new = new[I]
#
#fig = plt.figure()
#ax = Axes3D(fig)
##ax.scatter(hpoints[:,0][::ds], hpoints[:,1][::ds], hpoints[:,2][::ds],
##           c=new, linewidths=0)
#
#ax.scatter(nx, ny, nz, c=new,
#           cmap=plt.cm.jet, linewidths=0)
#ax.invert_zaxis()
#
##import volume_slicer as VS
##m = VS.VolumeSlicer(data=ndata[3])
##m.configure_traits()
#
#
#
#
#import plotly as py
#from plotly.graph_objs import Scatter3d, Data, Layout, Figure
#import plotly.graph_objs as go
#from plotly.tools import FigureFactory as FF
#
#scat = go.Data([go.Scatter3d(x=nx, y=ny, z=-nz,
#                    marker=dict(
#                    size=4,
#                    color=new,
#                    colorscale='Rainbow',
#                    opacity=0.5),
#                    line=dict(
#                    color='rgb(204, 204, 204)',
#                    width=0.01),
#                    )
#                    ])
#                    
#layout = go.Layout(xaxis=go.XAxis(title='x'), 
#                   yaxis=go.YAxis(title='y'),
#                   #zaxis=go.ZAxis(title='z')
#                   )
#                   
#
#fig = go.Figure(data=scat, layout=layout)
#py.offline.plot(fig, filename='graphics/NZVELx50_300.html',
#                auto_open=False)
#
##np.savetxt("models/nzvel_even_grid.txt", np.vstack([nx, ny, nz, new]).transpose(),
##           header="x,y,z,vp")
#           
#           
"""
Greys, YlGnBu, Greens, YlOrRd, Bluered, RdBu, Reds, Blues, Picnic,
 Rainbow, Portland, Jet, Hot, Blackbody, Earth, Electric, Viridis
"""


#def i_xyz_below_surface(xyzs, surf, s=None):
#    points = xyzs[:,0:2]
#    #values = xyz[:,2]
#    hpoints = (surf[0].ravel(), surf[1].ravel())    
#    on_surf = griddata(hpoints, surf[2].ravel(), points, method='linear')
#    
#    #Iz = np.where(-xyz[:,2] < on_surf)#, xyz[:,2], np.nan)
#    #return xyz[Iz]
#    argi = np.where(xyzs[:,2] > -on_surf)
#    #xyzs[argi[:,0]] = np.nan    
#    xyzs[argi] = np.nan
#    return argi
#    
#m = i_xyz_below_surface(xyzs, [hiklong, hiklatg, hik])


