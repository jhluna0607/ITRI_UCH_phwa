#%%
import pandas as pd
import numpy as np
import datetime
import netCDF4 as nc
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)

#%% Read data =======================================================
rootgrp = nc.Dataset('/data3/jhluna/work/ITRI_penghu/d00/tide_tpxo9_dph.nc')
lat = np.array(rootgrp.variables['lat_rho'][1:-1:8,1:-1:8])
lon = np.array(rootgrp.variables['lon_rho'][1:-1:8,1:-1:8])
cangle = np.array(rootgrp.variables['tide_Cangle'][:,1:-1:8,1:-1:8])
cmin = np.array(rootgrp.variables['tide_Cmin'][:,1:-1:8,1:-1:8])
cmax = np.array(rootgrp.variables['tide_Cmax'][:,1:-1:8,1:-1:8])

latm = np.array(rootgrp.variables['lat_rho'][1:-1,1:-1])
lonm = np.array(rootgrp.variables['lon_rho'][1:-1,1:-1])
mask = np.array(rootgrp.variables['mask_rho'][1:-1,1:-1])
mask[np.where(mask==1)[:]] = np.nan

#%% Make tidal ellipse ==============================================
xe = np.zeros((2,7,7,500))
ye = np.zeros((2,7,7,500))

tide = [3,1]    # M2, K1 for v.9
#tide = [0,4]    # M2, K1 vor v.7
for t in range(2):
    for i in range(len(lat)):
       for j in range(len(lon)):

            tidec = tide[t]
            foo = np.linspace(0,2*np.pi,500)    # degree of a circle
            x = cmax[tidec,i,j]*np.cos(foo)
            y = cmin[tidec,i,j]*np.sin(foo)

            # turn the angle of ellipse
            [r,theta] = cart2pol(x,y)
            [xr,yr] = pol2cart(r,theta+cangle[tidec,i,j]*np.pi/180)     
            xe[t,i,j,:] = xr
            ye[t,i,j,:] = yr

# Extreme which closed to land
xe[:,3:5,3,:] = np.nan
ye[:,3:5,3,:] = np.nan

# For legend
xe[:,-1,0:2,:] = np.nan
ye[:,-1,0:2,:] = np.nan

#%% Plot ============================================================
fig,ax = plt.subplots(figsize=(12,6),nrows=1,ncols=2,facecolor='white')
plt.subplots_adjust(wspace=.1)

currentaxis = fig.gca()

f = 0.05
for i in range(len(lat)):
    for j in range(len(lon)):
        cn = ax[0].plot(xe[0,i,j,:]*f+lon[i,j],ye[0,i,j,:]*f+lat[i,j],'-k')
        ax[0].pcolormesh(lonm,latm,mask,cmap='gray',vmin=-1,vmax=1)
        ax[0].set_title('M2',fontsize=18,fontweight='semibold')
        ax[0].grid()
xl = 0.5*np.cos(foo)
yl = 0.5*np.sin(foo)
ax[0].add_patch(patches.Rectangle((118.9,23.85),0.4,0.15,edgecolor='gainsboro',facecolor='gainsboro'))
ax[0].plot(xl*f+lon[-1,0],yl*f+23.91,'-k')
ax[0].text(119.08,23.9,'0.5 m/s',fontsize=14)
ax[0].set_xlim([118.95,120.05])
ax[0].set_ylim([22.95,23.98])
ax[0].tick_params(axis='both',labelsize=14)


f = 0.3
for i in range(len(lat)):
    for j in range(len(lon)):
        cn = ax[1].plot(xe[1,i,j,:]*f+lon[i,j],ye[1,i,j,:]*f+lat[i,j],'-b')
        ax[1].pcolormesh(lonm,latm,mask,cmap='gray',vmin=-1,vmax=1)
        ax[1].set_yticklabels([])
        ax[1].set_title('K1',fontsize=18,fontweight='semibold')
        ax[1].grid()
xl = 0.1*np.cos(foo)
yl = 0.1*np.sin(foo)
ax[1].add_patch(patches.Rectangle((118.9,23.85),0.4,0.15,edgecolor='gainsboro',facecolor='gainsboro'))
ax[1].plot(xl*f+lon[-1,0],yl*f+23.91,'-b')
ax[1].text(119.08,23.9,'0.1 m/s',fontsize=14,color='b')
ax[1].set_xlim([118.95,120.05])
ax[1].set_ylim([22.95,23.98])
ax[1].tick_params(axis='both',labelsize=14)

#fig.savefig('sst_sfc10.png',bbox_inches='tight',dpi=300)
plt.show()


