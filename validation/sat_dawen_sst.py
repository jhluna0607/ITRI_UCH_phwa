#%%
import pandas as pd
import numpy as np
import datetime
import netCDF4 as nc
import cmaps
from os import walk
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.patches as patches

#%% Himawari8 ======================================================
dirnm = '/data3/jhluna/Himawari8/'
fnmlist = sorted(next(walk(dirnm), (None, None, []))[2])

sst_hima = np.zeros(len(fnmlist))
n = 0
for fnm in fnmlist:
    print(fnm)
    rootgrp = nc.Dataset(dirnm+fnm)
    lon = np.array(rootgrp.variables['lon'])
    lat = np.array(rootgrp.variables['lat'])
    sst = np.array(rootgrp.variables['sea_surface_temperature'])-273.15

    ilon = np.where((lon>=119.5280538) & (lon<=119.5649639))[0]
    ilat = np.where((lat>=23.3385435) & (lat<=23.3607716))[0]
    [ii,jj] = np.meshgrid(ilon,ilat)
    sst_hima[n] = np.nanmean(sst[jj,ii])
    n = n+1

sst_mon = np.zeros(12)
for m in range(12):
    sst_mon[m] = np.nanmean(sst_hima[m::12])
sst_hima = np.roll(sst_mon,7)
del dirnm,fnmlist,fnm,lon,lat,sst,ilon,ilat,ii,jj,n,sst_mon,m

#%% ROMS d01 =======================================================
root = '/data3/jhluna/work/ITRI_UCH_phdw/DATA/'
with nc.Dataset(root+'itri_uch_avg_d01_his_zlev.nc') as rootgrp:
    to = np.array(rootgrp.variables['SST'][:,1:-1,1:-1])
    lat = np.array(rootgrp.variables['lat'][1:-1,1])
    lon = np.array(rootgrp.variables['lon'][1,1:-1])
    to[np.where((to==0)|(to>35))[:]] = np.nan

ilon = np.where((lon>=119.5280538) & (lon<=119.5649639))[0]
ilat = np.where((lat>=23.3385435) & (lat<=23.3607716))[0]
[ii,jj] = np.meshgrid(ilon,ilat)
sst_roms_d01 = np.nanmean(np.nanmean(to[:,jj,ii],axis=2),axis=1)
del to,rootgrp

#%% ROMS d02 =======================================================
with nc.Dataset(root+'itri_uch_avg_d02_his_zlev.nc') as rootgrp:
    to = np.array(rootgrp.variables['SST'][:,1:-1,1:-1])
    to[np.where((to==0)|(to>35))[:]] = np.nan
sst_roms_d02 = np.nanmean(np.nanmean(to,axis=2),axis=1)
del to,rootgrp

#%% PLOT ==========================================================
fig = plt.figure(figsize=(7,4),facecolor='white')
ax = fig.add_subplot(111)
ax.plot(np.linspace(1,12+27/31,len(sst_roms_d01)),sst_roms_d01,color='tab:orange')
ax.plot(np.linspace(1,12+27/31,len(sst_roms_d02)),sst_roms_d02,color='tab:green')
ax.plot(np.arange(1,13,1)+.5,sst_hima,'-o',color='black')
plt.legend(['ROMS d01 (1985-2014)','ROMS d02 (1985-2014)','Himawari-8 (2015-2023)'],loc='lower center')
ax.set_xlim([1,13])
ax.set_ylim([19,30])
ax.set_xlabel('Month',fontsize=13)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
ax.set_title('Sea Surface Temperature (\N{DEGREE SIGN}C)',fontsize=15,\
             fontweight='bold',loc='left')
ax.text(10.5,30.3,'Dawen area',fontsize=13)
plt.grid()
#fig.savefig('d01/sat_dawen_sst.png',bbox_inches='tight',dpi=300)
plt.show()

# Quantitive validation
arr1 = np.linspace(1,12+27/31,len(sst_roms_d01))
arr2 = np.arange(1,13,1)+.5
sst_d01_it = np.interp(arr2,arr1,sst_roms_d01)
sst_d02_it = np.interp(arr2,arr1,sst_roms_d02)
df = pd.DataFrame({'A': sst_d01_it, 'B': sst_d02_it, 'D': sst_hima})
print(df.corr())

print(np.sqrt(np.nanmean((sst_hima-sst_d01_it)**2)))
print(np.sqrt(np.nanmean((sst_hima-sst_d02_it)**2)))
