#%%
import pandas as pd
import numpy as np
import datetime
import netCDF4 as nc
from os import walk
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

#%% MODIS ==========================================================
dirnm = '/data3/jhluna/Aqua_Modis/'
fnmlist = sorted(next(walk(dirnm), (None, None, []))[2])

sst_modis = np.zeros(len(fnmlist))
n = 0
for fnm in fnmlist:
    print(fnm)
    rootgrp = nc.Dataset(dirnm+fnm)
    lon = np.array(rootgrp.variables['lon'])
    lat = np.array(rootgrp.variables['lat'])
    sst = np.array(rootgrp.variables['sst'])    # (lat,lon)

    ilon = np.where((lon>=119.46) & (lon<=119.6))[0]
    ilat = np.where((lat>=23.3296) & (lat<=23.4052))[0]
    [ii,jj] = np.meshgrid(ilon,ilat)
    sst_modis[n] = np.nanmean(sst[jj,ii])
    n = n+1
del dirnm,fnmlist,fnm,lon,lat,sst,ilon,ilat,ii,jj,n

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

    ilon = np.where((lon>=119.46) & (lon<=119.6))[0]
    ilat = np.where((lat>=23.3296) & (lat<=23.4052))[0]
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
    to[np.where((to==0)|(to>35))[:]] = np.nan
    sst_roms = np.nanmean(np.nanmean(to,axis=2),axis=1)
del to,rootgrp

#%% ROMS dph =======================================================
# Historical
with nc.Dataset(root+'itri_uch_avg_d00c_his_zlev.nc') as rootgrp:
    to = np.array(rootgrp.variables['SST'][:,1:-1,1:-1])
    lat = np.array(rootgrp.variables['lat'][1:-1,1])
    lon = np.array(rootgrp.variables['lon'][1,1:-1])
to[np.where((to==0)|(to>35))[:]] = np.nan

ilon = np.where((lon>=119.46) & (lon<=119.6))[0]
ilat = np.where((lat>=23.3296) & (lat<=23.4052))[0]
[ii,jj] = np.meshgrid(ilon,ilat)
sst_his_dph = np.nanmean(np.nanmean(to[:,jj,ii],axis=2),axis=1)
del to,rootgrp

# SSP245
with nc.Dataset(root+'itri_uch_avg_d00c_ssp_zlev.nc') as rootgrp:
    to = np.array(rootgrp.variables['SST'][:,1:-1,1:-1])
to[np.where((to==0)|(to>35))[:]] = np.nan
[ii,jj] = np.meshgrid(ilon,ilat)
sst_ssp_dph = np.nanmean(np.nanmean(to[:,jj,ii],axis=2),axis=1)
del to,rootgrp,ii,jj

#%% PLOT validation =================================================
fig = plt.figure(figsize=(7,4),facecolor='white')
l1,=plt.plot(np.linspace(1,12+27/31,len(sst_his_dph)),sst_his_dph,color='tab:blue')
l2,=plt.plot(np.linspace(1,12+27/31,len(sst_roms)),sst_roms,color='tab:orange')
l3,=plt.plot(np.arange(1,13,1)+.5,sst_modis,'--x',color='black')
l3,=plt.plot(np.arange(1,13,1)+.5,sst_hima,'-o',color='black')
plt.legend(['ROMS dph (1985-2014)','ROMS d01 (1985-2014)',\
            'Aqua-MODIS (2002-2023)','Himawari-8 (2015-2023)'],loc='lower center')
plt.xlim([1,13])
plt.ylim([19,30])
plt.xlabel('Month',fontsize=13)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.title('Sea Surface Temperature (\N{DEGREE SIGN}C)',fontsize=15,\
             fontweight='bold',loc='left')
plt.text(10.3,30.3,'Wangan area',fontsize=13)
plt.grid()
plt.show()

#%% Quantitive validation
arr1 = np.linspace(1,12+27/31,len(sst_his_dph))
arr2 = np.arange(1,13,1)+.5
sst_dph_it = np.interp(arr2,arr1,sst_his_dph)
sst_d01_it = np.interp(arr2,arr1,sst_roms)
df = pd.DataFrame({'A': sst_dph_it, 'B': sst_d01_it, 'C': sst_modis, 'D': sst_hima})
print(df.corr())

print(np.sqrt(np.nanmean((sst_modis-sst_dph_it)**2)))
print(np.sqrt(np.nanmean((sst_modis-sst_d01_it)**2)))
print(np.sqrt(np.nanmean((sst_hima-sst_dph_it)**2)))
print(np.sqrt(np.nanmean((sst_hima-sst_d01_it)**2)))

#%% PLOT difference line ============================================
fig = plt.figure(figsize=(7,4),facecolor='white')
ax = fig.add_subplot(111)
ax2 = ax.twinx()

l1,=ax.plot(np.linspace(1,12+27/31,len(sst_his_dph)),sst_his_dph,color='black')
l2,=ax.plot(np.linspace(1,12+27/31,len(sst_ssp_dph)),sst_ssp_dph,color='red')
ax.set_xlim([1,13])
ax.set_ylim([19,30])
ax.set_xlabel('Month',fontsize=13)
ax.set_ylabel('Mean',fontsize=13)
ax.set_xticks(ticks=np.arange(2,13,2))
ax.set_yticks(ticks=np.arange(20,31,2))
ax.grid()

l3,=ax2.plot(np.linspace(1,12+27/31,len(sst_ssp_dph)),sst_ssp_dph-sst_his_dph,color='gray')
ax2.set_xlim([1,13])
ax2.set_ylim([0,2.2])
ax2.set_xlabel('Month',fontsize=13)
ax2.set_ylabel('Difference',fontsize=13)
ax2.set_yticks(ticks=np.arange(.2,2.3,.4))
ax2.yaxis.label.set_color(l3.get_color())
ax2.tick_params(axis='y',colors=l3.get_color())

plt.legend([l1,l2,l3],['Historical (1985-2014)','SSP2-4.5 (2040-2069)','Difference'],loc='upper left')
ax.set_title('Sea Surface Temperature (\N{DEGREE SIGN}C)',fontsize=16,\
             fontweight='bold',loc='left')
ax.text(10.3,30.3,'Wangan area',fontsize=13)
plt.show()
