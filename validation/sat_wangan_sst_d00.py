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

#%% ROMS dph =======================================================
root = '/data3/jhluna/work/ITRI_UCH_phdw/DATA/'
with nc.Dataset(root+'itri_uch_avg_d00_his_zlev.nc') as rootgrp:
    to = np.array(rootgrp.variables['SST'][:,1:-1,1:-1])
    lat = np.array(rootgrp.variables['lat'][1:-1,0])
    lon = np.array(rootgrp.variables['lon'][0,1:-1])
    to[np.where((to==0)|(to>35))[:]] = np.nan

    t = rootgrp.variables['time'][:]
    tu = rootgrp.variables['time'].units
    time_roms = np.array(nc.num2date(t,units=tu,calendar='standard'))
    time_roms = pd.to_datetime(time_roms, format='%Y-%m-%d %H:%M:%S.%f').to_numpy() + np.timedelta64(8,'h')

ilon = np.where((lon>=119.46) & (lon<=119.6))[0]
ilat = np.where((lat>=23.3296) & (lat<=23.4052))[0]
[ii,jj] = np.meshgrid(ilon,ilat)
sst_his_dph = np.nanmean(np.nanmean(to[:,jj,ii],axis=2),axis=1)
del to,rootgrp,ii,jj,ilon,ilat,t,tu

# Daily mean --------------------------------------------------------
days = np.unique(time_roms.astype('datetime64[D]'))
to_m = []
for day in days:
    to_m.append(np.nanmean(sst_his_dph[time_roms.astype('datetime64[D]') == day], axis=0))
sst_his_dph = np.array(to_m)
time_roms = days
del days,to_m,day

# Leap year ---------------------------------------------------------
mon = time_roms.astype('datetime64[M]').astype(int) % 12 + 1
days_numeric = time_roms.astype('datetime64[D]').astype(str)
day = np.array([int(d.split('-')[2]) for d in days_numeric])

mask = ~((mon == 2) & (day == 29))
sst_his_dph, time_roms = sst_his_dph[mask], time_roms[mask]
del days_numeric,mask,mon,day,time_roms

# Climatology -------------------------------------------------------
sst_his_dph = np.concatenate([np.full((2),np.nan), sst_his_dph, np.full((4),np.nan)])
sst_his_dph = np.reshape(sst_his_dph,(30,365)) 
sst_dph_per = np.percentile(sst_his_dph,[25,75],axis=0)
sst_his_dph = np.nanmean(sst_his_dph,axis=0)

#%% TaiESM-ROMS ====================================================
root = '/data3/jhluna/work/ITRI_UCH_phdw/DATA/'
with nc.Dataset(root+'taiesm_roms_zlev/ocean_d01_his_mon.nc') as rootgrp:
    to = np.array(rootgrp.variables['SST'][:])
    lat = np.array(rootgrp.variables['lat'][:])
    lon = np.array(rootgrp.variables['lon'][:])
    to[np.where((to==0)|(to>35))[:]] = np.nan

    t = rootgrp.variables['time'][:]
    tu = rootgrp.variables['time'].units
    time_roms = np.array(nc.num2date(t,units=tu,calendar='standard'))

ilon = np.where((lon>=119.46) & (lon<=119.6))[0]
ilat = np.where((lat>=23.3296) & (lat<=23.4052))[0]
[ii,jj] = np.meshgrid(ilon,ilat)
sst_his_rom = np.nanmean(np.nanmean(to[:,jj,ii],axis=2),axis=1)
del to,rootgrp,ii,jj,ilon,ilat,t,tu

# Climatology -------------------------------------------------------
sst_his_rom = np.reshape(sst_his_rom,(30,12)) 
sst_rom_per = np.percentile(sst_his_rom,[25,75],axis=0)
sst_his_rom = np.nanmean(sst_his_rom,axis=0)


#%% PLOT validation =================================================
x1 = np.arange(1,13,1)+.5
x2 = np.linspace(1,13,len(sst_his_dph))

fig = plt.figure(figsize=(7,4),facecolor='white')
l1,=plt.plot(x1,sst_his_rom,color='mediumpurple')
plt.fill_between(x1,sst_rom_per[0,:],sst_rom_per[1,:],alpha=0.3,facecolor='mediumpurple')
l2,=plt.plot(x2,sst_his_dph,color='tab:blue')
plt.fill_between(x2,sst_dph_per[0,:],sst_dph_per[1,:],alpha=0.3,facecolor='tab:blue')
l3,=plt.plot(x1,sst_modis,'--x',color='black')
l4,=plt.plot(x1,sst_hima,'-o',color='black')
plt.legend([l1,l2,l3,l4],['TaiESM-ROMS (1985-2014)','ROMS dph (1985-2014)',\
            'Aqua-MODIS (2002-2023)','Himawari-8 (2015-2023)'],loc='lower center')
plt.xlim([1,13])
plt.ylim([19,30.5])
plt.xlabel('Month',fontsize=13)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.title('Sea Surface Temperature (\N{DEGREE SIGN}C)',fontsize=15,\
             fontweight='bold',loc='left')
plt.text(10.3,30.8,'Wangan area',fontsize=13)
plt.grid()
plt.show()

#%% Quantitive validation
sst_dph_it = np.interp(x1,x2,sst_his_dph)
df = pd.DataFrame({'A': sst_dph_it, 'B': sst_his_rom, 'C': sst_modis, 'D': sst_hima})
print(df.corr())

print(np.sqrt(np.nanmean((sst_modis-sst_dph_it)**2)))
print(np.sqrt(np.nanmean((sst_modis-sst_his_rom)**2)))
print(np.sqrt(np.nanmean((sst_hima-sst_dph_it)**2)))
print(np.sqrt(np.nanmean((sst_hima-sst_his_rom)**2)))

