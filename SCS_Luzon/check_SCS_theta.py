#%% Plot temp and salinity in SCS
import glob
import cmaps
import numpy as np
import netCDF4 as nc
import cartopy as cart
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

LON, LAT, TEMP = {}, {}, {}
lon_lim = [105, 125]
lat_lim = [5, 27]
TITLE = ['HYCOM (1994-2015)','TaiESM1 (1985-2014)','TaiESM-ROMS (1985-2014)']

#%% HYCOM ===========================================================
fname = '/data3/data/HYCOM/monthly_1994_2015.nc'

with nc.Dataset(fname) as rootgrp:

    lat = rootgrp.variables['lat'][:]
    lon = rootgrp.variables['lon'][:]
    lev = rootgrp.variables['depth'][:]

    ilon = np.where((lon>=lon_lim[0])&(lon<=lon_lim[1]))[0]
    ilat = np.where((lat>=lat_lim[0])&(lat<=lat_lim[1]))[0]
    ilev = np.argmin(np.abs(lev-1000))
    LON['hycom'], LAT['hycom'] = lon[ilon], lat[ilat]

    temp = rootgrp.variables['water_temp'][:,ilev,ilat,ilon]    # (time,lat,lon)
    TEMP['hycom'] = np.nanmean(temp,axis=0)

print(lev[ilev])
del lon,lat,lev,ilon,ilat,ilev,temp,fname,rootgrp

#%% TaiESM1 =========================================================
fpnm = sorted(glob.glob('/data3/CMIP6/TaiESM1/historical/ocean/thetao*'))
fpnm = fpnm[135:]

with nc.Dataset(fpnm[0]) as rootgrp:

    lon = rootgrp.variables['longitude'][:]     # (lat,lon)
    lat = rootgrp.variables['latitude'][:]
    lev = rootgrp.variables['lev'][:]
    ilon1, ilon2, ilat1, ilat2 = [], [], [], []

    for i in range(lon.shape[0]):
        tmp = lon[i,:]
        try:
            ilon1.append(np.where(tmp == np.nanmax(tmp[np.where(tmp<=lon_lim[0])[0]]))[0])
            ilon2.append(np.where(tmp == np.nanmin(tmp[np.where(tmp>=lon_lim[1])[0]]))[0])
        except:
            pass

    for i in range(lat.shape[1]):
        tmp = lat[:,i]
        try:
            ilat1.append(np.where(tmp == np.nanmax(tmp[np.where(tmp<=lat_lim[0])[0]]))[0])
            ilat2.append(np.where(tmp == np.nanmin(tmp[np.where(tmp>=lat_lim[1])[0]]))[0])
        except:
            pass

    ilon = [np.nanmin(ilon1), np.nanmax(ilon2)]
    ilat = [np.nanmin(ilat1), np.nanmax(ilat2)]
    ilev = np.argmin(np.abs(lev-1000))

LON['taiesm'] = lon[ilat[0]:ilat[1]+1,ilon[0]:ilon[1]+1]
LAT['taiesm'] = lat[ilat[0]:ilat[1]+1,ilon[0]:ilon[1]+1]
temp = []
for fname in fpnm:
    with nc.Dataset(fname) as rootgrp:
        temp.append(rootgrp.variables['thetao'][:,ilev,ilat[0]:ilat[1]+1,ilon[0]:ilon[1]+1])
temp = np.nanmean(np.concatenate(temp,axis=0),axis=0)
temp[temp>40] = np.nan
TEMP['taiesm'] = temp

print(lev[ilev])
del i,fpnm,fname,lon,lat,lev,ilon1,ilon2,ilat1,ilat2,tmp,ilon,ilat,ilev,temp,rootgrp

#%% TaiESM-ROMS =====================================================
root = '/data3/jhluna/work/ITRI_UCH_phdw/DATA/'
fname  = root+'taiesm_roms_zlev/ocean_d01_his_1000_mon.nc'

with nc.Dataset(fname) as rootgrp:

    lon = rootgrp.variables['lon'][:]
    lat = rootgrp.variables['lat'][:]
    lev = -rootgrp.variables['lev'][:]

    ilon = np.where((lon>=lon_lim[0])&(lon<=lon_lim[1]))[0]
    ilat = np.where((lat>=lat_lim[0])&(lat<=lat_lim[1]))[0]
    LON['roms'], LAT['roms'] = lon[ilon], lat[ilat]

    temp = rootgrp.variables['to'][:,:,ilat,ilon]      # (time,lat,lon)
    TEMP['roms'] = np.squeeze(np.nanmean(temp,axis=0))

del lon,lat,lev,ilon,ilat,temp,fname,rootgrp


#%% Plot ============================================================
# Whole figure settings
lon_ticks = np.arange(-70,-54,5)
lat_ticks = np.arange(5,27,5)

land = cart.feature.NaturalEarthFeature('physical', 'land', scale='10m',\
                                         edgecolor='k', facecolor='#AAAAAA',zorder=1)
proj = ccrs.PlateCarree(central_longitude=180)
fig,ax = plt.subplots(figsize=(10,6), nrows=1, ncols=3, facecolor='white',\
                      subplot_kw={'projection':proj})
plt.subplots_adjust(wspace=.15, hspace=.15)
ax = ax.flatten()

vmin, vmax = 3.5, 7.5
clev = np.arange(vmin,vmax+.1,.2)

for m, model in enumerate(TEMP.keys()):
    ax[m].set_extent([-75,-55,5,27],crs=proj)
    ax[m].add_feature(land)
    cn = ax[m].contourf(LON[model], LAT[model], TEMP[model], cmap='turbo', vmin=vmin, vmax=vmax,\
                        levels=clev, extend='both', transform=ccrs.PlateCarree())
    ax[m].set_title(TITLE[m])

    ax[m].set_xticks(lon_ticks,crs=proj)
    ax[m].set_xticklabels([f'{t+180}\N{DEGREE SIGN}E' for t in lon_ticks])
    if m == 0:
        ax[m].set_yticks(lat_ticks,crs=proj)
        ax[m].set_yticklabels([f'{t}\N{DEGREE SIGN}N' for t in lat_ticks])


cbar_ax = fig.add_axes([0.94,0.29,0.01,0.4])
cbar = fig.colorbar(cn, ticks=np.arange(vmin,vmax+.1,1), cax=cbar_ax,\
                    orientation='vertical', extend='both')
cbar.set_label('\N{DEGREE SIGN}C')
# %%
