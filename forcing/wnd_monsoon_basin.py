#%%
import pandas as pd
import numpy as np
import cmaps
import datetime
import netCDF4 as nc
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import cartopy as cart
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import matplotlib.patches as patches

def curl(lat,lon,u,v):
    s = u.shape
    deg2m = 111200
    latm = (lat[2,1]-lat[1,1])*deg2m
    lonm = (lon[1,2]-lon[1,1])*np.cos(np.deg2rad(lat))*deg2m
    dudy = np.gradient (u,latm,axis=0)
    dvdx = np.gradient (v,axis=1)/lonm
    curlZ = dvdx-dudy
    return curlZ
    
#%% TaiESM-WRF ======================================================
# Initialize
his, ssp = {}, {}
fnm = ['/home/jhluna/ROMS/forcing/WRF_historical/gom_wind_TAIESM_his1WRFc.nc', \
       '/home/jhluna/ROMS/forcing/WRF_ssp245_2040_2069/gom_wind_TAIESM_ssp2451WRFc.nc']

# Define file and read variables
for fn, data_dict in enumerate([his, ssp]):
    uwnds, vwnds, curlzs = [], [], []
    with nc.Dataset(fnm[fn]) as rootgrp:
        uwnd = rootgrp.variables['Uwind'][31:,:,:]
        vwnd = rootgrp.variables['Vwind'][31:,:,:]
        lat_wrf = np.arange(4.5,49.6,.25)
        lon_wrf = np.arange(95.5,157.1,.25)
        [lon_wrf,lat_wrf] = np.meshgrid(lon_wrf,lat_wrf)

    # Calculate wind curl for all time steps
    curlz = np.array([curl(lat_wrf,lon_wrf,uwnd[t,:,:],vwnd[t,:,:]) for t in range(uwnd.shape[0])])

    # Define month
    st = datetime(2013, 1, 1)
    en = datetime(2013, 12, 31)
    time = np.arange(st, en, timedelta(days=1)).astype('datetime64[D]')
    time = pd.to_datetime(time)
    mon = time.month

    # Select season
    seasons = {'DJF': [12,1,2], 'MAM': [3,4,5], 'JJA': [6,7,8], 'SON': [9,10,11]}
    for idx, (season_name, months) in enumerate(seasons.items()):
        if idx != 0:
            loc = np.where((mon>=months[0]) & (mon<=months[2]))[0]
        else:
            loc = np.where((mon>=months[0]) | (mon<=months[2]))[0]
        uwnds.append(np.nanmean(uwnd[loc,:,:],0))
        vwnds.append(np.nanmean(vwnd[loc,:,:],0))
        curlzs.append(np.nanmean(curlz[loc,:,:],0))

    data_dict["uwnds"] = np.asarray(uwnds)
    data_dict["vwnds"] = np.asarray(vwnds)
    data_dict["wpds"] = np.sqrt(data_dict["uwnds"]**2+data_dict["vwnds"]**2)
    data_dict["curlzs"] = np.asarray(curlzs)

    # Calculate overall mean
    #uwnd = np.nanmean(uwnd,axis=0)
    #vwnd = np.nanmean(vwnd,axis=0)
    #curlZ_wrf = np.nanmean(curlZ_wrf,axis=0)
    del st,en,fn,data_dict,mon,rootgrp,time,idx,loc,months,season_name,seasons
    del uwnd,vwnd,curlz,uwnds,vwnds,curlzs

#%% Plot seasonal wind speed (local monsoon) ========================
season_name = ['Winter','Summer']
scenario_name = ['1985-2014','2040-2069']
sea_num = [0,2]
d, m = 3, 15
cmap = cmaps.WhiteBlueGreenYellowRed
lon_ticks = np.arange(-63,-56,3)
lat_ticks = np.arange(20,27,3)

proj = ccrs.PlateCarree(central_longitude=180)
fig,ax= plt.subplots(figsize=(10,6),nrows=2,ncols=3,facecolor='white',\
                     subplot_kw={'projection':proj})
plt.subplots_adjust(wspace=.02,hspace=.15)
ax = ax.flatten()

for s, seast in enumerate(season_name):
    for sc, data_dict in enumerate([his, ssp]):
        f = s*3+sc
        pc1 = ax[f].pcolormesh(lon_wrf,lat_wrf,data_dict['wpds'][sea_num[s],:,:],vmin=0,vmax=m,\
                               cmap=cmap,transform=ccrs.PlateCarree())
        Q = ax[f].quiver(lon_wrf[::d,::d],lat_wrf[::d,::d],\
                         data_dict['uwnds'][sea_num[s],::d,::d],data_dict['vwnds'][sea_num[s],::d,::d],\
                         width=0.005,scale=100,transform=ccrs.PlateCarree())
        #ax[f].set_extent([-64,-56,19,27],crs=proj)
        ax[f].set_extent([-67,-55,16,27],crs=proj)
        ax[f].coastlines(resolution='50m')
        ax[f].quiverkey(Q,.2,.92,10,'10 m/s',labelpos='S',zorder=3)
        ax[f].gridlines(crs=ccrs.PlateCarree(),draw_labels=False,color='gray',alpha=0.2)

        if s == 1:
            ax[f].set_xticks(lon_ticks,crs=proj)
            ax[f].set_xticklabels(('117\N{DEGREE SIGN}E','120\N{DEGREE SIGN}E','123\N{DEGREE SIGN}E'))
        else:
            ax[f].set_title(scenario_name[sc],fontsize=15)
        if sc == 0:
            ax[f].set_yticks(lat_ticks,crs=proj)
            ax[f].set_yticklabels(('20\N{DEGREE SIGN}N','23\N{DEGREE SIGN}N','26\N{DEGREE SIGN}N'))
            ax[f].text(-0.25, 0.5, seast, va='bottom', ha='center',\
                        rotation='vertical', rotation_mode='anchor',\
                        transform=ax[f].transAxes,fontsize=15)

    f = 2+s*3
    tmp = ssp['wpds'][sea_num[s],:,:]-his['wpds'][sea_num[s],:,:]
    tmpu = ssp['uwnds'][sea_num[s],:,:]-his['uwnds'][sea_num[s],:,:]
    tmpv = ssp['vwnds'][sea_num[s],:,:]-his['vwnds'][sea_num[s],:,:]
    pc2 = ax[f].pcolormesh(lon_wrf,lat_wrf,tmp,vmin=-1,vmax=1,\
                        cmap='RdBu_r',transform=ccrs.PlateCarree())
    Q = ax[f].quiver(lon_wrf[::d,::d],lat_wrf[::d,::d],\
                     tmpu[::d,::d],tmpv[::d,::d],\
                     width=0.005,scale=8,transform=ccrs.PlateCarree())
    #ax[f].set_extent([-64,-56,19,27],crs=proj)
    ax[f].set_extent([-67,-55,16,27],crs=proj)
    ax[f].coastlines(resolution='50m')
    ax[f].quiverkey(Q,.2,.92,1,'1 m/s',labelpos='S',zorder=3)
    ax[f].gridlines(crs=ccrs.PlateCarree(),draw_labels=False,color='gray',alpha=0.2)
    if s == 1:
        ax[f].set_xticks(lon_ticks,crs=proj)
        ax[f].set_xticklabels(('117\N{DEGREE SIGN}E','120\N{DEGREE SIGN}E','123\N{DEGREE SIGN}E'))
    else:
        ax[f].set_title('Difference',fontsize=15)

cbar_ax = fig.add_axes([0.16,0.02,0.46,0.02])
cbar = fig.colorbar(pc1,cax=cbar_ax,ticks=np.arange(0,16,3),orientation='horizontal',extend='max')
cbar.set_label('m / s')

cbar_ax = fig.add_axes([0.67,0.02,0.2,0.02])
cbar = fig.colorbar(pc2,cax=cbar_ax,ticks=np.arange(-1,1.2,.5),orientation='horizontal',extend='both')
cbar.set_label('m / s')
plt.show()



#%% Plot seasonal wind speed (basin scale) ==========================
season_name = ['Winter','Summer']
scenario_name = ['1985-2014','2040-2069']
sea_num = [0,2]
d, m, md = 10, 2e-5, 3e-6
lon_ticks = np.arange(-65,-30,10)
lat_ticks = np.arange(15,40,10)

proj = ccrs.PlateCarree(central_longitude=180)
fig,ax= plt.subplots(figsize=(10,5),nrows=2,ncols=3,facecolor='white',\
                     subplot_kw={'projection':proj})
plt.subplots_adjust(wspace=.1,hspace=.1)
ax = ax.flatten()

for s, seast in enumerate(season_name):
    for sc, data_dict in enumerate([his, ssp]):
        f = s*3+sc
        pc1 = ax[f].pcolormesh(lon_wrf,lat_wrf,data_dict['curlzs'][sea_num[s],:,:],vmin=-m,vmax=m,\
                               cmap='RdBu_r',transform=ccrs.PlateCarree())
        Q = ax[f].quiver(lon_wrf[::d,::d],lat_wrf[::d,::d],\
                         data_dict['uwnds'][sea_num[s],::d,::d],data_dict['vwnds'][sea_num[s],::d,::d],\
                         width=0.005,scale=100,transform=ccrs.PlateCarree())
        ax[f].set_extent([-70,-30,10,40],crs=proj)
        ax[f].coastlines(resolution='50m')
        rect = patches.Rectangle((-70,33),9,7,edgecolor='k',facecolor='w')
        ax[f].add_patch(rect)
        ax[f].quiverkey(Q,.11,.93,10,'10 m/s',labelpos='S',zorder=3)
        ax[f].gridlines(crs=ccrs.PlateCarree(),draw_labels=False,color='gray',alpha=0.2)

        if s == 1:
            ax[f].set_xticks(lon_ticks,crs=proj)
            ax[f].set_xticklabels(('115\N{DEGREE SIGN}E','125\N{DEGREE SIGN}E','135\N{DEGREE SIGN}E',\
                                   '145\N{DEGREE SIGN}E'))
        else:
            ax[f].set_title(scenario_name[sc],fontsize=15)
        if sc == 0:
            ax[f].set_yticks(lat_ticks,crs=proj)
            ax[f].set_yticklabels(('15\N{DEGREE SIGN}N','25\N{DEGREE SIGN}N','35\N{DEGREE SIGN}N'))
            ax[f].text(-0.25, 0.5, seast, va='bottom', ha='center',\
                        rotation='vertical', rotation_mode='anchor',\
                        transform=ax[f].transAxes,fontsize=15)

    f = 2+s*3
    tmp = ssp['wpds'][sea_num[s],:,:]-his['wpds'][sea_num[s],:,:]
    tmpc = ssp['curlzs'][sea_num[s],:,:]-his['curlzs'][sea_num[s],:,:]
    tmpu = ssp['uwnds'][sea_num[s],:,:]-his['uwnds'][sea_num[s],:,:]
    tmpv = ssp['vwnds'][sea_num[s],:,:]-his['vwnds'][sea_num[s],:,:]
    pc2 = ax[f].pcolormesh(lon_wrf,lat_wrf,tmpc,vmin=-md,vmax=md,\
                        cmap='PuOr_r',transform=ccrs.PlateCarree())
    Q = ax[f].quiver(lon_wrf[::d,::d],lat_wrf[::d,::d],\
                     tmpu[::d,::d],tmpv[::d,::d],\
                     width=0.005,scale=10,transform=ccrs.PlateCarree())
    ax[f].set_extent([-70,-30,10,40],crs=proj)
    ax[f].coastlines(resolution='50m')
    rect = patches.Rectangle((-70,33),9,7,edgecolor='k',facecolor='w')
    ax[f].add_patch(rect)
    ax[f].quiverkey(Q,.11,.93,1,'1 m/s',labelpos='S',zorder=3)
    ax[f].gridlines(crs=ccrs.PlateCarree(),draw_labels=False,color='gray',alpha=0.2)
    if s == 1:
        ax[f].set_xticks(lon_ticks,crs=proj)
        ax[f].set_xticklabels(('115\N{DEGREE SIGN}E','125\N{DEGREE SIGN}E','135\N{DEGREE SIGN}E',\
                                '145\N{DEGREE SIGN}E'))
    else:
        ax[f].set_title('Difference',fontsize=15)

cbar_ax = fig.add_axes([0.15,0.02,0.46,0.02])
cbar = fig.colorbar(pc1,cax=cbar_ax,ticks=np.arange(-m,m+1e-6,m*.2),orientation='horizontal',extend='max')
cbar.set_label('1 / s')

cbar_ax = fig.add_axes([0.68,0.02,0.2,0.02])
cbar = fig.colorbar(pc2,cax=cbar_ax,ticks=np.arange(-md,md+1e-7,md*.5),orientation='horizontal',extend='both')
cbar.set_label('1 / s')
plt.show()
# %%
