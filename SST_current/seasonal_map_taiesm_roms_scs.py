#%% Plot SST and Current in SCS and Taiwan from TaiESM-ROMS
import numpy as np
import netCDF4 as nc
import cartopy as cart
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

#%% =================================================================
# Data Process
# ===================================================================
# Read ncfile -------------------------------------------------------
root = '/data3/jhluna/work/ITRI_UCH_phdw/DATA/'
fnm = [root+'taiesm_roms_zlev/ocean_d01_his_mon.nc',\
       root+'taiesm_roms_zlev/ocean_d01_ssp_mon.nc']
scen = ['HIS', 'SSP']
sst, uo, vo, sss, time = {}, {}, {}, {}, {}

for f, fname in enumerate(fnm):
    with nc.Dataset(fname) as rootgrp:
        lon  = rootgrp.variables['lon'][:]
        lat  = rootgrp.variables['lat'][:]
        sst[scen[f]] = rootgrp.variables['SST'][:]      # (time,lat,lon)
        sss[scen[f]] = rootgrp.variables['so'][:,0,:,:]
        uo[scen[f]] = rootgrp.variables['uo'][:,0,:,:]
        vo[scen[f]] = rootgrp.variables['vo'][:,0,:,:]
        t = rootgrp.variables['time'][:]
        tu = rootgrp.variables['time'].units
        time[scen[f]] = nc.num2date(t,units=tu,calendar='standard')
del f,fname,t,tu,rootgrp

# Select month ------------------------------------------------------
SST, UO, VO, SSS = {}, {}, {}, {}
seas = ['DJF', 'JJA']
st_yr = 0

get_month = np.vectorize(lambda date: date.month)
mon = get_month(time['HIS'].compressed())
get_year = np.vectorize(lambda date: date.year)
yr = get_year(time['HIS'].compressed())
it = {'DJF': np.where(((mon>=12)|(mon<=2))&(yr>=1985+st_yr))[0], 
      'JJA': np.where(((mon>=6)&(mon<=8))&(yr>=1985+st_yr))[0]}

for se, sea in enumerate(seas):
    for sc, sce in enumerate(scen):
        SST[f"{sce}_{sea}"] = np.nanmean(sst[sce][it[sea],:,:],axis=0)
        SSS[f"{sce}_{sea}"] = np.nanmean(sss[sce][it[sea],:,:],axis=0)
        UO[f"{sce}_{sea}"] = np.nanmean(uo[sce][it[sea],:,:],axis=0)
        VO[f"{sce}_{sea}"] = np.nanmean(vo[sce][it[sea],:,:],axis=0)
    
    SST[f"DIFF_{sea}"] = SST[f"SSP_{sea}"]-SST[f"HIS_{sea}"]
    SSS[f"DIFF_{sea}"] = SSS[f"SSP_{sea}"]-SSS[f"HIS_{sea}"]
    UO[f"DIFF_{sea}"] = UO[f"SSP_{sea}"]-UO[f"HIS_{sea}"]
    VO[f"DIFF_{sea}"] = VO[f"SSP_{sea}"]-VO[f"HIS_{sea}"]
del sc,sce,se,sea,get_month,mon,get_year,yr,it,sst,uo,vo

#%% =================================================================
# Plot SST & current
# ===================================================================
# Whole figure settings ---------------------------------------------
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
land = cart.feature.NaturalEarthFeature('physical', 'land', scale='10m',\
                                         edgecolor='k', facecolor='#AAAAAA',zorder=1)
proj = ccrs.PlateCarree(central_longitude=180)
fig,ax = plt.subplots(figsize=(10,6), nrows=2, ncols=3, facecolor='white',\
                      subplot_kw={'projection':proj})
plt.subplots_adjust(wspace=.02, hspace=.15)
ax = ax.flatten()

# Plot subplots -----------------------------------------------------
for i, s in enumerate(SST.keys()):

    # Subplots' settings
    if i == 2 or i == 5:
        vmin, vmax = 0, 2
        clev = np.arange(vmin,vmax+.1,.2)
        cm = 'YlOrRd'
        dv, sc = 5, 3
    else:
        vmin, vmax = 18, 32
        clev = np.arange(vmin,vmax+.5,1)
        cm = 'turbo'
        dv, sc = 5, 5

    # Plotting
    #ax[i].set_extent([-64,-56,19,27],crs=proj)
    ax[i].set_extent([-67,-55,16,27],crs=proj)
    ax[i].add_feature(land)
    cn = ax[i].contourf(lon, lat, SST[s], cmap=cm, vmin=vmin, vmax=vmax, levels=clev,\
                        extend='both', transform=ccrs.PlateCarree())
    Q = ax[i].quiver(lon[7::dv], lat[7::dv], UO[s][7::dv,7::dv], VO[s][7::dv,7::dv],\
                     scale=sc, width=4e-3, transform=ccrs.PlateCarree())
    
    # Ticks
    gl = ax[i].gridlines(crs=ccrs.PlateCarree(), draw_labels=False, color='gray', alpha=0.2)
    gl.xlocator = mticker.FixedLocator(np.arange(117,124,3))
    gl.ylocator = mticker.FixedLocator(np.arange(20,27,3))
    gl.xformatter, gl.yformatter = LONGITUDE_FORMATTER, LATITUDE_FORMATTER
    if i == 0 or i == 3:
        ax[i].set_yticks(np.arange(17,27,3),crs=proj)
        ax[i].set_yticklabels(('17\N{DEGREE SIGN}N','20\N{DEGREE SIGN}N','23\N{DEGREE SIGN}N',\
                               '26\N{DEGREE SIGN}N'))
    if i >= 3:
        ax[i].set_xticks(np.arange(-65,-55,3),crs=proj)
        ax[i].set_xticklabels(('115\N{DEGREE SIGN}E','118\N{DEGREE SIGN}E','121\N{DEGREE SIGN}E',\
                               '124\N{DEGREE SIGN}E'))
    
    # Quiverkeys and colorbars
    qk = ax[i].quiverkey(Q,0.2,0.92,sc*.1,'{:.1f} m/s'.format(sc*.1),labelpos='S',zorder=3)
    if i == 4:
        cbar_ax = fig.add_axes([0.15,0.02,0.46,0.02])
        cbar = fig.colorbar(cn, ticks=np.arange(vmin,vmax+.1,2), cax=cbar_ax, \
                            orientation='horizontal',extend='both')
        cbar.set_label('\N{DEGREE SIGN}C')
    elif i == 5:
        cbar_ax = fig.add_axes([0.67,0.02,0.2,0.02])
        cbar = fig.colorbar(cn, ticks=np.arange(vmin,vmax+.1,.4), cax=cbar_ax, \
                            orientation='horizontal', extend='both')
        cbar.set_label('\N{DEGREE SIGN}C')

# Titles
ax[0].set_title(str(1985+st_yr)+'-2014',fontsize=15)
ax[1].set_title(str(2040+st_yr)+'-2069',fontsize=15)
ax[2].set_title('Difference',fontsize=15)
ax[0].text(-0.25, 0.5, 'Winter', va='bottom', ha='center',rotation='vertical', \
           rotation_mode='anchor', transform=ax[0].transAxes, fontsize=15)
ax[3].text(-0.25, 0.5, 'Summer', va='bottom', ha='center',rotation='vertical', \
           rotation_mode='anchor', transform=ax[3].transAxes, fontsize=15)
plt.show()

#%% =================================================================
# Plot SSS & current
# ===================================================================
# Whole figure settings ---------------------------------------------
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
land = cart.feature.NaturalEarthFeature('physical', 'land', scale='10m',\
                                         edgecolor='k', facecolor='#AAAAAA',zorder=1)
proj = ccrs.PlateCarree(central_longitude=180)
fig,ax = plt.subplots(figsize=(10,6), nrows=2, ncols=3, facecolor='white',\
                      subplot_kw={'projection':proj})
plt.subplots_adjust(wspace=.02, hspace=.15)
ax = ax.flatten()

# Plot subplots -----------------------------------------------------
for i, s in enumerate(SSS.keys()):

    # Subplots' settings
    if i == 2 or i == 5:
        vmin, vmax = -.3, .3
        clev = np.arange(vmin,vmax+.01,.05)
        cm = 'RdBu_r'
        dv, sc = 5, 3
    else:
        vmin, vmax = 32.7, 33.7
        clev = np.arange(vmin,vmax+.01,.1)
        cm = 'turbo'
        dv, sc = 5, 5

    # Plotting
    #ax[i].set_extent([-64,-56,19,27],crs=proj)
    ax[i].set_extent([-67,-55,16,27],crs=proj)
    ax[i].add_feature(land)
    cn = ax[i].contourf(lon, lat, SSS[s], cmap=cm, vmin=vmin, vmax=vmax, levels=clev,\
                        extend='both', transform=ccrs.PlateCarree())
    Q = ax[i].quiver(lon[7::dv], lat[7::dv], UO[s][7::dv,7::dv], VO[s][7::dv,7::dv],\
                     scale=sc, width=4e-3, transform=ccrs.PlateCarree())
    
    # Ticks
    gl = ax[i].gridlines(crs=ccrs.PlateCarree(), draw_labels=False, color='gray', alpha=0.2)
    gl.xlocator = mticker.FixedLocator(np.arange(117,124,3))
    gl.ylocator = mticker.FixedLocator(np.arange(20,27,3))
    gl.xformatter, gl.yformatter = LONGITUDE_FORMATTER, LATITUDE_FORMATTER
    if i == 0 or i == 3:
        ax[i].set_yticks(np.arange(17,27,3),crs=proj)
        ax[i].set_yticklabels(('17\N{DEGREE SIGN}N','20\N{DEGREE SIGN}N','23\N{DEGREE SIGN}N',\
                               '26\N{DEGREE SIGN}N'))
    if i >= 3:
        ax[i].set_xticks(np.arange(-65,-55,3),crs=proj)
        ax[i].set_xticklabels(('115\N{DEGREE SIGN}E','118\N{DEGREE SIGN}E','121\N{DEGREE SIGN}E',\
                               '124\N{DEGREE SIGN}E'))
    
    # Quiverkeys and colorbars
    qk = ax[i].quiverkey(Q,0.2,0.92,sc*.1,'{:.1f} m/s'.format(sc*.1),labelpos='S',zorder=3)
    if i == 4:
        cbar_ax = fig.add_axes([0.15,0.02,0.46,0.02])
        cbar = fig.colorbar(cn, ticks=np.arange(vmin,vmax+.01,.1), cax=cbar_ax, \
                            orientation='horizontal',extend='both')
        cbar.set_label('\N{DEGREE SIGN}C')
    elif i == 5:
        cbar_ax = fig.add_axes([0.67,0.02,0.2,0.02])
        cbar = fig.colorbar(cn, ticks=np.arange(vmin,vmax+.01,.2), cax=cbar_ax, \
                            orientation='horizontal', extend='both')
        cbar.set_label('\N{DEGREE SIGN}C')

# Titles
ax[0].set_title(str(1985+st_yr)+'-2014',fontsize=15)
ax[1].set_title(str(2040+st_yr)+'-2069',fontsize=15)
ax[2].set_title('Difference',fontsize=15)
ax[0].text(-0.25, 0.5, 'Winter', va='bottom', ha='center',rotation='vertical', \
           rotation_mode='anchor', transform=ax[0].transAxes, fontsize=15)
ax[3].text(-0.25, 0.5, 'Summer', va='bottom', ha='center',rotation='vertical', \
           rotation_mode='anchor', transform=ax[3].transAxes, fontsize=15)
plt.show()