#%% The SST and Current Change in Taiwan Strait from TaiESM-ROMS
import cmaps
import numpy as np
import netCDF4 as nc
import cartopy as cart
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.stats import ttest_ind

#%% =================================================================
# Data Process 
# ===================================================================
# Read ncfile -------------------------------------------------------
root = '/data3/jhluna/work/ITRI_UCH_phdw/DATA/'
fnm = [root+'taiesm_roms_zlev/ocean_d01_his_mon.nc',\
       root+'taiesm_roms_zlev/ocean_d01_ssp_mon.nc']
scen = ['HIS', 'SSP']
sst, uo, vo, time = {}, {}, {}, {}

for f, fname in enumerate(fnm):
    with nc.Dataset(fname) as rootgrp:
        lat = rootgrp.variables['lat'][:]
        lon = rootgrp.variables['lon'][:]
        sst[scen[f]] = rootgrp.variables['SST'][:]            # (time,lat,lon)
        uo[scen[f]] = rootgrp.variables['uo'][:,1,:,:]
        vo[scen[f]] = rootgrp.variables['vo'][:,1,:,:]
        t = rootgrp.variables['time'][:]
        tu = rootgrp.variables['time'].units
        time[scen[f]] = nc.num2date(t,units=tu,calendar='standard')
del f,fname,t,tu,rootgrp

# Select month ------------------------------------------------------
# Initicalize
SST, UO, VO, VEL = {}, {}, {}, {}
seas = ['DJF', 'MAM', 'JJA', 'SON']
st_yr = 0                               # exclude spin-up year or not

# Get year and month from time
get_month = np.vectorize(lambda date: date.month)
mon = get_month(time['HIS'].compressed())
get_year = np.vectorize(lambda date: date.year)
yr = get_year(time['HIS'].compressed())

# Select season and year
it = {'DJF': np.where(((mon>=12)|(mon<=2))&(yr>=1985+st_yr))[0], 
      'MAM': np.where(((mon>=3)&(mon<=5))&(yr>=1985+st_yr))[0],
      'JJA': np.where(((mon>=6)&(mon<=8))&(yr>=1985+st_yr))[0],
      'SON': np.where(((mon>=9)&(mon<=11))&(yr>=1985+st_yr))[0]}

# Select season to variable dictionary
for se, sea in enumerate(seas):
    for sc, sce in enumerate(scen):
        SST[f"{sce}_{sea}"] = sst[sce][it[sea],:,:]                 # (time,lev,lat)
        UO[f"{sce}_{sea}"] = uo[sce][it[sea],:,:]
        VO[f"{sce}_{sea}"] = vo[sce][it[sea],:,:]
        VEL[f"{sce}_{sea}"] = np.sqrt(UO[f"{sce}_{sea}"]**2+VO[f"{sce}_{sea}"]**2)
    
    # Change after warming
    SST[f"DIFF_{sea}"] = np.nanmean(SST[f"SSP_{sea}"]-SST[f"HIS_{sea}"],axis=0)
    UO[f"DIFF_{sea}"] = np.nanmean(UO[f"SSP_{sea}"]-UO[f"HIS_{sea}"],axis=0)
    VO[f"DIFF_{sea}"] = np.nanmean(VO[f"SSP_{sea}"]-VO[f"HIS_{sea}"],axis=0)
    VEL[f"DIFF_{sea}"] = np.nanmean(VEL[f"SSP_{sea}"]-VEL[f"HIS_{sea}"],axis=0)    # (lev,lat)

    # P-value
    SST[f"PV_{sea}"] = np.zeros(SST['HIS_DJF'].shape[1:])           # (lev,lat)
    VEL[f"PV_{sea}"] = np.zeros(VEL['HIS_DJF'].shape[1:])
    for v in range(SST['HIS_DJF'].shape[1]):                     
        for j in range(SST['HIS_DJF'].shape[2]):
            SST[f"PV_{sea}"][v,j] = ttest_ind(SST[f"HIS_{sea}"][:,v,j], SST[f"SSP_{sea}"][:,v,j]).pvalue
            VEL[f"PV_{sea}"][v,j] = ttest_ind(VEL[f"HIS_{sea}"][:,v,j], VEL[f"SSP_{sea}"][:,v,j]).pvalue

# Seasonal mean
for se, sea in enumerate(seas):
    for sc, sce in enumerate(scen):
        SST[f"{sce}_{sea}"] = np.nanmean(SST[f"{sce}_{sea}"],axis=0)
        UO[f"{sce}_{sea}"] = np.nanmean(UO[f"{sce}_{sea}"],axis=0)
        VO[f"{sce}_{sea}"] = np.nanmean(VO[f"{sce}_{sea}"],axis=0)
        VEL[f"{sce}_{sea}"] = np.nanmean(VEL[f"{sce}_{sea}"],axis=0)

del sc,sce,se,sea,get_month,get_year,mon,yr,it,sst,uo,vo

#%% =================================================================
# Plot SST & Current climatology (2d) 
# ===================================================================
# Whole figure settings ---------------------------------------------
sea_str = ['Winter','Spring','Summer','Autumn']
fig,ax = plt.subplots(figsize=(15,3.5), nrows=1, ncols=4, facecolor='white')
plt.subplots_adjust(wspace=.1)
vmin, vmax = 18, 32
dl = 4
clev = np.arange(vmin, vmax+.1, .5)
clev1 = np.arange(vmin, vmax+.1, 1)

# Plot subplots -----------------------------------------------------
for i, s in enumerate(seas):

    # Plotting
    ax[i].set_facecolor('darkgray')
    cn = ax[i].contourf(lon, lat, SST[f"HIS_{s}"], cmap='turbo', vmin=vmin, vmax=vmax,\
                        levels=clev, extend='both')
    cm = ax[i].contour(lon, lat, SST[f"HIS_{s}"], levels=clev1, colors='dimgrey', linewidths=.7)
    Q = ax[i].quiver(lon[::dl],lat[::dl],np.squeeze(UO[f"HIS_{s}"][::dl,::dl]),\
                     np.squeeze(VO[f"HIS_{s}"][::dl,::dl]),scale=6,width=4e-3)

    # Clabel
    if i == 0:
        manual_locations = [(118,23.5),(119,22.5),(120.5,22)]
    elif i == 1:
        manual_locations = [(118,24),(119,23),(120.5,22)]
    elif i == 2:
        manual_locations = [(118.2,24),(119,23.25)]
    elif i == 3:
        manual_locations = [(119,23)]
    ax[i].clabel(cm, fontsize=10, manual=manual_locations)
    
    # Ticks
    ax[i].grid()
    ax[i].set_xlim([117,121])
    ax[i].set_ylim([21.5,25])
    if i>=1:
        ax[i].set_yticks(np.arange(22,26,1))
        ax[i].set_yticklabels([])
    else:
        ax[i].set_yticks(np.arange(22,26,1))
        ax[i].set_yticklabels(('22\N{DEGREE SIGN}N','23\N{DEGREE SIGN}N','24\N{DEGREE SIGN}N',\
                               '25\N{DEGREE SIGN}N'))
    ax[i].set_xticks(np.arange(117,121,1))
    ax[i].set_xticklabels(('117\N{DEGREE SIGN}E','118\N{DEGREE SIGN}E','119\N{DEGREE SIGN}E',\
                           '120\N{DEGREE SIGN}E'))
    ax[i].set_title(sea_str[i],fontsize=15)

# Colorbars and quiverkey
plt.quiverkey(Q,1.13,-.02,.5,label='0.5 m/s',labelpos='S')
cbar_ax = fig.add_axes([.92,0.13,0.008,0.7])
cbar = fig.colorbar(cn,cax=cbar_ax,orientation='vertical',extend='both',ticks=np.arange(vmin,vmax+.1,2))
cbar.ax.set_title('\N{DEGREE SIGN}C')
plt.show()

#%% =================================================================
# Plot SST change (2d) 
# ===================================================================
# Whole figure settings ---------------------------------------------
sea_str = ['Winter','Spring','Summer','Autumn']
fig,ax = plt.subplots(figsize=(15,3.5), nrows=1, ncols=4, facecolor='white')
plt.subplots_adjust(wspace=.1)
vmin, vmax = .8, 2
clev = np.arange(vmin,vmax+.1,.1)
clev1 = np.arange(vmin,vmax+.1,.2)

# Plot subplots -----------------------------------------------------
for i, s in enumerate(seas):

    # Plotting
    ax[i].set_facecolor('darkgray')
    cn = ax[i].contourf(lon, lat, SST[f"DIFF_{s}"], cmap='YlOrRd', vmin=vmin, vmax=vmax,\
                        levels=clev, extend='both')
    cm = ax[i].contour(lon, lat, SST[f"DIFF_{s}"], levels=clev, colors='k', linewidths=.7)
    cm = ax[i].contour(lon, lat, SST[f"DIFF_{s}"], levels=clev1, colors='k', linewidths=.7)
    
    # Plot p-value
    n = 3
    for v in range(SST['PV_DJF'].shape[0]//n):
        for j in range(SST['PV_DJF'].shape[1]//n):
            if SST[f"PV_{s}"][v*n,j*n] <= 0.05:
                ax[i].scatter(lon[j*n],lat[v*n],s=4,marker='o',c='dimgray')

    # Clabel
    if i == 0:
        manual_locations = [(118,23),(119.5,22.5),(119,23.5)]
    elif i == 1:
        manual_locations = [(119,23),(118.5,24.5),(120,21.5)]
    elif i == 2:
        manual_locations = [(118.5,22),(117.5,22.5),(119,23.25),(118.5,24)]
    elif i == 3:
        manual_locations = [(118.5,22.25),(119.8,22.7),(118,23.5)]
    ax[i].clabel(cm, fontsize=10, manual=manual_locations)

    # Ticks
    ax[i].grid()
    ax[i].set_xlim([117,121])
    ax[i].set_ylim([21.5,25])
    if i>=1:
        ax[i].set_yticks(np.arange(22,26,1))
        ax[i].set_yticklabels([])
    else:
        ax[i].set_yticks(np.arange(22,26,1))
        ax[i].set_yticklabels(('22\N{DEGREE SIGN}N','23\N{DEGREE SIGN}N','24\N{DEGREE SIGN}N',\
                               '25\N{DEGREE SIGN}N'))
    ax[i].set_xticks(np.arange(117,121,1))
    ax[i].set_xticklabels(('117\N{DEGREE SIGN}E','118\N{DEGREE SIGN}E','119\N{DEGREE SIGN}E',\
                           '120\N{DEGREE SIGN}E'))
    ax[i].set_title(sea_str[i],fontsize=15)

# Colorbars
cbar_ax = fig.add_axes([.92,0.13,0.008,0.7])
cbar = fig.colorbar(cn,cax=cbar_ax,orientation='vertical',extend='both',ticks=np.arange(vmin,vmax+.1,.2))
cbar.ax.set_title('\N{DEGREE SIGN}C')
plt.show()

#%% =================================================================
# Plot Current change (2d)
# ===================================================================
# Whole figure settings ---------------------------------------------
fig,ax = plt.subplots(figsize=(15,3.5), nrows=1, ncols=4, facecolor='white')
plt.subplots_adjust(wspace=.1)
m, dl = .3, 4
cmap = cmaps.cmp_flux.to_seg(N=24)

# Plot subplots -----------------------------------------------------
for i, s in enumerate(seas):

    # Calculate
    DIF = np.sqrt(UO[f"SSP_{s}"]**2 + VO[f"SSP_{s}"]**2) - \
          np.sqrt(UO[f"HIS_{s}"]**2 + VO[f"HIS_{s}"]**2)

    # Plotting
    ax[i].set_facecolor('darkgray')
    cn = ax[i].pcolormesh(lon, lat, DIF, cmap=cmap, vmin=-m, vmax=m)
    Q = ax[i].quiver(lon[::dl],lat[::dl],np.squeeze(UO[f"DIFF_{s}"][::dl,::dl]),\
                     np.squeeze(VO[f"DIFF_{s}"][::dl,::dl]),scale=1,width=4e-3)

    # Plot p-value
    n = 3
    for v in range(VEL['PV_DJF'].shape[0]//n):
        for j in range(VEL['PV_DJF'].shape[1]//n):
            if VEL[f"PV_{s}"][v*n,j*n] <= 0.05:
                ax[i].scatter(lon[j*n],lat[v*n],s=8,marker='x',c='dimgray')

    # Ticks
    ax[i].grid()
    ax[i].set_xlim([117,121])
    ax[i].set_ylim([21.5,25])
    if i>=1:
        ax[i].set_yticks(np.arange(22,26,1))
        ax[i].set_yticklabels([])
    else:
        ax[i].set_yticks(np.arange(22,26,1))
        ax[i].set_yticklabels(('22\N{DEGREE SIGN}N','23\N{DEGREE SIGN}N','24\N{DEGREE SIGN}N',\
                               '25\N{DEGREE SIGN}N'))
    ax[i].set_xticks(np.arange(117,121,1))
    ax[i].set_xticklabels(('117\N{DEGREE SIGN}E','118\N{DEGREE SIGN}E','119\N{DEGREE SIGN}E',\
                           '120\N{DEGREE SIGN}E'))
    ax[i].set_title(sea_str[i],fontsize=15)

# Colorbars and quiverkey
plt.quiverkey(Q,1.13,-.02,.06,label='0.06 m/s',labelpos='S')
cbar_ax = fig.add_axes([.917,0.13,0.008,0.7])
cbar = fig.colorbar(cn,cax=cbar_ax,orientation='vertical',extend='both')
cbar.ax.set_title('m/s')
plt.show()