#%%
import cmaps
import pandas as pd
import numpy as np
import datetime
import cftime
import netCDF4 as nc
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

#%% =================================================================
# Data Process 
# ===================================================================
# Initialize --------------------------------------------------------
root = '/data3/jhluna/work/ITRI_UCH_phdw/DATA/'
fname = [root+'itri_uch_avg_d00_his_zlev.nc', root+'itri_uch_avg_d00_ssp_zlev.nc']
scenm = ['HIS','SSP']
sst, uo, vo, time = {}, {}, {}, {}

# Each scenario ncfile
for f, fnm in enumerate(fname):

    # Read data
    with nc.Dataset(fnm) as rootgrp:
        lat = np.array(rootgrp.variables['lat'][1:-1,1:-1])
        lon = np.array(rootgrp.variables['lon'][1:-1,1:-1])
        mask = np.array(rootgrp.variables['mask'][1:-1,1:-1])
        sst[scenm[f]] = np.array(rootgrp.variables['SST'][:,1:-1,1:-1])
        uo[scenm[f]] = np.array(rootgrp.variables['uo'][:,1,1:-1,1:-1])       # 10 m
        vo[scenm[f]] = np.array(rootgrp.variables['vo'][:,1,1:-1,1:-1])

        sst[scenm[f]][np.where((sst[scenm[f]]==0)|(sst[scenm[f]]>35))[:]] = np.nan
        uo[scenm[f]][np.where(np.abs(uo[scenm[f]]>10))[:]] = np.nan
        vo[scenm[f]][np.where(np.abs(vo[scenm[f]]>10))[:]] = np.nan

        t = rootgrp.variables['time'][:]
        tu = rootgrp.variables['time'].units
        time[scenm[f]] = nc.num2date(t,units=tu,calendar='standard')

del rootgrp,f,fnm,tu,t

# Select month ------------------------------------------------------
SST, UO, VO, VEL = {}, {}, {}, {}
seas = ['DJF', 'MAM', 'JJA', 'SON']
st_yr = 0                                # exclude spin-up year or not

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
    for sc, sce in enumerate(scenm):
        SST[f"{sce}_{sea}"] = sst[sce][it[sea],:,:]             # (time,lon,lat)
        UO[f"{sce}_{sea}"] = uo[sce][it[sea],:,:]
        VO[f"{sce}_{sea}"] = vo[sce][it[sea],:,:]
        VEL[f"{sce}_{sea}"] = np.sqrt(UO[f"{sce}_{sea}"]**2+VO[f"{sce}_{sea}"]**2)

    # Change after warming
    SST[f"DIFF_{sea}"] = np.nanmean(SST[f"SSP_{sea}"]-SST[f"HIS_{sea}"],axis=0)
    UO[f"DIFF_{sea}"] = np.nanmean(UO[f"SSP_{sea}"]-UO[f"HIS_{sea}"],axis=0)
    VO[f"DIFF_{sea}"] = np.nanmean(VO[f"SSP_{sea}"]-VO[f"HIS_{sea}"],axis=0)
    VEL[f"DIFF_{sea}"] = np.nanmean(VEL[f"SSP_{sea}"]-VEL[f"HIS_{sea}"],axis=0)    # (lon,lat)

    # P-value
    SST[f"PV_{sea}"] = np.zeros(SST['HIS_DJF'].shape[1:])           # (lon,lat)
    VEL[f"PV_{sea}"] = np.zeros(VEL['HIS_DJF'].shape[1:])
    for v in range(SST['HIS_DJF'].shape[1]):                     
        for j in range(SST['HIS_DJF'].shape[2]):
            SST[f"PV_{sea}"][v,j] = ttest_ind(SST[f"HIS_{sea}"][:,v,j], SST[f"SSP_{sea}"][:,v,j]).pvalue
            VEL[f"PV_{sea}"][v,j] = ttest_ind(VEL[f"HIS_{sea}"][:,v,j], VEL[f"SSP_{sea}"][:,v,j]).pvalue

# Seasonal mean
for se, sea in enumerate(seas):
    for sc, sce in enumerate(scenm):
        SST[f"{sce}_{sea}"] = np.nanmean(SST[f"{sce}_{sea}"],axis=0)
        UO[f"{sce}_{sea}"] = np.nanmean(UO[f"{sce}_{sea}"],axis=0)
        VO[f"{sce}_{sea}"] = np.nanmean(VO[f"{sce}_{sea}"],axis=0)
        VEL[f"{sce}_{sea}"] = np.nanmean(VEL[f"{sce}_{sea}"],axis=0)

del sc,sce,se,sea,get_month,get_year,mon,yr,st_yr,it,v,j

#%% =================================================================
# Plot SST climatology (2d) 
# ===================================================================
# Whole figure settings ---------------------------------------------
sea_str = ['Winter','Spring','Summer','Autumn']
cmap = cmaps.cmocean_balance.to_seg(N=20)
vmin, vmax = 20, 30
clev = np.arange(vmin,vmax+.1,.5)
clev1 = np.arange(vmin,vmax+.1,1)
ytick = np.arange(23.2,23.9,.2)
xtick = np.arange(119.2,119.9,.3)

fig,ax = plt.subplots(figsize=(15,3.5), nrows=1, ncols=4, facecolor='white')
plt.subplots_adjust(wspace=.1)

# Plot subplots -----------------------------------------------------
for i, s in enumerate(seas):

    # Plotting
    ax[i].set_facecolor('darkgray')
    cn = ax[i].contourf(lon, lat, SST[f"HIS_{s}"], cmap=cmap, vmin=vmin, vmax=vmax,\
                        levels=clev, extend='both')
    cm = ax[i].contour(lon, lat, SST[f"HIS_{s}"], levels=clev, colors='k', linewidths=.5)
    cm = ax[i].contour(lon, lat, SST[f"HIS_{s}"], levels=clev1, colors='k', linewidths=.7)

    # Clabel and ticks
    ax[i].set_xlim([lon[0],lon[-1]])
    ax[i].set_ylim([lat[0],lat[-1]])
    ax[i].clabel(cm, fontsize=10)
    ax[i].set_yticks(ytick)
    if i>=1:  
        ax[i].set_yticklabels([])
    else:
        ax[i].set_yticklabels([f'{t:.1f}\N{DEGREE SIGN}N' for t in ytick])
    ax[i].set_xticks(xtick)
    ax[i].set_xticklabels([f'{t:.1f}\N{DEGREE SIGN}E' for t in xtick])
    ax[i].set_title(sea_str[i],fontsize=15)
    ax[i].grid()

# Colorbars
cbar_ax = fig.add_axes([.92,0.13,0.008,0.7])
cbar = fig.colorbar(cn,cax=cbar_ax,orientation='vertical',extend='both',ticks=np.arange(20,30.1,2))
cbar.ax.set_title('\N{DEGREE SIGN}C')
plt.show()

#%% For ITRI --------------------------------------------------------
for i, s in enumerate(seas):
    fig,ax = plt.subplots(figsize=(4,4), nrows=1, ncols=1, facecolor='white')

    # Plotting
    ax.contourf(lon,lat,mask,cmap='gray',vmin=-10,vmax=10)
    cn = ax.contourf(lon, lat, SST[f"HIS_{s}"], cmap=cmap, vmin=vmin, vmax=vmax,\
                        levels=clev, extend='both')
    cm = ax.contour(lon, lat, SST[f"HIS_{s}"], levels=clev, colors='k', linewidths=.5)
    cm = ax.contour(lon, lat, SST[f"HIS_{s}"], levels=clev1, colors='k', linewidths=.7)

    # Clabel and ticks
    ax.set_xlim([lon[0],lon[-1]])
    ax.set_ylim([lat[0],lat[-1]])
    ax.clabel(cm, fontsize=10)
    ax.set_yticks(ytick)
    ax.set_yticklabels([f'{t:.1f}\N{DEGREE SIGN}N' for t in ytick])
    ax.set_xticks(xtick)
    ax.set_xticklabels([f'{t:.1f}\N{DEGREE SIGN}E' for t in xtick])
    ax.grid()
    fig.savefig('seasonal_map_d00_sst_mean_'+s+'.png',bbox_inches='tight', transparent=True, dpi=330)

# Colorbars
fig,ax = plt.subplots(figsize=(1,3.5))
plt.axis('off')
cbar_ax = fig.add_axes([.3,0.13,0.12,0.7])
cbar = fig.colorbar(cn,cax=cbar_ax,orientation='vertical',extend='both',ticks=np.arange(20,30.1,2))
cbar.ax.set_title('\N{DEGREE SIGN}C')
plt.show()
fig.savefig('seasonal_map_d00_sst_mean_cbar.png',bbox_inches='tight', transparent=True, dpi=330)

#%% =================================================================
# Plot Current climatology (2d) 
# ===================================================================
# Whole figure settings ---------------------------------------------
cmap = 'BuPu'
dl = 5
vmin, vmax = 0, 1

fig,ax = plt.subplots(figsize=(15,3.5), nrows=1, ncols=4, facecolor='white')
plt.subplots_adjust(wspace=.1)

# Plot subplots -----------------------------------------------------
for i, s in enumerate(seas):

    tmp = np.sqrt(UO[f"HIS_{s}"]**2 + VO[f"HIS_{s}"]**2)

    # Plotting
    ax[i].pcolormesh(lon,lat,mask,cmap='gray',vmin=-1,vmax=1)
    cn = ax[i].pcolormesh(lon, lat, tmp, cmap=cmap, vmin=vmin, vmax=vmax)
    Q = ax[i].quiver(lon[2::dl],lat[2::dl],np.squeeze(UO[f"HIS_{s}"][2::dl,2::dl]),\
                     np.squeeze(VO[f"HIS_{s}"][2::dl,2::dl]),scale=5,width=4e-3)

    # Clabel and ticks
    ax[i].set_xlim([lon[0],lon[-1]])
    ax[i].set_ylim([lat[0],lat[-1]])
    ax[i].set_yticks(ytick)
    if i>=1:  
        ax[i].set_yticklabels([])
    else:
        ax[i].set_yticklabels([f'{t:.1f}\N{DEGREE SIGN}N' for t in ytick])
    ax[i].set_xticks(xtick)
    ax[i].set_xticklabels([f'{t:.1f}\N{DEGREE SIGN}E' for t in xtick])
    ax[i].set_title(sea_str[i],fontsize=15)
    ax[i].grid()

# Colorbars
cbar_ax = fig.add_axes([.92,0.13,0.008,0.7])
cbar = fig.colorbar(cn,cax=cbar_ax,orientation='vertical',extend='both')
cbar.ax.set_title('m/s')
plt.show()

#%% For ITRI --------------------------------------------------------
for i, s in enumerate(seas):
    fig,ax = plt.subplots(figsize=(4,4), nrows=1, ncols=1, facecolor='white')

    # Plotting
    ax.pcolormesh(lon,lat,mask,cmap='gray',vmin=-1,vmax=1)
    cn = ax.pcolormesh(lon, lat, tmp, cmap=cmap, vmin=vmin, vmax=vmax)
    Q = ax.quiver(lon[2::dl],lat[2::dl],np.squeeze(UO[f"HIS_{s}"][2::dl,2::dl]),\
                     np.squeeze(VO[f"HIS_{s}"][2::dl,2::dl]),scale=5,width=4e-3)

    # Clabel and ticks
    ax.set_xlim([lon[0],lon[-1]])
    ax.set_ylim([lat[0],lat[-1]])
    ax.clabel(cm, fontsize=8)
    ax.set_yticks(ytick)
    ax.set_yticklabels([f'{t:.1f}\N{DEGREE SIGN}N' for t in ytick])
    ax.set_xticks(xtick)
    ax.set_xticklabels([f'{t:.1f}\N{DEGREE SIGN}E' for t in xtick])
    ax.grid()
    ax.set_facecolor('white')
    #fig.savefig('seasonal_map_d00_cur_mean_'+s+'.png',bbox_inches='tight', transparent=True, dpi=330)

# Colorbars
fig,ax = plt.subplots(figsize=(1,3.5))
plt.axis('off')
cbar_ax = fig.add_axes([.3,0.13,0.12,0.7])
cbar = fig.colorbar(cn,cax=cbar_ax,orientation='vertical',extend='both')
cbar.ax.set_title('m/s')
plt.show()
#fig.savefig('seasonal_map_d00_cur_mean_cbar.png',bbox_inches='tight', transparent=True, dpi=330)

#%% =================================================================
# Plot SST & Current climatology (2d) 
# ===================================================================
# Whole figure settings ---------------------------------------------
fig,ax = plt.subplots(figsize=(15,3.5), nrows=1, ncols=4, facecolor='white')
plt.subplots_adjust(wspace=.1)
vmin, vmax = 18, 32
dl = 5
clev = np.arange(vmin, vmax+.1, .5)
clev1 = np.arange(vmin, vmax+.1, .5)

# Plot subplots -----------------------------------------------------
for i, s in enumerate(seas):

    # Plotting
    ax[i].set_facecolor('darkgray')
    cn = ax[i].contourf(lon, lat, SST[f"HIS_{s}"], cmap='turbo', vmin=vmin, vmax=vmax,\
                        levels=clev, extend='both')
    cm = ax[i].contour(lon, lat, SST[f"HIS_{s}"], levels=clev1, colors='dimgrey', linewidths=.7)
    Q = ax[i].quiver(lon[2::dl],lat[2::dl],np.squeeze(UO[f"HIS_{s}"][2::dl,2::dl]),\
                     np.squeeze(VO[f"HIS_{s}"][2::dl,2::dl]),scale=7,width=4e-3)

    # Clabel
    if i == 0:
        manual_locations = [(119.25,23.75),(119.5,23.25)]
    elif i == 1:
        manual_locations = [(119,24),(119.75,23.2)]
    elif i == 2:
        manual_locations = [(119.85,23.1),(119.5,23.6),(119.8,23.7)]
    elif i == 3:
        manual_locations = [(119.5,23.1)]
    ax[i].clabel(cm, fmt='%d', fontsize=10, manual=manual_locations)
    
    # Ticks
    ax[i].grid()
    ax[i].set_xlim([lon[0],lon[-1]])
    ax[i].set_ylim([lat[0],lat[-1]])
    ax[i].set_yticks(ytick)
    if i>=1:
        ax[i].set_yticklabels([])
    else:
        ax[i].set_yticklabels([f'{t:.1f}\N{DEGREE SIGN}N' for t in ytick])
    ax[i].set_xticks(xtick)
    ax[i].set_xticklabels([f'{t}\N{DEGREE SIGN}E' for t in xtick])
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
vmin, vmax = .8, 2
clev = np.arange(vmin,vmax+.1,.1)
clev1 = np.arange(0,vmax+.1,.2)

fig,ax = plt.subplots(figsize=(15,3.5), nrows=1, ncols=4, facecolor='white')
plt.subplots_adjust(wspace=.1)

# Plot subplots -----------------------------------------------------
for i, s in enumerate(seas):

    # Plotting
    ax[i].set_facecolor('darkgray')
    cn = ax[i].contourf(lon, lat, SST[f"DIFF_{s}"], cmap='YlOrRd', vmin=vmin, vmax=vmax,\
                        levels=clev, extend='both')
    cm = ax[i].contour(lon, lat, SST[f"DIFF_{s}"], levels=clev1, colors='k', linewidths=.7)
    
    # Plot p-value
    n = 5
    for v in range(SST['PV_DJF'].shape[0]//n):
        for j in range(SST['PV_DJF'].shape[1]//n):
            if SST[f"PV_{s}"][v*n+4,j*n+4] <= 0.05:
                ax[i].scatter(lon[j*n+4],lat[v*n+4],s=4,marker='o',c='dimgray')

    # Clabel
    if i == 0:
        manual_locations = [(119.1,23.3),(119.5,23.6),(119.9,23.1)]
    elif i == 1:
        manual_locations = [(119,23.5),(119.25,23.75),(119.5,23.2),(119.9,23.5),(119.9,23.1)]
    elif i == 2:
        manual_locations = [(119.8,23.1),(119.5,23.6),(119.25,23.7),(119.2,23.8)]
    elif i == 3:
        manual_locations = [(119.25,23.5),(119.3,23.3),(119.1,23.8),(119.8,23.2)]
    ax[i].clabel(cm, fontsize=10, manual=manual_locations)

    # Ticks
    ax[i].set_xlim([lon[0],lon[-1]])
    ax[i].set_ylim([lat[0],lat[-1]])
    ax[i].set_yticks(ytick)
    if i>=1:  
        ax[i].set_yticklabels([])
    else:
        ax[i].set_yticklabels([f'{t:.1f}\N{DEGREE SIGN}N' for t in ytick])
    ax[i].set_xticks(xtick)
    ax[i].set_xticklabels([f'{t:.1f}\N{DEGREE SIGN}E' for t in xtick])
    ax[i].set_title(sea_str[i],fontsize=15)
    ax[i].grid()

# Colorbars
cbar_ax = fig.add_axes([.92,0.13,0.008,0.7])
cbar = fig.colorbar(cn,cax=cbar_ax,orientation='vertical',extend='both',ticks=np.arange(vmin,vmax+.1,.2))
cbar.ax.set_title('\N{DEGREE SIGN}C')
plt.show()

#%% For ITRI --------------------------------------------------------
for i, s in enumerate(seas):
    fig,ax = plt.subplots(figsize=(4,4), nrows=1, ncols=1, facecolor='white')

    # Plotting
    ax.set_facecolor('darkgray')
    ax.contourf(lon,lat,mask,cmap='gray',vmin=-10,vmax=10)
    cn = ax.contourf(lon, lat, SST[f"DIFF_{s}"], cmap='YlOrRd', vmin=vmin, vmax=vmax,\
                        levels=clev, extend='both')
    cm = ax.contour(lon, lat, SST[f"DIFF_{s}"], levels=clev1, colors='k', linewidths=.7)
    
    # Plot p-value
    n = 5
    for v in range(SST['PV_DJF'].shape[0]//n):
        for j in range(SST['PV_DJF'].shape[1]//n):
            if SST[f"PV_{s}"][v*n+4,j*n+4] <= 0.05:
                ax.scatter(lon[j*n+4],lat[v*n+4],s=4,marker='o',c='dimgray')

    # Clabel
    if i == 0:
        manual_locations = [(119.1,23.3),(119.5,23.6),(119.9,23.1)]
    elif i == 1:
        manual_locations = [(119,23.5),(119.25,23.75),(119.5,23.2),(119.9,23.5),(119.9,23.1)]
    elif i == 2:
        manual_locations = [(119.8,23.1),(119.5,23.6),(119.25,23.7),(119.2,23.8)]
    elif i == 3:
        manual_locations = [(119.25,23.5),(119.3,23.3),(119.1,23.8),(119.8,23.2)]
    ax.clabel(cm, fontsize=10, manual=manual_locations)

    # Clabel and ticks
    ax.set_xlim([lon[0],lon[-1]])
    ax.set_ylim([lat[0],lat[-1]])
    ax.clabel(cm, fontsize=8)
    ax.set_yticks(ytick)
    ax.set_yticklabels([f'{t:.1f}\N{DEGREE SIGN}N' for t in ytick])
    ax.set_xticks(xtick)
    ax.set_xticklabels([f'{t:.1f}\N{DEGREE SIGN}E' for t in xtick])
    ax.grid()
    #fig.savefig('seasonal_map_d00_sst_diff_'+s+'.png',bbox_inches='tight', transparent=True, dpi=330)

# Colorbars
fig,ax = plt.subplots(figsize=(1,3.5))
plt.axis('off')
cbar_ax = fig.add_axes([.3,0.13,0.12,0.7])
cbar = fig.colorbar(cn,cax=cbar_ax,orientation='vertical',extend='both')
cbar.ax.set_title('\N{DEGREE SIGN}C')
plt.show()
#fig.savefig('seasonal_map_d00_sst_diff_cbar.png',bbox_inches='tight', transparent=True, dpi=330)

#%% =================================================================
# Plot Current change (2d)
# ===================================================================
# Whole figure settings ---------------------------------------------
m, dl = .3, 5
cmap = cmaps.cmp_flux.to_seg(N=24)

fig,ax = plt.subplots(figsize=(15,3.5), nrows=1, ncols=4, facecolor='white')
plt.subplots_adjust(wspace=.1)

# Plot subplots -----------------------------------------------------
for i, s in enumerate(seas):

    # Calculate
    DIF = np.sqrt(UO[f"SSP_{s}"]**2 + VO[f"SSP_{s}"]**2) - \
          np.sqrt(UO[f"HIS_{s}"]**2 + VO[f"HIS_{s}"]**2)

    # Plotting
    ax[i].pcolormesh(lon,lat,mask,cmap='gray',vmin=-1,vmax=1)
    cn = ax[i].pcolormesh(lon, lat, DIF, cmap=cmap, vmin=-m, vmax=m)
    Q = ax[i].quiver(lon[2::dl],lat[2::dl],np.squeeze(UO[f"DIFF_{s}"][2::dl,2::dl]),\
                     np.squeeze(VO[f"DIFF_{s}"][2::dl,2::dl]),scale=1,width=4e-3)

    # Plot p-value
    n = 5
    for v in range(VEL['PV_DJF'].shape[0]//n):
        for j in range(VEL['PV_DJF'].shape[1]//n):
            if VEL[f"PV_{s}"][v*n+2,j*n+2] <= 0.05:
                ax[i].scatter(lon[j*n+2],lat[v*n+2],s=8,marker='x',c='dimgray')

    # Ticks
    ax[i].set_xlim([lon[0],lon[-1]])
    ax[i].set_ylim([lat[0],lat[-1]])
    ax[i].clabel(cm, fontsize=8)
    ax[i].set_yticks(ytick)
    if i>=1:  
        ax[i].set_yticklabels([])
    else:
        ax[i].set_yticklabels([f'{t:.1f}\N{DEGREE SIGN}N' for t in ytick])
    ax[i].set_xticks(xtick)
    ax[i].set_xticklabels([f'{t:.1f}\N{DEGREE SIGN}E' for t in xtick])
    ax[i].set_title(sea_str[i],fontsize=15)
    ax[i].grid()

# Colorbars and quiverkey
plt.quiverkey(Q,1.13,-.02,.06,label='0.06 m/s',labelpos='S')
cbar_ax = fig.add_axes([.917,0.13,0.008,0.7])
cbar = fig.colorbar(cn,cax=cbar_ax,orientation='vertical',extend='both')
cbar.ax.set_title('m/s')
plt.show()

#%% For ITRI --------------------------------------------------------
for i, s in enumerate(seas):
    fig,ax = plt.subplots(figsize=(4,4), nrows=1, ncols=1, facecolor='white')

    DIF = np.sqrt(UO[f"SSP_{s}"]**2 + VO[f"SSP_{s}"]**2) - \
          np.sqrt(UO[f"HIS_{s}"]**2 + VO[f"HIS_{s}"]**2)

    # Plotting
    ax.pcolormesh(lon,lat,mask,cmap='gray',vmin=-1,vmax=1)
    cn = ax.pcolormesh(lon, lat, DIF, cmap=cmap, vmin=-m, vmax=m)
    Q = ax.quiver(lon[2::dl],lat[2::dl],np.squeeze(UO[f"DIFF_{s}"][2::dl,2::dl]),\
                     np.squeeze(VO[f"DIFF_{s}"][2::dl,2::dl]),scale=1,width=4e-3)
    
    # Plot p-value
    n = 5
    for v in range(VEL['PV_DJF'].shape[0]//n):
        for j in range(VEL['PV_DJF'].shape[1]//n):
            if VEL[f"PV_{s}"][v*n+2,j*n+2] <= 0.05:
                ax.scatter(lon[j*n+2],lat[v*n+2],s=8,marker='x',c='dimgray')

    # Clabel and ticks
    ax.set_xlim([lon[0],lon[-1]])
    ax.set_ylim([lat[0],lat[-1]])
    ax.clabel(cm, fontsize=8)
    ax.set_yticks(ytick)
    ax.set_yticklabels([f'{t:.1f}\N{DEGREE SIGN}N' for t in ytick])
    ax.set_xticks(xtick)
    ax.set_xticklabels([f'{t:.1f}\N{DEGREE SIGN}E' for t in xtick])
    ax.grid()
    fig.savefig('seasonal_map_d00_cur_diff_'+s+'.png',bbox_inches='tight', transparent=True, dpi=330)

# Colorbars
fig,ax = plt.subplots(figsize=(1,3.5))
plt.axis('off')
cbar_ax = fig.add_axes([.3,0.13,0.12,0.7])
cbar = fig.colorbar(cn,cax=cbar_ax,orientation='vertical',extend='both')
cbar.ax.set_title('m/s')
plt.show()
fig.savefig('seasonal_map_d00_cur_diff_cbar.png',bbox_inches='tight', transparent=True, dpi=330)

