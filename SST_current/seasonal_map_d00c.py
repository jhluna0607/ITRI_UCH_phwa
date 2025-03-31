#%%
import pandas as pd
import numpy as np
import datetime
import netCDF4 as nc
import cmaps
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

#%% =============================================================== 
# Read ncfile 
# =================================================================
#  Initialize -----------------------------------------------------
root = '/data3/jhluna/work/ITRI_UCH_phdw/DATA/'
scen = ['his','ssp']
year = np.array([2013,2067])
y = 0

# Each scenario
for sc in (scen):

    # Read file
    fnm = root+'itri_uch_avg_d00c_'+sc+'_zlev.nc'
    print(fnm)
    with nc.Dataset(fnm) as rootgrp:
        lat = np.array(rootgrp.variables['lat'][1:-1,1])
        lon = np.array(rootgrp.variables['lon'][1,1:-1])
        to = np.array(rootgrp.variables['SST'][:,1:-1,1:-1])
        uo = np.array(rootgrp.variables['uo'][:,1,1:-1,1:-1])       # 10 m
        vo = np.array(rootgrp.variables['vo'][:,1,1:-1,1:-1])
        mask = np.array(rootgrp.variables['mask'][1:-1,1:-1])

    # Mask out the invalid grid
    to[np.where(to>40)[:]] = np.nan
    mask[np.where(mask==1)[:]] = np.nan

    # Time
    st = datetime(year[y],1,1,0,15,0,0)
    en = datetime(year[y],12,27,0,16,0,0)
    time = np.arange(st,en,timedelta(hours=3)).astype(datetime)
    del st,en,fnm,rootgrp

#  Seasonal mean --------------------------------------------------
    # Extract month number
    mon = time.astype('datetime64[M]').astype(int) % 12 + 1
    season = [[12,2],[3,5],[6,8],[9,11]]

    # Initialize
    if y == 0:
        us = np.zeros([4,uo.shape[1],uo.shape[2],2])
        vs = np.zeros([4,vo.shape[1],vo.shape[2],2])
        ts = np.zeros([4,to.shape[1],to.shape[2],2])

    # Find the seasonal location
    for s, sea in enumerate(season):
        if s != 0:
            loc = np.where((mon>=sea[0]) & (mon<=sea[1]))[0]
        else:
            loc = np.where((mon>=sea[0]) | (mon<=sea[1]))[0]

        # Seasonal mean
        us[s,:,:,y] = np.nanmean(uo[loc,:,:],0)
        vs[s,:,:,y] = np.nanmean(vo[loc,:,:],0)
        ts[s,:,:,y] = np.nanmean(to[loc,:,:],0)
            
    y = y+1
    del mon,season,uo,vo,to,s,sea,loc
del y,year,root,sc

# %% ================================================================
# Plot SST
# ===================================================================
# Each scenario -----------------------------------------------------
# Initialize the figure setting
sea_str = ['Winter','Spring','Summer','Autumn']
cmap = cmaps.cmocean_balance.to_seg(N=20)
vmin, vmax = 20, 30
clev = np.arange(vmin, vmax+.1, .5)
clev1 = np.arange(vmin, vmax+.1, 1)
xtick = np.arange(119.2,119.9,.3)
ytick = np.arange(23.2,23.9,.2)

# Each scenario
for y in range(2):

    # Create a blank canva
    fig,ax = plt.subplots(figsize=(15,3.5),nrows=1,ncols=4,facecolor='white')
    plt.subplots_adjust(wspace=.1)

    # Plot each season
    for s in range(4):
        ax[s].set_facecolor('darkgray')
        cn = ax[s].contourf(lon,lat,ts[s,:,:,y],cmap=cmap,vmin=vmin,vmax=vmax,levels=clev,extend='both')
        cm = ax[s].contour(lon,lat,ts[s,:,:,y],levels=clev,colors='k',linewidths=.5)
        cm = ax[s].contour(lon,lat,ts[s,:,:,y],levels=clev1,colors='k',linewidths=.7)

        # Ticks, labels and title 
        ax[s].clabel(cm, fontsize=10)
        ax[s].set_yticks(ytick)
        if s>=1:
            ax[s].set_yticklabels([])
        else:
            ax[s].set_yticklabels([f'{t:.1f}\N{DEGREE SIGN}N' for t in ytick])
        ax[s].set_xticks(xtick)
        ax[s].set_xticklabels([f'{t:.1f}\N{DEGREE SIGN}E' for t in xtick])
        ax[s].set_title(sea_str[s],fontsize=15)
        ax[s].grid()

    # Colorbar and quiver
    cbar_ax = fig.add_axes([.92,0.13,0.008,0.7])
    cbar = fig.colorbar(cn,cax=cbar_ax,orientation='vertical',extend='both',ticks=np.arange(20,30.1,2))
    cbar.ax.set_title('\N{DEGREE SIGN}C')
    plt.show()

# Difference --------------------------------------------------------
# Initialize the figure setting
vmin, vmax = .8, 2
clev = np.arange(vmin,vmax+.1,.1)
clev1 = np.arange(0,vmax+.1,.2)

# Create a blank canva
fig,ax = plt.subplots(figsize=(15,3.5),nrows=1,ncols=4,facecolor='white')
plt.subplots_adjust(wspace=.1)

# Plot each season
for s in range(4):
    ax[s].set_facecolor('darkgray')
    tmp = ts[s,:,:,1]-ts[s,:,:,0]
    cn = ax[s].contourf(lon,lat,tmp,cmap='YlOrRd',vmin=vmin,vmax=vmax,levels=clev,extend='both')
    cm = ax[s].contour(lon,lat,tmp,levels=clev1,colors='k',linewidths=.7)

    # Ticks, labels and title 
    ax[s].clabel(cm, fontsize=10)
    ax[s].set_yticks(ytick)
    if s>=1:
        ax[s].set_yticklabels([])
    else:
        ax[s].set_yticklabels([f'{t:.1f}\N{DEGREE SIGN}N' for t in ytick])
    ax[s].set_xticks(xtick)
    ax[s].set_xticklabels([f'{t:.1f}\N{DEGREE SIGN}E' for t in xtick])
    ax[s].set_title(sea_str[s],fontsize=15)
    ax[s].grid()

# Colorbar and quiver
cbar_ax = fig.add_axes([.92,0.13,0.008,0.7])
cbar = fig.colorbar(cn,cax=cbar_ax,orientation='vertical',extend='both')
cbar.ax.set_title('\N{DEGREE SIGN}C')
plt.show()

# %% ================================================================
# Plot Current
# ===================================================================
# Initialize the figure setting
vel = np.sqrt(us**2+vs**2)
sea_str = ['Winter','Spring','Summer','Autumn']
dl = 5
xtick = np.arange(119.2,119.9,.3)
ytick = np.arange(23.2,23.9,.2)

# Each scenario
for y in range(2):

    # Create a blank canva
    fig,ax = plt.subplots(figsize=(15,3.5),nrows=1,ncols=4,facecolor='white')
    plt.subplots_adjust(wspace=.1)

    # Plot each season
    for s in range(4):
        cn = ax[s].pcolormesh(lon,lat,vel[s,:,:,y],cmap='BuPu',vmin=0,vmax=1)
        Q = ax[s].quiver(lon[2::dl],lat[2::dl],np.squeeze(us[s,2::dl,2::dl,y]),\
                         np.squeeze(vs[s,2::dl,2::dl,y]),scale=5,width=4e-3)
        ax[s].pcolormesh(lon,lat,mask,cmap='gray',vmin=-1,vmax=1)

        # Ticks, labels and title 
        ax[s].set_yticks(ytick)
        if s>=1:  
            ax[s].set_yticklabels([])
        else:
            ax[s].set_yticklabels([f'{t:.1f}\N{DEGREE SIGN}N' for t in ytick])
        ax[s].set_xticks(xtick)
        ax[s].set_xticklabels([f'{t:.1f}\N{DEGREE SIGN}E' for t in xtick])
        ax[s].set_title(sea_str[s],fontsize=15)
        ax[s].grid()

    # Colorbar and quiver
    cbar_ax = fig.add_axes([.92,0.13,0.008,0.7])
    cbar = fig.colorbar(cn,cax=cbar_ax,orientation='vertical',extend='both')
    cbar.ax.set_title('m/s')
    plt.show()

# Difference --------------------------------------------------------
# Initialize the figure setting
m, dl = .3, 5
cmap = cmaps.cmp_flux.to_seg(N=24)

# Create a blank canva
fig,ax = plt.subplots(figsize=(15,3.5),nrows=1,ncols=4,facecolor='white')
plt.subplots_adjust(wspace=.1)

# Plot each season
for s in range(4):
    tmp = vel[s,:,:,1]-vel[s,:,:,0]
    utmp = us[s,:,:,1]-us[s,:,:,0]
    vtmp = vs[s,:,:,1]-vs[s,:,:,0]

    vtmp[vtmp>.1] = np.nan
    cn = ax[s].pcolormesh(lon,lat,tmp,cmap=cmap,vmin=-m,vmax=m)
    Q = ax[s].quiver(lon[2::dl],lat[2::dl],np.squeeze(utmp[2::dl,2::dl]),\
                    np.squeeze(vtmp[2::dl,2::dl]),scale=1,width=4e-3)
    ax[s].pcolormesh(lon,lat,mask,cmap='gray',vmin=-1,vmax=1)

    # Ticks, labels and title 
    ax[s].set_yticks(ytick)
    if s>=1:  
        ax[s].set_yticklabels([])
    else:
        ax[s].set_yticklabels([f'{t:.1f}\N{DEGREE SIGN}N' for t in ytick])
    ax[s].set_xticks(xtick)
    ax[s].set_xticklabels([f'{t:.1f}\N{DEGREE SIGN}E' for t in xtick])
    ax[s].set_title(sea_str[s],fontsize=15)
    ax[s].grid()

# Colorbar and quiver
plt.quiverkey(Q,1.13,-.02,.06,label='0.06 m/s',labelpos='S')
cbar_ax = fig.add_axes([.92,0.13,0.008,0.7])
cbar = fig.colorbar(cn,cax=cbar_ax,orientation='vertical',extend='both')
cbar.ax.set_title('m/s')
plt.show()
