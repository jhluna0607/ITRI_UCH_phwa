#%%
import pandas as pd
import numpy as np
import datetime
import netCDF4 as nc
import cmaps
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.patches as patches

#%% Read ncfile =====================================================
root = '/data3/jhluna/work/ITRI_UCH_phdw/DATA/'
scen = ['his','ssp']
year = np.array([2013,2067])
y = 0

for sc in (scen):
    fnm = root+'itri_uch_avg_d01_'+sc+'_zlev.nc'

    with nc.Dataset(fnm) as rootgrp:
        to = np.array(rootgrp.variables['SST'][:,1:-1,1:-1])
        lat = np.array(rootgrp.variables['lat'][1:-1,1])
        lon = np.array(rootgrp.variables['lon'][1,1:-1])
        uo = np.array(rootgrp.variables['uo'][:,1,1:-1,1:-1])
        vo = np.array(rootgrp.variables['vo'][:,1,1:-1,1:-1])
        mask = np.array(rootgrp.variables['mask'][1:-1,1:-1])

    to[np.where(to>40)[:]] = np.nan
    #mask[np.where(mask==1)[:]] = np.nan

    st = datetime(year[y],1,1,0,15,0,0)
    en = datetime(year[y],12,27,0,16,0,0)
    time = np.arange(st,en,timedelta(hours=3)).astype(datetime)
    del st,en

# Seasonal mean --------------------------------------------------
    mon = time.astype('datetime64[M]').astype(int) % 12 + 1
    season = [[12,2],[3,5],[6,8],[9,11]]
    if y == 0:
        us = np.zeros([4,uo.shape[1],uo.shape[2],2])
        vs = np.zeros([4,vo.shape[1],vo.shape[2],2])
        ts = np.zeros([4,to.shape[1],to.shape[2],2])
    for s in range(4):
        sea = season[s]
        if s != 0:
            loc = np.where((mon>=sea[0]) & (mon<=sea[1]))[0]
        else:
            loc = np.where((mon>=sea[0]) | (mon<=sea[1]))[0]
        us[s,:,:,y] = np.nanmean(uo[loc,:,:],0)
        vs[s,:,:,y] = np.nanmean(vo[loc,:,:],0)
        ts[s,:,:,y] = np.nanmean(to[loc,:,:],0)
    
    y = y+1
    del season,sea,s,loc

#%% Plot SST and current ============================================
sea_str = ['Winter','Spring','Summer','Autumn']
vel = np.sqrt(us**2+vs**2)
dl = 10
cmap = cmaps.cmocean_balance.to_seg(N=20)
clev = np.arange(20,30.1,.5)
clev1 = np.arange(20,30.1,1)
xtick = np.arange(119.47,119.59,0.03)
ytick = np.arange(23.34,23.41,.02)

for y in range(1):
    fig,ax = plt.subplots(figsize=(10,6),nrows=2,ncols=2,facecolor='white')
    plt.subplots_adjust(wspace=.1,hspace=.2)
    ax = ax.flatten()

    for s in range(4):
        cn = ax[s].contourf(lon,lat,ts[s,:,:,y],cmap=cmap,vmin=20,vmax=30,levels=clev,extend='both')
        cm = ax[s].contour(lon,lat,ts[s,:,:,y],levels=clev,colors='k',linewidths=.5)
        cm = ax[s].contour(lon,lat,ts[s,:,:,y],levels=clev1,colors='k',linewidths=.7)
        Q = ax[s].quiver(lon[::dl],lat[::dl],np.squeeze(us[s,::dl,::dl,y]),\
                         np.squeeze(vs[s,::dl,::dl,y]),scale=5,width=4e-3)
        
        # Clabel
        if s == 0:
            manual_locations = [(119.49,23.37)]
        elif s == 1:
            manual_locations = [(119.56,23.39)]
        elif s == 2:
            manual_locations = [(119.52,23.38)]
        elif s == 3:
            manual_locations = [(119.52,23.39)]
        ax[s].clabel(cm, fontsize=10, manual=manual_locations)

        ax[s].set_facecolor('darkgray')
        ax[s].set_yticks(ytick)
        if s==1 or s==3:
            ax[s].set_yticklabels([])
        else:
            ax[s].set_yticklabels([f'{t:.2f}\N{DEGREE SIGN}N' for t in ytick])
        ax[s].set_xticks(xtick)
        if s<=1:
            ax[s].set_xticklabels([])
        else:
            ax[s].set_xticklabels([f'{t:.2f}\N{DEGREE SIGN}E' for t in xtick])
        ax[s].set_title(sea_str[s])
        ax[s].grid(zorder=0)
    
    plt.quiverkey(Q,1.15,-.02,.3,label='0.3 m/s',labelpos='S')
    cbar_ax = fig.add_axes([.95,0.13,0.01,0.7])
    cbar = fig.colorbar(cn,cax=cbar_ax,orientation='vertical',extend='both',ticks=np.arange(20,30.1,2))
    cbar.ax.set_title('\N{DEGREE SIGN}C')
    plt.show()

#%% Plot SST and current for ITRI-----------------------------------------
sea_mon = ['DJF','MAM','JJA','SON']
x, y = 0.025, 0.013

for s in range(4):
    fig,ax = plt.subplots(figsize=(5,3),nrows=1,ncols=1,facecolor='white')
    currentaxis = fig.gca()

    ax.contourf(lon,lat,mask,cmap='gray',vmin=-10,vmax=10)
    cn = ax.contourf(lon,lat,ts[s,:,:,0],cmap=cmap,vmin=20,vmax=30,levels=clev,extend='both')
    cm = ax.contour(lon,lat,ts[s,:,:,0],levels=clev,colors='k',linewidths=.5)
    cm = ax.contour(lon,lat,ts[s,:,:,0],levels=clev1,colors='k',linewidths=.7)
    Q = ax.quiver(lon[::dl],lat[::dl],np.squeeze(us[s,::dl,::dl,0]),\
                         np.squeeze(vs[s,::dl,::dl,0]),scale=5,width=4e-3)
    
    # Clabel
    if s == 0:
        manual_locations = [(119.49,23.37)]
    elif s == 1:
        manual_locations = [(119.56,23.39)]
    elif s == 2:
        manual_locations = [(119.52,23.38)]
    elif s == 3:
        manual_locations = [(119.52,23.39)]
    ax.clabel(cm, fontsize=10, manual=manual_locations)

    #rect = patches.Rectangle((np.min(lon),np.max(lat)-y),x,y,edgecolor='k',facecolor='w')
    #currentaxis.add_patch(rect)
    #plt.quiverkey(Q,.09,.95,.3,label='0.3 m/s',labelpos='S')
    plt.quiverkey(Q,-.1,0,.3,label='0.3 m/s',labelpos='S', fontproperties={'size':8})

    ax.set_facecolor('darkgray')
    ax.set_yticks(ytick)
    ax.set_yticklabels([f'{t:.2f}\N{DEGREE SIGN}N' for t in ytick])
    ax.set_xticks(xtick)
    ax.set_xticklabels([f'{t:.2f}\N{DEGREE SIGN}E' for t in xtick])
    ax.grid(zorder=0)
    #fig.savefig('seasonal_map_d01_mean_'+sea_mon[s]+'.png',bbox_inches='tight', transparent=True, dpi=330)

# Colorbar
fig,ax = plt.subplots(figsize=(1,3))
plt.axis('off')
cbar_ax = fig.add_axes([.3,0.13,0.1,0.7])
cbar = fig.colorbar(cn,cax=cbar_ax,orientation='vertical',extend='both',ticks=np.arange(20,30.1,2))
cbar.ax.set_title('\N{DEGREE SIGN}C')
fig.savefig('seasonal_map_d01_mean_cbar.png',bbox_inches='tight', transparent=True, dpi=330)
plt.show()

# %% Difference =====================================================
vmin, vmax = .5, 1.5
clev = np.arange(vmin,vmax+.01,.05)
clev1 = np.arange(vmin+.1,vmax+.01,.1)

fig,ax = plt.subplots(figsize=(10,6),nrows=2,ncols=2,facecolor='white')
plt.subplots_adjust(wspace=.1,hspace=.2)
ax = ax.flatten()

for s in range(4):
    tmp = ts[s,:,:,1]-ts[s,:,:,0]
    tmpu = us[s,:,:,1]-us[s,:,:,0]
    tmpv = vs[s,:,:,1]-vs[s,:,:,0]
    cn = ax[s].contourf(lon,lat,tmp,cmap='YlOrRd',vmin=vmin,vmax=vmax,levels=clev,extend='both')
    cm = ax[s].contour(lon,lat,tmp,levels=clev,colors='k',linewidths=.5)
    cm = ax[s].contour(lon,lat,tmp,levels=clev1,colors='k',linewidths=.7)
    Q = ax[s].quiver(lon[::dl],lat[::dl],np.squeeze(tmpu[::dl,::dl]),\
                     np.squeeze(tmpv[::dl,::dl]),scale=.5,width=4e-3)
    
    # Clabel
    if s == 0:
        manual_locations = [(119.49,23.37)]
    elif s == 1:
        manual_locations = [(119.56,23.39),(119.49,23.38)]
    elif s == 2:
        manual_locations = [(119.56,23.385),(119.52,23.38)]
    elif s == 3:
        manual_locations = [(119.52,23.39),(119.49,23.39)]
    ax[s].clabel(cm, fontsize=10, manual=manual_locations)

    ax[s].set_facecolor('darkgray')
    ax[s].set_yticks(ytick)
    if s==1 or s==3:
        ax[s].set_yticklabels([])
    else:
        ax[s].set_yticklabels([f'{t:.2f}\N{DEGREE SIGN}N' for t in ytick])
    ax[s].set_xticks(xtick)
    if s<=1:
        ax[s].set_xticklabels([])
    else:
            ax[s].set_xticklabels([f'{t:.2f}\N{DEGREE SIGN}E' for t in xtick])
    ax[s].set_title(sea_str[s])
    ax[s].grid(zorder=0)
    
plt.quiverkey(Q,1.15,-.02,.03,label='0.03 m/s',labelpos='S')
cbar_ax = fig.add_axes([.95,0.13,0.01,0.7])
cbar = fig.colorbar(cn,cax=cbar_ax,orientation='vertical',extend='both',ticks=np.arange(.5,1.6,.2))
cbar.ax.set_title('\N{DEGREE SIGN}C')
plt.show()

#%% Plot SST and current for ITRI-----------------------------------------
clev1 = np.arange(vmin+.1,vmax+.01,.1)
x, y = 0.025, 0.013

for s in range(4):
    fig,ax = plt.subplots(figsize=(5,3),nrows=1,ncols=1,facecolor='white')
    currentaxis = fig.gca()

    tmp = ts[s,:,:,1]-ts[s,:,:,0]
    tmpu = us[s,:,:,1]-us[s,:,:,0]
    tmpv = vs[s,:,:,1]-vs[s,:,:,0]
    ax.contourf(lon,lat,mask,cmap='gray',vmin=-10,vmax=10)
    cn = ax.contourf(lon,lat,tmp,cmap='YlOrRd',vmin=vmin,vmax=vmax,levels=clev,extend='both')
    cm = ax.contour(lon,lat,tmp,levels=clev,colors='k',linewidths=.5)
    cm = ax.contour(lon,lat,tmp,levels=clev1,colors='k',linewidths=.7)
    Q = ax.quiver(lon[::dl],lat[::dl],np.squeeze(tmpu[::dl,::dl]),\
                     np.squeeze(tmpv[::dl,::dl]),scale=.5,width=4e-3)

    # Clabel
    if s == 0:
        manual_locations = [(119.49,23.37)]
    elif s == 1:
        manual_locations = [(119.56,23.39),(119.49,23.38)]
    elif s == 2:
        manual_locations = [(119.56,23.385),(119.52,23.38)]
    elif s == 3:
        manual_locations = [(119.52,23.39),(119.49,23.39)]
    ax.clabel(cm, fontsize=10, manual=manual_locations)

    rect = patches.Rectangle((np.min(lon),np.max(lat)-y),x,y,edgecolor='k',facecolor='w')
    currentaxis.add_patch(rect)
    plt.quiverkey(Q,.09,.95,.03,label='0.03 m/s',labelpos='S')
    #plt.quiverkey(Q,-.1,0,.03,label='0.03 m/s',labelpos='S', fontproperties={'size':8})

    ax.set_facecolor('darkgray')
    ax.set_yticks(ytick)
    ax.set_yticklabels([f'{t:.2f}\N{DEGREE SIGN}N' for t in ytick])
    ax.set_xticks(xtick)
    ax.set_xticklabels([f'{t:.2f}\N{DEGREE SIGN}E' for t in xtick])
    ax.grid(zorder=0)
    #fig.savefig('seasonal_map_d01_diff_'+sea_mon[s]+'_qbox.png',bbox_inches='tight', transparent=True, dpi=330)

# Colorbar
fig,ax = plt.subplots(figsize=(1,3))
plt.axis('off')
cbar_ax = fig.add_axes([.3,0.13,0.1,0.7])
cbar = fig.colorbar(cn,cax=cbar_ax,orientation='vertical',extend='both',ticks=np.arange(.5,1.6,.2))
cbar.ax.set_title('\N{DEGREE SIGN}C')
fig.savefig('seasonal_map_d01_diff_cbar.png',bbox_inches='tight', transparent=True, dpi=330)
plt.show()
