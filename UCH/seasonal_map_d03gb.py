#%%
import glob
import pandas as pd
import numpy as np
import datetime
import netCDF4 as nc
import cmaps
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import matplotlib.patches as patches

def reshape(thelist, rows, cols):  
    return [thelist[i:i + cols] for i in range(0, len(thelist), cols)]

#%% Read ncfile =====================================================
# Initialize
root = '/data3/jhluna/work/ITRI_UCH_phdw/DATA/'
us, vs, ts = [], [], []
scen = ['his', 'ssp']
ilev = 0

for sn, sc in enumerate(scen):
    uo, vo, to, time = [], [], [], []
    fnl = sorted(glob.glob(root+'raw/d03gb/'+sc+'/tccip_ocean_avg_d03gb_*'))

    # Read each avg file
    for fnm in fnl:
        print(fnm)
        with nc.Dataset(fnm) as rootgrp:

            h = np.array(rootgrp.variables['h'][1:-1,1:-1])
            u = np.array(rootgrp.variables['u_eastward'][:,ilev,1:-1,1:-1])
            v = np.array(rootgrp.variables['v_northward'][:,ilev,1:-1,1:-1])
            t = np.array(rootgrp.variables['temp'][:,ilev,1:-1,1:-1])
            lat = np.array(rootgrp.variables['lat_rho'][1:-1,1])
            lon = np.array(rootgrp.variables['lon_rho'][1,1:-1])
            mask = np.array(rootgrp.variables['mask_rho'][1:-1,1:-1])

            ti = rootgrp.variables['ocean_time'][:]
            tu = rootgrp.variables['ocean_time'].units
            ti = nc.num2date(ti,units=tu,calendar='standard')

            uo.append(u)
            vo.append(v)
            to.append(t)
            time.append(ti)

    uo = np.concatenate(uo, axis=0)      # (time,lat,lon)
    vo = np.concatenate(vo, axis=0)
    to = np.concatenate(to, axis=0)
    time = np.concatenate(time, axis=0)
    del u,v,t,rootgrp,fnm,ti,tu

    # Delete the spin-up period
    get_year = np.vectorize(lambda date: date.year)
    yr = get_year(time.compressed())
    it = np.where(yr==2012)[0]
    uo, vo, to = np.delete(uo,it,axis=0), np.delete(vo,it,axis=0), np.delete(to,it,axis=0)
    time = np.delete(time,it,axis=0)
    del it,yr,get_year

# Seasonal mean --------------------------------------------------
    get_month = np.vectorize(lambda date: date.month)
    mon = get_month(time.compressed())
    season = [[12,2],[3,5],[6,8],[9,11]]

    for s, sea in enumerate(season):

        if s != 0:
            loc = np.where((mon>=sea[0]) & (mon<=sea[1]))[0]
        else:
            loc = np.where((mon>=sea[0]) | (mon<=sea[1]))[0]

        us.append(np.nanmean(uo[loc,:,:],0))
        vs.append(np.nanmean(vo[loc,:,:],0))
        ts.append(np.nanmean(to[loc,:,:],0))

us, vs, ts = reshape(us,2,4), reshape(vs,2,4), reshape(ts,2,4)
del get_month,mon,season,s,sea,uo,vo,to

#%% Plot SST and current ============================================
xtick = np.arange(119.555,119.561,.005)
ytick = np.arange(23.345,23.3551,.005)

for ss in range(2):
    sea_str = ['Winter','Spring','Summer','Autumn']
    dl = 20
    clev = np.arange(20,30.1,.5)
    cmap = cmaps.cmocean_balance.to_seg(N=20)

    fig,ax = plt.subplots(figsize=(5,10),nrows=2,ncols=2,facecolor='white')
    plt.subplots_adjust(wspace=.15,hspace=.12)
    ax = ax.flatten()

    for s in range(4):
        cn = ax[s].contourf(lon,lat,ts[ss][s],cmap=cmap,vmin=20,vmax=30,levels=clev,extend='both')
        if ilev == -1:
            cm = ax[s].contour(lon,lat,ts[ss][s],levels=np.arange(20,30.1,.5),colors='k',linewidths=.5)
        else:
            if s == 0:
                cm = ax[s].contour(lon,lat,h,levels=np.arange(10,41,10),colors='darkgray',linewidths=.5)
            else:
                cm = ax[s].contour(lon,lat,h,levels=np.arange(10,41,10),colors='dimgrey',linewidths=.5)
        ax[s].clabel(cm, fontsize=8)
        Q = ax[s].quiver(lon[::dl],lat[::dl],np.squeeze(us[ss][s][::dl,::dl]),\
                np.squeeze(vs[ss][s][::dl,::dl]),scale=8,width=4e-3)
        ax[s].scatter(119.5565, 23.3495, s=20, c='yellow')
        ax[s].set_facecolor('darkgray')

        ax[s].set_yticks(ytick)
        if s==1 or s==3:
            ax[s].set_yticklabels([])
        else:
            ax[s].set_yticklabels([f'{t:.3f}\N{DEGREE SIGN}N' for t in ytick])
        ax[s].set_xticks(xtick)
        if s<=1:
            ax[s].set_xticklabels([])
        else:
            ax[s].set_xticklabels([f'{t:.3f}\N{DEGREE SIGN}E' for t in xtick])
        ax[s].set_title(sea_str[s])
        ax[s].grid(zorder=0)
        
    plt.quiverkey(Q,1.32,0,.8,label='0.8 m/s',labelpos='S')
    cbar_ax = fig.add_axes([1,0.13,0.022,0.7])
    cbar = fig.colorbar(cn,cax=cbar_ax,orientation='vertical',extend='both',ticks=np.arange(20,30.1,2))
    cbar.ax.set_title('\N{DEGREE SIGN}C')
    plt.show()

#%% Plot SST and current for ITRI-----------------------------------------
sea_mon = ['DJF','MAM','JJA','SON']
x, y = 0.003, 0.002

for s in range(4):
    fig,ax = plt.subplots(figsize=(2.5,5),nrows=1,ncols=1,facecolor='white')
    currentaxis = fig.gca()

    ax.contourf(lon,lat,mask,cmap='gray',vmin=-10,vmax=10)
    cn = ax.contourf(lon,lat,ts[0][s],cmap=cmap,vmin=20,vmax=30,levels=clev,extend='both')
    if s == 0:
        cm = ax.contour(lon,lat,h,levels=np.arange(10,41,10),colors='darkgray',linewidths=.5)
    else:
        cm = ax.contour(lon,lat,h,levels=np.arange(10,41,10),colors='dimgrey',linewidths=.5)
    Q = ax.quiver(lon[::dl],lat[::dl],np.squeeze(us[0][s][::dl,::dl]),\
            np.squeeze(vs[0][s][::dl,::dl]),scale=8,width=4e-3)
    ax.scatter(119.5565, 23.3495, s=20, c='yellow')

    ax.clabel(cm, fontsize=10)
    #rect = patches.Rectangle((np.max(lon)-x,np.min(lat)),x,y,edgecolor='k',facecolor='w')
    #currentaxis.add_patch(rect)
    #plt.quiverkey(Q,.85,.08,.5,label='0.5 m/s',labelpos='S')
    plt.quiverkey(Q,-.22,.05,.5,label='0.5 m/s',labelpos='S')

    ax.set_facecolor('darkgray')
    ax.set_yticks(ytick)
    ax.set_yticklabels([f'{t:.3f}\N{DEGREE SIGN}N' for t in ytick])
    ax.set_xticks(xtick)
    ax.set_xticklabels([f'{t:.3f}\N{DEGREE SIGN}E' for t in xtick])
    ax.grid(zorder=0)
    fig.savefig('seasonal_map_d03gb_mean_'+sea_mon[s]+'.png',bbox_inches='tight', transparent=True, dpi=330)

# Colorbar
fig,ax = plt.subplots(figsize=(1,5))
plt.axis('off')
cbar_ax = fig.add_axes([.3,0.13,0.1,0.7])
cbar = fig.colorbar(cn,cax=cbar_ax,orientation='vertical',extend='both',ticks=np.arange(20,30.1,2))
cbar.ax.set_title('\N{DEGREE SIGN}C')
fig.savefig('seasonal_map_d03gb_mean_cbar.png',bbox_inches='tight', transparent=True, dpi=330)
plt.show()


#%% Plot SST and current DIFFERENT ==================================
tmp = np.array(ts[1])-np.array(ts[0])
tmpu = np.array(us[1])-np.array(us[0])
tmpv = np.array(vs[1])-np.array(vs[0])

sea_str = ['Winter','Spring','Summer','Autumn']
dl = 20
clev = np.arange(.5,1.55,.05)
cmap = 'YlOrRd'

fig,ax = plt.subplots(figsize=(5,10),nrows=2,ncols=2,facecolor='white')
plt.subplots_adjust(wspace=.15,hspace=.12)
ax = ax.flatten()

for s in range(4):
    cn = ax[s].contourf(lon,lat,tmp[s,:,:],cmap=cmap,vmin=.5,vmax=1.5,levels=clev,extend='both')
    if ilev == -1:
        cm = ax[s].contour(lon,lat,tmp[s,:,:],levels=clev,colors='k',linewidths=.5)
    else:
        if s == 5:
            cm = ax[s].contour(lon,lat,h,levels=np.arange(10,41,10),colors='darkgray',linewidths=.5)
        else:
            cm = ax[s].contour(lon,lat,h,levels=np.arange(10,41,10),colors='dimgrey',linewidths=.5)
    ax[s].clabel(cm, fontsize=8)
    Q = ax[s].quiver(lon[::dl],lat[::dl],np.squeeze(tmpu[s,::dl,::dl]),\
                np.squeeze(tmpv[s,::dl,::dl]),scale=.25,width=4e-3)
    ax[s].scatter(119.5565, 23.3495, s=20, c='yellow')
    ax[s].set_facecolor('darkgray')

    ax[s].set_yticks(ytick)
    if s==1 or s==3:
        ax[s].set_yticklabels([])
    else:
        ax[s].set_yticklabels([f'{t:.3f}\N{DEGREE SIGN}N' for t in ytick])
    ax[s].set_xticks(xtick)
    if s<=1:
        ax[s].set_xticklabels([])
    else:
        ax[s].set_xticklabels([f'{t:.3f}\N{DEGREE SIGN}E' for t in xtick])
    ax[s].set_title(sea_str[s])
    ax[s].grid(zorder=0)
        
plt.quiverkey(Q,1.32,0,.03,label='0.03 m/s',labelpos='S')
cbar_ax = fig.add_axes([1,0.13,0.022,0.7])
cbar = fig.colorbar(cn,cax=cbar_ax,orientation='vertical',extend='both',ticks=np.arange(.5,1.6,.2))
cbar.ax.set_title('\N{DEGREE SIGN}C')
plt.show()

#%% Plot SST and current for ITRI-----------------------------------------
sea_mon = ['DJF','MAM','JJA','SON']
x, y = 0.0032, 0.002

for s in range(4):
    fig,ax = plt.subplots(figsize=(2.5,5),nrows=1,ncols=1,facecolor='white')
    currentaxis = fig.gca()

    ax.contourf(lon,lat,mask,cmap='gray',vmin=-10,vmax=10)
    cn = ax.contourf(lon,lat,tmp[s,:,:],cmap=cmap,vmin=.5,vmax=1.5,levels=clev,extend='both')
    cm = ax.contour(lon,lat,h,levels=np.arange(10,41,10),colors='dimgrey',linewidths=.5)
    Q = ax.quiver(lon[::dl],lat[::dl],np.squeeze(tmpu[s,::dl,::dl]),\
            np.squeeze(tmpv[s,::dl,::dl]),scale=.25,width=4e-3)
    ax.scatter(119.5565, 23.3495, s=20, c='yellow')

    ax.clabel(cm, fontsize=10)
    rect = patches.Rectangle((np.max(lon)-x,np.min(lat)),x,y,edgecolor='k',facecolor='w')
    currentaxis.add_patch(rect)
    plt.quiverkey(Q,.84,.08,.03,label='0.03 m/s',labelpos='S')
    #plt.quiverkey(Q,-.22,.05,.03,label='0.03 m/s',labelpos='S')

    ax.set_facecolor('darkgray')
    ax.set_yticks(ytick)
    ax.set_yticklabels([f'{t:.3f}\N{DEGREE SIGN}N' for t in ytick])
    ax.set_xticks(xtick)
    ax.set_xticklabels([f'{t:.3f}\N{DEGREE SIGN}E' for t in xtick])
    ax.grid(zorder=0)
    fig.savefig('seasonal_map_d03gb_diff_'+sea_mon[s]+'_qbox.png',bbox_inches='tight', transparent=True, dpi=330)

# Colorbar
fig,ax = plt.subplots(figsize=(1,5))
plt.axis('off')
cbar_ax = fig.add_axes([.3,0.13,0.1,0.7])
cbar = fig.colorbar(cn,cax=cbar_ax,orientation='vertical',extend='both',ticks=np.arange(.5,1.6,.2))
cbar.ax.set_title('\N{DEGREE SIGN}C')
#fig.savefig('seasonal_map_d03gb_diff_cbar.png',bbox_inches='tight', transparent=True, dpi=330)
plt.show()


#%% SST Seasonal change bar =========================================
blon, blat = 119.5565, 23.3495
ilon = np.argmin(np.abs(lon-blon))
ilat = np.argmin(np.abs(lat-blat))

tmp = tmp[:,ilat,ilon]
tmpb = {'Guangbing':tmp}
df = pd.DataFrame(tmpb,sea_str)

# Plot
ax = df.plot.bar(rot=0, figsize=(4,4), \
                 color={'Guangbing':'firebrick'})
ax.set_title('Difference (\N{DEGREE SIGN}C)',fontsize=15,fontweight='bold',loc='left')
ax.grid(axis='y',linestyle='--',alpha=0.7,zorder=0)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.ylim([0,1.5])
#plt.text(2.32,1.53,'Case location',fontsize=10)
plt.legend(loc='lower left')
#fig.savefig('sat_wangan_sst.png',bbox_inches='tight',dpi=300)
plt.show()
