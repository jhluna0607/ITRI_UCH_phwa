#%%
import glob
import pandas as pd
import numpy as np
import datetime
import netCDF4 as nc
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.ticker as mtick
from windrose import WindroseAxes

# Define function
def cc2pc(uo,vo):
    mag = np.sqrt(uo**2+vo**2)
    dir = []
    for i in range(len(uo)):
        uSign, vSign = np.sign(uo[i]), np.sign(vo[i])
        # Quadran 1
        if (uSign > 0 and vSign > 0):
            dir.append( np.rad2deg( np.arctan( uo[i] / vo[i] )))
        # Quadran 2
        elif (uSign > 0 and vSign < 0):
            dir.append( 90 + np.rad2deg( np.arctan( vo[i] / uo[i] )))
        # Quadran 3
        elif (uSign < 0 and vSign < 0):
            dir.append(180 + np.rad2deg( np.arctan( abs(uo[i]) / abs(vo[i]) )))
        # Quadran 4
        elif (uSign < 0 and vSign > 0):
            dir.append(270 + np.rad2deg( np.arctan( vo[i] / abs(uo[i]) )))
        elif (np.isnan(uo[i]) or np.isnan(vo[i])):
            dir.append(np.nan)
    dir = np.asarray(dir)
    return (mag,dir)


#%% Read ncfile =====================================================
# Initialize --------------------------------------------------------
root = '/data3/jhluna/work/ITRI_UCH_phdw/DATA/'
scen = ['his','ssp']
blon, blat = 119.5565, 23.3495
sea_mon = [[12,2],[3,5],[6,8],[9,11]]
sea_sss = ['djf','mam','jja','son']
sea_str = ['Winter','Spring','Summer','Autumn']
mag, dir, um, vm = {}, {}, {}, {}

# Read each scenario ------------------------------------------------
for sn, sc in enumerate(scen):
    u, v, time = [], [], []
    fnl = sorted(glob.glob(root+'raw/d03gb/'+sc+'/tccip_ocean_avg_d03gb_*'))

    # Read each avg file
    for fnm in fnl:
        print(fnm)

        with nc.Dataset(fnm) as rootgrp:

            lat = np.array(rootgrp.variables['lat_rho'][1:-1,1])
            lon = np.array(rootgrp.variables['lon_rho'][1,1:-1])
            ilon = np.argmin(np.abs(lon-blon))
            ilat = np.argmin(np.abs(lat-blat))

            u.append(np.array(rootgrp.variables['u_eastward'][:,0,ilat+1,ilon+1]))
            v.append(np.array(rootgrp.variables['v_northward'][:,0,ilat+1,ilon+1]))

            t = rootgrp.variables['ocean_time'][:]
            tu = rootgrp.variables['ocean_time'].units
            t = nc.num2date(t,units=tu,calendar='standard')
            time.append(t)

    u = np.concatenate(u, axis=0)      # (time)
    v = np.concatenate(v, axis=0)
    time = np.concatenate(time, axis=0)
    del rootgrp,fnm,t,tu,lon,lat,ilon,ilat

    # Delete the spin-up period
    get_year = np.vectorize(lambda date: date.year)
    yr = get_year(time.compressed())
    it = np.where(yr==2012)[0]
    u, v = np.delete(u,it,axis=0), np.delete(v,it,axis=0)
    time = np.delete(time,it,axis=0)
    del it,yr,get_year

# Seasonal mean ---------------------------------------------------
    get_month = np.vectorize(lambda date: date.month)
    mon = get_month(time.compressed())

    # Each season
    for s, sea in enumerate(sea_mon):

        # Seasonal location
        if s != 0:
            loc = np.where((mon>=sea[0]) & (mon<=sea[1]))[0]
        else:
            loc = np.where((mon>=sea[0]) | (mon<=sea[1]))[0]
        us, vs = u[loc], v[loc]

        # Convert (u,v) to (mag,dir)
        keynm = f'{sc}_{sea_sss[s]}'
        [mag[keynm], dir[keynm]] = cc2pc(us,vs)
        um[keynm], vm[keynm] = np.array([np.nanmean(us)]), np.array([np.nanmean(vs)])

    del get_month,mon,s,sea,loc,us,vs,keynm
del blon,blat,fnl,root,sn,sc

#%% Plot figure =======================================================
# All plots -----------------------------------------------------------
# Open a new canva
fig, ax = plt.subplots(ncols=4, nrows=2, figsize=(14,7),\
                        subplot_kw={'projection': 'windrose'})
plt.subplots_adjust(wspace=.3, hspace=.1)
ax = ax.flatten()
x = 0

# Each scenario
for sn, sc in enumerate(scen):

    # Each season
    for s, sea in enumerate(sea_mon):
        keynm = f'{sc}_{sea_sss[s]}'

        # Plot the Current Rose 
        ax[x].bar(dir[keynm], mag[keynm], normed=True, opening=.9, edgecolor='white',\
                  bins = np.arange(0,1.3,.2), cmap=cm.YlGnBu)
        
        # Plot the Quiver
        Q = ax[x].quiver(0, 0, um[keynm], vm[keynm], scale=2.7, color='black')
        if sn == 1:
            tmpu = um[f'ssp_{sea_sss[s]}'] - um[f'his_{sea_sss[s]}']
            tmpv = vm[f'ssp_{sea_sss[s]}'] - vm[f'his_{sea_sss[s]}']
            Qr = ax[x].quiver(0, 0, tmpu, tmpv, scale=.027, color='red')   
        else:
            ax[x].set_title(sea_str[s], fontsize=16, fontweight=550, y=1.2)
        ax[x].set_yticks(np.arange(20,81,20))
        ax[x].set_yticklabels(['20%','40%','60%','80%'], fontsize=8)
        ax[x].grid(linewidth=.2)
        x = x+1

plt.quiverkey(Q,1.45,.72,.3,label='0.3 m/s',labelpos='E')
plt.quiverkey(Qr,1.45,.62,.003,label='0.003 m/s',labelpos='E')
ax[3].legend(title='Current Speed (m/s)', bbox_to_anchor=(1.2,-.45))
ax[3].text(1.29, .45, 'Guangbing', fontsize=14, fontweight='bold',transform=ax[3].transAxes)

ax[0].text(-0.25, 0.5, 'Historical', va='bottom', ha='center', rotation='vertical', \
           rotation_mode='anchor', transform=ax[0].transAxes, fontsize=16, fontweight='bold')
ax[4].text(-0.25, 0.5, 'SSP 2-4.5', va='bottom', ha='center', rotation='vertical', \
           rotation_mode='anchor', transform=ax[4].transAxes, fontsize=16, fontweight='bold')

plt.show()

#%% Plot figure for ITRI ==============================================
# Each scenario
for sn, sc in enumerate(scen):

    # Each season
    for s, sea in enumerate(sea_mon):

        # Open a new canva
        keynm = f'{sc}_{sea_sss[s]}'
        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(3,3),\
                               subplot_kw={'projection': 'windrose'})

        # Plot the Current Rose 
        ax.bar(dir[keynm], mag[keynm], normed=True, opening=.9, edgecolor='white',\
                  bins = np.arange(0,1.3,.2), cmap=cm.YlGnBu)
        
        # Plot the Quiver
        Q = ax.quiver(0, 0, um[keynm], vm[keynm], scale=2.7, color='black')
        v = np.sqrt(um[keynm]**2+vm[keynm]**2)
        ax.text(160*np.pi/180, 30, f'{v[0]:.2f} m/s', fontsize=8)
        if sn == 1:
            tmpu = um[f'ssp_{sea_sss[s]}'] - um[f'his_{sea_sss[s]}']
            tmpv = vm[f'ssp_{sea_sss[s]}'] - vm[f'his_{sea_sss[s]}']
            v = np.sqrt(tmpu**2+tmpv**2)
            Qr = ax.quiver(0, 0, tmpu, tmpv, scale=.027, color='red')
            ax.text(330*np.pi/180, 10, f'{v[0]:.4f} m/s', fontsize=8, color='red') 
        ax.set_yticks(np.arange(20,81,20))
        ax.set_yticklabels(['20%','40%','60%','80%'], fontsize=8)
        ax.grid(linewidth=.2)
        #fig.savefig('wind_rose_d03gb_'+sc+'_'+sea_sss[s]+'.png', \
        #            bbox_inches='tight', transparent=True, dpi=330)


# Legend
fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(3,3),\
                       subplot_kw={'projection': 'windrose'})
ax.bar([0], [0], normed=True, opening=.9, edgecolor='white',\
        bins = np.arange(0,1.3,.2), cmap=cm.YlGnBu)
plt.axis('off')
ax.legend(title='Current Speed (m/s)', bbox_to_anchor=(.7,.5))
#fig.savefig('wind_rose_d03gb_legend.png', \
#                    bbox_inches='tight', transparent=True, dpi=330)

