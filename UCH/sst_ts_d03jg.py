#%%
import glob
import pandas as pd
import numpy as np
import datetime
import netCDF4 as nc
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

#%% Read d03jg ncfile ===============================================
# Initialize
root = '/data3/jhluna/work/ITRI_UCH_phdw/DATA/'
tso_jg, tbo_jg = [], []
scen = ['his','ssp']

for sn, sc in enumerate(scen):
    fnl = sorted(glob.glob(root+'raw/d03gb/'+sc+'/tccip_ocean_avg_d03gb_*'))
    tso, tbo, time = [], [], []

    # Read each avg file
    for fnm in fnl:
        print(fnm)
        with nc.Dataset(fnm) as rootgrp:

            ts = np.array(rootgrp.variables['temp'][:,-1,1:-1,1:-1])
            tb = np.array(rootgrp.variables['temp'][:,0,1:-1,1:-1])
            lat = np.array(rootgrp.variables['lat_rho'][1:-1,1])
            lon = np.array(rootgrp.variables['lon_rho'][1,1:-1])
            ti = rootgrp.variables['ocean_time'][:]
            tu = rootgrp.variables['ocean_time'].units
            ti = nc.num2date(ti,units=tu,calendar='standard')

            tso.append(ts)
            tbo.append(tb)
            time.append(ti)

    tso = np.concatenate(tso, axis=0)   # (time,lat,lon)
    tbo = np.concatenate(tbo, axis=0)
    tso[np.where((tso==0)|(tso>35))[:]] = np.nan
    tbo[np.where((tbo==0)|(tbo>35))[:]] = np.nan
    time = np.concatenate(time, axis=0)
    del ts,tb,rootgrp,fnm,ti,tu

    # Delete the spin-up period
    get_year = np.vectorize(lambda date: date.year)
    yr = get_year(time.compressed())
    it = np.where(yr==2012)[0]
    tso = np.delete(tso,it,axis=0)
    tbo = np.delete(tbo,it,axis=0)
    time = np.delete(time,it,axis=0)
    tso = tso[::6,:,:]
    tbo = tbo[::6,:,:]
    time = time[::6]

    # Area mean or point
    blon, blat = 119.5394, 23.35106
    ilon = np.argmin(np.abs(lon-blon))
    ilat = np.argmin(np.abs(lat-blat))
    tso_jg.append(tso[:,ilat,ilon])
    tbo_jg.append(tbo[:,ilat,ilon])
    del it,yr,get_year,fnl,ilon,ilat,blon,blat,tbo,tso

#%% Plotting ========================================================
fig = plt.figure(figsize=(7,4),facecolor='white')
ax = fig.add_subplot(111)
ax2 = ax.twinx()

ax.plot(np.linspace(1,12+27/31,len(tso_jg[0])),tso_jg[0],color='slateblue',\
        linewidth=8,alpha=.3)
l1,=ax.plot(np.linspace(1,12+27/31,len(tbo_jg[0])),tbo_jg[0],color='slateblue')

ax.plot(np.linspace(1,12+27/31,len(tso_jg[1])),tso_jg[1],color='gold',\
        linewidth=8,alpha=.3)
l2,=ax.plot(np.linspace(1,12+27/31,len(tbo_jg[1])),tbo_jg[1],color='gold')

ax.set_xlim([1,13])
ax.set_ylim([19,30])
ax.set_xlabel('Month',fontsize=13)
ax.set_ylabel('Mean',fontsize=13)
ax.set_xticks(ticks=np.arange(2,13,2))
ax.set_yticks(ticks=np.arange(20,31,2))
ax.grid()

ax2.plot(np.linspace(1,12+27/31,len(tbo_jg[0])),tso_jg[1]-tso_jg[0],color='firebrick',\
        linewidth=8,alpha=.3)
l3,=ax2.plot(np.linspace(1,12+27/31,len(tbo_jg[0])),tbo_jg[1]-tbo_jg[0],color='firebrick')

ax2.set_xlim([1,13])
ax2.set_ylim([0,2.2])
ax2.set_xlabel('Month',fontsize=13)
ax2.set_ylabel('Difference',fontsize=13)
ax2.set_yticks(ticks=np.arange(.2,2.3,.4))
ax2.yaxis.label.set_color(l3.get_color())
ax2.tick_params(axis='y',colors=l3.get_color())

plt.legend([l1,l2,l3],['Historical (1985-2014)','SSP2-4.5 (2040-2069)','Difference'],loc='upper left')
ax.set_title('Temperature (\N{DEGREE SIGN}C)',fontsize=15,\
             fontweight='bold',loc='left')
ax.text(9.7,30.3,'Jiangjun location',fontsize=13)
#fig.savefig('d01/sat_dawen_sst.png',bbox_inches='tight',dpi=300)
plt.show()

