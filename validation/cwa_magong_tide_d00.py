#%%
import pandas as pd
import numpy as np
import datetime
import netCDF4 as nc
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

#%% =======================================================
# Observation
# =========================================================
# Read data -----------------------------------------------
fnm = open('/data3/jhluna/work/ITRI_penghu/d00/cwa_obs/2013_1356_magong_tide.csv')
df = np.array(pd.read_csv(fnm,header=20))
tide = df[:,2:12]
tide = tide.ravel()*1e-3  # unit: m

# Fill missing data with nan--------------------------------
timed = df[:,1]           # daily time
timed = pd.to_datetime(timed.astype(str), format="%Y%m%d%H")
timem = np.concatenate([pd.date_range(start, periods=10, freq="6T") for start in timed])
time = pd.date_range(datetime(2013,1,1,0,0,0,0),datetime(2013,12,31,23,55,0,0),freq="6T")
time = np.array(time)

tide_f = np.full(time.shape, np.nan)
matched_indices = np.isin(time, timem) 
tide_f[matched_indices] = tide
tide = tide_f
del timed,timem,tide_f,matched_indices,df,fnm

#%% =======================================================
# ROMS
# =========================================================
# Read ncfile ---------------------------------------------
root = '/data3/jhluna/work/ITRI_UCH_phdw/DATA/'
with nc.Dataset(root+'itri_uch_avg_d00_his_zlev.nc') as rootgrp:
    lat = np.array(rootgrp.variables['lat'][:,1])
    lon = np.array(rootgrp.variables['lon'][1,:])
    zeta = np.array(rootgrp.variables['zeta'][:])

    time_roms = rootgrp.variables['time'][:]
    time_u = rootgrp.variables['time'].units 
    time_roms = np.array(nc.num2date(time_roms,units=time_u,calendar='standard'))
    time_roms = pd.to_datetime(time_roms, format='%Y-%m-%d %H:%M:%S.%f').to_numpy() + np.timedelta64(8,'h')
del rootgrp,time_u

# Magong loc -----------------------------------------------
ilon = np.nanargmin(abs(lon-119.546))
ilat = np.nanargmin(abs(lat-23.57))
zeta_magong = zeta[:,ilat,ilon]
del ilat,ilon,zeta

# Shift standard sea level ---------------------------------
mn_obs = np.nanmean(tide,dtype='float32')
mn_roms = np.nanmean(zeta_magong,dtype='float32')
zeta_magong = zeta_magong + (mn_obs-mn_roms) 
del mn_obs,mn_roms

#%% ========================================================
# Plot 
# ==========================================================
tmp = mdates.date2num(time)
time_numeric = mdates.date2num(time_roms)

fig,ax = plt.subplots(figsize=(10,5),nrows=2,ncols=1,facecolor='white')
plt.subplots_adjust(hspace=.3)

ax[0].plot(time,tide,linewidth=.4)
ax[0].set_ylim([-2.2,2.2])
ax[0].set_xlim([15706.333333333334,16066.458333333334])
ax[0].set_xticklabels([])
ax[0].set_ylabel('Height (m)',fontsize=12)
ax[0].set_title('Magong Tide Station',fontsize=13,fontweight='bold')
#ax[0].set_title('Magong Tide Station',fontsize=11,fontweight='bold',loc='left',x=.02,y=.82)
ax[0].grid()

ax[1].plot(time_numeric,zeta_magong,linewidth=.4)
ax[1].set_ylim([-2.2,2.2])
ax[1].set_xlim([15706.333333333334,16066.458333333334])
ax[1].set_ylabel('Height (m)',fontsize=12)
ax[1].set_xlabel('Month',fontsize=13)
ax[1].xaxis.set_major_formatter(mdates.DateFormatter('%m'))
ax[1].xaxis.set_major_locator(mdates.MonthLocator())  # Set ticks to the start of each month
ax[1].set_title('ROMS Simulation',fontsize=13,fontweight='bold')
#ax[1].set_title('ROMS Simulation',fontsize=11,fontweight='bold',loc='left',x=.02,y=.82)
ax[1].grid()
#fig.savefig('tide_check.png',bbox_inches='tight',dpi=300)
plt.show()

#%% ==========================================================
# Quantitive validation
# ============================================================
import pandas as pd

arr1 = time_roms.astype('datetime64[s]').astype(float)
arr2 = time.astype('datetime64[s]').astype(float)

tide_it = np.interp(arr2,arr1,np.squeeze(zeta_magong))
df = pd.DataFrame({'A': tide_it, 'B': tide})
print(df.corr())
del arr1,arr2

rmse = np.sqrt(np.nanmean((tide_it-tide)**2))

cwa_std = np.nanstd(tide)
roms_std = np.nanstd(zeta_magong)
print(roms_std)