#%%
import pandas as pd
import numpy as np
import datetime
import netCDF4 as nc
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

#%% =================================================================
# Temperature - Buoy
# ===================================================================
# Read data ---------------------------------------------------------
to, time = [], []
for y in range(2013, 2020):
    file_path = f'd00/cwa_obs/{y}_C5W09_dongi_stemp.csv'
    print(file_path)
    with open(file_path) as fnm:
        tmp = pd.read_csv(fnm, header=8).to_numpy()

# Remove repeated time ----------------------------------------------
    time_u, loc = np.unique(tmp[:,1],return_index=True)
    to_r = tmp[loc,3] 
    time_r = pd.to_datetime(time_u.astype(str), format="%Y%m%d%H%M")
    time_r = np.array(time_r)
    del loc,time_u,tmp

# Daily mean --------------------------------------------------------
    days = np.unique(time_r.astype('datetime64[D]'))
    to_m = []
    for day in days:
        mask = (time_r.astype('datetime64[D]') == day)
        if (np.sum(mask)) <= 12:
            to_m.append(np.nan)
        else:
            to_m.append(np.nanmean(to_r[time_r.astype('datetime64[D]') == day], axis=0))
    to_m = np.array(to_m)
    time_d = days
    del days,to_r,time_r,mask,day

# Fill missing data with nan-----------------------------------------
    time_f = pd.date_range(datetime(y,1,1,0),datetime(y,12,31,23),freq='1d')
    to_f = np.full(time_f.shape, np.nan)
    to_f[np.isin(time_f,time_d)] = to_m
    del time_d,to_m

# Leap year ---------------------------------------------------------
    if y == 2016:
        mask = ~((time_f.month == 2) & (time_f.day == 29))
        to_f, time_f = to_f[mask], time_f[mask]
        del mask

# Merge -------------------------------------------------------------
    to.append(to_f)
    time.append(time_f)
del to_f,time_f,y,file_path,fnm

# Climatology -------------------------------------------------------
toc_per = np.nanpercentile(to,[25,75],axis=0)
toc = np.nanmean(np.array(to),axis=0)
timec = np.array(time)[0,:]
del to,time

#%% =================================================================
# Temperature - ROMS dph
# ===================================================================
# Read data ---------------------------------------------------------
root = '/data3/jhluna/work/ITRI_UCH_phdw/DATA/'
with nc.Dataset(root+'itri_uch_avg_d00c_his_zlev.nc') as rootgrp:
    tmp = np.array(rootgrp.variables['to'][:,3,1:-1,1:-1])
    lat = np.array(rootgrp.variables['lat'][1:-1,1])
    lon = np.array(rootgrp.variables['lon'][1,1:-1])
    tmp[np.where((tmp==0)|(tmp>35))[:]] = np.nan

ilon = np.nanargmin(abs(lon-119.6839))
ilat = np.nanargmin(abs(lat-23.2564))
tmp = tmp[:,ilat,ilon]

y = 2013
st = datetime(y,1,1,0,15,0,0)
en = datetime(y,12,27,0,16,0,0)
time_roms = np.arange(st,en,timedelta(hours=3)).astype(datetime)

del st,en,rootgrp,y,lat,lon,ilat,ilon

# Daily mean --------------------------------------------------------
days = np.unique(time_roms.astype('datetime64[D]'))
toc_roms = []
for day in days:
    toc_roms.append(np.nanmean(tmp[time_roms.astype('datetime64[D]') == day], axis=0))
toc_roms = np.array(toc_roms)
time_roms = days
del days,day,tmp

#%% =======================================================
# Plot
# =========================================================
time_numeric = mdates.date2num(timec)

fig = plt.figure(figsize=(8,3),facecolor='white')
ax = fig.add_subplot(111)
ax.plot(time_numeric,toc,'-k')
ax.plot(time_numeric[:360],toc_roms[:360],'r',linewidth=2)
ax.fill_between(time_numeric,toc_per[0,:],toc_per[1,:],alpha=0.3,facecolor='k')
plt.legend(['Observation (2013-2019)','ROMS dph (1985-2014)'])
ax.set_xlabel('Month',fontsize=13)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m'))
ax.xaxis.set_major_locator(mdates.MonthLocator())  # Set ticks to the start of each month
plt.xticks(fontsize=13)
plt.ylim([20,28])
plt.yticks(np.arange(21,28,2),fontsize=13)
ax.set_title('Temperature (\N{DEGREE SIGN}C)',fontsize=15,fontweight='bold',loc='left')
ax.text(15857,28.3,"Dongji Island (23\N{DEGREE SIGN}15\'23\'\'N, 119\N{DEGREE SIGN}41\'2\'\'E)",\
             fontsize=13)

plt.grid()

#%% ==========================================================
# Quantitive validation
# ============================================================
df = pd.DataFrame({'A': toc[:360], 'B': toc_roms[:360]})
print(df.corr())

rmse = np.sqrt(np.nanmean((toc_roms[:360]-toc[:360])**2))
print(rmse)