#%%
import pandas as pd
import numpy as np
import datetime
import netCDF4 as nc
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Define function (mag,dir) -> (uo,vo)
def pc2cc(mag,dir):
    dd = np.zeros(dir.shape)
    dd = -1*(dir-360)
    for i in range(dd.size):
        if (dd[i]>=270) & (dd[i]<=360):
            dd[i] = dd[i]-270
        else:
            dd[i] = dd[i]+90
    dd = dd.astype(float)
    uu = mag*np.cos(np.deg2rad(dd))
    vv = mag*np.sin(np.deg2rad(dd))
    return (uu,vv)

#%% =================================================================
# Current - Buoy
# ===================================================================
# Read data ---------------------------------------------------------
uo, vo, time = [], [], []
for y in range(2015, 2024):
    #if y >= 2016:
    #    continue
    file_path = f'd00/cwa_obs/{y}_C6W10_cimei_curr.csv'
    print(file_path)
    with open(file_path) as fnm:
        tmp = pd.read_csv(fnm, header=8).to_numpy()

# Remove repeated time ----------------------------------------------
    time_u, loc = np.unique(tmp[:,1], return_index=True)
    vo_r, do_r = tmp[loc,3], tmp[loc,4]
    time_r = pd.to_datetime(time_u.astype(str), format="%Y%m%d%H%M")
    time_r = np.array(time_r)
    del time_u,loc,tmp

# (vo,do) -> (u,v) --------------------------------------------------
    [uo_r, vo_r] = pc2cc(vo_r, do_r) 

# Daily mean --------------------------------------------------------
    days = np.unique(time_r.astype('datetime64[D]'))
    uo_m, vo_m = [], []
    for day in days:
        mask = (time_r.astype('datetime64[D]') == day)
        if (np.sum(mask)) <= 12:
            uo_m.append(np.nan)
            vo_m.append(np.nan)
        else:
            uo_m.append(np.nanmean(uo_r[time_r.astype('datetime64[D]') == day], axis=0))
            vo_m.append(np.nanmean(vo_r[time_r.astype('datetime64[D]') == day], axis=0))
    uo_m = np.array(uo_m)
    vo_m = np.array(vo_m)
    time_d = days
    del days,do_r,uo_r,vo_r,time_r,mask,day

# Fill missing data with nan-----------------------------------------
    time_f = pd.date_range(datetime(y,1,1,0),datetime(y,12,31,23),freq='1d')
    uo_f, vo_f = np.full(time_f.shape,np.nan), np.full(time_f.shape,np.nan)
    uo_f[np.isin(time_f,time_d)] = uo_m
    vo_f[np.isin(time_f,time_d)] = vo_m
    del uo_m,vo_m,time_d

# Leap year ---------------------------------------------------------
    if (y==2016) or (y==2020):
        mask = ~((time_f.month == 2) & (time_f.day == 29))
        uo_f, vo_f, time_f = uo_f[mask], vo_f[mask], time_f[mask]
        del mask

# Merge -------------------------------------------------------------
    uo.append(uo_f)
    vo.append(vo_f)
    time.append(time_f)
del uo_f,vo_f,time_f,y,file_path,fnm

# Climatology -------------------------------------------------------
uoc_per = np.nanpercentile(uo,[25,75],axis=0)
voc_per = np.nanpercentile(vo,[25,75],axis=0)
uoc = np.nanmean(np.array(uo),axis=0)
voc = np.nanmean(np.array(vo),axis=0)
timec = np.array(time)[0,:]
del uo,vo,time

#%% =================================================================
# Current - ROMS dph
# ===================================================================
# Read data ---------------------------------------------------------
root = '/data3/jhluna/work/ITRI_UCH_phdw/DATA/'
with nc.Dataset(root+'itri_uch_avg_d00c_his_zlev.nc') as rootgrp:
    tmpu = np.array(rootgrp.variables['uo'][:,0,1:-1,1:-1])
    tmpv = np.array(rootgrp.variables['vo'][:,0,1:-1,1:-1])
    lat = np.array(rootgrp.variables['lat'][1:-1,1])
    lon = np.array(rootgrp.variables['lon'][1,1:-1])

ilon = np.nanargmin(abs(lon-119.66))
ilat = np.nanargmin(abs(lat-23.23))
d1, d2 = 3, 2
tmpu = np.nanmean(np.nanmean(tmpu[:,ilat-d1:ilat+d1+1,ilon-d2:ilon+d2+1],axis=2),axis=1)
tmpv = np.nanmean(np.nanmean(tmpv[:,ilat-d1:ilat+d2+1,ilon-d2:ilon+d2+1],axis=2),axis=1)

y = 2013
st = datetime(y,1,1,0,15,0,0)
en = datetime(y,12,27,0,16,0,0)
time_roms = np.arange(st,en,timedelta(hours=3)).astype(datetime)
del st,en,rootgrp,y,lat,lon,ilat,ilon,d1,d2

# Daily mean --------------------------------------------------------
days = np.unique(time_roms.astype('datetime64[D]'))
uoc_roms, voc_roms = [], []
for day in days:
    uoc_roms.append(np.nanmean(tmpu[time_roms.astype('datetime64[D]') == day], axis=0))
    voc_roms.append(np.nanmean(tmpv[time_roms.astype('datetime64[D]') == day], axis=0))
uoc_roms = np.array(uoc_roms)
voc_roms = np.array(voc_roms)
time_roms = days
del days,day,tmpu,tmpv


#%% =================================================================
# Plot
# ===================================================================
time_numeric = mdates.date2num(timec)

fig = plt.figure(figsize=(9,6),facecolor='white')
ax = fig.add_subplot(211)
ax.plot(time_numeric,uoc*1e-3,'-k')
ax.plot(time_numeric[:360],uoc_roms[:360],'r',linewidth=2)
ax.fill_between(time_numeric,uoc_per[0,:]*1e-3,uoc_per[1,:]*1e-3,alpha=0.3,facecolor='k')
ax.legend(['Observation (2015-2023)','ROMS dph (1985-2014)'],loc='lower center')
ax.xaxis.set_major_locator(mdates.MonthLocator())  # Set ticks to the start of each month
ax.set_xticklabels([])
plt.yticks(fontsize=13)
plt.ylim([-.35,.5])
plt.yticks(np.arange(-.2,.5,.2),fontsize=13)
ax.set_title('Zonal Current (m/s)',fontsize=15,fontweight='bold',loc='left')
ax.text(16615,0.535,"Chimi Buoy (23\N{DEGREE SIGN}11\'20\'\'N, 119\N{DEGREE SIGN}39\'54\'\'E)",\
             fontsize=13)
plt.grid()

ax = fig.add_subplot(212)
ax.plot(time_numeric,voc*1e-3,'-k')
ax.plot(time_numeric[:360],voc_roms[:360],'r',linewidth=2)
ax.fill_between(time_numeric,voc_per[0,:]*1e-3,voc_per[1,:]*1e-3,alpha=0.3,facecolor='k')
ax.legend(['Observation (2015-2023)','ROMS dph (1985-2014)'])
ax.set_title('Meridional Current (m/s)',fontsize=15,fontweight='bold',loc='left')
plt.ylim([-.25,.55])
ax.set_xlabel('Month',fontsize=13)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m'))
ax.xaxis.set_major_locator(mdates.MonthLocator())  # Set ticks to the start of each month
plt.xticks(fontsize=13)
plt.yticks(np.arange(-.2,.5,.2),fontsize=13)
plt.grid()

#%% =================================================================
# Quantitive validation
# ===================================================================
df = pd.DataFrame({'A': uoc[:360]*1e-3, 'B': uoc_roms[:360]})
print(df.corr())
df = pd.DataFrame({'A': voc[:360]*1e-3, 'B': voc_roms[:360]})
print(df.corr())

rmse = np.sqrt(np.nanmean((uoc_roms[:360]-uoc[:360]*1e-3)**2))
print(rmse)
rmse = np.sqrt(np.nanmean((voc_roms[:360]-voc[:360]*1e-3)**2))
print(rmse)