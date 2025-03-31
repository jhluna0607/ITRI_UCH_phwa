#%%
import pandas as pd
import numpy as np
import datetime
import cftime
import netCDF4 as nc
from os import walk
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import scipy.stats as stats

def select_wangan(var, lon, lat):
    ilon = np.where((lon>=119.46) & (lon<=119.6))[0]
    ilat = np.where((lat>=23.3296) & (lat<=23.4052))[0]
    [ii,jj] = np.meshgrid(ilon,ilat)
    var_wangan = np.nanmean(np.nanmean(var[:,jj,ii],axis=2),axis=1)
    return var_wangan


def select_season(var, tm, st_yr=0):   
    get_month = np.vectorize(lambda date: date.month)
    mon = get_month(tm.compressed())
    get_year = np.vectorize(lambda date: date.year)
    yr = get_year(tm.compressed())

    it = {'DJF': np.where(((mon>=12)|(mon<=2))&(yr>=1985+st_yr))[0], 
          'MAM': np.where(((mon>=3)&(mon<=5))&(yr>=1985+st_yr))[0],
          'JJA': np.where(((mon>=6)&(mon<=8))&(yr>=1985+st_yr))[0],
          'SON': np.where(((mon>=9)&(mon<=11))&(yr>=1985+st_yr))[0]}
    
    vars = {}
    for sea in it.keys():
        vars[sea] = np.nanmean(var[it[sea]], axis=0)
    return vars


def select_season_std(var, tm, st_yr=0):   
    get_month = np.vectorize(lambda date: date.month)
    mon = get_month(tm.compressed())
    get_year = np.vectorize(lambda date: date.year)
    yr = get_year(tm.compressed())

    it = {'DJF': np.where(((mon>=12)|(mon<=2))&(yr>=1985+st_yr))[0], 
          'MAM': np.where(((mon>=3)&(mon<=5))&(yr>=1985+st_yr))[0],
          'JJA': np.where(((mon>=6)&(mon<=8))&(yr>=1985+st_yr))[0],
          'SON': np.where(((mon>=9)&(mon<=11))&(yr>=1985+st_yr))[0]}
    
    # Each season
    vars = {}
    for sea in it.keys():           
        tmp = var[it[sea]]      # 90 months for each season (3mon*30yr)

        # Each year
        tmpy = []
        for y in np.unique(yr):     
            iy = np.where(yr[it[sea]]==y)[0]           # find each year in 90 month data
            tmpy.append(np.nanmean(tmp[iy],axis=0))    # yearly mean
        vars[sea] = np.array(tmpy)    # 30yr
    return vars


#%% TaiESM-ROMS =====================================================
# Initialize 
root = '/data3/jhluna/work/ITRI_UCH_phdw/DATA/'
fname = [root+'taiesm_roms_zlev/ocean_d01_his_mon.nc', \
         root+'taiesm_roms_zlev/ocean_d01_ssp_mon.nc']
scenm = ['HIS','SSP']
sst_was, time = {}, {}

# Read each scenario ncfile
for f, fnm in enumerate(fname):

    with nc.Dataset(fnm) as rootgrp:
        lat = np.array(rootgrp.variables['lat'][:])
        lon = np.array(rootgrp.variables['lon'][:])
        sst = np.array(rootgrp.variables['SST'][:])
        sst[np.where((sst==0)|(sst>35))[:]] = np.nan
        t = rootgrp.variables['time'][:]
        tu = rootgrp.variables['time'].units
        time = nc.num2date(t,units=tu,calendar='standard')

    # Select region
    sst_wa = select_wangan(sst,lon,lat)   # (time)

    # Seasonal mean (yearly data)
    sst_was[scenm[f]] = select_season_std(sst_wa, time)

# Difference and standard deviation
DIF, STD = [], []
for sea in sst_was['HIS'].keys():
    tmp = sst_was['SSP'][sea] - sst_was['HIS'][sea]
    DIF.append(np.nanmean(tmp))
    STD.append(np.nanstd(tmp))

del f,fname,fnm,lat,lon,rootgrp,sea,sst,sst_was,sst_wa,t,time,tmp,tu

#%% d00 climatological test =========================================
# Initialize 
fname = [root+'itri_uch_avg_d00c_his_zlev.nc', \
         root+'itri_uch_avg_d00c_ssp_zlev.nc']
sst_was, time = {}, {}

# Read each scenario ncfile
for f, fnm in enumerate(fname):

    with nc.Dataset(fnm) as rootgrp:
        sst = np.array(rootgrp.variables['SST'][:,1:-1,1:-1])
        lat = np.array(rootgrp.variables['lat'][1:-1,1])
        lon = np.array(rootgrp.variables['lon'][1,1:-1])
        sst[np.where((sst==0)|(sst>35))[:]] = np.nan
        t = rootgrp.variables['time'][:]
        tu = rootgrp.variables['time'].units
        time = nc.num2date(t,units=tu,calendar='standard')

    # Select region
    sst_wa = select_wangan(sst,lon,lat)         # (time)

    # Seasonal mean 
    sst_was[scenm[f]] = select_season(sst_wa, time)

# Difference
DIF_d00c = []
for sea in sst_was['HIS'].keys():
    tmp = sst_was['SSP'][sea] - sst_was['HIS'][sea]
    DIF_d00c.append(tmp)

del f,fname,fnm,lat,lon,rootgrp,sea,sst,sst_was,sst_wa,t,tu,time,tmp

#%% d00 30year test =================================================
# Initialize 
fname = [root+'itri_uch_avg_d00_his_zlev.nc',\
         root+'itri_uch_avg_d00_ssp_zlev.nc']
sst_was, time = {}, {}

# Read each scenario ncfile
for f, fnm in enumerate(fname):

    with nc.Dataset(fnm) as rootgrp:
        sst = np.array(rootgrp.variables['SST'][:,1:-1,1:-1])
        lat = np.array(rootgrp.variables['lat'][1:-1,1])
        lon = np.array(rootgrp.variables['lon'][1,1:-1])
        sst[np.where((sst==0)|(sst>35))[:]] = np.nan
        t = rootgrp.variables['time'][:]
        tu = rootgrp.variables['time'].units
        time = nc.num2date(t,units=tu,calendar='standard')

    # Select region
    sst_wa = select_wangan(sst,lon,lat)         # (time)

    # Seasonal mean 
    sst_was[scenm[f]] = select_season_std(sst_wa, time)

# Difference and standard deviation
DIF_d00, STD_d00 = [], []
for sea in sst_was['HIS'].keys():
    tmp = sst_was['SSP'][sea] - sst_was['HIS'][sea]
    DIF_d00.append(np.nanmean(tmp))
    STD_d00.append(np.nanstd(tmp))

del f,fname,fnm,lat,lon,rootgrp,sea,sst,sst_was,sst_wa,t,tu,time,tmp

#%% d01 test ========================================================
# Initialize 
fname = [root+'itri_uch_avg_d01_his_zlev.nc',\
         root+'itri_uch_avg_d01_ssp_zlev.nc']
sst_was, time = {}, {}

# Read each scenario ncfile
for f, fnm in enumerate(fname):

    with nc.Dataset(fnm) as rootgrp:
        sst = np.array(rootgrp.variables['SST'][:,1:-1,1:-1])
        lat = np.array(rootgrp.variables['lat'][1:-1,1])
        lon = np.array(rootgrp.variables['lon'][1,1:-1])
        sst[np.where((sst==0)|(sst>35))[:]] = np.nan
        t = rootgrp.variables['time'][:]
        tu = rootgrp.variables['time'].units
        time = nc.num2date(t,units=tu,calendar='standard')

    # Select region
    sst_wa = select_wangan(sst,lon,lat)         # (time)

    # Seasonal mean 
    sst_was[scenm[f]] = select_season(sst_wa, time)

# Difference
DIF_d01 = []
for sea in sst_was['HIS'].keys():
    tmp = sst_was['SSP'][sea] - sst_was['HIS'][sea]
    DIF_d01.append(tmp)

del f,fname,fnm,lat,lon,rootgrp,sea,sst,sst_was,sst_wa,t,tu,time,tmp

#%% Plot the bar chart ==============================================
x = ['Winter','Spring','Summer','Autumn']
y = {'TaiESM-ROMS':DIF, 'ROMS dph (clim.)':DIF_d00c, 'ROMS dph (30-yr)':DIF_d00, 'ROMS d01':DIF_d01}
err = {'TaiESM-ROMS':STD, 'ROMS dph (30-yr)':STD_d00}

df = pd.DataFrame(y,index=x)

ax = df.plot.bar(rot=0, figsize=(4,4),\
                 color={'TaiESM-ROMS':'dimgray','ROMS dph (clim.)':'tab:blue',\
                        'ROMS dph (30-yr)':'cornflowerblue','ROMS d01':'tab:orange'})
ax.set_title('Difference (\N{DEGREE SIGN}C)',fontsize=15,fontweight='bold',loc='left')
ax.grid(axis='y',linestyle='--',alpha=0.7,zorder=0)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.ylim([0,1.5])
plt.text(2.34,1.53,'Wangan area',fontsize=10)
plt.legend(loc='lower left')
plt.show()