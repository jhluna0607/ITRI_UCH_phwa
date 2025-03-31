#%% Plot SST and Current Change in Taiwan Strait from TaiESM-ROMS
import cmaps
import numpy as np
import pandas as pd
import netCDF4 as nc
import cartopy as cart
import cartopy.crs as ccrs
from scipy import integrate
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

#%% Read ncfile  ====================================================
root = '/data3/jhluna/work/ITRI_UCH_phdw/DATA/'
fnm = [root+'taiesm_roms_zlev/ocean_d01_his_mon.nc',\
       root+'taiesm_roms_zlev/ocean_d01_ssp_mon.nc']
scen = ['HIS', 'SSP']
uo, time = {}, {}
lon_scs = 120.8

for f, fname in enumerate(fnm):
    with nc.Dataset(fname) as rootgrp:
        lat = rootgrp.variables['lat'][:]
        lon = rootgrp.variables['lon'][:]
        lev = rootgrp.variables['lev'][:]
        ilon = np.argmin(np.abs(lon-lon_scs))        
        uo[scen[f]] = rootgrp.variables['uo'][:,:,:,ilon]      # (time,lev,lat)
        t = rootgrp.variables['time'][:]
        tu = rootgrp.variables['time'].units
        time[scen[f]] = nc.num2date(t,units=tu,calendar='standard')
del f,fname,t,tu,rootgrp,ilon,lon

#%% Select Seasons ==================================================
UO = {}
seas = ['DJF', 'MAM', 'JJA', 'SON']
st_yr = 0

get_month = np.vectorize(lambda date: date.month)
mon = get_month(time['HIS'].compressed())
get_year = np.vectorize(lambda date: date.year)
yr = get_year(time['HIS'].compressed())
it = {'DJF': np.where(((mon>=12)|(mon<=2))&(yr>=1985+st_yr))[0], 
      'MAM': np.where(((mon>=3)&(mon<=5))&(yr>=1985+st_yr))[0], 
      'SON': np.where(((mon>=9)&(mon<=11))&(yr>=1985+st_yr))[0], 
      'JJA': np.where(((mon>=6)&(mon<=8))&(yr>=1985+st_yr))[0]}

for se, sea in enumerate(seas):
    for sc, sce in enumerate(scen):

        # Different scenario and season, tmp (30 year*3 mon)
        tmp = uo[sce][it[sea],:,:]                  # (time,lev,lat)

        # Yearly data selected from tmp1, tmp2 (30 year)
        s = uo['HIS'][it['DJF'],:,:].shape
        tmp2 = np.zeros([int(s[0]/3),s[1],s[2]])    # (time,lev,lat)
        for v in range(s[1]):
            for j in range(s[2]):
                tmp1 = np.convolve(tmp[:,v,j], np.ones(3)/3, mode='valid')
                tmp2[:,v,j] = tmp1[0::3]

        UO[f"{sce}_{sea}"] = tmp2
    
del sc,sce,se,sea,get_month,mon,get_year,yr,it,tmp,tmp1,tmp2,v,j,s

#%% Calculate Transport per depth ===================================
SV = {}
lat2m = 111200
lat_scs = [17,23]
ilat = np.where((lat>=lat_scs[0])&(lat<=lat_scs[1]))[0]
dlat = (lat[ilat[1]]-lat[ilat[0]])*lat2m    # unit: m

for se, sea in enumerate(seas):
    for sc, sce in enumerate(scen):
        SV[f"{sce}_{sea}"] = np.nansum(UO[f"{sce}_{sea}"][:,:,ilat],axis=2)*dlat    # (time,lev)
    
#%% Calculate Volume Transport ======================================
dlev = -np.gradient(lev)
SV_upper_t, SV_upper, SV_upper_std = {}, {}, {}

# 500m (26); 700m (31)
d = 31
for se, sea in enumerate(seas):
    for sc, sce in enumerate(scen):
        SV_upper_t[f"{sce}_{sea}"] = np.nansum(SV[f"{sce}_{sea}"][:,:d]*dlev[:d],axis=1)*1e-6     # (time)
        #SV_upper_t[f"{sce}_{sea}"] = integrate.simpson(SV[f"{sce}_{sea}"][:,:d], x=-lev[:d],axis=1)*1e-6

# Standard deviation
for se, sea in enumerate(seas):
    for sc, sce in enumerate(scen):
        SV_upper_std[f"{sce}_{sea}"] = np.nanstd(SV_upper_t[f"{sce}_{sea}"])     # (time)
        SV_upper[f"{sce}_{sea}"] = np.nanmean(SV_upper_t[f"{sce}_{sea}"])


# %% Plot bar =======================================================
x = ['Winter','Spring','Summer','Autumn']
y = {'Historical': [SV_upper[f'HIS_{sea}'] for sea in seas],
     'SSP2-4.5': [SV_upper[f'SSP_{sea}'] for sea in seas]}
yerr = {'Historical': [SV_upper_std[f'HIS_{sea}'] for sea in seas],
     'SSP2-4.5': [SV_upper_std[f'SSP_{sea}'] for sea in seas]}

df = pd.DataFrame(y,index=x)
dfe = pd.DataFrame(yerr,index=x)

ax = df.plot.bar(rot=0, figsize=(5,3), yerr=dfe, error_kw=dict(lw=1, capsize=3, capthick=1), \
                 color={'Historical':'slateblue','SSP2-4.5':'gold'})
ax.set_title('Net transport in the upper layer (Sv)',fontsize=11,fontweight='semibold',loc='left')
ax.grid(axis='y',linestyle='--',alpha=0.7,zorder=0)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
#plt.ylim([0,1.5])
#plt.text(2.34,1.53,'Wangan area',fontsize=10)
plt.legend(loc='upper left',fontsize=8)
#plt.ylabel('(Sv)')

# %%
