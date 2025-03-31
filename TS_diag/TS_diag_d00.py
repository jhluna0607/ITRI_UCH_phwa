#%%
import numpy as np
import pandas as pd
import netCDF4 as nc
import matplotlib.pyplot as plt
from scipy import interpolate
import seawater as sw

def select_point_data(temp, salt, lon, lat, target_lon, target_lat, lat_range=0):
    ilon = np.nanargmin(abs(lon - target_lon))
    ilat = np.nanargmin(abs(lat - target_lat))
    temp_data = np.nanmean(temp[:, :, ilat-lat_range:ilat+lat_range+1, ilon], axis=2)
    salt_data = np.nanmean(salt[:, :, ilat-lat_range:ilat+lat_range+1, ilon], axis=2)
    return temp_data, salt_data

def select_seasonal_data(data, time, season):
    if season == 'DJF':
        it = np.where((time == 12) | (time == 1) | (time == 2))[0]
    elif season == 'JJA':
        it = np.where((time >= 6) & (time <= 8))[0]
    return np.nanmean(data[it, :], axis=0)

#%% SODA ============================================================
# Read ncfile -------------------------------------------------------
ftn = np.arange(1985,2014,1)
root = '/data/chc/DATA/SODA/v331/monthly/soda3.3.1_mn_ocean_reg_'
temp, salt = [], []

# Each year data
for f, year in enumerate(ftn):
    filepath = f"{root}{year}.nc"
    print(filepath)

    # Read ncfile with with
    with nc.Dataset(filepath) as rootgrp:

        # Domain selection
        if f == 0:
            lon = rootgrp.variables['longitude'][:]
            lat = rootgrp.variables['latitude'][:]
            lev = np.array(rootgrp.variables['depth'][:])
            ilon = np.where((lon >= 112) & (lon <= 125))[0]
            ilat = np.where((lat >= 16) & (lat <= 28))[0]
            ilev = np.where(lev <= 500)[0]
            lon, lat, lev = lon[ilon], lat[ilat], lev[ilev]
            
        # Read temperature and salinity
        to = rootgrp.variables['temp'][:,ilev,ilat,ilon]  # (mon, lev, lat, lon)
        so = rootgrp.variables['salt'][:,ilev,ilat,ilon]  # (mon, lev, lat, lon)

        # Save to big list
        temp.append(to)
        salt.append(so)

# Stick the whole list
temp = np.concatenate(temp, axis=0)  # (time, lev, lat, lon)
salt = np.concatenate(salt, axis=0)  # (time, lev, lat, lon)

# Remove strange value
temp[np.abs(temp) > 40] = np.nan
salt[np.abs(salt) > 40] = np.nan

del f,filepath,ilat,ilev,ilon,root,rootgrp,so,to,year

# Select water mass -------------------------------------------------
SODA = {}
regions = {"KW": {"lon": 123, "lat": 19, "lat_range": 0},
           "SCSW": {"lon": 116, "lat": 22.25, "lat_range": 2}}
seasons = ['DJF', 'JJA']
time = np.tile(np.arange(1, 13), len(ftn))  # (month)

for region_name, region_info in regions.items():    # (string, struct)

    # Select water mass location
    temp_data, salt_data = select_point_data(
        temp, salt, lon, lat, 
        target_lon=region_info["lon"], 
        target_lat=region_info["lat"], 
        lat_range=region_info["lat_range"])

    # Select season
    for season in seasons:
        SODA[f"temp_{region_name.lower()}_{season.lower()}"] = select_seasonal_data(temp_data, time, season)
        SODA[f"salt_{region_name.lower()}_{season.lower()}"] = select_seasonal_data(salt_data, time, season)

del lon,lat,region_info,region_name,salt,salt_data,season,temp,temp_data,time

#%% TaiESM-ROMS =====================================================
root = '/data3/jhluna/work/ITRI_UCH_phdw/DATA/taiesm_roms_zlev/'
TAIESM_ROMS = {}
temp_sea, salt_sea = [], []
fnm = [root+'ocean_d01_his_mon.nc', root+'ocean_d01_ssp_mon.nc']

# Each scenario data
for f, ftn in enumerate(fnm):
    print(ftn)

    # Read ncfile with with
    with nc.Dataset(ftn) as rootgrp:

        # Domain Selection
        lon  = rootgrp.variables['lon'][:]
        lat  = rootgrp.variables['lat'][:]
        levt = -np.array(rootgrp.variables['lev'][:])
        to = rootgrp.variables['to'][:]    # (time,lev,lat,lon)
        so = rootgrp.variables['so'][:]

        # Time
        time = rootgrp.variables['time'][:]
        time_units = rootgrp.variables['time'].units
        time = nc.num2date(time, units=time_units, calendar='standard')
        time = np.array([pd.Timestamp(t.strftime("%Y-%m-%d %H:%M:%S")) for t in time])
        mon = pd.DatetimeIndex(time).month

    # Select water mass
    regions = {"KW": {"lon": 123, "lat": 19, "lat_range": 0},
               "SCSW": {"lon": 116, "lat": 22.25, "lat_range": 10}}

    for region_name, region_info in regions.items():    # (string, struct)

        # Select water mass location
        temp_data, salt_data = select_point_data(
            to, so, lon, lat,
            target_lon=region_info["lon"], 
            target_lat=region_info["lat"], 
            lat_range=region_info["lat_range"])

        # Select season
        for season in seasons:
            temp_sea.append(select_seasonal_data(temp_data, mon, season))
            salt_sea.append(select_seasonal_data(salt_data, mon, season))

        # Calculate difference
            if f == 1: 
                dtemp = temp_sea[-1]-temp_sea[-5]
                dsalt = salt_sea[-1]-salt_sea[-5]

                # Interp z-coor from TaiESM-ROMS to SODA
                fz = interpolate.interp1d(levt, dtemp)
                TAIESM_ROMS[f"temp_{region_name.lower()}_{season.lower()}_diff"] = fz(lev)
                fz = interpolate.interp1d(levt, dsalt)
                TAIESM_ROMS[f"salt_{region_name.lower()}_{season.lower()}_diff"] = fz(lev)


del f,fnm,ftn,lat,lon,mon,region_name,region_info,rootgrp,salt_data,temp_data,to,so,season,time,time_units
del temp_sea,salt_sea,regions,dtemp,dsalt,fz,levt,lev

#%% Penghu downscaling ==============================================
root = '/data3/jhluna/work/ITRI_UCH_phdw/DATA/'
PH_ROMS = {}
fnm = [root+'itri_uch_avg_d00_his_zlev.nc',root+'itri_uch_avg_d00_ssp_zlev.nc']

# Each scenario data
for f, ftn in enumerate(fnm):
    print(ftn)

    # Read ncfile with with
    with nc.Dataset(ftn) as rootgrp:

        # Domain Selection
        lon  = rootgrp.variables['lon'][:]
        lat  = rootgrp.variables['lat'][:]
        lev  = np.array(rootgrp.variables['lev'])
        to = rootgrp.variables['to'][:]    # (time,lev,lat,lon)
        so = rootgrp.variables['so'][:]

        # Time
        time = rootgrp.variables['time'][:]
        time_units = rootgrp.variables['time'].units
        time = nc.num2date(time, units=time_units, calendar='standard')
        time = np.array([pd.Timestamp(t.strftime("%Y-%m-%d %H:%M:%S")) for t in time])
        mon = pd.DatetimeIndex(time).month

    # Select water mass location
    #ilon = np.where((lon>=119.6) & (lon<=120))[0]
    #ilat = np.where((lat>=23.2) & (lat<=23.8))[0]
    #[ilon,ilat] = np.meshgrid(ilon,ilat)
    #to = to[:,:,ilat,ilon]                  # (time,lev,lat,lon)
    #so = so[:,:,ilat,ilon]

    # Select season
    for season in seasons:
        if f == 0:
            PH_ROMS[f"temp_ph_{season.lower()}"] = []
            PH_ROMS[f"salt_ph_{season.lower()}"] = []
        PH_ROMS[f"temp_ph_{season.lower()}"].append(select_seasonal_data(to, mon, season))
        PH_ROMS[f"salt_ph_{season.lower()}"].append(select_seasonal_data(so, mon, season))
        if f == 1:
            PH_ROMS[f"temp_ph_{season.lower()}"] = np.asarray(PH_ROMS[f"temp_ph_{season.lower()}"])
            PH_ROMS[f"salt_ph_{season.lower()}"] = np.asarray(PH_ROMS[f"salt_ph_{season.lower()}"])
    del rootgrp,so,to,time,time_units

del f,fnm,ftn,lat,lon,mon

#%% Draw T-S diagram ================================================
# T-S density
x_salt = np.arange(33,35.2,.2)
y_temp = np.arange(7,32,1)
[xx,yy] = np.meshgrid(x_salt,y_temp)
den = sw.dens0(xx,yy)-1000

# Depth matrix
s = PH_ROMS['temp_ph_djf'].shape
dep = -np.tile(lev[:, np.newaxis, np.newaxis],(1,s[-2],s[-1]))
del xx,yy,s

# Plot
sc = [0,0,1,1]
fig,ax = plt.subplots(figsize=(6,5),nrows=2,ncols=2,facecolor='white')
plt.subplots_adjust(wspace=.1,hspace=.2)
ax = ax.flatten()

for i, axis in enumerate(ax):
    season = seasons[i%2].lower()
    if i <= 1: s = 0; 
    else: s = 1;
    cm = axis.contour(x_salt,y_temp,den,levels=np.arange(20,28.5,.5),colors='gray',linewidths=.5)
    cn = axis.scatter(PH_ROMS[f"salt_ph_{season}"][s],PH_ROMS[f"temp_ph_{season}"][s],s=1,c=dep)
    #ll = 45
    #cn = axis.scatter(PH_ROMS[f"salt_ph_{season}"][s,0,ll,0:25],PH_ROMS[f"temp_ph_{season}"][s,0,ll,0:25],s=1)
    axis.clabel(cm, fontsize=8)

for s, season in enumerate(seasons):
    ax[s].plot(SODA[f"salt_kw_{season.lower()}"],SODA[f"temp_kw_{season.lower()}"],'-k')
    ax[s].plot(SODA[f"salt_scsw_{season.lower()}"],SODA[f"temp_scsw_{season.lower()}"],'-k')

for s, season in enumerate(seasons):
    ax[s+2].plot(SODA[f"salt_kw_{season.lower()}"]+TAIESM_ROMS[f"salt_kw_{season.lower()}_diff"] \
                ,SODA[f"temp_kw_{season.lower()}"]+TAIESM_ROMS[f"temp_kw_{season.lower()}_diff"],'-k')
    ax[s+2].plot(SODA[f"salt_scsw_{season.lower()}"]+TAIESM_ROMS[f"salt_scsw_{season.lower()}_diff"] \
                ,SODA[f"temp_scsw_{season.lower()}"]+TAIESM_ROMS[f"temp_scsw_{season.lower()}_diff"],'-k')

ax[0].set_title('Winter',fontsize=13)
ax[1].set_title('Summer',fontsize=13)
for i in [0,2]:
    ax[i].set_ylabel('Temperature (\N{DEGREE SIGN}C)')
    ax[i].set_xlim([33.75,35.2])
    ax[i].set_ylim([15,31])
for i in [1,3]:
    ax[i].set_xlim([33,35.2])
    ax[i].set_ylim([15,31])
    ax[i].set_yticks([])
for i in [2,3]:
    ax[i].set_xlabel('Salinity (psu)')
ax[0].text(-0.3, 0.5, 'Historical', va='bottom', ha='center',rotation='vertical',\
            rotation_mode='anchor', transform=ax[0].transAxes,fontsize=13)
ax[2].text(-0.3, 0.5, 'SSP2-4.5', va='bottom', ha='center',rotation='vertical',\
            rotation_mode='anchor', transform=ax[2].transAxes,fontsize=13)
cbar_ax = fig.add_axes([.95,0.13,0.02,0.7])
cbar = fig.colorbar(cn,cax=cbar_ax,orientation='vertical',extend='both')
cbar.ax.set_title('m')
plt.show()
