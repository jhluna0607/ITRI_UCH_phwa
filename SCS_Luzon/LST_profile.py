#%% The zonal current and transport across Luzon Strait profile from TaiESM-ROMS
import cmaps
import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

#%% =================================================================
# Data Process
# ===================================================================
# Read ncfile -------------------------------------------------------
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

# Select month ------------------------------------------------------
# Initicalize
UO = {}
seas = ['DJF', 'JJA']
st_yr = 0                               # exclude spin-up year or not

# Get year and month from time
get_month = np.vectorize(lambda date: date.month)
mon = get_month(time['HIS'].compressed())
get_year = np.vectorize(lambda date: date.year)
yr = get_year(time['HIS'].compressed())

# Select season and year
it = {'DJF': np.where(((mon>=12)|(mon<=2))&(yr>=1985+st_yr))[0], 
      'JJA': np.where(((mon>=6)&(mon<=8))&(yr>=1985+st_yr))[0]}

# Select season to variable dictionary
for se, sea in enumerate(seas):
    for sc, sce in enumerate(scen):
        UO[f"{sce}_{sea}"] = uo[sce][it[sea],:,:]               # (time,lev,lat)

    # Change mean after warming
    UO[f"DIFF_{sea}"] = np.nanmean(UO[f"SSP_{sea}"]-UO[f"HIS_{sea}"],axis=0)    # (lev,lat)

    # P-value in each point
    UO[f"PV_{sea}"] = np.zeros(UO['HIS_DJF'].shape[1:])
    for v in range(UO['HIS_DJF'].shape[1]):                     # (lev,lat)
        for j in range(UO['HIS_DJF'].shape[2]):
            UO[f"PV_{sea}"][v,j] = ttest_ind(UO[f"HIS_{sea}"][:,v,j], UO[f"SSP_{sea}"][:,v,j]).pvalue

del sc,sce,se,sea,get_month,mon,get_year,yr,it,v,j

#%% =================================================================
# Calculate Transport per depth
# ===================================================================
SV, SV_std = {}, {}
lat2m = 111200
lat_scs = [17,23]
ilat = np.where((lat>=lat_scs[0])&(lat<=lat_scs[1]))[0]
dlat = (lat[ilat[1]]-lat[ilat[0]])*lat2m    # unit: m

for se, sea in enumerate(seas):
    for sc, sce in enumerate(scen):
        SV[f"{sce}_{sea}"] = np.nansum(UO[f"{sce}_{sea}"][:,:,ilat],axis=2)*dlat    # (time,lev)
        SV_std[f"{sce}_{sea}"] = np.nanstd(SV[f"{sce}_{sea}"][:,:],axis=0)          # (lev)
        SV[f"{sce}_{sea}"] = np.nanmean(SV[f"{sce}_{sea}"],axis=0)                  # (lev)
        UO[f"{sce}_{sea}"] = np.nanmean(UO[f"{sce}_{sea}"],axis=0)
    
    SV[f"DIFF_{sea}"] = SV[f"SSP_{sea}"]-SV[f"HIS_{sea}"]

#%% =================================================================
# Plot the Transport Profile
# ===================================================================
fig,ax = plt.subplots(figsize=(2,6), nrows=2, ncols=1)
plt.subplots_adjust(hspace=.15)
ax = ax.flatten()
ax[0].plot(SV['HIS_DJF']*1e-6,lev,color='midnightblue')
ax[0].plot(SV['SSP_DJF']*1e-6,lev,color='firebrick')
#ax[0].fill_betweenx(lev,(SV['HIS_DJF']-SV_std['HIS_DJF'])*1e-6,\
#                    (SV['HIS_DJF']+SV_std['HIS_DJF'])*1e-6,alpha=0.5,facecolor='midnightblue')
#ax[0].fill_betweenx(lev,(SV['SSP_DJF']-SV_std['SSP_DJF'])*1e-6,\
#                    (SV['SSP_DJF']+SV_std['SSP_DJF'])*1e-6,alpha=0.5,facecolor='firebrick')
ax[0].set_ylim([-3000,0])
#ax[0].set_ylim([-500,0])
ax[0].set_yticklabels([str(-d) for d in range(-3000,1,500)])
ax[0].grid()
ax[0].legend(['Historical','SSP2-4.5'], loc='lower left', fontsize=8)
ax[0].set_title('Transport')
ax[0].set_ylabel('Winter')

ax[1].plot(SV['HIS_JJA']*1e-6,lev,color='midnightblue')
ax[1].plot(SV['SSP_JJA']*1e-6,lev,color='firebrick')
ax[1].set_ylim([-3000,0])
#ax[1].set_ylim([-500,0])
ax[1].set_yticklabels([str(-d) for d in range(-3000,1,500)])
ax[1].grid()
ax[1].set_xlabel('Sv/m')
ax[1].set_ylabel('Summer')

#%% =================================================================
# Plot the Zonal Current Profile (2d: lat-lev)
# ===================================================================
fig,ax = plt.subplots(figsize=(10,6), nrows=2, ncols=3)
plt.subplots_adjust(wspace=.05, hspace=.15)
ax = ax.flatten()

selected_keys = [list(UO.keys())[i] for i in [0,1,2,4,5,6]]
for i, sce in enumerate(selected_keys):

    # Subplots' settings
    if i == 2 or i == 5:
        cm = cmaps.cmp_flux.to_seg(N=18)
        vmax = .3
        clev = np.arange(-vmax,vmax+1e-3,vmax/9)
    else:
        cm = 'Spectral_r'
        vmax = .8
        clev = np.arange(-vmax,vmax+.1,.1)

    # Plotting
    ax[i].set_facecolor('darkgray')
    cn = ax[i].contourf(lat, lev, UO[sce], cmap=cm, vmin=-vmax, vmax=vmax,\
                        levels=clev, extend='both')
    
    n = 3
    if i == 2:
        for v in range(UO['PV_DJF'].shape[0]//n):
            for j in range(UO['PV_DJF'].shape[1]//n):
                if UO['PV_DJF'][v*n,j*n] <= 0.05:
                    ax[i].scatter(lat[j*n],lev[v*n],s=1,marker='o',c='gray')

    if i == 5:
        for v in range(UO['PV_JJA'].shape[0]//n):
            for j in range(UO['PV_JJA'].shape[1]//n):
                if UO['PV_JJA'][v*n,j*n] <= 0.05:
                    ax[i].scatter(lat[j*n],lev[v*n],s=1,marker='o',c='gray')

    ax[i].set_xlim([18.3,22.3])
    ax[i].set_ylim([-3000,0])
    #ax[i].set_ylim([-500,0])

    # Ticks
    ax[i].grid()
    if i != 0 or i != 3:
        ax[i].set_yticklabels([])
    if i == 0 or i == 3:
        ax[i].set_yticks(np.arange(-3000,1,500))
        ax[i].set_yticklabels([str(-d) for d in range(-3000,1,500)])
        #ax[i].set_yticks(np.arange(-500,1,100))
        #ax[i].set_yticklabels([str(-d) for d in range(-500,1,100)])
    if i < 3:
        ax[i].set_xticklabels([])

    # Colobars
    if i == 4:
        cbar_ax = fig.add_axes([0.15,0.02,0.46,0.02])
        cbar = fig.colorbar(cn, ticks=np.arange(-vmax,vmax+.1,vmax/4), \
                            cax=cbar_ax, orientation='horizontal',extend='both')
        #cbar_ax = fig.add_axes([0.15,0.02,0.2,0.02])
        #cbar = fig.colorbar(cn, ticks=np.arange(-vmax,vmax+.1,vmax/2), \
        #                    cax=cbar_ax, orientation='horizontal',extend='both')
        cbar.set_label('m/s')
    elif i == 5:
        cbar_ax = fig.add_axes([0.67,0.02,0.2,0.02])
        cbar = fig.colorbar(cn, ticks=np.arange(-vmax,vmax+1e-3,.2), \
                            cax=cbar_ax, orientation='horizontal', extend='both')
        cbar.set_label('m/s')

# Titles
ax[0].set_title(str(1985+st_yr)+'-2014',fontsize=15)
ax[1].set_title(str(2040+st_yr)+'-2069',fontsize=15)
ax[2].set_title('Difference',fontsize=15)
ax[0].text(-0.25, 0.5, 'Winter', va='bottom', ha='center',rotation='vertical', \
           rotation_mode='anchor', transform=ax[0].transAxes, fontsize=15)
ax[3].text(-0.25, 0.5, 'Summer', va='bottom', ha='center',rotation='vertical', \
           rotation_mode='anchor', transform=ax[3].transAxes, fontsize=15)
plt.show()


# %%
