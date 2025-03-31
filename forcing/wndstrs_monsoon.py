#%%
import numpy as np
import cmaps
import netCDF4 as nc
import matplotlib.pyplot as plt
import cartopy as cart
import cartopy.crs as ccrs
from scipy.stats import ttest_ind
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

def curl(lat,lon,u,v):
    s = u.shape
    deg2m = 111200
    latm = (lat[2,1]-lat[1,1])*deg2m
    lonm = (lon[1,2]-lon[1,1])*np.cos(np.deg2rad(lat))*deg2m
    dudy = np.gradient (u,latm,axis=0)
    dvdx = np.gradient (v,axis=1)/lonm
    curlZ = dvdx-dudy
    return curlZ

#%% =================================================================
# Data Process
# ===================================================================
# Initialize --------------------------------------------------------
root = '/data3/jhluna/work/ITRI_UCH_phdw/DATA/'
fnm  = [root+'taiesm_roms_zlev/ocean_d01_his_mon.nc',\
        root+'taiesm_roms_zlev/ocean_d01_ssp_mon.nc']
his, ssp = {}, {}

# Read ncfile
for f, data_dict in enumerate([his, ssp]):
    sustrs, svstrs, curlzs = [], [], []

    with nc.Dataset(fnm[f]) as rootgrp:
        lon  = rootgrp.variables['lon'][:]
        lat  = rootgrp.variables['lat'][:]
        sustr = rootgrp.variables['sustr'][:]      # (time,lat,lon)
        svstr = rootgrp.variables['svstr'][:]
        t = rootgrp.variables['time'][:]
        tu = rootgrp.variables['time'].units
        time = nc.num2date(t,units=tu,calendar='standard')
    del t,tu,rootgrp

# Calculate wind curl -----------------------------------------------
    [lon2,lat2] = np.meshgrid(lon,lat)
    curlz = np.array([curl(lat2,lon2,sustr[t,:,:],svstr[t,:,:]) for t in range(sustr.shape[0])])

# Seasonal mean -----------------------------------------------------   
    # Define month
    get_month = np.vectorize(lambda date: date.month)
    mon = get_month(time.compressed())
    del get_month

    # Seasonal mean
    seasons = {'DJF': [12,1,2], 'MAM': [3,4,5], 'JJA': [6,7,8], 'SON': [9,10,11]}
    for idx, (season_name, months) in enumerate(seasons.items()):
        if idx != 0:
            loc = np.where((mon>=months[0]) & (mon<=months[2]))[0]
        else:
            loc = np.where((mon>=months[0]) | (mon<=months[2]))[0]
        
        # Different scenario and season (30 year*3 mon)
        u, v, c = sustr[loc,:,:], svstr[loc,:,:], curlz[loc,:,:]            # (time,lat,lon)
        s = sustr[loc,:,:].shape

        # 3-month running mean (30 year)
        sustrm = np.zeros([int(s[0]/3),s[1],s[2]])                          # (year,lat,lon)
        svstrm = np.zeros([int(s[0]/3),s[1],s[2]])
        curlzm = np.zeros([int(s[0]/3),s[1],s[2]])
        for i in range(s[1]):
            for j in range(s[2]):
                tmp = np.convolve(u[:,i,j], np.ones(3)/3, mode='valid')
                sustrm[:,i,j] = tmp[0::3]
                tmp = np.convolve(v[:,i,j], np.ones(3)/3, mode='valid')
                svstrm[:,i,j] = tmp[0::3]
                tmp = np.convolve(c[:,i,j], np.ones(3)/3, mode='valid')
                curlzm[:,i,j] = tmp[0::3]

        sustrs.append(sustrm)                       # (season,year,lat,lon)
        svstrs.append(svstrm)
        curlzs.append(curlzm)
    del idx,season_name,months,loc,mon

    # Save into dictionary
    data_dict["sustr"] = np.asarray(sustrs)         # (season,year,lat,lon)
    data_dict["svstr"] = np.asarray(svstrs)
    data_dict["stress"] = np.sqrt(data_dict["sustr"]**2+data_dict["svstr"]**2)
    data_dict["curlz"] = np.asarray(curlzs)

# Student-t test ----------------------------------------------------
    if f == 1:
        s = ssp['sustr'].shape
        pvalue_stress = np.zeros([s[0],s[2],s[3]])
        pvalue_curlz = np.zeros([s[0],s[2],s[3]])

        for idx, (season_name, months) in enumerate(seasons.items()):
            for i in range(s[2]):                   # (lat,lon)
                for j in range(s[3]):
                   
                   # stress
                   tmp1 = his["stress"][idx,:,i,j]
                   tmp2 = ssp["stress"][idx,:,i,j]
                   pvalue_stress[idx,i,j] = ttest_ind(tmp1,tmp2).pvalue

                   # curl
                   tmp1 = his["curlz"][idx,:,i,j]
                   tmp2 = ssp["curlz"][idx,:,i,j]
                   pvalue_curlz[idx,i,j] = ttest_ind(tmp1,tmp2).pvalue
        del s,idx,season_name,months,i,j

# Mean --------------------------------------------------------------
for f, data_dict in enumerate([his, ssp]):
    data_dict["sustr"] = np.nanmean(data_dict["sustr"],axis=1)         # (season,lat,lon)
    data_dict["svstr"] = np.nanmean(data_dict["svstr"],axis=1) 
    data_dict["stress"] = np.nanmean(data_dict["stress"],axis=1) 
    data_dict["curlz"] =np.nanmean(data_dict["curlz"],axis=1) 

del f,sustr,svstr,sustrs,svstrs,seasons

#%% =================================================================
# Plot wind stress around Taiwan Strait
# ===================================================================
season_name = ['Winter','Summer']
scenario_name = ['1985-2014','2040-2069']
sea_num = [0,2]
d, m, md = 10, 0.3, 0.03
cmap = cmaps.WhiteBlueGreenYellowRed
lon_ticks = np.arange(-66,-56,3)
lat_ticks = np.arange(17,27,3)

proj = ccrs.PlateCarree(central_longitude=180)
fig,ax= plt.subplots(figsize=(10,6),nrows=2,ncols=3,facecolor='white',\
                     subplot_kw={'projection':proj})
plt.subplots_adjust(wspace=.02,hspace=.15)
ax = ax.flatten()

for s, seast in enumerate(season_name):
    for sc, data_dict in enumerate([his, ssp]):
        f = s*3+sc
        ax[f].set_facecolor('darkgray')
        pc1 = ax[f].pcolormesh(lon2,lat2,data_dict['stress'][sea_num[s],:,:],vmin=0,vmax=m,cmap=cmap,transform=ccrs.PlateCarree())
        Q = ax[f].quiver(lon2[::d,::d],lat2[::d,::d],data_dict['sustr'][sea_num[s],::d,::d],data_dict['svstr'][sea_num[s],::d,::d],\
                         width=0.005,scale=2,transform=ccrs.PlateCarree())
        ax[f].set_extent([-67,-55,16,27],crs=proj)
        ax[f].coastlines(resolution='50m')
        ax[f].quiverkey(Q,.2,.92,0.2,'0.2 N / m$^{2}$',labelpos='S',zorder=3)
        ax[f].gridlines(crs=ccrs.PlateCarree(),draw_labels=False,color='gray',alpha=0.2)

        if s == 1:
            ax[f].set_xticks(lon_ticks,crs=proj)
            ax[f].set_xticklabels([f'{t+180}\N{DEGREE SIGN}E' for t in lon_ticks])
        else:
            ax[f].set_title(scenario_name[sc],fontsize=15)
        if sc == 0:
            ax[f].set_yticks(lat_ticks,crs=proj)
            ax[f].set_yticklabels([f'{t}\N{DEGREE SIGN}N' for t in lat_ticks])
            ax[f].text(-0.25, 0.5, seast, va='bottom', ha='center',rotation='vertical', rotation_mode='anchor',transform=ax[f].transAxes,fontsize=15)

    f = 2+s*3
    tmp = ssp['stress'][sea_num[s],:,:]-his['stress'][sea_num[s],:,:]
    tmpu = ssp['sustr'][sea_num[s],:,:]-his['sustr'][sea_num[s],:,:]
    tmpv = ssp['svstr'][sea_num[s],:,:]-his['svstr'][sea_num[s],:,:]
    ax[f].set_facecolor('darkgray')
    pc2 = ax[f].pcolormesh(lon2,lat2,tmp,vmin=-md,vmax=md,cmap='RdBu_r',transform=ccrs.PlateCarree())
    Q = ax[f].quiver(lon2[::d,::d],lat2[::d,::d],tmpu[::d,::d],tmpv[::d,::d],width=0.005,scale=.2,transform=ccrs.PlateCarree())
    ax[f].set_extent([-67,-55,16,27],crs=proj)
    ax[f].coastlines(resolution='50m')
    ax[f].quiverkey(Q,.2,.92,0.02,'0.02 N / m$^{2}$',labelpos='S',zorder=3)
    ax[f].gridlines(crs=ccrs.PlateCarree(),draw_labels=False,color='gray',alpha=0.2)

    if (f == 2) | (f==5):
        for i in range(pvalue_stress.shape[1]//d):
            for j in range(pvalue_stress.shape[2]//d):
                if pvalue_stress[sea_num[s],i*d,j*d] <= 0.05:
                    ax[f].scatter(lon[j*d],lat[i*d],s=10,marker='o',c='darkslategrey',transform=ccrs.PlateCarree())

    if s == 1:
        ax[f].set_xticks(lon_ticks,crs=proj)
        ax[f].set_xticklabels([f'{t+180}\N{DEGREE SIGN}E' for t in lon_ticks])
    else:
        ax[f].set_title('Difference',fontsize=15)

cbar_ax = fig.add_axes([0.16,0.02,0.46,0.02])
cbar = fig.colorbar(pc1,cax=cbar_ax,ticks=np.arange(0,m+1e-5,.05),orientation='horizontal',extend='max')
cbar.set_label('N / m$^{2}$')

cbar_ax = fig.add_axes([0.67,0.02,0.2,0.02])
cbar = fig.colorbar(pc2,cax=cbar_ax,ticks=np.arange(-md,md+1e-6,md),orientation='horizontal',extend='both')
cbar.set_label('N / m$^{2}$')
plt.show()

#%% =================================================================
# Plot wind stress around Taiwan Strait (Only difference)
# ===================================================================
season_name = ['Winter','Summer']
scenario_name = ['1985-2014','2040-2069']
sea_num = [0,2]
d, m, md = 10, 0.3, 0.03
cmap = cmaps.WhiteBlueGreenYellowRed
lon_ticks = np.arange(-66,-56,3)
lat_ticks = np.arange(17,27,3)

proj = ccrs.PlateCarree(central_longitude=180)
fig,ax= plt.subplots(figsize=(8,5),nrows=1,ncols=2,facecolor='white',\
                     subplot_kw={'projection':proj})
plt.subplots_adjust(wspace=.1)
ax = ax.flatten()

for s, seast in enumerate(season_name):

    tmp = ssp['stress'][sea_num[s],:,:]-his['stress'][sea_num[s],:,:]
    tmpu = ssp['sustr'][sea_num[s],:,:]-his['sustr'][sea_num[s],:,:]
    tmpv = ssp['svstr'][sea_num[s],:,:]-his['svstr'][sea_num[s],:,:]
    ax[s].set_facecolor('darkgray')
    pc2 = ax[s].pcolormesh(lon2,lat2,tmp,vmin=-md,vmax=md,cmap='RdBu_r',transform=ccrs.PlateCarree())
    Q = ax[s].quiver(lon2[::d,::d],lat2[::d,::d],tmpu[::d,::d],tmpv[::d,::d],width=0.005,scale=.2,transform=ccrs.PlateCarree())
    ax[s].set_extent([-67,-55,16,27],crs=proj)
    ax[s].coastlines(resolution='50m')
    ax[s].quiverkey(Q,.2,.92,0.02,'0.02 N / m$^{2}$',labelpos='S',zorder=3)
    ax[s].gridlines(crs=ccrs.PlateCarree(),draw_labels=False,color='gray',alpha=0.2)

    for i in range(pvalue_stress.shape[1]//d):
        for j in range(pvalue_stress.shape[2]//d):
            if pvalue_stress[sea_num[s],i*d,j*d] <= 0.05:
                ax[s].scatter(lon[j*d],lat[i*d],s=10,marker='o',c='darkslategrey',transform=ccrs.PlateCarree())

    ax[s].set_xticks(lon_ticks,crs=proj)
    ax[s].set_xticklabels([f'{t+180}\N{DEGREE SIGN}E' for t in lon_ticks])
    
    if s == 0:
        ax[s].set_yticks(lat_ticks,crs=proj)
        ax[s].set_yticklabels([f'{t}\N{DEGREE SIGN}N' for t in lat_ticks])

    ax[s].set_title(f'{seast}',fontsize=15)

cbar_ax = fig.add_axes([0.2,0.13,0.6,0.025])
cbar = fig.colorbar(pc2,cax=cbar_ax,ticks=np.arange(-md,md+1e-6,md/3),orientation='horizontal',extend='both')
cbar.set_label('N / m$^{2}$')
plt.show()
