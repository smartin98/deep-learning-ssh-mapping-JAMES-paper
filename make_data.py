# 2022-05-12 Scott Martin
# Code to generate training examples containing (1D L3 SLA input tracks, 1D L3 SLA output tracks, 2D DUACS MDT, 2D longitude, 2D latitude, 2D optimal interpolated SST) for 
# square grid region with data orthographically projected onto local ENU tangent plane

############ Final files saved: ###############
# input_tracks.npy : [n_t, N (varies per sample), 3 channels: (x, y, SLA)]
# output_tracks.npy : [n_t, N (varies per sample), 3 channels: (x, y, SLA)]
# duacs_data.npy : [n/4,n/4,3 channels: (MDT, lon, lat)], DUACS binned to lower-res grid to avoid gaps, can be linearly interpolated later
# sample_dates.npy : mid-point date of sample
# sst.npy : [n_t, n, n, channels = (SST)]
# input_ssh_grid.npy : [n_t, n, n, channels = (SSH along-track bin-averaged)]
# output_ssh_grid.npy : [n_t, n, n, channels = (SSH along-track bin-averaged)]
# coordinate_grid.npy : [n, n, channels = (lon, lat)]
###############################################

import numpy as np
from numpy.random import randint
import pyproj
import scipy.spatial.transform 
import scipy.stats as stats
from scipy import interpolate
import matplotlib.path as mpltPath
import xarray as xr 
import time
from datetime import date, timedelta
import os
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from random import shuffle
import copy

############ Definitions ######################
# function to list all files within a directory including within any subdirectories
def GetListOfFiles(dirName, ext = '.nc'):
    # create a list of file and sub directories 
    # names in the given directory 
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory 
        if os.path.isdir(fullPath):
            allFiles = allFiles + GetListOfFiles(fullPath)
        else:
            if fullPath.endswith(ext):
                allFiles.append(fullPath)               
    return allFiles

# Transform geodetic coordinates to xyz (ENU) coordinates with origin at (lat,lon,alt)_org
def ll2xyz(lat, lon, alt, lat_org, lon_org, alt_org, transformer):
    X = np.zeros([len(lat)])
    Y = np.zeros([len(lon)])
    Z = np.zeros([len(lon)])
    for i in range(len(X)):
        # transform geodetic coords to ECEF (https://en.wikipedia.org/wiki/Earth-centered,_Earth-fixed_coordinate_system)
        x, y, z = transformer.transform( lon[i],lat[i],  alt,radians=False)
        x_org, y_org, z_org = transformer.transform( lon_org,lat_org,  alt_org,radians=False)
        # define position of all points relative to origin of local tangent plane
        vec=np.array([[ x-x_org, y-y_org, z-z_org]]).T

        # define 3D rotation required to transform between ECEF and ENU coordinates (https://gssc.esa.int/navipedia/index.php/Transformations_between_ECEF_and_ENU_coordinates)
        rot1 =  scipy.spatial.transform.Rotation.from_euler('x', -(90-lat_org), degrees=True).as_matrix()
        rot3 =  scipy.spatial.transform.Rotation.from_euler('z', -(90+lon_org), degrees=True).as_matrix()
        rotMatrix = rot1.dot(rot3)    
       
        # rotate ECEF coordinates to ENU
        enu = rotMatrix.dot(vec).T.ravel()
        X[i] = enu.T[0]
        Y[i] = enu.T[1]
        Z[i] = enu.T[2]
    return X, Y, Z

# Transform xyz (ENU) coordinates with origin at (lat,lon,alt)_org to geodetic coordinates

def xyz2ll(x,y,z, lat_org, lon_org, alt_org, transformer1, transformer2):
    lat = np.zeros([len(x)])
    lon = np.zeros([len(x)])
    for i in range(len(x)):
        # transform origin of local tangent plane to ECEF coordinates (https://en.wikipedia.org/wiki/Earth-centered,_Earth-fixed_coordinate_system)
        x_org, y_org, z_org = transformer1.transform( lon_org,lat_org,  alt_org,radians=False)
        ecef_org=np.array([[x_org,y_org,z_org]]).T

        # define 3D rotation required to transform between ECEF and ENU coordinates (https://gssc.esa.int/navipedia/index.php/Transformations_between_ECEF_and_ENU_coordinates)
        rot1 =  scipy.spatial.transform.Rotation.from_euler('x', -(90-lat_org), degrees=True).as_matrix()
        rot3 =  scipy.spatial.transform.Rotation.from_euler('z', -(90+lon_org), degrees=True).as_matrix()
        rotMatrix = rot1.dot(rot3)

        # transform ENU coords to ECEF by rotating
        ecefDelta = rotMatrix.T.dot(np.array([[x[i],y[i],z]]).T)
        # add offset of all corrds on tangent plane to get all points in ECEF
        ecef = ecefDelta+ecef_org
        # transform to geodetic coordinates
        lon[i], lat[i], alt = transformer2.transform( ecef[0,0],ecef[1,0],ecef[2,0],radians=False)
    # only return lat, lon since we're interested in points on Earth. 
    # N.B. this amounts to doing an inverse stereographic projection from ENU to lat, lon so shouldn't be used to directly back calculate lat, lon from tangent plane coords
    # this is instead achieved by binning the data's lat/long variables onto the grid in the same way as is done for the variable of interest
    return lat, lon

# Generate rectangular box with number of points per side defined by refinement
def box(x_bounds, y_bounds, refinement=100):
    xs = []
    ys = []
    
    xs.append(np.linspace(x_bounds[0], x_bounds[-1], num=refinement))
    ys.append(np.linspace(y_bounds[0], y_bounds[0], num=refinement))
                
    xs.append(np.linspace(x_bounds[-1], x_bounds[-1], num=refinement))
    ys.append(np.linspace(y_bounds[0], y_bounds[-1], num=refinement))
                
    xs.append(np.linspace(x_bounds[-1], x_bounds[0], num=refinement))
    ys.append(np.linspace(y_bounds[-1], y_bounds[-1], num=refinement))
    
    xs.append(np.linspace(x_bounds[0], x_bounds[0], num=refinement))
    ys.append(np.linspace(y_bounds[-1], y_bounds[0], num=refinement))
    
    return np.concatenate(xs), np.concatenate(ys)

# Grid DUACS data onto local tangent plane centered at (lon0, lat0)
def grid_duacs(data_duacs, n, L_x, L_y, lon0, lat0, flag):
    longitude = data_duacs['longitude']
    latitude = data_duacs['latitude']
    longitude, latitude = np.meshgrid(longitude, latitude)
    longitude = longitude.flatten()
    latitude = latitude.flatten()
    if flag == 0:
        mdt = np.array(data_duacs['adt']-data_duacs['sla']).flatten()
        
    points = np.array([longitude, latitude])
    points = np.swapaxes(points,0,1)

    # inverse stereographic projection to create (lat,lon) polygon to select data to grid 
    buffer = 0.05e6 # buffer to avoid exluding data due to inverse stereographic (rather than orthographic) projection
    boxx, boxy = box([-L_x/2-buffer,L_x/2+buffer],[-L_y/2-buffer,L_y/2+buffer])

    alt0 = 0

    # rotate all data by -lon0 in longitude direction for gridding to avoid problems crossing +/- 180 degrees
    points_temp = points
    points_temp[:,0] = points_temp[:,0] - lon0
    lon_temp = points_temp[:,0]
    lon_temp[lon_temp>180] = lon_temp[lon_temp>180] - 360
    lon_temp[lon_temp<-180] = lon_temp[lon_temp<-180] + 360
    points_temp[:,0] = lon_temp

    # calculate shape of lat, lon region to grid data from
    boxlat, boxlong = xyz2ll(boxx, boxy, 0, lat0, 0, 0, transformer_ll2xyz, transformer_xyz2ll)

    # define polygon bounding region to extract from data
    area = np.array([boxlong, boxlat])
    area = np.reshape(area, (2, len(boxlat)))
    area = np.swapaxes(area, 0,1)
    path = mpltPath.Path(area)

    # check both datasets for which points lie inside this polygon and extract the relevant data into numpy arrays
    inside = path.contains_points(points_temp)
    if flag == 0:  
        mdt = mdt[inside]
    lon = points_temp[:,0]
    lat = points_temp[:,1]
    lon = lon[inside]
    lat = lat[inside]

    # N.B. when changing n you must check to make sure the grid size of AVISO doen't get too large that you need to interpolate the gridded data (see commented out code):
    
    # calculate ENU coords of duacs data on tangent plane
    x,y,_ = ll2xyz(lat, lon, 0, lat0, 0, 0, transformer_ll2xyz)
    
    if flag == 0:   
        # grid onto square grid with smaller extent than that defined earlier
        mdt_grid, _,_,_ = stats.binned_statistic_2d(x, y, mdt, statistic = 'mean', bins=n/4, range = [[-L_x/2, L_x/2],[-L_y/2, L_y/2]])
        mdt_grid = np.rot90(mdt_grid)
    else:
        mdt_grid = []
    return mdt_grid

# Grid along-track data on local tangent plane centered at (lon0, lat0)
def extract_tracked(data_tracked, L_x, L_y, lon0, lat0, path, transformer_ll2xyz):
    longitude = data_tracked['longitude']
    longitude[longitude>180] = longitude[longitude>180]-360
    latitude = data_tracked['latitude']
    sla_f = np.array(data_tracked['sla_filtered'])
    sla_f = sla_f.flatten()

    sla_uf = np.array(data_tracked['sla_unfiltered'])
    sla_uf = sla_uf.flatten()

    points = np.array([longitude, latitude])
    points= np.swapaxes(points,0,1)

    points_temp = points
    points_temp[:,0] = points_temp[:,0] - lon0
    lon_temp = points_temp[:,0]
    lon_temp[lon_temp>180] = lon_temp[lon_temp>180] - 360
    lon_temp[lon_temp<-180] = lon_temp[lon_temp<-180] + 360
    points_temp[:,0] = lon_temp

    inside = path.contains_points(points_temp)

    lon = points_temp[:,0]
    lat = points_temp[:,1]

    lon = lon[inside]
    lat = lat[inside]
    sla_f = sla_f[inside]
    sla_uf = sla_uf[inside]

    # calculate ENU coords of along-track obs
    x,y,_ = ll2xyz(lat, lon, 0, lat0, 0, 0, transformer_ll2xyz)

    points_temp = np.array([x, y])
    points_temp= np.swapaxes(points_temp,0,1)

    boxx, boxy = box([-L_x/2,L_x/2],[-L_y/2,L_y/2])

    area = np.array([boxx, boxy])
    area = np.reshape(area, (2, len(boxx)))
    area = np.swapaxes(area, 0,1)
    path = mpltPath.Path(area)

    inside = path.contains_points(points_temp)

    x = x[inside]
    y = y[inside]
    sla_f = sla_f[inside]
    sla_uf = sla_uf[inside]
    tracks = np.stack([x, y, sla_f, sla_uf], axis = -1)
    return tracks

def sst_loop(i, n, files_sst, lat0, lon0, lat_min, lat_max, long_min_unshifted, long_max_unshifted, path,x,y):
    date_loop = i
    date_string = f'{date_loop}'
    date_string = date_string.replace('-','')
    file = [f for f in files_sst if f'{date_string}' in f]

    if len(file)>0:
        data_sst = xr.open_dataset(file[0])

        if long_max_unshifted>long_min_unshifted:
            data_sst = data_sst.isel(lon = (data_sst.lon < long_max_unshifted) & (data_sst.lon > long_min_unshifted),drop = True)
        else:
            data_sst = data_sst.isel(lon = (data_sst.lon < long_max_unshifted) | (data_sst.lon > long_min_unshifted),drop = True)
        data_sst = data_sst.sel(lat=slice(lat_min,lat_max), drop = True)
        data_sst['lon'] = (data_sst['lon']-lon0+180)%360-180
        
        data_sst = data_sst.coarsen(lon=4, boundary = 'trim').mean().coarsen(lat=4, boundary = 'trim').mean()
        longitude = data_sst['lon']
        latitude = data_sst['lat']
        longitude, latitude = np.meshgrid(longitude, latitude)

        longitude = longitude.flatten()
        latitude = latitude.flatten()
        sst = np.array(data_sst['analysed_sst']).flatten()
        # sst_a = np.array(data_sst['sst_anomaly']).flatten()
        points = np.array([longitude, latitude])
        points = np.swapaxes(points,0,1)

        # check both datasets for which points lie inside this polygon and extract the relevant data into numpy arrays
        inside = path.contains_points(points)
        sst = sst[inside]

        # grid onto square grid with smaller extent than that defined earlier
        sst[np.isnan(sst)] = 0
        sst_grid, _,_,_ = stats.binned_statistic_2d(x, y, sst, statistic = 'mean', bins=n, range = [[-L_x/2, L_x/2],[-L_y/2, L_y/2]])
        sst_grid = np.rot90(sst_grid)

    else:
        sst_grid = np.zeros([n,n])

    return sst_grid

# @jit(nopython=True)
def ssh_loop(i, n, track_dirs, L_x, L_y, lat0, lon0, path):
    non_empty_sats = 0
    start_idx = 0
    track_data_in = np.zeros([int(1e6), 4])
    transformer_ll2xyz = pyproj.Transformer.from_crs(
    {"proj":'latlong', "ellps":'WGS84', "datum":'WGS84'},
    {"proj":'geocent', "ellps":'WGS84', "datum":'WGS84'},)
    transformer_xyz2ll = pyproj.Transformer.from_crs(
        {"proj":'geocent', "ellps":'WGS84', "datum":'WGS84'},
        {"proj":'latlong', "ellps":'WGS84', "datum":'WGS84'},)
    for s in range(len(track_dirs)):
        files_tracked = GetListOfFiles(track_dirs[s])
        date_loop = i

        file = [f for f in files_tracked if f'{date_loop}'.replace('-','') in f]
        if len(file)>0:
            non_empty_sats+=1
            data_tracked = xr.open_dataset(file[0])
            tracks = extract_tracked(data_tracked, L_x, L_y, lon0, lat0, path, transformer_ll2xyz)

            track_data_in[start_idx:(start_idx+tracks.shape[0]),:] = tracks
            start_idx_old = copy.deepcopy(start_idx)
            start_idx = start_idx+tracks.shape[0]
            if non_empty_sats == N_sats:
                last_sat_idx = copy.deepcopy(start_idx)
    end_idx = start_idx
    if non_empty_sats<(N_sats+1):
        last_sat_idx = start_idx_old
    
    output = [track_data_in, end_idx, last_sat_idx]

    return output

# Define the pyproj transformer objects used to transform coordinates between (lat,long,alt) and ECEF in both directions
transformer_ll2xyz = pyproj.Transformer.from_crs(
        {"proj":'latlong', "ellps":'WGS84', "datum":'WGS84'},
        {"proj":'geocent', "ellps":'WGS84', "datum":'WGS84'},
        )
transformer_xyz2ll = pyproj.Transformer.from_crs(
        {"proj":'geocent', "ellps":'WGS84', "datum":'WGS84'},
        {"proj":'latlong', "ellps":'WGS84', "datum":'WGS84'},
        )
########################################

########## SAMPLE PROPERTIES ###########
n_t = 60 # length of input time-series in days
n = 128 # number of grid points in each direction per sample
L_x, L_y = 960*1e3, 960*1e3 # size of input domain in km
N_samples = 10000 # no samples
save_dir = '960km 128x128 gulf stream'
n_cores = 1 # no CPUs when parallelising
N_sats = 5 # currently set to be number of input satellites available in the data challenge
start_date = date(2010,1,1) + timedelta(days = int(n_t/2)+1)
end_date = date(2020,12,31) - timedelta(days = int(n_t/2)+1)

########################################

# define list of available satellites in each period of satellite record:
satellites = ['alg','tpn','tp','s3b','s3a','j3','j2n','j2g','j2','j1n','j1g','j1','h2b','h2ag','h2a','g2','enn','en','e2','e1g','al','c2','c2n'] # CryoSat-2 excluded in 2017 for test purposes 

# load (long, lat) coords of grid points that lie within ocean
ocean_coords_filename = 'src/gs_ocean_coords.npy'
# ocean_coords_filename = 'src/global_ocean_coords.npy'

ocean_grid = np.load(ocean_coords_filename)
longitude_ocean = ocean_grid[:,0]
latitude_ocean = ocean_grid[:,1]

sat_dirs = []
for sat in satellites:
    sat_dirs.append('l3 sla data/'+sat)

grid_dir = 'duacs'
files_grid = GetListOfFiles(grid_dir)

total_days = (end_date-start_date).days

for sample_no in range(N_samples):
    start_time = time.time()

    dates_final = []
    duacs_data = np.zeros([32, 32, 1])

    sst_data_final = np.zeros([n_t, 128, 128])

    shift = randint(1, total_days-1)
########################
    # partitions dates into training and validation windows and ensures no overlap between the two
    check_date = start_date + timedelta(days=shift)
    check_val = np.floor(shift/n_t)%10

    if (check_val==4) or (check_val==5):
        check_date = start_date + timedelta(days=shift)
        val = 'val'
        t_0 = np.floor(shift/n_t)
        val_shift = shift - n_t*t_0
        if val_shift<n_t/2:
            if check_val==4:
                shift+=int(n_t/2)
        elif val_shift>n_t/2:
            if check_val==5:
                shift-=int(n_t/2)
    else:
        val = 'train'
    check_date = start_date + timedelta(days=shift)

    # ensure 2017 not in either training or validation sets
    while (check_date.year == 2017):
        shift = randint(1, total_days-1)
        check_date = start_date + timedelta(days=shift)

        check_val = np.floor(shift/n_t)%10
        if (check_val==4) or (check_val==5):
            check_date = start_date + timedelta(days=shift)
            val = 'val'
            t_0 = np.floor(shift/n_t)
            val_shift = shift - n_t*t_0
            if val_shift<n_t/2:
                if check_val==4:
                    shift+=int(n_t/2)
            elif val_shift>n_t/2:
                if check_val==5:
                    shift-=int(n_t/2)
        else:
            val = 'train'
        check_date = start_date + timedelta(days=shift)
#######################
    print(val)
    mid_date = start_date + timedelta(days=shift)
    print(mid_date)
    track_dirs = copy.deepcopy(sat_dirs)
    
    shuffle(track_dirs)
    # removes CryoSat-2 observations during the testing period (with \pm 2 month safety buffer) to allow for independent testing data.
    if (mid_date<date(2018,3,1)) and (mid_date>date(2016,11,1)):
        track_dirs.remove('l3 sla data/'+"c2")
        track_dirs.remove('l3 sla data/'+"c2n")
        
    dates = []
    for i in range(n_t):
        dates.append(mid_date-timedelta(days = n_t/2-i))

    track_data = np.zeros([n_t,n,n,len(track_dirs)])

    coord_idx = randint(0, len(latitude_ocean))
    lat0 = ocean_grid[coord_idx,1]
    lon0 = ocean_grid[coord_idx,0]
    print('DUACS')
    # Grid DUACS data for 1 timestep to get MDT, longitude, latitude for sample region
    flag = 0
    t = 0 
    while flag==0:
        date_loop = dates[t]

        file = [f for f in files_grid if f'{date_loop}' in f]

        if len(file)>0:
            data_duacs = xr.open_dataset(file[0])
            data_duacs = data_duacs.mean(dim = 'time')
            mdt_grid = grid_duacs(data_duacs, n, L_x, L_y, lon0, lat0, flag)
            flag = 1
            duacs_data[:,:,0] = mdt_grid
            # duacs_data[:,:,1] = adt_grid
            # duacs_data[:,:,1] = lon_grid
            # duacs_data[:,:,2] = lat_grid
        t+=1
    print('Tracks')
    
    # inverse stereographic projection to create (lat,lon) polygon to select data to grid 
    buffer = 0.05e6 # buffer to avoid exluding data due to inverse stereographic (rather than orthographic) projection
    boxx, boxy = box([-L_x/2-buffer,L_x/2+buffer],[-L_y/2-buffer,L_y/2+buffer])
    alt0 = 0

    # calculate shape of lat, lon region to grid data from
    boxlat, boxlong = xyz2ll(boxx, boxy, 0, lat0, lon0, 0, transformer_ll2xyz, transformer_xyz2ll)
    lat_max = np.max(boxlat)
    lat_min = np.min(boxlat)
    
    if ((np.size(boxlong[boxlong>175])>0) and (np.size(boxlong[boxlong<-175])>0)):
        long_max_unshifted = np.max(boxlong[boxlong<0])
        long_min_unshifted = np.min(boxlong[boxlong>0])
    else:
        long_max_unshifted = np.max(boxlong)
        long_min_unshifted = np.min(boxlong)

    boxlat, boxlong = xyz2ll(boxx, boxy, 0, lat0, 0, 0, transformer_ll2xyz, transformer_xyz2ll)
    lat_max = np.max(boxlat)
    lat_min = np.min(boxlat)
    long_max = np.max(boxlong)
    long_min = np.min(boxlong)
    
    # define polygon bounding region to extract from data
    area = np.array([boxlong, boxlat])
    area = np.reshape(area, (2, len(boxlat)))
    area = np.swapaxes(area, 0,1)
    path = mpltPath.Path(area)

    end_idx = []
    last_sat_idx = []
    in_list = Parallel(n_jobs=n_cores)(delayed(ssh_loop)(i, n, track_dirs, L_x, L_y, lat0, lon0, path) for i in dates)
    track_data_in = np.zeros([n_t,int(1e6),4])
    for t in range(n_t):
        track_data_in[t,:,:] = in_list[t][0]
        end_idx.append(in_list[t][1])
        last_sat_idx.append(in_list[t][2])
    
    # max_end_idx = max(end_idx)
    max_last_sat_idx = max(last_sat_idx)
    in_tracks_final = np.zeros([n_t, max_last_sat_idx, 4])
    idx_diff = []
    for t in range(n_t):
        idx_diff.append(end_idx[t]-last_sat_idx[t])

    out_tracks_final = np.zeros([n_t, max(idx_diff), 4])
    for t in range(n_t):
        in_tracks_final[t,:last_sat_idx[t],:] = track_data_in[t,:last_sat_idx[t],:]
        out_tracks_final[t,:idx_diff[t],:] = track_data_in[t,last_sat_idx[t]:end_idx[t],:]
    
    ssh_input_grid = np.zeros([n_t, n, n, 2])
    ssh_output_grid = np.zeros([n_t, n, n, 2])
    
    for t in range(n_t):
        x = in_tracks_final[t,:,0]

        y = in_tracks_final[t,:,1]
        ssh_f = in_tracks_final[t,:,2]
        ssh_uf = in_tracks_final[t,:,3]

        ssh_uf = ssh_uf[x<L_x/2]
        ssh_f = ssh_f[x<L_x/2]
        y = y[x<L_x/2]
        x = x[x<L_x/2]

        ssh_uf = ssh_uf[x>-L_x/2]
        ssh_f = ssh_f[x>-L_x/2]
        y = y[x>-L_x/2]
        x = x[x>-L_x/2]

        ssh_uf = ssh_uf[y>-L_y/2]
        ssh_f = ssh_f[y>-L_y/2]
        x = x[y>-L_y/2]
        y = y[y>-L_y/2]

        ssh_uf = ssh_uf[y<L_y/2]
        ssh_f = ssh_f[y<L_y/2]
        x = x[y<L_y/2]
        y = y[y<L_y/2]

        ssh_uf = ssh_uf[x!=0]
        ssh_f = ssh_f[x!=0]
        y = y[x!=0]
        x = x[x!=0]
        
        ssh_f_grid, _,_,_ = stats.binned_statistic_2d(x, y, ssh_f, statistic = 'mean', bins=n, range = [[-L_x/2, L_x/2],[-L_y/2, L_y/2]])
        ssh_f_grid = np.rot90(ssh_f_grid)
        
        ssh_uf_grid, _,_,_ = stats.binned_statistic_2d(x, y, ssh_uf, statistic = 'mean', bins=n, range = [[-L_x/2, L_x/2],[-L_y/2, L_y/2]])
        ssh_uf_grid = np.rot90(ssh_uf_grid)
        
        ssh_input_grid[t,:,:,0] = ssh_f_grid
        ssh_input_grid[t,:,:,1] = ssh_uf_grid
    
    for t in range(n_t):
        x = out_tracks_final[t,:,0]

        y = out_tracks_final[t,:,1]
        ssh_f = out_tracks_final[t,:,2]
        ssh_uf = out_tracks_final[t,:,3]

        ssh_uf = ssh_uf[x<L_x/2]
        ssh_f = ssh_f[x<L_x/2]
        y = y[x<L_x/2]
        x = x[x<L_x/2]

        ssh_uf = ssh_uf[x>-L_x/2]
        ssh_f = ssh_f[x>-L_x/2]
        y = y[x>-L_x/2]
        x = x[x>-L_x/2]

        ssh_uf = ssh_uf[y>-L_y/2]
        ssh_f = ssh_f[y>-L_y/2]
        x = x[y>-L_y/2]
        y = y[y>-L_y/2]

        ssh_uf = ssh_uf[y<L_y/2]
        ssh_f = ssh_f[y<L_y/2]
        x = x[y<L_y/2]
        y = y[y<L_y/2]

        ssh_uf = ssh_uf[x!=0]
        ssh_f = ssh_f[x!=0]
        y = y[x!=0]
        x = x[x!=0]
        
        ssh_f_grid, _,_,_ = stats.binned_statistic_2d(x, y, ssh_f, statistic = 'mean', bins=n, range = [[-L_x/2, L_x/2],[-L_y/2, L_y/2]])
        ssh_f_grid = np.rot90(ssh_f_grid)
        
        ssh_uf_grid, _,_,_ = stats.binned_statistic_2d(x, y, ssh_uf, statistic = 'mean', bins=n, range = [[-L_x/2, L_x/2],[-L_y/2, L_y/2]])
        ssh_uf_grid = np.rot90(ssh_uf_grid)
        
        ssh_output_grid[t,:,:,0] = ssh_f_grid
        ssh_output_grid[t,:,:,1] = ssh_uf_grid

     ##################################################
    print('SST')
    # Grid the SST data:
    files_sst = GetListOfFiles('sst high res')

    date_loop = dates[0]
    date_string = f'{date_loop}'
    date_string = date_string.replace('-','')
    file = [f for f in files_sst if f'{date_string}' in f]
    if t==0:
        transformer_ll2xyz = pyproj.Transformer.from_crs(
            {"proj":'latlong', "ellps":'WGS84', "datum":'WGS84'},
            {"proj":'geocent', "ellps":'WGS84', "datum":'WGS84'},
            )

    if len(file)>0:

        data_sst = xr.open_dataset(file[0])

        if long_max_unshifted>long_min_unshifted:
            data_sst = data_sst.isel(lon = (data_sst.lon < long_max_unshifted) & (data_sst.lon > long_min_unshifted),drop = True)
        else:
            data_sst = data_sst.isel(lon = (data_sst.lon < long_max_unshifted) | (data_sst.lon > long_min_unshifted),drop = True)
        data_sst = data_sst.sel(lat=slice(lat_min,lat_max), drop = True)

        data_sst['lon'] = (data_sst['lon']-lon0+180)%360-180
        data_sst = data_sst.coarsen(lon=4, boundary = 'trim').mean().coarsen(lat=4, boundary = 'trim').mean()
        
        longitude = data_sst['lon']
        latitude = data_sst['lat']
        longitude, latitude = np.meshgrid(longitude, latitude)

        longitude = longitude.flatten()
        latitude = latitude.flatten()
        sst = np.array(data_sst['analysed_sst']).flatten()
        # sst_a = np.array(data_sst['sst_anomaly']).flatten()
        points = np.array([longitude, latitude])
        points = np.swapaxes(points,0,1)

        # check both datasets for which points lie inside this polygon and extract the relevant data into numpy arrays
        inside = path.contains_points(points)
        lon = points[:,0]
        lat = points[:,1]
        lon = lon[inside]
        lat = lat[inside]
        sst = sst[inside]

        # calculate ENU coords of data on tangent plane
        x,y,_ = ll2xyz(lat, lon, 0, lat0, 0, 0, transformer_ll2xyz)

        # grid onto square grid with smaller extent than that defined earlier
        sst[np.isnan(sst)] = 0
        sst_grid, _,_,_ = stats.binned_statistic_2d(x, y, sst, statistic = 'mean', bins=128, range = [[-L_x/2, L_x/2],[-L_y/2, L_y/2]])
        sst_grid_0 = np.rot90(sst_grid)
        
        lon[np.isnan(lon)] = 0
        lon_grid, _,_,_ = stats.binned_statistic_2d(x, y, lon, statistic = 'mean', bins=128, range = [[-L_x/2, L_x/2],[-L_y/2, L_y/2]])
        lon_grid = np.rot90(lon_grid)
        
        lat[np.isnan(lat)] = 0
        lat_grid, _,_,_ = stats.binned_statistic_2d(x, y, lat, statistic = 'mean', bins=128, range = [[-L_x/2, L_x/2],[-L_y/2, L_y/2]])
        lat_grid = np.rot90(lat_grid)

    else:
        sst_grid_0 = np.zeros([128,128])
    lon_grid = lon_grid+lon0
    lon_grid = (lon_grid+180)%360-180
    coordinate_data_final = np.stack([lon_grid, lat_grid],axis=-1)
    sst_grid_list = Parallel(n_jobs=n_cores)(delayed(sst_loop)(i, 128, files_sst, lat0, lon0, lat_min, lat_max, long_min_unshifted, long_max_unshifted, path,x,y) for i in dates[1:]) 
    sst_grid_list.insert(0,sst_grid_0)
    for t in range(n_t):
        sst_data_final[t,:,:] = sst_grid_list[t]
    ##################################################

    dates_final.append(mid_date)

    dates_final = np.array(dates_final, dtype = 'datetime64[D]')
    
    # find next index to make sure doesn't overwrite previously made data:
    files_save_dir = GetListOfFiles(save_dir, ext = '.npy')
    files_in_tracks = [i for i in files_save_dir if 'input_tracks' in i]
    n_saved = len([i for i in files_in_tracks if val in i])
    
    np.save(save_dir+f'/input_tracks{n_saved+1}'+val+'.npy', in_tracks_final)
    np.save(save_dir+f'/output_tracks{n_saved+1}'+val+'.npy', out_tracks_final)     
    np.save(save_dir+f'/duacs_data{n_saved+1}'+val+'.npy', duacs_data)    
    np.save(save_dir+f'/dates{n_saved+1}'+val+'.npy', dates_final)    
    np.save(save_dir+f'/sst{n_saved+1}'+val+'.npy', sst_data_final)
    np.save(save_dir+f'/input_ssh_grid{n_saved+1}'+val+'.npy',ssh_input_grid)
    np.save(save_dir+f'/output_ssh_grid{n_saved+1}'+val+'.npy',ssh_output_grid)
    np.save(save_dir+f'/coordinate_grid{n_saved+1}'+val+'.npy',coordinate_data_final)
    end_time = time.time()
    print(end_time-start_time)
