# -*- coding: utf-8 -*-
"""
Functions used to produce a line forecast.  
@author: Elynn
"""
from mpl_toolkits.basemap import Basemap
from skimage.filters import sobel
import numpy as np
import pandas as pd
from netCDF4 import Dataset
from pvlib.solarposition import get_solarposition
from matplotlib.path import Path
import datetime

def nominal_reflectance_GOES15(data):
    """Calculate reflectance from raw GVAR (GOES15)
    Parameters
    ----------
    data: float, or array of float
        GOES15 visible channel GVAR
    Returns
    -------
    output: float, or array of float
        Nominal reflectance (albedo)
    """
    k = 0.001106
    Xspace = 29
    beta = k*Xspace
    c = 1.442
    output = (k*data-beta)*c
    return output

def nominal_reflectance_to_cloud(reflectance,time,region,background_albedo,threshold_cutoff):
    '''Takes in nominal reflectance then normalized it by cosine(solarZenith) to remove
       brightness difference caused by sun position. Then compares this albedo value to
       clearsky albedo. If the difference is larger than 15.5%, flag as a cloud.
    Parameters
    ----------
    reflectance: numpy array (2d)
        Nominal reflectance 
    time: datetime
        Current date/time in UTC
    region: str
        'BayArea' or 'SD'. Will use different lat/lon as zenith angle proxy.
    background_albedo: str
        Background albedo file
    Returns
    -------
    cloud: numpy array (2D)
        Cloud flag (0 is clear, everything larger than 0.155 is cloudy).
    '''
    if region == 'BayArea':
        lat_c = 37.6
        lon_c = -122
    elif region == 'SD':
        lat_c = 34.6
        lon_c = -119       
    cs = Dataset(background_albedo)
    cs_albedo = cs['cs_albedo'][:,:]
    zenith = get_solarposition(time,lat_c,lon_c)['zenith'].as_matrix()
    cos_zenith = np.cos(np.deg2rad(zenith[0]))
    cloud = reflectance/cos_zenith - cs_albedo
    cloud[cloud<threshold_cutoff] = 0
    return cloud

def get_albedo_from_VIS(current_t,albedo_cutoff):
    '''Get albedo at input time from VIS image
    Parameters
    ----------
    current_t: datetime.datetime
        Input time in UTC
    albedo_cutoff: float
        Albedo difference from clearsky albedo
    Returns
    -------
    lat, lon: np.array 2D float
        Latitude and longitude in decimal
    albedo: np.array 2D floar
        Albedo for current image
    '''
    year = str(current_t.year)
    month = str(current_t.month).zfill(2)
    if year == '2016':
        f = Dataset('~/Satellite_Images/GOES15/'+year+'_'+month+'/BayArea/clearsky_test/goes15.'+year+'.'+str(current_t.dayofyear)+'.'+str(current_t.hour)+str(current_t.minute).zfill(2)+'BAND_01.nc')
    elif year == '2015':
        f = Dataset('~/Satellite_Images/GOES15/'+year+'_'+month+'/goes15.'+year+'.'+str(current_t.dayofyear)+'.'+str(current_t.hour)+str(current_t.minute).zfill(2)+'BAND_01.nc')
    data = f['data'][0,:,:]
    lat = f['lat'][:,:]
    lon = f['lon'][:,:]
    data = data.astype(int)/32.
    f.close()
    albedo = nominal_reflectance_GOES15(data)
    cs_path = '~/Satellite_Images/GOES15/2016_08/BayArea/BayArea_clearsky_albedo_'+str(current_t.hour)+'Z_JJA_2016.nc'
    albedo = nominal_reflectance_to_cloud(albedo,current_t,'BayArea',cs_path,albedo_cutoff)
    return lat, lon, albedo

def get_longest_edge(m,x,y,edges):
    '''Get longest consecutive edge from sobel edge detection
    Parameters
    ----------
    m: mpl_toolkits.basemap.Basemap
        Basemap used in current plot
    x, y: np.array 2D float
        Longitude and latitude in map coordinate
    Returns
    -------
    x1, y1: np.array float
        Edge longitude and latitude in map coordinate
    lonpt, latpt: np.array float
        Edge longitude and latitude in decimal
    '''
    temp = m.contour(x,y,edges,[0,np.unique(edges)[-1]],cmap=plt.get_cmap('jet'),alpha=0.)
    n = len(temp.collections[0].get_paths())
    paths = []
    for cc in range(n):
        paths.append(len(temp.collections[0].get_paths()[cc]))
    index = np.argsort(paths)
    p = temp.collections[0].get_paths()[index[-1]]#[np.argmax(paths)]
    v = p.vertices
    x1 = v[:,0]
    y2 = v[:,1]
    lonpt, latpt = m(x1,y2,inverse=True)
    index = np.where((latpt>=34.5)&(latpt<=42.)) #adjust for different domain
    x1, y2, lonpt, latpt = x1[index], y2[index], lonpt[index], latpt[index]
    return x1, y2, lonpt, latpt

def best_fit(X, Y):
    '''Simple best fit line
    Returns
    -------
    a: float
        intercept
    b: float
        slope
    '''
    xbar = sum(X)/len(X)
    ybar = sum(Y)/len(Y)
    n = len(X) # or len(Y)
    numer = sum([xi*yi for xi,yi in zip(X, Y)]) - n * xbar * ybar
    denum = sum([xi**2 for xi in X]) - n * xbar**2
    b = numer / denum
    a = ybar - b * xbar
    return a, b

def extrapolate_elevation_in_time(issue_time,input_time,elevation,minute_ahead,station_elevation):
    '''Extrapolate elevation in time until current pixel elevation, returns burnoff time
    Parameters
    ----------
    issue_time: float
        Forecast issue time in UTC
    input_time: array of float
        Time stamps from first avaiable up to current time (forecast issue time)
    elevation: array of float
        Elevation (m) time series up to forecast issue time
    minute_ahead: int
        Forecast horizon in minute
    station_elevation: float
        Elevation for the station
    Returns
    -------
    extrapolated_t: array of float
        Extrpolated time in UTC
    extrapolated_elevation: array of float
        Elevation extrapolated up to minute_ahead from forecast issue time
    burnOff_time: float
        Burnoff time in UTC
    '''
    non_nan_index = ~np.isnan(elevation)
    a, b = best_fit(input_time[non_nan_index],elevation[non_nan_index])
    if b>0: #if elevation is increasing in time
        b = 0.
        a = elevation[-1] #kept at current elevation
        burnOff_time = np.nan #never burns off
    else:
        burnOff_time = (station_elevation-a)/(b)
        if burnOff_time<input_time[-1]:
            burnOff_time = (station_elevation-a)/(b)
    extrapolated_t = [input_time[-1]+minute/60. for minute in range(0,minute_ahead+15,15)]
    extrapolated_t = np.array(extrapolated_t)
    extrapolated_elevation = [a+b*ti for ti in extrapolated_t]
    extrapolated_elevation = np.array(extrapolated_elevation)
    extrapolated_elevation[extrapolated_elevation>elevation[-1]] = elevation[-1]
    return extrapolated_t,extrapolated_elevation,burnOff_time

def exponentially_interpolate_current_2_burnoff_time(extrapolated_t,current_kt,burnOff_time,region):
    '''Linearly interpolate between [current_time,burnoff_time] and [current_kt,1.0]
    Parameters
    ----------
    extrapolated_t: array of float
        Extrpolated time in UTC
    current_kt: float
        Clearsky index at current issue time
    burnOff_time: float
        Dissipation time
    region: str
    BayArea for northern california and SD for southern california
    Returns
    ------- 
    interpolated_kt: array of float
        Clearsky index at extrpolated_t
    '''
    if region == 'BayArea':
        a = 0.0025
        b = -5.959
        c = 0.0218 #coefficients can be found in /exp_fit_test/exponential_fit_2003_2015_coeff.csv
        sunrise = 14.
    elif region == 'SD':
        a = 0.0052#0.01237588
        b = -5.2376#-4.38004067
        c = 0.0137#0.00633801
        sunrise = 13.5
    if np.isnan(burnOff_time):
        exp_fitted_kt = np.ones(len(extrapolated_t))*current_kt
    elif burnOff_time < extrapolated_t[0]:
        exp_fitted_kt = np.ones(len(extrapolated_t))
        exp_fitted_kt[0] = current_kt
    else:
        scaled_t = (extrapolated_t - sunrise) / (burnOff_time - sunrise) #scale time then center at sunrise, 1330Z for June in SD, 14Z for Aug in NoCA
        scaled_t[scaled_t>1.] = 1. #all time after dissipation time should be 1
        exp_fitted_kt = (a * np.exp(-b * scaled_t) + c)*(1-current_kt)+current_kt #the first part is the exponential fit, then *(1-current)+current to shift back
        exp_fitted_kt[0] = current_kt
    return exp_fitted_kt

def exponentially_interpolate_current_2_burnoff_time_dynamic_sunrise_time(extrapolated_t,current_kt,burnOff_time,region,current_date,SZA):
    '''Linearly interpolate between [current_time,burnoff_time] and [current_kt,1.0]
    Parameters
    ----------
    extrapolated_t: array of float
        Extrpolated time in UTC
    current_kt: float
        Clearsky index at current issue time
    burnOff_time: float
        Dissipation time
    region: str
    BayArea for northern california and SD for southern california
    current_date: datetime
        Current date to decide the sunrise time (cos(SZA)>0.1)
    lat, lon: float
    Returns
    ------- 
    interpolated_kt: array of float
        Clearsky index at extrpolated_t
    '''
    cosSZA = np.cos(np.deg2rad(SZA))
    if (region == 'BayArea') & (cosSZA>0.1):
        a = 0.0025
        b = -5.959
        c = 0.0218 #coefficients can be found in /exp_fit_test/exponential_fit_2003_2015_coeff.csv
        sunrise = 14.0
    else:
        a = 0.0063
        b = -5.0406
        c = 0.02183437 #coefficients can be found in /exp_fit_test/exponential_fit_2003_2015_coeff_differnet_sunrise.csv
        sunrise = 14.5
    if np.isnan(burnOff_time):
        exp_fitted_kt = np.ones(len(extrapolated_t))*current_kt
    elif burnOff_time < extrapolated_t[0]:
        exp_fitted_kt = np.ones(len(extrapolated_t))
        exp_fitted_kt[0] = current_kt
    else:
        scaled_t = (extrapolated_t - sunrise) / (burnOff_time - sunrise) #scale time then center at sunrise, 1330Z for June in SD, 14Z for Aug in NoCA
        scaled_t[scaled_t>1.] = 1. #all time after dissipation time should be 1
        exp_fitted_kt = (a * np.exp(-b * scaled_t) + c)*(1-current_kt)+current_kt #the first part is the exponential fit, then *(1-current)+current to shift back
        exp_fitted_kt[0] = current_kt
    return exp_fitted_kt