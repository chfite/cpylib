
import numpy as np
import xarray as xr
import pandas as pd
from scipy.stats.mstats import linregress
from sklearn.metrics import mean_squared_error


def linear_regression(array1, array2):
    #ensure they are arrays
    array1 = np.asanyarray(array1) 
    array2 = np.asanyarray(array2) 
    #remove possible nans
    array1 = array1[~np.isnan(array1)]
    array2 = array2[~np.isnan(array2)]
    m, b, rval, pval, stderr = linregress(np.ravel(array1),np.ravel(array2))
    r2 = round(rval**2, 2) #round to two decimal places
    if pval <= 0.01: #if smaller, then print change to scientific notation
        pval = '{:.1e}'.format(pval) 
    else: #if larger, round to 2 decimalm places
        pval = round(pval,2) 
    m = round(m,2) #round to two decimal places
    return m, b, rval, r2, pval, stderr
    
    
 def rootmean_square_error(array1, array2):
    #ensure they are arrays
    array1 = np.asanyarray(array1) 
    array2 = np.asanyarray(array2) 
    #remove possible nans
    array1 = array1[~np.isnan(array1)]
    array2 = array2[~np.isnan(array2)]
    #calc rmse
    rmse = np.sqrt(mean_squared_error(np.ravel(array1),np.ravel(array2)))
    return rmse


def percent_change(oldval,newval): 
    perchange = (newval-oldval)/np.abs(oldval) * 100 
    return perchange 


def mean_bias_error(observed,predicted):
    '''
     note: this set up subtracts the observed from the predicted.
           Doing it this way means a (+) MBE means the prediction
           is overestimating and (-) the prediction is underestimating
    '''
    predicted, observed = np.asarray(predicted), np.asarray(observed)
    diff = predicted - observed
    sum_of_diffs = np.nansum(diff)
    num_points = len(diff[~np.isnan(diff)]) # number of points used in sum, with Nans removed
    mbe = sum_of_diffs / num_points
    return mbe   
    
    
    
    
    
    
    
    
    
