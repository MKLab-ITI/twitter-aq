import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import csc_matrix
from scipy.sparse import vstack
from scipy.sparse import hstack
import pandas as pd 

def vcorrcoef(X,y):
    """ Computes the vectorized correlation coefficient
    
    code taken from here: https://waterprogramming.wordpress.com/2014/06/13/numpy-vectorized-correlation-coefficient/
    
    """
    Xm = np.reshape(np.mean(X,axis=1),(X.shape[0],1))
    ym = np.mean(y)
    r_num = np.sum((X-Xm)*(y-ym),axis=1)
    r_den = np.sqrt(np.sum((X-Xm)**2,axis=1)*np.sum((y-ym)**2))
    r = r_num/r_den
    return r

def feature_selection(datasets,cities,bow_features_column,target_column,window,method = 'Conly'):
    """Performs feature selection on a spcified feature column of a pandas dataframe:
    
    args:
    datasets -- dict containing pandas dataframes from each city and window
    cities -- list of cities in a country(english_cites,american_cities)
    bow_features_column -- the column name of the bow feature
    target_column -- the ground truth column name
    window -- the number of  aggregated timesteps(hours) (valid values: 6,12,24)
    method -- the feature selection method (valid values: 'Conly','S&C','Sonly','Conly_changed','Conly')
    
    returns:
    a tuple containing a list of the feature scores in descending order and a list with their respective indices
    """
    print('feature selection')
    stacked_datasets = []
    correlations = []
    for city in cities:
        dataset = datasets[city+'_'+str(window)].copy()
        dataset.dropna(inplace=True)
        features = np.array(vstack(dataset.loc[:,bow_features_column].tolist()).todense())
        pm = dataset[target_column].values.reshape(-1,1).astype(np.float64)
        correlation  = vcorrcoef(features.T,pm.T)
        correlation  = correlation.reshape(1,-1)
        correlations.append(correlation)
        stacked_datasets.append(dataset)
    correlations = np.concatenate(correlations,axis=0)   
    
    allcities = pd.concat(stacked_datasets,axis=0)
    allcities.dropna(inplace=True)
    all_features= np.array(vstack(allcities.loc[:,bow_features_column].tolist()).todense())
    all_pm = allcities[target_column].values.reshape(-1,1).astype(np.float64)
    all_correlations  = vcorrcoef(all_features.T,all_pm.T)
    all_correlations  = all_correlations.reshape(-1)

    all_abs_correlations = np.abs(all_correlations)
    argsorted_all_abs_correlations =np.argsort(all_abs_correlations)[::-1]#descending
    sorted_all_abs_correlations = all_abs_correlations[argsorted_all_abs_correlations]
    isfinite = np.isfinite(sorted_all_abs_correlations)
    argsorted_all_abs_correlations = argsorted_all_abs_correlations[isfinite]
    sorted_all_abs_correlations = all_abs_correlations[argsorted_all_abs_correlations]
    
    if method == 'Conly_a':
        print(sorted_all_abs_correlations.shape)
        return(sorted_all_abs_correlations,argsorted_all_abs_correlations)

    mean_abs_sum_correlations = np.abs(np.sum(correlations,axis=0))/len(cities)
    argsorted_mean_abs_sum_correlations =np.argsort(mean_abs_sum_correlations)[::-1]#descending
    sorted_mean_abs_sum_correlations = mean_abs_sum_correlations[argsorted_mean_abs_sum_correlations]
    isfinite = np.isfinite(sorted_mean_abs_sum_correlations)
    argsorted_mean_abs_sum_correlations = argsorted_mean_abs_sum_correlations[isfinite]
    sorted_mean_abs_sum_correlations = mean_abs_sum_correlations[argsorted_mean_abs_sum_correlations]

    mean_sum_abs_correlations = np.sum(np.abs(correlations),axis=0)/len(cities)
    argsorted_mean_sum_abs_correlations =np.argsort(mean_sum_abs_correlations)[::-1]#descending
    sorted_mean_sum_abs_correlations = mean_sum_abs_correlations[argsorted_mean_sum_abs_correlations]
    isfinite = np.isfinite(sorted_mean_sum_abs_correlations)
    argsorted_mean_sum_abs_correlations = argsorted_mean_sum_abs_correlations[isfinite]
    sorted_mean_sum_abs_correlations = mean_abs_sum_correlations[argsorted_mean_sum_abs_correlations]

    variance_correlations = np.var(correlations,axis=0)
    argsorted_variance_correlations =np.argsort(variance_correlations)
    sorted_variance_correlations = variance_correlations[argsorted_variance_correlations]
    isfinite = np.isfinite(sorted_variance_correlations)
    argsorted_variance_correlations = argsorted_variance_correlations[isfinite]
    sorted_variance_correlations = variance_correlations[argsorted_variance_correlations]

    if method == 'Conly':
        return(sorted_mean_abs_sum_correlations,argsorted_mean_abs_sum_correlations)
    if method == 'Conly_changed':
        return(sorted_mean_sum_abs_correlations,argsorted_mean_sum_abs_correlations)
    if method == 'Sonly':
        return(sorted_variance_correlations,argsorted_variance_correlations)

    if method=='S&C':
        order = []
        for i in list(argsorted_variance_correlations): 
            order.append(list(argsorted_mean_abs_sum_correlations).index(i) +list(argsorted_variance_correlations).index(i))
            arg_order=np.argsort(np.array(order))
            arg_sorted=argsorted_variance_correlations[arg_order]
        return (None,arg_sorted)
    if method=='S&C_changed':
        order = []
        for i in list(argsorted_variance_correlations): 
            order.append(list(argsorted_mean_sum_abs_correlations).index(i) +list(argsorted_variance_correlations).index(i))
            arg_order=np.argsort(np.array(order))
            arg_sorted=argsorted_variance_correlations[arg_order]
        return (None,arg_sorted)