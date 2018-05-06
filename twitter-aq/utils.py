import zipfile
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import csc_matrix
from scipy.sparse import vstack
from scipy.sparse import hstack
from sklearn.preprocessing import normalize
pd.options.mode.chained_assignment = None
np.seterr(divide='ignore', invalid='ignore')
import os
import datetime
import sys
import scipy

def unzip_file(path,extract_path=None):
    """ Unzip a zip file in a certain directory
    
    args:
    path -- path of the zip file
    extract_path -- directory to extract the file (default: None)
    """
    
    with zipfile.ZipFile(path,"r") as zip_ref:
        zip_ref.extractall(extract_path)
        print('{} extracted to {}'.format(path,extract_path))
        
def windowed_dataset(dataset,window,function='sum'):
    """ Aggregates the columns of a dataframe grouping them by hourly timesteps using 'date' and 'hour' columns.
    
    args:
    dataset -- a pandas dataset with date and hour columns
    window -- the number of  aggregated timesteps(hours) (valid values: 6,12,24)
    function -- aggregation method (default: sum)
    """
    
    nbins = int(24/window)
    bins = [i*window-1 for i in range((nbins+1))]
    group_names = ['window_'+str(i+1) for i in range(nbins)]
    dataset.reset_index(inplace=True)
    dataset['window'] = pd.cut(dataset['hour'], bins, labels=group_names) #create a helper column which is the name of the window each hour corresponds (e.g if window == 6 hour 10:00 correspnds to window2)
    dataset.drop('hour',axis=1,inplace=True)
    if function=='sum':
        dataset =dataset.groupby(['date','window']).sum()
    elif function=='max':
        dataset =dataset.groupby(['date','window']).max()
    elif function=='last':
        dataset =dataset.groupby(['date','window']).last()
    elif function=='mean':
        dataset =dataset.groupby(['date','window']).mean()
    return dataset


def create_index_from_timestamps(dataset,window):
    """ Recreate a pandas index from the timestamps
    
    args:
    dataset -- the raw dataframe from read_csv function
    window -- the window of the dataset (valid values: 6,12,24)
    """

    dataset['min_timestamp'] = pd.to_datetime(dataset['min_timestamp'])
    dataset['date'] = dataset['min_timestamp'].dt.date
    dataset['hour'] = dataset['min_timestamp'].dt.hour
    index = windowed_dataset(dataset,window).index #take the index of a windowed dataset
    dataset.drop('date',axis=1,inplace=True)
    dataset.drop('window',axis=1,inplace=True)
    dataset.drop('index',axis=1,inplace=True)
    new_dataset = pd.DataFrame(index=index,data=dataset.values,columns=dataset.columns)#create a new dataset with index created
    return new_dataset    


def csv_format_to_sparse(vector,shape):
    """ Tranforms tab seperated sparse vector representation (index -> value) to csr_matrix
    
    args:
    vector -- the csv vector representation 
    shape -- the shape of the sparse array
    
    returns:
    a csr_matrix
    """

    if pd.isnull(vector):
        return np.nan
    else:    
        indices = [int(i.split('->')[0]) for i in vector.split('\t')]#get the indices
        indices_x = [0]*len(indices)#x axis indices are zero because it is a vector
        data = [float(i.split('->')[1]) for i in vector.split('\t')]# get the data
        return csr_matrix((data,(indices_x,indices)),shape=shape)
    
def create_aggregated_versions_of_bow_vector(dataset,bow_vector,aggregate_timesteps = [2,3,4]):
    """ Creation of columns with aggregated versions of a bag-of-words feature on a pandas dataframe
    
    args:
    dataset -- pandas dataframe that has a sparse bag-of-words feature
    bow_vector -- the bag-of-words feature column name
    aggregate_timesteps --  how many timesteps behind will be aggregated (e.g 2 means current and previous), (default:[2,3,4])    
    """
    
    for n in aggregate_timesteps:
        #to calculate moving sum we replace NaN values with sparse arrays with no items
        dataset_replaced_nan =  dataset.copy()
        dataset_replaced_nan.loc[dataset_replaced_nan[bow_vector].isnull(),bow_vector] = dataset_replaced_nan.loc[dataset_replaced_nan[bow_vector].isnull(),bow_vector].apply(lambda x: csr_matrix((1,10000)))
        
        ret = np.nancumsum(dataset_replaced_nan[bow_vector])
        ret[n:] = ret[n:] - ret[:-n]
        
        #after this we replace zero sparse arrays with NaN
        ret[np.where(dataset_replaced_nan[bow_vector].apply(lambda x: True if len(x.data)==0 else False))[0]]= np.nan
       
        dataset[bow_vector+'_'+str(n)] = ret


def normalize_bow_feature(dataset,bow_vector,axis = 1,suffix = '_normalized'):
    """ L2 normalization on a sparse bag-of-words feature on a pandas dataframe
    
    args:
    dataset -- pandas dataframe that has a sparse bow feature
    bow_vector -- the bow feature vector
    axis -- (valid values: (0,1))
    suffix : suffix for the new column name (default: '_normalized')
    """
    stacked = vstack(dataset[bow_vector].loc[~dataset[bow_vector].isnull()])#vertical stacking  of all sparse vectors to one universal sparse array (ignoring NaNs)
    dataset[bow_vector+suffix] =dataset[bow_vector]
    dataset[bow_vector+suffix].loc[~dataset[bow_vector].isnull()] =list(normalize(stacked,norm='l2',axis=1))#l2 normalization of not NaN rows
    
def create_dataset_from_csv(path,window,features):
    """ Transforms csv files to pandas dataframes with proper sparse representation of bag-of-words feature and aggregated versions of them
    
    args:
    path -- the path of the csv file
    window -- window -- the window of the dataset (valid values: 6,12,24)
    feature -- the sparse bag-of_-ords feature
    
    returns:
    a pandas dataframe
    """
    dataset = pd.read_csv(path)
    dataset = create_index_from_timestamps(dataset,window)
    for feature in features:
        dataset[feature] = dataset[feature].apply(lambda x: csv_format_to_sparse(x,(1,10000)))
        create_aggregated_versions_of_bow_vector(dataset,feature)
    dataset = dataset.astype(dtype= {"#aqs":np.float64,"#aqs":np.float64,"#high":np.float64,"#tw":np.float64,"pm25":np.float64,"idw_pm25":np.float64})
    return dataset

def save_results(filename,results):
    """ Creates a csv file with results to results/ directory
    
    args:
    filename -- the name of the file
    results -- a list with results

    """
    #save results
    columns = ['country','city','window','setup','baseline','sample_weights','fs_method','feature_number',
                    'feature_type','feature_details','representation','1_step_regressor',
     '2_step_regressor','rmse','mae','precision(high)','recall(high)','f_measure(high)']

    data = np.concatenate(results,axis=0).reshape(-1,len(columns))#concat results
    
    #create a dataframe and save it to a csv file
    dataframe = pd.DataFrame(data =data,columns=columns)
    dataframe[['rmse','mae','precision(high)','recall(high)','f_measure(high)']] = dataframe[['rmse','mae','precision(high)','recall(high)','f_measure(high)']].astype(np.float64)
    results_path = 'results/'
    if not os.path.exists(results_path):
        os.mkdir(results_path)
    dataframe.to_csv(os.path.join(results_path,filename+'.csv'),index=False)       

