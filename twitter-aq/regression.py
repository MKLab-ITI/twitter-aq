import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import csc_matrix
from scipy.sparse import vstack
from scipy.sparse import hstack
from sklearn.model_selection import KFold
pd.options.mode.chained_assignment = None
np.seterr(divide='ignore', invalid='ignore')
import os
import sys
import scipy
from sklearn.metrics import precision_recall_fscore_support
import sklearn.metrics as sm
from feature_selection import feature_selection

def training(regressor,xcols,ycol,dataset,mask = None,additional_features = None,keep=None,weights=False,verbose=False):
    """ Training a specified regressor from a pandas dataframe.
    
    args:
    regressor -- an sklearn regressor (e.g GradientBoostingRegressor)
    xcols -- the columns used for training samples
    ycol -- the column of ground truth values
    dataset -- the training pandas dataframe
    mask -- feature selection mask for keeping certain feature of a sparse feature vector (default: None)
    additional_features -- numpy array of feature to be concatenated with training samples (default: None)
    keep -- features to keep in the dataset in order to drop NaN values only from those (usually contains xcols plus other features used for 2-step regression)
    weights -- whether or not to weight each training sample based on distance of each city (used in cross city setups) (default: False)
    verbose -- (default: False)
    
    returns:
    a tuple of a training classifier and training predictions
    """
    
    predictors = xcols.copy()
    train_dataset = dataset.copy()
    if keep is None:
        columns = xcols+[ycol]
    else:
        columns = keep+[ycol]
        
    if weights:
        columns = columns + ['weights']
        
    train_dataset= train_dataset.loc[:,columns].dropna() # keep only certain columns and drop NaN values
            
    if verbose:
        print( 'Train dataset shape: ',train_dataset.shape)
        print( 'Train dataset columns: ',train_dataset.columns)
    if train_dataset.shape[0] == 0:
        print( 'There are no valid training data')
        sys.exit()
    
    #horizontal contatenation of features (if there is a sparse feature then all features are tranformed to a sparse array)
    processed_predictors=[]   
    sparse = False
    for predictor in predictors:
        if isinstance(train_dataset.loc[:,predictor].iloc[0],scipy.sparse.csr.csr_matrix): #if it is a sparse feature
            pred_list = train_dataset.loc[:,predictor].tolist()
            stacked = vstack(pred_list)
            processed_predictors.append(stacked)
            sparse =True
        else:
            pred_list =train_dataset.loc[:,predictor].as_matrix().reshape(-1, 1).astype(np.float64)
            processed_predictors.append(pred_list)
    if sparse:
        X = hstack(processed_predictors)
        if mask is not None:
            if len(processed_predictors)==1: # apply mask only if there is only one sparse feature (did not implement this with other features)
                X = X[:,mask]
            else:
                print('MASK NOT APPLIED: there are multiple features')
        if additional_features is not None:
            X= hstack([X,additional_features])   
    else:
        X = np.concatenate(processed_predictors,axis=1)
        if additional_features is not None:
            X= np.concatenate([X,additional_features],axis=1) 
        
    y= train_dataset[ycol].as_matrix().reshape(-1, 1) 
   
    features_number=X.shape[1]
    if verbose:
        print( 'X train dataset shape: ',X.shape)
        print( 'y train dataset shape: ',y.shape)
    #training
    if weights:
        fit =regressor.fit(X,y.ravel(),sample_weight = train_dataset['weights'].values)
    else:
        fit =regressor.fit(X,y.ravel()) 
    ypred = fit.predict(X)
    rmse_train = np.sqrt(sm.mean_squared_error(y, ypred))
    mae_train = sm.mean_absolute_error(y, ypred)

    if verbose:
        print( 'stats on training set')
        print( 'MSE : ',sm.mean_squared_error(y, ypred)  )
        print( 'RMSE : ',rmse_train )
        print( 'MAE : ',mae_train )
        print( '--------------------------' )
    return (fit,ypred.reshape(-1,1))




def testing(fit,xcols,ycol,dataset,mask = None,additional_features=None,keep=None,classification = False,verbose=False):
    """ Testing a specified regressor from a pandas dataframe.
    
    args:
    fit -- a trained regressor
    xcols -- the columns used for testing samples
    ycol -- the column of ground truth values
    dataset -- the test pandas dataframe
    mask -- feature selection mask for keeping certain feature of a sparse feature vector (default: None)
    additional_features -- numpy array of feature to be concatenated with testing samples (default: None)
    keep -- features to keep in the dataset in order to drop NaN values only from those (usually contains xcols plus other features used for 2-step regression)
    classification -- whether or not to calculate classification metrics after transforming grand trith values to labels (default: False)
    verbose -- (default: False)
    
    returns:
    if classification == False: a tuple of a test RMSE,test MAE and test predictions
    if classification == True: a tuple of a test RMSE,test MAE and test predictions, precision, recall, fscore
    """ 
    
    predictors = xcols.copy()  
    test_dataset = dataset.copy()
         
    if keep is None:
        test_dataset = test_dataset.loc[:,xcols+[ycol]].dropna()
    else:
        test_dataset = test_dataset.loc[:,keep+[ycol]].dropna()
                
    if verbose:
        print( 'test dataset shape: ',test_dataset.shape)
        print( 'test dataset columns: ',test_dataset.columns)
    if test_dataset.shape[0] == 0:
        print( 'there are no valid test data')
        sys.exit()
        
    processed_predictors=[]   
    sparse = False
    for predictor in predictors:
        if isinstance(test_dataset.loc[:,predictor].iloc[0],scipy.sparse.csr.csr_matrix):
            pred_list = test_dataset.loc[:,predictor].tolist()
            stacked = vstack(pred_list)
            processed_predictors.append(stacked)
            sparse =True
        else:
            pred_list =test_dataset.loc[:,predictor].as_matrix().reshape(-1, 1).astype(np.float64)
            processed_predictors.append(pred_list)
    if sparse:
        Xtest= hstack(processed_predictors)
        if mask is not None:
            if len(processed_predictors)==1: # apply mask only if there is only one sparse feature
                Xtest = Xtest[:,mask]
            else:
                print('MASK NOT APPLIED: there are multiple features')
        if additional_features is not None:
            Xtest= hstack([Xtest,additional_features])   
    else:
        Xtest = np.concatenate(processed_predictors,axis=1)
        if additional_features is not None:
            Xtest= np.concatenate([Xtest,additional_features],axis=1)   
        
    ytest= test_dataset[ycol].as_matrix().reshape(-1, 1) 
    if verbose:
        print( 'X test dataset shape: ',Xtest.shape)
        print( 'y test dataset shape: ',ytest.shape)
        
    ypredtest = fit.predict(Xtest)
    rmse_test = np.sqrt(sm.mean_squared_error(ytest, ypredtest))
    mae_test = sm.mean_absolute_error(ytest, ypredtest)    
    results = (rmse_test,mae_test,ypredtest.reshape(-1,1))   
    
    if(classification==True):
        test_dataset[ycol+'_cat'] = test_dataset[ycol].apply(to_labels)
        ytest_cat= test_dataset[ycol+'_cat'].as_matrix().reshape(-1, 1)
        to_labels_v = np.vectorize(to_labels)
        ypredtest_cat = to_labels_v(ypredtest)
        precision = precision_recall_fscore_support(ytest_cat,ypredtest_cat,labels=['good','bad'])[0]
        recall = precision_recall_fscore_support(ytest_cat,ypredtest_cat,labels=['good','bad'])[1]
        fscore = precision_recall_fscore_support(ytest_cat,ypredtest_cat,labels=['good','bad'])[2]
        results = (rmse_test,mae_test,ypredtest.reshape(-1,1),precision,recall,fscore)   
     
    if verbose:
        print( 'stats on test set')
        print( 'MSE : ',sm.mean_squared_error(ytest, ypredtest)  )
        print('RMSE : ',rmse_test)
        print( 'MAE : ',mae_test)
        print( '--------------------------' )
    return results
    
def first_step_regression(train,test,regressor,first_predictor,mask=None,keep =None,weights=False,verbose=False):
    """Perfrorm training and testing using the first step predictors
    
    args:
    train -- the training pandas dataframe
    test -- the testing pandas dataframe
    regressor --  an sklearn regressor (e.g GradientBoostingRegressor)
    first_predictor -- a list of first step predictors
    mask -- feature selection mask for keeping certain feature of a sparse feature vector (default: None)
    keep -- features to keep in the dataset in order to drop NaN values only from those (usually contains xcols plus other features used for 2-step regression)
    weights -- whether or not to weight each training sample based on distance of each city (used in cross city setups) (default: False)
    verbose -- (default: False)
    
    returns:
    a tuple of arrays with training and test predictions
    """ 
    
    train_list = []
    test_list = []
    if weights:
        fit,train_predictions = training(regressor,first_predictor,'pm25',train,mask=mask,verbose=verbose,keep=keep,weights=True)
    else:
        fit,train_predictions = training(regressor,first_predictor,'pm25',train,mask=mask,verbose=verbose,keep=keep)
    test_rmse,test_mae,test_predictions = testing(fit,first_predictor,'pm25',test,mask=mask,verbose=verbose,keep=keep)
    return(train_predictions,test_predictions)


def to_labels(x,threshold = 20):
    """Tranform pm2.5 value to label
    
    args:
    x -- a pm2.5 value
    threshold -- threshold to assign labels (default: 20)
    
    returns:
    a label
    """ 
    if np.isnan(x):
        return np.nan
    if x < threshold:
        return 'good'
    else:
        return 'bad'
    
#split datasets in test and train by odd or even month
def split_dataset_even_odd_months(dataset,odd='train'):
    """Split a pandas dataframe to even and odd months
    
    args:
    dataset -- a pandas dataframe with a date and hour multiIndex
    odd -- which dataset to include odd months (default: 'train')
    
    returns:
    a tuple of traing and testing dataframes
    """ 
    dataset['index_col'] = dataset.index.get_level_values(0)
    dataset['index_col'] = pd.to_datetime(dataset['index_col'])
    dataset['month'] = dataset.index_col.dt.month
    if odd=='train':
        train_dataset = dataset.loc[dataset.month % 2 == 1]
        test_dataset = dataset.loc[dataset.month % 2 == 0]
    elif odd=='test':
        train_dataset = dataset.loc[dataset.month % 2 == 0]
        test_dataset = dataset.loc[dataset.month % 2 == 1]
        
    dataset.drop(['month','index_col'],axis=1,inplace=True) # remove helper columns
    train_dataset.drop(['month','index_col'],axis=1,inplace=True) 
    test_dataset.drop(['month','index_col'],axis=1,inplace=True) 
    return (train_dataset,test_dataset)

def create_all_vs_one_datasets(datasets,cities_dict,city,window,weights=None):
    """ Create 'cross city (i.e. all to one)' datasets
    
    args:
    datasets -- dict containing pandas dataframes from each city and window
    cities_dict -- a dict which as the country code as key (e.g UK) and the list of cities in this country as a value
    city -- the city name to be used as testing    
    window -- the number of  aggregated timesteps(hours) (valid values: 6,12,24)
    weights --  an Inverse Distance Weighting dataframe for each city in every country, if weights is not None then the column weight is added to the dataframe representing the inverse distance weighting value for each training sample (defaul: None)

    returns:
    a tuple of traing and testing dataframes
    """ 
    for code in cities_dict:
        if city in cities_dict[code]:
            use_cities = cities_dict[code].copy()
    
    test_dataset = datasets[city+'_'+str(window)]
    
    use_cities.remove(city)
    if weights is not None:
        weight_matrix = weights.loc[city].values
    stacked_datasets = []
    for c,city in enumerate(use_cities):
        dataframe = datasets[city+'_'+str(window)]
        if weights is not None:
            dataframe['weights'] = weight_matrix[c]#create a weight column with the corresponding wieght of each city if 
        stacked_datasets.append(dataframe)        
    train_dataset = pd.concat(stacked_datasets,axis=0)     
    return (train_dataset,test_dataset)

def create_cv_bow_model(regressor,xcols,ycol,dataset,mask=None,keep=None,weights=False,cv=3,verbose=False):
    """Create a cross validation model of a bog_of_words feature and calculate train predictions
    in order to use them in a two step regression setup
    
    args:
    regressor --  an sklearn regressor (e.g GradientBoostingRegressor)
    xcols -- the columns used for training samples
    ycol -- the column of ground truth values
    dataset -- the training pandas dataframe
    mask -- feature selection mask for keeping certain feature of a sparse feature vector (default: None)
    keep -- features to keep in the dataset in order to drop NaN values only from those (usually contains xcols plus other features used for 2-step regression)
    weights -- whether or not to weight each training sample based on distance of each city (used in cross city setups) (default: False)
    cv -- number of KFold splits (default: 3)
    verbose -- (default: False)
    
    returns:
    a tuple of the training dataset and sorted training predictions
    """ 
    
    predictors = xcols.copy()
    full_dataset = dataset.copy()
    
    if keep is None:
        columns = xcols+[ycol]
    else:
        columns = keep+[ycol]
        
    if weights:
        columns = columns + ['weights']
        
    full_dataset= full_dataset.loc[:,columns].dropna() # keep only certain columns and drop NaN values
            
    if verbose:
        print( 'train dataset shape: ',full_dataset.shape)
        print( 'train dataset columns: ',full_dataset.columns)
    if full_dataset.shape[0] == 0:
        print( 'there are no valid train data')
        return np.nan
    
    kf = KFold(n_splits = cv, shuffle = True)
    
    index= []
    predictions = []
    for i,result in enumerate(kf.split(full_dataset)):
        #print('cv:'+str(i+1))
        train_dataset = full_dataset.iloc[result[0]]
        test_dataset =  full_dataset.iloc[result[1]]
        
        processed_predictors_train=[]    
        processed_predictors_test=[] 
        sparse=False
        #horizontal contatenation of features (if there is a sparse feature then all features are tranformed to a sparse array)
        for predictor in predictors:
            if isinstance(train_dataset.loc[:,predictor].iloc[0],scipy.sparse.csr.csr_matrix):
                pred_list_train = train_dataset.loc[:,predictor].tolist()
                pred_list_test = test_dataset.loc[:,predictor].tolist()
                stacked_train = vstack(pred_list_train)
                processed_predictors_train.append(stacked_train)
                stacked_test = vstack(pred_list_test)
                processed_predictors_test.append(stacked_test)
                sparse=True
            else:
                pred_list_train =train_dataset.loc[:,predictor].as_matrix().reshape(-1, 1).astype(np.float64)
                processed_predictors_train.append(pred_list_train)
                pred_list_test =test_dataset.loc[:,predictor].as_matrix().reshape(-1, 1).astype(np.float64)
                processed_predictors_test.append(pred_list_test)
        if sparse:
            X = hstack(processed_predictors_train)
            Xtest = hstack(processed_predictors_test)
        else:
            X = np.concatenate(processed_predictors_train,axis=1)
            Xtest = np.concatenate(processed_predictors_test,axis=1)
            
        if mask is not None:
            X = X[:,mask]
            Xtest = Xtest[:,mask]

        y= train_dataset[ycol].as_matrix().reshape(-1, 1)
        ytest= test_dataset[ycol].as_matrix().reshape(-1, 1)
        

        if weights:
            model =regressor.fit(X,y.ravel(),sample_weight=train_dataset['weights'].as_matrix())
        else:
            model =regressor.fit(X,y.ravel())
        ypredtest = model.predict(Xtest)
        predictions.append(ypredtest)
        index.append(result[1])
    feature_number = X.shape[1]
    
    bow_predictions = np.concatenate(predictions,axis=0)
    index = np.concatenate(index,axis=0)
    
    sorted_index = np.argsort(index)
    sorted_bow_predictions  = bow_predictions[sorted_index]
    
    assert full_dataset.pm25.values.shape == bow_predictions.shape
    return (full_dataset,sorted_bow_predictions)


def create_second_step_bow_model(regressor,dataset,second_step_features,predictions,weights=False): 
    """ Creation of the second step model using second step feature training data and training predictions from cv bow model
    
    args:
    regressor --  an sklearn regressor (e.g GradientBoostingRegressor)
    dataset -- the training pandas dataframe
    second_step_feature -- the second step feature
    predictions -- the training predictions calculated from the cv bow model
    weights -- whether or not to weight each training sample based on distance of each city (used in cross city setups) (default: False)

    
    returns:
    a final regression model to perform testing 
    """ 
    sparse = False
    processed_features = []
    #horizontal contatenation of features (if there is a sparse feature then all features are tranformed to a sparse array)
    for feature in second_step_features:
        if isinstance(dataset.loc[:,feature].iloc[0],scipy.sparse.csr.csr_matrix):
            pred_list = dataset.loc[:,feature].tolist()
            stacked = vstack(pred_list)
            processed_features.append(stacked)
            sparse =True
        else:
            pred_list =dataset.loc[:,feature].as_matrix().reshape(-1, 1).astype(np.float64)
            processed_features.append(pred_list)
    if sparse:
        temp= hstack(processed_features)
        predictions = predictions.reshape(-1,1)
        X = hstack([temp,predictions])
    else:
        temp = np.concatenate(processed_features,axis=1)
        predictions = predictions.reshape(-1,1)
        X = np.concatenate([temp,predictions],axis=1)
    

    if weights:
        model = regressor.fit(X,dataset.pm25.values.ravel(),sample_weight=dataset['weights'].as_matrix())
    else:
        model = regressor.fit(X,dataset.pm25.values.ravel())
    return model


def get_inverse_distance_weights(distances):
    """"Calculate Inverse Distance Weighting matrix for each city of a country
    
    args:
    distances -- list of lists that contains the distance of a city with respect to others in a country
    
    returns:
    Inverse Distance Weighting matrix 
    """
    
    dist = np.array(distances)
    dist = dist.astype(np.float32)
    dist2 = dist.sum(axis=1).reshape(5,1) - dist
    d = dist2 / dist2.sum(axis=1).reshape(5,1)
    return d

def get_universal_inverse_distance_weights(cities,distances):
    """"Calculate Inverse Distance Weighting matrix for all cities
    args:
    cities -- list of cities
    distances -- list of lists that contains the distance of a city with respect to others in a country
    returns:
    Inverse Distance Weighting dataframe for each city in every country
    """
    weights = [ get_inverse_distance_weights(w) for w in distances]   
    weights_frame = pd.DataFrame(data = np.concatenate(weights),index=cities)#concat matricies to a dataframe
    return weights_frame

def compute_regression_results(datasets,cities_dict,city,window,setup,baseline,fs_method,fs_feature_num,features,feature_types,feature_details,representation,regressor,regressor_name,weights=None):
    """ Compute regression results
    
    args:
    datasets -- dict containing pandas dataframes from each city and window
    cities_dict -- a dict which as the country code as key (e.g UK) and the list of cities in this country as a value
    city -- the string name of the city
    window -- the number of  aggregated timesteps(hours) (valid values: 6,12,24)
    setup -- the regression setup (valid values: ('cross city (i.e. all to one)','within city (i.e. same city)'))
    baseline -- string to indicate whether this experiment is baseline or not (used only in results)
    fs_method -- the feature selection method ('Conly':features with highier correlation with PM2.5 in all cities.(used in paper),
'                                              'Sonly'':features with lowest correlation variance with PM2.5 in all cities,
                                               'S&C':combination of previous methods,
                                               'None':No feature selection)
    fs_feature_num -- number of best features to keep after performing feature selection  or 'None'
    features -- list of features in one step regression (e.g ['#aqs','bow_10k_unigrams_normalized']) or list of lists of features for two step regression (e.g [[bow_10k_unigrams_normalized'],['nearby_ground_truth_pm25']])
    feature_types -- description of the feature (used only in results)
    feature_details -- details of the feature (used only in results)
    representation -- the representation of the bow feature
    regressors -- an sklearn regressor for one step regression or a list of two sklearn regressors in two step regression setup
    regressor_name -- the regressor name for one step regression or a list of two regression names in two step regression setup
    weights -- the inverse distance weight matrix from all cities (used only in cross city setups) in order to weight each training sample when training the model or None
    """
    
    print(city+' -> '+ str(window)+' in '+setup+' setup with features-> '+str(features) )
    for code in cities_dict:
        if city in cities_dict[code]:
            country_code = code
            country = cities_dict[code]
      
    weights_flag = False
    if weights is not None:
        weights_flag = True
    
    if setup == 'within city (i.e. same city)':
        dataset = datasets[city+'_'+str(window)]
        train,test = split_dataset_even_odd_months(dataset)
        if weights is not None:
            raise Exception('Need to use cross city setup when using weights')
    elif setup == 'cross city (i.e. all to one)':
        train,test = create_all_vs_one_datasets(datasets,cities_dict,city,window,weights=weights) 
    
    #check if it is a 2 step regression
    if isinstance(features[0],list):
        two_step = True
        first_feature = features[0]#feature selection only for the first feature#(currently implemented  to work only for one bag of words feature)
    else:
        two_step = False
        first_feature = features
              
    if fs_method != 'NULL':
        if fs_feature_num != 'NULL':
            if len(first_feature) > 1:
                raise Exception('You have to use only one bow feature for feature selection') #(currently implemented  to work only for one bag of words feature)
            _,mask = feature_selection(datasets,country,first_feature[0],'pm25',window,method=fs_method)
            mask = mask[:fs_feature_num]#get the fs_feature_num top features
    else:
        mask = None
        
    if two_step:
        #calculate the training predictions using KFold cross validation to train the second step regression model
        train_dataset,bow_predictions = create_cv_bow_model(regressor[0],
                           features[0],'pm25',train,mask=mask,keep=features[0]+features[1],weights=weights_flag,cv=3)
        #train and test a regressor with first step features
        _,test_predictions =first_step_regression(train,test,regressor[0],features[0],mask=mask
                                                ,keep =features[0]+features[1],weights=weights_flag) 
        
        #train second step regressor with second step features and bow prediction
        model = create_second_step_bow_model(regressor[1],train_dataset,features[1],bow_predictions,weights=weights_flag)
        #use above model to test second step features with test predictions from first step features
        rmse_res,mae_res,test_prediction,precision,recall,fscore = testing(model,features[1],'pm25',test,mask=None,additional_features= test_predictions,
                        keep=features[0]+features[1],classification=True)
        return[country_code,city,window,setup,baseline,weights_flag,fs_method,fs_feature_num,
               feature_types,feature_details,representation,regressor_name[0],regressor_name[1],rmse_res,mae_res,precision[1],recall[1],fscore[1]]

    else:
        #training
        model,train_prediction = training(regressor,features,'pm25',train,mask=mask,weights=weights_flag) #add mask for feature selecation
        #testing
        rmse_res,mae_res,test_prediction,precision,recall,fscore = testing(model,features,'pm25',test,mask=mask,classification=True,verbose=False)
        return[country_code,city,window,setup,baseline,weights_flag,fs_method,fs_feature_num,
               feature_types,feature_details,representation,regressor_name,'NULL',rmse_res,mae_res,precision[1],recall[1],fscore[1]]
    
    
def aggregated_regression_experiments(datasets,cities_dict,cities,windows,setup,baseline,fs_methods,fs_feature_nums,features,feature_types,feature_details,representations,
                                      regressors,regressor_names,weights=None):
    """Run multiple regression experiments
    
    args:
    datasets -- dict containing pandas dataframes from each city and window
    cities_dict -- a dict which as the country code as key (e.g UK) and the list of cities in this country as a value
    cities -- list of cities
    windows -- list of lists of numbers of  aggregated timesteps(hours) (valid values: 6,12,24)
    setup -- the regression setup (valid values: ('cross city (i.e. all to one)','within city (i.e. same city)'))
    baseline -- string to indicate whether this experiment is baseline or not (used only in results)
    fs_methods -- lists of feature selection methods ('Conly':features with highier correlation with PM2.5 in all cities.(used in paper)
                                               'Sonly'':features with lowest correlation variance with PM2.5 in all cities
                                               'S&C':combination of previous methods
                                               'None':No feature selection)
                or 'None'
    fs_feature_nums -- list of numbers of best features to keep after performing feature selection or 'None'
    features -- list of lists of features in one step regression (e.g [['#aqs','bow_10k_unigrams_normalized'],['#aqs']]) 
                or lists of lists of lists of features for two step regression (e.g [[[bow_10k_unigrams_normalized'],['nearby_ground_truth_pm25']],[[#tw'],['nearby_ground_truth_pm25']]])
    feature_types -- lists of descriptions of the feature (used only in results)
    feature_details -- lists of details of the feature (used only in results)
    representations -- lists the representation of the bow feature
    regressors -- list of lists of sklearn regressors for one step regression or a list of lists of lists of sklearn regressors in two step regression setup
    regressor_name -- list of lists of regressor names for one step regression or a list of lists of lists of regression names in two step regression setup
    weights -- the inverse distance weight matrix from all cities (used only in cross city setups) in order to weight each training sample when training the model or None
    """
    
    results = []
    for i,feature in enumerate(features):
        for city in cities:
            for window in windows[i]:
                if fs_methods !='NULL': # feature selecion
                    for fs_method in fs_methods:
                        for fs_feature_num in fs_feature_nums:
                            if isinstance(regressors[i][0],list):#if it is a two step regression get the r-th regressor each of the two nested lists (same with regressor names)
                                for r,regressor in enumerate(regressors[i][0]):
                                    regr = [regressors[i][0][r],regressors[i][1][r]]
                                    regr_names = [regressor_names[i][0][r],regressor_names[i][1][r]]
                                    results.append(compute_regression_results(datasets,cities_dict,city,window,setup,baseline,fs_method,fs_feature_num,feature,
                                       feature_types[i],feature_details[i],representations[i],regr,regr_names,weights=weights))
                            else:#if it is not a two step regression just test every regressor in the list with his corresponding name
                                for r,regressor in enumerate(regressors[i]):
                                    regr = regressor
                                    regr_names = regressor_names[i][r]
                                    results.append(compute_regression_results(datasets,cities_dict,city,window,setup,baseline,fs_method,fs_feature_num,feature,
                                       feature_types[i],feature_details[i],representations[i],regr,regr_names,weights=weights))
                else:# no feature selection
                    if isinstance(regressors[i][0],list):#if it is a two step regression get the r-th regressor each of the two nested lists (same with regressor names)
                        for r,regressor in enumerate(regressors[i][0]):
                            regr = [regressors[i][0][r],regressors[i][1][r]]
                            regr_names = [regressor_names[i][0][r],regressor_names[i][1][r]]
                            results.append(compute_regression_results(datasets,cities_dict,city,window,setup,baseline,'NULL','NULL',feature,
                                       feature_types[i],feature_details[i],representations[i],regr,regr_names,weights=weights))
                    else:#if it is not a two step regression just test every regressor in the list with his corresponding name
                        for r,regressor in enumerate(regressors[i]):
                            regr = regressor
                            regr_names = regressor_names[i][r]
                            results.append(compute_regression_results(datasets,cities_dict,city,window,setup,baseline,'NULL','NULL',feature,
                               feature_types[i],feature_details[i],representations[i],regr,regr_names,weights=weights))
    return results
