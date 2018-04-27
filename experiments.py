import sys
import os
sys.path.insert(0, 'twitter-aq/')
from utils import *
from regression import *
from feature_selection import *
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import normalize
from sklearn.linear_model import LassoCV
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import ElasticNetCV
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.svm import LinearSVR

datasets_path = 'datasets/'
zip_files = ['uk_cities.zip','us_cities.zip']
vocabularies_path = 'datasets/vocabularies/'
windows = [6,12,24]
cities = ['London','Manchester','Leeds','Liverpool','Birmingham','Boston','NewYork','Philadelphia','Baltimore','Pittsburgh']
english_cities = ['London','Manchester','Leeds','Liverpool','Birmingham']
american_cities = ["Boston","NewYork","Philadelphia","Baltimore","Pittsburgh"]
cities_dict ={'UK':english_cities,'US':american_cities}
english_distances = [[163,178,169,101],[163,31,35,70],[178,31,64,78],[169,35,64,92],[101,70,78,92]]
american_distances = [[306,436,579,775],[306,129,272,507],[436,129,144,414],[579,272,144,316],[775,507,414,316]]

def main():
    
    #unzip files
    for file in zip_files:
        unzip_file(os.path.join(datasets_path,file),datasets_path)
    english_datasets_path = os.path.join(datasets_path,zip_files[0].split('.')[0])
    american_datasets_path = os.path.join(datasets_path,zip_files[1].split('.')[0])
    
    datasets = {}
    #reconstructing datasets from files
    print('--english cities--')
    feature_names = ['bow_10k_unigrams']
    for city in english_cities:
        print(city)
        for window in windows:
            dataset = create_dataset_from_csv(os.path.join(english_datasets_path,city+'_'+str(window)+'.csv'),window,feature_names)
            datasets[city+'_'+str(window)] = dataset

    print('--american cities--')
    for city in american_cities:
        print(city)
        for window in windows:
            dataset = create_dataset_from_csv(os.path.join(american_datasets_path,city+'_'+str(window)+'.csv'),window,feature_names)
            datasets[city+'_'+str(window)] = dataset
            
    #create normalized columns
    for city in cities:
        #print(city)
        for window in windows:
            for feature in [x for x in  datasets[city+'_'+str(window)].columns if 'bow_10k_unigrams' in x and 'normalized' not in x]:
                normalize_bow_feature(datasets[city+'_'+str(window)],feature,axis = 1,suffix = '_normalized')
                
    #calculate inverse distance weight dataframe for all cities
    weights_frame  = get_universal_inverse_distance_weights(cities,[english_distances,american_distances])
    
    
    print('experiment 1 - within city')
    
    paramsrgs = {'n_estimators': 200,'max_features':0.3,'learning_rate':0.01}
    paramsrgs_second = {'n_estimators': 200,'learning_rate':0.01}
    setup='within city (i.e. same city)'
    baseline='no'
    weights=None 
    fs_methods='NULL'
    fs_feature_nums='NULL'

    features= [
        ['bow_10k_unigrams'],
        ['bow_10k_unigrams_2'],
        ['bow_10k_unigrams_3'],
        ['bow_10k_unigrams_4'],
        ['bow_10k_unigrams_normalized'],
        ['bow_10k_unigrams_2_normalized'],
        ['bow_10k_unigrams_3_normalized'],
        ['bow_10k_unigrams_4_normalized'],

    ]

    feature_types = [
                'BOW','BOW','BOW','BOW','BOW','BOW','BOW','BOW'
                ]

    feature_details = [
                 'BOW10k_unigrams', 'BOW10k_unigrams_lag2', 'BOW10k_unigrams_lag3', 'BOW10k_unigrams_lag4', 'BOW10k_unigrams_normalized',
                    'BOW10k_unigrams_lag2_normalized', 'BOW10k_unigrams_lag3_normalized', 'BOW10k_unigrams_lag4_normalized'
                ]
    representations = [
                'uni_tf','uni_tf','uni_tf','uni_tf','uni_tf','uni_tf','uni_tf','uni_tf',
    ]

    regressors = [
             [GradientBoostingRegressor(**paramsrgs)]
    ]*len(features)

    regressor_names = [ 
        ['GBR_0.3_200'], 
    ]*len(features)

    win = [
        [6,12,24],
    ]*len(features)

    results1 = aggregated_regression_experiments(datasets,cities_dict,cities,win,setup,baseline,fs_methods,fs_feature_nums,features,feature_types,feature_details,representations,
                                          regressors,regressor_names,weights=weights)
    
    print('experiment 2 - cross city')    
    setup='cross city (i.e. all to one)'
    weights=weights_frame
    fs_methods='NULL'
    fs_feature_nums='NULL'

    features= [
        #two step regressions
        [['bow_10k_unigrams_normalized'],['nearby_ground_truth_pm25']],
        [['bow_10k_unigrams_2_normalized'],['nearby_ground_truth_pm25']],
        [['bow_10k_unigrams_3_normalized'],['nearby_ground_truth_pm25']],
        [['bow_10k_unigrams_4_normalized'],['nearby_ground_truth_pm25']],
    ]

    feature_types = [
                'BOW+nearby'
                ]*len(features)

    feature_details = [
                 'BOW10k_unigrams_normalized+nearby_2step',
                    'BOW10k_unigrams_lag2_normalized+nearby_2step', 'BOW10k_unigrams_lag3_normalized+nearby_2step', 'BOW10k_unigrams_lag4_normalized+nearby_2step'
                ]
    representations = [
                'uni_tf'
    ]*len(features)

    regressors = [
             [[GradientBoostingRegressor(**paramsrgs)],[GradientBoostingRegressor(**paramsrgs_second)]]
    ]*len(features)

    regressor_names = [ 
        [['GBR_0.3_200'],['GBR_200']]
    ]*len(features)

    win = [
        [6,12,24],
    ]*len(features)
    # without weighted samples
    results2 = aggregated_regression_experiments(datasets,cities_dict,cities,win,setup,baseline,fs_methods,fs_feature_nums,features,feature_types,feature_details,representations,
                                          regressors,regressor_names,weights=None)
    # with weighted samples
    results3 = aggregated_regression_experiments(datasets,cities_dict,cities,win,setup,baseline,fs_methods,fs_feature_nums,features,feature_types,feature_details,representations,
                                          regressors,regressor_names,weights=weights)
    
    print('experiment 3 - cross city + feature selection')
    baseline='no'
    paramsrgs = {'n_estimators': 200,'max_features':0.3,'learning_rate':0.01}
    paramsrgs_second = {'n_estimators': 200,'learning_rate':0.01}
    setup='cross city (i.e. all to one)'
    weights=weights_frame
    fs_methods=['Conly']
    fs_feature_nums=[20,50,100,500,1000]

    features= [
        [['bow_10k_unigrams_normalized'],['nearby_ground_truth_pm25']],
        [['bow_10k_unigrams_2_normalized'],['nearby_ground_truth_pm25']],
        [['bow_10k_unigrams_3_normalized'],['nearby_ground_truth_pm25']],
        [['bow_10k_unigrams_4_normalized'],['nearby_ground_truth_pm25']],
    ]

    feature_types = [
                'BOW+nearby'
                ]*len(features)

    feature_details = [
                 'BOW10k_unigrams_normalized+nearby_2step',
                    'BOW10k_unigrams_lag2_normalized+nearby_2step', 'BOW10k_unigrams_lag3_normalized+nearby_2step', 'BOW10k_unigrams_lag4_normalized+nearby_2step'
                ]
    representations = [
                'uni_tf'
    ]*len(features)

    regressors = [
             [[GradientBoostingRegressor(**paramsrgs)],[GradientBoostingRegressor(**paramsrgs_second)]]
    ]*len(features)

    regressor_names = [ 
        [['GBR_0.3_200'],['GBR_200']]
    ]*len(features)

    win = [
        [6,12,24],
    ]*len(features)

    # without weighted samples
    results4 = aggregated_regression_experiments(datasets,cities_dict,cities,win,setup,baseline,fs_methods,fs_feature_nums,features,feature_types,feature_details,representations,
                                          regressors,regressor_names,weights=None)
    # with weighted samples
    results5=aggregated_regression_experiments(datasets,cities_dict,cities,win,setup,baseline,fs_methods,fs_feature_nums,features,feature_types,feature_details,representations,
                                          regressors,regressor_names,weights=weights)
        
    #save results
    columns = ['country','city','window','setup','baseline','sample_weights','fs_method','feature_number',
                    'feature_type','feature_details','representation','1_step_regressor',
     '2_step_regressor','rmse','mae','precision(high)','recall(high)','f_measure(high)']

    results=results1+results2+results3+results4+results5
    data = np.concatenate(results,axis=0).reshape(-1,len(columns))
    print(data.shape)
    dataframe = pd.DataFrame(data =data,columns=columns)
    dataframe[['rmse','mae','precision(high)','recall(high)','f_measure(high)']] = dataframe[['rmse','mae','precision(high)','recall(high)','f_measure(high)']].astype(np.float64)
    results_path = 'results/
    if not os.path.exists(results_path):
        os.mkdir(results_path)
    dataframe.to_csv(os.path.join(results_path,'experiments.csv'),index=False)          

if __name__ == "__main__":
    main()
    

