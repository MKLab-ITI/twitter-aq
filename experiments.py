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
import pickle
import warnings
warnings.filterwarnings("ignore")

datasets_path = 'datasets/'
processed_datasets_path =  os.path.join(datasets_path,'processed_datasets')
zip_files = ['uk_cities.zip','us_cities.zip']
vocabularies_path = 'datasets/vocabularies/'
windows = [6,12,24]
english_cities = ['London','Manchester','Leeds','Liverpool','Birmingham']
american_cities = ["Boston","NewYork","Philadelphia","Baltimore","Pittsburgh"]
cities = english_cities + american_cities
cities_dict ={'UK':english_cities,'US':american_cities}
english_distances = [[163,178,169,101],[163,31,35,70],[178,31,64,78],[169,35,64,92],[101,70,78,92]]
american_distances = [[306,436,579,775],[306,129,272,507],[436,129,144,414],[579,272,144,316],[775,507,414,316]]

def main():
    
    if os.path.exists(processed_datasets_path):
        print('loading processed datasets')
        #load datasets dict from file if it has already been created
        files = [f for f in os.listdir(processed_datasets_path) if os.path.isfile(os.path.join(processed_datasets_path, f))]
        datasets = {}
        for file in files:
            with open(os.path.join(processed_datasets_path,file), 'rb') as f:
                key = file.split('.')[0]
                datasets[key] =  pickle.load(f)
    else:
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
        print('saving datasets')
        os.makedirs(os.path.join(datasets_path,'processed_datasets/'))#create dir to save dataset files
        for key, value in datasets.items():
            with open(os.path.join(os.path.join(datasets_path,'processed_datasets/'),key+'.pkl'), 'wb') as f:
                pickle.dump(value, f, pickle.HIGHEST_PROTOCOL)
            

                
    #calculate inverse distance weight dataframe for all cities
    weights_frame  = get_universal_inverse_distance_weights(cities,[english_distances,american_distances])
    
    def print_menu():   
        print(30 * "-" , "MENU" , 30 * "-")
        print("0. Experiment 0 - baselines")
        print("1. Experiment 1 - within city")
        print("2. Experiment 2 - cross city")
        print("3. Experiment 3 - feature selection")
        print("4. Exit")
        print(67 * "-")
  
    loop=True      
  
    while loop:          
        print_menu()    
        choice = int(input("Enter your choice [1-5]: "))
        print(choice)
        if choice==0:     
            ############################################################### experiment 0 ( baselines )################################################################
            print('experiment 0 - baselines')
            results = []
            setup='within city'
            weights=None 
            fs_methods='NULL'
            fs_feature_nums='NULL'

            features= [
                ['NULL'],
            ]
            regressors = [
                     ['NULL']
            ]*len(features)

            win = [
                [6,12,24],
            ]*len(features)

            setup='within city'
            baseline='idw'
            results.append(aggregated_regression_experiments(datasets,cities_dict,cities,win,setup,baseline,fs_methods,fs_feature_nums,features,regressors,weights=weights))
            baseline='mean'
            results.append(aggregated_regression_experiments(datasets,cities_dict,cities,win,setup,baseline,fs_methods,fs_feature_nums,features,regressors,weights=weights))

            setup='cross city'
            baseline='idw'
            results.append(aggregated_regression_experiments(datasets,cities_dict,cities,win,setup,baseline,fs_methods,fs_feature_nums,features,regressors,weights=weights))
            baseline='mean'
            results.append(aggregated_regression_experiments(datasets,cities_dict,cities,win,setup,baseline,fs_methods,fs_feature_nums,features,regressors,weights=weights))

            save_results('experiment_0',results)

            ################################################################ end of experiment 0 ################################################################


        elif choice==1:
            ############################################################### experiment 1 ( within city experiments )################################################################
            print('experiment 1 - within city')
            results=[]
            #first classifier parameters
            paramsrgs = {'n_estimators': 200,'max_features':0.3,'learning_rate':0.01}
            #second step classifier parameters (if exists)
            paramsrgs_second = {'n_estimators': 200,'learning_rate':0.01}
            setup='within city'
            baseline= 'NULL'
            weights=None 
            fs_methods='NULL'
            fs_feature_nums='NULL'

            features= [
                ['bow_10k_unigrams_normalized'],
                ['bow_10k_unigrams_2_normalized'],
                ['bow_10k_unigrams_3_normalized'],

            ]

            regressors = [
                     [GradientBoostingRegressor(**paramsrgs)]
            ]*len(features)

            win = [
                [6,12,24],
            ]*len(features)

            results.append(aggregated_regression_experiments(datasets,cities_dict,cities,win,setup,baseline,fs_methods,fs_feature_nums,features,regressors,weights=weights))

            save_results('experiment_1',results)

            ################################################################ end of experiment 1 ################################################################
        elif choice==2:
            ################################################################ experiment 2 (cross city experiments) ################################################################

            print('experiment 2 - cross city')   
            paramsrgs = {'n_estimators': 200,'max_features':0.3,'learning_rate':0.01}
            #second step classifier parameters (if exists)
            paramsrgs_second = {'n_estimators': 200,'learning_rate':0.01}
            results=[]
            setup='cross city'
            weights=weights_frame
            baseline= 'NULL'
            fs_methods='NULL'
            fs_feature_nums='NULL'

            features= [
                #two step regressions
                [['bow_10k_unigrams_normalized'],['idw_pm25']],
                [['bow_10k_unigrams_2_normalized'],['idw_pm25']],
                [['bow_10k_unigrams_3_normalized'],['idw_pm25']],
            ]

            regressors = [
                     [[GradientBoostingRegressor(**paramsrgs)],[GradientBoostingRegressor(**paramsrgs_second)]]
            ]*len(features)

            win = [
                [6,12,24],
            ]*len(features)
            # without weighted samples
            results.append(aggregated_regression_experiments(datasets,cities_dict,cities,win,setup,baseline,fs_methods,fs_feature_nums,features,regressors,weights=None))
            # with weighted samples
            results.append(aggregated_regression_experiments(datasets,cities_dict,cities,win,setup,baseline,fs_methods,fs_feature_nums,features,regressors,weights=weights))

            save_results('experiment_2',results)
            ################################################################ end of experiment 2 ################################################################

        elif choice==3:
            ################################################################ experiment 3 ( cross city with feature selection) ################################################################

            print('experiment 3 - cross city + feature selection')
            results=[]
            baseline='NULL'
            paramsrgs = {'n_estimators': 200,'max_features':0.3,'learning_rate':0.01}
            paramsrgs_second = {'n_estimators': 200,'learning_rate':0.01}
            setup='cross city'
            weights=weights_frame
            fs_methods=['Conly']
            fs_feature_nums=[20,50,100,500,1000]

            features= [
                [['bow_10k_unigrams_normalized'],['idw_pm25']],
                [['bow_10k_unigrams_2_normalized'],['idw_pm25']],
                [['bow_10k_unigrams_3_normalized'],['idw_pm25']],
            ]

            regressors = [
                     [[GradientBoostingRegressor(**paramsrgs)],[GradientBoostingRegressor(**paramsrgs_second)]]
            ]*len(features)

            win = [
                [6,12,24],
            ]*len(features)

            # without weighted samples
            results.append(aggregated_regression_experiments(datasets,cities_dict,cities,win,setup,baseline,fs_methods,fs_feature_nums,features,regressors,weights=None))
            # with weighted samples
            results.append(aggregated_regression_experiments(datasets,cities_dict,cities,win,setup,baseline,fs_methods,fs_feature_nums,features,regressors,weights=weights))

            save_results('experiment_3',results)

            ################################################################ end of experiment 3 ################################################################
        elif choice==4:
            print("Exiting")
            ## You can add your code or functions here
            loop=False # This will make the while loop to end as not value of loop is set to False
        else:
            # Any integer inputs other than values 1-5 we print an error message
            input("Wrong option selection. Enter any key to try again..")
    

    
    
    
    

    
    
    

            

if __name__ == "__main__":
    main()
    

