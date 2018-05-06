# twitter-aq
Dataset and code to reproduce results of IVMSP 2018 paper: "Twitter-based Sensing of City-level Air Quality"

## Python version
The project is compatible with Python `3.5`

## How to run experiments

* Recreate pandas datasets from `.csv` files using `create_dataset_from_csv(path,window,features)`:

Argument | Description | Valid values
--- | ----- | ---
`path` | path to a `.csv` file in the `datasets/` dir | a string path
`window` | the temporal bin size (in hours) | 6, 12, 24
`features` | a list with all bag-of-word features | currently supporting only 'bow_10k_unigrams'

*e.g:* `London_dataset = create_dataset_from_csv(path_to_London_dataset,6,['bow_10k_unigrams'])`

`London_dataset` is a pandas dataframe that contains Twitter features (`#aqs`,`#high`,`#tw`), bag-of-word features (`bow_10k_unigrams`) in sparse representation and lagged versions of them, ground truth PM2.5 values (`pm25`) and inverse dinstance weighted (IDW) estimates based on ground truth PM2.5 values of nearby cities ('nearby_ground_truth_pm25') 

* Compute regression results using `compute_regression_results(datasets,cities_dict,city,window,setup,baseline,fs_method,fs_feature_num,features,regressor,weights=None)`

Arguments | Description | Valid values
--- | --- | ---
`datasets` | dict with city names and window as key and respective datasets as values | `datasets['London_6'] = London_dataset`
`cities_dict` | dict with available countries and respective cities | cities_dict['UK'] = `['London','Liverpool', ...]`
`city` | the city to make air quality predictions | a valid city name string
`window` | the temporal bin size (in hours) | 6, 12, 24
`setup` | whether to perform within city predictions (using odd months for training and even months for testing) or to perform cross-city predictions (by using `city` dataset for testing and all other datasets for training) | 'cross city' (i.e. all to one),'within city' (i.e. same city)
`baseline` | string to indicate whether this experiment is baseline, by defining the prediction metric, or not | 'idw','mean','NULL'
`fs_method` | the feature selection method | 'Conly':features with highiest correlation with PM2.5 (*used in paper*),<br />'Sonly'':features with lowest correlation variance with PM2.5,<br />'S&C':combination of previous methods,<br />'None':No feature selection
`fs_feature_num` | number of best features to keep after performing feature selection | *e.g* 100,500,'None'
`features` | features to use for training the regression model. if a single list is supplied, one regression model is built on a concatenated feature vector. if multiple lists are supplied, the outputs of the individual regression models are used as inputs in a second-stage regression model | one step regression (*e.g* ['#aqs','bow_10k_unigrams'])<br /> two step regression (*e.g* [[bow_10k_unigrams'],['nearby_ground_truth_pm25']])
`regressor` | the type of regressor  | an sklearn regressor for one step regression setup<br />a list of two sklearn regressors for two step regression setup
`weights` | the inverse distance weight matrix of all cities (used only in cross-city setups) in order to weight each training example | 

`compute_regression_results(arguments)` *returns a list which is described below*

Position | Description 
--- | --- 
0 | country code ('UK','US)
1 | city
2 | window
3 | the setup 
4 | baseline 
5 | inverse distance weight boolean flag
6 | the feature selection method
7 | number of best features to keep after performing feature selection
8 | feature_types
9 | feature_details 
10 | representation 
11 | one step regressor name
12 | two step regressor name
13 | Root Mean Square Error between ground truth and testing predictions
14 | Mean Absolute  Error between ground truth and testing predictions
15 | Precision in high pollution class after transformed regression to classification
16 | Recall in high pollution class after transformed regression to classification
17 | F-score in high pollution class after transformed regression to classification

* Compute regression results iteratively using the function: `aggregated_regression_experiments(datasets,cities_dict,cities,windows,setup,baseline,fs_methods,fs_feature_nums,features,regressors,weights=None)`

for more examples see `experiments.py`

## TODOs
* Test on additional cities
* Test additional textual representations (e.g. n-grams, word-to-vec)

## License
This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details


## Citation
If you find these dataset and code useful in your research please cite:
Charitidis, P., Spyromitros-Xioufis, E., Papadopoulos, S., & Kompatsiaris, Y. (2018). Twitter-based Sensing of City-level Air Quality. In Image, Video, and Multidimensional Signal Processing Workshop (IVMSP), 2018 IEEE 13th. IEEE.
