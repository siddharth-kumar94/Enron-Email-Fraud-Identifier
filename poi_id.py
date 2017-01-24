# coding: utf-8

#!/usr/bin/python

import sys
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

features_list = [
    'poi', 'salary', 'deferral_payments',
    'total_payments', 'loan_advances', 'bonus',
    'restricted_stock_deferred', 'deferred_income',
    'total_stock_value', 'expenses', 'exercised_stock_options',
    'other', 'long_term_incentive', 'restricted_stock',
    'director_fees', 'to_messages',
    'from_poi_to_this_person', 'from_messages',
    'from_this_person_to_poi', 'shared_receipt_with_poi'
    ]

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

#Visually detect outliers
#Converts data_dict to Pandas DataFrame and transpose for easy manipulation
data_dict = pd.DataFrame(data_dict).transpose()
plt.scatter(data_dict['salary'], data_dict['bonus'])
plt.xlabel('Salary')
plt.ylabel('Bonus')
plt.show(block=False)

#Replaces all NaN and infinity values with 0
data_dict.replace('NaN', 0, inplace=True)

#Determines what the outlier is called
print 'outliers', data_dict[data_dict['salary'] > 2e7].index, '\n'

#Removes outlier 'TOTAL' from dataset
data_dict.drop('TOTAL', inplace=True)

#Checks to see if there are more outliers
plt.scatter(data_dict['salary'], data_dict['bonus'])
plt.xlabel('Salary')
plt.ylabel('Bonus')
plt.show(block=False)

### Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict
my_dataset['poi_inbox_percentage'] = my_dataset['from_poi_to_this_person'].divide(my_dataset['to_messages'], fill_value = 0)
my_dataset['poi_outbox_percentage'] = my_dataset['from_this_person_to_poi'].divide(my_dataset['from_messages'], fill_value = 0)
my_dataset['poi_shared_receipt_percentage'] = my_dataset['shared_receipt_with_poi'].divide(my_dataset['to_messages'], fill_value = 0)

#Replaces all NaN and infinity values with 0
my_dataset.replace([np.nan, np.inf], 0, inplace=True)

#Finds and prints all pairs of highly correlated features
def find_highly_correlated_features(dataset):
    pairwise_corr = dataset.corr()
    for j in range(0, len(pairwise_corr.columns)):
        for i in range(0, j):
            if abs(pairwise_corr.iloc[i, j]) > 0.75:
                print (pairwise_corr.columns[i], pairwise_corr.index[j]), ': ', pairwise_corr.iloc[i, j]

find_highly_correlated_features(my_dataset)
print '\n'

#Manually remove one feature from each pair that had highly correlated features
features_list.remove('other')
features_list.remove('to_messages')
features_list.remove('loan_advances')
features_list.remove('exercised_stock_options')
features_list.remove('restricted_stock')

#transpose dataset back to agree with featureFormat function
my_dataset = my_dataset.transpose()

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Try a varity of classifiers
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
import tester

#First classifier

#Estimators and params used for pipeline and Grid Search
estimators = []
params = dict()

#Feature Selector
#SelectKBest
estimators.append(('feature_selector', SelectKBest()))
params.update(dict(feature_selector__k=[10, 12, 'all']))

#Classifier
##Naive Bayes
estimators.append(('clf', GaussianNB()))

#Cross Validation
pipe = Pipeline(estimators).fit(features, labels)
grid_search = GridSearchCV(pipe, param_grid=params, scoring='f1')
grid_search.fit(features, labels)
    
clf = grid_search.best_estimator_
print 'Naive Bayes score: ', grid_search.best_score_, '\n'

#Second classifier

#Estimators and params used for pipeline and Grid Search
estimators = []
params = dict()

#Feature Selector
#SelectKBest
estimators.append(('feature_selector', SelectKBest()))
params.update(dict(feature_selector__k=[10, 12, 'all']))

#Classifier
##AdaBoost
estimators.append(('clf', AdaBoostClassifier()))
params.update({'clf__n_estimators': [50, 75, 100], 'clf__learning_rate': [.9, 1]})

#Cross Validation
pipe = Pipeline(estimators).fit(features, labels)
grid_search = GridSearchCV(pipe, param_grid=params, scoring='f1')
grid_search.fit(features, labels)
    
clf = grid_search.best_estimator_
print 'AdaBoost score: ', grid_search.best_score_, '\n'

#Finds and prints best features and corresponding scores
def get_best_features(clf):
    best_feature_indexes = clf.named_steps['feature_selector'].get_support()
    for i in range (0, len(best_feature_indexes)):
        if best_feature_indexes[i]:
            print features_list[i+1], pipe.named_steps['feature_selector'].scores_[i]

get_best_features(clf)
print '\n'

#Uses methods from tester.py to used Stratified Shuffle Split for final cross validation step
dump_classifier_and_data(clf, my_dataset, features_list)
tester.main()

