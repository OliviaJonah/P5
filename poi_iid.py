# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 16:42:50 2017

@author: Victor
"""
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import Imputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pickle
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
import tester

features_list = ['poi',
                'salary',
                'bonus', 
                'long_term_incentive', 
                'deferred_income', 
                'deferral_payments',
                'loan_advances', 
                'other',
                'expenses', 
                'director_fees',
                'total_payments',
                'exercised_stock_options',
                'restricted_stock',
                'restricted_stock_deferred',
                'total_stock_value',
                'to_messages',
                'from_messages',
                'from_this_person_to_poi',
                'from_poi_to_this_person']

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

# Transform data from dictionary to the Pandas DataFrame
df = pd.DataFrame.from_dict(data_dict, orient = 'index')
#Order columns in DataFrame, exclude email column
df = df[features_list]
df = df.replace('NaN', np.nan)
df.info()
#split of POI and non-POI in the dataset
poi_non_poi = df.poi.value_counts()
poi_non_poi.index=['non-POI', 'POI']
print "POI / non-POI split"
poi_non_poi
