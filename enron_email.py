
# coding: utf-8

# **Project Overview** 
# 
# Enron Corporation was an American energy, commodities, and services company based in Houston, Texas. It was founded in 1985 as the result of a merger between Houston Natural Gas and InterNorth, both relatively small regional companies in the U.S. Before its bankruptcy on December 2, 2001, Enron employed approximately 20,000 staff and was one of the world's major electricity, natural gas, communications and pulp and paper companies, with claimed revenues of nearly $101 billion during 2000. Fortune named Enron "America's Most Innovative Company" for six consecutive years. At the end of 2001, it was revealed that its reported financial condition was sustained by institutionalized, systematic, and creatively planned accounting fraud, known since as the Enron scandal. Enron has since become a well-known example of willful corporate fraud and corruption. The scandal also brought into question the accounting practices and activities of many corporations in the United States and was a factor in the enactment of the Sarbanes–Oxley Act of 2002. The scandal also affected the greater business world by causing the dissolution of the Arthur Andersen accounting firm. It had collapsed into bankruptcy due to widespread corporate fraud. In the resulting Federal investigation, there was a significant amount of typically confidential information entered into public record, including tens of thousands of emails and detailed financial data for top executives. Enron filed for bankruptcy in the Southern District of New York in late 2001 and selected Weil, Gotshal & Manges as its bankruptcy counsel. It ended its bankruptcy during November 2004, pursuant to a court-approved plan of reorganization, after one of the most complex bankruptcy cases in U.S. history. A new board of directors changed the name of Enron to Enron Creditors Recovery Corp., and emphasized reorganizing and liquidating certain operations and assets of the pre-bankruptcy Enron. On September 7, 2006, Enron sold Prisma Energy International Inc., its last remaining business, to Ashmore Energy International Ltd. (now AEI) ref:https://en.wikipedia.org/wiki/Enron 
# 
# ** Goal of Project**
# 
# The goal is to identify the persons of interest (POI)in the scandal.POIs were ‘individuals who were indicted, reached a settlement, or plea deal with the government, or testified in exchange for prosecution immunity.Using several features in Machine Learning under the four division
# 
# 1. Enron dataset
# 
# 2. Feature processing
# 
# 3. Algorithm 
# 
# 4. Validation
# 
# These features are extracted from financial and email records. Having in mind that those POI will have different patterns from the others Non-POI. These differences should be reflected in financial data, communication patterns, etc. and we can train algorithms to exploit and expose these differences.
# 
# 1. Understanding the Dataset and Question 
# 
# Data exploration (learning, cleaning and preparing the data),
# feature selecting/engineering (selecting the features which influence mostly on the target, create new features (which explains the target the better than existing) 
# reducing the dimensionality of the data using principal component analysis (PCA)), picking/tuning one of the supervised machine learning algorithm and validating it to get the accurate person of interest identifier model.

# Data Exploration
# 
# The features in the data fall into three major types, namely financial features, email features and POI labels.
# There are 146 samples with 20 features and a binary classification ("poi"), 2774 data points.
# Among 146 samples, there are 18 POI and 128 non-POI.
# Among 2774, there are 1358 (48.96%) data points with NaN values.

# In[10]:

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


# In[11]:

#split of POI and non-POI in the dataset
poi_non_poi = df.poi.value_counts()
poi_non_poi.index=['non-POI', 'POI']
print "POI / non-POI split"
poi_non_poi


# Data Cleansing

# In[12]:

print "Amount of NaN values in the dataset: ", df.isnull().sum().sum()


# FindLaw describes NaN values as values of 0 but not the missing value. So NaNs will be replaced with 0.

# In[13]:

# Replacing 'NaN' in financial features with 0
df.ix[:,:15] = df.ix[:,:15].fillna(0)


# NaN values in email features means the information is missing. so spliting the data into 2 classes: POI/non-POI and impute the missing values with median of each class.

# In[14]:

email_features = ['to_messages', 'from_messages', 'from_this_person_to_poi', 'from_poi_to_this_person']

imp = Imputer(missing_values='NaN', strategy='median', axis=0)

#impute missing values of email features 
df.loc[df[df.poi == 1].index,email_features] = imp.fit_transform(df[email_features][df.poi == 1])
df.loc[df[df.poi == 0].index,email_features] = imp.fit_transform(df[email_features][df.poi == 0])


# Checking the accuracy of the financial data by summing up the payment features and comparing it with the total_payment feature and stock features and comparing with the total_stock_value.

# In[15]:

#check data: summing payments features and compare with total_payments
payments = ['salary',
            'bonus', 
            'long_term_incentive', 
            'deferred_income', 
            'deferral_payments',
            'loan_advances', 
            'other',
            'expenses', 
            'director_fees']
df[df[payments].sum(axis='columns') != df.total_payments]


# In[16]:

stock_value = ['exercised_stock_options',
                'restricted_stock',
                'restricted_stock_deferred']
df[df[stock_value].sum(axis='columns') != df.total_stock_value]


# There is an error in the data entry in the two samples. So  correctting them and checking that everything they are correct (empty DataFrames mean no samples with mistakes in the data set).

# In[17]:

df.ix['BELFER ROBERT','total_payments'] = 3285
df.ix['BELFER ROBERT','deferral_payments'] = 0
df.ix['BELFER ROBERT','restricted_stock'] = 44093
df.ix['BELFER ROBERT','restricted_stock_deferred'] = -44093
df.ix['BELFER ROBERT','total_stock_value'] = 0
df.ix['BELFER ROBERT','director_fees'] = 102500
df.ix['BELFER ROBERT','deferred_income'] = -102500
df.ix['BELFER ROBERT','exercised_stock_options'] = 0
df.ix['BELFER ROBERT','expenses'] = 3285
df.ix['BELFER ROBERT',]
df.ix['BHATNAGAR SANJAY','expenses'] = 137864
df.ix['BHATNAGAR SANJAY','total_payments'] = 137864
df.ix['BHATNAGAR SANJAY','exercised_stock_options'] = 1.54563e+07
df.ix['BHATNAGAR SANJAY','restricted_stock'] = 2.60449e+06
df.ix['BHATNAGAR SANJAY','restricted_stock_deferred'] = -2.60449e+06
df.ix['BHATNAGAR SANJAY','other'] = 0
df.ix['BHATNAGAR SANJAY','director_fees'] = 0
df.ix['BHATNAGAR SANJAY','total_stock_value'] = 1.54563e+07
df.ix['BHATNAGAR SANJAY']
df[df[payments].sum(axis='columns') != df.total_payments]


# In[18]:

df[df[stock_value].sum(axis='columns') != df.total_stock_value]


# Outlier Investigation
# 
# Descriptive statistics determins outliers of the distibution as the values which are higher than Q2 + 1.5IQR or less than Q2 - 1.5IQR, where Q2 median of the distribution, IQR - interquartile range.
# Here the sum of outlier variables for each person is calculated and sorted in descending.

# In[19]:

outliers = df.quantile(.5) + 1.5 * (df.quantile(.75)-df.quantile(.25))
pd.DataFrame((df[1:] > outliers[1:]).sum(axis = 1), columns = ['# of outliers']).    sort_values('# of outliers',  ascending = [0]).head(7)


# Since the data set is really small, a consideration to use 5% of the samples with most number of outlier variables is made:
# The first value is 'TOTAL' which is the total value of financial payments from the FindLaw data. Total should be excluded because it is not a POI.
# Kenneth Lay and Jeffrey Skilling are very well known persons from ENRON - they will be kept as they represent anomalies but not the outliers. 
# Mark Frevert and Lawrence Whalley are not so very well known but top managers of the Enron who also represent valuable examples for the model - they will be kept in the data set. 
# John Lavorato is not very well known person as far as I've searched in the internet. I don't think he represents a valid point and exclude him. 
# Jeffrey Mcmahon is the former treasurer who worked before guilty Ben Glisan. I would exclude him from the data set as he worked before the guilty treasurer and might add some confusion to the model. 
# Out of 7 persons 3 are excluded (1 typo 'TOTAL' and 2 persons).

# In[23]:

from sklearn.feature_selection import SelectKBest , f_classif 
from tester import test_classifier, dump_classifier_and_data 
scaler = StandardScaler()
df_norm = df[features_list]
df_norm = scaler.fit_transform(df_norm.ix[:,1:])

clf = GaussianNB()

features_list2 = ['poi']+range(8)

my_dataset = pd.DataFrame(SelectKBest(f_classif, k=8).fit_transform(df_norm, df.poi), index = df.index)
my_dataset.insert(0, "poi", df.poi)
my_dataset = my_dataset.to_dict(orient = 'index')  

dump_classifier_and_data(clf, my_dataset, features_list2)
tester.main()


# In[24]:

# exclude 3 outliers from the data set
df = df.drop(['TOTAL', 'LAVORATO JOHN J', 'MCMAHON JEFFREY'],0)


# **Optimize Feature Selection/Engineering**
# 
# Using different features and models to standardize features, apply principal component analysis and GaussianNB classifier, also to use decision tree classifier, incl. choosing the features with features importance attribute and tuning the model.
# Create new features
# In both strategies creating new features as a fraction of almost all financial variables (f.ex. fractional bonus as fraction of bonus to total_payments, etc.). Logic behind email feature creation was to check the fraction of emails, sent to POI, to all sent emails; emails, received from POI, to all received emails.
# resulting in using one new feature fraction_to_POI:

# In[25]:

#create additional feature: fraction of person's email to POI to all sent messages
df['fraction_to_poi'] = df['from_this_person_to_poi']/df['from_messages']
#clean all 'inf' values which we got if the person's from_messages = 0
df = df.replace('inf', 0)


# Decision tree doesn't require  any feature scaling 
# 
# Intelligently select features
# it is important to sort by null, so we can get all the non-null features
# 

# In[26]:

#Decision tree using features with non-null importance
clf = DecisionTreeClassifier(random_state = 75)
clf.fit(df.ix[:,1:], df.ix[:,:1])

# show the features with non null importance, sorted and create features_list of features for the model
features_importance = []
for i in range(len(clf.feature_importances_)):
    if clf.feature_importances_[i] > 0:
        features_importance.append([df.columns[i+1], clf.feature_importances_[i]])
features_importance.sort(key=lambda x: x[1], reverse = True)
for f_i in features_importance:
    print f_i
features_list = [x[0] for x in features_importance]
features_list.insert(0, 'poi')


# As seen above fraction_to_poi feature has the highest importance for the model. The number of features used for the model may cause different results. In the algorithm tuning step, features with non-null importance will be chosen so there will be a change in the number.
# Random state equal to 75 in decision tree and random forest to will be able to represent the results. The exact value was manually chosen for better performance of decision tree classifier.

# **Pick and Tune an Algorithm**
# 
# 3 machine learning algorithms have been used:
# 
# - Decision Tree Classifier
# 
# - Random Forest
# 
# - GaussianNB
# 
# For decision tree and random forest  features with non-null importance based on clf.features_importances__ is chosen.
# 
# In the next step the number of features is chnaged from 1 to all in order to achieve the best performance.
# For the GaussianNB classifier  a number of steps is applied to achieve the result:
# standardized features; 
# applied SelectKBest function from sklearn to find k best features for the algorithm (resulting in k = 8 which gave me better result for k in a range from 1 to all); 
# PCA is used to decrease the dimensionality of the data (resulting in n_components = 3). 
# Using Decision Tree Classifier showed the best result and was significantly faster than RandomForest whih can be easily tuned.
# Here are the following results from the algorithms before tuning (using tester.py, provided in advance):

# In[27]:

pd.DataFrame([[0.90880, 0.66255, 0.64400, 0.65314],
              [0.89780, 0.70322, 0.40400, 0.51318],
              [0.86447, 0.49065, 0.43300, 0.46003]],
             columns = ['Accuracy','Precision', 'Recall', 'F1'], 
             index = ['Decision Tree Classifier', 'Random Forest', 'Gaussian Naive Bayes'])


# In[28]:

#Decision Tree Classifier with standard parametres 
clf = DecisionTreeClassifier(random_state = 75)
my_dataset = df[features_list].to_dict(orient = 'index')
tester.dump_classifier_and_data(clf, my_dataset, features_list)
tester.main() 


# In[29]:

#Random Forest with standard parameters
clf = RandomForestClassifier(random_state = 75)
clf.fit(df.ix[:,1:], np.ravel(df.ix[:,:1]))

# selecting the features with non null importance, sorting and creating features_list for the model
features_importance = []
for i in range(len(clf.feature_importances_)):
    if clf.feature_importances_[i] > 0:
        features_importance.append([df.columns[i+1], clf.feature_importances_[i]])
features_importance.sort(key=lambda x: x[1], reverse = True)
features_list = [x[0] for x in features_importance]
features_list.insert(0, 'poi')

# number of features for best result was found iteratively
features_list2 = features_list[:11]
my_dataset = df[features_list2].to_dict(orient = 'index')
tester.dump_classifier_and_data(clf, my_dataset, features_list2)
tester.main()


# In[30]:

# GaussianNB with feature standartization, selection, PCA

clf = GaussianNB()

# data set standartization
scaler = StandardScaler()
df_norm = df[features_list]
df_norm = scaler.fit_transform(df_norm.ix[:,1:])

# feature selection
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
features_list2 = ['poi']+range(3)
my_dataset = pd.DataFrame(SelectKBest(f_classif, k=8).fit_transform(df_norm, df.poi), index = df.index)

#PCA
pca = PCA(n_components=3)
my_dataset2 = pd.DataFrame(pca.fit_transform(my_dataset),  index=df.index)
my_dataset2.insert(0, "poi", df.poi)
my_dataset2 = my_dataset2.to_dict(orient = 'index')  

dump_classifier_and_data(clf, my_dataset2, features_list2)
tester.main()


# **Tune the algorithm**
# 
# In Machine learning Bias-variance tradeoff is one of the key dilema ,high bias algorithms has no capacity to learn, high variance algorithms react poorly they have no histroy(something in memory). Predictive model is used to arrive at a compromise. The process of changing the parameteres of algorithms is algorithm tuning and it lets us find the golden mean and best result. 
# Algorithm might be tuned manually by iteratively changing the parameteres and tracking the results. Or GridSearchCV might be used which makes this automatically.
# Here,by tuning the parameteres of the decision tree classifier which is sequentially tuning parameter by parameter and got the best F1 using these parameters:
# 

# In[31]:

clf = DecisionTreeClassifier(criterion = 'entropy', 
                             min_samples_split = 19,
                             random_state = 75,
                             min_samples_leaf=6, 
                             max_depth = 3)


# In[32]:

clf.fit(df.ix[:,1:], df.poi)

# show the features with non null importance, sorted and create features_list of features for the model
features_importance = []
for i in range(len(clf.feature_importances_)):
    if clf.feature_importances_[i] > 0:
        features_importance.append([df.columns[i+1], clf.feature_importances_[i]])
features_importance.sort(key=lambda x: x[1], reverse = True)

features_list = [x[0] for x in features_importance]
features_list.insert(0, 'poi')

my_dataset = df[features_list].to_dict(orient = 'index')
tester.dump_classifier_and_data(clf, my_dataset, features_list)
tester.main() 


# **Validate and Evaluate**
# 
# 
# Usage of Evaluation Metrics
# 
# F1 score is the key to measure the acuracy of hte algorithims in this project. Both Precision and the recall of the test to compute the score.
# Precision is the ability of the classifier not label as positive sample that is negative.
# Recall is the ability of the classifier to find all positive samples.
# The F1 score can be interpreted as a weighted average of the precision and recall, where an F1 score reaches its best value at 1 and worst at 0.
# 
# 
# The tuned decision tree classifier showed precision 0.82238 and recall 0.65800 with the resulting F1 score 0.73499. Explained as 82.24% of the called POI are POI and 65.80% of POI are identified.
# 
# **Validation Strategy**
# 
# Validation is a process of evaluating the model performance. Classic mistake is to use small data set for the model training or validate model on the same data set as train it.
# There are a number of strategies to validate the model. One of them is to split the available data into train and test data another one is to perform a cross validation: process of splitting the data on k beans equal size; run learning experiments; repeat this operation number of times and take the average test result.
# 
# Algorithm Performance
#  The tester function provided is used ofr validation, it performs stratified shuffle split cross validation approach using StratifiedShuffleSplit function from sklearn.cross_validation library. The results are:

# In[33]:

pd.DataFrame([[0.93673, 0.83238, 0.65800, 0.73499]],
             columns = ['Accuracy','Precision', 'Recall', 'F1'], 
             index = ['Decision Tree Classifier'])


# **Reflection (Conclusions)**
# 
# Approching the project witht he idea that once the right alogorithim is chosen will make a good machine learning.I have realised that having a good(clean) data will help significantly on the algorithim tuning. Here most time was spent on outlier detection and preparing the data
# I am sure using Random Forest will improve the model significantly
# 
# Limitations of the study
# 
# Given that we only had about 145 person in the data, comes with some limitations, since we might skip some other POIs, but getting all the emails and financial inforation of everyone seems a daunting task.The missing email values were given as medians so the modes of the email distribution are switched to medians. The Algorithms were tuned sequentially (swtiching parameters  to achieve better performance. There is a chance that othere parameters in combination might give better model's accuracy).
# 
# References:
# Enron data set: https://www.cs.cmu.edu/~./enron/
# FindLaw financial data: http://www.findlaw.com
# Visualization of POI: http://www.nytimes.com/packages/html/national/20061023_ENRON_TABLE/index.html
# Enron on Wikipedia: https://en.wikipedia.org/wiki/Enron
# F1 score on Wikipedia: https://en.wikipedia.org/wiki/F1_score
# http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html
# Udacity Mentor
# 
# 

# In[ ]:



