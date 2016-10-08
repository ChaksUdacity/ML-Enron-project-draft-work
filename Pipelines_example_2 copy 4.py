#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

import pandas
 
from pandas import read_csv
from sklearn.cross_validation import KFold
from sklearn.cross_validation import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn import cross_validation
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','total_stock_value', 'exercised_stock_options','bonus' ] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
    
# Removing Obvious Outliners from Data_dict
    data_dict.pop('TOTAL',0)  
    data_dict.pop("THE TRAVEL AGENCY IN THE PARK", 0)

# DATA -- > Pandas DataFrame; 
import pandas as pd
df = pd.DataFrame.from_records(list(data_dict.values()))
employees = pd.Series(list(data_dict.keys())) 
df.set_index(employees, inplace=True) # Index is employees name
df_dict = df.to_dict('index')


### Task 2: Remove outliers
# Drop features that have more than 80 missing values 
df.drop(['loan_advances',
         'director_fees',
         'restricted_stock_deferred',
         'deferral_payments',
         'deferred_income',
         'long_term_incentive',
         'email_address'],axis=1,inplace=True)


# replace NaN by 0, and all should be float
df.replace(to_replace= 'NaN', value= 0,inplace=True)
df=df.astype(float)



outlier_people =['LAVORATO JOHN J','KAMINSKI WINCENTY J',
                 'FREVERT MARK A','WHITE JR THOMAS E'] 

# Drop those people, outliers
df.drop(['LAVORATO JOHN J','KAMINSKI WINCENTY J',
         'FREVERT MARK A','WHITE JR THOMAS E',
        'PAI LOU L','BHATNAGAR SANJAY'] ,axis=0, inplace=True)


# reshape the numpy arrays:
from_poi = df.from_poi_to_this_person.values.reshape(-1, 1)
to_poi =df.from_this_person_to_poi.values.reshape(-1, 1)
to_messages=df.to_messages.values.reshape(-1, 1)
from_messages=df.from_messages.values.reshape(-1, 1)

# Feature Scaling 
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

n_from_poi =scaler.fit_transform(from_poi)
n_to_poi=scaler.fit_transform(to_poi)

n_to_messages=scaler.fit_transform(to_messages)
n_from_messages=scaler.fit_transform(from_messages)

### Task 3: Create new feature(s)
# Creating new feature poi emails 
poi_emails = n_from_poi + n_to_poi

# Create lists from the results:
poi_emails = [n[0] for n in list(poi_emails)]

poi_emails = pd.Series(poi_emails)

# Creating new feature all emails
all_emails = n_to_messages + n_from_messages

# Create lists from the results:
all_emails = [n[0] for n in list(all_emails)]

all_emails = pd.Series(all_emails)

#Creating new feature the ratio poi emails in all emails
poi_email_ratio = list((poi_emails+1)/(all_emails+1))

# Add new feature poi email ratio to the dataset
df['poi_email_ratio'] = pd.Series(poi_email_ratio, index=df.index)


#  Peek at the Data before adding and deleting data
print " Top rows of data before adding and deleting data columns"
# head
print df.head()


df['new_poi'] = df['poi']

# for convenience, we drop the original feature
df.drop('poi', axis=1, inplace=True)



#  Peek at the Data after adding and deleting data
print " Top rows of data after before adding and deleting data columns"
# head
print df.head()
#print df.info

# Transform that dataframe into a dictionary so that you can use it for the rest of your code:


### features_list is a list of strings, each of which is a feature name
### first feature must be "poi", as this will be singled out as the label

new_features_list = df.columns.values
# add a print statement to see the features list
# check that `'poi'` is the first item in that list (for later):
print new_features_list

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
# The dictionary, derived from that dataframe is given by:

df_dict = df.to_dict('index')

# The above two lines, after we have finished with the dataframe, give a dictionary that is compatible with the rest of the code in 'poi_id.py'.

#That is, we use that dictionary, 'df_dict', instead of reloading the original dictionary

# my_dataset will be the dictionary generated from pandas:
my_dataset = df_dict

### Extract features and labels from dataset for local testing
#these two lines extract the features specified in features_list
# and extract them from data_dict, returning a numpy array
data = featureFormat(df_dict, new_features_list)
print "new_features_list:\n", new_features_list

# if we are creating new features, could also do that here
labels, features = targetFeatureSplit(data)

#Y = labels
#X = features

array = df.values
X = array[:,0:14]
Y = array[:,14]
#validation_size = 0.20
#seed = 7

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html
from sklearn import preprocessing
# create feature union

#Algorithm Chains and Pipelines
print "Algorithm Chains and Pipelines"
from sklearn.svm import SVC
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# load and split the data
my_dataset = df_dict
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, random_state=0)


from sklearn.model_selection import GridSearchCV

#Building Pipelines
print "Building Pipelines"

from sklearn.pipeline import Pipeline
pipe = Pipeline([("scaler", MinMaxScaler()), ("svm", SVC())])
#pipe.fit(X_train, y_train)

#print("Test score: {:.2f}".format(pipe.score(X_test, y_test)))

#Using Pipelines in Grid-searches
print "Using Pipelines in Grid-searches"
param_grid = {'svm__C': [0.001, 0.01, 0.1, 1, 10, 100],
              'svm__gamma': [0.001, 0.01, 0.1, 1, 10, 100]}


clf = GridSearchCV(pipe, param_grid=param_grid, cv=5)
clf.fit(X_train, y_train)
print("Best cross-validation accuracy: {:.2f}".format(clf.best_score_))
print("Test set score: {:.2f}".format(clf.score(X_test, y_test)))
print("Best parameters: {}".format(clf.best_params_))

print "``````````````````````````````````````````````````````````````````"


from sklearn.linear_model import LogisticRegression

pipe = Pipeline([("scaler", MinMaxScaler()), (LogisticRegression())])
param_grid = {'logisticregression__C': [0.01, 0.1, 1, 10, 100]}

# load and split the data
my_dataset = df_dict
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, random_state=0)

clf = GridSearchCV(pipe, param_grid=param_grid, cv=5)
print clf.fit(X_train, y_train)
