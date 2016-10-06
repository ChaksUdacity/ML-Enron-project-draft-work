#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")
import pandas


import pandas
import numpy as np
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import cross_validation
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import decomposition
from sklearn.decomposition import PCA

from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.preprocessing import scale
from sklearn.preprocessing import MinMaxScaler 

#DataFrame = pandas.read_pickle("../final_project/final_project_dataset.pkl")
#print DataFrame


# DATA : 'data_dict' is created automatically by the following lines:

# The file 'final_project_dataset.pkl' is in the 'final_project' folder of the download and 
#  -the file 'poi_id.py' is in the same folder. 
#   -So, as long as we run 'poi_id.py' from the folder that we found it in, 
#    -then 'data_dict' will be created for you by that script

# One we have 'data_dict' loaded, using the command in 'explore_enron_data.py', 
#  -we can change the format from a python dictionary to a pandas dataframe using 
#   -pandas commands specifically designed to do that.

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

#  
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

n_from_poi =scaler.fit_transform(from_poi)
n_to_poi=scaler.fit_transform(to_poi)

n_to_messages=scaler.fit_transform(to_messages)
n_from_messages=scaler.fit_transform(from_messages)

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

#  Peek at the Data
print " Top rows of data"
# head
print df.head()
#print df.info

#print df['poi_email_ratio']

# Dimensions of Dataset
print "Data shape" 
# shape
print(df.shape)
print "dataframe.dtype:\n", df.dtypes

print "poi", ['poi'>0]

# Statistical Summary
print "Statistical Summary"
# descriptions
print(df.describe())
print "***********************************************************"

# class distribution
print(df.groupby('poi').size())
print "***********************************************************"

print "Correlations Between Attributes"
"""
Correlation refers to the relationship between two variables and how they may or may not
 change together. The most common method for calculating correlation is Pearsons Correlation
  -Coefficient that assumes a normal distribution of the attributes involved.
   A correlation of -1 or 1 shows a full negative or positive correlation respectively. 
   Whereas a value of 0 shows no correlation at all. Some machine learning algorithms 
   like linear and logistic regression can suffer poor performance if there are highly 
   correlated attributes in your dataset. As such, it is a good idea to review all of
    the pair-wise correlations of the attributes in your dataset. You can use the corr() 
    function on the Pandas DataFrame to calculate a correlation matrix.
"""

# Pairwise Pearson correlations
import pandas

pandas.set_option('display.width', 100)
pandas.set_option('precision', 3)
correlations = df.corr(method='pearson')
print "Correlations Between Attributes:\n" , (correlations)
print "***********************************************************"

print "Skew of Univariate Distributions"
# Skew for each attribute
import pandas

skew = df.skew()
#print(skew)

print "***********************************************************"

print "Data Visualization"
print  "Univariate Plots:\n"
# Univariate plots to better understand each attribute.
# box and whisker plots
import matplotlib.pyplot as plt
df.plot(kind='box', subplots=True, layout=(5,5), sharex=False, sharey=False,title = "Univariate plots to better understand each attribute")
plt.show()

print "Histogram"
# histograms
import numpy as np
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt

df.hist()
plt.show()

# Multivariate Plots
# we can look at the interactions between the variables.

# scatter plot matrix
print "Scatterplot Matrix"
import matplotlib.pyplot as plt
import pandas
from pandas.tools.plotting import scatter_matrix
scatter_matrix(df)
plt.show()


# Correction Matrix Plot
import matplotlib.pyplot as plt
import pandas
import numpy
names = ['bonus', 'exercised_stock_options','expenses','from_poi_to_this_person','from_this_person_to_poi', 'other','poi','restricted_stock', 'salary','shared_receipt_with_poi','to_messages','total_payments','total_stock_value','poi_emails', 'all_emails','poi_email_ratio']                        
#names = ['bonus', 'exercised_stock_options','expenses','from_messages','from_poi_to_this_person','from_this_person_to_poi', 'other','poi','restricted_stock', 'salary','shared_receipt_with_poi','to_messages','total_payments','total_stock_value','poi_email_ratio']                        
#names = ['exercised_stock_options','from_messages','from_poi_to_this_person', 'other']                        

correlations = df.corr()
# plot correlation matrix
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = numpy.arange(0,15,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(names)
ax.set_yticklabels(names)
plt.show()

print "+++++++++++++++++++"
import matplotlib.pyplot as plt
import pandas
import numpy as np
df.poi.hist(bins=30)
plt.title('POI');
plt.xlabel('[poi]')
plt.show()

#"It is common to apply a log transform to this type of variable to reduce the skewness."
# take the log of bonus
# - this variable follows approximately a log-normal distribution
df['new_poi'] = df['poi']

# for convenience, we drop the original feature
df.drop('poi', axis=1, inplace=True)
print df.head()

print "+++++++++++++++++++"


print "Test options and evaluation metric "
# We will use 10-fold cross validation to estimate accuracy.
# This will split our dataset into 10 parts, train on 9 and test on 1 and repeat for all combinations of train-test splits.

print "Evaluate Some Algorithms"
print  "Create a Validation Dataset"
# Split-out validation dataset
array = df.values
X = array[:,0:14]
Y = array[:,14]
validation_size = 0.20
seed = 7


X_train, X_validation, Y_train, Y_validation = cross_validation.train_test_split(X, Y, test_size=validation_size, random_state=seed)

print " Principal Component Analysis (or PCA)"
# Feature Extraction with PCA
import numpy
from pandas import read_csv
from sklearn.decomposition import PCA
# load data
array = df.values
# feature extraction
pca = PCA(n_components=4)
#pca = PCA(n_components=4)
fit = pca.fit(X)
# summarize components
print("Explained Variance: %s") % fit.explained_variance_ratio_
print(fit.components_)

# Make a scatter plot of each, shaded according to cluster assignment.
#plt.scatter(x=fit[:,0], y=fit[:,1], c=labels)
# Show the plot.
#plt.show()

#print "first_pc :\n", pca.components_[0]
#print "second_pc :\n", pca.components_[1]
#print "third_pc :\n", pca.components_[2]
#print "fourth_pc :\n", pca.components_[3]
#print "fifth_pc :\n", pca.components_[4]

#X_train_pca = fit.transform(X_train)
#X_validation_pca = fit.transform(X_validation)

# helper functions for the demonstration

#The amount of variance that each PC explains
var= pca.explained_variance_ratio_
print pca.components_

print "pca ratio,\n", pca.explained_variance_ratio_/np.linalg.norm(pca.explained_variance_ratio_)

#Cumulative Variance explains
var1=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)
print "Cumulative Variance explains", var1
plt.plot(var1)
plt.show()

#Looking at above plot I'm taking 30 variables
#pca = PCA(n_components=10)
#pca.fit(X)
X=pca.fit_transform(X)
#print X1

#transformed = pca.transform(X)
#print transformed

#print utils.components_table(pca, dataset)
#utils.biplot(dataset, transformed, pca)




print "Test options and evaluation metric "
# We will use 10-fold cross validation to estimate accuracy.
# This will split our dataset into 10 parts, train on 9 and test on 1 and repeat for all combinations of train-test splits.num_folds = 10

num_folds = 10
num_instances = len(X_train)
seed = 7
scoring = 'accuracy'

print "Build Models"

"""
Lets evaluate 6 different algorithms
Logistic Regression (LR)
Linear Discriminant Analysis (LDA)
K-Nearest Neighbors (KNN).
Classification and Regression Trees (CART).
Gaussian Naive Bayes (NB).
Support Vector Machines (SVM).

This is a good mixture of simple linear (LR and LDA), 
nonlinear (KNN, CART, NB and SVM) algorithms. 
We reset the random number seed before each run to ensure that
 the evaluation of each algorithm is performed using exactly 
 the same data splits. It ensures the results are directly comparable.

Lets build and evaluate our five models:
"""
names = ['bonus', 'exercised_stock_options','expenses','from_poi_to_this_person','from_this_person_to_poi', 'other','poi','restricted_stock', 'salary','shared_receipt_with_poi','to_messages','total_payments','total_stock_value','poi_emails', 'all_emails','poi_email_ratio']                        

#names = ['bonus', 'exercised_stock_options','expenses','from_messages','from_poi_to_this_person','from_this_person_to_poi', 'other','restricted_stock', 'salary','shared_receipt_with_poi','to_messages','total_payments','total_stock_value','poi_email_ratio', 'new_poi']
# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
# evaluate each model in turn
results = []
names = []

#http://scikit-learn.org/stable/modules/cross_validation.html

for name, model in models:
  kfold = cross_validation.KFold(n=num_instances, n_folds=num_folds, random_state=seed)
  cv_results = cross_validation.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
  results.append(cv_results)
  names.append(name)
  #print "model, X_train, Y_train, cv=kfold, scoring=scoring:\n", model, X_train, Y_train, cv, scoring
  msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
  #print "name:\n", name
  #print "model:\n", model
  print "name, cv_results.mean(), cv_results.std():\n", (msg)

print "  Select Best Model "
"""
We can also create a plot of the model evaluation results 
and compare the spread and the mean accuracy of each model. 
There is a population of accuracy measures for each algorithm 
because each algorithm was evaluated 10 times (10 fold cross validation).
"""

# Compare Algorithms
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
ax.set_yticklabels(names)
plt.show()

df.new_poi.hist(bins=30)
plt.title('poi distribution');
plt.xlabel('[new_poi]')
plt.show()
print "+++++++++++++++++++"

"""
KNN algorithm  model : 
Now we want to get an idea of the accuracy of the model on our validation set.

This will give us an independent final check on the accuracy of the best model. 
It is valuable to keep a validation set just in case you made a slip during training,
 such as overfitting to the training set or a data leak. Both will result 
 in an overly optimistic result.

We can run the KNN model directly on the validation set and summarize the results 
as a final accuracy score, a confusion matrix and a classification report.

"""
# Make predictions on validation dataset
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_validation)
print "Predictions using KNeighborsClassifier:\n", predictions

print "accuracy score:\n", (accuracy_score(Y_validation, predictions))
print "confusion_matrix:\n",(confusion_matrix(Y_validation, predictions))
print "classification_report:\n", (classification_report(Y_validation, predictions))

print "************************"




