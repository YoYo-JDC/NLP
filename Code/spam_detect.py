# -*- coding: utf-8 -*-
"""
Spyder Editor

4/22/2023
author: Jingze Jiang

"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.model_selection import train_test_split
import datetime as dt
import chardet
import os
os.chdir('C:/Users/jjian/OneDrive/Full Professor/Github/NPL Folder/')
from parameters import *



pd.set_option('display.max_columns',None)
analy_dt=dt.datetime.now().strftime("%Y-%m-%d")

# =============================================================================
# setup output folder
# =============================================================================
output_path=filepath

try:
    os.mkdir(output_path)
except FileExistsError:
    print('Directory Already Exists')
    
try:
    os.mkdir(output_path+'/'+analy_dt)
except FileExistsError:
    print('Directory Already Exists')




    
#import the dataset
#the dataset is download from the kaggle "SMS Spam Collection Dataset"
#https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset?resource=download
file='Data/spam.csv'

#using pd.read_csv will generate error code 
#UnicodeDecodeError: 'utf-8' codec can't decode bytes in position 606-607: invalid continuation byte
#we are using the following code to identify which obs trigger the error and 
#fix it
with open(file, 'rb') as rawdata:
    result = chardet.detect(rawdata.read(100000))
result


#import the data
data = pd.read_csv(file,encoding='Windows-1252')
data.head()

#eleminating the extra columns that do not contain any obs

data=data[['v1','v2']]
data.head()

#define the y variable
y=data['v1']

#split the data into training and testing sets

X_train, X_test, y_train, y_test=train_test_split(data['v2'],y,test_size=0.33,random_state=53)


#processing the training text data for our ML model

# =============================================================================
# #Using count to vecterize the text
# 
# =============================================================================
#initiate the countvecterizer
count_vecterizer=CountVectorizer(stop_words='english')

#vectorize the training data
count_train=count_vecterizer.fit_transform(X_train.values)

#vectorize the testing data
count_test=count_vecterizer.transform(X_test.values)

#print the first 10 feature
print(count_vecterizer.get_feature_names_out()[:10])

# =============================================================================
# Using tfidf model to vectorize the data
# =============================================================================

#initiate the tfidf vecterizer
tfidf_vectorizer=TfidfVectorizer(stop_words='english',max_df=0.7)

#vecterize the X variable for both training and testing set
#vectorizing the training set
tfidf_train=tfidf_vectorizer.fit_transform(X_train.values)

#vectorizing the testing set
tfidf_test=tfidf_vectorizer.transform(X_test.values)

#print the first 10 features
print(tfidf_vectorizer.get_feature_names_out()[:10])

#print the 5 vector in the training set

print(tfidf_train.A[:,5])

#we will adopt the Multinomial ML model to perform the 
#superised ML

# =============================================================================
# Based on CountVectorizer 
# =============================================================================

#initiate the MultinomialNB
model_count=MultinomialNB()

#fit the model
model_count.fit(count_train,y_train)

#predict model
pred_count=model_count.predict(count_test)

#measure the model quality using accuracy score 
print(metrics.accuracy_score(y_test,pred_count))

#measure the model quality using confusion matrix
print(metrics.confusion_matrix(y_test,pred_count,labels=['spam','ham']))

# =============================================================================
# Based on tfidfvectorizer
# =============================================================================
#initiate the MultinomialNB
model_tfidf=MultinomialNB()

#fit the model
model_tfidf.fit(tfidf_train,y_train)

#predict model
pred_tfidf=model_tfidf.predict(tfidf_test)

#measure the model quality using accuracy score 
print(metrics.accuracy_score(y_test,pred_tfidf))

#measure the model quality using confusion matrix
print(metrics.confusion_matrix(y_test,pred_tfidf,labels=['spam','ham']))


#improving the tfidfvectorizer model

#set up the smoothing variable possiable values
alphas=np.arange(0,1,0.1)


#build a function that pass alphas
def train_and_predict(alpha):
    #initiate the ML model
    nb_classifier=MultinomialNB(alpha=alpha)
    #fit the model with training set
    nb_classifier.fit(tfidf_train,y_train)
    #using the trained model to do prediction
    pred=nb_classifier.predict(tfidf_test)
    #measure the quality of the model using accuracy score
    accuracy=metrics.accuracy_score(y_test, pred)
    
    return(accuracy)

#build a loop to show all the results under different alpha value

for i in alphas:
    print ('alpha is', i)
    print ('accuracy score is', train_and_predict(i))
    print ('')
    
    
#inspect the model develped


# Get the class labels: class_labels
class_labels = model_tfidf.classes_

# Extract the features: feature_names
feature_names = tfidf_vectorizer.get_feature_names_out()

# Zip the feature names together with the coefficient array and sort by weights: feat_with_weights
feat_with_weights = sorted(zip(model_tfidf.feature_log_prob_[0],feature_names))

# Print the first class label and the top 20 feat_with_weights entries
print(class_labels[0], feat_with_weights[:20])

# Print the second class label and the bottom 20 feat_with_weights entries
print(class_labels[1], feat_with_weights[-20:])
    