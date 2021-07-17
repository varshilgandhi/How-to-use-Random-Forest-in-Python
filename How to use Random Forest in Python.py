# -*- coding: utf-8 -*-
"""
Created on Tue May 11 08:45:31 2021

@author: abc
"""

import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

#read our data
df = pd.read_csv('images_analyzed_productivity1.csv')
print(df.head())

# Let's see how many values are divided into good or bad
sizes = df['Productivity'].value_counts(sort=1)
print(sizes)

#our data to be able to predict what are the independent variables into our dataset
#here we drop independent variables 
#In the dataset user, Images_analyzed are independent variables so we drop it.

#drop independent variables
df.drop(['Images_Analyzed'], axis=1, inplace=True)
df.drop(['User'], axis=1, inplace=True)
print(df.head())

#Handle missing values
#if there is any missing value occur so it drop that overall row
df = df.dropna()  #in our dataset there is noone null value

#Handle non numeric data
#Convert non-numeric data into numeric data
#In our dataset good or bad let convert into 1 or 2
df.Productivity[df.Productivity == 'Good'] = 1
df.Productivity[df.Productivity == 'Bad'] = 2
print(df.head())

#Define dependent variable 
Y = df['Productivity'].values
#Convert Y object into integer 
Y=Y.astype('int')

#Define independent variables
X = df.drop(labels=['Productivity'], axis=1)


#######################################################################################

#Let's Split our data into training and testing purpose

import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

df = pd.read_csv('images_analyzed_productivity1.csv')

#Convert Non numeric value into Numeric value
df.Productivity[df.Productivity == 'Good'] = 1
df.Productivity[df.Productivity == 'Bad'] = 2

#define dependent variables
Y = df['Productivity'].values
#Convert Y object into integer
Y=Y.astype('int')

#Define independent variables
X = df.drop(labels=['Productivity'], axisi=1)

# Split our data into Training and Testing dataset
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4, random_state=20)
#test_size = 0.4 means 40% of our data randomly selected and assign for testing purpose
#random_state = 20 means whatever randomply picked that 40% data we return every time same value of data

print(X_train) #It's our 40% trained data
print(X_test) #It's our 40% tested data

##########################################################################################

#Let's Apply random forest

import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

#read our data
df = pd.read_csv('images_analyzed_productivity1.csv') 

#drop independent variables
df.drop(['Images_Analyzed'], axis=1, inplace=True)
df.drop(['User'], axis=1, inplace=True)

#Convert Non numeric value into numeric value
df.Productivity[df.Productivity == 'Good'] = 1
df.Productivity[df.Productivity == 'Bad'] = 2

#define dependent variables
Y = df['Productivity'].values
#Convert Y object into integer
Y=Y.astype('int')

#Define independent varibles
X = df.drop(labels=['Productivity'], axis=1)

#Split our data into Training and Testing dataset
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4, random_state = 20)
#test_size = 0.4 means 40% of our data randomly selected and assign for testing purpose
#random_state = 20 means whatever randomly picked that 40% data we return every time same value of data

#Apply random forest
#we use RandomForestClassifier because in our dataset Productivity it's classify the value into  good or bad
from sklearn.ensemble import RandomForestClassifier 
model = RandomForestClassifier(n_estimators=10, random_state=30)

#fit data
model.fit(X_train, Y_train)

#Predict our dataset
prediction_test = model.predict(X_test)
print(prediction_test)

#find our prediction accuracy
from sklearn import metrics
print("Accuracy = ", metrics.accuracy_score(Y_test, prediction_test))

#find what parameters or features occurs in our model
features_list = list(X.columns)
feature_imp = pd.Series(model.feature_importances_, index=features_list).sort_values(ascending=False) 
print(feature_imp)

############################################################################################################

                               #THANK YOU
















































































































