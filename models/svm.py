import numpy
import pandas
from numpy import genfromtxt
from sklearn import datasets
from sklearn import metrics
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

my_data = genfromtxt('cartrain.csv', delimiter=',',dtype=int,usecols=(0,1,2,3,4,5))
mytarget= genfromtxt('target.csv',delimiter=',',dtype=int)

trainx=my_data[0:760]
trainy=mytarget[0:760]
testx=my_data[760:]
testy=mytarget[760:]
model=SVC()
model.fit(trainx,trainy)
print(model.predict(testx))
print(testy)
print(model.score(testx,testy))
