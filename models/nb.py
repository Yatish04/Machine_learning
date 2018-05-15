from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
import numpy as np
import pandas
from numpy import genfromtxt
from sklearn import datasets
from sklearn import metrics
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.ensemble import ExtraTreesClassifier

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification


my_data = genfromtxt('cartrain.csv', delimiter=',',dtype=int,usecols=(0,1,2,3,4,5))
print(my_data)
mytarget= genfromtxt('target.csv',delimiter=',',dtype=int)

trainx=my_data[0:760]
trainy=mytarget[0:760]
testx=my_data[760:]
testy=mytarget[760:]


clf = MLPClassifier(solver='lbfgs', alpha=1e-10,hidden_layer_sizes=(20, 15), random_state=15)

clf.fit(trainx, trainy)
print(clf.score(testx,testy))