from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

from sklearn.tree import ExtraTreeClassifier

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
from sklearn.neighbors import KNeighborsClassifier
my_data = genfromtxt('cartrain.csv', delimiter=',',dtype=int,usecols=(0,1,2,3,4,5))
mytarget= genfromtxt('target.csv',delimiter=',',dtype=int)

trainx=my_data[250:]
trainy=mytarget[250:]
testx=my_data[:249]
testy=mytarget[:249]

model=DecisionTreeClassifier(random_state=1)

#model = KNeighborsClassifier(n_neighbors=5)
model.fit(trainx, trainy)
print("Expected")
print(testy)
print("Found")
print(model.predict(testx))
print("Score: ",model.score(testx,testy))
