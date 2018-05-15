import numpy
import pandas
from numpy import genfromtxt
from sklearn.ensemble import RandomForestClassifier
my_data = genfromtxt('cartrain.csv', delimiter=',',dtype=int,usecols=(0,1,2,3,4,5))
mytarget= genfromtxt('target.csv',delimiter=',',dtype=int)

trainx=my_data[0:760]
trainy=mytarget[0:760]
testx=my_data[760:]
testy=mytarget[760:]
model=RandomForestClassifier(n_estimators=15, random_state=0)
model.fit(trainx,trainy)

print("actual:",testy)

print("predicted: ",model.predict(testx))
print(model.score(testx,testy))