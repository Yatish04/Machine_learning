import numpy
import pandas

from numpy import genfromtxt

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.ensemble import RandomForestClassifier



my_data = genfromtxt('cartrain.csv', delimiter=',',dtype=int,usecols=(0,1,2,3,4,5))
print(my_data)
mytarget= genfromtxt('target.csv',delimiter=',',dtype=int)

testx=genfromtxt('car_test.csv',delimiter=',',dtype=int)
trainx=my_data[:]
trainy=mytarget[:]

model=RandomForestClassifier(n_estimators=14, random_state=0)
model.fit(trainx,trainy)
mdata=model.predict(testx)
with open('temp.txt','w') as fh:
	for i in mdata:
		fh.write(str(i))
		fh.write("\n") 
