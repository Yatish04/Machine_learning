import numpy
import pandas
from numpy import genfromtxt
from sklearn import datasets
from sklearn import metrics
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.ensemble import ExtraTreesClassifier

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

my_data = genfromtxt('cartrain.csv', delimiter=',',dtype=int)

mytarget= genfromtxt('target.csv',delimiter=',',dtype=int)


model = ExtraTreesClassifier()
model.fit(my_data, mytarget)
print(model.feature_importances_)
print("---------------------Selectkbest-------------------------------")
test = SelectKBest(score_func=chi2, k=5 )
fit = test.fit(my_data, mytarget)

numpy.set_printoptions(precision=3)
print(fit.scores_)
features = fit.transform(my_data)
# summarize selected features
print(features[0:5,:])

print("-------------------------rfe with randomforest------------------------------------------")
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

model = RandomForestClassifier()
rfe = RFE(model, 4)
fit = rfe.fit(my_data, mytarget)
print("Num Features: %d" % fit.n_features_)
print("Selected Features: %s" % fit.support_)
print("Feature Ranking: %s" % fit.ranking_)

print("------------------------------randomforest---------------------------------------")

model=RandomForestClassifier(max_depth=2, random_state=0)
model.fit(my_data,mytarget)
rfe = RFE(model, 4)
fit = rfe.fit(my_data, mytarget)
print("Num Features: %d" % fit.n_features_)
print("Selected Features: %s" % fit.support_)
print("Feature Ranking: %s" % fit.ranking_)
print(model.feature_importances_)

print("----------------------------------pca----------------------------------------------")
from sklearn.decomposition import PCA
pca = PCA(n_components=5)
fit = pca.fit(my_data)

# summarize components
print("Explained Variance: %s" % fit.explained_variance_ratio_)
print(fit.components_)