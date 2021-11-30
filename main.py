#cite:https://machinelearningmastery.com/elastic-net-regression-in-python/
         
         
import os
import pandas as pd
from numpy import mean
from numpy import std
from numpy import absolute
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import ElasticNet

path = os.path.dirname(os.path.abspath(__file__))
#print('getcwd:      ', os.getcwd())
#print('__file__:    ', __file__)

#load dataset
data = pd.read_csv('Data558.csv')
#print(data.shape)
#print(data.head())

y = data.EPDS_9
X = data[['AGE', 'DeliveryWays','Marital','EDU','Q21.Planed','FamilySupport1', 'FamilySupport2','FamilySupport3',
         'FamilySupport4','FamilySupport5','PartnerSupport1','PartnerSupport2','PartnerSupport3',
         'FearCovid1','FearCovid2','FearCovid3','FearCovid4','FearCovid5','FearCovid6','FearCovid7','Income',
         #alcohol, drug, stubstance
         'Q55.1　AlcoholDrug','Q55.2','Q55.3','Q55.4','Q55.5','Q55.6','Q55.7','Q55.8','Q55.9',
         #1hypertention, 2diabetes,15depression
         'Q61.1','Q61.2','Q61.3','Q61.4','Q61.5','Q61.6','Q61.7','Q61.8','Q61.9','Q61.10','Q61.11','Q61.12',
         'Q61.13','Q61.14','Q61.15','Q61.16']]
#print(X)
#define model
model = ElasticNet(alpha=1.0, l1_ratio=0.5)
#define model evaluation method
cv = RepeatedKFold (n_splits=10, n_repeats=3, random_state=1)
#evaluate model
scores = cross_val_score(model, X,y, scoring='neg_mean_absolute_error',cv=cv, n_jobs=-1)
#force scores to be positive
scores = absolute(scores)
print('Mean MAE: %.3f (%.3f)' % (mean(scores),std(scores)))
