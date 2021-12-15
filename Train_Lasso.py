import pandas as pd
import numpy as np
from sklearn.preprocessing import Normalizer
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import GridSearchCV

filename = 'Data558.csv'
data = pd.read_csv(filename)
#peek = data.head(30)
#hape = data.shape
#ypes = data.dtypes
#pd.set_option('display.max_rows', None, 'display.max_columns', None)
#pd.set_option('precision', 3)
#description = data.describe()
#EPDS_counts = data.groupby('EPDS_9').size()       #0/1 n=365/n=193
#skew = data.skew()

X = data[['AGE', 'DeliveryWays', 'Marital', 'EDU', 'Income',
          'Q21.Planed', 'Q21.2', 'Q21.3', 'Q21.4', 'Q21.5', 'Q21.12', 'Q21.13', 'Q21.14', 'Q21.15', 'Q21.16', 'Q21.17', 'Q21.18',
          #After Birth
          'Q22.1', 'Q22.2', 'Q22.3', 'Q22.4', 'Q22.5', 'Q22.6', 'Q22.7', 'Q22.8', 'Q22.9', 'Q22.10', 'Q22.11', 'Q22.12',
          #pregnancy complication Eg., 23.1:Multiple pregnancy, 23.12:Pre-eclampsia
          'Q23.1', 'Q23.2', 'Q23.3', 'Q23.4', 'Q23.5', 'Q23.6', 'Q23.7', 'Q23.8', 'Q23.9', 'Q23.10', 'Q23.11', 'Q23.12',
          #baby health
          'Q24.1', 'Q24.2', 'Q24.3', 'Q24.4', 'Q24.5',
          #breastfeeding
          'Q25', 'Q26.1', 'Q26.2', 'Q26.3', 'Q27.1', 'Q27.2', 'Q27.3', 'Q27.4', 'Q27.5', 'Q27.6', 'Q27.7', 'Q27.8', 'Q27.9', 'Q28.1', 'Q28.2', 'Q28.3',
          'FamilySupport1', 'FamilySupport2', 'FamilySupport3', 'FamilySupport4', 'FamilySupport5',
          'PartnerSupport1', 'PartnerSupport2', 'PartnerSupport3',
          'FearCovid1', 'FearCovid2', 'FearCovid3', 'FearCovid4', 'FearCovid5', 'FearCovid6', 'FearCovid7',
          #UCLA 3items
          'Q49.7', 'Q49.8', 'Q49.9',
          #Personality
          'Q52.1', 'Q52.2', 'Q52.3', 'Q52.4', 'Q52.5', 'Q52.6', 'Q52.7', 'Q52.8', 'Q52.9', 'Q52.10',
          #alcohol, drug, substance use
          'Q55.1　AlcoholDrug', 'Q55.2', 'Q55.3', 'Q55.4', 'Q55.5', 'Q55.6', 'Q55.7', 'Q55.8', 'Q55.9',
          #1:hypertention, 2:diabetes,15:depression
          'Q61.1', 'Q61.2', 'Q61.3', 'Q61.4', 'Q61.5', 'Q61.6', 'Q61.7', 'Q61.8', 'Q61.9', 'Q61.10', 'Q61.11', 'Q61.12',
          'Q61.13', 'Q61.14', 'Q61.15', 'Q61.16']].values
# X[558,115]
y = data['EPDS_9'].values

##Lasso Regression
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
#lasso = Lasso(alpha=1.0)
#lasso.fit(X_train, y_train)
#print(lasso.score(X_test, y_test), lasso.score(X_train, y_train))
#print(lasso.coef_)


#model = Lasso(alpha=1.0)
#cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
#scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
#scores = np.absolute(scores)
#average mean absolute error (MAE)
#print('Mean MAE: %.3f(%.3f)' % (np.mean(scores), np.std(scores)))
#Mean MAE: 0.454(0.022)


#model = Lasso()
#cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
#grid = dict()
#grid['alpha'] = np.arange(0, 1, 0.001)
#search = GridSearchCV(model, grid, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
#results = search.fit(X, y)
#print('MAE: %.3f' % results.best_score_)
#print('Config: %s' % results.best_params_)
#MAE: -0.348 Config: {'alpha': 0.01}

#Normalized X
#scaler = Normalizer().fit(X)
#normalizedX = scaler.transform(X)
#model = Lasso()
#cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
#grid = dict()
#grid['alpha'] = np.arange(0, 1, 0.01)
#search = GridSearchCV(model, grid, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
#results = search.fit(normalizedX, y)
#print('MAE: %.3f' % results.best_score_)
#print('Config: %s' % results.best_params_)
#MAE: -0.376 Config: {'alpha': 0.0}

