import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import shap



filename = 'Data0420.csv'
data = pd.read_csv(filename)

y = data['EPDS_01'].values
X = data[['AGE', 'Income1', 'Income2', 'Income3', 'Income4', 'Income5', 'Income6', 'Income7', 'TimeBirth1', 'TimeBirth2', 'TimeBirth3',
          'Childbirth1', 'Childbirth2', 'Childbirth3', 'Childbirth4', 'Q2.1LiveSum', 'LiveChild', 'Q3.1Live_husband', 'Live_parents',
          'Employment1', 'Employment2',	'Employment3', 'Employment4', 'Employment5', 'FirmSize1', 'FirmSize2', 'FirmSize3', 'FirmSize4',
          'JobType1', 'JobType2', 'JobType3', 'JobType4',
          'JobDemand_1', 'JobDemand_2', 'JobDemand_3', 'JobControl_1', 'JobControl_2', 'JobControl_3', 'JobInterpersonal_1', 'JobInterpersonal_2', 'JobInterpersonal_3',
          'JobEnvironment_1', 'JobEnvironment_2', 'JobEnvironment_3', 'JobEngagement_1', 'JobEngagement_2', 'JobEngagement_3','JobBullying_1', 'JobBullying_2', 'JobBullying_3',
          'JobBossSupport_1', 'JobBossSupport_2', 'JobBossSupport_3', 'JobCoworkerSupport_1', 'JobCoworkerSupport_2', 'JobCoworkerSupport_3',
          'JobFamilySupport_1', 'JobFamilySupport_2', 'JobFamilySupport_3','Q13Marital',	'EDU_01', 'House01', 'WorkStatus1', 'WorkStatus2', 'WorkStatus3', 'WorkStatus4',
          'Q21.1Planned', 'Q21.2', 'Q21.3', 'Q21.4', 'Q21.5', 'Q21.14', 'Q21.15', 'Q21.16', 'Q21.17', 'Q21.8_1', 'Q21.8_2', 'Q21.8_3',
          'Q22.3', 'Q22.4', 'Q22.5', 'Q22.6', 'Q22.7', 'Q22.9', 'Q22.11', 'Q22.13',
          'Q23.1Complications', 'Q23.2', 'Q23.3', 'Q23.4', 'Q23.5', 'Q23.6', 'Q23.7', 'Q23.8', 'Q23.9', 'Q23.10', 'Q23.11', 'Q23.12',
          'Q24.1BirthWeight', 'Q24.2', 'Q24.3',  'Q24.4', 'Q24.5', 'Breastfeed1', 'Breastfeed2', 'Breastfeed3',
          'Q28.1', 'Q28.2', 'Q28.3', 'Q28.4', 'PhysicalLoad1', 'PhysicalLoad2',	'PhysicalLoad3', 'MentalLoad1',	'MentalLoad2', 'MentalLoad3',
          'CovidWork1',	'CovidWork2', 'CovidWork3',	'VDTApril_1', 'VDTApril_2', 'VDTApril_3', 'VDTApril_4', 'SitApril_1', 'SitApril_2', 'SitApril_3', 'SitApril_4', 'SitApril_5',
          'WalkApril_1', 'WalkApril_2', 'WalkApril_3', 'WalkApril_4', 'SportsApril_1', 'SportsApril_2', 'SportsApril_3', 'SportsApril_4',
          'SleepApril_1', 'SleepApril_2', 'SleepApril_3', 'SleepApril_4', 'SleepApril_5', 'WashApril_1', 'WashApril_2', 'WashApril_3', 'WashApril_4', 'WashApril_5',
          'ChildApril_1', 'ChildApril_2', 'ChildApril_3', 'ChildApril_4', 'VDTJune_1', 'VDTJune_2', 'VDTJune_3', 'VDTJune_4',
          'SitJune_1', 'SitJune_2', 'SitJune_3', 'SitJune_4', 'SitJune_5', 'WalkJune_1', 'WalkJune_2', 'WalkJune_3', 'WalkJune_4',
          'SportsJune_1', 'SportsJune_2', 'SportsJune_3', 'SportsJune_4',  'SleepJune_1', 'SleepJune_2', 'SleepJune_3', 'SleepJune_4', 'SleepJune_5',
          'WashJune_1', 'WashJune_2', 'WashJune_3', 'WashJune_4', 'WashJune_5', 'ChildJune_1', 'ChildJune_2', 'ChildJune_3', 'ChildJune_4',
          'Q35.15', 'Q35.16', 'Q35.17', 'Q35.19', 'CoupleArgue1', 'CoupleArgue2', 'CoupleArgue3', 'CoupleArgue4',
          'YearIncome',	'YearIncome1', 'YearIncome2', 'YearIncome3', 'YearIncome4', 'Influenza1', 'Influenza2',	'Influenza3',
          'Impact40_1', 'Impact40_2', 'Impact40_3','Impact40_4',
          'ChildMental1', 'ChildMental2', 'ChildMental3','ChildViolent1', 'ChildViolent2',	'ChildViolent3', 'ChildShout1',	'ChildShout2', 'ChildShout3',
          'ChildLearn1', 'ChildLearn2',	'ChildLearn3','ChildSchool1', 'ChildSchool2', 'ChildSchool3',
          'Q43.9', 'Q43.12', 'Q43.13', 'Q48.1',	'Q48.2', 'Q48.3', 'Q48.4', 'Q48.5',	'Q48.13', 'Q48.15',
          'WorkPerform1', 'WorkPerform2', 'WorkPerform3', 'WorkPerform4', 'HouseworkPerform1', 'HouseworkPerform2', 'HouseworkPerform3', 'HouseworkPerform4',
          'Q51.1TrustNeighbor',	'Q51.2NeighborHelp', 'Q51.4TrustGovern', 'Alcohol1', 'Alcohol2', 'Alcohol3', 'Drug1', 'Drug2', 'Drug3',
          'Smoke1',	'Smoke2', 'Smoke3', 'Q61.15Depression',	'Q61.16Psychiatry',
          ######################
          #continus variables
          ######################
          'Q36.1Lifestyle', 'Q36.2', 'Q36.3', 'Q36.4', 'Q36.5', 'Q36.6', 'Q36.7', 'Q36.8', 'Q36.9',
          'SumQ26LearnBreastfeed', 'NumCovidSymptom', 'NumQ32EventsApril', 'NumQ35PreventCovid', 'SumQ42ChildAbuse',
          'SumQ43StressEvent', 'SumQ44FearCovid', 'SumQ47Information', 'Violence', 'SumLoneliness', 'NumIllness',
          'SumFamilysupport', 'SumPartnersupport', 'Extroversion', 'Agreeable', 'Conscientious', 'Emotion', 'Open']]
y = data['EPDS_01'].values

print(X.size)
# scaler = MinMaxScaler()
# X = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1, stratify=y)
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)
# print('Parameters currently in use:\n', clf.get_params())

#GridSearch with hyperparameters
# clf = RandomForestClassifier(max_depth = 10, n_estimators = 1000, random_state=42)
# clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
print(cnf_matrix)
class_names = [0, 1]
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')

print('Accuracy:', metrics.accuracy_score(y_test, clf.predict(X_test)))
print('Precision:', metrics.precision_score(y_test, clf.predict(X_test)))
print('Recall:', metrics.recall_score(y_test, clf.predict(X_test)))
#plt1 = plt.figure(1)
#plt1.show()

y_pred_pro = clf.predict_proba(X_test)[::, 1]
#print('predicted probability of y:', y_pred_pro)
print('Area under the curve:', metrics.roc_auc_score(y_test, y_pred_pro))
metrics.RocCurveDisplay.from_predictions(y_test, y_pred_pro)
#plt.show()

cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=1)
params = {'n_estimators': [100, 300, 500, 1000],
          'max_depth': [3, 5, 10, 20],
          'max_features': [20, 30, 50, 90],
          # 'min_samples_split': min_samples_split,
          # 'min_samples_leaf': min_samples_leaf
}

search = GridSearchCV(clf, params, scoring='accuracy', n_jobs=-1, cv=cv)
result = search.fit(X, y)
print('Best Score: %s' % result.best_score_)
print('Best Hyperparameters: %s' % result.best_params_)

