from __future__ import print_function
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.optimizers import SGD, Adam, rmsprop
from keras.utils import np_utils

import pandas as pd
import numpy as np
import xgboost as xgb
import sys

import random
from time import time

def read_file(path):
	d = pd.read_csv(path, header=0, date_parser = dateparse, parse_dates=['date_recorded'])
	d['date_recorded'] = d['date_recorded'].dt.year

	return d

def feature_extraction(train, test, y_train):
	z = ['id','amount_tsh',  'num_private', 'region', 
		'quantity', 'quality_group', 'source_type', 'payment', 
		'waterpoint_type_group',
		'extraction_type_group', 'scheme_name']
	for i in z:
		del train[i]
		del test[i]
	
	for data in [train, test]:
		data['construction_year'].replace(0, data[data['construction_year'] != 0]['construction_year'].mean(), inplace=True)

		data['date_recorded'] = pd.to_datetime(data['date_recorded'])
		data['year_recorded'] = data['date_recorded'].apply(lambda x: x.year)
		data['month_recorded'] = data['date_recorded'].apply(lambda x: x.month)
		data['date_recorded'] = (pd.to_datetime(data['date_recorded'])).apply(lambda x: x.toordinal())	

	for z in ['month_recorded', 'year_recorded']:
		train[z] = train[z].apply(lambda x: str(x))
		test[z] = test[z].apply(lambda x: str(x))
		good_cols = [z+'_'+i for i in train[z].unique() if i in test[z].unique()]
		train = pd.concat((train, pd.get_dummies(train[z], prefix = z)[good_cols]), axis = 1)
		test = pd.concat((test, pd.get_dummies(test[z], prefix = z)[good_cols]), axis = 1)

	z = ['public_meeting', 'permit']
	for data in [train, test]:
		for i in z:
			data[i].fillna(False, inplace = True)
			data[i] = data[i].apply(lambda x: float(x))

	trans = ['longitude', 'latitude', 'gps_height', 'population']
	for data in [train, test]:
		data.loc[data.longitude == 0, 'latitude'] = 0
	for z in trans:
		for data in [train, test]:
			data[z].replace(0., np.NaN, inplace = True)
			data[z].replace(1., np.NaN, inplace = True)
        
		for j in ['subvillage', 'district_code', 'basin']:
			train['mean'] = train.groupby([j])[z].transform('mean')
			train[z] = train[z].fillna(train['mean'])
			o = train.groupby([j])[z].mean()
			fill = pd.merge(test, pd.DataFrame(o), left_on=[j], right_index=True, how='left').iloc[:,-1]
			test[z] = test[z].fillna(fill)
		
		train[z] = train[z].fillna(train[z].mean())
		test[z] = test[z].fillna(train[z].mean())
		del train['mean']        

	train['population'] = np.log(train['population'])
	test['population'] = np.log(test['population'])

	cols = [i for i in train.columns if type(train[i].iloc[0]) == str]
	train[cols] = train[cols].where(train[cols].apply(lambda x: x.map(x.value_counts())) > 100, "other")
	for column in cols:
		for i in test[column].unique():
			if i not in train[column].unique():
				test[column].replace(i, 'other', inplace=True)

	columns = [i for i in train.columns if type(train[i].iloc[0]) == str]
	for column in columns:
		train[column].fillna('NULL', inplace = True)
		good_cols = [column+'_'+i for i in train[column].unique() if i in test[column].unique()]
		train = pd.concat((train, pd.get_dummies(train[column], prefix = column)[good_cols]), axis = 1)
		test = pd.concat((test, pd.get_dummies(test[column], prefix = column)[good_cols]), axis = 1)
		del train[column]
		del test[column]

	return train, test


trainFile = sys.argv[1]
labelFile = sys.argv[2]
testFile = sys.argv[3]
threshold = 1000
NUM_ROUNDS = 500

print('Reading data...')
dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d')
x_train = read_file(trainFile)
y_train = pd.read_csv(labelFile, header=0)
del y_train['id']
test_data = read_file(testFile)
print(y_train.values.ravel())

y_train['status_group'] = y_train['status_group'].astype('category')
int_to_label = y_train['status_group'].cat.categories

print('Feature Extraction')
x_train, test_data = feature_extraction(x_train, test_data, y_train)
print(x_train.columns.unique())
print(len(x_train.columns.unique()))

n_samples = x_train.shape[0]

if '--val' in sys.argv:
	print('Splitting train and validation...')
	amount = int(0.8*x_train.shape[0])
	x_validation = x_train[amount:]
	x_train = x_train[:amount]
	y_validation = y_train[amount:]
	y_train = y_train[:amount]

print('Training...')
if '--rf' in sys.argv:
	tuned_parameters = { 'n_estimators':[700,900,1100]}

	clf = RandomForestClassifier(max_features = 'auto', n_estimators = 1000, n_jobs = -1)
	#clf = GridSearchCV( clf, tuned_parameters, cv=3, scoring='accuracy', n_jobs = -1)
	#clf = GridSearchCV( AdaBoostClassifier(base_estimator = DecisionTreeClassifier), tuned_parameters, cv=5,scoring='accuracy', n_jobs = -1)
	t0 = time()
	clf.fit(x_train, y_train.values.ravel())
	print("---done in %0.3fs" % (time() - t0))
	print("---training score : %.5f " % (clf.score(x_train, y_train)))
	#print("Best parameters set:")
	#best_parameters = clf.best_estimator_.get_params()
	#best_parameters = clf.get_params()
	#print(best_parameters)
	#importance = (clf.feature_importances_)
	#clf = clf.best_estimator_
	if '--val' in sys.argv:
		y_validation = list(y_validation)
		print("---valadation score : %.5f " % (clf.score(x_validation, y_validation)))

elif '--xgb' in sys.argv:
	f = y_train['status_group'].cat.codes
	x_train = x_train.as_matrix()

	dtrain = xgb.DMatrix( x_train, label=f)
	param = {'max_depth':14, 'eta':.1, 'silent':1, 'objective':'multi:softmax', 
			 'num_class' : 3, 'colsample_bytree': .6, 'subsample': 0.8}
	param['nthread'] = 4
	param['eval_metric'] = 'merror'
	num_round = NUM_ROUNDS

	for i in range(2,13):
		param['seed'] = i
		t0 = time()
		clf = xgb.train(param, dtrain, num_round)
		print("---model with seed {} done in {}s".format(i, (time() - t0)))
		clf.save_model('000{}.model'.format(i))

		d_train = xgb.DMatrix(x_train)
		pred = clf.predict(d_train)
		error_rate = np.sum(pred == f) / x_train.shape[0]
		print('Train acc = {}'.format(error_rate))

		if '--val' in sys.argv:
			f_val = y_validation['status_group'].cat.codes
			x_val = x_validation.as_matrix()
			d_val = xgb.DMatrix(x_val)
			pred = clf.predict(d_val)
			error_rate = np.sum(pred == f_val) / x_val.shape[0]
			print('Val acc = {}'.format(error_rate))

# Testing
n_test_samples = test_data.shape[0]
print('Testing...')
if '--xgb' in sys.argv:
	test_data = test_data.as_matrix()
	test_data = xgb.DMatrix(test_data)
ans = clf.predict(test_data).astype(np.int32)
if '--xgb' in sys.argv:
	ans = int_to_label[ans]
print(ans)

x_test= read_file(testFile).as_matrix() # in order to read id
result = open('result.csv', 'w')
result.write('id,status_group\n')
for i in range(n_test_samples):

	result.write('{},{}\n'.format(x_test[i][0], ans[i]))

result.close()

