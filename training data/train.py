import pandas
import numpy as np
import pickle
from sklearn.ensemble import GradientBoostingClassifier 
from sklearn.grid_search import GridSearchCV

dataname = 'database.csv'
data = pandas.read_csv(dataname, index_col='N')
data = data[['MeanGreen', 'StdGreen', 'MeanRed', 'StdRed', 'MeanInfrared', 'StdInfrared', 'MeanIntensity', 'StdIntensity',
 'MeanNDSM', 'StdNDSM', 'MeanNDVI', 'StdNDVI', 'Colour']]
data.dropna(axis = 0, inplace = True)

aim = np.array(data['Colour']) 
X = data.as_matrix()[:, :-1]


'''
param_grid = {
    'max_depth': [2, 3, 4, 5],
    'learning_rate': [0.1, 0.3, 0.5],
    'n_estimators': [50, 100, 400] 
}

test_gbt = GradientBoostingClassifier(n_estimators=10) 

CV_gbt = GridSearchCV(estimator = test_gbt, param_grid=param_grid)

test_data = data
test_data.sample(frac=1)
test_data = test_data[:500000]
test_aim = np.array(test_data['Colour'])
test_X = data.as_matrix()[:, :-1]

CV_gbt.fit(test_X, test_aim)

gbm = GradientBoostingClassifier(CV_gbt.best_params_)
gbm.fit(X, aim)
'''
gbm = GradientBoostingClassifier(n_estimators=100, verbose=1)
gbm.fit(X, aim)

filename = 'rot_model.sav'
pickle.dump(gbm, open('/home/sanity-seeker/Programming/Projects/semantic_project/testing data/' + filename, 'wb'))
