import pandas
import numpy as np
import pickle
import csv
import ruamel.yaml
from sklearn.ensemble import GradientBoostingClassifier 
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import accuracy_score 
from sklearn.metrics import confusion_matrix

filename = 'rot_model.sav'
gbm = pickle.load(open(filename, 'rb'))

tablename = '32.csv'
validation = pandas.read_csv(tablename, index_col='N')
validation = validation[['MeanGreen', 'StdGreen', 'MeanRed', 'StdRed', 'MeanInfrared', 'StdInfrared', 'MeanIntensity', 'StdIntensity',
 'MeanNDSM', 'StdNDSM', 'MeanNDVI', 'StdNDVI', 'Colour']]
validation.dropna(axis = 0, inplace = True)

test = validation.as_matrix()[:, :-1]
truth = np.array(validation[['Colour']])

predicted = gbm.predict(test)
precision, recall, fscore, support = score(truth, predicted)

matrix = np.vstack((precision, recall, fscore, support))

prfs = pandas.DataFrame(matrix, index = ['Precision', 'Recall', 'F1','Support'], columns = ['imp_surf', 'building', 'low_veg', 'tree', 'car', 'clutter'])
conf_mat = pandas.DataFrame(confusion_matrix(truth, predicted), index = ['imp_surf', 'building', 'low_veg','tree', 'car', 'clutter'], columns = ['imp_surf', 'building', 'low_veg', 'tree', 'car', 'clutter'])
frames = [conf_mat, prfs]
result = pandas.concat(frames)

result.to_csv('confusion_matrix_for_' + tablename)

fd = open('confusion_matrix_for_' + tablename, 'a')
fd.write(str(accuracy_score(truth, predicted)))
fd.close()

data = dict(enumerate(predicted.tolist(), start=1))
ruamel.yaml.dump(data, open('/home/sanity-seeker/Programming/Projects/segmentation/semantic_project/testing data/32.yml', 'w'), default_flow_style=False)
