import pandas as pd
from fairnessMeasuresAPI import fm
from SimpleFairTree import SimpleFairTree
from sklearn.model_selection import train_test_split
import numpy as np
import kagglehub


# Download latest version of law school dataset
path = kagglehub.dataset_download("danofer/law-school-admissions-bar-passage")
file = path + "/bar_pass_prediction.csv"





dataset = pd.read_csv(file, sep=',', header='infer')
# Add a column 'is_white' that is 0 if race1 is 'white', 1 otherwise
dataset['is_white'] = np.where(dataset['race1'] == 'white', 0, 1)
toDrop = ['decile1b', 'decile3', 'ID', 'decile1', 'race', 'cluster', 'lsat', 'ugpa', 'zfygpa', 'grad', 'zgpa', 'bar1', 'bar1_yr', 'bar2', 'bar2_yr', 'age', 'gender', 'race2', 'race1', 'bar', 'Dropout', 'other', 'asian', 'black', 'hisp', 'pass_bar', 'tier', 'index6040',   'indxgrp', 'indxgrp2', 'gpa', 'dnn_bar_pass_prediction', 'gpa', 'lsat']


dataset = dataset.drop(columns=toDrop)



# Define fairness measures parameters
sensible_attribute = 'is_white'
success_attribute = 'bar_passed'


sft = SimpleFairTree()


X = pd.get_dummies(dataset)
y = dataset[success_attribute]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
sft.fit(X_train, success_attribute)

predictions_test = sft.predict(X_test)
predictions_train = sft.predict(X_train)
print("Simple tree")
d11 = X_test.copy()
d11['y_hat'] = predictions_test
f11 = fm(dataset=d11, sensible_attribute=sensible_attribute, true_success=success_attribute, prediction='y_hat')
ts = f11.measure(measure='ts', result='abs')
sp = f11.measure(measure='sp', result='abs')
ta = f11.measure(measure='ta', result='abs')
print ("Test")
print (f"True Statistical Parity: {ts}, Statistical Parity: {sp}, Total Accuracy: {ta}")
tp = d11[(d11['y_hat'] == 1) & (d11[success_attribute] == 1)].shape[0]
tn = d11[(d11['y_hat'] == 0) & (d11[success_attribute] == 0)].shape[0]
accuracy = (tp + tn) / d11.shape[0]
print(f"Accuracy: {accuracy}")

d12 = X_train.copy()
d12['y_hat'] = predictions_train
f12 = fm(dataset=d12, sensible_attribute=sensible_attribute, true_success=success_attribute, prediction='y_hat')
ts = f12.measure(measure='ts', result='abs')
sp = f12.measure(measure='sp', result='abs')
ta = f12.measure(measure='ta', result='abs')
print ("Train")
print (f"True Statistical Parity: {ts}, Statistical Parity: {sp}, Total Accuracy: {ta}")
tp = d12[(d12['y_hat'] == 1) & (d12[success_attribute] == 1)].shape[0]
tn = d12[(d12['y_hat'] == 0) & (d12[success_attribute] == 0)].shape[0]
accuracy = (tp + tn) / d12.shape[0]
print(f"Accuracy: {accuracy}")
print("Tree statistics")
print(f"Feature importance: {sft.get_feature_importance()}")
sft.print_tree_by_level()


f = fm(dataset=X_train, sensible_attribute=sensible_attribute, true_success=success_attribute, prediction='y_hat')
sft.fit(X_train, success_attribute, fairness_measures=f, measure='sp', fair_weight=0.1)

predictions_test = sft.predict(X_test)
predictions_train = sft.predict(X_train)
print("Simple FAIR tree")
d21 = X_test.copy()
d21['y_hat'] = predictions_test
f21 = fm(dataset=d21, sensible_attribute=sensible_attribute, true_success=success_attribute, prediction='y_hat')
ts = f21.measure(measure='ts', result='abs')
sp = f21.measure(measure='sp', result='abs')
ta = f21.measure(measure='ta', result='abs')
print ("Test")
print (f"True Statistical Parity: {ts}, Statistical Parity: {sp}, Total Accuracy: {ta}")
tp = d21[(d21['y_hat'] == 1) & (d21[success_attribute] == 1)].shape[0]
tn = d21[(d21['y_hat'] == 0) & (d21[success_attribute] == 0)].shape[0]
accuracy = (tp + tn) / d21.shape[0]
print(f"Accuracy: {accuracy}")

d22 = X_train.copy()
d22['y_hat'] = predictions_train
f22 = fm(dataset=d22, sensible_attribute=sensible_attribute, true_success=success_attribute, prediction='y_hat')
ts = f22.measure(measure='ts', result='abs')
sp = f22.measure(measure='sp', result='abs')
ta = f22.measure(measure='ta', result='abs')
print ("Train")
print (f"True Statistical Parity: {ts}, Statistical Parity: {sp}, Total Accuracy: {ta}")
tp = d22[(d22['y_hat'] == 1) & (d22[success_attribute] == 1)].shape[0]
tn = d22[(d22['y_hat'] == 0) & (d22[success_attribute] == 0)].shape[0]
accuracy = (tp + tn) / d22.shape[0]
print(f"Accuracy: {accuracy}")
print("Tree statistics")
print(f"Feature importance: {sft.get_feature_importance()}")
sft.print_tree_by_level()