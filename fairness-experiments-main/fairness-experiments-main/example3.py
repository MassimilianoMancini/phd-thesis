import pandas as pd
from fairnessMeasuresAPI import fm
from SimpleFairTree import SimpleFairTree
from sklearn.model_selection import train_test_split
import numpy as np
import kagglehub


# Download latest version of law school dataset
path = kagglehub.dataset_download("uciml/adult-census-income")
file = path + "/adult.csv"

# Define fairness measures parameters
sensible_attribute = 'sex'
success_attribute = 'income'

dataset = pd.read_csv(file, sep=',', header='infer')

dataset['income'] = dataset['income'].apply(lambda x: 1 if x == '>50K' else 0)

dataset = dataset[dataset['sex'].notna()]

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
ts = f11.measure(measure='ts')
sp = f11.measure(measure='sp')
ta = f11.measure(measure='ta')
print ("Test")
print (f"True Statistical Parity: {ts}, Statistical Parity: {sp}, Total Accuracy: {ta}")
tp = d11[(d11['y_hat'] == 1) & (d11[success_attribute] == 1)].shape[0]
tn = d11[(d11['y_hat'] == 0) & (d11[success_attribute] == 0)].shape[0]
accuracy = (tp + tn) / d11.shape[0]
print(f"Accuracy: {accuracy}")

d12 = X_train.copy()
d12['y_hat'] = predictions_train
f12 = fm(dataset=d12, sensible_attribute=sensible_attribute, true_success=success_attribute, prediction='y_hat')
ts = f12.measure(measure='ts')
sp = f12.measure(measure='sp')
ta = f12.measure(measure='ta')
print ("Train")
print (f"True Statistical Parity: {ts}, Statistical Parity: {sp}, Total Accuracy: {ta}")
tp = d12[(d12['y_hat'] == 1) & (d12[success_attribute] == 1)].shape[0]
tn = d12[(d12['y_hat'] == 0) & (d12[success_attribute] == 0)].shape[0]
accuracy = (tp + tn) / d12.shape[0]
print(f"Accuracy: {accuracy}")


f = fm(dataset=X_train, sensible_attribute=sensible_attribute, true_success=success_attribute, prediction='y_hat', sigma=0.05, tau=0.95, gamma=0.0)

sft.fit(X_train, success_attribute, fairness_measures=f)

predictions_test = sft.predict(X_test)
predictions_train = sft.predict(X_train)
print("Simple FAIR tree")
d21 = X_test.copy()
d21['y_hat'] = predictions_test
f21 = fm(dataset=d21, sensible_attribute=sensible_attribute, true_success=success_attribute, prediction='y_hat')
ts = f21.measure(measure='ts')
sp = f21.measure(measure='sp')
ta = f21.measure(measure='ta')
print ("Test")
print (f"True Statistical Parity: {ts}, Statistical Parity: {sp}, Total Accuracy: {ta}")
tp = d21[(d21['y_hat'] == 1) & (d21[success_attribute] == 1)].shape[0]
tn = d21[(d21['y_hat'] == 0) & (d21[success_attribute] == 0)].shape[0]
accuracy = (tp + tn) / d21.shape[0]
print(f"Accuracy: {accuracy}")

d22 = X_train.copy()
d22['y_hat'] = predictions_train
f22 = fm(dataset=d22, sensible_attribute=sensible_attribute, true_success=success_attribute, prediction='y_hat')
ts = f22.measure(measure='ts')
sp = f22.measure(measure='sp')
ta = f22.measure(measure='ta')
print ("Train")
print (f"True Statistical Parity: {ts}, Statistical Parity: {sp}, Total Accuracy: {ta}")
tp = d22[(d22['y_hat'] == 1) & (d22[success_attribute] == 1)].shape[0]
tn = d22[(d22['y_hat'] == 0) & (d22[success_attribute] == 0)].shape[0]
accuracy = (tp + tn) / d22.shape[0]
print(f"Accuracy: {accuracy}")