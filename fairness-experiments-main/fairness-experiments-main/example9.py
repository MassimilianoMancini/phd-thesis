import pandas as pd
from fairnessMeasuresAPI import fm
from SimpleFairTree import SimpleFairTree
from sklearn.model_selection import train_test_split
import numpy as np

# Number of samples
N = 20

# For reproducibility
np.random.seed(4)

# Create a dataset
dataset = pd.DataFrame()

# Populate the dataset with 10 males (0) and 10 females (1)
genders = [0]*10 + [1]*10
dataset['gender'] = genders


# Add 'degree_score' (final degree grade) with normal distribution centered at 2.5
dataset['degree_score'] = np.random.normal(loc=2.5, scale=0.8, size=N).round().astype(int)
# Ensure 'degree_score' is between 1 and 4
dataset['degree_score'] = dataset['degree_score'].clip(1, 4)

# Assign 'success' with different probability functions for gender
degree_min = dataset['degree_score'].min()
degree_max = dataset['degree_score'].max()
degree_norm = (dataset['degree_score'] - degree_min) / (degree_max - degree_min)

# Success probability
success_probs = np.clip(0.3 + 0.7 * degree_norm, 0, 1)

dataset['success'] = (np.random.rand(N) < success_probs).astype(int)


# Define fairness measures parameters
sensible_attribute = 'gender'
success_attribute = 'success'
sft = SimpleFairTree(max_depth=5)

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
ts = f11.measure(measure='ts', result='matrix')
sp = f11.measure(measure='sp', result='matrix')
ta = f11.measure(measure='ta', result='matrix')
print ("Test")
print (f"True Statistical Parity: {ts}, Statistical Parity: {sp}, Total Accuracy: {ta}")
tp = d11[(d11['y_hat'] == 1) & (d11[success_attribute] == 1)].shape[0]
tn = d11[(d11['y_hat'] == 0) & (d11[success_attribute] == 0)].shape[0]
accuracy = (tp + tn) / d11.shape[0]
print(f"Accuracy: {accuracy}")

d12 = X_train.copy()
d12['y_hat'] = predictions_train
f12 = fm(dataset=d12, sensible_attribute=sensible_attribute, true_success=success_attribute, prediction='y_hat')
ts = f12.measure(measure='ts', result='matrix')
sp = f12.measure(measure='sp', result='matrix')
ta = f12.measure(measure='ta', result='matrix')
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
sft.fit(X_train, success_attribute, fairness_measures=f, measure='sp', fair_weight=1)

predictions_test = sft.predict(X_test)
predictions_train = sft.predict(X_train)
print("Simple FAIR tree")
d21 = X_test.copy()
d21['y_hat'] = predictions_test
f21 = fm(dataset=d21, sensible_attribute=sensible_attribute, true_success=success_attribute, prediction='y_hat')
ts = f21.measure(measure='ts', result='matrix')
sp = f21.measure(measure='sp', result='matrix')
ta = f21.measure(measure='ta', result='matrix')
print ("Test")
print (f"True Statistical Parity: {ts}, Statistical Parity: {sp}, Total Accuracy: {ta}")
tp = d21[(d21['y_hat'] == 1) & (d21[success_attribute] == 1)].shape[0]
tn = d21[(d21['y_hat'] == 0) & (d21[success_attribute] == 0)].shape[0]
accuracy = (tp + tn) / d21.shape[0]
print(f"Accuracy: {accuracy}")

d22 = X_train.copy()
d22['y_hat'] = predictions_train
f22 = fm(dataset=d22, sensible_attribute=sensible_attribute, true_success=success_attribute, prediction='y_hat')
ts = f22.measure(measure='ts', result='matrix')
sp = f22.measure(measure='sp', result='matrix')
ta = f22.measure(measure='ta', result='matrix')
print ("Train")
print (f"True Statistical Parity: {ts}, Statistical Parity: {sp}, Total Accuracy: {ta}")
tp = d22[(d22['y_hat'] == 1) & (d22[success_attribute] == 1)].shape[0]
tn = d22[(d22['y_hat'] == 0) & (d22[success_attribute] == 0)].shape[0]
accuracy = (tp + tn) / d22.shape[0]
print(f"Accuracy: {accuracy}")
print("Tree statistics")
print(f"Feature importance: {sft.get_feature_importance()}")
sft.print_tree_by_level()
