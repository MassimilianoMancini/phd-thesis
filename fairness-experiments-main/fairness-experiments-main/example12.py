import pandas as pd
from fairnessMeasuresAPI import fm
from SimpleFairTree import SimpleFairTree
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

# Number of samples
N = 20

# For reproducibility
np.random.seed(1)

# Create a dataset
dataset = pd.DataFrame()

# Populate the dataset with 10 males (0) and 10 females (1)
genders = [0]*10 + [1]*10
dataset['gender'] = genders

# Add 'race' with specified probabilities
race_categories = ['caucasian', 'african', 'latin', 'other']
race_probs = [0.55, 0.20, 0.15, 0.10]
dataset['race'] = np.random.choice(race_categories, size=N, p=race_probs)

# Add 'degree_score' (final degree grade) with normal distribution centered at 90
dataset['degree_score'] = np.random.normal(loc=90, scale=5, size=N).round().astype(int)
# Ensure 'degree_score' is between 0 and 100
dataset['degree_score'] = dataset['degree_score'].clip(0, 100)

# Assign 'success' with different probability functions for gender
degree_min = dataset['degree_score'].min()
degree_max = dataset['degree_score'].max()
degree_norm = (dataset['degree_score'] - degree_min) / (degree_max - degree_min)


# Define fairness measures parameters
sensible_attribute = 'gender'
success_attribute = 'success'

# Success probability: higher for males (gender=0), lower for females (gender=1)
success_probs = np.where(
    dataset['gender'] == 0,
    0.3 + 0.6 * degree_norm,   # Higher base probability for males
    0.1 + 0.5 * degree_norm    # Lower base probability for females
)
success_probs = np.clip(success_probs, 0, 1)

dataset['success'] = (np.random.rand(N) < success_probs).astype(int)
X = pd.get_dummies(dataset)
y = dataset[success_attribute]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

sft1 = SimpleFairTree(max_depth=5)
sft1.fit(X_train, success_attribute)

predictions_test = sft1.predict(X_test)
predictions_train = sft1.predict(X_train)
d11 = X_test.copy()
d11['y_hat'] = predictions_test
d11['y_hat'] = d11['y_hat'].astype(int)
f21 = fm(dataset=d11, sensible_attribute=sensible_attribute, true_success=success_attribute, prediction='y_hat')
ts11 = f21.measure(measure='ts', result='matrix')
sp11 = f21.measure(measure='sp', result='matrix')
ta11 = f21.measure(measure='ta', result='matrix')
tp = d11[(d11['y_hat'] == 1) & (d11[success_attribute] == 1)].shape[0]
tn = d11[(d11['y_hat'] == 0) & (d11[success_attribute] == 0)].shape[0]
test_accuracy = (tp + tn) / d11.shape[0]

d12 = X_train.copy()
d12['y_hat'] = predictions_train
f22 = fm(dataset=d12, sensible_attribute=sensible_attribute, true_success=success_attribute, prediction='y_hat')
d12['y_hat'] = d12['y_hat'].astype(int)
ts12 = f22.measure(measure='ts', result='matrix')
sp12 = f22.measure(measure='sp', result='matrix')
ta12 = f22.measure(measure='ta', result='matrix')
tp = d12[(d12['y_hat'] == 1) & (d12[success_attribute] == 1)].shape[0]
tn = d12[(d12['y_hat'] == 0) & (d12[success_attribute] == 0)].shape[0]
train_accuracy = (tp + tn) / d12.shape[0]

sft2 = SimpleFairTree(max_depth=5, splitter='fair', fair_measure='sp', fair_weight=0.3, fair_attribute=sensible_attribute, random_state=1)
sft2.fit(X_train, success_attribute)

predictions_test = sft2.predict(X_test)
predictions_train = sft2.predict(X_train)
d21 = X_test.copy()
d21['y_hat'] = predictions_test
d21['y_hat'] = d21['y_hat'].astype(int)
f21 = fm(dataset=d21, sensible_attribute=sensible_attribute, true_success=success_attribute, prediction='y_hat')
ts21 = f21.measure(measure='ts', result='matrix')
sp21 = f21.measure(measure='sp', result='matrix')
ta21 = f21.measure(measure='ta', result='matrix')
tp = d21[(d21['y_hat'] == 1) & (d21[success_attribute] == 1)].shape[0]
tn = d21[(d21['y_hat'] == 0) & (d21[success_attribute] == 0)].shape[0]
test_accuracy_fair = (tp + tn) / d21.shape[0]

d22 = X_train.copy()
d22['y_hat'] = predictions_train
f22 = fm(dataset=d22, sensible_attribute=sensible_attribute, true_success=success_attribute, prediction='y_hat')
d22['y_hat'] = d22['y_hat'].astype(int)
ts22 = f22.measure(measure='ts', result='matrix')
sp22 = f22.measure(measure='sp', result='matrix')
ta22 = f22.measure(measure='ta', result='matrix')
tp = d22[(d22['y_hat'] == 1) & (d22[success_attribute] == 1)].shape[0]
tn = d22[(d22['y_hat'] == 0) & (d22[success_attribute] == 0)].shape[0]
train_accuracy_fair = (tp + tn) / d22.shape[0]

print(f"Test Accuracy: {test_accuracy:.4f}, SP: {sp11:.4f}, TS: {ts11:.4f}, TA: {ta11:.4f}")
print(f"Train Accuracy: {train_accuracy:.4f}, SP: {sp12:.4f}, TS: {ts12:.4f}, TA: {ta12:.4f}")

print(f"Test Accuracy Fair: {test_accuracy_fair:.4f}, SP: {sp21:.4f}, TS: {ts21:.4f}, TA: {ta21:.4f}")
print(f"Train Accuracy Fair: {train_accuracy_fair:.4f}, SP: {sp22:.4f}, TS: {ts22:.4f}, TA: {ta22:.4f}")
sft1.plot_tree(filename='sft1_tree.png')
sft2.plot_tree(filename='sft2_tree.png')
