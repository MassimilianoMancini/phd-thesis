import pandas as pd
import kagglehub
from fairnessMeasuresAPI import fm
from SimpleFairTree import SimpleFairTree
from sklearn.model_selection import train_test_split

# Download latest version of law school dataset
path = kagglehub.dataset_download("danofer/law-school-admissions-bar-passage")
file = path + "/bar_pass_prediction.csv"

# Define fairness measures parameters
sensible_attribute = 'male'
success_attribute = 'bar_passed'
subset = 'X_test'
sft = SimpleFairTree()

toDrop = ['decile1b', 'decile3', 'ID', 'decile1', 'race', 'cluster', 'lsat', 'ugpa', 'zfygpa', 'grad', 'zgpa', 'bar1', 'bar1_yr', 'bar2', 'bar2_yr', 'age', 'gender', 'race2', 'bar', 'Dropout', 'other', 'asian', 'black', 'hisp', 'pass_bar', 'tier', 'index6040',   'indxgrp', 'indxgrp2', 'gpa', 'dnn_bar_pass_prediction', 'gpa', 'lsat']

dataset = pd.read_csv(file, sep=',', header='infer')

# preprocess dataset
dataset = dataset.drop(toDrop, axis=1)
dataset = dataset[dataset[sensible_attribute].notna()]
dataset = dataset[dataset[success_attribute].notna()]
dataset['DOB_yr'] = dataset['DOB_yr'].apply(lambda x: x + 1900)


X = pd.get_dummies(dataset)
y = dataset[success_attribute]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
sft.fit(X_train, success_attribute)

predictions_test = sft.predict(X_test)
predictions_train = sft.predict(X_train)

f = fm(dataset=X_train, sensible_attribute=sensible_attribute, true_success=success_attribute, prediction='y_hat')

sft.fit(X_train, success_attribute, fairness_measures=f)

predictions_test = sft.predict(X_test)
predictions_train = sft.predict(X_train)

sp = f.measure(measure='ts')
ta = f.measure(measure='ta')