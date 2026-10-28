import pandas as pd
from fairnessMeasuresAPI import fm
from SimpleFairTree import SimpleFairTree
from sklearn.model_selection import train_test_split
import numpy as np


# Number of samples
N = 2000

# For reproducibility
np.random.seed(1)

# Create a dataset
dataset = pd.DataFrame()

# Populate the dataset with 1000 males (0) and 1000 females (1)
genders = [0]*1000 + [1]*1000
dataset['gender'] = genders

# Add 'race' with specified probabilities
race_categories = [0, 1]
race_probs = [0.55, 0.45]
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
sensible_attributes = ['gender', 'race']
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


for sft in (SimpleFairTree(max_depth=5), SimpleFairTree(max_depth=5, splitter='fair', fairness_measure='sp', fairness_weights=0.3, fairness_attributes=sensible_attributes, random_state=1)):

    sft.fit(X_train, success_attribute)
    predictions_test = sft.predict(X_test)
    predictions_train = sft.predict(X_train)
    d_test = X_test.copy()
    d_test['y_hat'] = predictions_test
    d_test['y_hat'] = d_test['y_hat'].astype(int)
    d_train = X_train.copy()
    d_train['y_hat'] = predictions_train
    for sensible_attribute in sensible_attributes:

        f_test = fm(dataset=d_test, sensible_attribute=sensible_attribute, true_success=success_attribute, prediction='y_hat')
        ts_test = f_test.measure(measure='ts', result='matrix')
        sp_test = f_test.measure(measure='sp', result='matrix')
        ta_test = f_test.measure(measure='ta', result='matrix')
        tp_test = d_test[(d_test['y_hat'] == 1) & (d_test[success_attribute] == 1)].shape[0]
        tn_test = d_test[(d_test['y_hat'] == 0) & (d_test[success_attribute] == 0)].shape[0]
        test_accuracy = (tp_test + tn_test) / d_test.shape[0]


        f_train = fm(dataset=d_train, sensible_attribute=sensible_attribute, true_success=success_attribute, prediction='y_hat')
        d_train['y_hat'] = d_train['y_hat'].astype(int)
        ts_train = f_train.measure(measure='ts', result='matrix')
        sp_train = f_train.measure(measure='sp', result='matrix')
        ta_train = f_train.measure(measure='ta', result='matrix')
        tp_train = d_train[(d_train['y_hat'] == 1) & (d_train[success_attribute] == 1)].shape[0]
        tn_train = d_train[(d_train['y_hat'] == 0) & (d_train[success_attribute] == 0)].shape[0]
        train_accuracy = (tp_train + tn_train) / d_train.shape[0]

        print(f"SimpleFairTree: {sft.splitter}")
        print(f"Sensible Attribute: {sensible_attribute}")
        print(f"Test Accuracy: {test_accuracy:.4f}, SP: {sp_test:.4f}, TS: {ts_test:.4f}, TA: {ta_test:.4f}")
        print(f"Train Accuracy: {train_accuracy:.4f}, SP: {sp_train:.4f}, TS: {ts_train:.4f}, TA: {ta_train:.4f}")
