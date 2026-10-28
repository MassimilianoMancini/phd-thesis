import pandas as pd
from fairnessMeasuresAPI import fm
from SimpleFairTree import SimpleFairTree
from sklearn.model_selection import train_test_split
import numpy as np


# Number of samples
N = 20

# For reproducibility
np.random.seed(1)

# Create a dataset
dataset = pd.DataFrame()

# Populate the dataset with half males (0) and half females (1)
genders = [0]*(N//2) + [1]*(N//2)
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

# Shuffle the dataset to ensure randomness
dataset = dataset.sample(frac=1).reset_index(drop=True)

# Define fairness measures parameters
sensible_attributes = ['gender', 'race']
success_attribute = 'success'

# Success probability: even lower overall, and larger gap between males and females
success_probs = np.where(
    dataset['gender'] == 0,
    0.4 + 0.6 * degree_norm,   # Much lower base probability for males
    0.2 + 0.4 * degree_norm    # Much lower base probability for females
)
success_probs = np.clip(success_probs, 0, 1)

dataset['success'] = (np.random.rand(N) < success_probs).astype(int)
X = dataset
y = dataset[success_attribute]


# Create a table with counts of gender and success
count_table = dataset.groupby(['gender', 'success']).size().unstack(fill_value=0)
print(count_table)

for sft in (SimpleFairTree(max_depth=5), SimpleFairTree(max_depth=5, splitter='fair', fairness_measure='sp', fairness_weights=[0.7, 0.3], fairness_attributes=sensible_attributes, random_state=1)):

    sft.fit(X, success_attribute)
    predictions = sft.predict(X)
    d = X.copy()
    d['y_hat'] = predictions
    d['y_hat'] = d['y_hat'].astype(int)
    print(f"SimpleFairTree: {sft.splitter}")
    for sensible_attribute in sensible_attributes:

        f = fm(dataset=d, sensible_attribute=sensible_attribute, true_success=success_attribute, prediction='y_hat')
        ts = f.measure(measure='ts', result='matrix')
        sp = f.measure(measure='sp', result='matrix')
        ta = f.measure(measure='ta', result='matrix')
        tp = d[(d['y_hat'] == 1) & (d[success_attribute] == 1)].shape[0]
        tn = d[(d['y_hat'] == 0) & (d[success_attribute] == 0)].shape[0]
        accuracy = (tp + tn) / d.shape[0]


        print(f"\tSensible Attribute: {sensible_attribute}")
        print(f"\t\tAccuracy: {accuracy:.4f}")
        print(f"\t\tTS: {ts:.4f}, SP: {sp:.4f}, TA: {ta:.4f}")
