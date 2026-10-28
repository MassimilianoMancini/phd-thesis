import pandas as pd
from fairnessMeasuresAPI import fm
from SimpleFairTree import SimpleFairTree
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

# Number of samples
N = 20000

# For reproducibility
np.random.seed(1)

# Create a dataset
dataset = pd.DataFrame()

# Populate the dataset with 10000 males (0) and 10000 females (1)
dataset['gender'] = np.random.uniform(size=N) < 0.5
dataset['gender'] = dataset['gender'].astype(int)

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


results = []

for i in np.arange(0.1, 2, 0.1):
    # Success probability: higher for males (gender=0), lower for females (gender=1)
    success_probs = np.where(
        dataset['gender'] == 0,
        0.3 + 0.7 * degree_norm,   # Higher base probability for males
        0.1 + 0.4 * degree_norm    # Lower base probability for females
    )
    success_probs = np.clip(success_probs, 0, 1)

    dataset['success'] = (np.random.rand(N) < success_probs).astype(int)
    X = pd.get_dummies(dataset)
    y = dataset[success_attribute]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    sft = SimpleFairTree(max_depth=5, splitter='fair', fair_measure='sp', fair_weight=i, fair_attribute=sensible_attribute, random_state=1)
    sft.fit(X_train, success_attribute)

    predictions_test = sft.predict(X_test)
    predictions_train = sft.predict(X_train)
    print(f"Simple tree with weight {i}")
    d21 = X_test.copy()
    d21['y_hat'] = predictions_test
    d21['y_hat'] = d21['y_hat'].astype(int)
    f21 = fm(dataset=d21, sensible_attribute=sensible_attribute, true_success=success_attribute, prediction='y_hat')
    ts1 = f21.measure(measure='ts', result='matrix')
    sp1 = f21.measure(measure='sp', result='matrix')
    ta1 = f21.measure(measure='ta', result='matrix')
    tp = d21[(d21['y_hat'] == 1) & (d21[success_attribute] == 1)].shape[0]
    tn = d21[(d21['y_hat'] == 0) & (d21[success_attribute] == 0)].shape[0]
    test_accuracy = (tp + tn) / d21.shape[0]

    d22 = X_train.copy()
    d22['y_hat'] = predictions_train
    f22 = fm(dataset=d22, sensible_attribute=sensible_attribute, true_success=success_attribute, prediction='y_hat')
    d22['y_hat'] = d22['y_hat'].astype(int)
    ts2 = f22.measure(measure='ts', result='matrix')
    sp2 = f22.measure(measure='sp', result='matrix')
    ta2 = f22.measure(measure='ta', result='matrix')
    tp = d22[(d22['y_hat'] == 1) & (d22[success_attribute] == 1)].shape[0]
    tn = d22[(d22['y_hat'] == 0) & (d22[success_attribute] == 0)].shape[0]
    train_accuracy = (tp + tn) / d22.shape[0]
    results.append({
        'fair_weight': i,
        'test_sp': sp1,
        'test_ts': ts1,
        'test_ta': ta1,
        'train_sp': sp2,
        'train_ts': ts2,
        'train_ta': ta2,
        'test_accuracy': test_accuracy,
        'train_accuracy': train_accuracy
    })

plt.figure(figsize=(10, 6))
plt.plot([r['fair_weight'] for r in results], [r['test_sp'] for r in results], label='Test SP', marker='o')
plt.plot([r['fair_weight'] for r in results], [r['train_sp'] for r in results], label='Train SP', marker='o')
plt.plot([r['fair_weight'] for r in results], [r['test_ts'] for r in results], label='Test TS', marker='o')
plt.plot([r['fair_weight'] for r in results], [r['train_ts'] for r in results], label='Train TS', marker='o')
plt.plot([r['fair_weight'] for r in results], [r['test_ta'] for r in results], label='Test TA', marker='o')
plt.plot([r['fair_weight'] for r in results], [r['train_ta'] for r in results], label='Train TA', marker='o')
plt.plot([r['fair_weight'] for r in results], [r['test_accuracy'] for r in results], label='Test Accuracy', marker='o')
plt.plot([r['fair_weight'] for r in results], [r['train_accuracy'] for r in results], label='Train Accuracy', marker='o')
plt.xlabel('Fair Weight')
plt.ylabel('Measure Value')
plt.title('Fairness Measures vs Fair Weight')
plt.legend()
plt.grid()
plt.savefig("fairness_measures_vs_fair_weight.png")  # Save the figure to a file
plt.show()