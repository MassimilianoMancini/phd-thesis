import pandas as pd
import kagglehub
import numpy as np
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import random

# Reuse same helpers
def success_func(r_hat):
    return 1 if r_hat > 3 else 0

def categorize_lsat(lsat):
    if pd.isna(lsat):
        return None
    lower_min = 10
    lower_max = 29
    lower_range = lower_max - lower_min
    lower_step = lower_range / 3
    upper_min = 29
    upper_max = 48
    upper_range = upper_max - upper_min
    upper_step = upper_range / 3
    if lsat <= 29:
        class_idx = int((lsat - lower_min) / lower_step)
        class_idx = min(class_idx, 2)
        return class_idx + 1
    else:
        class_idx = int((lsat - upper_min) / upper_step)
        class_idx = min(class_idx, 2)
        return class_idx + 4

# Download dataset
path = kagglehub.dataset_download("danofer/law-school-admissions-bar-passage")
file = path + "/bar_pass_prediction.csv"

toDrop = ['decile1b', 'decile3', 'ID', 'race', 'lsat', 'zfygpa', 'grad', 'zgpa', 'bar1', 'bar1_yr', 
          'bar2', 'bar2_yr', 'age', 'gender', 'male', 'race2', 'bar', 'Dropout', 'other', 'asian', 
          'black', 'hisp', 'pass_bar', 'bar_passed', 'tier', 'index6040', 'indxgrp', 'indxgrp2', 
          'gpa', 'dnn_bar_pass_prediction']

base_dataset = pd.read_csv(file, sep=',', header='infer')
base_dataset['DOB_yr'] = base_dataset['DOB_yr'].apply(lambda x: x + 1900)

print('='*100)
print('TUNED MLP: nested RandomizedSearchCV (inner) + outer 5-fold CV')
print('='*100) 

# Prepare dataset
dataset = base_dataset.copy()
dataset['r'] = dataset['lsat'].apply(categorize_lsat)
dataset = dataset.drop(toDrop, axis=1)
dataset.dropna(inplace=True)

X = pd.get_dummies(dataset.drop('r', axis=1))
y_r = (dataset['r'].astype(int) - 1).values
y_y = dataset['r'].apply(success_func).values

print(f"Samples: {X.shape[0]}, Features: {X.shape[1]}")

# Outer CV
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Pipeline (scaler + MLP)
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('mlp', MLPClassifier(max_iter=500, random_state=42))
])

# Parameter grid for randomized search (search over pipeline param names)
param_dist = {
    'mlp__hidden_layer_sizes': [(128,64), (256,128), (256,128,64), (512,256,128)],
    'mlp__alpha': [1e-3, 5e-4, 1e-4, 5e-5, 1e-5],
    'mlp__learning_rate_init': [1e-3, 5e-4, 1e-4],
    'mlp__batch_size': [64, 128, 256],
    'mlp__learning_rate': ['constant', 'adaptive']
}

# RandomizedSearchCV settings
random_search = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=param_dist,
    n_iter=12,
    scoring='accuracy',
    cv=3,
    verbose=0,
    random_state=42,
    n_jobs=-1,
    refit=True
)

# Storage for outer results
r_acc_train = []
r_acc_val = []
y_acc_val_list = []

fold_idx = 0
for train_idx, test_idx in outer_cv.split(X, y_r):
    fold_idx += 1
    print(f"\n=== Outer fold {fold_idx} ===")
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y_r[train_idx], y_r[test_idx]

    # fit randomized search on the training fold (inner CV)
    random_search.fit(X_train, y_train)
    best = random_search.best_estimator_
    print(f"Best params (fold {fold_idx}): {random_search.best_params_}")

    # evaluate on train and test using best estimator
    y_pred_train = best.predict(X_train)
    y_pred_val = best.predict(X_test)

    acc_tr = accuracy_score(y_train, y_pred_train)
    acc_val = accuracy_score(y_test, y_pred_val)

    r_acc_train.append(acc_tr)
    r_acc_val.append(acc_val)

    # derive y from r predictions
    y_true = y_y[test_idx]
    y_pred_y = np.array([success_func(r + 1) for r in y_pred_val])
    ya = accuracy_score(y_true, y_pred_y)
    y_acc_val_list.append(ya)

    print(f"Fold {fold_idx}: Train acc={acc_tr:.4f}, Val acc={acc_val:.4f}, Y-Val acc={ya:.4f}")

# Aggregate
r_acc_train = np.array(r_acc_train)
r_acc_val = np.array(r_acc_val)
y_acc_val = np.array(y_acc_val_list)

print('\n=== Summary tuned MLP ===')
print(f"R-Accuracy CV: {r_acc_val.mean():.4f} ± {r_acc_val.std():.4f}")
print(f"Y-Accuracy CV: {y_acc_val.mean():.4f} ± {y_acc_val.std():.4f}")
print('Finished')
