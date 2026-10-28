import pandas as pd
import kagglehub
import numpy as np
from fairnessMeasuresAPI import fm
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from seaborn import heatmap


# utility functions copied from the evaluation scripts
def success_func(r_hat):
    """Converte r (categorizzazione LSAT) in y (pass/fail)"""
    return 1 if r_hat > 3 else 0

def categorize_lsat(lsat):
    """
    Categorizza i punteggi LSAT in 6 classi:
    - 3 equidistanti sotto 29
    - 3 equidistanti da 29 in su
    Numerate da 1 a 6.
    (identica a quanto utilizzato in eval_lsat_intervals_cv_*.py)
    """
    if pd.isna(lsat):
        return None
    
    # Range sotto 29: da 11 a 29
    lower_min = 11
    lower_max = 29
    lower_range = lower_max - lower_min
    lower_step = lower_range / 3
    
    # Range da 29 in su: da 29 a 48
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

# download and prepare dataset
path = kagglehub.dataset_download("danofer/law-school-admissions-bar-passage")
file = path + "/bar_pass_prediction.csv"

# sensible attribute and subset used later by the fairness API
group = 'race1'

# variables dropped from the dataset

toDrop = [
    'decile1b', 'decile3', 'ID', 'race', 'lsat', 'zfygpa', 'grad', 'zgpa',
    'bar1', 'bar1_yr', 'bar2', 'bar2_yr', 'age', 'gender', 'male',
    'race2', 'bar', 'Dropout', 'other', 'asian', 'black', 'hisp',
    'pass_bar', 'bar_passed', 'tier', 'index6040', 'indxgrp', 'indxgrp2',
    'gpa', 'dnn_bar_pass_prediction'
]

# read and clean
dataset = pd.read_csv(file, sep=',', header='infer')
# calculate r and normalize DOB
dataset['r'] = dataset['lsat'].apply(categorize_lsat)
dataset['DOB_yr'] = dataset['DOB_yr'].apply(lambda x: x + 1900)

# drop unwanted fields and missing rows
dataset = dataset.drop(toDrop, axis=1)
dataset.dropna(inplace=True)


# prepare features/labels
score = 'r'
X = pd.get_dummies(dataset.drop(score, axis=1))
le = LabelEncoder()
y_encoded = le.fit_transform(dataset[score])

n_classes = len(le.classes_)

print(f'File is {file}')
print(f'Sensible attribute is {group}')
print(f'Score is {score}')
print(dataset.race1.value_counts())

# hold models with the same hyperparameters as in the evaluation scripts
models = {
    'dt': DecisionTreeClassifier(
        criterion='gini',
        max_depth=5,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42
    ),
    'xgb': XGBClassifier(
        objective='multi:softprob',
        num_class=n_classes,
        eval_metric='mlogloss',
        max_depth=4,
        learning_rate=0.05,
        n_estimators=300,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        gamma=0,
        reg_lambda=1,
        random_state=42
    ),
    'mlp': Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(
            hidden_layer_sizes=(256, 128, 64),
            activation='relu',
            solver='adam',
            alpha=5e-5,
            batch_size=128,
            learning_rate='adaptive',
            learning_rate_init=5e-4,
            max_iter=500,
            tol=1e-5,
            early_stopping=True,
            n_iter_no_change=20,
            random_state=42,
            verbose=False,
        ))
    ])
}

# cross-validation metrics using all data
# use plain KFold instead of StratifiedKFold for speed
cv = KFold(n_splits=5, shuffle=True, random_state=42)

# loop over algorithms and compute fairness measures plus statistics
for name, alg in models.items():
    
    # train on full dataset for fairness evaluation
    alg.fit(X, y_encoded)
    predictions_all = alg.predict(X)

    # restore original labels
    predictions_all_orig = le.inverse_transform(predictions_all)

    # copy full frame for fairness
    X_all = X.copy()
    X_all['r'] = dataset.loc[X_all.index][score]
    X_all['r_hat'] = predictions_all_orig
    X_all['y'] = X_all['r'].apply(success_func)
    X_all['y_hat'] = X_all['r_hat'].apply(success_func)

    X_all[group] = dataset.loc[X_all.index][group]

    f = fm(X_all, group, 'y', 'y_hat', 'r_hat')

    ts = f.measure(measure='ts', result='matrix')
    sp = f.measure(measure='sp', result='matrix')
    ta = f.measure(measure='ta', result='matrix')
    ca = f.measure(measure='ca', result='matrix')

    header = sorted(X_all[group].unique().tolist())
    pd.options.display.float_format = '{:,.3f}'.format

    print("==============================================")
    print(f'Algorithm is {name}')
    print(f'n_classes: {n_classes}')
    print("==============================================")

    print("Classes:")
    for index, value in enumerate(header):
        print(f"{index}:{value}")

    print("==============================================")

    # convenience loop for printing and plotting heatmaps
    for measure, matrix in [('ts', ts), ('sp', sp), ('ta', ta), ('ca', ca)]:
        if measure == 'ts':
            print('True statistical parity')
        elif measure == 'sp':
            print('Statistical parity')
        elif measure == 'ta':
            print('Total accuracy')
        elif measure == 'ca':
            print('Calibration')
        print('Pairwise matrix')
        print(pd.DataFrame(matrix))
        print(f.measure(measure=measure, result='latex'))
        plt.clf()
        hm = heatmap(matrix, cmap="coolwarm", cbar=False, annot=True, fmt=".3f",
                     xticklabels=header + ['1-vs-all'], yticklabels=header)
        hm.figure.savefig(f'{name}-{measure}.png')





    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y_encoded)):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y_encoded[train_idx], y_encoded[test_idx]
        alg.fit(X_tr, y_tr)
        pred_test = alg.predict(X_te)
        pred_train = alg.predict(X_tr)
        # convert r to y using success_func for accuracy calculation
        y_true_test = np.array([success_func(r+1) for r in y_te])
        y_pred_test = np.array([success_func(r+1) for r in pred_test])
        
        # r accuracy
        r_acc_test = accuracy_score(y_te, pred_test)
        r_acc_train = accuracy_score(y_tr, pred_train)

        # y accuracy using success_func
        y_true_train = np.array([success_func(r+1) for r in y_tr])
        y_pred_train = np.array([success_func(r+1) for r in pred_train])

        y_acc_test = accuracy_score(y_true_test, y_pred_test)
        y_acc_train = accuracy_score(y_true_train, y_pred_train)
        print(f"  Fold {fold+1}:", end='')
        print(f" R-Train={r_acc_train:.4f}, R-Test={r_acc_test:.4f}", end='')
        print(f" | Y-Train={y_acc_train:.4f}, Y-Test={y_acc_test:.4f}")


    print("==============================================")
    print()
