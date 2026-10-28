import pandas as pd
import kagglehub
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from SimpleMLP import SimpleMLPClassifier


def success_func(r_hat):
    """Converte r (categorizzazione LSAT) in y (pass/fail)"""
    return 1 if r_hat > 3 else 0


def categorize_lsat(lsat):
    """
    Categorizza i punteggi LSAT in 6 classi:
    - 3 equidistanti sotto 29
    - 3 equidistanti da 29 in su
    Numerate da 1 a 6.
    """
    if pd.isna(lsat):
        return None
    
    # Range sotto 29: da 10 a 29
    lower_min = 10
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


# Download dataset
path = kagglehub.dataset_download("danofer/law-school-admissions-bar-passage")
file = path + "/bar_pass_prediction.csv"

toDrop = ['decile1b', 'decile3', 'ID', 'race', 'lsat', 'zfygpa', 'grad', 'zgpa', 'bar1', 'bar1_yr', 
          'bar2', 'bar2_yr', 'age', 'gender', 'male', 'race2', 'bar', 'Dropout', 'other', 'asian', 
          'black', 'hisp', 'pass_bar', 'bar_passed', 'tier', 'index6040', 'indxgrp', 'indxgrp2', 
          'gpa', 'dnn_bar_pass_prediction']

# Leggi dataset
base_dataset = pd.read_csv(file, sep=',', header='infer')
base_dataset['DOB_yr'] = base_dataset['DOB_yr'].apply(lambda x: x + 1900)

print("=" * 100)
print("VALUTAZIONE CON CROSS-VALIDATION (5-FOLD)")
print("6 intervalli: 3 sotto 29 + 3 da 29 in su")
print("=" * 100 + "\n")

# Copia dataset
dataset = base_dataset.copy()

# Crea colonna 'r' con categorizzazione LSAT
dataset['r'] = dataset['lsat'].apply(categorize_lsat)

# Filtra colonne
dataset = dataset.drop(toDrop, axis=1)
dataset.dropna(inplace=True)

print(f"Dataset size: {len(dataset)}")
print(f"Distribution of r (LSAT classes):")
print(dataset['r'].value_counts().sort_index())
print()

# Prepara dati
X = pd.get_dummies(dataset.drop('r', axis=1))
y_r = (dataset['r'].astype(int) - 1).values  # Classi da 0 a 5
y_y = dataset['r'].apply(success_func).values

print(f"Features: {X.shape[1]}")
print(f"Samples: {X.shape[0]}\n")

# Setup cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("=" * 100)
print("R-ACCURACY (Predizione della categorizzazione LSAT) - SimpleMLP")
print("=" * 100)

# SimpleMLPClassifier model
model_r = SimpleMLPClassifier(seed=42, verbose=False)

# lists for metrics
r_acc_train = []
r_acc_val = []

y_acc_val_list = []

print(f"\nFold results (R-Accuracy, Y-Accuracy):")
for fold, (train_idx, test_idx) in enumerate(cv.split(X, y_r)):
    X_train, X_test = X.iloc[train_idx].values, X.iloc[test_idx].values
    y_train, y_test = y_r[train_idx], y_r[test_idx]

    # train model on this fold
    model_r.fit(X_train, y_train)

    # predictions for r
    y_pred_train = model_r.predict(X_train)
    y_pred_val = model_r.predict(X_test)

    # r metrics
    acc_tr = accuracy_score(y_train, y_pred_train)
    acc_val = accuracy_score(y_test, y_pred_val)

    r_acc_train.append(acc_tr)
    r_acc_val.append(acc_val)

    # compute y predictions via success_func
    y_true = y_y[test_idx]
    y_pred = np.array([success_func(int(r) + 1) for r in y_pred_val])
    ya = accuracy_score(y_true, y_pred)

    y_acc_val_list.append(ya)

    print(f"  Fold {fold+1}: Train={acc_tr:.4f}", end='')
    print(f", Val={acc_val:.4f}", end='')
    print(f" | Y-Val={ya:.4f}")

# convert to arrays for aggregation
r_acc_train = np.array(r_acc_train)
r_acc_val = np.array(r_acc_val)

y_acc_val = np.array(y_acc_val_list)

print(f"\nR-Accuracy (mean): {r_acc_val.mean():.4f} ± {r_acc_val.std():.4f}")
print(f"Train Score (mean acc): {r_acc_train.mean():.4f}")


print("\n" + "=" * 100)
print("RIEPILOGO FINALE")
print("=" * 100)

print("\nIntervalli LSAT:")
lower_min = 10
lower_max = 29
lower_range = lower_max - lower_min
lower_step = lower_range / 3

upper_min = 29
upper_max = 48
upper_range = upper_max - upper_min
upper_step = upper_range / 3

print(f"  Classe 1: [{lower_min:.2f}, {lower_min + lower_step:.2f})")
print(f"  Classe 2: [{lower_min + lower_step:.2f}, {lower_min + 2*lower_step:.2f}]")
print(f"  Classe 3: [{lower_min + 2*lower_step:.2f}, {lower_max:.2f}]")
print(f"  Classe 4: ({upper_min:.2f}, {upper_min + upper_step:.2f})")
print(f"  Classe 5: [{upper_min + upper_step:.2f}, {upper_min + 2*upper_step:.2f})")
print(f"  Classe 6: [{upper_min + 2*upper_step:.2f}, {upper_max:.2f}]")

print(f"\nR-Accuracy CV:  {r_acc_val.mean():.4f} ± {r_acc_val.std():.4f}")
print(f"Y-Accuracy CV:  {y_acc_val.mean():.4f} ± {y_acc_val.std():.4f}")

print("\n" + "=" * 100)
print("\nNote sulla Cross-Validation:")
print("- La CV non è addestrata su un set separato (train/test split)")
print("- Ogni iterazione usa k-1 fold per addestrare e 1 fold per validare")
print("- Il modello viene re-addestrato da zero in ogni fold")
print("- Il modello finale riportato è la media delle performance sui fold di validation")
print("=" * 100)
