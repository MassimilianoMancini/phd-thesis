import pandas as pd
import kagglehub
import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

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
    
    # Range sotto 29: da 11 a 28.999...
    lower_min = 11
    lower_max = 29
    lower_range = lower_max - lower_min
    lower_step = lower_range / 3
    
    # Range da 29.01 in su: da 29.01 a 48
    upper_min = 29.01
    upper_max = 48
    upper_range = upper_max - upper_min
    upper_step = upper_range / 3
    
    if lsat < 29:
        # Calcola quale classe sotto 29 (1 a 3)
        class_idx = int((lsat - lower_min) / lower_step)
        class_idx = min(class_idx, 2)
        return class_idx + 1
    else:
        # Calcola quale classe da 29.01 in su (4 a 6)
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
print("ADDESTRAMENTO SU TUTTO IL DATASET")
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

print("=" * 100)
print("R-ACCURACY (Predizione della categorizzazione LSAT)")
print("=" * 100)

# Addestra modello R su TUTTO il dataset
model_r = XGBClassifier(eval_metric='mlogloss', random_state=42)
model_r.fit(X, y_r)

# Predizioni su tutto il dataset
y_r_pred = model_r.predict(X)
r_accuracy = accuracy_score(y_r, y_r_pred)

print(f"\nR-Accuracy (su tutto il dataset): {r_accuracy:.4f}")

print("\n" + "=" * 100)
print("Y-ACCURACY (Predizione del bar pass)")
print("=" * 100)

# Addestra modello Y su TUTTO il dataset
model_y = XGBClassifier(eval_metric='logloss', random_state=42)
model_y.fit(X, y_y)

# Predizioni su tutto il dataset
y_y_pred = model_y.predict(X)
y_accuracy = accuracy_score(y_y, y_y_pred)

print(f"\nY-Accuracy (su tutto il dataset): {y_accuracy:.4f}")

print("\n" + "=" * 100)
print("RIEPILOGO FINALE")
print("=" * 100)

print("\nIntervalli LSAT:")
lower_min = 11
lower_max = 29
lower_range = lower_max - lower_min
lower_step = lower_range / 3

upper_min = 29.01
upper_max = 48
upper_range = upper_max - upper_min
upper_step = upper_range / 3

print(f"  Classe 1: [{lower_min:.2f}, {lower_min + lower_step:.2f})")
print(f"  Classe 2: [{lower_min + lower_step:.2f}, {lower_min + 2*lower_step:.2f})")
print(f"  Classe 3: [{lower_min + 2*lower_step:.2f}, {lower_max:.2f})")
print(f"  Classe 4: [{upper_min:.2f}, {upper_min + upper_step:.2f})")
print(f"  Classe 5: [{upper_min + upper_step:.2f}, {upper_min + 2*upper_step:.2f})")
print(f"  Classe 6: [{upper_min + 2*upper_step:.2f}, {upper_max:.2f}]")

print(f"\nR-Accuracy (Training): {r_accuracy:.4f}")
print(f"Y-Accuracy (Training): {y_accuracy:.4f}")
print(f"Average:              {(r_accuracy + y_accuracy) / 2:.4f}")

print("\n" + "=" * 100)
print("\nNote:")
print("- Il modello è addestraito su TUTTO il dataset")
print("- L'accuracy è calcolata sugli STESSI dati di addestramento")
print("- Questi sono i valori massimi raggiungibili (upper bound)")
print("- Per una valutazione realistica, usare cross-validation o train/test split")
print("=" * 100)
