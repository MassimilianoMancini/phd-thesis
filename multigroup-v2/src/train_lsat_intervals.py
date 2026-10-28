import pandas as pd
import kagglehub
import numpy as np
from fairnessMeasuresAPI import fm
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, top_k_accuracy_score
import matplotlib
matplotlib.use('Agg')

def success_func(r_hat):
    return 1 if r_hat > 3 else 0

def categorize_lsat(lsat):
    """
    Categorizza i punteggi LSAT in 6 classi:
    - 3 equidistanti sotto 29
    - 3 equidistanti da 29.01 in su
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

group = 'race1'
subset = 'X_test'

toDrop = ['decile1b', 'decile3', 'ID', 'race', 'lsat', 'zfygpa', 'grad', 'zgpa', 'bar1', 'bar1_yr', 
          'bar2', 'bar2_yr', 'age', 'gender', 'male', 'race2', 'bar', 'Dropout', 'other', 'asian', 
          'black', 'hisp', 'pass_bar', 'bar_passed', 'tier', 'index6040', 'indxgrp', 'indxgrp2', 
          'gpa', 'dnn_bar_pass_prediction']

# Leggi dataset una volta
base_dataset = pd.read_csv(file, sep=',', header='infer')
base_dataset['DOB_yr'] = base_dataset['DOB_yr'].apply(lambda x: x + 1900)

# Risultati
results = []

# Test solo per n_classes = 6
print(f"\n{'='*60}")
print(f"Testing con 6 intervalli equidistanti (TRAINING SET)")
print(f"3 sotto 29 + 3 da 29.01 in su")
print(f"{'='*60}")

# Copia il dataset
dataset = base_dataset.copy()

# Crea colonna 'r' con categorizzazione LSAT
dataset['r'] = dataset['lsat'].apply(categorize_lsat)

# Filtra e prepara
dataset = dataset.drop(toDrop, axis=1)
dataset.dropna(inplace=True)

score = 'r'

# Prepara i dati
X = pd.get_dummies(dataset.drop(score, axis=1))
le = LabelEncoder()
y_encoded = le.fit_transform(dataset[score])

n_classes_actual = len(le.classes_)
algorithm = XGBClassifier(eval_metric='mlogloss', random_state=42)

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Train
algorithm.fit(X_train, y_train)
predictions_test = algorithm.predict(X_test)
predictions_train = algorithm.predict(X_train)

# Convert predictions back
predictions_test_orig = le.inverse_transform(predictions_test)
predictions_train_orig = le.inverse_transform(predictions_train)

# Prepara i dati per il calcolo delle metriche (usando TRAINING SET)
X_train_copy = X_train.copy()
X_train_copy['r_hat'] = predictions_train_orig
X_train_copy[score] = dataset.loc[X_train.index][score]
X_train_copy['y_hat'] = X_train_copy['r_hat'].apply(success_func)
X_train_copy['y'] = X_train_copy[score].apply(success_func)

# Accuracy
r_accuracy = accuracy_score(X_train_copy['r_hat'], X_train_copy[score])
y_accuracy = accuracy_score(X_train_copy['y'], X_train_copy['y_hat'])

print(f"Classes: {le.classes_}")
print(f"Number of classes: {n_classes_actual}")
print(f"R-Accuracy (score prediction): {r_accuracy:.4f}")
print(f"Y-Accuracy (success prediction): {y_accuracy:.4f}")

results.append({
    'n_classes': 6,
    'n_actual_classes': n_classes_actual,
    'r_accuracy': r_accuracy,
    'y_accuracy': y_accuracy
})

# Stampa riepilogo
print(f"\n{'='*60}")
print("RISULTATI FINALI")
print(f"{'='*60}\n")

# Stampa intervalli
lower_min = 11
lower_max = 29
lower_range = lower_max - lower_min
lower_step = lower_range / 3

upper_min = 29.01
upper_max = 48
upper_range = upper_max - upper_min
upper_step = upper_range / 3

print("INTERVALLI:")
print(f"Classe 1: [{lower_min:.2f}, {lower_min + lower_step:.2f})")
print(f"Classe 2: [{lower_min + lower_step:.2f}, {lower_min + 2*lower_step:.2f})")
print(f"Classe 3: [{lower_min + 2*lower_step:.2f}, {lower_max:.2f})")
print(f"Classe 4: [{upper_min:.2f}, {upper_min + upper_step:.2f})")
print(f"Classe 5: [{upper_min + upper_step:.2f}, {upper_min + 2*upper_step:.2f})")
print(f"Classe 6: [{upper_min + 2*upper_step:.2f}, {upper_max:.2f}]")
print()

result = results[0]
print(f"N. Classi: {int(result['n_classes'])}")
print(f"R-Accuracy: {result['r_accuracy']:.4f}")
print(f"Y-Accuracy: {result['y_accuracy']:.4f}")
print(f"{'='*60}\n")
