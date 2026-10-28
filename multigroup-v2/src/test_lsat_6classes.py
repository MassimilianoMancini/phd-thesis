import pandas as pd
import kagglehub
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import matplotlib
matplotlib.use('Agg')
from itertools import combinations

def categorize_lsat_custom(lsat, split_points):
    """
    Categorizza LSAT usando split_points.
    split_points è una lista ordinata di valori di split.
    La classe è determinata da quale intervallo cade il lsat.
    """
    if pd.isna(lsat):
        return None
    
    for i, point in enumerate(split_points):
        if lsat < point:
            return i + 1
    return len(split_points) + 1

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

print(f"{'='*80}")
print(f"Testing approfondito con n_classes = 6")
print(f"{'='*80}\n")

n_classes = 6
half_classes = 3  # 3 sotto 29, 3 sopra/compreso 29

# Per 6 classi abbiamo bisogno di 5 split points: 2 sotto 29, 29, 2 sopra 29
# Generiamo una griglia fitta di candidati
lower_candidates = np.arange(12, 29, 0.5)  # Split sotto 29
upper_candidates = np.arange(29.5, 48, 0.5)  # Split sopra 29

results = []
test_count = 0
max_tests = 200  # Numero massimo di test

print(f"Testando combinazioni di split points...")
print(f"Split points sotto 29: {len(lower_candidates)} candidati")
print(f"Split points sopra 29: {len(upper_candidates)} candidati\n")

# Genera combinazioni di 2 split points sotto 29 e 2 sopra 29
for lower_combo in combinations(lower_candidates, 2):
    if test_count >= max_tests:
        break
    
    for upper_combo in combinations(upper_candidates, 2):
        if test_count >= max_tests:
            break
        
        # Crea split_points ordinati
        split_points = sorted(list(lower_combo) + [29] + list(upper_combo))
        
        # Copia il dataset
        dataset = base_dataset.copy()
        
        # Crea colonna 'r' con categorizzazione customizzata
        dataset['r'] = dataset['lsat'].apply(lambda x: categorize_lsat_custom(x, split_points))
        
        # Filtra e prepara
        dataset = dataset.drop(toDrop, axis=1)
        dataset.dropna(inplace=True)
        
        score = 'r'
        
        try:
            # Prepara i dati
            X = pd.get_dummies(dataset.drop(score, axis=1))
            le = LabelEncoder()
            y_encoded = le.fit_transform(dataset[score])
            
            n_classes_actual = len(le.classes_)
            
            # Verifica che abbiamo le classi attese
            if n_classes_actual != n_classes:
                continue
            
            algorithm = XGBClassifier(eval_metric='mlogloss', random_state=42)
            
            # Split train/test
            X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
            
            # Train
            algorithm.fit(X_train, y_train)
            predictions_test = algorithm.predict(X_test)
            
            # Convert predictions back
            predictions_test_orig = le.inverse_transform(predictions_test)
            
            # Prepara i dati per il calcolo delle metriche
            X_test_copy = X_test.copy()
            X_test_copy['r_hat'] = predictions_test_orig
            X_test_copy[score] = dataset.loc[X_test.index][score]
            
            # R-Accuracy
            r_accuracy = accuracy_score(X_test_copy['r_hat'], X_test_copy[score])
            
            results.append({
                'split_points': tuple(np.round(split_points, 2)),
                'split_points_str': str(np.round(split_points, 2)),
                'r_accuracy': r_accuracy,
                'lower_1': split_points[0],
                'lower_2': split_points[1],
                'upper_1': split_points[3],
                'upper_2': split_points[4]
            })
            
            test_count += 1
            
            if test_count % 50 == 0:
                print(f"Test completati: {test_count}")
                
        except Exception as e:
            continue

print(f"\nTest totali eseguiti: {test_count}\n")

if results:
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values('r_accuracy', ascending=False)
    
    print(f"{'='*80}")
    print("TOP 10 CONFIGURAZIONI")
    print(f"{'='*80}\n")
    
    for idx, (i, row) in enumerate(df_results.head(10).iterrows(), 1):
        print(f"{idx}. R-Accuracy: {row['r_accuracy']:.4f}")
        print(f"   Split points: {row['split_points_str']}")
        print(f"   Intervalli:")
        print(f"     Classe 1: LSAT < {row['lower_1']:.2f}")
        print(f"     Classe 2: {row['lower_1']:.2f} ≤ LSAT < {row['lower_2']:.2f}")
        print(f"     Classe 3: {row['lower_2']:.2f} ≤ LSAT < 29")
        print(f"     Classe 4: 29 ≤ LSAT < {row['upper_1']:.2f}")
        print(f"     Classe 5: {row['upper_1']:.2f} ≤ LSAT < {row['upper_2']:.2f}")
        print(f"     Classe 6: LSAT ≥ {row['upper_2']:.2f}")
        print()
    
    best = df_results.iloc[0]
    print(f"{'='*80}")
    print("CONFIGURAZIONE MIGLIORE")
    print(f"{'='*80}")
    print(f"R-Accuracy: {best['r_accuracy']:.4f}")
    print(f"Split points: {best['split_points_str']}")
    print(f"{'='*80}\n")
