import pandas as pd
import kagglehub
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import matplotlib
matplotlib.use('Agg')

def success_func(r_hat):
    return 1 if r_hat > 3 else 0

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

# Risultati
all_results = []

# Test per n_classes da 4 a 8 (solo pari)
for n_classes in [4, 6, 8]:
    print(f"\n{'='*80}")
    print(f"Testing con n_classes = {n_classes}")
    print(f"{'='*80}\n")
    
    half_classes = n_classes // 2
    
    # Genera split points candidati
    # Sotto 29: split points tra 11 e 29
    lower_splits_candidates = np.linspace(11.5, 28.5, 10)
    # Sopra 29: split points tra 29 e 48
    upper_splits_candidates = np.linspace(29.5, 47.5, 10)
    
    best_accuracy = 0
    best_config = None
    results_for_this_n_classes = []
    
    # Genera combinazioni di split points
    # Per velocità, testiamo un subset ragionevole
    test_count = 0
    max_tests = 50  # Limita il numero di test per n_classes
    
    for lower_sp in [np.linspace(12, 28.5, half_classes + 1)[1:-1]]:
        for upper_sp in [np.linspace(29.5, 47.5, half_classes + 1)[1:-1]]:
            if test_count >= max_tests:
                break
            
            # Crea split_points: [lower_sp1, lower_sp2, ..., 29, upper_sp1, upper_sp2, ...]
            split_points = sorted(list(lower_sp) + [29] + list(upper_sp))
            
            # Rimuovi duplicati e ordina
            split_points = sorted(list(set(split_points)))
            
            # Verifica che 29 sia sempre un split point
            if 29 not in split_points:
                split_points.append(29)
                split_points = sorted(split_points)
            
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
                
                algorithm = XGBClassifier(eval_metric='mlogloss', random_state=42, verbose=0)
                
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
                
                results_for_this_n_classes.append({
                    'n_classes': n_classes,
                    'split_points': str(np.round(split_points, 2)),
                    'r_accuracy': r_accuracy
                })
                
                if r_accuracy > best_accuracy:
                    best_accuracy = r_accuracy
                    best_config = split_points
                
                test_count += 1
            except:
                continue
    
    # Aggiungi i risultati equidistanti (baseline)
    dataset = base_dataset.copy()
    from test_lsat_intervals import categorize_lsat
    dataset['r'] = dataset['lsat'].apply(lambda x: categorize_lsat(x, n_classes))
    dataset = dataset.drop(toDrop, axis=1)
    dataset.dropna(inplace=True)
    
    X = pd.get_dummies(dataset.drop('r', axis=1))
    le = LabelEncoder()
    y_encoded = le.fit_transform(dataset['r'])
    algorithm = XGBClassifier(eval_metric='mlogloss', random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
    algorithm.fit(X_train, y_train)
    predictions_test = algorithm.predict(X_test)
    predictions_test_orig = le.inverse_transform(predictions_test)
    X_test_copy = X_test.copy()
    X_test_copy['r_hat'] = predictions_test_orig
    X_test_copy['r'] = dataset.loc[X_test.index]['r']
    baseline_accuracy = accuracy_score(X_test_copy['r_hat'], X_test_copy['r'])
    
    print(f"Migliore R-Accuracy trovata: {best_accuracy:.4f}")
    if best_config is not None:
        print(f"Split points: {np.round(best_config, 2)}")
    print(f"Baseline (equidistante): {baseline_accuracy:.4f}")
    
    # Mostra top 5
    if results_for_this_n_classes:
        df_results = pd.DataFrame(results_for_this_n_classes)
        df_results = df_results.sort_values('r_accuracy', ascending=False).head(5)
        print(f"\nTop 5 configurazioni:")
        print(df_results.to_string(index=False))
    
    all_results.extend(results_for_this_n_classes)

print(f"\n{'='*80}")
print("ANALISI COMPLETA")
print(f"{'='*80}\n")

if all_results:
    df_all = pd.DataFrame(all_results)
    best_overall = df_all.loc[df_all['r_accuracy'].idxmax()]
    print(f"Configurazione migliore globale:")
    print(f"  n_classes: {best_overall['n_classes']}")
    print(f"  Split points: {best_overall['split_points']}")
    print(f"  R-Accuracy: {best_overall['r_accuracy']:.4f}\n")
    
    for nc in [4, 6, 8]:
        df_nc = df_all[df_all['n_classes'] == nc]
        if not df_nc.empty:
            best_for_nc = df_nc.loc[df_nc['r_accuracy'].idxmax()]
            print(f"Migliore per n_classes={nc}: {best_for_nc['r_accuracy']:.4f}")
