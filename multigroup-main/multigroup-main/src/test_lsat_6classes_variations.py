import pandas as pd
import kagglehub
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import matplotlib
matplotlib.use('Agg')
from itertools import product

def categorize_lsat_custom(lsat, split_points):
    """
    Categorizza LSAT usando split_points con 29 come hard boundary.
    split_points ha 4 elementi: [sp1_under, sp2_under, sp1_upper, sp2_upper]
    29 è il limite superiore della classe 3 e inferiore della classe 4.
    """
    if pd.isna(lsat):
        return None
    
    sp1_under, sp2_under, sp1_upper, sp2_upper = split_points
    
    if lsat < sp1_under:
        return 1
    elif lsat < sp2_under:
        return 2
    elif lsat < 29:
        return 3
    elif lsat < sp1_upper:
        return 4
    elif lsat < sp2_upper:
        return 5
    else:
        return 6

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
print(f"Testing con variazioni di ±4 punti da intervalli equidistanti")
print(f"{'='*80}\n")

n_classes = 6
half_classes = 3

# Calcola intervalli equidistanti con 29 come hard boundary
# Range totale sotto 29: 11 a 29 (18 punti)
# Range totale sopra 29: 29 a 48 (19 punti)

lower_step = 18 / 3  # 6 punti per classe
upper_step = 19 / 3  # ~6.33 punti per classe

# Split points equidistanti
equidistant_lower = [11 + lower_step, 11 + 2*lower_step]  # 17, 23
equidistant_upper = [29 + upper_step, 29 + 2*upper_step]  # ~35.33, ~40.67

equidistant_split_points = [equidistant_lower[0], equidistant_lower[1], equidistant_upper[0], equidistant_upper[1]]

print(f"Split points equidistanti: {np.round(equidistant_split_points, 2)}")
print(f"(con 29 come hard boundary tra classe 3 e 4)\n")

# Genera variazioni di ±4 punti per ogni split point
variations = [-4, -3, -2, -1, 0, 1, 2, 3, 4]

results = []
test_count = 0
max_tests = 500

print(f"Testando combinazioni di variazioni...")

# Generiamo combinazioni di variazioni per i 4 split points
for var_lower_1 in variations:
    for var_lower_2 in variations:
        for var_upper_1 in variations:
            for var_upper_2 in variations:
                if test_count >= max_tests:
                    break
                
                # Applica variazioni
                sp1 = equidistant_lower[0] + var_lower_1
                sp2 = equidistant_lower[1] + var_lower_2
                sp3 = equidistant_upper[0] + var_upper_1
                sp4 = equidistant_upper[1] + var_upper_2
                
                # Valida i limiti
                if sp1 < 11.5 or sp1 >= 28.9 or sp2 < 11.5 or sp2 >= 28.9 or sp3 < 29.1 or sp3 > 47.5 or sp4 < 29.1 or sp4 > 47.5:
                    continue
                
                # Assicura ordine corretto
                if sp1 >= sp2 or sp3 >= sp4:
                    continue
                
                split_points = [sp1, sp2, sp3, sp4]
                
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
                        'var_lower_1': var_lower_1,
                        'var_lower_2': var_lower_2,
                        'var_upper_1': var_upper_1,
                        'var_upper_2': var_upper_2
                    })
                    
                    test_count += 1
                    
                    if test_count % 100 == 0:
                        print(f"Test completati: {test_count}")
                        
                except Exception as e:
                    continue

print(f"\nTest totali eseguiti: {test_count}\n")

if results:
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values('r_accuracy', ascending=False)
    
    print(f"{'='*80}")
    print("TOP 15 CONFIGURAZIONI")
    print(f"{'='*80}\n")
    
    for idx, (i, row) in enumerate(df_results.head(15).iterrows(), 1):
        sp = row['split_points']
        print(f"{idx}. R-Accuracy: {row['r_accuracy']:.4f}")
        print(f"   Variazioni: lower_1={row['var_lower_1']:+d}, lower_2={row['var_lower_2']:+d}, upper_1={row['var_upper_1']:+d}, upper_2={row['var_upper_2']:+d}")
        print(f"   Split points: {row['split_points_str']}")
        print(f"   Intervalli:")
        print(f"     Classe 1: LSAT < {sp[0]:.2f}")
        print(f"     Classe 2: {sp[0]:.2f} <= LSAT < {sp[1]:.2f}")
        print(f"     Classe 3: {sp[1]:.2f} <= LSAT < 29")
        print(f"     Classe 4: 29 < LSAT < {sp[2]:.2f}")
        print(f"     Classe 5: {sp[2]:.2f} <= LSAT < {sp[3]:.2f}")
        print(f"     Classe 6: LSAT >= {sp[3]:.2f}")
        print()
    
    best = df_results.iloc[0]
    print(f"{'='*80}")
    print("CONFIGURAZIONE MIGLIORE")
    print(f"{'='*80}")
    print(f"R-Accuracy: {best['r_accuracy']:.4f}")
    print(f"Split points: {best['split_points_str']}")
    print(f"{'='*80}\n")
