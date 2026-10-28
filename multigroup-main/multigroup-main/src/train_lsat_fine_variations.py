import pandas as pd
import kagglehub
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

def categorize_lsat_custom(lsat, split_points):
    """
    Categorizza LSAT con split_points: [sp1_under, sp2_under, sp1_upper, sp2_upper]
    """
    if pd.isna(lsat):
        return None
    
    sp1_under, sp2_under, sp1_upper, sp2_upper = split_points
    
    if lsat <= sp1_under:
        return 1
    elif lsat <= sp2_under:
        return 2
    elif lsat < 29:
        return 3
    elif lsat <= sp1_upper:
        return 4
    elif lsat <= sp2_upper:
        return 5
    else:
        return 6

# Download dataset
path = kagglehub.dataset_download("danofer/law-school-admissions-bar-passage")
file = path + "/bar_pass_prediction.csv"

toDrop = ['decile1b', 'decile3', 'ID', 'race', 'lsat', 'zfygpa', 'grad', 'zgpa', 'bar1', 'bar1_yr', 
          'bar2', 'bar2_yr', 'age', 'gender', 'male', 'race2', 'bar', 'Dropout', 'other', 'asian', 
          'black', 'hisp', 'pass_bar', 'bar_passed', 'tier', 'index6040', 'indxgrp', 'indxgrp2', 
          'gpa', 'dnn_bar_pass_prediction']

base_dataset = pd.read_csv(file, sep=',', header='infer')
base_dataset['DOB_yr'] = base_dataset['DOB_yr'].apply(lambda x: x + 1900)

print("=" * 120)
print("RICERCA VARIAZIONI SUI SPLIT POINTS SUPERIORI A 29 (TRAINING SET)")
print("=" * 120 + "\n")

lsat_values = base_dataset['lsat'].dropna().values

# Split points fissi SOTTO 29
sp1_under = 25.0
sp2_under = 27.0

# Split points BASE SOPRA 29 (da ottimizzare)
sp1_upper_base = 35.0
sp2_upper_base = 40.0

print(f"Split points FISSI sotto 29: [{sp1_under}, {sp2_under}]")
print(f"Split points BASE sopra 29: [{sp1_upper_base}, {sp2_upper_base}]")
print(f"Testando variazioni +/-1, +/-2, +/-3 ai split points sopra 29\n")

results = []

# Genera tutte le combinazioni con variazioni ±1, ±2, ±3
variations = [-3, -2, -1, 0, 1, 2, 3]

for var1 in variations:
    for var2 in variations:
        sp1_upper = sp1_upper_base + var1
        sp2_upper = sp2_upper_base + var2
        
        # Verifica che sp1_upper < sp2_upper
        if sp1_upper >= sp2_upper:
            continue
        
        split_points = [sp1_under, sp2_under, sp1_upper, sp2_upper]
        
        # Conta distribuzione
        c1 = np.sum((lsat_values <= sp1_under))
        c2 = np.sum((lsat_values > sp1_under) & (lsat_values <= sp2_under))
        c3 = np.sum((lsat_values > sp2_under) & (lsat_values < 29))
        c4 = np.sum((lsat_values >= 29) & (lsat_values <= sp1_upper))
        c5 = np.sum((lsat_values > sp1_upper) & (lsat_values <= sp2_upper))
        c6 = np.sum((lsat_values > sp2_upper))
        
        counts = [c1, c2, c3, c4, c5, c6]
        
        # Calcola varianza sopra 29
        above_29_counts = [c4, c5, c6]
        variance_above = np.var(above_29_counts)
        
        # Addestra modello per verificare R-accuracy
        dataset = base_dataset.copy()
        dataset['r'] = dataset['lsat'].apply(lambda x: categorize_lsat_custom(x, split_points))
        dataset = dataset.dropna(subset=['r'])
        
        X = dataset.drop(columns=toDrop)
        X = X.drop(columns=['r'])
        
        # Encode categorical columns
        for col in X.select_dtypes(include=['object', 'str']).columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
        
        y = (dataset['r'].astype(int) - 1)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = XGBClassifier(max_depth=6, random_state=42)
        model.fit(X_train, y_train)
        y_pred_train = model.predict(X_train)
        acc_train = accuracy_score(y_train, y_pred_train)
        
        results.append({
            'sp1_var': var1,
            'sp2_var': var2,
            'sp1_upper': sp1_upper,
            'sp2_upper': sp2_upper,
            'c4': c4,
            'c5': c5,
            'c6': c6,
            'variance': variance_above,
            'accuracy': acc_train
        })

# Ordina per accuracy decrescente
results_sorted = sorted(results, key=lambda x: x['accuracy'], reverse=True)

print("\n" + "=" * 120)
print("TOP 20 RISULTATI (Ordinati per R-Accuracy su Training Set)")
print("=" * 120 + "\n")

print(f"{'Var1':>5} {'Var2':>5} {'SP1':>6} {'SP2':>6} {'C4':>6} {'C5':>6} {'C6':>6} {'Var[C4,C5,C6]':>12} {'R-Accuracy':>12}")
print("-" * 120)

for i, result in enumerate(results_sorted[:20]):
    print(f"{result['sp1_var']:>5} {result['sp2_var']:>5} {result['sp1_upper']:>6.1f} {result['sp2_upper']:>6.1f} "
          f"{result['c4']:>6} {result['c5']:>6} {result['c6']:>6} {result['variance']:>12.1f} {result['accuracy']:>12.4f}")

print("\n" + "=" * 120)
print("TOP 10 RISULTATI (Ordinati per Varianza - Classi più bilanciate)")
print("=" * 120 + "\n")

results_var_sorted = sorted(results, key=lambda x: x['variance'])

print(f"{'Var1':>5} {'Var2':>5} {'SP1':>6} {'SP2':>6} {'C4':>6} {'C5':>6} {'C6':>6} {'Var[C4,C5,C6]':>12} {'R-Accuracy':>12}")
print("-" * 120)

for i, result in enumerate(results_var_sorted[:10]):
    print(f"{result['sp1_var']:>5} {result['sp2_var']:>5} {result['sp1_upper']:>6.1f} {result['sp2_upper']:>6.1f} "
          f"{result['c4']:>6} {result['c5']:>6} {result['c6']:>6} {result['variance']:>12.1f} {result['accuracy']:>12.4f}")

print("\n" + "=" * 120)
print(f"Totale configurazioni testate: {len(results)}")
print("=" * 120)
