import pandas as pd
import kagglehub
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import matplotlib
matplotlib.use('Agg')

def categorize_lsat_custom(lsat, split_points):
    """
    Categorizza LSAT usando split_points con 29 come hard boundary.
    split_points ha 4 elementi: [sp1_under, sp2_under, sp1_upper, sp2_upper]
    29 è il limite superiore della classe 3 e il limite inferiore (esclusivo) della classe 4.
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

toDrop = ['decile1b', 'decile3', 'ID', 'race', 'lsat', 'zfygpa', 'grad', 'zgpa', 'bar1', 'bar1_yr', 
          'bar2', 'bar2_yr', 'age', 'gender', 'male', 'race2', 'bar', 'Dropout', 'other', 'asian', 
          'black', 'hisp', 'pass_bar', 'bar_passed', 'tier', 'index6040', 'indxgrp', 'indxgrp2', 
          'gpa', 'dnn_bar_pass_prediction']

# Leggi dataset una volta
base_dataset = pd.read_csv(file, sep=',', header='infer')
base_dataset['DOB_yr'] = base_dataset['DOB_yr'].apply(lambda x: x + 1900)

print(f"{'='*80}")
print(f"Testing con distribuzione equi-numerosa degli elementi")
print(f"(6 intervalli con numero simile di elementi, 29 come limite superiore classe 3)")
print(f"{'='*80}\n")

# Estrai i valori LSAT validi
lsat_values = base_dataset['lsat'].dropna().sort_values().values

print(f"Statistiche LSAT:")
print(f"  Min: {lsat_values.min():.2f}, Max: {lsat_values.max():.2f}")
print(f"  Totale elementi: {len(lsat_values)}")
print(f"  Elementi per classe (target): {len(lsat_values) / 6:.0f}\n")

# Strategia: 
# - Calcola i quantili per 3 classi sotto 29
# - Calcola i quantili per 3 classi da 29+ in su
# - Assicura che il 3° intervallo termini a 29

# Elementi sotto 29
lsat_under_29 = lsat_values[lsat_values < 29]
lsat_above_29 = lsat_values[lsat_values >= 29]

print(f"Elementi sotto 29: {len(lsat_under_29)} ({len(lsat_under_29)/len(lsat_values)*100:.1f}%)")
print(f"Elementi >= 29: {len(lsat_above_29)} ({len(lsat_above_29)/len(lsat_values)*100:.1f}%)\n")

# Dividi equi-numerosamente
# I quantili sono frazioni: 1/3 e 2/3 per dividi i dati in 3 parti uguali
# Sotto 29: 3 classi
sp1_under = np.quantile(lsat_under_29, 1/3)
sp2_under = np.quantile(lsat_under_29, 2/3)

# Sopra 29: 3 classi
sp1_upper = np.quantile(lsat_above_29, 1/3)
sp2_upper = np.quantile(lsat_above_29, 2/3)

# Arrotonda leggermente
sp1_under = np.round(sp1_under, 2)
sp2_under = np.round(sp2_under, 2)
sp1_upper = np.round(sp1_upper, 2)
sp2_upper = np.round(sp2_upper, 2)

split_points_base = [sp1_under, sp2_under, sp1_upper, sp2_upper]

print(f"Split points calcolati (equi-numerosi):")
print(f"  Sotto 29: {sp1_under:.2f}, {sp2_under:.2f}")
print(f"  Sopra 29: {sp1_upper:.2f}, {sp2_upper:.2f}\n")

# Verifica distribuzione
def count_per_class(lsat_data, split_points):
    sp1_under, sp2_under, sp1_upper, sp2_upper = split_points
    c1 = len(lsat_data[lsat_data < sp1_under])
    c2 = len(lsat_data[(lsat_data >= sp1_under) & (lsat_data < sp2_under)])
    c3 = len(lsat_data[(lsat_data >= sp2_under) & (lsat_data < 29)])
    c4 = len(lsat_data[(lsat_data >= 29) & (lsat_data < sp1_upper)])
    c5 = len(lsat_data[(lsat_data >= sp1_upper) & (lsat_data < sp2_upper)])
    c6 = len(lsat_data[lsat_data >= sp2_upper])
    return [c1, c2, c3, c4, c5, c6]

counts_base = count_per_class(lsat_values, split_points_base)
print(f"Distribuzione elementi per classe (split points base):")
for i, c in enumerate(counts_base, 1):
    print(f"  Classe {i}: {c} ({c/len(lsat_values)*100:.1f}%)")

# Test con il dataset
results = []
test_count = 0

dataset = base_dataset.copy()
dataset['r'] = dataset['lsat'].apply(lambda x: categorize_lsat_custom(x, split_points_base))
dataset = dataset.drop(toDrop, axis=1)
dataset.dropna(inplace=True)

score = 'r'

try:
    X = pd.get_dummies(dataset.drop(score, axis=1))
    le = LabelEncoder()
    y_encoded = le.fit_transform(dataset[score])
    
    n_classes_actual = len(le.classes_)
    algorithm = XGBClassifier(eval_metric='mlogloss', random_state=42)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
    
    algorithm.fit(X_train, y_train)
    predictions_test = algorithm.predict(X_test)
    predictions_test_orig = le.inverse_transform(predictions_test)
    
    X_test_copy = X_test.copy()
    X_test_copy['r_hat'] = predictions_test_orig
    X_test_copy[score] = dataset.loc[X_test.index][score]
    
    r_accuracy = accuracy_score(X_test_copy['r_hat'], X_test_copy[score])
    
    results.append({
        'split_points': split_points_base,
        'r_accuracy': r_accuracy,
        'counts': counts_base
    })
    
except Exception as e:
    print(f"Errore: {e}")

print(f"\n{'='*80}")
print(f"RISULTATO - DISTRIBUZIONE EQUI-NUMEROSA")
print(f"{'='*80}\n")

if results:
    result = results[0]
    sp = result['split_points']
    counts = result['counts']
    
    print(f"R-Accuracy: {result['r_accuracy']:.4f}\n")
    print(f"Split points: [{sp[0]:.2f}, {sp[1]:.2f}, 29, {sp[2]:.2f}, {sp[3]:.2f}]\n")
    print(f"Intervalli:")
    print(f"  Classe 1: LSAT < {sp[0]:.2f}          [{counts[0]:5d} elementi, {counts[0]/len(lsat_values)*100:5.1f}%]")
    print(f"  Classe 2: {sp[0]:.2f} <= LSAT < {sp[1]:.2f}  [{counts[1]:5d} elementi, {counts[1]/len(lsat_values)*100:5.1f}%]")
    print(f"  Classe 3: {sp[1]:.2f} <= LSAT < 29      [{counts[2]:5d} elementi, {counts[2]/len(lsat_values)*100:5.1f}%]")
    print(f"  Classe 4: 29 < LSAT < {sp[2]:.2f}       [{counts[3]:5d} elementi, {counts[3]/len(lsat_values)*100:5.1f}%]")
    print(f"  Classe 5: {sp[2]:.2f} <= LSAT < {sp[3]:.2f} [{counts[4]:5d} elementi, {counts[4]/len(lsat_values)*100:5.1f}%]")
    print(f"  Classe 6: LSAT >= {sp[3]:.2f}        [{counts[5]:5d} elementi, {counts[5]/len(lsat_values)*100:5.1f}%]\n")

print(f"{'='*80}\n")

# Salva la configurazione migliore
print(f"Configurazione migliore (equi-numerosa):")
print(f"split_points = {split_points_base}")
