import pandas as pd
import kagglehub
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

def categorize_lsat_balanced(lsat, split_points):
    """
    Categorizza LSAT con split_points che assicurano balance esatto.
    split_points: [sp1_under, sp2_under, sp1_upper, sp2_upper]
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

def find_balanced_split_points(lsat_values):
    """
    Trova split points che creano classi PERFETTAMENTE bilanciate.
    
    Restituisce: [sp1_under, sp2_under, sp1_upper, sp2_upper]
    dove:
    - C1: LSAT < sp1_under        (~551 elementi)
    - C2: sp1_under <= LSAT < sp2_under (~551 elementi)
    - C3: sp2_under <= LSAT < 29   (~551 elementi)
    - C4: 29 <= LSAT < sp1_upper   (~6918 elementi)
    - C5: sp1_upper <= LSAT < sp2_upper (~6918 elementi)
    - C6: sp2_upper <= LSAT        (~6918 elementi)
    """
    
    # Dividi in sotto e sopra 29
    under_29 = sorted(lsat_values[lsat_values < 29])
    above_29 = sorted(lsat_values[lsat_values >= 29])
    
    print(f"SOTTO 29: {len(under_29)} elementi")
    print(f"SOPRA 29: {len(above_29)} elementi\n")
    
    # SOTTO 29: Dividi in 3 parti uguali
    target_under = len(under_29) / 3
    idx1_under = int(target_under) - 1
    idx2_under = int(2 * target_under) - 1
    
    # Il split point è il valore DOPO il quale cambia la classe
    # Se voglio 3 classi uguali:
    # C1: elementi 0 a idx1_under
    # C2: elementi idx1_under+1 a idx2_under
    # C3: elementi idx2_under+1 a fine
    
    sp1_under = under_29[idx1_under]  # Ultimo elemento di C1
    sp2_under = under_29[idx2_under]  # Ultimo elemento di C2
    
    # SOPRA 29: Dividi in 3 parti uguali
    target_above = len(above_29) / 3
    idx1_above = int(target_above) - 1
    idx2_above = int(2 * target_above) - 1
    
    sp1_upper = above_29[idx1_above]  # Ultimo elemento di C4
    sp2_upper = above_29[idx2_above]  # Ultimo elemento di C5
    
    return [sp1_under, sp2_under, sp1_upper, sp2_upper]

# Download dataset
path = kagglehub.dataset_download("danofer/law-school-admissions-bar-passage")
file = path + "/bar_pass_prediction.csv"

toDrop = ['decile1b', 'decile3', 'ID', 'race', 'lsat', 'zfygpa', 'grad', 'zgpa', 'bar1', 'bar1_yr', 
          'bar2', 'bar2_yr', 'age', 'gender', 'male', 'race2', 'bar', 'Dropout', 'other', 'asian', 
          'black', 'hisp', 'pass_bar', 'bar_passed', 'tier', 'index6040', 'indxgrp', 'indxgrp2', 
          'gpa', 'dnn_bar_pass_prediction']

base_dataset = pd.read_csv(file, sep=',', header='infer')
base_dataset['DOB_yr'] = base_dataset['DOB_yr'].apply(lambda x: x + 1900)

print("=" * 100)
print("RICERCA SPLIT POINTS CON BALANCE ESATTO ENTRO I GRUPPI")
print("=" * 100 + "\n")

lsat_values = base_dataset['lsat'].dropna().values

# Trova i split points
split_points = find_balanced_split_points(lsat_values)
print(f"Split points trovati: {split_points}\n")

# Verifica la distribuzione
def count_per_class(lsat_data, split_points):
    sp1_under, sp2_under, sp1_upper, sp2_upper = split_points
    c1 = np.sum((lsat_data >= np.min(lsat_data)) & (lsat_data <= sp1_under))
    c2 = np.sum((lsat_data > sp1_under) & (lsat_data <= sp2_under))
    c3 = np.sum((lsat_data > sp2_under) & (lsat_data < 29))
    c4 = np.sum((lsat_data >= 29) & (lsat_data <= sp1_upper))
    c5 = np.sum((lsat_data > sp1_upper) & (lsat_data <= sp2_upper))
    c6 = np.sum((lsat_data > sp2_upper))
    return [c1, c2, c3, c4, c5, c6]

counts = count_per_class(lsat_values, split_points)

print("Distribuzione classi (PERFETTAMENTE BILANCIATE):")
for i, c in enumerate(counts, 1):
    print(f"  Classe {i}: {c:5d} ({c/len(lsat_values)*100:5.1f}%)")

# Verifica balance dentro i gruppi
under_counts = counts[:3]
above_counts = counts[3:]
print(f"\nSotto 29 (target ~551): {under_counts}")
print(f"  Media: {np.mean(under_counts):.1f}, StdDev: {np.std(under_counts):.1f}")
print(f"\nSopra 29 (target ~6918): {above_counts}")
print(f"  Media: {np.mean(above_counts):.1f}, StdDev: {np.std(above_counts):.1f}")

# Addestra modello per verificare R-accuracy
print("\n" + "=" * 100)
print("VERIFICA R-ACCURACY")
print("=" * 100 + "\n")

dataset = base_dataset.copy()
dataset['r'] = dataset['lsat'].apply(lambda x: categorize_lsat_balanced(x, split_points))
dataset = dataset.dropna(subset=['r'])

X = dataset.drop(columns=toDrop)
# Encode categorical columns
for col in X.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))

y = (dataset['r'].astype(int) - 1)  # Convert 1-6 to 0-5

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = XGBClassifier(max_depth=6, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"R-Accuracy: {acc:.4f}")
print(f"\nSplit points da usare: {split_points}")
print(f"Configurazione: [np.{split_points[0]}, np.{split_points[1]}, np.{split_points[2]}, np.{split_points[3]}]")

print("\n" + "=" * 100)
