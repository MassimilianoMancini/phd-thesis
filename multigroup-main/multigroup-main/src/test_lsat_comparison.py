import pandas as pd
import kagglehub
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

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

# Leggi dataset
base_dataset = pd.read_csv(file, sep=',', header='infer')
base_dataset['DOB_yr'] = base_dataset['DOB_yr'].apply(lambda x: x + 1900)

print("=" * 100)
print("CONFRONTO TRE APPROCCI DI CATEGORIZZAZIONE LSAT")
print("=" * 100)

lsat_values = base_dataset['lsat'].dropna().values
print(f"\nDati LSAT: {len(lsat_values)} elementi, range [{lsat_values.min():.0f}, {lsat_values.max():.0f}]")

# ===========================
# APPROCCIO 1: EQUIDISTANTE
# ===========================
print("\n" + "=" * 100)
print("APPROCCIO 1: EQUIDISTANTE")
print("=" * 100)

# Split the range: 3 classes under 29, 3 classes from 29 onward
# Under 29: 1 to 29 - divide into 3 equal parts
# Above 29: 29 to 48 - divide into 3 equal parts

range_under = 29 - lsat_values[lsat_values < 29].min()
range_above = lsat_values[lsat_values >= 29].max() - 29

sp1_eq = lsat_values[lsat_values < 29].min() + range_under / 3
sp2_eq = lsat_values[lsat_values < 29].min() + 2 * range_under / 3
sp1_above_eq = 29 + range_above / 3
sp2_above_eq = 29 + 2 * range_above / 3

split_points_eq = [sp1_eq, sp2_eq, sp1_above_eq, sp2_above_eq]

# Calcola distribuzione
counts_eq = []
c1 = len(lsat_values[lsat_values < sp1_eq])
c2 = len(lsat_values[(lsat_values >= sp1_eq) & (lsat_values < sp2_eq)])
c3 = len(lsat_values[(lsat_values >= sp2_eq) & (lsat_values < 29)])
c4 = len(lsat_values[(lsat_values >= 29) & (lsat_values < sp1_above_eq)])
c5 = len(lsat_values[(lsat_values >= sp1_above_eq) & (lsat_values < sp2_above_eq)])
c6 = len(lsat_values[lsat_values >= sp2_above_eq])
counts_eq = [c1, c2, c3, c4, c5, c6]

# Addestra modello
dataset_eq = base_dataset.copy()
dataset_eq['r'] = dataset_eq['lsat'].apply(lambda x: categorize_lsat_custom(x, split_points_eq))
dataset_eq = dataset_eq.dropna(subset=['r'])
X_eq = dataset_eq.drop(columns=toDrop)
# Encode categorical columns
for col in X_eq.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    X_eq[col] = le.fit_transform(X_eq[col].astype(str))
y_eq = (dataset_eq['r'].astype(int) - 1)  # Convert 1-6 to 0-5
X_train_eq, X_test_eq, y_train_eq, y_test_eq = train_test_split(X_eq, y_eq, test_size=0.2, random_state=42)
model_eq = XGBClassifier(max_depth=6, random_state=42, verbose=0)
model_eq.fit(X_train_eq, y_train_eq)
y_pred_eq = model_eq.predict(X_test_eq)
acc_eq = accuracy_score(y_test_eq, y_pred_eq)

print(f"\nSplit points: [{sp1_eq:.2f}, {sp2_eq:.2f}, {sp1_above_eq:.2f}, {sp2_above_eq:.2f}]")
print(f"\nDistribuzione classi:")
for i, c in enumerate(counts_eq, 1):
    print(f"  Classe {i}: {c:5d} ({c/len(lsat_values)*100:5.1f}%)")
print(f"\nR-Accuracy: {acc_eq:.4f}")

# ===========================
# APPROCCIO 2: VARIAZIONALE ±4
# ===========================
print("\n" + "=" * 100)
print("APPROCCIO 2: VARIAZIONALE ±4 (BEST ACCURACY)")
print("=" * 100)

# Best found from test_lsat_6classes_variations.py
split_points_var = [13, 19, 29, 31.33, 45.67]
# Formatta come [sp1_under, sp2_under, sp1_upper, sp2_upper]
split_points_var = [13, 19, 31.33, 45.67]

# Calcola distribuzione
counts_var = []
c1 = len(lsat_values[lsat_values < split_points_var[0]])
c2 = len(lsat_values[(lsat_values >= split_points_var[0]) & (lsat_values < split_points_var[1])])
c3 = len(lsat_values[(lsat_values >= split_points_var[1]) & (lsat_values < 29)])
c4 = len(lsat_values[(lsat_values >= 29) & (lsat_values < split_points_var[2])])
c5 = len(lsat_values[(lsat_values >= split_points_var[2]) & (lsat_values < split_points_var[3])])
c6 = len(lsat_values[lsat_values >= split_points_var[3]])
counts_var = [c1, c2, c3, c4, c5, c6]

# Addestra modello
dataset_var = base_dataset.copy()
dataset_var['r'] = dataset_var['lsat'].apply(lambda x: categorize_lsat_custom(x, split_points_var))
dataset_var = dataset_var.dropna(subset=['r'])
X_var = dataset_var.drop(columns=toDrop)
# Encode categorical columns
for col in X_var.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    X_var[col] = le.fit_transform(X_var[col].astype(str))
y_var = (dataset_var['r'].astype(int) - 1)  # Convert 1-6 to 0-5
X_train_var, X_test_var, y_train_var, y_test_var = train_test_split(X_var, y_var, test_size=0.2, random_state=42)
model_var = XGBClassifier(max_depth=6, random_state=42)
model_var.fit(X_train_var, y_train_var)
y_pred_var = model_var.predict(X_test_var)
acc_var = accuracy_score(y_test_var, y_pred_var)

print(f"\nSplit points: {split_points_var}")
print(f"\nDistribuzione classi:")
for i, c in enumerate(counts_var, 1):
    print(f"  Classe {i}: {c:5d} ({c/len(lsat_values)*100:5.1f}%)")
print(f"\nR-Accuracy: {acc_var:.4f}")

# ===========================
# APPROCCIO 3: EQUI-NUMEROSO
# ===========================
print("\n" + "=" * 100)
print("APPROCCIO 3: EQUI-NUMEROSO (BALANCED CLASSES)")
print("=" * 100)

# Usa i quantili per dividere ogni sottogruppo in 3 parti uguali
lsat_under_29 = lsat_values[lsat_values < 29]
lsat_above_29 = lsat_values[lsat_values >= 29]

sp1_under = np.quantile(lsat_under_29, 1/3)
sp2_under = np.quantile(lsat_under_29, 2/3)
sp1_upper = np.quantile(lsat_above_29, 1/3)
sp2_upper = np.quantile(lsat_above_29, 2/3)

split_points_equ = [sp1_under, sp2_under, sp1_upper, sp2_upper]

# Calcola distribuzione
counts_equ = []
c1 = len(lsat_values[lsat_values < split_points_equ[0]])
c2 = len(lsat_values[(lsat_values >= split_points_equ[0]) & (lsat_values < split_points_equ[1])])
c3 = len(lsat_values[(lsat_values >= split_points_equ[1]) & (lsat_values < 29)])
c4 = len(lsat_values[(lsat_values >= 29) & (lsat_values < split_points_equ[2])])
c5 = len(lsat_values[(lsat_values >= split_points_equ[2]) & (lsat_values < split_points_equ[3])])
c6 = len(lsat_values[lsat_values >= split_points_equ[3]])
counts_equ = [c1, c2, c3, c4, c5, c6]

# Addestra modello
dataset_equ = base_dataset.copy()
dataset_equ['r'] = dataset_equ['lsat'].apply(lambda x: categorize_lsat_custom(x, split_points_equ))
dataset_equ = dataset_equ.dropna(subset=['r'])
X_equ = dataset_equ.drop(columns=toDrop)
# Encode categorical columns
for col in X_equ.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    X_equ[col] = le.fit_transform(X_equ[col].astype(str))
y_equ = (dataset_equ['r'].astype(int) - 1)  # Convert 1-6 to 0-5
X_train_equ, X_test_equ, y_train_equ, y_test_equ = train_test_split(X_equ, y_equ, test_size=0.2, random_state=42)
model_equ = XGBClassifier(max_depth=6, random_state=42, verbose=0)
model_equ.fit(X_train_equ, y_train_equ)
y_pred_equ = model_equ.predict(X_test_equ)
acc_equ = accuracy_score(y_test_equ, y_pred_equ)

print(f"\nSplit points: [{split_points_equ[0]:.2f}, {split_points_equ[1]:.2f}, {split_points_equ[2]:.2f}, {split_points_equ[3]:.2f}]")
print(f"\nDistribuzione classi:")
for i, c in enumerate(counts_equ, 1):
    print(f"  Classe {i}: {c:5d} ({c/len(lsat_values)*100:5.1f}%)")
print(f"\nR-Accuracy: {acc_equ:.4f}")

# ===========================
# TABELLA COMPARATIVA FINALE
# ===========================
print("\n" + "=" * 100)
print("RIEPILOGO COMPARATIVO")
print("=" * 100)

print(f"\n{'Metodo':<25} {'R-Accuracy':<15} {'Imbalance Ratio':<20} {'Note':<30}")
print("-" * 90)

# Calcola imbalance ratio (max/min elementi per classe)
imbalance_eq = max(counts_eq) / min(counts_eq)
imbalance_var = max(counts_var) / min(counts_var)
imbalance_equ = max(counts_equ) / min(counts_equ)

print(f"{'Equidistante':<25} {acc_eq:<15.4f} {imbalance_eq:<20.2f}x {'Range-based split':<30}")
print(f"{'Variazionale ±4':<25} {acc_var:<15.4f} {imbalance_var:<20.2f}x {'Best accuracy':<30}")
print(f"{'Equi-numeroso':<25} {acc_equ:<15.4f} {imbalance_equ:<20.2f}x {'Balanced within groups':<30}")

print("\nNote sulla distribuzione:")
print(f"  - Sotto 29: {len(lsat_under_29)} elementi (7.4%)")
print(f"  - Sopra 29: {len(lsat_above_29)} elementi (92.6%)")
print(f"  - Rapporto: {len(lsat_above_29)/len(lsat_under_29):.1f}x")
print(f"\nVincolo biologico del dataset:")
print(f"  Non è possibile avere 6 classi perfettamente bilanciate (3735 elementi/classe)")
print(f"  mantenendo 29 come boundary, dato che solo il 7.4% dei dati è sotto 29.")
print(f"\nRaccomandiazione:")
if acc_var > acc_eq and acc_var > acc_equ:
    print(f"  -> APPROCCIO 2 (Variazionale ±4): Best accuracy ({acc_var:.4f})")
    print(f"    Sacrifica un po' di balance per una migliore predizione del modello")
elif acc_equ > acc_eq and acc_equ > acc_var:
    print(f"  -> APPROCCIO 3 (Equi-numeroso): Best balance ({imbalance_equ:.2f}x)")
    print(f"    Sacrifica un po' di accuracy per classi più equilibrate")
else:
    print(f"  -> APPROCCIO 1 (Equidistante): Trade-off moderato")

print("\n" + "=" * 100)
