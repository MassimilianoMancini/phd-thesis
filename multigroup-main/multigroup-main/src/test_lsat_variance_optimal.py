import pandas as pd
import kagglehub
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

def categorize_lsat_balanced(lsat, split_points):
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

base_dataset = pd.read_csv(file, sep=',', header='infer')
lsat_values = base_dataset['lsat'].dropna().values

print("=" * 100)
print("RICERCA SPLIT POINTS CON VARIANZA MINIMA SOPRA 29")
print("=" * 100 + "\n")

# Dividi i dati
under_29 = np.sort(lsat_values[lsat_values < 29])
above_29 = np.sort(lsat_values[lsat_values >= 29])

print(f"SOTTO 29: {len(under_29)} elementi")
print(f"SOPRA 29: {len(above_29)} elementi\n")

# Split points fissi SOTTO 29 (già ottimi da ricerca precedente)
target_under = len(under_29) / 3
idx1_under = int(target_under) - 1
idx2_under = int(2 * target_under) - 1
sp1_under = under_29[idx1_under]
sp2_under = under_29[idx2_under]

print(f"Split points SOTTO 29 (FISSI): [{sp1_under}, {sp2_under}]")
print(f"  C1: {len(under_29[:idx1_under+1])} elementi")
print(f"  C2: {len(under_29[idx1_under+1:idx2_under+1])} elementi")
print(f"  C3: {len(under_29[idx2_under+1:])} elementi\n")

# Ricerca sui split points SOPRA 29
target_above = len(above_29) / 3
best_variance = float('inf')
best_split = None
best_counts = None

# Ottieni valori unici LSAT sopra 29 per fare ricerca efficiente
unique_above_29 = sorted(np.unique(above_29))

print(f"Ricerca split points sopra 29...")
print(f"Target elementi per classe: {target_above:.0f}\n")

# Ricerca su tutti i possibili split point
for i, sp1_candidate in enumerate(unique_above_29):
    # Elementi fino al primo split
    c4 = np.sum((above_29 >= 29) & (above_29 <= sp1_candidate))
    
    # Cerchiamo il secondo split fra i candidati dopo il primo
    for j in range(i+1, len(unique_above_29)):
        sp2_candidate = unique_above_29[j]
        
        # Conta gli elementi per ogni classe
        c4_count = np.sum((above_29 >= 29) & (above_29 <= sp1_candidate))
        c5_count = np.sum((above_29 > sp1_candidate) & (above_29 <= sp2_candidate))
        c6_count = np.sum((above_29 > sp2_candidate))
        
        # Calcola varianza
        counts = [c4_count, c5_count, c6_count]
        variance = np.var(counts)
        
        # Aggiorna il migliore
        if variance < best_variance:
            best_variance = variance
            best_split = [sp1_candidate, sp2_candidate]
            best_counts = counts

print(f"RISULTATO MIGLIORE TROVATO")
print(f"Split points sopra 29: {best_split}")
print(f"Classi: {best_counts}")
print(f"  C4: {best_counts[0]} elementi ({best_counts[0]/len(above_29)*100:.1f}%)")
print(f"  C5: {best_counts[1]} elementi ({best_counts[1]/len(above_29)*100:.1f}%)")
print(f"  C6: {best_counts[2]} elementi ({best_counts[2]/len(above_29)*100:.1f}%)")
print(f"Media: {np.mean(best_counts):.1f}, Varianza: {best_variance:.1f}, StdDev: {np.std(best_counts):.1f}\n")

# Crea split points finali
split_points_final = [sp1_under, sp2_under, best_split[0], best_split[1]]

# Verifica completa della distribuzione
print("=" * 100)
print("DISTRIBUZIONE COMPLETA")
print("=" * 100 + "\n")

c1 = np.sum((lsat_values <= sp1_under))
c2 = np.sum((lsat_values > sp1_under) & (lsat_values <= sp2_under))
c3 = np.sum((lsat_values > sp2_under) & (lsat_values < 29))
c4 = np.sum((lsat_values >= 29) & (lsat_values <= best_split[0]))
c5 = np.sum((lsat_values > best_split[0]) & (lsat_values <= best_split[1]))
c6 = np.sum((lsat_values > best_split[1]))

all_counts = [c1, c2, c3, c4, c5, c6]
print(f"Split points: {split_points_final}\n")
print("=" * 100)
print("DISTRIBUZIONE DETTAGLIATA PER CLASSE")
print("=" * 100 + "\n")

# Dettagli per classe
class_info = [
    (1, f"LSAT <= {sp1_under}", all_counts[0]),
    (2, f"{sp1_under} < LSAT <= {sp2_under}", all_counts[1]),
    (3, f"{sp2_under} < LSAT < 29", all_counts[2]),
    (4, f"29 <= LSAT <= {best_split[0]}", all_counts[3]),
    (5, f"{best_split[0]} < LSAT <= {best_split[1]}", all_counts[4]),
    (6, f"LSAT > {best_split[1]}", all_counts[5]),
]

for class_num, interval, count in class_info:
    pct = count/len(lsat_values)*100
    print(f"Classe {class_num}: {interval:<30} -> {count:5d} elementi ({pct:5.1f}%)")

print(f"\n" + "=" * 100)
print("STATISTICHE PER GRUPPO")
print("=" * 100 + "\n")

under_counts = all_counts[:3]
print(f"SOTTO 29:")
for i, c in enumerate(under_counts, 1):
    print(f"  Classe {i}: {c:5d} elementi")
print(f"  Media: {np.mean(under_counts):.1f}, StdDev: {np.std(under_counts):.1f}")

above_counts = all_counts[3:]
print(f"\nSOPRA 29:")
for i, c in enumerate(above_counts, 4):
    print(f"  Classe {i}: {c:5d} elementi")
print(f"  Media: {np.mean(above_counts):.1f}, StdDev: {np.std(above_counts):.1f}")

# Test R-accuracy
print("\n" + "=" * 100)
print("VERIFICA R-ACCURACY")
print("=" * 100 + "\n")

toDrop = ['decile1b', 'decile3', 'ID', 'race', 'lsat', 'zfygpa', 'grad', 'zgpa', 'bar1', 'bar1_yr', 
          'bar2', 'bar2_yr', 'age', 'gender', 'male', 'race2', 'bar', 'Dropout', 'other', 'asian', 
          'black', 'hisp', 'pass_bar', 'bar_passed', 'tier', 'index6040', 'indxgrp', 'indxgrp2', 
          'gpa', 'dnn_bar_pass_prediction']

dataset = base_dataset.copy()
dataset['r'] = dataset['lsat'].apply(lambda x: categorize_lsat_balanced(x, split_points_final))
dataset = dataset.dropna(subset=['r'])

print("\n" + "=" * 100)
print("PRIME 10 RIGHE DEL DATASET CON CLASSI ASSEGNATE")
print("=" * 100 + "\n")
print(dataset[['lsat', 'r']].head(10).to_string())
print("\n")

X = dataset.drop(columns=toDrop)
# Rimuovi anche 'r' dalle features (è il TARGET, non una feature!)
X = X.drop(columns=['r'])
# Encode categorical columns
for col in X.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))

print("\n" + "=" * 100)
print("INFORMAZIONI SUL DATASET PER IL MODELLO")
print("=" * 100 + "\n")
print(f"Colonne usate dal modello (X): {X.columns.tolist()}")
print(f"Numero di features: {X.shape[1]}")
print(f"Numero di campioni: {X.shape[0]}")
print(f"\nLSAT è presente in X? {'lsat' in X.columns}")
print(f"Colonne droppate: {toDrop}\n")

y = (dataset['r'].astype(int) - 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = XGBClassifier(max_depth=6, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

# Calcola accuracy anche su train set
y_pred_train = model.predict(X_train)
acc_train = accuracy_score(y_train, y_pred_train)

print(f"R-Accuracy (Train): {acc_train:.4f}")
print(f"R-Accuracy (Test):  {acc:.4f}")

# Feature importance
print(f"\nImportanza features (top 5):")
feature_importance = sorted(zip(X.columns, model.feature_importances_), key=lambda x: x[1], reverse=True)
for feat, imp in feature_importance[:5]:
    print(f"  {feat}: {imp:.4f}")

# Se accuracy è 1.0 e train/test sono uguali, c'è overfitting
if acc == 1.0:
    print(f"\nAttention: Accuracy perfetta! Controllare:")
    print(f"  - Train accuracy: {acc_train:.4f} (overfitting se < 1.0)")
    print(f"  - Test size: {len(y_test)} campioni")
    print(f"  - Classi uniche in y_test: {len(np.unique(y_test))}")
    
    # Mostra qualche predizione
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\nMatrice di confusione:")
    print(cm)
print(f"\nSplit points ottimali: {split_points_final}")

print("\n" + "=" * 100)
