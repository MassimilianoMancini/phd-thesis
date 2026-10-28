import pandas as pd
import kagglehub
import matplotlib.pyplot as plt
import seaborn as sns

# Download dataset
path = kagglehub.dataset_download("danofer/law-school-admissions-bar-passage")
file = path + "/bar_pass_prediction.csv"

# Load dataset
dataset = pd.read_csv(file, sep=',', header='infer')

# Crea colonna di raggruppamento per intervalli di LSAT
def categorize_lsat(lsat):
    if pd.isna(lsat):
        return None
    elif lsat < 20:
        return '1_Molto_Basso (<20)'
    elif lsat < 25:
        return '2_Basso (20-24)'
    elif lsat < 30:
        return '3_Basso-Medio (25-29)'
    elif lsat < 35:
        return '4_Medio (30-34)'
    elif lsat < 40:
        return '5_Medio-Alto (35-39)'
    else:
        return '6_Alto (40+)'

dataset['lsat_category'] = dataset['lsat'].apply(categorize_lsat)

# Visualizza la distribuzione di LSAT
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Istogramma
axes[0].hist(dataset['lsat'], bins=30, edgecolor='black', alpha=0.7, color='skyblue')
axes[0].set_xlabel('LSAT Score')
axes[0].set_ylabel('Frequenza')
axes[0].set_title('Distribuzione LSAT - Istogramma')
axes[0].grid(axis='y', alpha=0.3)

# Density plot
dataset['lsat'].plot(kind='density', ax=axes[1], color='steelblue', linewidth=2)
axes[1].set_xlabel('LSAT Score')
axes[1].set_ylabel('Densità')
axes[1].set_title('Distribuzione LSAT - Density Plot')
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('lsat_distribution.png', dpi=300, bbox_inches='tight')
print("Grafico salvato come 'lsat_distribution.png'")

# Stampa statistiche
print("\n=== Statistiche LSAT ===")
print(dataset['lsat'].describe())
print(f"\nValori mancanti: {dataset['lsat'].isna().sum()}")
print(f"Totale osservazioni: {len(dataset)}")

# Conteggio elementi per ogni valore di LSAT
print("\n=== Conteggio elementi per valore LSAT ===")
lsat_counts = dataset['lsat'].value_counts().sort_index()
print(lsat_counts)
print(f"\nTotale valori unici di LSAT: {len(lsat_counts)}")

# Distribuzione per categoria
print("\n=== Distribuzione per categoria LSAT ===")
category_counts = dataset['lsat_category'].value_counts().sort_index()
print(category_counts)
print(f"\nPercentuali:")
print(dataset['lsat_category'].value_counts(normalize=True).sort_index() * 100)
