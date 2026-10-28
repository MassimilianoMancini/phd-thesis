import pandas as pd
import kagglehub
import numpy as np
from fairnessMeasuresAPI import fm
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib
# from SimpleMLP import SimpleMLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, top_k_accuracy_score
matplotlib.use('Agg') 

import matplotlib.pyplot as plt
from seaborn import heatmap


# Success function
def success_func(r_hat):
    return 1 if r_hat > 3 else 0

def rBoundaryAccuracy(r, s, y, h, d=1):
    return 1 if abs(r - s) <= d and y==h else 0

def rBoundaryAccuracy2(r, s, d=1):
    return 1 if abs(r - s) <= d else 0

def categorize_lsat_optimized(lsat):
    if pd.isna(lsat):
        return None
    
    if lsat <= 25.0:
        return 1
    elif lsat <= 27.0:
        return 2
    elif lsat < 29:
        return 3
    elif lsat <= 35.0:
        return 4
    elif lsat <= 39.7:
        return 5
    else:
        return 6


# Download latest version of law school dataset
path = kagglehub.dataset_download("danofer/law-school-admissions-bar-passage")
file = path + "/bar_pass_prediction.csv"

group = 'race1'
score = 'r'
subset = 'X_test'
algorithm = XGBClassifier(objective='multi:softprob')

toDrop     = ['decile1b', 'decile3', 'ID', 'race', 'lsat', 'zfygpa', 'grad', 'zgpa', 'bar1', 'bar1_yr', 'bar2', 'bar2_yr', 'age', 'gender', 'male', 'race2', 'bar', 'Dropout', 'other', 'asian', 'black', 'hisp', 'pass_bar', 'bar_passed', 'tier', 'index6040',   'indxgrp', 'indxgrp2', 'gpa', 'dnn_bar_pass_prediction', 'gpa' ]

dataset = pd.read_csv(file, sep=',', header='infer')

# Crea colonna 'r' con categorizzazione LSAT ottimizzata
dataset['r'] = dataset['lsat'].apply(lambda x: categorize_lsat_optimized(x))

#dataset = dataset[dataset['race1'].notna()]
#dataset = dataset[dataset['r'].notna()]

dataset['DOB_yr'] = dataset['DOB_yr'].apply(lambda x: x + 1900)

dataset = dataset.drop(toDrop, axis=1)
dataset.dropna(inplace=True)

print("\n" + "="*80)
print("DISTRIBUZIONE RECORD PER CLASSE LSAT")
print("="*80)
class_counts = dataset['r'].value_counts().sort_index()
for class_num, count in class_counts.items():
    pct = (count / len(dataset)) * 100
    print(f"Classe {int(class_num)}: {count:5d} record ({pct:5.1f}%)")
print(f"TOTALE: {len(dataset)} record")
print("="*80 + "\n")

print(dataset.race1.value_counts())

X = pd.get_dummies(dataset.drop(score, axis=1))
le = LabelEncoder()
y_encoded = le.fit_transform(dataset[score])  # ensures same mapping for train/test

n_classes = len(le.classes_)
algorithm = XGBClassifier()


X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
algorithm.fit(X_train, y_train)
predictions_test = algorithm.predict(X_test)
predictions_train = algorithm.predict(X_train)

# convert predictions back to the original label scale
predictions_test_orig = le.inverse_transform(predictions_test)
predictions_train_orig = le.inverse_transform(predictions_train)

# Probabilities (needed for top-k accuracy)
y_proba_test  = algorithm.predict_proba(X_test)   # shape: (n_samples, n_classes)
y_proba_train = algorithm.predict_proba(X_train)

# Top-k accuracy (top-1 == standard accuracy)
top1_test = top_k_accuracy_score(y_test, y_proba_test, k=1, labels=np.arange(n_classes))
top3_test = top_k_accuracy_score(y_test, y_proba_test, k=3, labels=np.arange(n_classes))
top5_test = top_k_accuracy_score(y_test, y_proba_test, k=5, labels=np.arange(n_classes))

top1_train = top_k_accuracy_score(y_train, y_proba_train, k=1, labels=np.arange(n_classes))
top3_train = top_k_accuracy_score(y_train, y_proba_train, k=3, labels=np.arange(n_classes))
top5_train = top_k_accuracy_score(y_train, y_proba_train, k=5, labels=np.arange(n_classes))


X_test['r_hat'] = predictions_test_orig
X_test[score] = dataset.loc[X_test.index][score]
X_test['y_hat'] = X_test['r_hat'].apply(success_func)
X_test['y'] = X_test[score].apply(success_func)


X_train['r_hat'] = predictions_train_orig
X_train[score] = dataset.loc[X_train.index][score]
X_train['y_hat'] = X_train['r_hat'].apply(success_func)
X_train['y'] = X_train[score].apply(success_func)

d1 = eval(subset)

d = dataset.loc[d1.index].copy()
d['r_hat'] = d1['r_hat']
d['y'] = d1['y']
d['y_hat'] = d1['y_hat']

f = fm(d, group, 'y', 'y_hat', 'r_hat', success_func)

ts = f.measure(measure='ts', result='matrix')
sp = f.measure(measure='sp', result='matrix')
ta = f.measure(measure='ta', result='matrix')
ca = f.measure(measure='ca', result='matrix')

header = sorted(d[group].unique().tolist())
pd.options.display.float_format = '{:,.3f}'.format

r_accuracy = accuracy_score(d['r_hat'], d['r'])
y_accuracy = accuracy_score(d['y'], d['y_hat'])

print ("==============================================")
print (f'Algorithm is {algorithm.__class__.__name__}')
print (f'File is {file}')
print (f'Subset is {subset}')
print (f'Sensible attribute is {group}')
print (f'Score is {score}')
print (f'Y-Accuracy of success prediction {y_accuracy}')
print (f'R-Accuracy of score prediction {r_accuracy}')

print ("==============================================")

print ("Classes:")
for index, value in enumerate(header):
    print(f"{index}:{value}")

print ("==============================================")

print ('True statistical parity')
print ('Pairwise matrix')
print (pd.DataFrame(ts))
print (f.measure(measure='ts', result='latex'))
plt.clf()
hm = heatmap(ts, cmap="coolwarm", cbar=False, annot=True, fmt=".3f", xticklabels=header + ['1-vs-all'], yticklabels=header)
hm.figure.savefig('x-ts.png')



print ("==============================================")
print ('Statistical parity')
print ('Pairwise matrix')
print (pd.DataFrame(sp))
print (f.measure(measure='sp', result='latex'))
plt.clf()
hm = heatmap(sp, cmap="coolwarm", cbar=False, annot=True, fmt=".3f", xticklabels=header + ['1-vs-all'], yticklabels=header)
hm.figure.savefig(f'x-sp.png')


print ("==============================================")
print ('Total accuracy')
print ('Pairwise matrix')
print (pd.DataFrame(ta))
print (f.measure(measure='ta', result='latex'))
plt.clf()
hm = heatmap(ta, cmap="coolwarm", cbar=False, annot=True, fmt=".3f", xticklabels=header + ['1-vs-all'], yticklabels=header)
hm.figure.savefig('x-ta.png')


print ("==============================================")
print ('Calibration')
print ('Pairwise matrix')
print (pd.DataFrame(ca))
print (f.measure(measure='ca', result='latex'))
plt.clf()
hm = heatmap(ca, cmap="coolwarm", cbar=False, annot=True, fmt=".3f", xticklabels=header + ['1-vs-all'], yticklabels=header)
hm.figure.savefig('x-ca.png')


print ("==============================================")
print ("==============================================")
print()



