import pandas as pd
import kagglehub
import numpy as np
from fairnessMeasuresAPI import fm
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib
from SimpleMLP import SimpleMLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, top_k_accuracy_score
matplotlib.use('Agg') 

import matplotlib.pyplot as plt
from seaborn import heatmap


# Success function
def success_func(r):
    return 1 if r > 29 else 0

def rBoundaryAccuracy(r, s, y, h, d=1):
    return 1 if abs(r - s) <= d and y==h else 0

def rBoundaryAccuracy2(r, s, d=1):
    return 1 if abs(r - s) <= d else 0


# Download latest version of law school dataset
path = kagglehub.dataset_download("danofer/law-school-admissions-bar-passage")
file = path + "/bar_pass_prediction.csv"

group = 'race1'
score = 'lsatInt'
subset = 'X_test'
algorithm = XGBClassifier(objective='multi:softprob')

toDrop     = ['decile1b', 'decile3', 'ID', 'decile1', 'race', 'cluster', 'lsat', 'ugpa', 'zfygpa', 'grad', 'zgpa', 'bar1', 'bar1_yr', 'bar2', 'bar2_yr', 'age', 'gender', 'male', 'race2', 'bar', 'Dropout', 'other', 'asian', 'black', 'hisp', 'pass_bar', 'bar_passed', 'tier', 'index6040',   'indxgrp', 'indxgrp2', 'gpa', 'dnn_bar_pass_prediction', 'gpa' ]

dataset = pd.read_csv(file, sep=',', header='infer')

#dataset = dataset[dataset['race1'].notna()]
#dataset = dataset[dataset['lsat'].notna()]

dataset['lsatInt'] = dataset['lsat'].apply(lambda x: round(x))
dataset['DOB_yr'] = dataset['DOB_yr'].apply(lambda x: x + 1900)

dataset = dataset.drop(toDrop, axis=1)
dataset.dropna(inplace=True)

print(dataset.race1.value_counts())

X = pd.get_dummies(dataset.drop(score, axis=1))
le = LabelEncoder()
y_encoded = le.fit_transform(dataset[score])  # ensures same mapping for train/test

n_classes = len(le.classes_)
algorithm = XGBClassifier(objective='multi:softprob', num_class=n_classes, eval_metric='mlogloss')


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


X_test['r'] = predictions_test_orig
X_test[score] = dataset.loc[X_test.index][score]
X_test['y_hat'] = X_test['r'].apply(success_func)
X_test['y'] = X_test[score].apply(success_func)
X_test['rd'] = X_test.apply(lambda row: rBoundaryAccuracy(row['r'], row['lsatInt'], row['y'], row['y_hat'], d=4), axis=1)
X_test['rd2'] = X_test.apply(lambda row: rBoundaryAccuracy2(row['r'], row['lsatInt'], d=4), axis=1)


X_train['r'] = predictions_train_orig
X_train[score] = dataset.loc[X_train.index][score]
X_train['y_hat'] = X_train['r'].apply(success_func)
X_train['y'] = X_train[score].apply(success_func)
X_train['rd'] = X_train.apply(lambda row: rBoundaryAccuracy(row['r'], row['lsatInt'], row['y'], row['y_hat'], d=4), axis=1)
X_train['rd2'] = X_train.apply(lambda row: rBoundaryAccuracy2(row['r'], row['lsatInt'], d=4), axis=1)

d1 = eval(subset)

d = dataset.loc[d1.index].copy()
d['r'] = d1['r']
d['y'] = d1['y']
d['y_hat'] = d1['y_hat']
d['rd'] = d1['rd']
d['rd2'] = d1['rd2']

f = fm(d, group, 'y', 'y_hat', 'r', success_func)

ts = f.measure(measure='ts', result='matrix')
sp = f.measure(measure='sp', result='matrix')
ta = f.measure(measure='ta', result='matrix')
ca = f.measure(measure='ca', result='matrix')

header = sorted(d[group].unique().tolist())
pd.options.display.float_format = '{:,.3f}'.format

# Compute accuracy
true_positive = len(d[(d['y'] == 1) & (d['y_hat'] == 1)])
true_negative = len(d[(d['y'] == 0) & (d['y_hat'] == 0)])
accuracy = (true_positive + true_negative) / len(d)

r_accuracy = accuracy_score(d['lsatInt'], d['r'])
y_accuracy = accuracy_score(d['y'], d['y_hat'])
rdaccuracy = sum(d['rd']) / len(d)
rd2accuracy = sum(d['rd2']) / len(d)

print ("==============================================")
print (f'Algorithm is {algorithm.__class__.__name__}')
print (f'File is {file}')
print (f'Subset is {subset}')
print (f'Sensible attribute is {group}')
print (f'Score is {score}')
print (f'old Y-Accuracy of success prediction {accuracy}')
print (f'Y-Accuracy of success prediction {y_accuracy}')
print (f'R-Accuracy of score prediction {r_accuracy}')
print (f'R-Distance-bounded Accuracy of score prediction {rdaccuracy}')
print (f'R2-Distance not bounded Accuracy of score prediction {rd2accuracy}')

print(f'n_classes: {n_classes}')

print (f'Top-1 accuracy (test)  : {top1_test:.4f}')
print (f'Top-3 accuracy (test)  : {top3_test:.4f}')
print (f'Top-5 accuracy (test)  : {top5_test:.4f}')
print (f'Top-1 accuracy (train) : {top1_train:.4f}')
print (f'Top-3 accuracy (train) : {top3_train:.4f}')
print (f'Top-5 accuracy (train) : {top5_train:.4f}')


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



