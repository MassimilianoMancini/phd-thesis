import pandas as pd
import kagglehub
import numpy as np
from fairnessMeasuresAPI import fm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from seaborn import heatmap

# Import dal tuo package
from MLP.SimpleMLP import SimpleMLPClassifier


# Success function
def success_func(r):
    return 1 if r > 29 else 0


# ----------------------------
# Download + load dataset
# ----------------------------
path = kagglehub.dataset_download("danofer/law-school-admissions-bar-passage")
file = path + "/bar_pass_prediction.csv"

group = 'race1'
score = 'lsatInt'
subset = 'X_test'

toDrop = [
    'decile1b', 'decile3', 'ID', 'decile1', 'race', 'cluster', 'lsat', 'ugpa', 'zfygpa',
    'grad', 'zgpa', 'bar1', 'bar1_yr', 'bar2', 'bar2_yr', 'age', 'gender', 'male', 'race2',
    'bar', 'Dropout', 'other', 'asian', 'black', 'hisp', 'pass_bar', 'bar_passed', 'tier',
    'index6040', 'indxgrp', 'indxgrp2', 'gpa', 'dnn_bar_pass_prediction', 'gpa'
]

dataset = pd.read_csv(file, sep=',', header='infer')

# Target: lsat rounded to int
dataset['lsatInt'] = dataset['lsat'].apply(lambda x: round(x))

# Fix DOB year (come nel tuo script)
dataset['DOB_yr'] = dataset['DOB_yr'].apply(lambda x: x + 1900)

# Drop colonne e NaN
dataset = dataset.drop(toDrop, axis=1)
dataset.dropna(inplace=True)

print(dataset.race1.value_counts())


# ----------------------------
# One-hot + label encoding
# ----------------------------
X_df = pd.get_dummies(dataset.drop(score, axis=1))

le = LabelEncoder()
y_encoded = le.fit_transform(dataset[score])  # class indices per MLP

X_train_df, X_test_df, y_train, y_test = train_test_split(
    X_df, y_encoded, test_size=0.2, random_state=42
)

# ----------------------------
# Scaling (importante per MLP)
# ----------------------------
scaler = StandardScaler(with_mean=True, with_std=True)
X_train_scaled = scaler.fit_transform(X_train_df.values.astype(np.float32))
X_test_scaled  = scaler.transform(X_test_df.values.astype(np.float32))


# ----------------------------
# Train SimpleMLP
# ----------------------------
algorithm = SimpleMLPClassifier(
    hidden=(128, 64),
    dropout=0.2,
    lr=1e-3,
    weight_decay=1e-4,
    batch_size=256,
    epochs=30,
    verbose=True
)

algorithm.fit(X_train_scaled, y_train)

predictions_test = algorithm.predict(X_test_scaled)
predictions_train = algorithm.predict(X_train_scaled)

# Convert predictions back to original lsatInt values
predictions_test_orig = le.inverse_transform(predictions_test)
predictions_train_orig = le.inverse_transform(predictions_train)


# ----------------------------
# Ricostruisci DF per fairness (come nel tuo flusso)
# ----------------------------
X_test = X_test_df.copy()
X_train = X_train_df.copy()

X_test['r'] = predictions_test_orig
X_test[score] = dataset.loc[X_test.index][score]
X_test['y_hat'] = X_test['r'].apply(success_func)
X_test['y'] = X_test[score].apply(success_func)

X_train['r'] = predictions_train_orig
X_train[score] = dataset.loc[X_train.index][score]
X_train['y_hat'] = X_train['r'].apply(success_func)
X_train['y'] = X_train[score].apply(success_func)

d1 = eval(subset)

d = dataset.loc[d1.index].copy()
d['r'] = d1['r']
d['y'] = d1['y']
d['y_hat'] = d1['y_hat']


# ----------------------------
# Fairness measures
# ----------------------------
f = fm(d, group, 'y', 'y_hat', 'r', success_func)

ts = f.measure(measure='ts', result='matrix')
sp = f.measure(measure='sp', result='matrix')
ta = f.measure(measure='ta', result='matrix')
ca = f.measure(measure='ca', result='matrix')

header = sorted(d[group].unique().tolist())
pd.options.display.float_format = '{:,.3f}'.format


# ----------------------------
# Accuracy (success prediction)
# ----------------------------
true_positive = len(X_test[(X_test['y'] == 1) & (X_test['y_hat'] == 1)])
true_negative = len(X_test[(X_test['y'] == 0) & (X_test['y_hat'] == 0)])
accuracy = (true_positive + true_negative) / len(X_test)

print("==============================================")
print(f'Algorithm is {algorithm.__class__.__name__}')
print(f'File is {file}')
print(f'Subset is {subset}')
print(f'Sensible attribute is {group}')
print(f'Score is {score}')
print(f'Accuracy of success prediction {accuracy}')
print("==============================================")


print("Classes:")
for index, value in enumerate(header):
    print(f"{index}:{value}")

# ----------------------------
# Plots (heatmaps) + latex
# ----------------------------
print("==============================================")
print('True statistical parity')
print('Pairwise matrix')
print(pd.DataFrame(ts))
print(f.measure(measure='ts', result='latex'))
plt.clf()
hm = heatmap(ts, cmap="coolwarm", cbar=False, annot=True, fmt=".3f",
            xticklabels=header + ['1-vs-all'], yticklabels=header)
hm.figure.savefig('mlp-ts.png')

print("==============================================")
print('Statistical parity')
print('Pairwise matrix')
print(pd.DataFrame(sp))
print(f.measure(measure='sp', result='latex'))
plt.clf()
hm = heatmap(sp, cmap="coolwarm", cbar=False, annot=True, fmt=".3f",
            xticklabels=header + ['1-vs-all'], yticklabels=header)
hm.figure.savefig('mlp-sp.png')

print("==============================================")
print('Total accuracy')
print('Pairwise matrix')
print(pd.DataFrame(ta))
print(f.measure(measure='ta', result='latex'))
plt.clf()
hm = heatmap(ta, cmap="coolwarm", cbar=False, annot=True, fmt=".3f",
            xticklabels=header + ['1-vs-all'], yticklabels=header)
hm.figure.savefig('mlp-ta.png')

print("==============================================")
print('Calibration')
print('Pairwise matrix')
print(pd.DataFrame(ca))
print(f.measure(measure='ca', result='latex'))
plt.clf()
hm = heatmap(ca, cmap="coolwarm", cbar=False, annot=True, fmt=".3f",
            xticklabels=header + ['1-vs-all'], yticklabels=header)
hm.figure.savefig('mlp-ca.png')

print("==============================================")
print("Done.")
