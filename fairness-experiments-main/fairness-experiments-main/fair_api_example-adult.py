import pandas as pd
from fairness_measures_api import fairness_measures_api
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from seaborn import heatmap
import matplotlib.pyplot as plt
import kagglehub


# Download latest version of law school dataset
path = kagglehub.dataset_download("uciml/adult-census-income")
file = path + "/adult.csv"

# Main
group = 'race'
score = 'income'
subset = 'X_test'
algorithm = XGBClassifier()

dataset = pd.read_csv(file, sep=',', header='infer')

dataset = dataset[dataset['race'].notna()]
dataset = dataset[dataset['income'].notna()]

dataset['income'] = dataset['income'].apply(lambda x: 1 if x == '>50K' else 0)


X = pd.get_dummies(dataset.drop(score, axis=1))
le = LabelEncoder()
y = le.fit_transform(dataset[score])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
y_train = le.fit_transform(y_train)
algorithm.fit(X_train, y_train)
predictions_test = algorithm.predict(X_test)
predictions_train = algorithm.predict(X_train)

X_test['y_hat'] = predictions_test
X_test['y'] = dataset.loc[X_test.index][score]

X_train['y_hat'] = predictions_train
X_train['y'] = dataset.loc[X_train.index][score]

d1 = eval(subset)

d = dataset.loc[d1.index].copy()
d['y'] = d1['y']
d['y_hat'] = d1['y_hat']

f = fairness_measures_api(d, group, 'y', None, 'y_hat', None)

tsp = f.true_statistical_parity()
sp = f.statistical_parity()
ta = f.total_accuracy()

header = sorted(d[group].unique().tolist())
pd.options.display.float_format = '{:,.3f}'.format

# Compute accuracy
true_positive = len(X_test[(X_test['y'] == 1) & (X_test['y_hat'] == 1)])
true_negative = len(X_test[(X_test['y'] == 0) & (X_test['y_hat'] == 0)])
accuracy = (true_positive + true_negative) / len(X_test)

print ("==============================================")
print (f'Algorithm is {algorithm}')
print (f'File is {file}')
print (f'Subset is {subset}')
print (f'Sensible attribute is {group}')
print (f'Score is {score}')
print (f'Accuracy of prediction {accuracy}')

print ("==============================================")

print ("Classes:")
for index, value in enumerate(header):
    print(f"{index}:{value}")

print ("==============================================")

print ('True statistical parity')
print ('Pairwise matrix')
print (pd.DataFrame(tsp))
plt.figure(figsize=(10, 10))
heatmap(pd.DataFrame(tsp), cmap="coolwarm", cbar=False, annot=True, fmt=".3f", xticklabels=header, yticklabels=header)
plt.savefig('api/heatmaps/tsp-' + score + '.png')
plt.close()

print ("==============================================")
print ('Statistical parity')
print ('Pairwise matrix')
print (pd.DataFrame(sp))
plt.figure(figsize=(10, 10))
heatmap(pd.DataFrame(sp), cmap="coolwarm", cbar=False, annot=True, fmt=".3f", xticklabels=header, yticklabels=header)
plt.savefig('api/heatmaps/sp-' + score + '.png')
plt.close()

print ("==============================================")
print ('Total accuracy')
print ('Pairwise matrix')
print (pd.DataFrame(ta))
plt.figure(figsize=(10, 10))
heatmap(pd.DataFrame(ta), cmap="coolwarm", cbar=False, annot=True, fmt=".3f", xticklabels=header, yticklabels=header)
plt.savefig('api/heatmaps/ta-' + score + '.png')
plt.close()

print ("==============================================")
print ("==============================================")
print()