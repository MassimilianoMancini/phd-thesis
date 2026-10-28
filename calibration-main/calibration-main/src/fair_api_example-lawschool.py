import pandas as pd
import kagglehub
from fairnessMeasuresAPI import fm
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split



# Success function
def success_func(r):
    return 1 if r > 29 else 0


# Download latest version of law school dataset
path = kagglehub.dataset_download("danofer/law-school-admissions-bar-passage")
file = path + "/bar_pass_prediction.csv"

group = 'race1'
score = 'lsatInt'
subset = 'X_test'
algorithm = DecisionTreeClassifier()

toDrop     = ['decile1b', 'decile3', 'ID', 'decile1', 'race', 'cluster', 'lsat', 'ugpa', 'zfygpa', 'grad', 'zgpa', 'bar1', 'bar1_yr', 'bar2', 'bar2_yr', 'age', 'gender', 'male', 'race2', 'bar', 'Dropout', 'other', 'asian', 'black', 'hisp', 'pass_bar', 'bar_passed', 'tier', 'index6040',   'indxgrp', 'indxgrp2', 'gpa', 'dnn_bar_pass_prediction', 'gpa' ]

dataset = pd.read_csv(file, sep=',', header='infer')


dataset = dataset[dataset['race1'].notna()]
dataset = dataset[dataset['lsat'].notna()]

dataset['lsatInt'] = dataset['lsat'].apply(lambda x: round(x))
dataset['DOB_yr'] = dataset['DOB_yr'].apply(lambda x: x + 1900)

dataset = dataset.drop(toDrop, axis=1)

print(dataset.race1.value_counts())


X = pd.get_dummies(dataset.drop(score, axis=1))
y = dataset[score]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
algorithm.fit(X_train, y_train)
predictions_test = algorithm.predict(X_test)
predictions_train = algorithm.predict(X_train)

X_test['r'] = predictions_test + 12
X_test[score] = dataset.loc[X_test.index][score]
X_test['y_hat'] = X_test['r'].apply(success_func)
X_test['y'] = X_test[score].apply(success_func)

X_train['r'] = predictions_train + 12
X_train[score] = dataset.loc[X_train.index][score]
X_train['y_hat'] = X_train['r'].apply(success_func)
X_train['y'] = X_train[score].apply(success_func)

d1 = eval(subset)

d = dataset.loc[d1.index].copy()
d['r'] = d1['r']
d['y'] = d1['y']
d['y_hat'] = d1['y_hat']

f = fm(d, group, 'y', 'r', 'y_hat', success_func)

tsp = f.measure(measure='ts')
sp = f.measure(measure='sp')
ta = f.measure(measure='ta')
ca = f.measure(measure='ca')
al = f.measure(measure='all')

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
print (f'Accuracy of success prediction {accuracy}')

print ("==============================================")

print ("Classes:")
for index, value in enumerate(header):
    print(f"{index}:{value}")

print ("==============================================")

print ('True statistical parity')
print ('Pairwise matrix')
print (pd.DataFrame(tsp))
print ('maxdiff')
print (f.measure(measure='ts', result='maxdiff'))
print ('pca')
print (f.measure(measure='ts', result='pca'))


print ("==============================================")
print ('Statistical parity')
print ('Pairwise matrix')
print (pd.DataFrame(sp))
print ('maxdiff')
print (f.measure(measure='sp', result='maxdiff'))
print ('pca')
print (f.measure(measure='sp', result='pca'))

print ("==============================================")
print ('Total accuracy')
print ('Pairwise matrix')
print (pd.DataFrame(ta))
print ('maxdiff')
print (f.measure(measure='ta', result='maxdiff'))
print ('pca')
print (f.measure(measure='ta', result='pca'))

print ("==============================================")
print ('Calibration')
print ('Pairwise matrix')
print (pd.DataFrame(ca))
print ('maxdiff')
print (f.measure(measure='ca', result='maxdiff'))
print ('pca')
print (f.measure(measure='ca', result='pca'))

print ("==============================================")
print ('All')
print ('Pairwise matrix')
print (pd.DataFrame(al))
print ('maxdiff')
print (f.measure(measure='all', result='maxdiff'))
print ('pca')
print (f.measure(measure='all', result='pca'))

print ("==============================================")
print ("==============================================")
print()



