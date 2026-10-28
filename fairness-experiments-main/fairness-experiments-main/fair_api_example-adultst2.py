import pandas as pd
import kagglehub
from fairnessMeasuresAPI import fm
from SimpleFairTree import SimpleFairTree
from sklearn.model_selection import train_test_split



# Download latest version of law school dataset
path = kagglehub.dataset_download("uciml/adult-census-income")
file = path + "/adult.csv"

# Main
group = 'sex'
score = 'income'
subset = 'X_test'

sft = SimpleFairTree()

dataset = pd.read_csv(file, sep=',', header='infer')


dataset = dataset[dataset['sex'].notna()]
dataset = dataset[dataset['income'].notna()]


print(dataset.sex.value_counts())


X = pd.get_dummies(dataset.drop(score, axis=1))
y = dataset[score]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
sft.fit(X_train, y_train)
predictions_test = sft.predict(X_test)
predictions_train = sft.predict(X_train)

X_test['y_hat'] = predictions_test
X_test['y'] = dataset.loc[X_test.index][score]

X_train['y_hat'] = predictions_train
X_train['y'] = dataset.loc[X_train.index][score]


d1 = eval(subset)

d = dataset.loc[d1.index].copy()
d['y'] = d1['y']
d['y_hat'] = d1['y_hat']

f = fm(dataset=d, sensible_group=group, true_success='y', prediction='y_hat')

tsp = f.measure(measure='ts')
sp = f.measure(measure='sp')
ta = f.measure(measure='ta')


header = sorted(d[group].unique().tolist())
pd.options.display.float_format = '{:,.3f}'.format

# Compute accuracy
true_positive = len(X_test[(X_test['y'] == 1) & (X_test['y_hat'] == 1)])
true_negative = len(X_test[(X_test['y'] == 0) & (X_test['y_hat'] == 0)])
accuracy = (true_positive + true_negative) / len(X_test)


print ("==============================================")
print (f'Algorithm is SimpleTree')
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


print ("==============================================")
print ('Statistical parity')
print ('Pairwise matrix')
print (pd.DataFrame(sp))
print ('maxdiff')
print (f.measure(measure='sp', result='maxdiff'))

print ("==============================================")
print ('Total accuracy')
print ('Pairwise matrix')
print (pd.DataFrame(ta))
print ('maxdiff')
print (f.measure(measure='ta', result='maxdiff'))


print ("==============================================")
print ("==============================================")
print()



