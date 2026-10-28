import pandas as pd
import os
from fairnessMeasuresAPI import fm
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn import svm

# Success function
def success_func(r):
    return 1 if r >= 10 else 0

file = 'student-mat.csv'
subset = "X_test"
group = "sex"
algorithm = XGBClassifier()

# Load the dataset
data = pd.read_csv(f'src/{file}', sep=';', header='infer')

# Drop columns with strong correlation
# data = data.drop(['G1', 'G2'], axis=1)

X = pd.get_dummies(data.drop('G3', axis=1))
le = LabelEncoder()
y = le.fit_transform(data['G3'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

algorithm.fit(X_train, y_train)

# Predizioni
predictions_test = algorithm.predict(X_test)
predictions_train = algorithm.predict(X_train)

# Decodifica dei valori predetti
original_preds_test = le.inverse_transform(predictions_test)
original_preds_train = le.inverse_transform(predictions_train)

# Decodifica anche i valori reali
original_y_test = le.inverse_transform(y_test)
original_y_train = le.inverse_transform(y_train)

# Ora original_preds_test corrisponde a X_test, original_preds_train a X_train

X_test['r'] = original_preds_test
X_test['G3'] = original_y_test
X_test['y_hat'] = X_test['r'].apply(success_func)
X_test['y'] = X_test['G3'].apply(success_func)

X_train['r'] = original_preds_train
X_train['G3'] = original_y_train
X_train['y_hat'] = X_train['r'].apply(success_func)
X_train['y'] = X_train['G3'].apply(success_func)



d1 = eval(subset)

d = data.loc[eval(subset).index].copy()
d['r'] = d1['r']
d['y'] = d1['y']
d['y_hat'] = d1['y_hat']

f = fm(d, group, 'y', 'y_hat', 'r', success_func)

sp = f.measure(measure='sp')
ta = f.measure(measure='ta')
ca = f.measure(measure='ca')

pd.options.display.float_format = '{:,.3f}'.format

accuracy = accuracy_score(original_y_test, original_preds_test)
y_accuracy = accuracy_score(d['y'], d['y_hat'])


print ("==============================================")
print (f'Algorithm is {algorithm.__class__.__name__}')
print (f'File is {file}')
print (f'Subset is {subset}')
print (f'Sensible attribute is {group}')
print (f'Target attribute is G3')
print (f'Accuracy on r {accuracy}')
print (f'Accuracy on y_hat {y_accuracy}')
print (f'Groups order is {f.gs}')
print (f'Positive values are in favor of group {f.gs[0]}')
print (f'Success function is 1 if r >= 10 else 0')
print ("==============================================")
print ("==============================================")
print ('Statistical parity')
print (sp)
print ("==============================================")
print ('Total accuracy')
print (ta)
print ("==============================================")
print ('Calibration')
print (ca)
print ("==============================================")
print ("==============================================")
print()
