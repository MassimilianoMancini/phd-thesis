import pandas as pd
import os
from fairness_measures_api import fairness_measures_api
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

def cal(file, subset, algorithm, g):

    data = pd.read_csv(file, sep=';', header='infer')
    
    X = pd.get_dummies(data.drop('G3', axis=1))
    le = LabelEncoder()
    y = le.fit_transform(data['G3'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = algorithm
    model.fit(X_train, y_train)
    predictions_test = model.predict(X_test)
    predictions_train = model.predict(X_train)

    X_test['r'] = predictions_test
    X_test['G3'] = data.iloc[X_test.index]['G3']
    X_test['y_hat'] = (X_test['r'] >= 10).astype('int')
    X_test['y'] = (X_test['G3'] >= 10).astype('int')

    X_train['r'] = predictions_train
    X_train['G3'] = data.iloc[X_train.index]['G3']
    X_train['y_hat'] = (X_train['r'] >= 10).astype('int')
    X_train['y'] = (X_train['G3'] >= 10).astype('int')

    d1 = eval(subset)

    d = data.iloc[d1.index].copy()
    d['r'] = d1['r']
    d['G3'] = d1['G3']
    d['y'] = d1['y']
    d['y_hat'] = d1['y_hat']

    f = fairness_measures_api(d, g, 'y', 'r', 'y_hat')

    tsp = f.true_statistical_parity()
    sp = f.statistical_parity()
    ta = f.total_accuracy()
    ca = f.calibration()

    header = sorted(d[g].unique().tolist())
    pd.options.display.float_format = '{:,.3f}'.format
    
    print ("==============================================")
    print (f'Algorithm is {model.__class__.__name__}')
    print (f'File is {file}')
    print (f'Subset is {subset}')
    print (f'Sensible attribute is {g}')

    print ("==============================================")

    print ('True statistical parity')
    print ('Pairwise matrix')
    print (pd.DataFrame(tsp[0], header, header))
    print (f'Mean: {tsp[2]:.3f}')
    print (f'Variance: {tsp[3]:.3f}')
    print (f'Max: {tsp[7]:.3f} given by {tsp[10]} - {tsp[11]}')
    print (f'Min: {tsp[6]:.3f} given by {tsp[8]} - {tsp[9]}')
    print ('One against others')
    print (pd.DataFrame(tsp[1], header, [' ']))
    print (f'Mean: {tsp[4]:.3f}')
    print (f'Variance: {tsp[5]:.3f}')

    print ("==============================================")
    print ('Statistical parity')
    print ('Pairwise matrix')
    print (pd.DataFrame(sp[0], header, header))
    print (f'Mean: {sp[2]:.3f}')
    print (f'Variance: {sp[3]:.3f}')
    print (f'Max: {sp[7]:.3f} given by {sp[10]} - {sp[11]}')
    print (f'Min: {sp[6]:.3f} given by {sp[8]} - {sp[9]}')
    print ('One against others')
    print (pd.DataFrame(sp[1], header, [' ']))
    print (f'Mean: {sp[4]:.3f}')
    print (f'Variance: {sp[5]:.3f}')
    print ("==============================================")
    print ('Total accuracy')
    print ('Pairwise matrix')
    print (pd.DataFrame(ta[0], header, header))
    print (f'Mean: {ta[2]:.3f}')
    print (f'Variance: {ta[3]:.3f}')
    print (f'Max: {ta[7]:.3f} given by {ta[10]} - {ta[11]}')
    print (f'Min: {ta[6]:.3f} given by {ta[8]} - {ta[9]}')
    print ('One against others')
    print (pd.DataFrame(ta[1], header, [' ']))
    print (f'Mean: {ta[4]:.3f}')
    print (f'Variance: {ta[5]:.3f}')
    print ("==============================================")
    print ('Calibration')
    print ('Pairwise matrix')
    print (pd.DataFrame(ca[0], header, header))
    print (f'Mean: {ca[2]:.3f}')
    print (f'Variance: {ca[3]:.3f}')
    print (f'Max: {ca[7]:.3f} given by {ca[10]} - {ca[11]}')
    print (f'Min: {ca[6]:.3f} given by {ca[8]} - {ca[9]}')
    print ('One against others')
    print (pd.DataFrame(ca[1], header, [' ']))
    print (f'Mean: {ca[4]:.3f}')
    print (f'Variance: {ca[5]:.3f}')
    print ("==============================================")
    print ("==============================================")
    print()


# Main
files      = ['student-mat.csv']
groups     = ['Medu', 'sex']
subsets    = ['X_test']
algorithms = [XGBClassifier()]

os.system('cls')

for f in files:
    for g in groups:
        for s in subsets:
            for a in algorithms:
                cal(f, s, a, g)