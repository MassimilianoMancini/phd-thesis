import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from SimpleTree import SimpleTree

# Load dataset Iris
iris = load_iris()
df_iris = pd.DataFrame(iris.data, columns=iris.feature_names)
df_iris['target'] = iris.target

# Creazione e addestramento dell'albero decisionale
tree = SimpleTree()

X_train, X_test, y_train, y_test = train_test_split(df_iris, iris.target, test_size=0.2, random_state=42)


tree.fit(X_train, 'target')

print(tree)

predictions = tree.predict(X_test)

print("Predizioni sui nuovi dati:", predictions.tolist())
print("Valori reali:", y_test.tolist())
