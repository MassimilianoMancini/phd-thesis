import xgboost, inspect
print('version', xgboost.__version__)
print('init', inspect.signature(xgboost.XGBClassifier.__init__))
print('fit', inspect.signature(xgboost.XGBClassifier.fit))
