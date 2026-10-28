import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def split_data(df, target_col="Y", test_size=0.2, random_state=42):

    y = df[target_col]
    X = df.drop([target_col], axis=1, errors="ignore")
    
    X_train, X_test, _, _ = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )
    df.loc[X_test.index, "SPLIT"] = "test"
    df.loc[X_train.index, "SPLIT"] = "train"    
    return df

def train_xgboost_classifier(df, target_col="Y", random_state=42):
    features_df = split_data(df)
    features_df = df.drop(columns=["CODICESCUOLA", "2026", "DIFFERENCE", "INCREMENT", "R"], errors="ignore")
    features_df_train = features_df[features_df["SPLIT"] == "train"]
    features_df_test = features_df[features_df["SPLIT"] == "test"]

    y_train = features_df_train[target_col]
    X_train = features_df_train.drop([target_col, "SPLIT"], axis=1, errors="ignore")
    X_test = features_df_test.drop([target_col, "SPLIT"], axis=1, errors="ignore")
    y_test_true = features_df_test[target_col]

    X_all = pd.get_dummies(pd.concat([X_train, X_test], axis=0), dummy_na=False)
    X_train = X_all.loc[X_train.index]
    X_test = X_all.loc[X_test.index]

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalid = xgb.DMatrix(X_test, label=y_test_true)

    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "max_depth": 4,
        "eta": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "gamma": 1,
        "alpha": 0.5,
        "lambda": 1.0,
        "seed": random_state,
    }

    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=200,
        evals=[(dvalid, "valid")],
        early_stopping_rounds=10,
        verbose_eval=False,
    )

    y_test_pred = (booster.predict(dvalid) >= 0.5).astype(int)
    y_train_pred = (booster.predict(dtrain) >= 0.5).astype(int)

    xgb_predictions = pd.Series(pd.NA, index=df.index, dtype="Int64")
    xgb_predictions.loc[features_df_train.index] = y_train_pred
    xgb_predictions.loc[features_df_test.index] = y_test_pred
    df["XGB"] = xgb_predictions.astype("Int64")

    return df

def train_random_forest_classifier(df, target_col="Y", random_state=42):
    features_df = split_data(df)
    features_df = df.drop(columns=["CODICESCUOLA", "2026", "DIFFERENCE", "INCREMENT", "R"], errors="ignore")
    features_df_train = features_df[features_df["SPLIT"] == "train"]
    features_df_test = features_df[features_df["SPLIT"] == "test"]

    y_train = features_df_train[target_col]
    X_train = features_df_train.drop([target_col, "SPLIT"], axis=1, errors="ignore")
    X_test = features_df_test.drop([target_col, "SPLIT"], axis=1, errors="ignore")

    X_all = pd.get_dummies(pd.concat([X_train, X_test], axis=0), dummy_na=False)
    X_train = X_all.loc[X_train.index]
    X_test = X_all.loc[X_test.index]

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features="sqrt",
        bootstrap=True,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_test_pred = model.predict(X_test)
    y_train_pred = model.predict(X_train)

    rf_predictions = pd.Series(pd.NA, index=df.index, dtype="Int64")
    rf_predictions.loc[features_df_train.index] = y_train_pred
    rf_predictions.loc[features_df_test.index] = y_test_pred
    df["RF"] = rf_predictions.astype("Int64")

    return df


def predict_with_xgboost(df):
    return train_xgboost_classifier(df)

def predict_with_random_forest(df):
    return train_random_forest_classifier(df)