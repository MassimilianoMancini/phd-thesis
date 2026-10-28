import importlib
import time
import numpy as np


def _get_pd(retries: int = 5, delay: float = 0.5):
    for attempt in range(retries):
        try:
            pd = importlib.import_module("pandas")
            return pd
        except KeyboardInterrupt:
            if attempt == retries - 1:
                raise
            time.sleep(delay)
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(delay)

from data_import import (
    import_files,
    clean_data,
    remove_columns,
    group_by_year,
    create_target_category,
    apply_target_bins_to_column,
    count_target_rows,
    import_school_registry,
    find_unmatched_grouped,
    merge_registry_and_grouped,
    calculate_statistical_parity,
)


def train_xgboost_classifier(final_df, target_col="TARGET", test_size=0.2, random_state=42):
    pd = _get_pd()
    try:
        import xgboost as xgb
    except ImportError as exc:
        raise ImportError("xgboost is required to train the classifier. Install it with 'pip install xgboost'.") from exc

    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    features_df = final_df.drop(columns=["CODICESCUOLA", target_col, 2026], errors="ignore")
    features_df = features_df.select_dtypes(include=["number"])
    if features_df.empty:
        raise ValueError("No numeric features available for XGBoost training.")

    data = pd.concat([features_df, final_df[target_col]], axis=1).dropna()
    X = data[features_df.columns]
    # Factorize target so labels are 0..n-1 for the classifier and keep uniques to map back
    y, uniques = pd.factorize(data[target_col])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    # Calcola y_hat per il dataset di test: se target_2025 < TARGET allora y_hat = -1 altrimenti y_hat = 1
    try:
        test_idx = X_test.index
        test_df = final_df.loc[test_idx].copy()
        test_df["y_hat"] = np.where(test_df["target_2025"] > test_df[target_col], -1, 1)
        n_pos = int((test_df["y_hat"] == 1).sum())
        n_neg = int((test_df["y_hat"] == -1).sum())
    except Exception:
        test_df = pd.DataFrame()
        n_pos = 0
        n_neg = 0

    print(f"Test set - Conteggio y_hat=1: {n_pos} | y_hat=-1: {n_neg}")
    print("Test set head y_hat=1:")
    try:
        print(test_df[test_df["y_hat"] == 1].head())
    except Exception:
        print(pd.DataFrame())
    print("Test set head y_hat=-1:")
    try:
        print(test_df[test_df["y_hat"] == -1].head())
    except Exception:
        print(pd.DataFrame())

    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=random_state)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"XGBoost accuracy: {accuracy:.4f}")
    return model, accuracy, uniques


def main():
    file_names = [
        "data/BISPROGANNO201920240925.csv",
        "data/BISPROGANNO202020240925.csv",
        "data/BISPROGANNO202120240925.csv",
        "data/BISPROGANNO202220240925.csv",
        "data/BISPROGANNO202320241220.csv",
        "data/BISPROGANNO202420251220.csv",
        "data/BISPROGANNO202520260320.csv",
        "data/BISPROGANNO202620260421.csv",
    ]

    combined_df = import_files(file_names)
    cleaned_df = clean_data(combined_df)
    cleaned_df = remove_columns(cleaned_df)

    grouped_df = group_by_year(cleaned_df)
    target_bins = create_target_category(grouped_df)
    
    # Crea target_2025 usando gli stessi bin di TARGET
    grouped_df = apply_target_bins_to_column(grouped_df, 2025, target_bins, "target_2025")

    print("Intervalli dei 4 bin TARGET:")
    print(target_bins)

    print("Conteggio righe per TARGET:")
    print(count_target_rows(grouped_df))

    registry_files = [
        "data/SCUANAGRAFESTAT20262720260901.csv",
        "data/SCUANAGRAFESTAT20252620250901.csv",
        "data/SCUANAGRAFESTAT20242520250831.csv",
        "data/SCUANAGRAFESTAT20232420240831.csv",
        "data/SCUANAGRAFESTAT20222320230831.csv",
        "data/SCUANAGRAFESTAT20212220220831.csv",
        "data/SCUANAGRAFESTAT20202120210831.csv",
        "data/SCUANAGRAFESTAT20192020200831.csv",
        "data/SCUANAGRAFESTAT20181920190831.csv",
    ]
    registry_df = import_school_registry(registry_files)

    unmatched_df = find_unmatched_grouped(grouped_df, registry_df)
    print("Righe di grouped_df senza corrispondenza nel registry:")
    print(unmatched_df)

    final_df = merge_registry_and_grouped(grouped_df, registry_df)
    print("Dataset unito con colonne del registry e grouped:")
    print(final_df.head())

    print("Avvio training XGBoost su final_df...")
    model, accuracy, uniques = train_xgboost_classifier(final_df)

    # Predici target_hat su tutto il dataset usando il modello addestrato
    pd = _get_pd()
    features_all = final_df.drop(columns=["CODICESCUOLA", "TARGET", 2026], errors="ignore")
    features_all = features_all.select_dtypes(include=["number"])

    # Predici solo sulle righe senza NaN nelle feature numeriche
    valid_idx = features_all.dropna().index
    target_hat_series = pd.Series(index=final_df.index, dtype=float)
    if len(valid_idx) > 0:
        preds_codes = model.predict(features_all.loc[valid_idx])
        # map prediction codes back to original TARGET values
        preds = uniques[preds_codes]
        target_hat_series.loc[valid_idx] = preds

    final_df["target_hat"] = target_hat_series

    # Calcola y_hat: se target_2025 < target_hat allora y_hat = -1 altrimenti y_hat = 1
    final_df["y_hat"] = np.where(final_df["target_2025"] > final_df["target_hat"], -1, 1)

    # Stampa riepilogo
    n_pos = int((final_df["y_hat"] == 1).sum())
    n_neg = int((final_df["y_hat"] == -1).sum())
    print(f"Conteggio y_hat=1: {n_pos} | y_hat=-1: {n_neg}")

    print(final_df.head())
    print(final_df[final_df["y_hat"] == 1].head())
    print(final_df[final_df["y_hat"] == -1].head())
    print(final_df["TARGET"].value_counts())

    # Ciclo su attributi per calcolare statistical parity
    print("\n" + "="*80)
    print("REPORT STATISTICAL PARITY")
    print("="*80 + "\n")
    
    attributes = [
        "AREAGEOGRAFICA",
        "REGIONE",
        "PROVINCIA",
        "DESCRIZIONETIPOLOGIAGRADOISTRUZIONESCUOLA",
    ]
    
    for attribute in attributes:
        print(f"\n{'='*80}")
        print(f"STATISTICAL PARITY - {attribute}")
        print(f"{'='*80}")
        
        try:
            sp_report = calculate_statistical_parity(final_df, attribute, y_hat_col="y_hat")
            print(sp_report.to_string(index=False))
            print(f"\nMedia Statistical Parity per {attribute}: {sp_report['statistical_parity'].mean():.4f}")
            print(f"Deviazione Standard: {sp_report['statistical_parity'].std():.4f}")
            print(f"Min: {sp_report['statistical_parity'].min():.4f}, Max: {sp_report['statistical_parity'].max():.4f}")
        except Exception as e:
            print(f"Errore nel calcolo della statistical parity per {attribute}: {str(e)}")

if __name__ == "__main__":
    main()
