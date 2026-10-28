import csv
import os
from datetime import datetime
import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression

def read_files():
    file_names = [
        "data/BISPROGANNO201920240925.csv",
        "data/BISPROGANNO202020240925.csv", 
        "data/BISPROGANNO202120240925.csv", 
        "data/BISPROGANNO202220240925.csv", 
        "data/BISPROGANNO202320241220.csv", 
        "data/BISPROGANNO202420251220.csv",
        "data/BISPROGANNO202520260320.csv",
        "data/BISPROGANNO202620260421.csv"]

    date_format = "%Y-%m-%d"
    ANNO_MIN, ANNO_MAX = 2019, 2026
    valid_rows = []

    for filename in file_names:

        with open(filename, mode="r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f) 
            for row in reader:
                if not all(value is not None and value.strip() != "" for value in row.values()):
                    continue

                if row["TIPOLOGIAVOCE"] != "ENTRATA":
                    continue
                
                try:
                    amount = float(row["IMPORTO"].replace(",", "."))
                except (ValueError, KeyError):
                    continue

                try:
                    parsed_date = datetime.strptime(row["DATAAPPROVAZIONECONSIGLIO"], date_format)
                except (ValueError, KeyError):
                    continue
                
                if amount <= 0:
                    continue

                if not (ANNO_MIN <= parsed_date.year <= ANNO_MAX):
                    continue
                row["IMPORTO"] = amount
                row["DATAAPPROVAZIONECONSIGLIO"] = parsed_date
                valid_rows.append(row)

    return pd.DataFrame(valid_rows)

def group_rows_by_month_year(df):
    df["ANNO"] = df["DATAAPPROVAZIONECONSIGLIO"].dt.year
    df["MESE"] = df["DATAAPPROVAZIONECONSIGLIO"].dt.month

    return (
        df.groupby(["ANNO", "MESE", "REGIONE", "TIPOLOGIAVOCE"])["IMPORTO"]
        .sum()
        .reset_index()
    )

def build_features(df):
    df = df.sort_values(["REGIONE", "ANNO", "MESE"]).copy()

    df["MESE_SIN"] = np.sin(2 * np.pi * df["MESE"] / 12)
    df["MESE_COS"] = np.cos(2 * np.pi * df["MESE"] / 12)

    df["TREND"] = (df["ANNO"] - df["ANNO"].min()) * 12 + df["MESE"]

    df["REGIONE"] = df["REGIONE"].astype("category")

    return df

def train_and_evaluate(df):
    df = build_features(df)

    feature_cols = ["ANNO", "MESE_SIN", "MESE_COS", "TREND", "REGIONE"]
    target_col = "IMPORTO"

    X = df[feature_cols]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        enable_categorical=True,
        random_state=42
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print(f"MAE:  {mean_absolute_error(y_test, y_pred):,.2f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):,.2f}")
    print(f"R2:   {r2_score(y_test, y_pred):.3f}")

    return model, X_test, y_test, y_pred

def predict_future(model, regioni, anno=2027, mesi=range(1, 7)):
    rows = [{"REGIONE": r, "ANNO": anno, "MESE": m} for r in regioni for m in mesi]
    future_df = pd.DataFrame(rows)
    future_df = build_features(future_df)

    X_future = future_df[["ANNO", "MESE_SIN", "MESE_COS", "TREND", "REGIONE"]]
    future_df["IMPORTO_PREVISTO"] = model.predict(X_future)

    return future_df

def print_accuracy_report(X_test, y_test, y_pred):
    results = X_test.copy()
    results["IMPORTO_REALE"] = y_test.values
    results["IMPORTO_PREVISTO"] = y_pred
    results["ERRORE_ASSOLUTO"] = (results["IMPORTO_REALE"] - results["IMPORTO_PREVISTO"]).abs()
    results["ERRORE_PERCENTUALE"] = (
        results["ERRORE_ASSOLUTO"] / results["IMPORTO_REALE"].replace(0, np.nan)
    ) * 100

    print("\n" + "=" * 50)
    print("REPORT ACCURATEZZA MODELLO")
    print("=" * 50)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mape = results["ERRORE_PERCENTUALE"].mean()

    print(f"MAE:  {mae:,.2f}")
    print(f"RMSE: {rmse:,.2f}")
    print(f"R2:   {r2:.3f}")
    print(f"MAPE: {mape:.1f}%")

    print("\n--- Errore medio assoluto per regione (peggiori 5) ---")
    errore_per_regione = (
        results.groupby("REGIONE", observed=True)["ERRORE_ASSOLUTO"]
        .mean()
        .sort_values(ascending=False)
    )
    print(errore_per_regione.head())

    print("\n--- Esempi di previsione (prime 10 righe) ---")
    print(
        results[["REGIONE", "ANNO", "IMPORTO_REALE", "IMPORTO_PREVISTO", "ERRORE_ASSOLUTO"]]
        .head(10)
        .to_string(index=False)
    )
    print("=" * 50 + "\n")

    return results

def show_average_by_region_year(df):
    avg_df = (
        df.groupby(["REGIONE", "ANNO"], observed=True)["IMPORTO"]
        .mean()
        .reset_index()
        .rename(columns={"IMPORTO": "IMPORTO_MEDIO"})
    )

    print("\n" + "=" * 50)
    print("MEDIA IMPORTI PER REGIONE E ANNO")
    print("=" * 50)
    print(avg_df.to_string(index=False))
    print("=" * 50 + "\n")

    return avg_df

def predict_2027_by_region_linear(df):
    """
    Per ogni regione, allena una regressione lineare IMPORTO ~ ANNO
    (sui dati aggregati annualmente) ed estrapola la previsione per il 2027.
    """
    # Aggrega per regione/anno (media annuale, coerente con l'approccio precedente)
    yearly_df = (
        df.groupby(["REGIONE", "ANNO"], observed=True)["IMPORTO"]
        .mean()
        .reset_index()
    )

    risultati = []

    for regione, gruppo in yearly_df.groupby("REGIONE", observed=True):
        X = gruppo[["ANNO"]].values
        y = gruppo["IMPORTO"].values

        if len(gruppo) < 2:
            # Servono almeno 2 punti per stimare una retta
            print(f"Attenzione: regione '{regione}' ha meno di 2 anni di dati, saltata.")
            continue

        model = LinearRegression()
        model.fit(X, y)

        previsione_2027 = model.predict([[2027]])[0]

        risultati.append({
            "REGIONE": regione,
            "IMPORTO_PREVISTO_2027": previsione_2027,
            "COEFFICIENTE_ANNUO": model.coef_[0],
            "INTERCETTA": model.intercept_,
            "R2_TRAIN": model.score(X, y)
        })

    risultati_df = pd.DataFrame(risultati).sort_values("IMPORTO_PREVISTO_2027", ascending=False)

    print("\n" + "=" * 50)
    print("PREVISIONE IMPORTI 2027 PER REGIONE (Regressione Lineare)")
    print("=" * 50)
    print(risultati_df.to_string(index=False))
    print("=" * 50 + "\n")

    return risultati_df

def validate_linear_regression_on_2026(df):
    """
    Per ogni regione, allena una regressione lineare IMPORTO ~ ANNO
    usando solo il 2019-2025, poi confronta la previsione per il 2026
    con il valore reale osservato in quell'anno.
    """
    # Aggrega per regione/anno (media annuale)
    yearly_df = (
        df.groupby(["REGIONE", "ANNO"], observed=True)["IMPORTO"]
        .mean()
        .reset_index()
    )

    risultati = []

    for regione, gruppo in yearly_df.groupby("REGIONE", observed=True):
        train = gruppo[gruppo["ANNO"] <= 2025]
        test = gruppo[gruppo["ANNO"] == 2026]

        if len(train) < 2:
            print(f"Attenzione: regione '{regione}' ha meno di 2 anni di dati fino al 2025, saltata.")
            continue

        if test.empty:
            print(f"Attenzione: regione '{regione}' non ha dati reali per il 2026, saltata.")
            continue

        X_train = train[["ANNO"]].values
        y_train = train["IMPORTO"].values

        model = LinearRegression()
        model.fit(X_train, y_train)

        previsione_2026 = model.predict([[2026]])[0]
        reale_2026 = test["IMPORTO"].values[0]

        errore_assoluto = abs(reale_2026 - previsione_2026)
        errore_percentuale = (errore_assoluto / reale_2026) * 100 if reale_2026 != 0 else np.nan

        risultati.append({
            "REGIONE": regione,
            "IMPORTO_REALE_2026": reale_2026,
            "IMPORTO_PREVISTO_2026": previsione_2026,
            "ERRORE_ASSOLUTO": errore_assoluto,
            "ERRORE_PERCENTUALE": errore_percentuale,
            "R2_TRAIN": model.score(X_train, y_train)
        })

    risultati_df = pd.DataFrame(risultati).sort_values("ERRORE_PERCENTUALE", ascending=False)

    print("\n" + "=" * 60)
    print("VALIDAZIONE REGRESSIONE LINEARE: PREVISTO vs REALE 2026")
    print("=" * 60)
    print(risultati_df.to_string(index=False))

    print(f"\nErrore percentuale medio (MAPE): {risultati_df['ERRORE_PERCENTUALE'].mean():.1f}%")
    print(f"Errore assoluto medio (MAE): {risultati_df['ERRORE_ASSOLUTO'].mean():,.2f}")
    print("=" * 60 + "\n")

    return risultati_df


def validate_linear_regression_on_2026_by_school(df, output_csv="data/scuole.csv"):
    """
    Per ogni CODICESCUOLA, allena una regressione lineare IMPORTO ~ ANNO
    usando solo il 2019-2025, poi confronta la previsione per il 2026
    con il valore reale osservato in quell'anno.
    Restituisce un DataFrame con una riga per scuola.
    """
    yearly_df = (
        df.groupby(["CODICESCUOLA", "REGIONE", "ANNO"], observed=True)["IMPORTO"]
        .sum()
        .reset_index()
    )

    risultati = []

    for (scuola, regione), gruppo in yearly_df.groupby(["CODICESCUOLA", "REGIONE"], observed=True):
        train = gruppo[gruppo["ANNO"] <= 2025]
        test = gruppo[gruppo["ANNO"] == 2026]

        if len(train) < 2:
            # Servono almeno 2 punti per stimare una retta
            continue

        if test.empty:
            # Nessun dato reale 2026 per questa scuola: non possiamo validare
            continue

        X_train = train[["ANNO"]].values
        y_train = train["IMPORTO"].values

        model = LinearRegression()
        model.fit(X_train, y_train)

        previsione_2026 = model.predict([[2026]])[0]
        reale_2026 = test["IMPORTO"].values[0]

        errore_assoluto = abs(reale_2026 - previsione_2026)
        errore_percentuale = (errore_assoluto / reale_2026) * 100 if reale_2026 != 0 else np.nan

        risultati.append({
            "CODICESCUOLA": scuola,
            "REGIONE": regione,
            "IMPORTO_REALE_2026": reale_2026,
            "IMPORTO_PREVISTO_2026": previsione_2026,
            "ERRORE_ASSOLUTO": errore_assoluto,
            "ERRORE_PERCENTUALE": errore_percentuale,
            "R2_TRAIN": model.score(X_train, y_train),
            "N_ANNI_TRAIN": len(train)
        })

    risultati_df = pd.DataFrame(risultati).sort_values("ERRORE_PERCENTUALE", ascending=False)

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    risultati_df.to_csv(output_csv, index=False)

    print("\n" + "=" * 60)
    print("VALIDAZIONE REGRESSIONE LINEARE PER SCUOLA: PREVISTO vs REALE 2026")
    print("=" * 60)
    print(f"Totale scuole valutate: {len(risultati_df)}")
    print(risultati_df.to_string(index=False))  # <-- stampa tutte le righe per scuola

    print(f"Errore percentuale medio (MAPE): {risultati_df['ERRORE_PERCENTUALE'].mean():.1f}%")
    print(f"Errore assoluto medio (MAE):     {risultati_df['ERRORE_ASSOLUTO'].mean():,.2f}")
    print("=" * 60 + "\n")

    return risultati_df


def aggregate_school_predictions_by_region(school_results_df):
    """
    Aggrega per REGIONE i risultati calcolati a livello di singola scuola
    (validate_linear_regression_on_2026_by_school), sommando gli importi reali
    e previsti e ricalcolando l'errore assoluto e percentuale sui totali di regione.
    """
    agg_df = (
        school_results_df
        .groupby("REGIONE", observed=True)
        .agg(
            N_SCUOLE=("CODICESCUOLA", "count"),
            IMPORTO_REALE_2026=("IMPORTO_REALE_2026", "sum"),
            IMPORTO_PREVISTO_2026=("IMPORTO_PREVISTO_2026", "sum"),
            MAE_SCUOLE=("ERRORE_ASSOLUTO", "mean"),
            MAPE_SCUOLE=("ERRORE_PERCENTUALE", "mean"),
        )
        .reset_index()
    )

    agg_df["ERRORE_ASSOLUTO"] = (
        agg_df["IMPORTO_REALE_2026"] - agg_df["IMPORTO_PREVISTO_2026"]
    ).abs()
    agg_df["ERRORE_PERCENTUALE"] = (
        agg_df["ERRORE_ASSOLUTO"] / agg_df["IMPORTO_REALE_2026"].replace(0, np.nan)
    ) * 100

    agg_df = agg_df.sort_values("ERRORE_PERCENTUALE", ascending=False)

    print("\n" + "=" * 60)
    print("AGGREGAZIONE PER REGIONE (bottom-up, previsione per scuola)")
    print("=" * 60)
    print(agg_df.to_string(index=False))
    print("=" * 60 + "\n")

    return agg_df


def main():
    detailed_df = read_files()
    print(detailed_df.head())
    print(f"Totale righe valide: {len(detailed_df)}")

    # Diagnostica: guarda i valori originali della data per le righe con ANNO anomalo
    df_check = detailed_df.copy()
    df_check["DATAAPPROVAZIONECONSIGLIO_PARSED"] = pd.to_datetime(
        df_check["DATAAPPROVAZIONECONSIGLIO"], errors="coerce"
    )
    df_check["ANNO_TEST"] = df_check["DATAAPPROVAZIONECONSIGLIO_PARSED"].dt.year

    # Mostra le righe con anni sospetti (es. fuori da un range ragionevole)
    anomale = df_check[
        (df_check["ANNO_TEST"] < 2000) | (df_check["ANNO_TEST"] > 2030)
    ]
    print(anomale[["DATAAPPROVAZIONECONSIGLIO", "DATAAPPROVAZIONECONSIGLIO_PARSED"]].drop_duplicates())
    print(f"Numero di righe anomale: {len(anomale)}")



    grouped_df = group_rows_by_month_year(detailed_df)
    print(grouped_df.head())
    print(f"Totale righe valide: {len(grouped_df)}")

    avg_df = show_average_by_region_year(detailed_df)
    lr1 = validate_linear_regression_on_2026(detailed_df)
    lr2 = predict_2027_by_region_linear(detailed_df)

    build_features(grouped_df)
    model, X_test, y_test, y_pred = train_and_evaluate(grouped_df)
    print_accuracy_report(X_test, y_test, y_pred)


    school_results = validate_linear_regression_on_2026_by_school(detailed_df)
    region_from_schools = aggregate_school_predictions_by_region(school_results)

if __name__ == "__main__":
    main()
