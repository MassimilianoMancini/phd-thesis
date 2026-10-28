import pandas as pd

def _coerce_numeric_columns(df, columns):
    for column in columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df

def _import_files(file_names):
    frames = []
    budget_columns = [
        "ANNOSCOLASTICO",
        "ANNOFINANZIARIO",
        "REGIONE",
        "CODICESCUOLA",
        "TIPOLOGIAVOCE",
        "IMPORTO",
    ]
    for file_name in file_names:
        frame = pd.read_csv(file_name, usecols=budget_columns)
        frames.append(_coerce_numeric_columns(frame, ["ANNOFINANZIARIO", "IMPORTO"]))
    return pd.concat(frames, ignore_index=True)


def _clean_data(df):
    df = df[(df["IMPORTO"] != 0) & (df["IMPORTO"].notna())].reset_index(drop=True)
    df = df[df["TIPOLOGIAVOCE"] == "ENTRATA"].reset_index(drop=True)
    df = df[(df["ANNOFINANZIARIO"] >= 2019) & (df["ANNOFINANZIARIO"] <= 2026)].reset_index(drop=True)
    return df


def _remove_columns(df):
    return df.drop(columns=["ANNOSCOLASTICO", "TIPOLOGIAVOCE"]).reset_index(drop=True)

def _import_school_registry(file_paths):
    registry_df = None
    registry_columns = [
        "AREAGEOGRAFICA",
        "REGIONE",
        "PROVINCIA",
        "CAPSCUOLA",
        "CODICECOMUNESCUOLA",
        "DESCRIZIONECARATTERISTICASCUOLA",
        "DESCRIZIONETIPOLOGIAGRADOISTRUZIONESCUOLA",
        "CODICESCUOLA",
    ]
    for file_path in file_paths:
        current_df = pd.read_csv(file_path, usecols=registry_columns)
        current_df = current_df.astype({"CODICESCUOLA": str})
        if registry_df is None:
            registry_df = current_df
        else:
            unique_rows = current_df[~current_df["CODICESCUOLA"].isin(registry_df["CODICESCUOLA"])]
            registry_df = pd.concat([registry_df, unique_rows], ignore_index=True)

    return registry_df

def import_and_clean_data():
    transactions_file_names = [
        "data/BISPROGANNO201920240925.csv",
        "data/BISPROGANNO202020240925.csv",
        "data/BISPROGANNO202120240925.csv",
        "data/BISPROGANNO202220240925.csv",
        "data/BISPROGANNO202320241220.csv",
        "data/BISPROGANNO202420251220.csv",
        "data/BISPROGANNO202520260320.csv",
        "data/BISPROGANNO202620260421.csv",
    ]
    schools_file_names = [
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

    tdf = _import_files(transactions_file_names)
    tdf = _clean_data(tdf)
    tdf = _remove_columns(tdf)
    sdf = _import_school_registry(schools_file_names)
    return tdf, sdf