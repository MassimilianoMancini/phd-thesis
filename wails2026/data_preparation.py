import pandas as pd
import numpy as np

def _group_by_year(df):
    df = df.copy()

    year_columns = [str(year) for year in [2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026]]

    grouped = (
        df.groupby(["CODICESCUOLA", "ANNOFINANZIARIO"], as_index=False)["IMPORTO"]
        .sum()
        .pivot(index="CODICESCUOLA", columns="ANNOFINANZIARIO", values="IMPORTO")
    )

    grouped.columns = grouped.columns.astype(str)

    return grouped.reindex(columns=year_columns, fill_value=0).reset_index().fillna(0)


def _create_avg_column(df):
    year_columns = [str(year) for year in [2019, 2020, 2021, 2022, 2023, 2024, 2025]]
    df["AVG"] = df[year_columns].mean(axis=1)
    return df


def _create_target_columns(df, comp = "2025"):
    comp = str(comp)
    df["DIFFERENCE"] = df["2026"] - df[comp]
    df["Y"] = df["DIFFERENCE"].apply(lambda x: 1 if x >= 0 else 0)
    df["INCREMENT"] = np.where(
        (df[comp] * df["2026"]) != 0,
        df["DIFFERENCE"] / (df[comp] + df["2026"]),
        0
    )
    
    R = np.zeros(len(df), dtype=int)
    
    neg_mask = df["INCREMENT"] < 0
    zero_mask = df["INCREMENT"] == 0
    pos_mask = df["INCREMENT"] > 0
    
    if neg_mask.any():
        neg_values = df.loc[neg_mask, "INCREMENT"]
        neg_median = neg_values.median()
        R[neg_mask & (df["INCREMENT"] < neg_median)] = 0
        R[neg_mask & (df["INCREMENT"] >= neg_median)] = 1
    
    R[zero_mask] = 2
    
    if pos_mask.any():
        pos_values = df.loc[pos_mask, "INCREMENT"]
        pos_median = pos_values.median()
        R[pos_mask & (df["INCREMENT"] <= pos_median)] = 3
        R[pos_mask & (df["INCREMENT"] > pos_median)] = 4
    
    df["R"] = R
    
    cut = pd.Categorical(R, categories=[0, 1, 2, 3, 4], ordered=True)
    
    return cut.categories

def _merge_registry_and_grouped(grouped_df, registry_df):
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
    merged_df = grouped_df.merge(
        registry_df[registry_columns],
        on="CODICESCUOLA",
        how="left",
        validate="m:1",
    )
    cols = [
        "AREAGEOGRAFICA",
        "REGIONE",
        "PROVINCIA",
        "CAPSCUOLA",
        "CODICECOMUNESCUOLA",
        "DESCRIZIONECARATTERISTICASCUOLA",
        "DESCRIZIONETIPOLOGIAGRADOISTRUZIONESCUOLA",
    ]
    ordered_cols = cols + [c for c in merged_df.columns if c not in cols]
    return merged_df[ordered_cols]


def _create_raggruppamento_column(df):
    mapping = {
        "ISTITUTO COPMPRENSIVO": "ISTITUTO COMPRENSIVO",
        "ISTITUTO COMPRENSIVO": "ISTITUTO COMPRENSIVO",
        "ISTITUTO SUPERIORE": "ISTITUTO SUPERIORE",
        "SCUOLA PRIMARIA": "SCUOLA PRIMARIA",
        "LICEO SCIENTIFICO": "LICEO SCIENTIFICO",
        "SCUOLA PRIMO GRADO": "SCUOLA PRIMO GRADO",
        "LICEO CLASSICO": "LICEO CLASSICO",
    }
    df["RAGGRUPPAMENTO"] = (
        df["DESCRIZIONETIPOLOGIAGRADOISTRUZIONESCUOLA"]
        .map(mapping)
        .fillna("ALTRO")
    )
    return df


def _define_string_columns_as_category(df):
    string_columns = df.select_dtypes(include=["str"]).columns
    for col in string_columns:
        df[col] = df[col].astype("category")
    return df

def prepare_and_merge_data(tdf, sdf):
    grouped_df = _group_by_year(tdf)
    grouped_df = _create_avg_column(grouped_df)
    cut_categories = _create_target_columns(grouped_df, comp="AVG")
    final_df = _merge_registry_and_grouped(grouped_df, sdf)
    final_df = _create_raggruppamento_column(final_df)
    final_df = _define_string_columns_as_category(final_df)
    return final_df, cut_categories
