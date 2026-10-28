import pandas as pd

from data_import import import_and_clean_data
from data_preparation import prepare_and_merge_data
from fairness_measures import cosine_distance, cosine_distance_equity, ep_dm, group_cardinality, pairwise_difference_matrix, statistical_parity, total_accuracy, total_positives_ref, total_positives_sp, total_positives_ta
from predict_data import predict_with_random_forest, predict_with_xgboost
from data_controls import controls


def main():

    EQUITY_REFERENCE = pd.Series({
        "ABRUZZO": 0.96,
        "BASILICATA": 1.00,
        "CALABRIA": 0.97,
        "CAMPANIA": 0.98,
        "EMILIA ROMAGNA": 0.72,
        "FRIULI-VENEZIA G.": 0.73,
        "LAZIO": 0.84,
        "LIGURIA": 0.74,
        "LOMBARDIA": 0.70,
        "MARCHE": 0.86,
        "MOLISE": 0.99,
        "PIEMONTE": 0.73,
        "PUGLIA": 0.98,
        "SARDEGNA": 0.95,
        "SICILIA": 0.99,
        "TOSCANA": 0.85,
        "UMBRIA": 0.83,
        "VENETO": 0.74,
    }, name="eq_i")
    GROUP_EQUITY_REFERENCE = pd.Series(1.0, index=[
        "ALTRO", "ISTITUTO COMPRENSIVO", "ISTITUTO SUPERIORE",
        "LICEO CLASSICO", "LICEO SCIENTIFICO", "SCUOLA PRIMO GRADO",
        "SCUOLA PRIMARIA",
    ], name="eq_i")

    tdf, sdf = import_and_clean_data()
    df, _ = prepare_and_merge_data(tdf, sdf)
    df_xgb = predict_with_xgboost(df)
    print("\nControllo delle predizioni del modello XGBoost:")
    controls(df_xgb, pred_column="XGB")

    df_rf = predict_with_random_forest(df)
    print("\nControllo delle predizioni del modello Random Forest:")
    controls(df_rf, pred_column="RF")

    def print_region_totals(data, classifier):
        cardinalities = group_cardinality(data, "REGIONE").rename("numero di record")
        sp = total_positives_sp(data, "REGIONE", classifier).rename("SP")
        ta = total_positives_ta(data, "REGIONE", classifier, "Y").rename("TA")
        ref = cardinalities.mul(EQUITY_REFERENCE.reindex(cardinalities.index)).rename("REF")

        table = pd.concat([cardinalities, sp, ta, ref], axis=1)
        for column in ["numero di record", "SP", "TA"]:
            table[column] = table[column].astype(int)
        totals = table.sum().to_frame().T
        for column in ["numero di record", "SP", "TA"]:
            totals[column] = totals[column].astype(int)
        totals.insert(0, "REGIONE", "TOTALE")
        table = pd.concat([table.reset_index(), totals], ignore_index=True)

        print(f"\nTotali per regione ({classifier}):")
        print(table.to_string(index=False))

    print("REGIONE")
    print_region_totals(df, "Y")
    print_region_totals(df_xgb, "XGB")
    print_region_totals(df_rf, "RF")

    print("Statistical Parity vera:")
    sp_y = statistical_parity(df, 'REGIONE', 'Y')
    print(sp_y)
    ep_y, dm_y = ep_dm(pairwise_difference_matrix(sp_y))
    print(f"EP: {ep_y}, DM: {dm_y}")
    print(f"Mtsp: {total_positives_sp(df, 'REGIONE', 'Y')}")

    print("Statistical Parity per XGBoost:")
    sp_xgb = statistical_parity(df_xgb, 'REGIONE', 'XGB')
    print(sp_xgb)
    print("EP e DM per XGBoost:")
    ep_xgb, dm_xgb = ep_dm(pairwise_difference_matrix(sp_xgb))
    print(f"EP: {ep_xgb}, DM: {dm_xgb}")
    print(f"Mxsp: {total_positives_sp(df_xgb, 'REGIONE', 'XGB')}")

    print("Statistical Parity per Random Forest:")
    sp_rf = statistical_parity(df_rf, 'REGIONE', 'RF')
    print(sp_rf)
    print("EP e DM per Random Forest:")
    ep_rf, dm_rf = ep_dm(pairwise_difference_matrix(sp_rf))
    print(f"EP: {ep_rf}, DM: {dm_rf}")
    print(f"Mrsp: {total_positives_sp(df_rf, 'REGIONE', 'RF')}")

    print(f"Mref: {total_positives_ref(df, 'REGIONE', EQUITY_REFERENCE)}")

    print("Total Accuracy per XGBoost:")
    ta_xgb = total_accuracy(df_xgb, 'REGIONE', 'XGB', 'Y')
    print(ta_xgb)

    print("Total Accuracy per Random Forest:")
    ta_rf = total_accuracy(df_rf, 'REGIONE', 'RF', 'Y')
    print(ta_rf)

    print("Matrice delle differenze pairwise per XGBoost:")
    pdm_xgb = pairwise_difference_matrix(sp_xgb)
    print(pdm_xgb)

    print("EP e DM per XGBoost:")
    ep_xgb, dm_xgb = ep_dm(pdm_xgb)
    print(f"EP: {ep_xgb}, DM: {dm_xgb}")

    print("Distanza del coseno per Y Vera:")
    cosine_distance_y = cosine_distance(sp_y)
    print(cosine_distance_y)

    print("Distanza del coseno per SP XGBoost:")
    cosine_distance_xgb = cosine_distance(sp_xgb)
    print(cosine_distance_xgb)

    print("Distanza del coseno per SP Random Forest:")
    cosine_distance_rf = cosine_distance(sp_rf)
    print(cosine_distance_rf)

    print("Distanza del coseno per TA XGBoost:")
    cosine_distance_xgb = cosine_distance(ta_xgb)
    print(cosine_distance_xgb)

    print("Distanza del coseno per TA Random Forest:")
    cosine_distance_rf = cosine_distance(ta_rf)
    print(cosine_distance_rf)

    print("Distanza del coseno per Y Vera equity:")
    cosine_distance_y = cosine_distance_equity(sp_y, EQUITY_REFERENCE)
    print(cosine_distance_y)

    print("Distanza del coseno per SP XGBoost equity:")
    cosine_distance_xgb = cosine_distance_equity(sp_xgb, EQUITY_REFERENCE)
    print(cosine_distance_xgb)

    print("Distanza del coseno per SP Random Forest equity:")
    cosine_distance_rf = cosine_distance_equity(sp_rf, EQUITY_REFERENCE)
    print(cosine_distance_rf)

    print("Distanza del coseno per TA XGBoost equity:")
    cosine_distance_xgb = cosine_distance_equity(ta_xgb, EQUITY_REFERENCE)
    print(cosine_distance_xgb)

    print("Distanza del coseno per TA Random Forest equity:")
    cosine_distance_rf = cosine_distance_equity(ta_rf, EQUITY_REFERENCE)
    print(cosine_distance_rf)


    print("RAGGRUPPAMENTO")

    print("Statistical Parity vera:")
    sp_y = statistical_parity(df, 'RAGGRUPPAMENTO', 'Y')
    print(sp_y)

    print("Statistical Parity per XGBoost:")
    sp_xgb = statistical_parity(df_xgb, 'RAGGRUPPAMENTO', 'XGB')
    print(sp_xgb)

    print("Statistical Parity per Random Forest:")
    sp_rf = statistical_parity(df_rf, 'RAGGRUPPAMENTO', 'RF')
    print(sp_rf)

    print("Total Accuracy per XGBoost:")
    ta_xgb = total_accuracy(df_xgb, 'RAGGRUPPAMENTO', 'XGB', 'Y')
    print(ta_xgb)

    print("Total Accuracy per Random Forest:")
    ta_rf = total_accuracy(df_rf, 'RAGGRUPPAMENTO', 'RF', 'Y')
    print(ta_rf)

    print("Matrice delle differenze pairwise per XGBoost:")
    pdm_xgb = pairwise_difference_matrix(sp_xgb)
    print(pdm_xgb)

    print("EP e DM per XGBoost:")
    ep_xgb, dm_xgb = ep_dm(pdm_xgb)
    print(f"EP: {ep_xgb}, DM: {dm_xgb}")

    print("Distanza del coseno per Y Vera:")
    cosine_distance_y = cosine_distance(sp_y)
    print(cosine_distance_y)

    print("Distanza del coseno per SP XGBoost:")
    cosine_distance_xgb = cosine_distance(sp_xgb)
    print(cosine_distance_xgb)

    print("Distanza del coseno per SP Random Forest:")
    cosine_distance_rf = cosine_distance(sp_rf)
    print(cosine_distance_rf)

    print("Distanza del coseno per TA XGBoost:")
    cosine_distance_xgb = cosine_distance(ta_xgb)
    print(cosine_distance_xgb)

    print("Distanza del coseno per TA Random Forest:")
    cosine_distance_rf = cosine_distance(ta_rf)
    print(cosine_distance_rf)

    print("Distanza del coseno per Y Vera equity:")
    cosine_distance_y = cosine_distance_equity(sp_y, GROUP_EQUITY_REFERENCE)
    print(cosine_distance_y)

    print("Distanza del coseno per SP XGBoost equity:")
    cosine_distance_xgb = cosine_distance_equity(sp_xgb, GROUP_EQUITY_REFERENCE)
    print(cosine_distance_xgb)

    print("Distanza del coseno per SP Random Forest equity:")
    cosine_distance_rf = cosine_distance_equity(sp_rf, GROUP_EQUITY_REFERENCE)
    print(cosine_distance_rf)

    print("Distanza del coseno per TA XGBoost equity:")
    cosine_distance_xgb = cosine_distance_equity(ta_xgb, GROUP_EQUITY_REFERENCE)
    print(cosine_distance_xgb)

    print("Distanza del coseno per TA Random Forest equity:")
    cosine_distance_rf = cosine_distance_equity(ta_rf, GROUP_EQUITY_REFERENCE)
    print(cosine_distance_rf)

if __name__ == "__main__":
    main()
