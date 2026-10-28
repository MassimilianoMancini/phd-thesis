from data_import import import_and_clean_data
from data_preparation import prepare_and_merge_data
from fairness_measures import cosine_distance, ep_dm, pairwise_difference_matrix, statistical_parity, total_accuracy
from predict_data import predict_with_random_forest, predict_with_xgboost
from data_controls import controls


def main():
    tdf, sdf = import_and_clean_data()
    df, _ = prepare_and_merge_data(tdf, sdf)
    df_xgb = predict_with_xgboost(df)
    print("\nControllo delle predizioni del modello XGBoost:")
    controls(df_xgb, pred_column="XGB")

    df_rf = predict_with_random_forest(df)
    print("\nControllo delle predizioni del modello Random Forest:")
    controls(df_rf, pred_column="RF")

    print("REGIONE")

    print("Statistical Parity vera:")
    sp_y = statistical_parity(df, 'REGIONE', 'Y')
    print(sp_y)
    ep_y, dm_y = ep_dm(pairwise_difference_matrix(sp_y))
    print(f"EP: {ep_y}, DM: {dm_y}")

    print("Statistical Parity per XGBoost:")
    sp_xgb = statistical_parity(df_xgb, 'REGIONE', 'XGB')
    print(sp_xgb)
    print("EP e DM per XGBoost:")
    ep_xgb, dm_xgb = ep_dm(pairwise_difference_matrix(sp_xgb))
    print(f"EP: {ep_xgb}, DM: {dm_xgb}")

    print("Statistical Parity per Random Forest:")
    sp_rf = statistical_parity(df_rf, 'REGIONE', 'RF')
    print(sp_rf)
    print("EP e DM per Random Forest:")
    ep_rf, dm_rf = ep_dm(pairwise_difference_matrix(sp_rf))
    print(f"EP: {ep_rf}, DM: {dm_rf}")

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

if __name__ == "__main__":
    main()
