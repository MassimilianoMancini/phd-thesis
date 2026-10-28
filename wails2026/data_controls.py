from sklearn.metrics import accuracy_score, recall_score, precision_score, fbeta_score

def controls(df, pred_column):
    # Stampa numero di righe dove Y = y_pred suddivise per split
    correct_train = ((df["Y"] == df[pred_column]) & (df["SPLIT"] == "train")).sum()
    correct_test = ((df["Y"] == df[pred_column]) & (df["SPLIT"] == "test")).sum()
    total_rows = len(df)
    train_rows = (df["SPLIT"] == "train").sum()
    test_rows = (df["SPLIT"] == "test").sum()
    
    print(f"\nRighe totali: {total_rows}")
    print(f"  Train: {train_rows}")
    print(f"  Test: {test_rows}")
    print(f"\nRighe con Y = {pred_column}:")
    print(f"  Train: {correct_train}")
    print(f"  Test: {correct_test}")
    
    # Calcola metriche per train e test
    train_mask = df["SPLIT"] == "train"
    test_mask = df["SPLIT"] == "test"
    
    # Train metrics
    train_acc = accuracy_score(df.loc[train_mask, "Y"], df.loc[train_mask, pred_column])
    train_recall = recall_score(df.loc[train_mask, "Y"], df.loc[train_mask, pred_column], zero_division=0)
    train_precision = precision_score(df.loc[train_mask, "Y"], df.loc[train_mask, pred_column], zero_division=0)
    train_f2 = fbeta_score(df.loc[train_mask, "Y"], df.loc[train_mask, pred_column], beta=2, zero_division=0)
    
    # Test metrics
    test_acc = accuracy_score(df.loc[test_mask, "Y"], df.loc[test_mask, pred_column])
    test_recall = recall_score(df.loc[test_mask, "Y"], df.loc[test_mask, pred_column], zero_division=0)
    test_precision = precision_score(df.loc[test_mask, "Y"], df.loc[test_mask, pred_column], zero_division=0)
    test_f2 = fbeta_score(df.loc[test_mask, "Y"], df.loc[test_mask, pred_column], beta=2, zero_division=0)
    
    print(f"\nMetriche TRAIN:")
    print(f"  Accuracy:  {train_acc:.4f}")
    print(f"  Recall:    {train_recall:.4f}")
    print(f"  Precision: {train_precision:.4f}")
    print(f"  F2:        {train_f2:.4f}")
    
    print(f"\nMetriche TEST:")
    print(f"  Accuracy:  {test_acc:.4f}")
    print(f"  Recall:    {test_recall:.4f}")
    print(f"  Precision: {test_precision:.4f}")
    print(f"  F2:        {test_f2:.4f}")
    