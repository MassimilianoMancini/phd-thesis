"""
Simple script to check whether LSAT value 29 acts as a threshold
for bar passage using the law school dataset.

The script downloads the same dataset used by the other modules,
loads it with pandas, and computes pass rates for scores below 29,
equal to 29, and above 29.
"""

import pandas as pd
import kagglehub


def main():
    # download dataset (cached by kagglehub)
    path = kagglehub.dataset_download("danofer/law-school-admissions-bar-passage")
    file = path + "/bar_pass_prediction.csv"

    df = pd.read_csv(file)

    # keep only records with a numeric lsat and a bar outcome
    df = df.dropna(subset=["lsat", "bar1"])
    # create a boolean passed indicator using the bar1 column ('P'=pass)
    df["passed"] = df["bar1"].astype(str).str.upper() == "P"

    # overall counts around the 29 threshold
    below = df[df["lsat"] < 29]
    at29 = df[df["lsat"] == 29]
    above = df[df["lsat"] > 29]

    def report(group, label):
        n = len(group)
        p = group["passed"].mean() if n > 0 else float("nan")
        print(f"{label}: samples={n}, pass_rate={p:.3f}")

    print("LSAT threshold analysis")
    print("========================")
    report(below, "LSAT < 29")
    report(at29, "LSAT == 29")
    report(above, "LSAT > 29")

    # additional detail: pass rates for each LSAT score near 29
    neighbourhood = df[df["lsat"].between(9, 50)]
    print("\nDetailed pass rate by score (9..50):")
    print(neighbourhood.groupby("lsat")["passed"].mean())


if __name__ == "__main__":
    main()
