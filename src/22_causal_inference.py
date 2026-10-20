"""Causal inference using statsmodels logistic regression with clustered SEs."""

import json
import time
from pathlib import Path

import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_V2_PATH = PROJECT_ROOT / "data" / "processed" / "data_v2.npz"
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "causal_analysis.json"


def main():
    start_time = time.time()

    # Load raw data
    data = np.load(DATA_V2_PATH, allow_pickle=True)
    X_train = data["X_train"]
    X_test = data["X_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]
    patient_ids_train = data["patient_ids_train"]
    patient_ids_test = data["patient_ids_test"]
    feature_names = data["feature_names"].astype(str).tolist()
    feature_means = data["feature_means"].astype(np.float32)
    feature_stds = data["feature_stds"].astype(np.float32)

    # Combine train and test
    X_all = np.concatenate([X_train, X_test], axis=0)
    y_all = np.concatenate([y_train, y_test], axis=0)
    patient_ids_all = np.concatenate([patient_ids_train, patient_ids_test], axis=0)

    # Get last hour values
    last_hour = X_all[:, -1, :]

    # Denormalize
    last_hour_denorm = last_hour * feature_stds + feature_means

    # Build dataframe
    n_samples = len(y_all)
    df_data = {
        "sepsis": y_all,
        "patient_id": patient_ids_all,
    }

    # Add variables of interest
    variables = ["Lactate", "HR", "MAP", "Resp", "Temp"]
    for var in variables:
        idx = feature_names.index(var)
        values = last_hour_denorm[:, idx]
        # Standardize per 1 SD
        std_val = np.std(values)
        df_data[var] = (values - np.mean(values)) / std_val

    # Add covariates
    idx_iculos = feature_names.index("ICULOS")
    idx_age = feature_names.index("Age")
    idx_gender = feature_names.index("Gender")

    df_data["ICULOS"] = last_hour_denorm[:, idx_iculos]
    df_data["Age"] = last_hour_denorm[:, idx_age]
    df_data["Gender"] = last_hour_denorm[:, idx_gender]

    # Create pandas DataFrame
    import pandas as pd
    df = pd.DataFrame(df_data)

    print(f"Total samples: {len(df)}")
    print(f"Positive cases: {df['sepsis'].sum()} ({df['sepsis'].mean():.2%})")

    # Run analysis for each variable
    results = {}

    for var in variables:
        print(f"\n=== {var} ===")

        # Naive model
        formula_naive = f"sepsis ~ {var}"
        model_naive = smf.logit(formula_naive, data=df).fit(disp=0)
        naive_coef = model_naive.params[var]
        naive_se = model_naive.bse[var]
        naive_ci_low = model_naive.conf_int().loc[var, 0]
        naive_ci_high = model_naive.conf_int().loc[var, 1]

        # Adjusted model with clustered standard errors
        formula_adj = f"sepsis ~ {var} + ICULOS + Age + Gender"
        model_adj = smf.logit(formula_adj, data=df).fit(cov_type="cluster", cov_kwds={"groups": df["patient_id"]}, disp=0)

        adj_coef = model_adj.params[var]
        adj_se = model_adj.bse[var]
        adj_ci_low = model_adj.conf_int().loc[var, 0]
        adj_ci_high = model_adj.conf_int().loc[var, 1]

        # Odds ratios
        naive_or = np.exp(naive_coef)
        naive_or_ci = (np.exp(naive_ci_low), np.exp(naive_ci_high))
        adj_or = np.exp(adj_coef)
        adj_or_ci = (np.exp(adj_ci_low), np.exp(adj_ci_high))

        # Percent change in log-odds
        pct_change = ((adj_coef - naive_coef) / abs(naive_coef)) * 100 if naive_coef != 0 else 0

        print(f"Naive OR: {naive_or:.3f} [{naive_or_ci[0]:.3f}, {naive_or_ci[1]:.3f}]")
        print(f"Adjusted OR: {adj_or:.3f} [{adj_or_ci[0]:.3f}, {adj_or_ci[1]:.3f}]")
        print(f"Percent change in log-odds: {pct_change:.1f}%")

        # Plain language read
        if adj_or > 1:
            direction = "increases"
        else:
            direction = "decreases"
        read = f"After adjusting for ICULOS, Age, and Gender, each 1 SD increase in {var} {direction} sepsis odds by {abs(adj_or - 1):.1%} (OR={adj_or:.3f}, 95% CI [{adj_or_ci[0]:.3f}, {adj_or_ci[1]:.3f}])."

        results[var] = {
            "naive": {
                "coefficient": float(naive_coef),
                "std_error": float(naive_se),
                "or": float(naive_or),
                "or_ci_low": float(naive_or_ci[0]),
                "or_ci_high": float(naive_or_ci[1]),
            },
            "adjusted": {
                "coefficient": float(adj_coef),
                "std_error": float(adj_se),
                "or": float(adj_or),
                "or_ci_low": float(adj_or_ci[0]),
                "or_ci_high": float(adj_or_ci[1]),
            },
            "percent_change_log_odds": float(pct_change),
            "plain_language": read,
        }

    # Add disclaimer
    results["disclaimer"] = "These are covariate-adjusted associations from observational data, not proof of causation."

    # Save results
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved results to {OUTPUT_PATH}")

    # Print table
    print("\n=== Causal Inference Table ===")
    print(f"{'Variable':<10} {'Naive OR':<12} {'Adjusted OR':<14} {'95% CI':<20} {'Change in log-odds':<20}")
    print("-" * 80)
    for var in variables:
        r = results[var]
        adj_or = r["adjusted"]["or"]
        ci_low = r["adjusted"]["or_ci_low"]
        ci_high = r["adjusted"]["or_ci_high"]
        pct_change = r["percent_change_log_odds"]
        naive_or = r["naive"]["or"]
        print(f"{var:<10} {naive_or:<12.3f} {adj_or:<14.3f} [{ci_low:.3f}, {ci_high:.3f}]   {pct_change:<20.1f}%")

    print("\n=== Plain Language Summaries ===")
    for var in variables:
        print(f"{var}: {results[var]['plain_language']}")

    print(f"\nDisclaimer: {results['disclaimer']}")

    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed:.1f}s")

    # Update progress log
    with open(PROJECT_ROOT / "outputs" / "progress_log.md", "a") as f:
        f.write(f"| PART 3: Causal inference | Analyzed {len(variables)} variables with clustered SEs | {elapsed:.1f} |\n")


if __name__ == "__main__":
    main()
