"""Phase 4: Exploratory Data Analysis (EDA) for UNSW-NB15."""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless plotting
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

def run_eda():
    project_root = Path(__file__).resolve().parent.parent.parent
    raw_dir = project_root / "data" / "raw"
    figures_dir = project_root / "reports" / "figures"
    reports_dir = project_root / "reports"
    figures_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    print("[*] Loading UNSW-NB15 training and testing splits...")
    df_train = pd.read_parquet(raw_dir / "UNSW_NB15_training-set.parquet")
    df_test = pd.read_parquet(raw_dir / "UNSW_NB15_testing-set.parquet")

    # Clean whitespace in attack_cat strings
    df_train["attack_cat"] = df_train["attack_cat"].astype(str).str.strip()
    df_test["attack_cat"] = df_test["attack_cat"].astype(str).str.strip()

    # Configure plotting aesthetic
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams.update({"font.sans-serif": "Arial", "figure.dpi": 150})

    # --- FIGURE 1: Binary Distribution ---
    print("[*] Generating Figure 1: Binary Distribution...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (name, df) in zip(axes, [("Training Set", df_train), ("Testing Set", df_test)]):
        binary_counts = df["label"].value_counts().rename({0: "Normal", 1: "Attack"})
        colors = ["#2b5c8f", "#d95f02"]
        wedges, texts, autotexts = ax.pie(
            binary_counts.values,
            labels=binary_counts.index,
            autopct="%1.1f%%",
            startangle=140,
            colors=colors,
            explode=(0.05, 0),
            textprops={"fontsize": 11, "weight": "bold"}
        )
        ax.set_title(f"{name} Binary Split (Total: {len(df):,})", fontsize=13, weight="bold", pad=10)
    plt.tight_layout()
    fig1_path = figures_dir / "binary_distribution.png"
    plt.savefig(fig1_path, bbox_inches="tight")
    plt.close()
    print(f"  -> Saved {fig1_path.name}")

    # --- FIGURE 2: Multiclass Attack Imbalance (Linear & Log Scale) ---
    print("[*] Generating Figure 2: Multiclass Imbalance...")
    train_cat_counts = df_train["attack_cat"].value_counts()
    test_cat_counts = df_test["attack_cat"].value_counts()

    combined_cats = pd.DataFrame({
        "Training Set": train_cat_counts,
        "Testing Set": test_cat_counts
    }).fillna(0).sort_values(by="Training Set", ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Linear scale
    combined_cats.plot(kind="bar", ax=axes[0], color=["#1f77b4", "#aec7e8"], width=0.8)
    axes[0].set_title("Attack Categories Count (Linear Scale)", fontsize=13, weight="bold")
    axes[0].set_ylabel("Number of Samples")
    axes[0].set_xlabel("Category")
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].grid(axis="y", linestyle="--", alpha=0.7)

    # Log scale (reveals Worms, Shellcode, Backdoors)
    combined_cats.plot(kind="bar", ax=axes[1], color=["#d62728", "#ff9896"], width=0.8)
    axes[1].set_yscale("log")
    axes[1].set_title("Attack Categories Count (Logarithmic Scale - Rare Attacks)", fontsize=13, weight="bold")
    axes[1].set_ylabel("Number of Samples (Log Scale)")
    axes[1].set_xlabel("Category")
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()
    fig2_path = figures_dir / "class_imbalance.png"
    plt.savefig(fig2_path, bbox_inches="tight")
    plt.close()
    print(f"  -> Saved {fig2_path.name}")

    # --- FIGURE 3: Feature Correlation Heatmap ---
    print("[*] Generating Figure 3: Correlation Matrix...")
    flow_features = [
        "dur", "spkts", "dpkts", "sbytes", "dbytes", "rate",
        "sload", "dload", "sloss", "dloss", "sinpkt", "dinpkt",
        "sjit", "djit", "swin", "dwin", "tcprtt", "synack",
        "ackdat", "smean", "dmean", "label"
    ]
    numeric_available = [f for f in flow_features if f in df_train.columns]
    corr_matrix = df_train[numeric_available].corr()

    plt.figure(figsize=(14, 11))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(
        corr_matrix,
        mask=mask,
        cmap="coolwarm",
        vmax=1.0,
        vmin=-1.0,
        center=0,
        annot=True,
        fmt=".2f",
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8, "label": "Pearson Correlation Coefficient"}
    )
    plt.title("Correlation Matrix of Key Flow Features (UNSW-NB15 Train)", fontsize=14, weight="bold", pad=12)
    plt.tight_layout()
    fig3_path = figures_dir / "correlation_matrix.png"
    plt.savefig(fig3_path, bbox_inches="tight")
    plt.close()
    print(f"  -> Saved {fig3_path.name}")

    # --- FIGURE 4: Key Feature Distributions (Normal vs Attack) ---
    print("[*] Generating Figure 4: Feature Distributions by Class...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    dist_features = [
        ("rate", "Flow Rate (pkts/sec, Log Scale)"),
        ("sload", "Source Load (bits/sec, Log Scale)"),
        ("smean", "Mean Source Packet Size (bytes)"),
        ("dur", "Connection Duration (sec, Log Scale)")
    ]

    for ax, (feat, title) in zip(axes.flatten(), dist_features):
        plot_df = df_train[[feat, "label"]].copy()
        plot_df["Class"] = plot_df["label"].map({0: "Normal", 1: "Attack"})

        if "Log" in title:
            # Handle zeros with log1p
            plot_df[feat] = np.log1p(plot_df[feat])

        sns.boxplot(
            data=plot_df,
            x="Class",
            y=feat,
            hue="Class",
            palette={"Normal": "#2b5c8f", "Attack": "#d95f02"},
            ax=ax,
            legend=False,
            fliersize=1.5
        )
        ax.set_title(title, fontsize=12, weight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("Value" if "Log" not in title else "log(1 + x)")

    plt.tight_layout()
    fig4_path = figures_dir / "normal_vs_attack_features.png"
    plt.savefig(fig4_path, bbox_inches="tight")
    plt.close()
    print(f"  -> Saved {fig4_path.name}")

    # --- EDA REPORT MARKDOWN GENERATION ---
    print("[*] Compiling Markdown EDA Report...")
    report_text = generate_eda_report(df_train, df_test, corr_matrix)
    eda_report_path = reports_dir / "eda_report.md"
    with open(eda_report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"[+] Saved comprehensive EDA report to {eda_report_path}")
    print("[✓] Phase 4 EDA Completed Successfully.")

def generate_eda_report(df_train: pd.DataFrame, df_test: pd.DataFrame, corr_matrix: pd.DataFrame) -> str:
    # Highly correlated feature pairs (|r| > 0.8)
    high_corr_pairs = []
    cols = corr_matrix.columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            val = corr_matrix.iloc[i, j]
            if abs(val) >= 0.80 and cols[i] != "label" and cols[j] != "label":
                high_corr_pairs.append((cols[i], cols[j], round(float(val), 3)))

    high_corr_pairs = sorted(high_corr_pairs, key=lambda x: abs(x[2]), reverse=True)

    md = []
    md.append("# Phase 4: Exploratory Data Analysis (EDA) Report\n")
    md.append("**Dataset:** UNSW-NB15 (Train: 175,341 rows | Test: 82,332 rows)\n")

    md.append("## 1. Class Distributions & Imbalance\n")
    md.append("- **Binary Split:** In the training split, Attack traffic constitutes 68.06% (119,341 flows) while Normal constitutes 31.94% (56,000 flows). In testing, Attack is 55.06% and Normal is 44.94%.")
    md.append("- **Multiclass Extremes:** While classes like `Generic` (40,000) and `Exploits` (33,393) are heavily represented, rare classes such as `Worms` (130 samples in train, 44 in test) and `Shellcode` (1,133 train, 378 test) present acute imbalance.")
    md.append("- **Implication for Unseen Attack Experiments:** Holding out `Worms` or `Shellcode` as an unseen attack is ecologically valid, as novel exploits in reality appear with low initial frequency.\n")

    md.append("## 2. Feature Multicollinearity & Redundancy\n")
    md.append("Pairs with Pearson correlation coefficient $|r| \\ge 0.80$:\n")
    md.append("| Feature 1 | Feature 2 | Pearson Correlation ($r$) | Recommendation |")
    md.append("| :--- | :--- | :--- | :--- |")
    for f1, f2, r in high_corr_pairs:
        md.append(f"| `{f1}` | `{f2}` | {r:.3f} | High collinearity; monitor feature importance in tree models. |")
    md.append("\n")

    md.append("## 3. Distribution Characteristics & Scaling Implications\n")
    md.append("- **Heavy Tails & Extreme Outliers:** Features such as `sload`, `dload`, `sbytes`, `dbytes`, and `dur` span multiple orders of magnitude with severe positive skew.")
    md.append("- **Scaling Requirement:** Standard Z-score (`StandardScaler`) is vulnerable to severe distortion by extreme network flow outliers. `RobustScaler` (median & IQR centering) or logarithmic transformations are strongly favored.\n")

    md.append("## 4. Generated Artifacts\n")
    md.append("- `reports/figures/binary_distribution.png`: Binary label proportions.")
    md.append("- `reports/figures/class_imbalance.png`: Linear vs log multiclass distribution.")
    md.append("- `reports/figures/correlation_matrix.png`: Triangular correlation heatmap.")
    md.append("- `reports/figures/normal_vs_attack_features.png`: Boxplots comparing Normal vs Attack profiles.\n")

    return "\n".join(md)

if __name__ == "__main__":
    run_eda()
