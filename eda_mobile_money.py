"""
==========================================================
Uganda Mobile Money Dataset
Exploratory Data Analysis (EDA)

Author: Charis Opol
Dissertation

Hybrid Explainable LSTM-FNN Framework
==========================================================
"""

from pathlib import Path

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import (
    skew,
    kurtosis,
    zscore
)

from scipy import stats

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# ==========================================================
# DIRECTORIES
# ==========================================================

PROJECT_DIR = Path(
    r"C:\Users\chari\Desktop\Hybrid_LSTM-FNN-Explainable_framework_for_credtirisk-assessment"
)

DATA_PATH = (
    PROJECT_DIR
    / "data"
    / "raw"
    / "uganda_mobile_money_master.csv"
)

EDA_DIR = PROJECT_DIR / "EDA"

FIGURE_DIR = EDA_DIR / "figures"

TABLE_DIR = EDA_DIR / "tables"

REPORT_DIR = EDA_DIR / "reports"

for folder in [
    EDA_DIR,
    FIGURE_DIR,
    TABLE_DIR,
    REPORT_DIR
]:

    folder.mkdir(
        parents=True,
        exist_ok=True
    )


# ==========================================================
# PLOT STYLE
# ==========================================================

plt.style.use("ggplot")

sns.set_theme(
    style="whitegrid"
)

plt.rcParams["figure.figsize"] = (10,6)

plt.rcParams["figure.dpi"] = 150


# ==========================================================
# SAVE FIGURE
# ==========================================================

def save_plot(filename):

    plt.tight_layout()

    plt.savefig(

        FIGURE_DIR / filename,

        dpi=300,

        bbox_inches="tight"

    )

    plt.close()


# ==========================================================
# LOAD DATASET
# ==========================================================

print("="*70)
print("Loading Dataset")
print("="*70)

df = pd.read_csv(

    DATA_PATH,

    parse_dates=["timestamp"]

)

print(df.head())

print()

print(df.shape)

print()

print(df.info())

print()


# ==========================================================
# SAVE COLUMN INFORMATION
# ==========================================================

columns = pd.DataFrame({

    "Column":df.columns,

    "DataType":df.dtypes.astype(str)

})

columns.to_csv(

    TABLE_DIR/"column_information.csv",

    index=False

)


# ==========================================================
# DATASET DIMENSIONS
# ==========================================================

summary = {

    "Rows":[len(df)],

    "Columns":[len(df.columns)],

    "Borrowers":[

        df.borrower_id.nunique()

    ],

    "Transactions":[

        len(df)

    ]

}

summary = pd.DataFrame(summary)

summary.to_csv(

    TABLE_DIR/"dataset_summary.csv",

    index=False

)

print(summary)


# ==========================================================
# MEMORY USAGE
# ==========================================================

memory = (

    df.memory_usage(

        deep=True

    )/1024**2

)

memory = memory.reset_index()

memory.columns=[

    "Column",

    "Memory_MB"

]

memory.to_csv(

    TABLE_DIR/"memory_usage.csv",

    index=False

)


# ==========================================================
# MISSING VALUES
# ==========================================================

missing = pd.DataFrame({

    "Missing":

    df.isnull().sum(),

    "Percent":

    df.isnull().mean()*100

})

missing = missing.sort_values(

    "Percent",

    ascending=False

)

missing.to_csv(

    TABLE_DIR/"missing_values.csv"

)

plt.figure(figsize=(12,6))

sns.barplot(

    x=missing.index,

    y=missing["Percent"]

)

plt.xticks(rotation=90)

plt.ylabel("Percent")

plt.title("Missing Values")

save_plot(

    "missing_values.png"

)


# ==========================================================
# DUPLICATES
# ==========================================================

duplicates = df.duplicated().sum()

duplicate_table = pd.DataFrame({

    "Duplicate Records":[duplicates]

})

duplicate_table.to_csv(

    TABLE_DIR/"duplicates.csv",

    index=False

)

print()

print("Duplicate Records:",duplicates)


# ==========================================================
# DESCRIPTIVE STATISTICS
# ==========================================================

description = df.describe(

    include="all"

)

description.to_csv(

    TABLE_DIR/"descriptive_statistics.csv"

)


# ==========================================================
# NUMERIC VARIABLES
# ==========================================================

numeric_columns = df.select_dtypes(

    include=np.number

).columns.tolist()

print()

print(numeric_columns)


# ==========================================================
# MEAN
# ==========================================================

means = df[numeric_columns].mean()

medians = df[numeric_columns].median()

modes = df[numeric_columns].mode().iloc[0]

variances = df[numeric_columns].var()

std = df[numeric_columns].std()

minimum = df[numeric_columns].min()

maximum = df[numeric_columns].max()

iqr = (

    df[numeric_columns].quantile(.75)

    -

    df[numeric_columns].quantile(.25)

)

statistics = pd.DataFrame({

    "Mean":means,

    "Median":medians,

    "Mode":modes,

    "Variance":variances,

    "Std":std,

    "Minimum":minimum,

    "Maximum":maximum,

    "IQR":iqr

})

statistics.to_csv(

    TABLE_DIR/"summary_statistics.csv"

)


# ==========================================================
# SKEWNESS
# ==========================================================

skewness = df[numeric_columns].apply(

    skew

)

# ==========================================================
# KURTOSIS
# ==========================================================

kurt = df[numeric_columns].apply(

    kurtosis

)

shape = pd.DataFrame({

    "Skewness":skewness,

    "Kurtosis":kurt

})

shape.to_csv(

    TABLE_DIR/"skewness_kurtosis.csv"

)

print()

print(shape)


# ==========================================================
# HISTOGRAMS
# ==========================================================

for column in numeric_columns:

    plt.figure(figsize=(8,5))

    sns.histplot(

        df[column],

        kde=True,

        bins=40

    )

    plt.title(

        f"Distribution of {column}"

    )

    save_plot(

        f"hist_{column}.png"

    )


# ==========================================================
# BOXPLOTS
# ==========================================================

for column in numeric_columns:

    plt.figure(figsize=(10,2))

    sns.boxplot(

        x=df[column]

    )

    plt.title(

        f"Boxplot - {column}"

    )

    save_plot(

        f"box_{column}.png"

    )


# ==========================================================
# OUTLIER ANALYSIS
# ==========================================================

outlier_results = []

for column in numeric_columns:

    q1 = df[column].quantile(.25)

    q3 = df[column].quantile(.75)

    iqr = q3-q1

    lower = q1-1.5*iqr

    upper = q3+1.5*iqr

    count = (

        (

            df[column]<lower

        )|

        (

            df[column]>upper

        )

    ).sum()

    outlier_results.append({

        "Feature":column,

        "Outliers":count

    })

outliers = pd.DataFrame(

    outlier_results

)

outliers.to_csv(

    TABLE_DIR/"outliers.csv",

    index=False

)


# ==========================================================
# Z SCORE OUTLIERS
# ==========================================================

z_summary = []

for column in numeric_columns:

    z = np.abs(

        zscore(

            df[column],

            nan_policy="omit"

        )

    )

    count = (z>3).sum()

    z_summary.append({

        "Feature":column,

        "ZScore_Outliers":count

    })

pd.DataFrame(

    z_summary

).to_csv(

    TABLE_DIR/"zscore_outliers.csv",

    index=False

)

print()

print("Part 1 Complete")

# ==========================================================
# CORRELATION ANALYSIS
# ==========================================================

print("\nComputing Correlation Matrices...")

pearson_corr = df[numeric_columns].corr(method="pearson")

spearman_corr = df[numeric_columns].corr(method="spearman")

pearson_corr.to_csv(
    TABLE_DIR/"pearson_correlation.csv"
)

spearman_corr.to_csv(
    TABLE_DIR/"spearman_correlation.csv"
)

plt.figure(figsize=(14,10))

sns.heatmap(
    pearson_corr,
    cmap="coolwarm",
    center=0,
    square=True
)

plt.title("Pearson Correlation Matrix")

save_plot(
    "pearson_correlation_heatmap.png"
)

plt.figure(figsize=(14,10))

sns.heatmap(
    spearman_corr,
    cmap="viridis",
    center=0,
    square=True
)

plt.title("Spearman Correlation Matrix")

save_plot(
    "spearman_correlation_heatmap.png"
)


# ==========================================================
# TARGET DISTRIBUTION
# ==========================================================

plt.figure(figsize=(6,5))

sns.countplot(
    data=df,
    x="default_label"
)

plt.title(
    "Loan Default Distribution"
)

save_plot(
    "default_distribution.png"
)

target_summary = (

    df["default_label"]

    .value_counts()

    .rename_axis("Default")

    .reset_index(name="Count")

)

target_summary.to_csv(

    TABLE_DIR/"target_distribution.csv",

    index=False

)


# ==========================================================
# TRANSACTION TYPE
# ==========================================================

plt.figure(figsize=(7,5))

sns.countplot(
    data=df,
    x="transaction_type"
)

plt.title(
    "Transaction Type Distribution"
)

save_plot(
    "transaction_type_distribution.png"
)


# ==========================================================
# MONTHLY TRANSACTION COUNT
# ==========================================================

monthly_transactions = (

    df.groupby("month")

    .size()

    .reset_index(name="Transactions")

)

monthly_transactions.to_csv(

    TABLE_DIR/"monthly_transaction_count.csv",

    index=False

)

plt.figure(figsize=(10,5))

sns.lineplot(

    data=monthly_transactions,

    x="month",

    y="Transactions",

    marker="o"

)

plt.title(
    "Monthly Transaction Count"
)

save_plot(
    "monthly_transaction_count.png"
)


# ==========================================================
# MONTHLY CASHFLOW
# ==========================================================

monthly_cashflow = (

    df.groupby("month")["amount"]

    .sum()

    .reset_index()

)

monthly_cashflow.to_csv(

    TABLE_DIR/"monthly_cashflow.csv",

    index=False

)

plt.figure(figsize=(10,5))

sns.barplot(

    data=monthly_cashflow,

    x="month",

    y="amount"

)

plt.title(
    "Monthly Transaction Value"
)

save_plot(
    "monthly_cashflow.png"
)


# ==========================================================
# OCCUPATION DISTRIBUTION
# ==========================================================

occupation_counts = (

    df["occupation"]

    .value_counts()

    .reset_index()

)

occupation_counts.columns=[

    "Occupation",

    "Count"

]

occupation_counts.to_csv(

    TABLE_DIR/"occupation_distribution.csv",

    index=False

)

plt.figure(figsize=(12,6))

sns.countplot(

    data=df,

    y="occupation",

    order=df["occupation"].value_counts().index

)

plt.title(
    "Occupation Distribution"
)

save_plot(
    "occupation_distribution.png"
)


# ==========================================================
# NETWORK DISTRIBUTION
# ==========================================================

plt.figure(figsize=(7,5))

sns.countplot(
    data=df,
    x="network"
)

plt.title(
    "Preferred Network"
)

save_plot(
    "network_distribution.png"
)


# ==========================================================
# CHANNEL DISTRIBUTION
# ==========================================================

plt.figure(figsize=(8,5))

sns.countplot(
    data=df,
    x="channel"
)

plt.title(
    "Transaction Channel"
)

save_plot(
    "channel_distribution.png"
)


# ==========================================================
# BORROWER LEVEL FEATURES
# ==========================================================

print("\nCreating Borrower-Level Features...")

received = (

    df[df["transaction_type"]=="Received"]

    .groupby("borrower_id")["amount"]

    .sum()

)

sent = (

    df[df["transaction_type"]=="Sent"]

    .groupby("borrower_id")["amount"]

    .sum()

)

borrower = (

    df.groupby("borrower_id")

    .agg({

        "amount":[

            "count",

            "mean",

            "median",

            "std",

            "sum"

        ],

        "balance_after":"mean",

        "financial_health":"mean",

        "default_label":"max",

        "age":"first",

        "monthly_income":"first",

        "loan_amount":"first",

        "interest_rate":"first",

        "loan_term_months":"first",

        "occupation":"first",

        "region":"first",

        "district":"first"

    })

)

borrower.columns=[

    "_".join(col).strip("_")

    for col in borrower.columns

]

borrower.reset_index(inplace=True)


# ==========================================================
# NET CASHFLOW
# ==========================================================

borrower["received"] = (

    borrower["borrower_id"]

    .map(received)

    .fillna(0)

)

borrower["sent"] = (

    borrower["borrower_id"]

    .map(sent)

    .fillna(0)

)

borrower["net_cashflow"] = (

    borrower["received"]

    -

    borrower["sent"]

)


# ==========================================================
# TRANSACTION VELOCITY
# ==========================================================

days_active = (

    df.groupby("borrower_id")["timestamp"]

    .agg(

        lambda x:

        (x.max()-x.min()).days+1

    )

)

borrower["days_active"] = (

    borrower["borrower_id"]

    .map(days_active)

)

borrower["transaction_velocity"] = (

    borrower["amount_count"]

    /

    borrower["days_active"]

)

borrower.to_csv(

    TABLE_DIR/

    "borrower_features.csv",

    index=False

)


# ==========================================================
# TRANSACTION VELOCITY HISTOGRAM
# ==========================================================

plt.figure(figsize=(8,5))

sns.histplot(

    borrower["transaction_velocity"],

    bins=40,

    kde=True

)

plt.title(
    "Transaction Velocity"
)

save_plot(
    "transaction_velocity.png"
)


# ==========================================================
# NET CASHFLOW HISTOGRAM
# ==========================================================

plt.figure(figsize=(8,5))

sns.histplot(

    borrower["net_cashflow"],

    bins=40,

    kde=True

)

plt.title(
    "Net Cashflow"
)

save_plot(
    "net_cashflow.png"
)


# ==========================================================
# CHI SQUARE TESTS
# ==========================================================

categorical = [

    "occupation",

    "gender",

    "education",

    "marital_status",

    "network",

    "channel"

]

chi_results=[]

for col in categorical:

    table = pd.crosstab(

        df[col],

        df["default_label"]

    )

    chi,p,_,_ = stats.chi2_contingency(table)

    chi_results.append({

        "Feature":col,

        "ChiSquare":chi,

        "PValue":p

    })

pd.DataFrame(

    chi_results

).to_csv(

    TABLE_DIR/

    "chi_square_results.csv",

    index=False

)


# ==========================================================
# T TESTS
# ==========================================================

ttest=[]

for col in numeric_columns:

    a = df.loc[

        df.default_label==0,

        col

    ]

    b = df.loc[

        df.default_label==1,

        col

    ]

    t,p = stats.ttest_ind(

        a,

        b,

        equal_var=False,

        nan_policy="omit"

    )

    ttest.append({

        "Feature":col,

        "TStatistic":t,

        "PValue":p

    })

pd.DataFrame(

    ttest

).to_csv(

    TABLE_DIR/

    "t_test_results.csv",

    index=False

)


# ==========================================================
# ANOVA
# ==========================================================

anova=[]

for feature in [

    "monthly_income",

    "loan_amount",

    "financial_health"

]:

    groups=[]

    for occ in df["occupation"].unique():

        groups.append(

            df.loc[

                df["occupation"]==occ,

                feature

            ]

        )

    f,p = stats.f_oneway(*groups)

    anova.append({

        "Feature":feature,

        "FStatistic":f,

        "PValue":p

    })

pd.DataFrame(

    anova

).to_csv(

    TABLE_DIR/

    "anova_results.csv",

    index=False

)

print("\nPart 2 Complete.")

# ==========================================================
# BORROWER PCA ANALYSIS
# ==========================================================

print("\nRunning PCA...")

borrower_numeric = borrower.select_dtypes(
    include=np.number
).copy()

# Remove identifiers and target
drop_columns = [
    "borrower_id",
    "default_label_max"
]

existing = [
    c for c in drop_columns
    if c in borrower_numeric.columns
]

borrower_numeric = borrower_numeric.drop(
    columns=existing
)

feature_names = borrower_numeric.columns.tolist()

scaler = StandardScaler()

X = scaler.fit_transform(
    borrower_numeric
)

pca = PCA()

X_pca = pca.fit_transform(X)

explained = pd.DataFrame({

    "Component":
    np.arange(
        1,
        len(feature_names)+1
    ),

    "ExplainedVariance":
    pca.explained_variance_ratio_,

    "CumulativeVariance":
    np.cumsum(
        pca.explained_variance_ratio_
    )

})

explained.to_csv(

    TABLE_DIR/
    "pca_explained_variance.csv",

    index=False

)


# ==========================================================
# SCREE PLOT
# ==========================================================

plt.figure(figsize=(9,5))

plt.plot(

    explained["Component"],

    explained["ExplainedVariance"],

    marker="o"

)

plt.xlabel("Principal Component")

plt.ylabel("Explained Variance")

plt.title("PCA Scree Plot")

save_plot(
    "pca_scree_plot.png"
)


# ==========================================================
# CUMULATIVE VARIANCE
# ==========================================================

plt.figure(figsize=(9,5))

plt.plot(

    explained["Component"],

    explained["CumulativeVariance"],

    marker="o"

)

plt.axhline(

    0.95,

    color="red",

    linestyle="--"

)

plt.xlabel("Principal Component")

plt.ylabel("Cumulative Variance")

plt.title("PCA Cumulative Explained Variance")

save_plot(
    "pca_cumulative_variance.png"
)


# ==========================================================
# PCA 2D
# ==========================================================

pca2 = PCA(
    n_components=2
)

coords = pca2.fit_transform(X)

pca_df = pd.DataFrame({

    "PC1":coords[:,0],

    "PC2":coords[:,1],

    "Default":
    borrower["default_label_max"]

})

pca_df.to_csv(

    TABLE_DIR/

    "borrower_pca2.csv",

    index=False

)

plt.figure(figsize=(9,7))

sns.scatterplot(

    data=pca_df,

    x="PC1",

    y="PC2",

    hue="Default",

    alpha=.7

)

plt.title(
    "Borrower PCA (2 Components)"
)

save_plot(
    "borrower_pca_2d.png"
)


# ==========================================================
# PCA 3D
# ==========================================================

from mpl_toolkits.mplot3d import Axes3D

pca3 = PCA(
    n_components=3
)

coords3 = pca3.fit_transform(X)

fig = plt.figure(figsize=(9,7))

ax = fig.add_subplot(
    111,
    projection="3d"
)

scatter = ax.scatter(

    coords3[:,0],

    coords3[:,1],

    coords3[:,2],

    c=borrower["default_label_max"],

    alpha=.7

)

ax.set_xlabel("PC1")

ax.set_ylabel("PC2")

ax.set_zlabel("PC3")

plt.title("Borrower PCA 3D")

save_plot(
    "borrower_pca_3d.png"
)


# ==========================================================
# PCA LOADINGS
# ==========================================================

loadings = pd.DataFrame(

    pca.components_.T,

    columns=[

        f"PC{i}"

        for i in range(

            1,

            len(feature_names)+1

        )

    ],

    index=feature_names

)

loadings.to_csv(

    TABLE_DIR/

    "pca_loadings.csv"

)


# ==========================================================
# TOP CONTRIBUTING FEATURES
# ==========================================================

importance = (

    loadings["PC1"]

    .abs()

    .sort_values(

        ascending=False

    )

)

importance.to_csv(

    TABLE_DIR/

    "feature_importance_pc1.csv"

)

plt.figure(figsize=(10,6))

importance.head(15).plot.bar()

plt.ylabel("Absolute Loading")

plt.title(
    "Top Features Contributing to PC1"
)

save_plot(
    "pc1_feature_importance.png"
)


# ==========================================================
# CORRELATION WITH DEFAULT
# ==========================================================

print("\nComputing Point-Biserial Correlations...")

target = borrower["default_label_max"]

numeric = borrower.select_dtypes(
    include=np.number
)

corr_results = []

for column in numeric.columns:

    if column == "default_label_max":
        continue

    try:

        r,p = stats.pointbiserialr(

            target,

            numeric[column]

        )

        corr_results.append({

            "Feature":column,

            "Correlation":r,

            "PValue":p

        })

    except Exception:

        pass

corr_results = pd.DataFrame(
    corr_results
)

corr_results.to_csv(

    TABLE_DIR/

    "point_biserial_results.csv",

    index=False

)


# ==========================================================
# TOP CORRELATIONS
# ==========================================================

top_corr = (

    corr_results

    .reindex(

        corr_results["Correlation"]

        .abs()

        .sort_values(

            ascending=False

        ).index

    )

)

plt.figure(figsize=(10,6))

plt.bar(

    top_corr["Feature"][:15],

    top_corr["Correlation"][:15]

)

plt.xticks(rotation=90)

plt.ylabel("Correlation")

plt.title(
    "Top Numerical Features Correlated with Default"
)

save_plot(
    "default_correlations.png"
)


# ==========================================================
# CLASS IMBALANCE
# ==========================================================

class_counts = (

    borrower["default_label_max"]

    .value_counts()

)

imbalance = pd.DataFrame({

    "Class":class_counts.index,

    "Count":class_counts.values,

    "Percentage":

    class_counts.values

    /

    class_counts.sum()

    *100

})

imbalance.to_csv(

    TABLE_DIR/

    "class_balance.csv",

    index=False

)


# ==========================================================
# SAVE SUMMARY REPORT
# ==========================================================

with open(

    REPORT_DIR/

    "EDA_Report.txt",

    "w",

    encoding="utf-8"

) as report:

    report.write("="*70+"\n")

    report.write("UGANDA MOBILE MONEY DATASET\n")

    report.write("EXPLORATORY DATA ANALYSIS\n")

    report.write("="*70+"\n\n")

    report.write(f"Rows: {len(df):,}\n")

    report.write(f"Columns: {len(df.columns)}\n")

    report.write(
        f"Borrowers: {df.borrower_id.nunique():,}\n"
    )

    report.write(
        f"Transactions: {len(df):,}\n\n"
    )

    report.write("Missing Values\n")

    report.write(
        str(
            missing
        )
    )

    report.write("\n\n")

    report.write("Summary Statistics Saved.\n")

    report.write("Correlation Analysis Saved.\n")

    report.write("Borrower Features Saved.\n")

    report.write("PCA Analysis Saved.\n")

    report.write("Statistical Tests Saved.\n")

    report.write("Figures Generated Successfully.\n")


# ==========================================================
# FINISHED
# ==========================================================

print("\n" + "="*70)
print("EDA COMPLETE")
print("="*70)

print(f"Figures saved to : {FIGURE_DIR}")
print(f"Tables saved to  : {TABLE_DIR}")
print(f"Reports saved to : {REPORT_DIR}")

print("\nReady for Feature Engineering.")