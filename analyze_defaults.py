import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

csv_path = Path("data/raw/uganda_mobile_money_master_5000.csv")

if not csv_path.exists():
    raise FileNotFoundError(f"Could not find {csv_path}. Run the generator first so the CSV exists.")

df = pd.read_csv(csv_path)

if "default_label" not in df.columns:
    raise KeyError("The CSV does not contain a 'default_label' column.")

counts = df["default_label"].value_counts().sort_index()
print("Default label counts:")
print(counts)

plt.figure(figsize=(6, 4))
plt.hist(df["default_label"], bins=[-0.5, 0.5, 1.5], color="steelblue", edgecolor="black")
plt.xticks([0, 1], ["No Default", "Default"])
plt.xlabel("default_label")
plt.ylabel("Count")
plt.title("Distribution of default_label")
plt.tight_layout()
plt.show()

data = pd.read_csv("data/raw/uganda_mobile_money_master_5000.csv")
per_borrower_label = data.groupby("borrower_id")["default_label"].agg(["max", "mean", "nunique"])
print(per_borrower_label["nunique"].value_counts())  # do borrowers have mixed 0/1 rows?
print((per_borrower_label["max"] == 1).mean())        # fraction of borrowers with ANY positive row
print((per_borrower_label["mean"] == 1).mean())       # fraction of borrowers where ALL rows are positive

import pandas as pd
data = pd.read_csv("data/raw/uganda_mobile_money_master_5000.csv")
data = data.sort_values(["borrower_id", "timestamp"])  # match your real date column name

# What does the LAST row per borrower say, vs. the max?
last_label = data.groupby("borrower_id")["default_label"].last()
max_label = data.groupby("borrower_id")["default_label"].max()

print("Last-row positive rate:", last_label.mean())
print("Max-row positive rate:", max_label.mean())
