# Import the needed libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import kagglehub
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Download latest version


def load_gold_data(path="gold_data_2015_25.csv"):
    """
    Load Gold Prices dataset from CSV file.
    Returns a pandas DataFrame with Date as index.
    """
    df = pd.read_csv(path)

    # Make sure the date column is parsed as datetime
    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)

    return df


# Example usage
gold_df = load_gold_data("gold_data_2015_25.csv")

# Explore data
print(gold_df.head())
print(gold_df.describe())
print(gold_df.info())
print(list(gold_df.columns))

gold_col = "GLD"  # Gold prices
index_cols = ["SPX", "USO", "SLV", "EUR/USD"]  # S&P 500, Oil, Silver, Euro/USD

gold_mean = gold_df["GLD"].mean()
gold_median = gold_df["GLD"].median()
gold_mode = gold_df["GLD"].mode()[0]

print("Gold Mean:", round(gold_mean, 2))
print("Gold Median:", round(gold_median, 2))
print("Gold Mode:", round(gold_mode, 2))

# Add a Year column
gold_df["Year"] = gold_df.index.year

# Group by Year and compute summary stats
yearly_stats = gold_df.groupby("Year")["GLD"].agg(
    ["mean", "median", "std", "min", "max", "count"]
)
yearly_stats.head()

gold_df["GLD"].plot(
    kind="hist", bins=30, figsize=(8, 5), title="Distribution of Gold Prices"
)
plt.xlabel("Gold Price")
plt.show()

comp = gold_df[[gold_col] + index_cols].dropna()

normalized = comp / comp.iloc[0] * 100

plt.figure(figsize=(12, 6))
for col in normalized.columns:
    plt.plot(normalized.index, normalized[col], label=col)
plt.title("Gold vs Indexes (Normalized, start=100)")
plt.ylabel("Index Value")
plt.legend()
plt.show()

outperf_counts = {}
for idx in index_cols:
    mask = normalized[gold_col] > normalized[idx]
    outperf_counts[idx] = mask.sum()
print("Days Gold outperformed:")
print(outperf_counts)

annual = comp.resample("Y").last().pct_change() * 100
annual.index = annual.index.year
annual.round(2)

# Features and target
X = gold_df.drop(columns=["GLD"])
y = gold_df["GLD"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=False
)  # shuffle=False keeps time order if you want chronological split

# Define and train model
model = XGBRegressor(n_estimators=300, learning_rate=0.05,
                     max_depth=4, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("MSE:", mse)
print("RMSE:", rmse)
print("R²:", r2)
