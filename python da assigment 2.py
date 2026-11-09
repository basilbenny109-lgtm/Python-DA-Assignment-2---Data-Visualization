import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns




# =====================================
# 1. LOAD THE TAXIS DATASET
# =====================================

df = sns.load_dataset("taxis")

print("First five rows:")
print(df.head())

print("\nDataset information:")
print(df.info())

# =====================================
# 2. HANDLE MISSING VALUES
# =====================================

print("\nMissing values before cleaning:")
print(df.isnull().sum())

# Fill numerical columns using median
numerical_columns = df.select_dtypes(include=np.number).columns

for column in numerical_columns:
    df[column] = df[column].fillna(df[column].median())

# Fill categorical columns using mode
categorical_columns = df.select_dtypes(
    include=["object", "category"]
).columns

for column in categorical_columns:
    if df[column].isnull().sum() > 0:
        df[column] = df[column].fillna(df[column].mode()[0])

# Convert pickup into datetime
df["pickup"] = pd.to_datetime(df["pickup"])

# Remove rows missing essential chart information
df = df.dropna(
    subset=["pickup", "fare", "distance", "pickup_borough"]
)

print("\nMissing values after cleaning:")
print(df.isnull().sum())

# =====================================
# 3. MATPLOTLIB VISUALIZATIONS
# =====================================

# Line Chart
plt.figure(figsize=(10, 5))
plt.plot(df["pickup"], df["fare"])
plt.title("Taxi Fare Over Time")
plt.xlabel("Pickup Date")
plt.ylabel("Fare")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Bar Chart
borough_fare = df.groupby("pickup_borough")["fare"].sum()

plt.figure(figsize=(8, 5))
borough_fare.plot(kind="bar")
plt.title("Total Fare by Pickup Borough")
plt.xlabel("Pickup Borough")
plt.ylabel("Total Fare")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Pie Chart
payment_counts = df["payment"].value_counts().dropna()

plt.figure(figsize=(7, 7))
plt.pie(
    payment_counts,
    labels=payment_counts.index,
    autopct="%1.1f%%",
    startangle=90
)
plt.title("Trips by Payment Method")
plt.show()

# Histogram
plt.figure(figsize=(8, 5))
plt.hist(df["distance"].dropna(), bins=20)
plt.title("Distribution of Taxi Distance")
plt.xlabel("Distance")
plt.ylabel("Number of Trips")
plt.show()

# Box Plot
plt.figure(figsize=(9, 5))
sns.boxplot(data=df, x="pickup_borough", y="tip")
plt.title("Tip Amount by Pickup Borough")
plt.xlabel("Pickup Borough")
plt.ylabel("Tip")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# =====================================
# 4. SEABORN VISUALIZATIONS
# =====================================

# Count Plot
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="pickup_borough")
plt.title("Number of Trips by Pickup Borough")
plt.xlabel("Pickup Borough")
plt.ylabel("Number of Trips")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Scatter Plot
plt.figure(figsize=(9, 5))
sns.scatterplot(
    data=df,
    x="distance",
    y="fare",
    hue="pickup_borough"
)
plt.title("Distance vs Fare")
plt.show()

# Heatmap
correlation = df[
    ["distance", "fare", "tip", "tolls", "total"]
].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)
plt.title("Correlation Heatmap")
plt.show()

# Pair Plot
sns.pairplot(
    df,
    vars=["distance", "fare", "tip", "total"],
    hue="pickup_zone"
)
plt.show()

# Violin Plot
plt.figure(figsize=(8, 5))
sns.violinplot(
    data=df,
    x="payment",
    y="fare"
)
plt.title("Fare Distribution by Payment Method")
plt.xlabel("Payment Method")
plt.ylabel("Fare")
plt.show()
