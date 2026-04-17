import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("../data/data.csv")

# --- Data Understanding ---
print("First 5 rows:")
print(df.head())

print("\nSummary Statistics:")
print(df.describe())

# --- Features ---
X = df[["admissions"]]

# --- Models ---
# Beds model
y_beds = df["beds_used"]
model_beds = LinearRegression()
model_beds.fit(X, y_beds)

# Oxygen model
y_oxygen = df["oxygen_used"]
model_oxygen = LinearRegression()
model_oxygen.fit(X, y_oxygen)

# --- Prediction ---
admissions = int(input("\nEnter expected admissions: "))
input_data = pd.DataFrame([[admissions]], columns=["admissions"])

beds_pred = model_beds.predict(input_data)
oxygen_pred = model_oxygen.predict(input_data)

print(f"\nPredicted beds needed: {beds_pred[0]:.2f}")
print(f"Predicted oxygen needed: {oxygen_pred[0]:.2f}")

# --- Insight ---
if beds_pred[0] > 70:
    print("⚠️ High demand expected! Arrange extra beds and oxygen.")
else:
    print("✅ Resources are sufficient.")

# --- Visualization ---
import matplotlib.pyplot as plt

# --- 1. Beds Trend ---
plt.figure()
plt.plot(df["day"], df["beds_used"])
plt.xlabel("Day")
plt.ylabel("Beds Used")
plt.title("Hospital Bed Usage Trend")
plt.savefig("beds_trend.png")
print("Saved beds_trend.png")

# --- 2. Oxygen Trend ---
plt.figure()
plt.plot(df["day"], df["oxygen_used"])
plt.xlabel("Day")
plt.ylabel("Oxygen Used")
plt.title("Oxygen Usage Trend")
plt.savefig("oxygen_trend.png")
print("Saved oxygen_trend.png")

# --- 3. Relationship Plot ---
plt.figure()
plt.scatter(df["admissions"], df["beds_used"])
plt.xlabel("Admissions")
plt.ylabel("Beds Used")
plt.title("Admissions vs Beds Relationship")
plt.savefig("relationship.png")
print("Saved relationship.png")