import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle

# Load dataset
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Select only important features
selected_features = [
    "tenure","MonthlyCharges","TotalCharges",
    "Contract","InternetService",
    "TechSupport","OnlineSecurity","PaymentMethod","Churn"
]

df = df[selected_features]

# Fix TotalCharges
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors='coerce')
df.fillna(0, inplace=True)

# Convert target
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# Encode categorical
for col in df.columns:
    if df[col].dtype == 'object':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

# Split
X = df.drop("Churn", axis=1)
y = df["Churn"]

# Scale
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Save scaler
pickle.dump(scaler, open("model/scaler.pkl", "wb"))

# Save data
pd.DataFrame(X).to_csv("model/X.csv", index=False)
pd.DataFrame(y).to_csv("model/y.csv", index=False)

print("✅ Preprocessing done (Reduced features)")