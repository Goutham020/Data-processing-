import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the data
train = pd.read_csv(r"C:\Users\Admin\Downloads\archive (1)\train_genetic_disorders.csv")
test = pd.read_csv(r"C:\Users\Admin\Downloads\archive (1)\test_genetic_disorders.csv")

# Explore the data
print("✅ Data loaded")
print("Train shape:", train.shape)
print("\nTrain Info:")
print(train.info())
print("\nMissing values:\n", train.isnull().sum())
# Fill missing values
for df in [train, test]:
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype == 'object':
                df[col] = df[col].fillna(df[col].mode()[0])
            else:
                df[col] = df[col].fillna(df[col].median())

print("✅ Missing values filled.")
# Drop target column temporarily from train
target_column = 'genetic_disorder'  # change if your dataset has a different target
train_features = train.drop(columns=[target_column])
combined = pd.concat([train_features, test], axis=0)

# One-hot encode
combined_encoded = pd.get_dummies(combined)

# Split back
train_encoded = combined_encoded.iloc[:len(train), :]
test_encoded = combined_encoded.iloc[len(train):, :]

# Add back target
train_encoded[target_column] = train[target_column].values

print("✅ Encoding done. Encoded shape:", train_encoded.shape)
# Drop target column temporarily from train
target_column = 'genetic_disorder'  # change if your dataset has a different target
train_features = train.drop(columns=[target_column])
combined = pd.concat([train_features, test], axis=0)

# One-hot encode
combined_encoded = pd.get_dummies(combined)

# Split back
train_encoded = combined_encoded.iloc[:len(train), :]
test_encoded = combined_encoded.iloc[len(train):, :]

# Add back target
train_encoded[target_column] = train[target_column].values

print("✅ Encoding done. Encoded shape:", train_encoded.shape)
from sklearn.preprocessing import StandardScaler

X = train_encoded.drop(target_column, axis=1)
y = train_encoded[target_column]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
test_scaled = scaler.transform(test_encoded)

print("✅ Feature scaling complete")
# Select 2 numerical columns (if available)
numeric_cols = X.select_dtypes(include='number').columns[:2]

for col in numeric_cols:
    plt.figure(figsize=(6, 3))
    sns.boxplot(x=X[col])
    plt.title(f"Boxplot - {col}")
    plt.show()

# Function to remove outliers
def remove_outliers_iqr(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return df[(df[col] >= lower) & (df[col] <= upper)]

# Apply to original (non-scaled) train_encoded
for col in numeric_cols:
    train_encoded = remove_outliers_iqr(train_encoded, col)

print("✅ Outliers removed")

