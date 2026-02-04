import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE

# 1. Loading
print("Loading dataset...")
df = pd.read_csv('creditcard.csv')
print("Dataset loaded successfully!\n")

# 2. Preprocessing
print("Performing data preprocessing...")
scaler = StandardScaler()
df['Amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
df = df.drop(['Time'], axis=1)

X = df.drop('Class', axis=1)
y = df['Class']

# Handling class imbalance
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X, y)

print("Splitting dataset into training and testing sets...\n")
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

# 3. Training
print("Training model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print("Model trained successfully!\n")

# 4. Predictions
print("Making predictions on test data...")
y_pred = model.predict(X_test)
print("Predictions completed.\n")

# 5. Final Output
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print("\nProject Completed Successfully! 🚀")