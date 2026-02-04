import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("titanic.csv")

# Select required columns
data = data[['Survived', 'Pclass', 'Sex', 'Age', 'Fare']]

# Handle missing Age values safely (no warning)
data.loc[:, 'Age'] = data['Age'].fillna(data['Age'].median())

# Convert Sex column to numeric
data.loc[:, 'Sex'] = data['Sex'].map({'male': 0, 'female': 1})

# Split features and target
X = data.drop('Survived', axis=1)
y = data['Survived']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Print accuracy
print("Model Accuracy:", accuracy_score(y_test, y_pred))
