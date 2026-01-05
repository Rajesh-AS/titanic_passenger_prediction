import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# Load data
data = pd.read_csv("train.csv")

# Select features
features = ["p_class", "sex", "age", "sib_sp", "parch", "fare", "embarked"]
target = "survived"

# Handle missing values
data["age"].fillna(data["age"].median(), inplace=True)
data["fare"].fillna(data["fare"].median(), inplace=True)
data["embarked"].fillna(data["embarked"].mode()[0], inplace=True)

# Encode categorical columns
le_sex = LabelEncoder()
le_embarked = LabelEncoder()

data["sex"] = le_sex.fit_transform(data["sex"])
data["embarked"] = le_embarked.fit_transform(data["embarked"])

# Train-test split
X = data[features]
y = data[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model and encoders
with open("titanic_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("encoders.pkl", "wb") as f:
    pickle.dump((le_sex, le_embarked), f)

print("✅ Model trained and saved successfully!")
