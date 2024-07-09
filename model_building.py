import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Load the data
df = pd.read_csv('email_expense_transactions.csv')

# Fill missing 'Description' values with an empty string
df['Description'] = df['Description'].fillna('')

# Ensure 'toAccount' is treated as a categorical variable
df['toAccount'] = df['toAccount'].astype('category')

# Define features and target
features = ['amount', 'recipient', 'expense_account', 'Description']
features_without_description = ['amount', 'recipient', 'expense_account']
target = 'toAccount'

# Preprocess categorical features using one-hot encoding
categorical_features_with_description = ['recipient', 'expense_account', 'Description']
categorical_features_without_description = ['recipient', 'expense_account']
numeric_features = ['amount']

# Define the preprocessor for data with the 'Description' feature
preprocessor_with_description = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='mean'), numeric_features),
        ('cat', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ]), categorical_features_with_description)
    ])

# Define the preprocessor for data without the 'Description' feature
preprocessor_without_description = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='mean'), numeric_features),
        ('cat', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ]), categorical_features_without_description)
    ])

# Split the data into training and testing sets
X_with_description = df[features]
X_without_description = df[features_without_description]
y = df[target]

# Identify rows with NaNs in the target
nan_rows = y.isna()

# Use rows with NaNs in the target for prediction
X_to_predict_with_description = X_with_description[nan_rows]
X_to_predict_without_description = X_without_description[nan_rows]
X_with_description = X_with_description[~nan_rows]
X_without_description = X_without_description[~nan_rows]
y = y[~nan_rows]

# Print initial data shapes
print(f"Initial X_with_description shape: {X_with_description.shape}")
print(f"Initial X_without_description shape: {X_without_description.shape}")
print(f"Initial y shape: {y.shape}")
print(f"Initial X_to_predict_with_description shape: {X_to_predict_with_description.shape}")
print(f"Initial X_to_predict_without_description shape: {X_to_predict_without_description.shape}")

# Split the data into training and testing sets
X_train_with_description, X_test_with_description, y_train, y_test = train_test_split(X_with_description, y, test_size=0.2, random_state=42)
X_train_without_description, X_test_without_description, y_train_without_description, y_test_without_description = train_test_split(X_without_description, y, test_size=0.2, random_state=42)

# Check for NaNs
print("NaNs in X_train_with_description:", X_train_with_description.isnull().sum())
print("NaNs in X_test_with_description:", X_test_with_description.isnull().sum())
print("NaNs in y_train:", y_train.isnull().sum())
print("NaNs in y_test:", y_test.isnull().sum())

# Print data splits
print(f"X_train_with_description shape: {X_train_with_description.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_test_with_description shape: {X_test_with_description.shape}")
print(f"y_test shape: {y_test.shape}")
print(f"X_train_without_description shape: {X_train_without_description.shape}")
print(f"y_train_without_description shape: {y_train_without_description.shape}")
print(f"X_test_without_description shape: {X_test_without_description.shape}")
print(f"y_test_without_description shape: {y_test_without_description.shape}")

# Create a pipeline that includes preprocessing and the model
pipeline_with_description = Pipeline(steps=[('preprocessor', preprocessor_with_description),
                                            ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'))])

pipeline_without_description = Pipeline(steps=[('preprocessor', preprocessor_without_description),
                                               ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'))])

# Train the models
print("Training the model with description...")
pipeline_with_description.fit(X_train_with_description, y_train)
print("Training the model without description...")
pipeline_without_description.fit(X_train_without_description, y_train_without_description)

# Make predictions
y_pred_with_description = pipeline_with_description.predict(X_test_with_description)
y_pred_without_description = pipeline_without_description.predict(X_test_without_description)

# Evaluate the models
print("Evaluating the model with description...")
accuracy_with_description = accuracy_score(y_test, y_pred_with_description)
print(f"Accuracy with description: {accuracy_with_description}")
print(classification_report(y_test, y_pred_with_description, zero_division=0))

print("Evaluating the model without description...")
accuracy_without_description = accuracy_score(y_test_without_description, y_pred_without_description)
print(f"Accuracy without description: {accuracy_without_description}")
print(classification_report(y_test_without_description, y_pred_without_description, zero_division=0))

# Function to predict toAccount
def predict_toAccount(data, model_pipeline):
    return model_pipeline.predict(data)

# Predicting with description
print("Predicting with description...")
df['predicted_toAccount_with_description'] = predict_toAccount(df[features], pipeline_with_description)

# Predicting without description
print("Predicting without description...")
df['predicted_toAccount_without_description'] = predict_toAccount(df[features_without_description], pipeline_without_description)

# Use the model to predict 'toAccount' for the rows with NaNs in the target
if not X_to_predict_with_description.empty:
    df.loc[nan_rows, 'predicted_toAccount_with_description'] = predict_toAccount(X_to_predict_with_description, pipeline_with_description)
if not X_to_predict_without_description.empty:
    df.loc[nan_rows, 'predicted_toAccount_without_description'] = predict_toAccount(X_to_predict_without_description, pipeline_without_description)

# Save the updated dataframe to a new CSV file
df.to_csv('predicted_email_expense_transactions.csv', index=False)

print("Prediction process complete. Results saved to 'predicted_email_expense_transactions.csv'.")
