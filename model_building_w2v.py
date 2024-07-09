import pandas as pd
import gensim
from gensim.models import Word2Vec
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import numpy as np

# Load the data
df = pd.read_csv('email_expense_transactions.csv')

# Fill missing 'Description' values with an empty string
df['Description'] = df['Description'].fillna('')
print("Data loaded and 'Description' column filled with empty strings if missing.")

# Ensure 'toAccount' is treated as a categorical variable
df['toAccount'] = df['toAccount'].astype('category')
print("'toAccount' column converted to categorical.")

# Train Word2Vec model on the 'Description' column
descriptions = df['Description'].apply(lambda x: x.split())
word2vec_model = Word2Vec(sentences=descriptions, vector_size=100, window=5, min_count=1, workers=4)
print("Word2Vec model trained on 'Description' column.")

# Function to transform a description to its Word2Vec embedding
def get_word2vec_vector(description):
    words = description.split()
    vector = np.mean([word2vec_model.wv[word] for word in words if word in word2vec_model.wv], axis=0)
    if isinstance(vector, np.ndarray):
        return vector
    else:
        return np.zeros(word2vec_model.vector_size)

# Apply the function to the 'Description' column
df['Description_w2v'] = df['Description'].apply(get_word2vec_vector)
print("Word2Vec vectors generated for 'Description' column.")

# Combine Word2Vec vectors with other features
word2vec_features = pd.DataFrame(df['Description_w2v'].tolist(), index=df.index)
word2vec_features.columns = word2vec_features.columns.astype(str)
df = pd.concat([df, word2vec_features], axis=1)
print("Word2Vec vectors combined with other features.")

# Define features and target
word2vec_columns = list(word2vec_features.columns)
features_with_w2v = ['amount', 'recipient', 'expense_account'] + word2vec_columns
target = 'toAccount'

# Preprocess categorical features using one-hot encoding
categorical_features = ['recipient', 'expense_account']
numeric_features = ['amount'] + word2vec_columns

preprocessor_with_w2v = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='mean'), numeric_features),
        ('cat', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ]), categorical_features)
    ])
print("Preprocessor defined with one-hot encoding and imputation.")

# Split the data into training and testing sets
X_with_w2v = df[features_with_w2v]
y = df[target]

# Identify rows with NaNs in the target
nan_rows = y.isna()

# Use rows with NaNs in the target for prediction
X_to_predict_with_w2v = X_with_w2v[nan_rows]
X_with_w2v = X_with_w2v[~nan_rows]
y = y[~nan_rows]
print("Data split into training and prediction sets.")

# Print initial data shapes
print(f"Initial X_with_w2v shape: {X_with_w2v.shape}")
print(f"Initial y shape: {y.shape}")
print(f"Initial X_to_predict_with_w2v shape: {X_to_predict_with_w2v.shape}")

# Split the data into training and testing sets
X_train_with_w2v, X_test_with_w2v, y_train, y_test = train_test_split(X_with_w2v, y, test_size=0.2, random_state=42)
print("Data split into training and testing sets.")

# Check for NaNs
print("NaNs in X_train_with_w2v:", X_train_with_w2v.isnull().sum())
print("NaNs in X_test_with_w2v:", X_test_with_w2v.isnull().sum())
print("NaNs in y_train:", y_train.isnull().sum())
print("NaNs in y_test:", y_test.isnull().sum())

# Print data splits
print(f"X_train_with_w2v shape: {X_train_with_w2v.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_test_with_w2v shape: {X_test_with_w2v.shape}")
print(f"y_test shape: {y_test.shape}")

# Create a pipeline that includes preprocessing and the model
pipeline_with_w2v = Pipeline(steps=[('preprocessor', preprocessor_with_w2v),
                                    ('classifier', GradientBoostingClassifier(n_estimators=100, random_state=42))])
print("Pipeline created with preprocessing and Gradient Boosting Classifier.")

# Train the model
print("Training the model with Word2Vec...")
pipeline_with_w2v.fit(X_train_with_w2v, y_train)
print("Model trained.")

# Make predictions
y_pred_with_w2v = pipeline_with_w2v.predict(X_test_with_w2v)
print("Predictions made.")

# Evaluate the model
print("Evaluating the model with Word2Vec...")
accuracy_with_w2v = accuracy_score(y_test, y_pred_with_w2v)
print(f"Accuracy with Word2Vec: {accuracy_with_w2v}")
print(classification_report(y_test, y_pred_with_w2v, zero_division=0))

# Function to predict toAccount
def predict_toAccount(data, model_pipeline):
    return model_pipeline.predict(data)

# Predicting with Word2Vec
print("Predicting with Word2Vec...")
df['predicted_toAccount_with_w2v'] = predict_toAccount(X_with_w2v, pipeline_with_w2v)

# Use the model to predict 'toAccount' for the rows with NaNs in the target
if not X_to_predict_with_w2v.empty:
    df.loc[nan_rows, 'predicted_toAccount_with_w2v'] = predict_toAccount(X_to_predict_with_w2v, pipeline_with_w2v)

# Save the updated dataframe to a new CSV file
df.to_csv('predicted_email_expense_transactions_with_w2v.csv', index=False)

print("Prediction process complete. Results saved to 'predicted_email_expense_transactions_with_w2v.csv'.")
