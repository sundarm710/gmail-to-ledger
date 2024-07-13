import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
#import streamlit as st

# Load data
data = pd.read_csv('email_expense_transactions.csv')

# Encode categorical variables
label_encoder = LabelEncoder()
data['recipient_encoded'] = label_encoder.fit_transform(data['recipient'])
data['account_last4_encoded'] = label_encoder.fit_transform(data['account_last4'])

# Feature selection
features = ['amount', 'recipient_encoded', 'account_last4_encoded']
X = data[features]
y_description = data['Description']
y_toAccount = data['toAccount']

# Split data
print()
X_train_desc, X_test_desc, y_train_desc, y_test_desc = train_test_split(X, y_description, test_size=0.2, random_state=42)
X_train_acc, X_test_acc, y_train_acc, y_test_acc = train_test_split(X, y_toAccount, test_size=0.2, random_state=42)

# Train models for Description and toAccount
model_desc = RandomForestClassifier()
model_desc.fit(X_train_desc, y_train_desc)

model_acc = RandomForestClassifier()
model_acc.fit(X_train_acc, y_train_acc)


# Function to predict description and account
def predict_expense(amount, recipient, account_last4):
    recipient_encoded = label_encoder.transform([recipient])[0]
    account_last4_encoded = label_encoder.transform([account_last4])[0]
    
    input_data = pd.DataFrame([[amount, recipient_encoded, account_last4_encoded]], columns=features)
    
    predicted_desc = model_desc.predict(input_data)[0]
    predicted_acc = model_acc.predict(input_data)[0]
    
    return predicted_desc, predicted_acc

# Streamlit app
st.title("Expense Recorder")

amount = st.number_input("Amount", min_value=0.0)
recipient = st.text_input("Recipient")
account_last4 = st.text_input("Account Last 4 Digits")

if st.button("Predict"):
    description, to_account = predict_expense(amount, recipient, account_last4)
    st.write(f"Predicted Description: {description}")
    st.write(f"Predicted toAccount: {to_account}")

    confirmed_desc = st.text_input("Confirm or Edit Description", value=description)
    confirmed_acc = st.text_input("Confirm or Edit toAccount", value=to_account)
    
    if st.button("Save"):
        new_data = {
            "date": pd.Timestamp.now().strftime('%Y-%m-%d'),
            "amount": amount,
            "recipient": recipient,
            "account_last4": account_last4,
            "type": "HDFC Savings Account",
            "expense_account": "Assets:Banking:HDFC",
            "timestamp": pd.Timestamp.now().strftime('%a, %d %b %Y %H:%M:%S %z'),
            "Description": confirmed_desc,
            "toAccount": confirmed_acc
        }
        st.write("Expense recorded successfully!", new_data)

# "import pandas as pd
# import gensim
# from gensim.models import Word2Vec
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# from sklearn.impute import SimpleImputer
# from sklearn.ensemble import GradientBoostingClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report, accuracy_score
# import numpy as np

# # Load the data
# df = 

# # Fill missing 'Description' values with an empty string
# df['Description'] = df['Description'].fillna('')
# print("Data loaded and 'Description' column filled with empty strings if missing.")

# # Ensure 'toAccount' is treated as a categorical variable
# df['toAccount'] = df['toAccount'].astype('category')
# print("'toAccount' column converted to categorical.")

# # Train Word2Vec model on the 'Description' column
# descriptions = df['Description'].apply(lambda x: x.split())
# word2vec_model = Word2Vec(sentences=descriptions, vector_size=100, window=5, min_count=1, workers=4)
# print("Word2Vec model trained on 'Description' column.")

# # Function to transform a description to its Word2Vec embedding
# def get_word2vec_vector(description):
#     words = description.split()
#     vector = np.mean([word2vec_model.wv[word] for word in words if word in word2vec_model.wv], axis=0)
#     if isinstance(vector, np.ndarray):
#         return vector
#     else:
#         return np.zeros(word2vec_model.vector_size)

# # Apply the function to the 'Description' column
# df['Description_w2v'] = df['Description'].apply(get_word2vec_vector)
# print("Word2Vec vectors generated for 'Description' column.")

# # Combine Word2Vec vectors with other features
# word2vec_features = pd.DataFrame(df['Description_w2v'].tolist(), index=df.index)
# word2vec_features.columns = word2vec_features.columns.astype(str)
# df = pd.concat([df, word2vec_features], axis=1)
# print("Word2Vec vectors combined with other features.")

# # Define features and target
# word2vec_columns = list(word2vec_features.columns)
# features_with_w2v = ['amount', 'recipient', 'expense_account'] + word2vec_columns
# target = 'toAccount'

# # Preprocess categorical features using one-hot encoding
# categorical_features = ['recipient', 'expense_account']
# numeric_features = ['amount'] + word2vec_columns

# preprocessor_with_w2v = ColumnTransformer(
#     transformers=[
#         ('num', SimpleImputer(strategy='mean'), numeric_features),
#         ('cat', Pipeline(steps=[
#             ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
#             ('onehot', OneHotEncoder(handle_unknown='ignore'))
#         ]), categorical_features)
#     ])
# print("Preprocessor defined with one-hot encoding and imputation.")

# # Split the data into training and testing sets
# X_with_w2v = df[features_with_w2v]
# y = df[target]

# # Identify rows with NaNs in the target
# nan_rows = y.isna()

# # Use rows with NaNs in the target for prediction
# X_to_predict_with_w2v = X_with_w2v[nan_rows]
# X_with_w2v = X_with_w2v[~nan_rows]
# y = y[~nan_rows]
# print("Data split into training and prediction sets.")

# # Print initial data shapes
# print(f"Initial X_with_w2v shape: {X_with_w2v.shape}")
# print(f"Initial y shape: {y.shape}")
# print(f"Initial X_to_predict_with_w2v shape: {X_to_predict_with_w2v.shape}")

# # Split the data into training and testing sets
# X_train_with_w2v, X_test_with_w2v, y_train, y_test = train_test_split(X_with_w2v, y, test_size=0.2, random_state=42)
# print("Data split into training and testing sets.")

# # Check for NaNs
# print("NaNs in X_train_with_w2v:", X_train_with_w2v.isnull().sum())
# print("NaNs in X_test_with_w2v:", X_test_with_w2v.isnull().sum())
# print("NaNs in y_train:", y_train.isnull().sum())
# print("NaNs in y_test:", y_test.isnull().sum())

# # Print data splits
# print(f"X_train_with_w2v shape: {X_train_with_w2v.shape}")
# print(f"y_train shape: {y_train.shape}")
# print(f"X_test_with_w2v shape: {X_test_with_w2v.shape}")
# print(f"y_test shape: {y_test.shape}")

# # Create a pipeline that includes preprocessing and the model
# pipeline_with_w2v = Pipeline(steps=[('preprocessor', preprocessor_with_w2v),
#                                     ('classifier', GradientBoostingClassifier(n_estimators=100, random_state=42))])
# print("Pipeline created with preprocessing and Gradient Boosting Classifier.")

# # Train the model
# print("Training the model with Word2Vec...")
# pipeline_with_w2v.fit(X_train_with_w2v, y_train)
# print("Model trained.")

# # Make predictions
# y_pred_with_w2v = pipeline_with_w2v.predict(X_test_with_w2v)
# print("Predictions made.")

# # Evaluate the model
# print("Evaluating the model with Word2Vec...")
# accuracy_with_w2v = accuracy_score(y_test, y_pred_with_w2v)
# print(f"Accuracy with Word2Vec: {accuracy_with_w2v}")
# print(classification_report(y_test, y_pred_with_w2v, zero_division=0))

# # Function to predict toAccount
# def predict_toAccount(data, model_pipeline):
#     return model_pipeline.predict(data)

# # Predicting with Word2Vec
# print("Predicting with Word2Vec...")
# df['predicted_toAccount_with_w2v'] = predict_toAccount(X_with_w2v, pipeline_with_w2v)

# # Use the model to predict 'toAccount' for the rows with NaNs in the target
# if not X_to_predict_with_w2v.empty:
#     df.loc[nan_rows, 'predicted_toAccount_with_w2v'] = predict_toAccount(X_to_predict_with_w2v, pipeline_with_w2v)

# # Save the updated dataframe to a new CSV file
# df.to_csv('predicted_email_expense_transactions_with_w2v.csv', index=False)

# print("Prediction process complete. Results saved to 'predicted_email_expense_transactions_with_w2v.csv'.")
