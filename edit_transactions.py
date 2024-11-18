import streamlit as st
import pandas as pd
from datetime import datetime
import pytz
from email.utils import parsedate_to_datetime
import shutil

st.set_page_config(layout="wide")

def parse_timestamp(ts):
    if pd.isna(ts) or ts == 'time':
        return None
    try:
        # First try parsing as email format
        return parsedate_to_datetime(ts)
    except:
        try:
            # Then try parsing as ISO format
            return pd.to_datetime(ts)
        except:
            return None

def load_transactions(file_path):
    df = pd.read_csv(file_path)
    
    # Convert timestamp column to datetime explicitly
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    
    df.info()
    return df

def format_datetime(timestamp):
    if timestamp is None:
        return "No timestamp available"
    
    # Convert to IST
    ist = pytz.timezone('Asia/Kolkata')
    if timestamp.tzinfo is None:
        timestamp = pytz.utc.localize(timestamp)
    timestamp_ist = timestamp.astimezone(ist)
    return timestamp_ist.strftime("%A, %B %d, %Y at %I:%M %p IST")

def main():
    st.title("Transaction Editor")
    print("Starting the application")
    # Load the CSV file
    df = load_transactions('email_expense_transactions.csv')
    print("Loaded DataFrame:", df.shape)  # Print DataFrame dimensions
    
    with st.expander("View all transactions"):
        st.write(df)
    
    # Find rows with either empty or None Description or toAccount
    blank_rows = df[
        (df['Description'].isna()) |
        (df['Description'].str.strip() == '') |
        (df['toAccount'].isna()) |
        (df['toAccount'].str.strip() == '')
    ].copy()
    print("Blank rows found:", len(blank_rows))  # Print number of blank rows

    if blank_rows.empty:
        st.success("No blank transactions to edit!")
        return

    # Display count of remaining blank transactions
    st.info(f"Remaining blank transactions: {len(blank_rows)}")

    st.write(blank_rows)

    # Get the first blank transaction
    current_transaction = blank_rows.iloc[0]

    # Create two columns for the layout
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Current Transaction")
        # Display the current transaction details
        st.write(current_transaction[['date', 'amount', 'recipient', 'Description', 'toAccount', 'timestamp']])

        # Show timestamp in readable format
        st.write("Transaction Time:")
        if pd.api.types.is_numeric_dtype(current_transaction['timestamp']):
            timestamp = pd.to_datetime(current_transaction['timestamp'], unit='s')
        else:
            timestamp = pd.to_datetime(current_transaction['timestamp'])
        st.write(format_datetime(parse_timestamp(timestamp)))

        # Text input for Description
        new_description = st.text_input("Edit Description", "")

        # Select box for toAccount
        unique_accounts = df['toAccount'].dropna().value_counts().index.tolist()
        new_to_account = st.selectbox("Select toAccount", unique_accounts)
        text_to_account = st.text_input("New toAccount", value=new_to_account)

        if text_to_account != new_to_account:
            new_to_account = text_to_account

    with col2:
        st.subheader("Previous Transactions")
        # Find previous transactions with the same recipient
        same_recipient = df[
            (df['recipient'] == current_transaction['recipient']) &
            (df['Description'].notna()) &  # Only show transactions with descriptions
            (df['Description'].str.strip() != '')
        ].sort_values('date', ascending=False)

        if not same_recipient.empty:
            # Display previous transactions
            st.dataframe(
                same_recipient[['date', 'amount', 'Description', 'toAccount']],
                hide_index=True
            )
        else:
            st.write("No previous transactions found with this recipient")

    # Add a save button
    if st.button("Save Changes"):
        print("Before update - Current transaction:", current_transaction)  # Print current state
        print("Edited values: Description:", new_description, "toAccount:", new_to_account)  # Print edited values
        
        # Update the main DataFrame with edited values
        df.loc[current_transaction.name, 'Description'] = new_description
        df.loc[current_transaction.name, 'toAccount'] = new_to_account
        
        print("After update - Modified row:", df.loc[current_transaction.name])  # Print updated state
        
        # Backup the original CSV
        shutil.copy('email_expense_transactions.csv', 'email_expense_transactions_backup.csv')
        
        # Save back to CSV
        df.to_csv('email_expense_transactions.csv', index=False)
        st.success("Transaction updated! Refresh the page to continue editing.")
        st.rerun()

if __name__ == "__main__":
    main()