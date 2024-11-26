import streamlit as st
import pandas as pd
from datetime import datetime
import pytz
from email.utils import parsedate_to_datetime
from fetch_expense_emails import update_transactions_csv
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
    
    # # Convert timestamp column to datetime explicitly
    # df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    
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

    with st.expander("View remaining blank transactions"):
        st.write(blank_rows)

    if 'current_index' not in st.session_state:
        st.session_state.current_index = blank_rows.index[0]
    current_transaction = df.loc[st.session_state.current_index]

    # Create two columns for the layout
    col1, col2, col3 = st.columns([1, 2, 1])

    with col1:
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
            # Prefill with last known values
            last_known_description = same_recipient.iloc[0]['Description']
            last_known_to_account = same_recipient.iloc[0]['toAccount']
        else:
            st.write("No previous transactions found with this recipient")
            last_known_description = ""
            last_known_to_account = ""

    with col2:
        
        # Show timestamp in readable format
        st.write("Transaction Time:")
        if pd.api.types.is_numeric_dtype(current_transaction['timestamp']):
            timestamp = pd.to_datetime(current_transaction['timestamp'], unit='s')
        else:
            timestamp = pd.to_datetime(current_transaction['timestamp'])
        st.write(format_datetime(parse_timestamp(timestamp)))

        # Text input for Description, prefilled with last known value
        new_description = st.text_input("Edit Description", last_known_description)

        # Select box for toAccount, prefilled with last known value
        unique_accounts = df['toAccount'].dropna().value_counts().index.tolist()
        new_to_account = st.selectbox("Select toAccount", unique_accounts, index=unique_accounts.index(last_known_to_account) if last_known_to_account in unique_accounts else 0)
        text_to_account = st.text_input("New toAccount", value=new_to_account)

        if text_to_account != new_to_account:
            new_to_account = text_to_account

        # Add navigation buttons in a new row
        nav_col1, nav_col2, nav_col3 = st.columns([1, 1, 1])
        
        # Store current index in session state if not already present
        if 'current_index' not in st.session_state:
            st.session_state.current_index = blank_rows.index[0]
        
        with nav_col1:
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
                
                # Move to next blank transaction
                current_idx = blank_rows.index.get_loc(st.session_state.current_index)
                if current_idx < len(blank_rows) - 1:
                    st.session_state.current_index = blank_rows.index[current_idx + 1]
                
                st.success("Transaction updated!")
                st.rerun()

        with nav_col2:
            if st.button("Previous Transaction"):
                current_idx = blank_rows.index.get_loc(st.session_state.current_index)
                if current_idx > 0:
                    st.session_state.current_index = blank_rows.index[current_idx - 1]
                    st.rerun()

        with nav_col3:
            if st.button("Next Transaction"):
                current_idx = blank_rows.index.get_loc(st.session_state.current_index)
                if current_idx < len(blank_rows) - 1:
                    st.session_state.current_index = blank_rows.index[current_idx + 1]
                    st.rerun()


    with col3:
        st.subheader("Current Transaction")
        # Display the current transaction details
        st.write(current_transaction[['date', 'amount', 'recipient', 'Description', 'toAccount', 'timestamp']])


    # Update the current transaction selection based on navigation
    current_transaction = df.loc[st.session_state.current_index]

    # After the save button logic
    st.divider()  # Add a visual separator
    
    # Add date picker and ledger format button
    st.subheader("Generate Ledger Format")
    selected_date = st.date_input(
        "Show transactions after date:",
        value=datetime.now().date(),
        format="YYYY-MM-DD"
    )
    
    if st.button("Generate in Ledger Format"):
        # Filter DataFrame for non-blank descriptions and dates after selected date
        filtered_df = df[
            (df['Description'].notna()) & 
            (df['Description'].str.strip() != '') &
            (df['toAccount'].notna()) & 
            (df['toAccount'].str.strip() != '') &
            (pd.to_datetime(df['date']).dt.date > selected_date)
        ].sort_values(by='date', ascending=True)
        
        # Generate ledger entries
        ledger_entries = []
        for _, row in filtered_df.iterrows():
            date = pd.to_datetime(row['date']).strftime('%Y/%m/%d')
            description = row['Description']
            expense_account = row['expense_account']
            amount = f"₹{row['amount']:,.2f}"
            to_account = row['toAccount']
            
            ledger_entry = f"{date} {description}\n"
            ledger_entry += f"    {to_account}    {amount}\n"
            ledger_entry += f"    {expense_account}"
            ledger_entries.append(ledger_entry)
        
        # Display the ledger entries in a monospace font
        if ledger_entries:
            st.text("Ledger Format Output:")
            st.code('\n\n'.join(ledger_entries), language=None)
        else:
            st.warning("No transactions found after the selected date.")

if __name__ == "__main__":
    if 'transactions_updated' not in st.session_state:
        print("Updating transactions CSV file")
        # Update the transactions CSV file
        update_transactions_csv()
        st.session_state.transactions_updated = True
    main()