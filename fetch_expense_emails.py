import yaml
import logging
import imaplib
import email
from email.header import decode_header
import pandas as pd
import re
import os
from datetime import datetime, timedelta

def load_credentials(filepath):
    try:
        with open(filepath, 'r') as file:
            credentials = yaml.safe_load(file)
            user = credentials['user']
            password = credentials['password']
            print("Credentials loaded:", user)
            return user, password
    except Exception as e:
        logging.error("Failed to load credentials: {}".format(e))
        raise

def connect_to_gmail_imap(user, password):
    imap_url = 'imap.gmail.com'
    try:
        mail = imaplib.IMAP4_SSL(imap_url)
        mail.login(user, password)
        mail.select('inbox')  # Connect to the inbox.
        print("Connected to Gmail IMAP server.")
        return mail
    except Exception as e:
        logging.error("Connection failed: {}".format(e))
        raise

def get_expense_emails(mail, label, since_date):
    try:
        # Select the specified label
        mail.select(label)
        print(f"Selected label: {label}")

        # Search for emails matching the query and after the since_date
        search_query = f'(SINCE {since_date})'
        status, messages = mail.search(None, search_query)
        print(f"Search status: {status}")

        email_ids = messages[0].split()
        print(f"Found {len(email_ids)} emails matching the query.")

        transactions = []

        for email_id in email_ids:
            status, msg_data = mail.fetch(email_id, '(RFC822)')
            print(f"\n Fetching email ID: {email_id}")
            for response_part in msg_data:
                if isinstance(response_part, tuple):
                    msg = email.message_from_bytes(response_part[1])
                    subject = decode_header(msg['Subject'])[0][0]
                    if isinstance(subject, bytes):
                        subject = subject.decode()
                    print(f"Email Subject: {subject}")

                    if msg.is_multipart():
                        for part in msg.walk():
                            content_type = part.get_content_type()
                            content_disposition = str(part.get("Content-Disposition"))
                            if "attachment" not in content_disposition:
                                try:
                                    body = part.get_payload(decode=True).decode()
                                    print("Email body fetched and decoded.")
                                except AttributeError as e:
                                    print (content_disposition)
                                    print(f"Error decoding email body part: {e}")
                                    continue
                                break
                    else:
                        try:
                            body = msg.get_payload(decode=True).decode()
                            print("Single-part email body fetched and decoded.")
                        except AttributeError as e:
                            print(f"Error decoding single-part email body: {e}")
                            continue

                    # Parse the email body to extract transaction details
                    transaction = parse_transaction_details(subject, body)
                    if transaction:
                        # Add timestamp from email
                        transaction['timestamp'] = msg['Date']
                        transactions.append(transaction)
                        print("Transaction added:", transaction)

        # Convert the list of transactions into a DataFrame
        df = pd.DataFrame(transactions)
        df['Description'] = ''  # Add a blank Description column
        df['toAccount'] = '' # Add a blank toAccount column
        
        # Convert timestamp column to datetime explicitly
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        print("Transactions DataFrame created with blank Description column.")
        return df

    except Exception as e:
        logging.error("Failed to get expense emails: {}".format(e))
        raise

# Define patterns and corresponding transaction types and accounts
PATTERNS = {
    'HDFC Savings Account': {
        'pattern': r'Rs\.(\d+\.\d{2}) has been debited from account \*\*(\d{4}) to VPA ([\w\.\-]+@[\w\.\-]+) on (\d{2}-\d{2}-\d{2})',
        'expense_account': 'Assets:Banking:HDFC'
    },
    'HDFC Savings Account New Format': {
        'pattern': r'Rs\.(\d+\.\d{2}) has been debited from account \*\*(\d{4}) to VPA ([\w\.\-]+@[\w\.\-]+) ([\w\s]+) on (\d{2}-\d{2}-\d{2})',
        'expense_account': 'Assets:Banking:HDFC'
    },
    'Liabilities Credit HDFCMoneyBack': {
        'pattern': r'HDFC Bank Credit Card ending (\d{4}) for Rs (\d+\.\d{2}) at ([\w\.\-]+) on (\d{2}-\d{2}-\d{4} \d{2}:\d{2}:\d{2})',
        'expense_account': 'Liabilities:Credit:HDFCMoneyBack'
    },
    'Liabilities Credit HDFCMoneyBack SmartPay': {
        'pattern': r'Your ([\w\.\-] +) of Rs. (\d+\.\d{2}) for',
        'expense_account': 'Liabilities:Credit:HDFCMoneyBack'
    },
    'Liabilities Credit ICICI': {
        'pattern': r'ICICI Bank Credit Card XX(\d{4}) has been used for a transaction of INR (\d+\.\d{2}) on (\w+ \d{2}, \d{4} at \d{2}:\d{2}:\d{2})\. Info: ([\w\s\.]+)',
        'expense_account': 'Liabilities:Credit:ICICIAmazonPay'
    },
    'Liabilities Credit ICICI': {
        'pattern': r'ICICI Bank Credit Card XX(\d{4}) has been used for a transaction of INR (\d+\,\d+\.\d{2}) on (\w+ \d{2}, \d{4} at \d{2}:\d{2}:\d{2})\. Info: ([\w\s\.]+)',
        'expense_account': 'Liabilities:Credit:ICICIAmazonPay'
    },
    'SBI Debit Card': {
        'pattern': r'Your A/C \w+(\d{4}) has a debit by transfer of Rs (\d+,\d+\.\d{2}) on (\d{2}/\d{2}/\d{2})',
        'expense_account': 'Assets:Banking:SBI'
    },
    'SBI NACH': {
        'pattern': r'Your A/C \w+(\d{4}) has a debit by NACH of Rs (\d+,\d+,\d+\.\d{2}) on (\d{2}/\d{2}/\d{2})',
        'expense_account': 'Assets:Banking:SBI'
    },
    'HDFC Debit Card': {
        'pattern': r'HDFC Bank Debit Card ending (\d{4}) for Rs (\d+\.\d{2}) at ([\w\s\.]+) on (\d{2}-\d{2}-\d{4} \d{2}:\d{2}:\d{2})',
        'expense_account': 'Assets:Banking:HDFC'
    }
}

def parse_transaction_details(subject, body):
    try:
        for transaction_type, details in PATTERNS.items():
            pattern = details['pattern']
            expense_account = details['expense_account']
            
            match = re.search(pattern, body)
            if match:
                groups = match.groups()
                
                if transaction_type == 'HDFC Savings Account':
                    amount, account_last4, recipient, date = groups
                    date = pd.to_datetime(date, format='%d-%m-%y').strftime('%Y-%m-%d')
                elif transaction_type == 'HDFC Savings Account New Format':
                    amount, account_last4, recipient, description, date = groups
                    date = pd.to_datetime(date, format='%d-%m-%y').strftime('%Y-%m-%d')
                elif transaction_type == 'Liabilities Credit HDFCMoneyBack':
                    account_last4, amount, recipient, datetime_str = groups
                    date = pd.to_datetime(datetime_str, format='%d-%m-%Y %H:%M:%S').strftime('%Y-%m-%d')
                elif transaction_type == 'Liabilities Credit ICICI':
                    account_last4, amount, datetime_str, recipient = groups
                    date = pd.to_datetime(datetime_str, format='%b %d, %Y at %H:%M:%S').strftime('%Y-%m-%d')
                    amount = float(amount.replace(',', ''))
                elif transaction_type == 'SBI Debit Card':
                    account_last4, amount, date = groups
                    date = pd.to_datetime(date, format='%d/%m/%y').strftime('%Y-%m-%d')
                    amount = float(amount.replace(',', ''))
                    recipient = "Transfer"
                elif transaction_type == 'SBI NACH':
                    account_last4, amount, date = groups
                    date = pd.to_datetime(date, format='%d/%m/%y').strftime('%Y-%m-%d')
                    amount = float(amount.replace(',', ''))
                    recipient = "NACH Debit"
                elif transaction_type == 'HDFC Debit Card':
                    account_last4, amount, recipient, datetime_str = groups
                    date = pd.to_datetime(datetime_str, format='%d-%m-%Y %H:%M:%S').strftime('%Y-%m-%d')
                elif transaction_type == 'Liabilities Credit HDFCMoneyBack SmartPay':
                    recipient, amount = groups
                    date = pd.to_datetime('today').strftime('%Y-%m-%d')
                    amount = float(amount)
                    account_last4 = 'XXXX'

                return {
                    'date': date,
                    'amount': float(amount),
                    'recipient': recipient.strip(),
                    'account_last4': account_last4,
                    'type': transaction_type,
                    'expense_account': expense_account
                }
        
        print("No matching transaction details for subject:", subject)
    except Exception as e:
        logging.error("Failed to parse transaction details: {}".format(e))
        raise

    return None

import os
import pandas as pd

def clean_and_normalize_data(df):
    """Normalize and clean the input DataFrame."""
    # Ensure all relevant columns exist
    required_columns = ['date', 'amount', 'recipient', 'timestamp', 'Description']
    for col in required_columns:
        if col not in df.columns:
            df[col] = None

    # Fill missing descriptions with empty strings
    df['Description'] = df['Description'].fillna('')

    # Normalize 'recipient' by stripping and converting to lowercase
    df['recipient'] = df['recipient'].str.strip().str.lower()

    # Convert 'date' and 'timestamp' columns to datetime
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce', utc=True)  # Updated with utc=True

    # Convert 'amount' to numeric
    df['amount'] = pd.to_numeric(df['amount'], errors='coerce')

    return df



def prioritize_descriptions_and_deduplicate(df):
    """Remove duplicates, prioritizing rows with descriptions."""
    # Add a helper column to prioritize rows with non-empty descriptions
    df['has_description'] = df['Description'].apply(lambda x: bool(str(x).strip()))

    # Sort to prioritize rows with descriptions and earlier timestamps
    df = df.sort_values(by=['has_description', 'timestamp'], ascending=[False, True])

    # Drop duplicates based on key columns
    df = df.drop_duplicates(subset=['date', 'amount', 'recipient', 'timestamp'], keep='first')

    # Drop the helper column
    df = df.drop(columns=['has_description'])
    return df


def merge_with_existing_csv(df, csv_file):
    """Merge a new DataFrame with an existing CSV file."""
    # Normalize and clean the incoming DataFrame
    df = clean_and_normalize_data(df)

    if os.path.exists(csv_file):
        print(f"CSV file '{csv_file}' exists. Merging with new data.")
        # Load and clean the existing CSV
        existing_df = pd.read_csv(csv_file)
        existing_df = clean_and_normalize_data(existing_df)

        # Combine existing and new data
        combined_df = pd.concat([existing_df, df], ignore_index=True)
    else:
        print(f"CSV file '{csv_file}' does not exist. Creating new file.")
        combined_df = df

    # Deduplicate and prioritize descriptions
    combined_df = prioritize_descriptions_and_deduplicate(combined_df)

    # Sort for consistency
    combined_df = combined_df.sort_values(by=['date', 'timestamp'], ascending=[False, False]).reset_index(drop=True)

    return combined_df


def update_transactions_csv():
    credentials = load_credentials('credentials.yaml')
    mail = connect_to_gmail_imap(*credentials)
    # Use the label filter with the SINCE filter
    label = 'Finances/Expenses'
    one_week_ago = datetime.now() - timedelta(days=4)

    # Format the date as 'DD-MMM-YYYY'
    since_date = one_week_ago.strftime('%d-%b-%Y')
    df = get_expense_emails(mail, label, since_date)

    if not df.empty:
        csv_file = 'email_expense_transactions.csv'
        merged_df = merge_with_existing_csv(df, csv_file)
        merged_df.to_csv(csv_file, index=False)
        print(f"Transactions saved to '{csv_file}'")
    else:
        print("No transactions found.")

if __name__ == "__main__":
    update_transactions_csv()
