import yaml
import logging
import imaplib
import email
from email.header import decode_header
import pandas as pd
import re
import os

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
            print(f"Fetching email ID: {email_id}")
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
        print("Transactions DataFrame created with blank Description column.")
        return df

    except Exception as e:
        logging.error("Failed to get expense emails: {}".format(e))
        raise

def parse_transaction_details(subject, body):
    try:
        if 'You have done a UPI txn. Check details!' in subject:
            # Case 1: HDFC Savings Account
            pattern = r'Rs\.(\d+\.\d{2}) has been debited from account \*\*(\d{4}) to VPA ([\w\.\@]+) on (\d{2}-\d{2}-\d{2})'
            match = re.search(pattern, body)
            if match:
                amount, account_last4, recipient, date = match.groups()
                date = pd.to_datetime(date, format='%d-%m-%y').strftime('%Y-%m-%d')  # Parse and format date
                print("HDFC Savings Account details extracted:", amount, account_last4, recipient, date)
                return {
                    'date': date,
                    'amount': float(amount),
                    'recipient': recipient,
                    'account_last4': account_last4,
                    'type': 'HDFC Savings Account',
                    'expense_account': 'Assets:Banking:HDFC'
                }
        elif 'Update on your HDFC Bank Credit Card' in subject:
            print ("HDFC Credit Card processing...")
            # Case 2: HDFC Credit Card
            if '6815' in body:
                pattern = r'HDFC Bank Credit Card ending (\d{4}) for Rs (\d+\.\d{2}) at ([\w\.\-]+) on (\d{2}-\d{2}-\d{4})'
                match = re.search(pattern, body)
                if match:
                    card_last4, amount, recipient, date = match.groups()
                    print (date)
                    formatted_date = pd.to_datetime(date, format="%d-%m-%Y").strftime("%Y-%m-%d")
                    print (formatted_date)
                    print("HDFC Credit Card details extracted:", amount, card_last4, recipient, formatted_date)
                    return {
                        'date': formatted_date,
                        'amount': float(amount),
                        'recipient': recipient,
                        'account_last4': card_last4,
                        'type': 'Liabilities Credit HDFCMoneyBack',
                        'expense_account': 'Liabilities:Credit Cards:HDFC'
                    }
        elif 'Alert : Transaction alert for your ICICI Bank Credit Card' in subject:
            # Case 3: ICICI Credit Card
            if '8004' in body:
                pattern = r'ICICI Bank Credit Card XX(\d{4}) has been used for a transaction of INR (\d+\.\d{2}) on (\d{2}-\d{2}-\d{4} \d{2}:\d{2}:\d{2}) at ([\w\s\.]+)'
                match = re.search(pattern, body)
                if match:
                    card_last4, amount, date, recipient = match.groups()
                    date = pd.to_datetime(date, format='%d-%m-%y').strftime('%Y-%m-%d')  # Parse and format date
                    print("ICICI Credit Card details extracted:", amount, card_last4, recipient, date)
                    return {
                        'date': date,
                        'amount': float(amount),
                        'recipient': recipient.strip(),
                        'account_last4': card_last4,
                        'type': 'Liabilities Credit ICICI',
                        'expense_account': 'Liabilities:Credit Cards:ICICI'
                    }
        else:
            print("No matching transaction details for subject:", subject)
    except Exception as e:
        logging.error("Failed to parse transaction details: {}".format(e))
        raise
    return None

def merge_with_existing_csv(df, csv_file):
    if os.path.exists(csv_file):
        print(f"CSV file '{csv_file}' exists. Cleaning and merging with new data.")
        existing_df = pd.read_csv(csv_file)

        # Ensure the 'Description' column is present in the existing CSV
        if 'Description' not in existing_df.columns:
            existing_df['Description'] = ''

        # Convert 'Description' to string to avoid AttributeError on NaN values
        existing_df['Description'] = existing_df['Description'].astype(str)

        # Step 1: Clean existing data by removing duplicates, giving precedence to non-empty descriptions
        existing_df['has_description'] = existing_df['Description'].apply(lambda x: bool(str(x).strip()))
        existing_df = existing_df.sort_values(by=['has_description', 'timestamp'], ascending=[False, True])
        existing_df = existing_df.drop_duplicates(subset=['date', 'amount', 'recipient', 'account_last4'], keep='first')
        # existing_df = existing_df.drop(columns=['has_description'])

        # # Convert new data 'Description' column to string
        df['Description'] = df['Description'].astype(str)

        # # Step 2: Concatenate the cleaned existing DataFrame with the new DataFrame
        combined_df = pd.concat([existing_df, df])
        combined_df = combined_df.reset_index(drop=True)

        # # Step 3: Remove duplicates again, giving precedence to non-empty descriptions
        combined_df['has_description'] = combined_df['Description'].apply(lambda x: bool(str(x).strip()))
        combined_df = combined_df.sort_values(by=['has_description', 'timestamp'], ascending=[False, True])
        combined_df = combined_df.drop_duplicates(subset=['date', 'amount', 'recipient', 'timestamp'], keep='first')
        combined_df = combined_df.drop(columns=['has_description'])
        # print (combined_df)
        # Sort by date to keep it organized
        combined_df = combined_df.sort_values(by=['date', 'timestamp'], ascending=[False, True]).reset_index(drop=True)
        print("Data cleaned and merged successfully.")
    else:
        print(f"CSV file '{csv_file}' does not exist. Creating new file.")
        combined_df = df

    return combined_df


def main():
    credentials = load_credentials('credentials.yaml')
    mail = connect_to_gmail_imap(*credentials)
    # Use the label filter with the SINCE filter
    label = 'Finances/Expenses'
    since_date = '13-Jun-2024'
    df = get_expense_emails(mail, label, since_date)

    if not df.empty:
        csv_file = 'email_expense_transactions.csv'
        merged_df = merge_with_existing_csv(df, csv_file)
        merged_df.to_csv(csv_file, index=False)
        print(f"Transactions saved to '{csv_file}'")
    else:
        print("No transactions found.")

if __name__ == "__main__":
    main()
