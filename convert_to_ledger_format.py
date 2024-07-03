import pandas as pd

def convert_csv_to_ledger(csv_file, output_file):
    # Read the CSV file
    df = pd.read_csv(csv_file)

    df = df.sort_values(by='date', ascending=True)

    # Initialize an empty list to hold the formatted ledger entries
    ledger_entries = []

    for index, row in df.iterrows():
        # Parse and format the date
        date = pd.to_datetime(row['date']).strftime('%Y/%m/%d')

        # Extract the Description
        description = row['Description']

        # Extract the expense_account and amount, format the amount with the rupee symbol
        expense_account = row['expense_account']
        amount = f"₹{row['amount']:,.2f}"

        # Extract the toAccount
        to_account = row['toAccount']

        # Format each transaction in the Ledger format
        ledger_entry = f"{date} {description}\n"
        ledger_entry += f"    {to_account}    {amount}\n"
        ledger_entry += f"    {expense_account}"

        # Append the formatted entry to the list
        ledger_entries.append(ledger_entry)

    # Write the formatted ledger entries to the output file
    with open(output_file, 'w') as file:
        for entry in ledger_entries:
            file.write(entry)
            file.write("\n")  # Adds a blank line for separation between entries
            file.write("\n")  # Adds a blank line for separation between entries

if __name__ == "__main__":
    # The CSV file to be converted
    csv_file = 'email_expense_transactions.csv'
    # The output file where ledger entries will be saved
    output_file = 'ledger_entries_obsidian.txt'
    convert_csv_to_ledger(csv_file, output_file)

    print(f"Ledger entries have been saved to {output_file}")
