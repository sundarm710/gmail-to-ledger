# Personal Finance Tracker

A Python-based personal finance management system that processes transactions and generates ledger-formatted entries.

## Features

- Automated transaction processing from email
- Interactive transaction editing with Streamlit interface
- Ledger-format output generation
- CSV to ledger format conversion
- Transaction history tracking

## Installation

1. Clone the repository
2. Install dependencies:
3. pip install -r requirements.txt

## Configuration

1. Create a `credentials.yaml` file with your email credentials (this file is gitignored)
2. Ensure you have proper permissions for Gmail API access

## Usage

### Transaction Editor
Run the Streamlit interface: streamlit run edit_transactions.py


Features:
- View and edit transaction details
- Navigate through blank transactions
- View transaction history
- Generate ledger format output
- Date-based filtering

### CSV to Ledger Converter
Convert CSV transactions to ledger format: python convert_to_ledger_format.py


## File Structure

- `edit_transactions.py`: Main Streamlit interface for transaction editing
- `convert_to_ledger_format.py`: CSV to ledger format converter
- `email_expense_transactions.csv`: Transaction database (gitignored)
- `ledger.txt`: Generated ledger format output (gitignored)

## Output Format

The ledger format follows this structure:
YYYY/MM/DD Description
Account1 ₹Amount
Account2


## Security

Sensitive files are gitignored:
- credentials.yaml
- email_expense_transactions.csv
- email_expense_transactions_backup.csv
- ledger_entries.txt
- ledger_entries_obsidian.txt
- ledger.txt

## Dependencies

Key dependencies include:
- pandas
- streamlit
- google-api-python-client
- google-auth-oauthlib

For complete list, see requirements.txt


## License

MIT License

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request
