import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_test_data(n_samples=100):
    """Generate sample Excel test data for money laundering detection"""
    
    # Set random seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Generate sample data
    data = []
    
    # Sample banks and locations
    banks = ['Bank_A', 'Bank_B', 'Bank_C', 'Bank_D', 'Bank_E']
    locations = ['Jakarta', 'Surabaya', 'Bandung', 'Medan', 'Makassar', 'Singapore', 'Malaysia', 'Thailand']
    currencies = ['IDR', 'USD', 'SGD', 'EUR', 'MYR']
    payment_types = ['Transfer', 'Cash', 'Check', 'Card', 'Online']
    
    start_date = datetime(2023, 1, 1)
    
    for i in range(n_samples):
        # Generate random date and time
        random_days = np.random.randint(0, 365)
        date = start_date + timedelta(days=random_days)
        hour = np.random.randint(0, 24)
        minute = np.random.randint(0, 60)
        second = np.random.randint(0, 60)
        time = f"{hour:02d}:{minute:02d}:{second:02d}"
        
        # Generate account numbers
        sender_account = f"ACC{np.random.randint(100000, 999999)}"
        receiver_account = f"ACC{np.random.randint(100000, 999999)}"
        
        # Generate currencies
        payment_currency = np.random.choice(currencies)
        received_currency = np.random.choice(currencies)
        
        # Generate locations
        sender_location = np.random.choice(locations)
        receiver_location = np.random.choice(locations)
        
        # Generate payment type
        payment_type = np.random.choice(payment_types)
        
        # Generate amount (with some high amounts for potential laundering)
        if np.random.random() < 0.1:  # 10% chance of high amount (suspicious)
            amount = np.random.uniform(100000, 1000000)
            is_laundering = 1 if np.random.random() < 0.7 else 0  # 70% chance of being laundering
        else:
            amount = np.random.uniform(100, 50000)
            is_laundering = 1 if np.random.random() < 0.05 else 0  # 5% chance of being laundering
        
        data.append({
            'Date': date.strftime('%Y-%m-%d'),
            'Time': time,
            'Sender_account': sender_account,
            'Receiver_account': receiver_account,
            'Payment_currency': payment_currency,
            'Received_currency': received_currency,
            'Sender_bank_location': sender_location,
            'Receiver_bank_location': receiver_location,
            'Payment_type': payment_type,
            'Amount': round(amount, 2),
            'Is_laundering': is_laundering
        })
    
    return pd.DataFrame(data)

if __name__ == "__main__":
    # Generate test data
    test_df = generate_test_data(500)
    
    # Save as Excel file
    test_df.to_excel('data/test_data.xlsx', index=False)
    print(f"Generated test data with {len(test_df)} samples")
    print(f"Money laundering cases: {test_df['Is_laundering'].sum()}")
    print(f"Normal cases: {(test_df['Is_laundering'] == 0).sum()}")
    print("\nFirst 5 rows:")
    print(test_df.head())
    
    # Also save as CSV for backup
    test_df.to_csv('data/test_data.csv', index=False)
    print("\nSaved as both Excel and CSV files in data/ directory")
