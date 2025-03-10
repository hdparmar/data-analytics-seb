"""
Main processor for the SEB banking data
"""
from src.data.loader import SEBDataLoader
import pandas as pd
import os

def main():
    print("*" * 50)
    print("STARTING THE OPERATION, FAMILY BUSINESS TIME")
    print("*" * 50)
    
    # Check if the raw data exists
    if not os.path.exists('data/raw/kontoutdrag.csv'):
        print("Oi bruv, where's the CSV file? Put it in data/raw/kontoutdrag.csv")
        return
    
    # Peek at the raw CSV 
    print("\nPeeking at the raw CSV format...")
    try:
        with open('data/raw/kontoutdrag.csv', 'r', encoding='utf-8') as f:
            first_lines = [next(f) for _ in range(5) if _ < 5]
        print("First 5 lines of CSV:")
        for line in first_lines:
            print(f"  {line.strip()}")
    except Exception as e:
        print(f"Error reading CSV directly: {str(e)}")
        try:
            with open('data/raw/kontoutdrag.csv', 'r', encoding='latin-1') as f:
                first_lines = [next(f) for _ in range(5) if _ < 5]
            print("First 5 lines of CSV (with latin-1 encoding):")
            for line in first_lines:
                print(f"  {line.strip()}")
        except Exception as e2:
            print(f"Error reading CSV with latin-1 encoding: {str(e2)}")
    
    # Initialize the data loader
    loader = SEBDataLoader('data/raw/kontoutdrag.csv')
    
    # Load and clean the data
    print("\nLoading the money data...")
    data = loader.load_data()
    
    if data is None or len(data) == 0:
        print("Failed to load data properly. Check your CSV format.")
        return
    
    print("\nLet's see what we're working with here:")
    print("Column types:")
    print(data.dtypes)
    print("\nFirst row sample:")
    print(data.iloc[0])
    
    print("\nCleaning the books...")
    data = loader.clean_data()
    
    # Basic data exploration
    print("\nTaking a look at what we've got:")
    print(f"Total transactions: {len(data)}")
    print("\nFirst few transactions:")
    print(data.head())
    
    # Get summary statistics
    try:
        print("\nThe big picture:")
        stats = loader.get_summary_stats()
        for key, value in stats.items():
            print(f"  {key}: {value}")
    except Exception as e:
        print(f"Error getting summary stats: {str(e)}")
    
    # Extract basic categories
    try:
        print("\nSorting the transactions into categories...")
        data = loader.extract_basic_categories()
    except Exception as e:
        print(f"Error categorizing transactions: {str(e)}")
        data['Category'] = 'Uncategorized'  # Fallback
    
    # Create the processed directory if it doesn't exist
    os.makedirs('data/processed', exist_ok=True)
    
    # Save the processed data
    processed_path = 'data/processed/transactions_categorized.csv'
    loader.save_processed_data(processed_path)
    
    print(f"\nCategory breakdown:")
    print(data['Category'].value_counts())
    
    print("\nJob done. The processed data is ready for the family/individual at:")
    print(f"  {processed_path}")
    print("*" * 50)

if __name__ == "__main__":
    main()