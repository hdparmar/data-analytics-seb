"""
SEB Bank Statement Data Loader
------------------------------
Handles loading and initial processing of SEB bank statement CSV files.
"""
import pandas as pd
import os
from datetime import datetime
import re

class SEBDataLoader:
    def __init__(self, file_path):
        """Initialize with the path to the CSV file."""
        self.file_path = file_path
        self.data = None
        
    def load_data(self):
        """Load the CSV data with proper encoding and delimiter."""
        try:
            # First, peek at the file to see what we're dealing with
            with open(self.file_path, 'r', encoding='utf-8') as f:
                first_lines = [next(f) for _ in range(5) if _ < 5]
            
            print("First few lines of the CSV:")
            for line in first_lines:
                print(f"  {line.strip()}")
            
            # See if we can detect the delimiter
            delimiters = [';', ',', '\t']
            detected_delimiter = None
            for delimiter in delimiters:
                if delimiter in first_lines[0]:
                    detected_delimiter = delimiter
                    break
            
            if detected_delimiter:
                print(f"Detected delimiter: '{detected_delimiter}'")
            else:
                print("Could not detect delimiter, defaulting to ';'")
                detected_delimiter = ';'
            
            
            self.data = pd.read_csv(
                self.file_path, 
                delimiter=detected_delimiter, 
                encoding='utf-8',
                header=0,  # Assume headers are present
                error_bad_lines=False  # Skip lines with too many fields
            )
            
            # Check what columns we have
            print(f"Columns found: {list(self.data.columns)}")
            
            # The column name might contain spaces or be a single string with semicolons
            # Let's check if we have a single column and need to split it
            if len(self.data.columns) == 1 and detected_delimiter in self.data.columns[0]:
                print("Found a single column that needs splitting. Reloading...")
                column_names = self.data.columns[0].split(detected_delimiter)
                
                # Reload with proper column names
                self.data = pd.read_csv(
                    self.file_path, 
                    delimiter=detected_delimiter, 
                    encoding='utf-8',
                    header=0,
                    names=column_names,
                    error_bad_lines=False
                )
                print(f"Columns after splitting: {list(self.data.columns)}")
            
            # Try to convert date columns if they exist
            date_columns = [col for col in self.data.columns if 'date' in str(col).lower() or 'dag' in str(col).lower()]
            for col in date_columns:
                try:
                    self.data[col] = pd.to_datetime(self.data[col])
                    print(f"Converted {col} to datetime")
                except Exception as e:
                    print(f"Could not convert {col} to datetime: {str(e)}")
            
            print(f"Successfully loaded {len(self.data)} transactions.")
            return self.data
            
        except Exception as e:
            print(f"Error loading CSV with UTF-8 encoding: {str(e)}")
            # Try alternative approach with different encoding
            try:
                print("Trying with latin-1 encoding...")
                self.data = pd.read_csv(
                    self.file_path,
                    delimiter=detected_delimiter if 'detected_delimiter' in locals() else ';',
                    encoding='latin-1',
                    error_bad_lines=False
                )
                print(f"Successfully loaded with latin-1 encoding: {len(self.data)} transactions.")
                return self.data
            except Exception as e2:
                print(f"Second attempt failed: {str(e2)}")
                
                # Last resort: try to read it more manually
                try:
                    print("Trying manual approach...")
                    # Read raw lines
                    with open(self.file_path, 'r', encoding='latin-1') as f:
                        lines = f.readlines()
                    
                    # Try to determine structure
                    if len(lines) > 1:
                        header = lines[0].strip().split(';')
                        data_rows = []
                        
                        for line in lines[1:]:
                            values = line.strip().split(';')
                            # Ensure each row has the right number of fields
                            if len(values) == len(header):
                                data_rows.append(values)
                        
                        # Create DataFrame
                        self.data = pd.DataFrame(data_rows, columns=header)
                        print(f"Manually loaded {len(self.data)} transactions.")
                        return self.data
                    else:
                        raise ValueError("File appears to be empty or has only one line")
                    
                except Exception as e3:
                    print(f"All attempts failed. Last error: {str(e3)}")
                    raise
    
    def clean_data(self):
        """Basic cleaning of the data."""
        if self.data is None:
            self.load_data()
        
        print("Column types before conversion:")
        print(self.data.dtypes)
            
        # Convert amount to numeric
        amount_cols = [col for col in self.data.columns if 'amount' in str(col).lower() or 'belopp' in str(col).lower()]
        for col in amount_cols:
            try:
                # Check if column is not already numeric
                if not pd.api.types.is_numeric_dtype(self.data[col]):
                    print(f"Converting {col} to numeric...")
                    # First convert to string in case it's not
                    self.data[col] = self.data[col].astype(str)
                    # Replace comma with dot for decimal separator
                    self.data[col] = self.data[col].str.replace(',', '.').str.replace(' ', '')
                    # Convert to float
                    self.data[col] = pd.to_numeric(self.data[col], errors='coerce')
                    print(f"Converted {col} successfully")
            except Exception as e:
                print(f"Error converting {col}: {str(e)}")
                
        # Handle balance column
        balance_cols = [col for col in self.data.columns if 'balance' in str(col).lower() or 'saldo' in str(col).lower()]
        for col in balance_cols:
            try:
                # Check if column is not already numeric
                if not pd.api.types.is_numeric_dtype(self.data[col]):
                    print(f"Converting {col} to numeric...")
                    # First convert to string in case it's not
                    self.data[col] = self.data[col].astype(str)
                    # Replace comma with dot for decimal separator
                    self.data[col] = self.data[col].str.replace(',', '.').str.replace(' ', '')
                    # Convert to float
                    self.data[col] = pd.to_numeric(self.data[col], errors='coerce')
                    print(f"Converted {col} successfully")
            except Exception as e:
                print(f"Error converting {col}: {str(e)}")
                
        # Remove any trailing/leading whitespace in text fields
        text_cols = [col for col in self.data.columns if 'text' in str(col).lower() or 'beskrivning' in str(col).lower()]
        for col in text_cols:
            try:
                if pd.api.types.is_object_dtype(self.data[col]):
                    self.data[col] = self.data[col].astype(str).str.strip()
            except Exception as e:
                print(f"Error cleaning text in {col}: {str(e)}")
        
        print("Column types after conversion:")
        print(self.data.dtypes)
            
        return self.data
    
    def get_summary_stats(self):
        """Get summary statistics about the transactions."""
        if self.data is None or len(self.data) == 0:
            raise ValueError("Data must be loaded and cleaned first")
        
        # Try to identify amount column
        amount_cols = [col for col in self.data.columns if 'amount' in str(col).lower() or 'belopp' in str(col).lower()]
        amount_col = amount_cols[0] if amount_cols else None
        
        # Try to identify date column
        date_cols = [col for col in self.data.columns if 'date' in str(col).lower() or 'dag' in str(col).lower()]
        date_col = None
        for col in date_cols:
            if pd.api.types.is_datetime64_dtype(self.data[col]):
                date_col = col
                break
        
        # Prepare summary
        summary = {
            "total_transactions": len(self.data)
        }
        
        # Add date range if available
        if date_col:
            summary["date_range"] = (
                self.data[date_col].min().strftime('%Y-%m-%d'), 
                self.data[date_col].max().strftime('%Y-%m-%d')
            )
        
        # Add amount statistics if available
        if amount_col:
            income = self.data[self.data[amount_col] > 0][amount_col].sum()
            expenses = self.data[self.data[amount_col] < 0][amount_col].sum()
            
            summary.update({
                "total_income": income,
                "total_expenses": expenses,
                "net_cashflow": income + expenses,
                "average_transaction": self.data[amount_col].mean()
            })
        
        return summary
    
    def extract_basic_categories(self):
        """Extract basic transaction categories based on text patterns."""
        if self.data is None:
            raise ValueError("Data must be loaded first")
        
        # Try to identify text column
        text_cols = [col for col in self.data.columns if 'text' in str(col).lower() or 'beskrivning' in str(col).lower()]
        text_col = text_cols[0] if text_cols else None
        
        if not text_col:
            print("Could not identify text column for categorization")
            self.data['Category'] = 'Unknown'
            return self.data
            
        # Create a new column for categories
        self.data['Category'] = 'Other'
        
        # Define some basic category patterns - adjusted for SEB
        category_patterns = {
            'Salary': [r'lön', r'salary', r'payroll', r'arbetsgivare'],
            'Groceries': [r'ica', r'coop', r'willys', r'hemköp', r'lidl', r'netto', r'mat', r'livs'],
            'Restaurants': [r'restaurant', r'restaurang', r'pizzeria', r'burger', r'mcdonalds', r'max'],
            'Transport': [r'sl\b', r'sj\b', r'uber', r'taxi', r'parkering', r'bensin', r'buss', r'reskassa'],
            'Shopping': [r'h\s*&\s*m', r'zara', r'lindex', r'åhlens', r'ikea', r'elgiganten', r'media\s*markt'],
            'Entertainment': [r'bio', r'cinema', r'spotify', r'netflix', r'hbo', r'sf\s'],
            'Rent': [r'hyra', r'rent', r'bostad'],
            'Utilities': [r'el\b', r'electricity', r'vatten', r'water', r'internet', r'telia', r'telenor'],
            'Insurance': [r'försäkring', r'insurance', r'trygg', r'folksam'],
            'Medical': [r'apotek', r'pharmacy', r'läkare', r'vårdcentral', r'tandläkare']
        }
        
        # Apply the patterns to categorize transactions
        for category, patterns in category_patterns.items():
            pattern = '|'.join(patterns)
            mask = self.data[text_col].str.lower().str.contains(pattern, na=False, regex=True)
            self.data.loc[mask, 'Category'] = category
            
        # Additional logic for transfers and withdrawals - common in Swedish banking
        self.data.loc[self.data[text_col].str.contains('Överföring|Overforing', case=False, na=False), 'Category'] = 'Transfer'
        self.data.loc[self.data[text_col].str.contains('Uttag|Uttagsautomat', case=False, na=False), 'Category'] = 'Withdrawal'
        
        # Print category distribution
        print("\nCategory distribution:")
        print(self.data['Category'].value_counts())
        
        return self.data
        
    def save_processed_data(self, output_path):
        """Save the processed data to a new CSV file."""
        if self.data is not None:
            output_dir = os.path.dirname(output_path)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                
            self.data.to_csv(output_path, index=False, encoding='utf-8')
            print(f"Processed data saved to {output_path}")
        else:
            print("No data to save. Please load and process data first.")