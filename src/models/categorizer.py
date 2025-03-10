"""
Advanced Transaction Categorizer
-------------------------------
Uses machine learning to categorize bank transactions with high accuracy.
"""
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os
import re
import string
from datetime import datetime

class TransactionCategorizer:
    def __init__(self, model_path=None):
        """Initialize the categorizer, optionally loading a pre-trained model."""
        self.vectorizer = TfidfVectorizer(lowercase=True, 
                                          min_df=2, 
                                          stop_words='english',
                                          ngram_range=(1, 2))
        self.model = None
        self.categories = None
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def preprocess_text(self, text):
        """Clean and normalize transaction text."""
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove numbers but keep words
        text = re.sub(r'\b\d+\b', 'NUM', text)
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def extract_features(self, df):
        """Extract features from transaction data."""
        # Preprocess transaction text
        texts = df['Text'].apply(self.preprocess_text)
        
        # Add day of week and month as features
        if 'Booking date' in df.columns:
            df['DayOfWeek'] = df['Booking date'].dt.dayofweek
            df['Month'] = df['Booking date'].dt.month
        
        # Create text features using TF-IDF
        if self.vectorizer.vocabulary_ is None:
            text_features = self.vectorizer.fit_transform(texts)
        else:
            text_features = self.vectorizer.transform(texts)
        
        # Add amount-based features
        if 'Amount' in df.columns:
            amount_features = np.array([
                df['Amount'],
                df['Amount'].abs(),
                (df['Amount'] > 0).astype(int),  # Is income flag
            ]).T
        else:
            amount_features = np.zeros((len(df), 3))
        
        # Add time-based features if available
        if 'DayOfWeek' in df.columns and 'Month' in df.columns:
            time_features = np.array([
                df['DayOfWeek'],
                df['Month']
            ]).T
        else:
            time_features = np.zeros((len(df), 2))
            
        return text_features, amount_features, time_features
    
    def train(self, transactions_df, target_col='Category', test_size=0.2, random_state=42):
        """Train the categorization model using labeled transaction data."""
        # Store unique categories
        self.categories = transactions_df[target_col].unique()
        print(f"Training on {len(transactions_df)} transactions with {len(self.categories)} categories")
        
        # Extract features
        text_features, amount_features, time_features = self.extract_features(transactions_df)
        
        # Split data into training and testing sets
        X_text_train, X_text_test, X_amount_train, X_amount_test, X_time_train, X_time_test, y_train, y_test = train_test_split(
            text_features, amount_features, time_features, 
            transactions_df[target_col], 
            test_size=test_size, 
            random_state=random_state,
            stratify=transactions_df[target_col] if len(transactions_df) > 100 else None
        )
        
        # Train model (RandomForest for this example)
        # In a real implementation, you might want to use more sophisticated models or ensemble methods
        self.model = RandomForestClassifier(n_estimators=100, random_state=random_state)
        
        # Combine features - for a RandomForest, we need to convert sparse matrix to dense
        X_train_combined = np.hstack((X_text_train.toarray(), X_amount_train, X_time_train))
        
        # Train the model
        self.model.fit(X_train_combined, y_train)
        
        # Evaluate on test set
        X_test_combined = np.hstack((X_text_test.toarray(), X_amount_test, X_time_test))
        y_pred = self.model.predict(X_test_combined)
        
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model accuracy: {accuracy:.2f}")
        print(classification_report(y_test, y_pred))
        
        return accuracy
    
    def predict(self, transactions_df):
        """Predict categories for new transactions."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first or load a pre-trained model.")
        
        # Extract features
        text_features, amount_features, time_features = self.extract_features(transactions_df)
        
        # Combine features
        X_combined = np.hstack((text_features.toarray(), amount_features, time_features))
        
        # Predict
        predictions = self.model.predict(X_combined)
        
        # Add predictions to dataframe
        result_df = transactions_df.copy()
        result_df['PredictedCategory'] = predictions
        
        return result_df
    
    def save_model(self, model_path):
        """Save the trained model and vectorizer to a file."""
        if self.model is None:
            raise ValueError("No trained model to save.")
        
        model_dir = os.path.dirname(model_path)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        # Save both the model and vectorizer
        joblib.dump({
            'model': self.model,
            'vectorizer': self.vectorizer,
            'categories': self.categories
        }, model_path)
        
        print(f"Model saved to {model_path}")
    
    def load_model(self, model_path):
        """Load a trained model and vectorizer from a file."""
        try:
            saved_data = joblib.load(model_path)
            self.model = saved_data['model']
            self.vectorizer = saved_data['vectorizer']
            self.categories = saved_data['categories']
            print(f"Model loaded from {model_path}")
            print(f"Categories: {self.categories}")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise


if __name__ == "__main__":
    # Example usage
    # Load the processed transactions
    data_path = "data/processed/transactions_categorized.csv"
    transactions = pd.read_csv(data_path, parse_dates=['Booking date', 'Value date'])
    
    # Initialize and train the categorizer
    categorizer = TransactionCategorizer()
    accuracy = categorizer.train(transactions)
    
    # Save the trained model
    categorizer.save_model("models/transaction_categorizer.joblib")
    
    # Make predictions on new data (using the same data for demonstration)
    # In a real scenario, you'd use a separate dataset
    predictions = categorizer.predict(transactions)
    
    # Show some examples
    print("\nSample Predictions:")
    print(predictions[['Text', 'Amount', 'Category', 'PredictedCategory']].head(10))
    
    # Calculate accuracy
    correct = predictions['Category'] == predictions['PredictedCategory']
    accuracy = correct.mean()
    print(f"\nPrediction accuracy on training data: {accuracy:.4f}")