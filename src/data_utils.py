"""
Data utilities for fraud detection preprocessing and feature engineering.
Part of the FinTechCo Fraud Detection Demo - Production System.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional, Dict, Any
import joblib
import os


class DataPreprocessor:
    """
    Production-ready data preprocessing pipeline for fraud detection.

    Handles feature scaling, train-test splits, and data validation
    for credit card fraud detection using PCA-transformed features.
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_names = None
        self.is_fitted = False

    def load_data(self, data_path: str) -> pd.DataFrame:
        """
        Load credit card fraud dataset with validation.

        Args:
            data_path: Path to creditcard.csv file

        Returns:
            DataFrame with validated fraud detection data

        Raises:
            FileNotFoundError: If data file doesn't exist
            ValueError: If data format is invalid
        """
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")

        try:
            df = pd.read_csv(data_path)
        except Exception as e:
            raise ValueError(f"Error loading data: {str(e)}")

        # Validate required columns
        required_cols = ['Class'] + [f'V{i}' for i in range(1, 29)] + ['Time', 'Amount']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Validate data types and ranges
        if df['Class'].min() < 0 or df['Class'].max() > 1:
            raise ValueError("Class column must contain only 0 and 1 values")

        if df.isnull().sum().sum() > 0:
            raise ValueError("Dataset contains missing values")

        print(f"✅ Data loaded successfully: {df.shape}")
        print(f"   • Total transactions: {len(df):,}")
        print(f"   • Fraud transactions: {df['Class'].sum():,} ({df['Class'].mean()*100:.3f}%)")
        print(f"   • Normal transactions: {(df['Class']==0).sum():,} ({(df['Class']==0).mean()*100:.3f}%)")

        return df

    def prepare_features(self, df: pd.DataFrame, fit_scaler: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare features and target for modeling.

        Args:
            df: Input DataFrame with fraud data
            fit_scaler: Whether to fit the scaler (True for training, False for inference)

        Returns:
            Tuple of (X_scaled, y) arrays
        """
        # Separate features and target
        X = df.drop('Class', axis=1)
        y = df['Class'].values

        # Store feature names
        if self.feature_names is None:
            self.feature_names = X.columns.tolist()

        # Scale features
        if fit_scaler:
            X_scaled = self.scaler.fit_transform(X)
            self.is_fitted = True
            print(f"✅ Scaler fitted on {X.shape[0]:,} samples with {X.shape[1]} features")
        else:
            if not self.is_fitted:
                raise ValueError("Scaler must be fitted before transform. Set fit_scaler=True.")
            X_scaled = self.scaler.transform(X)

        return X_scaled, y

    def create_train_test_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Create stratified train-test split for imbalanced data.

        Args:
            X: Feature matrix
            y: Target vector
            test_size: Proportion of data for testing
            random_state: Random seed for reproducibility

        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y  # Maintain class distribution
        )

        print(f"✅ Train-test split created:")
        print(f"   • Train: {len(X_train):,} samples ({y_train.sum()} fraud, {y_train.mean()*100:.3f}%)")
        print(f"   • Test: {len(X_test):,} samples ({y_test.sum()} fraud, {y_test.mean()*100:.3f}%)")

        return X_train, X_test, y_train, y_test

    def validate_transaction(self, transaction: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and prepare a single transaction for scoring.

        Args:
            transaction: Dictionary with transaction features

        Returns:
            Validated and formatted transaction data

        Raises:
            ValueError: If transaction format is invalid
        """
        if not isinstance(transaction, dict):
            raise ValueError("Transaction must be a dictionary")

        # Check for required features
        if self.feature_names is None:
            raise ValueError("Preprocessor must be fitted before validating transactions")

        missing_features = [f for f in self.feature_names if f not in transaction]
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")

        # Convert to DataFrame for consistent processing
        df = pd.DataFrame([transaction])

        # Validate feature ranges (basic sanity checks)
        if 'Amount' in transaction and transaction['Amount'] < 0:
            raise ValueError("Transaction amount cannot be negative")

        if 'Time' in transaction and transaction['Time'] < 0:
            raise ValueError("Transaction time cannot be negative")

        return transaction

    def save_preprocessor(self, filepath: str) -> None:
        """Save fitted preprocessor to disk."""
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted preprocessor")

        preprocessor_data = {
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'is_fitted': self.is_fitted
        }

        joblib.dump(preprocessor_data, filepath)
        print(f"✅ Preprocessor saved to {filepath}")

    def load_preprocessor(self, filepath: str) -> None:
        """Load fitted preprocessor from disk."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Preprocessor file not found: {filepath}")

        preprocessor_data = joblib.load(filepath)
        self.scaler = preprocessor_data['scaler']
        self.feature_names = preprocessor_data['feature_names']
        self.is_fitted = preprocessor_data['is_fitted']

        print(f"✅ Preprocessor loaded from {filepath}")

    def get_feature_stats(self) -> Optional[Dict[str, Any]]:
        """Get statistics about the fitted preprocessor."""
        if not self.is_fitted:
            return None

        return {
            'n_features': len(self.feature_names),
            'feature_names': self.feature_names,
            'feature_means': self.scaler.mean_.tolist(),
            'feature_stds': self.scaler.scale_.tolist()
        }


def create_sample_transaction(fraud: bool = False, random_state: Optional[int] = None) -> Dict[str, Any]:
    """
    Create a sample transaction for testing/demo purposes.

    Args:
        fraud: Whether to create a fraud-like transaction
        random_state: Random seed for reproducibility

    Returns:
        Dictionary with sample transaction features
    """
    if random_state is not None:
        np.random.seed(random_state)

    # Create base transaction with PCA features
    transaction = {}

    # Time feature (in seconds from start of dataset)
    transaction['Time'] = np.random.uniform(0, 172792)  # 2 days in seconds

    # PCA features V1-V28 (different distributions for fraud vs normal)
    if fraud:
        # Fraud transactions tend to have different PCA patterns
        for i in range(1, 29):
            if i in [14, 4, 11, 12, 18]:  # Key discriminative features
                transaction[f'V{i}'] = np.random.normal(0, 3)  # More extreme values
            else:
                transaction[f'V{i}'] = np.random.normal(0, 1.5)
        # Higher amount for fraud
        transaction['Amount'] = np.random.lognormal(4, 1.8)  # Higher amounts
    else:
        # Normal transaction patterns
        for i in range(1, 29):
            transaction[f'V{i}'] = np.random.normal(0, 1)
        # Normal amount distribution
        transaction['Amount'] = np.random.lognormal(3, 1.5)

    return transaction


if __name__ == "__main__":
    # Demo usage
    print("🔧 Data Preprocessing Utils Demo")

    # Create sample transactions
    normal_txn = create_sample_transaction(fraud=False, random_state=42)
    fraud_txn = create_sample_transaction(fraud=True, random_state=123)

    print(f"\\n📊 Sample Normal Transaction:")
    print(f"   Amount: ${normal_txn['Amount']:.2f}")
    print(f"   Time: {normal_txn['Time']:.0f} seconds")
    print(f"   V1: {normal_txn['V1']:.3f}")

    print(f"\\n🚨 Sample Fraud Transaction:")
    print(f"   Amount: ${fraud_txn['Amount']:.2f}")
    print(f"   Time: {fraud_txn['Time']:.0f} seconds")
    print(f"   V1: {fraud_txn['V1']:.3f}")

    print("\\n✅ Data utilities ready for production use!")