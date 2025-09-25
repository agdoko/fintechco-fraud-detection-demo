"""
Production fraud detection system - Main API for real-time transaction scoring.
Part of the FinTechCo Fraud Detection Demo - Production System.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Union, Optional
import os
import time
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

from .data_utils import DataPreprocessor
from .model_utils import ModelTrainer


class FraudDetector:
    """
    Production-ready fraud detection system for real-time transaction scoring.

    This class provides the main API for fraud detection, integrating data preprocessing,
    model inference, and explainability features in a production-ready interface.
    """

    def __init__(self, model_dir: Optional[str] = None, model_name: str = "fraud_detector"):
        """
        Initialize fraud detection system.

        Args:
            model_dir: Directory containing trained model files (None for training mode)
            model_name: Base name for model files
        """
        self.preprocessor = DataPreprocessor()
        self.model_trainer = ModelTrainer()
        self.model_dir = model_dir
        self.model_name = model_name

        self.optimal_threshold = 0.5
        self.model_metadata = {}
        self.performance_stats = {}
        self.is_ready = False

        # Load model if directory provided
        if model_dir and os.path.exists(model_dir):
            self.load_model()

    def train_system(
        self,
        data_path: str,
        save_dir: str = "outputs",
        test_size: float = 0.2,
        optimize_threshold: bool = True
    ) -> Dict[str, Any]:
        """
        Train the complete fraud detection system from data.

        Args:
            data_path: Path to creditcard.csv dataset
            save_dir: Directory to save trained model
            test_size: Proportion of data for testing
            optimize_threshold: Whether to optimize decision threshold

        Returns:
            Training and evaluation results
        """
        print("🚀 Training complete fraud detection system...")
        training_start = datetime.now()

        # Load and preprocess data
        df = self.preprocessor.load_data(data_path)
        X_scaled, y = self.preprocessor.prepare_features(df, fit_scaler=True)
        X_train, X_test, y_train, y_test = self.preprocessor.create_train_test_split(
            X_scaled, y, test_size=test_size
        )

        # Train model
        training_metadata = self.model_trainer.train_production_model(X_train, y_train)
        training_metadata['feature_names'] = self.preprocessor.feature_names

        # Evaluate model
        evaluation_results = self.model_trainer.evaluate_model(X_test, y_test)

        # Optimize threshold if requested
        if optimize_threshold:
            print("🎯 Optimizing decision threshold for maximum business value...")
            optimization_results = self.model_trainer.optimize_threshold(X_test, y_test)
            self.optimal_threshold = optimization_results['optimal_threshold']

            # Re-evaluate with optimal threshold
            evaluation_results = self.model_trainer.evaluate_model(X_test, y_test, self.optimal_threshold)

        # Save everything
        os.makedirs(save_dir, exist_ok=True)
        saved_files = self.model_trainer.save_model(save_dir, self.model_name)
        self.preprocessor.save_preprocessor(os.path.join(save_dir, f"{self.model_name}_preprocessor.pkl"))

        # Save threshold and results
        system_config = {
            'optimal_threshold': self.optimal_threshold,
            'evaluation_results': evaluation_results,
            'training_metadata': training_metadata
        }

        config_path = os.path.join(save_dir, f"{self.model_name}_config.json")
        with open(config_path, 'w') as f:
            json.dump(system_config, f, indent=2)

        saved_files['config'] = config_path
        saved_files['preprocessor'] = os.path.join(save_dir, f"{self.model_name}_preprocessor.pkl")

        # Update internal state
        self.model_dir = save_dir
        self.model_metadata = training_metadata
        self.performance_stats = evaluation_results
        self.is_ready = True

        training_time = (datetime.now() - training_start).total_seconds()

        results = {
            'training_time_seconds': training_time,
            'model_performance': evaluation_results,
            'optimal_threshold': self.optimal_threshold,
            'saved_files': saved_files,
            'system_status': 'ready'
        }

        print(f"✅ Fraud detection system trained successfully in {training_time:.1f} seconds!")
        print(f"🎯 Fraud detection rate: {evaluation_results['fraud_detection_rate']:.1f}%")
        print(f"💰 Business savings: ${evaluation_results['business_metrics']['net_business_savings']:,.0f}")

        return results

    def load_model(self) -> None:
        """Load trained model and all components."""
        if not self.model_dir or not os.path.exists(self.model_dir):
            raise ValueError(f"Model directory not found: {self.model_dir}")

        print(f"📂 Loading fraud detection system from {self.model_dir}...")

        # Load model components
        self.model_trainer.load_model(self.model_dir, self.model_name)
        self.preprocessor.load_preprocessor(os.path.join(self.model_dir, f"{self.model_name}_preprocessor.pkl"))

        # Load system configuration
        config_path = os.path.join(self.model_dir, f"{self.model_name}_config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
                self.optimal_threshold = config.get('optimal_threshold', 0.5)
                self.performance_stats = config.get('evaluation_results', {})
                self.model_metadata = config.get('training_metadata', {})

        self.is_ready = True
        print("✅ Fraud detection system loaded and ready!")

    def predict_transaction(
        self,
        transaction: Union[Dict[str, Any], pd.DataFrame],
        explain: bool = False
    ) -> Dict[str, Any]:
        """
        Score a single transaction for fraud probability.

        Args:
            transaction: Transaction data as dict or DataFrame
            explain: Whether to include SHAP explanations

        Returns:
            Prediction results with fraud probability, decision, and optional explanations
        """
        if not self.is_ready:
            raise ValueError("Fraud detection system not ready. Train or load model first.")

        start_time = time.time()

        # Validate and prepare transaction
        if isinstance(transaction, dict):
            validated_txn = self.preprocessor.validate_transaction(transaction)
            df = pd.DataFrame([validated_txn])
        else:
            df = transaction.copy()

        # Preprocess
        X_scaled, _ = self.preprocessor.prepare_features(df, fit_scaler=False)

        # Predict
        fraud_probability = self.model_trainer.model.predict_proba(X_scaled)[0, 1]
        is_fraud = fraud_probability >= self.optimal_threshold

        # Calculate confidence level
        confidence = max(fraud_probability, 1 - fraud_probability)

        prediction_time = (time.time() - start_time) * 1000  # Convert to milliseconds

        result = {
            'fraud_probability': float(fraud_probability),
            'is_fraud': bool(is_fraud),
            'confidence': float(confidence),
            'decision_threshold': float(self.optimal_threshold),
            'prediction_time_ms': float(prediction_time),
            'risk_level': self._get_risk_level(fraud_probability),
            'timestamp': datetime.now().isoformat()
        }

        # Add explanations if requested
        if explain:
            try:
                from .explainer import FraudExplainer
                explainer = FraudExplainer(self.model_trainer.model)
                explanations = explainer.explain_prediction(X_scaled[0], self.preprocessor.feature_names)
                result['explanations'] = explanations
            except ImportError:
                result['explanations'] = {'error': 'SHAP explainer not available'}

        return result

    def predict_batch(
        self,
        transactions: Union[pd.DataFrame, List[Dict[str, Any]]],
        include_details: bool = False
    ) -> Dict[str, Any]:
        """
        Score multiple transactions in batch for efficiency.

        Args:
            transactions: DataFrame or list of transaction dictionaries
            include_details: Whether to include detailed per-transaction results

        Returns:
            Batch prediction results with summary statistics
        """
        if not self.is_ready:
            raise ValueError("Fraud detection system not ready. Train or load model first.")

        start_time = time.time()

        # Convert to DataFrame if needed
        if isinstance(transactions, list):
            df = pd.DataFrame(transactions)
        else:
            df = transactions.copy()

        print(f"🔄 Scoring {len(df)} transactions...")

        # Preprocess all transactions
        X_scaled, _ = self.preprocessor.prepare_features(df, fit_scaler=False)

        # Batch prediction
        fraud_probabilities = self.model_trainer.model.predict_proba(X_scaled)[:, 1]
        fraud_predictions = fraud_probabilities >= self.optimal_threshold

        processing_time = (time.time() - start_time) * 1000

        # Calculate summary statistics
        n_transactions = len(df)
        n_fraud_detected = fraud_predictions.sum()
        avg_fraud_probability = fraud_probabilities.mean()
        max_fraud_probability = fraud_probabilities.max()
        high_risk_count = (fraud_probabilities >= 0.8).sum()

        results = {
            'batch_summary': {
                'total_transactions': int(n_transactions),
                'fraud_detected': int(n_fraud_detected),
                'fraud_rate': float(n_fraud_detected / n_transactions * 100),
                'avg_fraud_probability': float(avg_fraud_probability),
                'max_fraud_probability': float(max_fraud_probability),
                'high_risk_transactions': int(high_risk_count),
                'processing_time_ms': float(processing_time),
                'throughput_tps': float(n_transactions / (processing_time / 1000))
            }
        }

        # Include detailed results if requested
        if include_details:
            transaction_results = []
            for i, (prob, pred) in enumerate(zip(fraud_probabilities, fraud_predictions)):
                transaction_results.append({
                    'transaction_id': i,
                    'fraud_probability': float(prob),
                    'is_fraud': bool(pred),
                    'risk_level': self._get_risk_level(prob)
                })
            results['transaction_details'] = transaction_results

        print(f"✅ Batch scoring completed: {n_fraud_detected}/{n_transactions} flagged as fraud")

        return results

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status and health metrics."""
        status = {
            'is_ready': self.is_ready,
            'model_loaded': self.model_trainer.is_trained,
            'preprocessor_fitted': self.preprocessor.is_fitted,
            'optimal_threshold': self.optimal_threshold,
            'system_timestamp': datetime.now().isoformat()
        }

        if self.is_ready:
            status.update({
                'model_metadata': self.model_metadata,
                'performance_stats': self.performance_stats,
                'model_directory': self.model_dir
            })

        return status

    def get_feature_importance(self) -> Dict[str, float]:
        """Get model feature importance rankings."""
        if not self.is_ready:
            raise ValueError("System not ready")

        return self.model_trainer.get_feature_importance(self.preprocessor.feature_names)

    def _get_risk_level(self, fraud_probability: float) -> str:
        """Convert fraud probability to human-readable risk level."""
        if fraud_probability >= 0.8:
            return "HIGH"
        elif fraud_probability >= 0.5:
            return "MEDIUM"
        elif fraud_probability >= 0.2:
            return "LOW"
        else:
            return "MINIMAL"

    def simulate_real_time_monitoring(self, duration_seconds: int = 60) -> Dict[str, Any]:
        """
        Simulate real-time transaction monitoring for demo purposes.

        Args:
            duration_seconds: How long to simulate monitoring

        Returns:
            Monitoring results and statistics
        """
        if not self.is_ready:
            raise ValueError("System not ready for monitoring")

        from .data_utils import create_sample_transaction
        import random

        print(f"🔄 Simulating {duration_seconds} seconds of real-time monitoring...")

        monitoring_start = time.time()
        transactions_processed = 0
        fraud_detected = 0
        alerts_generated = []

        while (time.time() - monitoring_start) < duration_seconds:
            # Generate random transaction (90% normal, 10% fraud-like)
            is_suspicious = random.random() < 0.1
            transaction = create_sample_transaction(fraud=is_suspicious)

            # Score transaction
            result = self.predict_transaction(transaction)

            transactions_processed += 1

            # Generate alert if fraud detected
            if result['is_fraud']:
                fraud_detected += 1
                alert = {
                    'timestamp': result['timestamp'],
                    'fraud_probability': result['fraud_probability'],
                    'risk_level': result['risk_level'],
                    'transaction_amount': transaction['Amount']
                }
                alerts_generated.append(alert)

            # Simulate realistic transaction rate (delay between transactions)
            time.sleep(random.uniform(0.1, 0.5))

        monitoring_duration = time.time() - monitoring_start

        results = {
            'monitoring_summary': {
                'duration_seconds': float(monitoring_duration),
                'transactions_processed': transactions_processed,
                'fraud_detected': fraud_detected,
                'fraud_rate': float(fraud_detected / transactions_processed * 100) if transactions_processed > 0 else 0.0,
                'avg_processing_speed_tps': float(transactions_processed / monitoring_duration),
                'alerts_generated': len(alerts_generated)
            },
            'alerts': alerts_generated[-10:],  # Last 10 alerts
            'status': 'monitoring_complete'
        }

        print(f"✅ Monitoring complete: {fraud_detected}/{transactions_processed} fraud detected")

        return results


if __name__ == "__main__":
    # Demo usage
    print("🔧 Fraud Detection System Demo")

    # Create system instance
    detector = FraudDetector()

    print(f"✅ FraudDetector initialized")
    print(f"🎯 System status: {detector.get_system_status()['is_ready']}")
    print("🚀 Ready for training or inference!")