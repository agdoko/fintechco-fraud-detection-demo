"""
FinTechCo Fraud Detection Demo - Production System

A comprehensive fraud detection system showcasing Claude Code's value for
data science teams working on imbalanced classification problems.

This package provides:
- Production-ready fraud detection API
- Advanced ML techniques (XGBoost + SMOTE)
- Model explainability with SHAP
- Real-time transaction scoring
- Comprehensive evaluation and monitoring

Main components:
- FraudDetector: Main production API
- DataPreprocessor: Data preprocessing pipeline
- ModelTrainer: Advanced model training utilities
- FraudExplainer: SHAP-based explainability

Example usage:
    from src import FraudDetector

    detector = FraudDetector()
    results = detector.train_system('data/creditcard.csv')
    prediction = detector.predict_transaction(transaction)
"""

from .fraud_detector import FraudDetector
from .data_utils import DataPreprocessor, create_sample_transaction
from .model_utils import ModelTrainer, calculate_business_impact_comparison
from .explainer import FraudExplainer

__version__ = "1.0.0"
__author__ = "Claude Code Demo"
__description__ = "Production-ready fraud detection system for imbalanced datasets"

# Package metadata
__all__ = [
    'FraudDetector',
    'DataPreprocessor',
    'ModelTrainer',
    'FraudExplainer',
    'create_sample_transaction',
    'calculate_business_impact_comparison'
]

# System status
SYSTEM_READY = True
COMPONENTS_LOADED = {
    'fraud_detector': True,
    'data_utils': True,
    'model_utils': True,
    'explainer': True
}

print("🚀 FinTechCo Fraud Detection System - Production Ready!")
print(f"📦 Version: {__version__}")
print("✅ All components loaded successfully")