"""
Model utilities for fraud detection training, evaluation, and persistence.
Part of the FinTechCo Fraud Detection Demo - Production System.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve,
    accuracy_score, precision_score, recall_score, f1_score
)
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import joblib
import json
import os
from datetime import datetime
from typing import Dict, Any, Tuple, Optional, List
import warnings
warnings.filterwarnings('ignore')


class ModelTrainer:
    """
    Production-ready model training and evaluation for fraud detection.

    Implements the winning XGBoost + SMOTE approach from Milestone 3
    with production-grade error handling and model persistence.
    """

    def __init__(self):
        self.model = None
        self.smote = None
        self.model_metadata = {}
        self.is_trained = False

    def train_production_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        smote_strategy: float = 0.3,
        xgb_params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Train the production fraud detection model using XGBoost + SMOTE.

        Args:
            X_train: Training features
            y_train: Training labels
            smote_strategy: SMOTE sampling strategy (fraud ratio in balanced data)
            xgb_params: XGBoost parameters (uses optimized defaults if None)

        Returns:
            Training results and metadata
        """
        print("🔄 Training production fraud detection model...")

        # Record training start time
        training_start = datetime.now()

        # Default XGBoost parameters optimized for fraud detection
        if xgb_params is None:
            xgb_params = {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42,
                'eval_metric': 'logloss',
                'use_label_encoder': False
            }

        # Apply SMOTE for balanced training data
        print(f"📊 Applying SMOTE with {smote_strategy*100:.0f}% fraud ratio...")
        self.smote = SMOTE(sampling_strategy=smote_strategy, random_state=42)
        X_train_balanced, y_train_balanced = self.smote.fit_resample(X_train, y_train)

        print(f"   • Original training: {len(X_train):,} samples ({y_train.sum()} fraud)")
        print(f"   • Balanced training: {len(X_train_balanced):,} samples ({y_train_balanced.sum()} fraud)")
        print(f"   • New fraud ratio: {y_train_balanced.mean()*100:.1f}%")

        # Train XGBoost model
        print("🚀 Training XGBoost model...")
        self.model = xgb.XGBClassifier(**xgb_params)
        self.model.fit(X_train_balanced, y_train_balanced)

        # Calculate training time
        training_time = (datetime.now() - training_start).total_seconds()

        # Store training metadata
        self.model_metadata = {
            'model_type': 'XGBoost + SMOTE',
            'training_samples': len(X_train),
            'balanced_samples': len(X_train_balanced),
            'original_fraud_rate': float(y_train.mean()),
            'balanced_fraud_rate': float(y_train_balanced.mean()),
            'smote_strategy': smote_strategy,
            'xgb_params': xgb_params,
            'training_time_seconds': training_time,
            'training_date': training_start.isoformat(),
            'feature_names': None,  # To be set by caller
            'n_features': X_train.shape[1]
        }

        self.is_trained = True

        print(f"✅ Model training completed in {training_time:.1f} seconds")

        return self.model_metadata

    def evaluate_model(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        Comprehensive model evaluation with business metrics.

        Args:
            X_test: Test features
            y_test: Test labels
            threshold: Decision threshold for predictions

        Returns:
            Comprehensive evaluation results
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")

        print(f"📊 Evaluating model with threshold {threshold:.2f}...")

        # Predictions
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba >= threshold).astype(int)

        # Technical metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        pr_auc = average_precision_score(y_test, y_pred_proba)

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()

        # Business metrics
        avg_fraud_amount = 150
        cost_per_fp = 10
        investigation_cost = 25

        missed_fraud_cost = fn * avg_fraud_amount
        false_positive_cost = fp * cost_per_fp
        investigation_cost_total = tp * investigation_cost
        prevented_fraud_savings = tp * avg_fraud_amount

        total_cost = missed_fraud_cost + false_positive_cost + investigation_cost_total
        net_savings = prevented_fraud_savings - total_cost

        # Compile results
        results = {
            'threshold': threshold,
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'roc_auc': float(roc_auc),
            'pr_auc': float(pr_auc),
            'confusion_matrix': {
                'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)
            },
            'business_metrics': {
                'missed_fraud_cost': float(missed_fraud_cost),
                'false_positive_cost': float(false_positive_cost),
                'investigation_cost': float(investigation_cost_total),
                'prevented_fraud_savings': float(prevented_fraud_savings),
                'net_business_savings': float(net_savings)
            },
            'fraud_detection_rate': float(recall * 100),
            'false_positive_rate': float((fp / (fp + tn)) * 100) if (fp + tn) > 0 else 0.0,
            'evaluation_date': datetime.now().isoformat()
        }

        # Print summary
        print(f"✅ Model Evaluation Results:")
        print(f"   🎯 Fraud Detection Rate: {results['fraud_detection_rate']:.1f}%")
        print(f"   🎯 Precision: {precision*100:.1f}%")
        print(f"   🎯 F1-Score: {f1:.3f}")
        print(f"   💰 Net Business Savings: ${net_savings:,.0f}")

        return results

    def optimize_threshold(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        thresholds: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Optimize decision threshold for maximum business value.

        Args:
            X_test: Test features
            y_test: Test labels
            thresholds: Array of thresholds to test (defaults to 0.1-0.9)

        Returns:
            Optimization results with best threshold
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before threshold optimization")

        if thresholds is None:
            thresholds = np.arange(0.1, 0.9, 0.05)

        print(f"🎯 Optimizing threshold across {len(thresholds)} values...")

        # Get probabilities once
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]

        # Test each threshold
        results = []
        for threshold in thresholds:
            evaluation = self.evaluate_model_at_threshold(y_test, y_pred_proba, threshold)
            results.append(evaluation)

        # Find optimal threshold (maximize business value)
        best_idx = np.argmax([r['business_metrics']['net_business_savings'] for r in results])
        best_result = results[best_idx]

        optimization_results = {
            'optimal_threshold': float(thresholds[best_idx]),
            'best_result': best_result,
            'all_results': results,
            'thresholds_tested': thresholds.tolist()
        }

        print(f"✅ Optimal threshold found: {optimization_results['optimal_threshold']:.2f}")
        print(f"   💰 Maximum business savings: ${best_result['business_metrics']['net_business_savings']:,.0f}")

        return optimization_results

    def evaluate_model_at_threshold(
        self,
        y_test: np.ndarray,
        y_pred_proba: np.ndarray,
        threshold: float
    ) -> Dict[str, Any]:
        """Helper method for threshold optimization."""
        y_pred = (y_pred_proba >= threshold).astype(int)

        # Calculate metrics quickly
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        # Business metrics
        avg_fraud_amount, cost_per_fp, investigation_cost = 150, 10, 25
        missed_fraud_cost = fn * avg_fraud_amount
        false_positive_cost = fp * cost_per_fp
        investigation_cost_total = tp * investigation_cost
        prevented_fraud_savings = tp * avg_fraud_amount
        net_savings = prevented_fraud_savings - (missed_fraud_cost + false_positive_cost + investigation_cost_total)

        return {
            'threshold': float(threshold),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'confusion_matrix': {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)},
            'business_metrics': {'net_business_savings': float(net_savings)}
        }

    def get_feature_importance(self, feature_names: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Get feature importance from trained XGBoost model.

        Args:
            feature_names: List of feature names

        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self.is_trained:
            raise ValueError("Model must be trained to get feature importance")

        importances = self.model.feature_importances_

        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(len(importances))]

        importance_dict = dict(zip(feature_names, importances.astype(float)))

        # Sort by importance
        importance_dict = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))

        return importance_dict

    def save_model(self, model_dir: str, model_name: str = "fraud_detector") -> Dict[str, str]:
        """
        Save trained model and preprocessing components.

        Args:
            model_dir: Directory to save model files
            model_name: Base name for model files

        Returns:
            Dictionary with paths to saved files
        """
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")

        os.makedirs(model_dir, exist_ok=True)

        # Save paths
        model_path = os.path.join(model_dir, f"{model_name}_xgboost.pkl")
        smote_path = os.path.join(model_dir, f"{model_name}_smote.pkl")
        metadata_path = os.path.join(model_dir, f"{model_name}_metadata.json")

        # Save model components
        joblib.dump(self.model, model_path)
        joblib.dump(self.smote, smote_path)

        # Save metadata
        with open(metadata_path, 'w') as f:
            json.dump(self.model_metadata, f, indent=2)

        saved_files = {
            'model': model_path,
            'smote': smote_path,
            'metadata': metadata_path
        }

        print(f"✅ Model saved successfully:")
        for component, path in saved_files.items():
            print(f"   • {component}: {path}")

        return saved_files

    def load_model(self, model_dir: str, model_name: str = "fraud_detector") -> None:
        """
        Load trained model and preprocessing components.

        Args:
            model_dir: Directory containing model files
            model_name: Base name for model files
        """
        # File paths
        model_path = os.path.join(model_dir, f"{model_name}_xgboost.pkl")
        smote_path = os.path.join(model_dir, f"{model_name}_smote.pkl")
        metadata_path = os.path.join(model_dir, f"{model_name}_metadata.json")

        # Check files exist
        missing_files = []
        for name, path in [('model', model_path), ('smote', smote_path), ('metadata', metadata_path)]:
            if not os.path.exists(path):
                missing_files.append(f"{name}: {path}")

        if missing_files:
            raise FileNotFoundError(f"Missing model files: {missing_files}")

        # Load components
        self.model = joblib.load(model_path)
        self.smote = joblib.load(smote_path)

        # Load metadata
        with open(metadata_path, 'r') as f:
            self.model_metadata = json.load(f)

        self.is_trained = True

        print(f"✅ Model loaded successfully from {model_dir}")
        print(f"   • Model type: {self.model_metadata.get('model_type', 'Unknown')}")
        print(f"   • Training date: {self.model_metadata.get('training_date', 'Unknown')}")


def calculate_business_impact_comparison(baseline_results: Dict[str, Any],
                                       advanced_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate business impact comparison between baseline and advanced models.

    Args:
        baseline_results: Results from baseline model evaluation
        advanced_results: Results from advanced model evaluation

    Returns:
        Comparison metrics and improvements
    """
    baseline_savings = baseline_results['business_metrics']['net_business_savings']
    advanced_savings = advanced_results['business_metrics']['net_business_savings']

    improvement = advanced_savings - baseline_savings
    improvement_pct = ((advanced_savings - baseline_savings) / abs(baseline_savings) * 100) if baseline_savings != 0 else float('inf')

    baseline_recall = baseline_results['recall'] * 100
    advanced_recall = advanced_results['recall'] * 100
    recall_improvement = advanced_recall - baseline_recall

    comparison = {
        'baseline_business_impact': float(baseline_savings),
        'advanced_business_impact': float(advanced_savings),
        'business_improvement': float(improvement),
        'business_improvement_pct': float(improvement_pct) if improvement_pct != float('inf') else 'infinite',
        'baseline_fraud_detection': float(baseline_recall),
        'advanced_fraud_detection': float(advanced_recall),
        'fraud_detection_improvement': float(recall_improvement),
        'summary': f"Advanced techniques improved business value by ${improvement:,.0f} ({improvement_pct:.0f}%) and fraud detection by {recall_improvement:.1f} percentage points"
    }

    return comparison


if __name__ == "__main__":
    # Demo usage
    print("🔧 Model Training Utils Demo")
    print("✅ Production-ready model utilities loaded!")
    print("🚀 Ready for XGBoost + SMOTE training pipeline!")