"""
SHAP-based model explainer for fraud detection interpretability.
Part of the FinTechCo Fraud Detection Demo - Production System.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Union
import shap
import matplotlib.pyplot as plt
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class FraudExplainer:
    """
    Production-ready model explainer using SHAP for fraud detection interpretability.

    Provides individual prediction explanations, feature importance analysis,
    and visualization capabilities for regulatory compliance and trust building.
    """

    def __init__(self, model, model_type: str = "tree"):
        """
        Initialize SHAP explainer for fraud detection model.

        Args:
            model: Trained model (XGBoost, etc.)
            model_type: Type of SHAP explainer ("tree", "linear", "kernel")
        """
        self.model = model
        self.model_type = model_type
        self.explainer = None
        self.background_data = None
        self.is_fitted = False

    def fit_explainer(self, X_background: np.ndarray, max_samples: int = 100) -> None:
        """
        Fit SHAP explainer with background data for accurate explanations.

        Args:
            X_background: Background dataset for SHAP explanations
            max_samples: Maximum samples to use for background (for efficiency)
        """
        print("🔄 Fitting SHAP explainer for fraud detection model...")

        # Sample background data if needed
        if len(X_background) > max_samples:
            indices = np.random.choice(len(X_background), max_samples, replace=False)
            self.background_data = X_background[indices]
        else:
            self.background_data = X_background

        # Initialize appropriate SHAP explainer
        if self.model_type == "tree":
            # For tree-based models (XGBoost, RandomForest)
            self.explainer = shap.TreeExplainer(self.model)
        elif self.model_type == "linear":
            # For linear models
            self.explainer = shap.LinearExplainer(self.model, self.background_data)
        else:
            # For any model (slower but universal)
            self.explainer = shap.KernelExplainer(self.model.predict_proba, self.background_data)

        self.is_fitted = True
        print(f"✅ SHAP explainer fitted with {len(self.background_data)} background samples")

    def explain_prediction(
        self,
        transaction: np.ndarray,
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Generate SHAP explanation for a single transaction prediction.

        Args:
            transaction: Single transaction feature vector
            feature_names: List of feature names for interpretation

        Returns:
            Dictionary with SHAP values and explanations
        """
        if not self.is_fitted:
            raise ValueError("Explainer must be fitted before generating explanations")

        # Ensure transaction is 2D
        if transaction.ndim == 1:
            transaction = transaction.reshape(1, -1)

        # Calculate SHAP values
        if self.model_type == "tree":
            shap_values = self.explainer.shap_values(transaction)
            # For binary classification, XGBoost returns SHAP values for positive class
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # Fraud class
        else:
            shap_values = self.explainer.shap_values(transaction)

        # Flatten if needed
        if shap_values.ndim > 1:
            shap_values = shap_values[0]

        # Create feature importance ranking
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(len(shap_values))]

        # Create explanation dictionary
        feature_contributions = {}
        for name, value in zip(feature_names, shap_values):
            feature_contributions[name] = float(value)

        # Sort by absolute contribution
        sorted_contributions = sorted(
            feature_contributions.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )

        # Get top contributing features
        top_fraud_contributors = [(name, contrib) for name, contrib in sorted_contributions if contrib > 0][:5]
        top_normal_contributors = [(name, contrib) for name, contrib in sorted_contributions if contrib < 0][:5]

        # Calculate base value (expected model output)
        if hasattr(self.explainer, 'expected_value'):
            if isinstance(self.explainer.expected_value, np.ndarray):
                base_value = float(self.explainer.expected_value[1])  # Fraud class
            else:
                base_value = float(self.explainer.expected_value)
        else:
            base_value = 0.0

        # Model prediction
        fraud_probability = self.model.predict_proba(transaction)[0, 1]

        explanation = {
            'fraud_probability': float(fraud_probability),
            'base_value': base_value,
            'prediction_impact': float(np.sum(shap_values)),
            'total_shap_values': len(shap_values),
            'feature_contributions': feature_contributions,
            'top_fraud_contributors': top_fraud_contributors,
            'top_normal_contributors': top_normal_contributors,
            'explanation_summary': self._create_explanation_summary(
                top_fraud_contributors, top_normal_contributors, fraud_probability
            ),
            'timestamp': datetime.now().isoformat()
        }

        return explanation

    def explain_batch(
        self,
        transactions: np.ndarray,
        feature_names: Optional[List[str]] = None,
        max_samples: int = 50
    ) -> Dict[str, Any]:
        """
        Generate SHAP explanations for multiple transactions.

        Args:
            transactions: Array of transaction feature vectors
            feature_names: List of feature names
            max_samples: Maximum number of transactions to explain (for efficiency)

        Returns:
            Batch explanation results
        """
        if not self.is_fitted:
            raise ValueError("Explainer must be fitted before generating explanations")

        # Sample transactions if too many
        if len(transactions) > max_samples:
            indices = np.random.choice(len(transactions), max_samples, replace=False)
            sample_transactions = transactions[indices]
        else:
            sample_transactions = transactions

        print(f"🔄 Generating SHAP explanations for {len(sample_transactions)} transactions...")

        # Calculate SHAP values
        if self.model_type == "tree":
            shap_values = self.explainer.shap_values(sample_transactions)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # Fraud class
        else:
            shap_values = self.explainer.shap_values(sample_transactions)

        # Feature names
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(shap_values.shape[1])]

        # Calculate feature importance (mean absolute SHAP values)
        feature_importance = np.mean(np.abs(shap_values), axis=0)
        importance_ranking = dict(zip(feature_names, feature_importance))
        importance_ranking = dict(sorted(importance_ranking.items(), key=lambda x: x[1], reverse=True))

        # Predictions
        fraud_probabilities = self.model.predict_proba(sample_transactions)[:, 1]

        batch_explanation = {
            'n_transactions_explained': len(sample_transactions),
            'avg_fraud_probability': float(np.mean(fraud_probabilities)),
            'feature_importance_ranking': importance_ranking,
            'top_10_features': list(importance_ranking.keys())[:10],
            'shap_summary': {
                'mean_positive_impact': float(np.mean(shap_values[shap_values > 0])) if np.any(shap_values > 0) else 0.0,
                'mean_negative_impact': float(np.mean(shap_values[shap_values < 0])) if np.any(shap_values < 0) else 0.0,
                'total_features_analyzed': len(feature_names)
            }
        }

        print("✅ Batch SHAP explanations generated")

        return batch_explanation

    def create_explanation_plot(
        self,
        transaction: np.ndarray,
        feature_names: Optional[List[str]] = None,
        max_features: int = 10,
        save_path: Optional[str] = None
    ) -> str:
        """
        Create SHAP waterfall plot for single transaction explanation.

        Args:
            transaction: Single transaction feature vector
            feature_names: List of feature names
            max_features: Maximum number of features to show
            save_path: Path to save plot (optional)

        Returns:
            Path to saved plot or status message
        """
        if not self.is_fitted:
            raise ValueError("Explainer must be fitted before creating plots")

        # Ensure transaction is 2D
        if transaction.ndim == 1:
            transaction = transaction.reshape(1, -1)

        # Calculate SHAP values
        if self.model_type == "tree":
            shap_values = self.explainer.shap_values(transaction)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # Fraud class
        else:
            shap_values = self.explainer.shap_values(transaction)

        if shap_values.ndim > 1:
            shap_values = shap_values[0]

        # Feature names
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(len(shap_values))]

        try:
            # Create SHAP waterfall plot
            plt.figure(figsize=(10, 8))

            # Create explanation object for waterfall plot
            if hasattr(self.explainer, 'expected_value'):
                expected_value = self.explainer.expected_value
                if isinstance(expected_value, np.ndarray):
                    expected_value = expected_value[1]  # Fraud class
            else:
                expected_value = 0.0

            # Sort features by absolute SHAP value
            feature_importance = list(zip(feature_names, shap_values, transaction[0]))
            feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)

            # Take top features
            top_features = feature_importance[:max_features]

            # Create manual waterfall plot
            feature_names_plot = [f[0] for f in top_features]
            shap_values_plot = [f[1] for f in top_features]

            # Create horizontal bar plot
            colors = ['red' if val > 0 else 'blue' for val in shap_values_plot]
            y_pos = np.arange(len(feature_names_plot))

            plt.barh(y_pos, shap_values_plot, color=colors, alpha=0.7)
            plt.yticks(y_pos, feature_names_plot)
            plt.xlabel('SHAP Value (Impact on Prediction)')
            plt.title('Feature Contributions to Fraud Prediction')
            plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)

            # Add prediction info
            fraud_prob = self.model.predict_proba(transaction)[0, 1]
            plt.text(0.02, 0.98, f'Fraud Probability: {fraud_prob:.3f}',
                    transform=plt.gca().transAxes, fontsize=12,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.5),
                    verticalalignment='top')

            plt.tight_layout()

            # Save plot if path provided
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
                return save_path
            else:
                plt.show()
                return "Plot displayed successfully"

        except Exception as e:
            return f"Error creating plot: {str(e)}"

    def get_global_feature_importance(self) -> Dict[str, float]:
        """Get global feature importance from SHAP values."""
        if not self.is_fitted or self.background_data is None:
            raise ValueError("Explainer must be fitted with background data")

        # Calculate SHAP values for background data
        if self.model_type == "tree":
            shap_values = self.explainer.shap_values(self.background_data)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # Fraud class
        else:
            shap_values = self.explainer.shap_values(self.background_data)

        # Calculate mean absolute SHAP values
        feature_importance = np.mean(np.abs(shap_values), axis=0)

        return feature_importance.tolist()

    def _create_explanation_summary(
        self,
        fraud_contributors: List[tuple],
        normal_contributors: List[tuple],
        fraud_probability: float
    ) -> str:
        """Create human-readable explanation summary."""
        risk_level = "HIGH" if fraud_probability >= 0.8 else "MEDIUM" if fraud_probability >= 0.5 else "LOW"

        summary = f"This transaction has a {risk_level} fraud risk ({fraud_probability:.1%} probability). "

        if fraud_contributors:
            top_fraud_feature = fraud_contributors[0][0]
            summary += f"The main fraud indicator is {top_fraud_feature}. "

        if normal_contributors:
            top_normal_feature = normal_contributors[0][0]
            summary += f"However, {top_normal_feature} suggests normal behavior. "

        return summary

    def save_explainer(self, filepath: str) -> None:
        """Save fitted explainer to disk."""
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted explainer")

        import joblib
        explainer_data = {
            'explainer': self.explainer,
            'background_data': self.background_data,
            'model_type': self.model_type,
            'is_fitted': self.is_fitted
        }

        joblib.dump(explainer_data, filepath)
        print(f"✅ SHAP explainer saved to {filepath}")

    def load_explainer(self, filepath: str) -> None:
        """Load fitted explainer from disk."""
        import joblib
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Explainer file not found: {filepath}")

        explainer_data = joblib.load(filepath)
        self.explainer = explainer_data['explainer']
        self.background_data = explainer_data['background_data']
        self.model_type = explainer_data['model_type']
        self.is_fitted = explainer_data['is_fitted']

        print(f"✅ SHAP explainer loaded from {filepath}")


if __name__ == "__main__":
    # Demo usage
    print("🔧 SHAP Explainer Demo")
    print("✅ Fraud explainer utilities loaded!")
    print("🎯 Ready for model interpretability and compliance!")