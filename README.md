# FinTechCo Fraud Detection Demo

A comprehensive demonstration of Claude Code's value for data science teams working on imbalanced fraud detection datasets.

## 🎯 Project Overview

This project showcases how Claude Code accelerates fraud detection workflows that typically take data science teams days to complete. We're using the Kaggle Credit Card Fraud Detection dataset to demonstrate real-world challenges with extreme class imbalance.

### Dataset Characteristics
- **Total Transactions**: 284,807
- **Fraudulent Transactions**: 492 (0.172%)
- **Imbalance Ratio**: 577:1
- **Features**: 28 PCA-transformed + Time + Amount

## 🚨 Key DS Pain Points Addressed

1. **Extreme Class Imbalance** - Traditional ML fails completely
2. **Misleading Accuracy Metrics** - 99.83% accuracy with 0% fraud detection
3. **Feature Engineering Complexity** - PCA features lack interpretability
4. **Business Cost Trade-offs** - Precision vs Recall impacts bottom line

## 📊 Project Structure

```
fintechco-fraud-detection-demo/
├── data/
│   └── creditcard.csv           # Kaggle dataset
├── notebooks/
│   ├── 01_initial_exploration.ipynb    # ✅ Comprehensive EDA
│   ├── 02_baseline_model.ipynb         # 🔄 Next: Baseline failures
│   ├── 03_advanced_techniques.ipynb    # 🔄 SMOTE + Advanced methods
│   └── 04_production_ready.ipynb       # 🔄 Final production system
├── src/                         # Utility functions
├── outputs/                     # Model artifacts and plots
├── README.md
└── pyproject.toml              # UV dependencies
```

## 🚀 Claude Code Value Proposition

### ⏱️ Time Savings
- **Traditional Approach**: 4-6 hours for comprehensive EDA
- **With Claude Code**: 10 minutes for publication-quality analysis

### 📈 Analysis Depth
- Complete class imbalance analysis with visualizations
- Feature importance ranking and business cost calculations
- Time-series pattern analysis
- Professional presentation-ready outputs

### 🎯 Business Focus
Analysis directly connects technical metrics to business impact, showing why accuracy is misleading for fraud detection.

## 🛠️ Setup Instructions

```bash
# Clone and setup
git clone [repo-url]
cd fintechco-fraud-detection-demo

# Install dependencies with UV
uv sync

# Setup Kaggle API (if dataset not included)
# 1. Download kaggle.json from Kaggle Account settings
# 2. Place in ~/.kaggle/kaggle.json
# 3. Set permissions: chmod 600 ~/.kaggle/kaggle.json

# Download dataset (if needed)
uv run kaggle datasets download -d mlg-ulb/creditcardfraud --path data --unzip

# Start Jupyter
uv run jupyter lab
```

## 📈 Milestones

- [x] **Milestone 1**: Initial EDA & Problem Diagnosis (10 min)
- [ ] **Milestone 2**: Baseline Model Failures (10 min)
- [ ] **Milestone 3**: Advanced Imbalanced Techniques (15 min)
- [ ] **Milestone 4**: Production-Ready Solution (10 min)

## 💰 Business Impact

Current "high accuracy" models that predict all transactions as normal:
- **Accuracy**: 99.83% (misleading!)
- **Fraud Detection**: 0% (business disaster)
- **Cost**: $14,700+ in missed fraud losses

## 🔗 Key Insights

1. **The Accuracy Trap**: High accuracy ≠ good fraud detection
2. **Feature Challenges**: PCA anonymization limits interpretability
3. **Business Costs**: Must optimize for $ impact, not accuracy
4. **Advanced Techniques Needed**: SMOTE, class weights, proper metrics

---

*Built with Claude Code to demonstrate AI-accelerated data science workflows*