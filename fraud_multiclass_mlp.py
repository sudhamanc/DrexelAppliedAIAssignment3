"""
Credit Card Fraud Detection - Multiclass Classification
Neural Network (MLP) ONLY with SMOTE
Risk Levels: Low-Risk Legit, High-Risk Legit, Small Fraud, Large Fraud
Target: 90%+ Accuracy
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, confusion_matrix, classification_report,
                            balanced_accuracy_score)
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

def load_and_create_risk_levels():
    """Load dataset and create 4 risk level classes"""
    print("\n" + "="*80)
    print("MULTICLASS CLASSIFICATION: RISK LEVEL DETECTION - MLP ONLY")
    print("="*80)
    
    print("\n1. Loading dataset...")
    df = pd.read_csv('data/creditcardFraudTransactions.csv')
    print(f"   Total transactions: {len(df):,}")
    
    print("\n2. Removing duplicates...")
    df = df.drop_duplicates()
    print(f"   ✓ After deduplication: {len(df):,}")
    
    print("\n3. Creating risk level classes...")
    amount_threshold = 50  # $50 threshold
    
    conditions = [
        (df['Class'] == 0) & (df['Amount'] < amount_threshold),   # Low-Risk Legitimate
        (df['Class'] == 0) & (df['Amount'] >= amount_threshold),  # High-Risk Legitimate
        (df['Class'] == 1) & (df['Amount'] < amount_threshold),   # Small Fraud
        (df['Class'] == 1) & (df['Amount'] >= amount_threshold)   # Large Fraud
    ]
    choices = [0, 1, 2, 3]
    df['RiskLevel'] = np.select(conditions, choices)
    
    print("\n4. Risk level distribution:")
    for level in range(4):
        count = (df['RiskLevel'] == level).sum()
        pct = count / len(df) * 100
        level_names = ['Low-Risk Legit (<$50)', 'High-Risk Legit (≥$50)', 
                      'Small Fraud (<$50)', 'Large Fraud (≥$50)']
        print(f"   Class {level} ({level_names[level]}): {count:,} ({pct:.2f}%)")
    
    X = df.drop(['Class', 'RiskLevel'], axis=1)
    y = df['RiskLevel']
    
    return X, y

def feature_engineering_and_selection(X, y, n_features=20):
    """Feature scaling and selection"""
    print("\n" + "="*80)
    print("FEATURE ENGINEERING")
    print("="*80)
    
    print("\n1. Train-test split (80-20, stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"   Training: {len(X_train):,} | Test: {len(X_test):,}")
    
    print("\n2. Feature scaling...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"\n3. Feature selection (top {n_features})...")
    selector = SelectKBest(f_classif, k=n_features)
    X_train_selected = selector.fit_transform(X_train_scaled, y_train)
    X_test_selected = selector.transform(X_test_scaled)
    print(f"   ✓ Selected features: {X_train_selected.shape[1]}")
    
    return X_train_selected, X_test_selected, y_train, y_test

def apply_smote(X_train, y_train):
    """Apply SMOTE for perfect 4-way class balance"""
    print("\n" + "="*80)
    print("SMOTE OVERSAMPLING (4-WAY BALANCE)")
    print("="*80)
    
    print("\nOriginal distribution:")
    for cls in range(4):
        print(f"   Class {cls}: {(y_train == cls).sum():,}")
    
    smote = SMOTE(random_state=42, k_neighbors=5, sampling_strategy='all')
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    print("\nAfter SMOTE:")
    for cls in range(4):
        print(f"   Class {cls}: {(y_train_resampled == cls).sum():,}")
    
    print(f"\n✓ Perfectly balanced! Total: {len(X_train_resampled):,}")
    
    return X_train_resampled, y_train_resampled

def train_mlp(X_train, y_train):
    """Train and tune Neural Network (MLP)"""
    print("\n" + "="*80)
    print("NEURAL NETWORK (MLP) TRAINING - TARGET: 90%+ ACCURACY")
    print("="*80)
    
    mlp_param_dist = {
        'hidden_layer_sizes': [(200, 100), (150, 100, 50), (200, 100, 50)],
        'activation': ['relu'],
        'alpha': [0.0001, 0.001],
        'learning_rate': ['adaptive'],
        'learning_rate_init': [0.001, 0.01],
        'max_iter': [500]
    }
    
    print("\nHyperparameter tuning (4 iterations, 3-fold CV)...")
    
    cv_strategy = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    mlp_search = RandomizedSearchCV(
        MLPClassifier(random_state=42, early_stopping=True, validation_fraction=0.1),
        mlp_param_dist,
        n_iter=4,
        cv=cv_strategy,
        scoring='f1_weighted',
        n_jobs=-1,
        verbose=1,
        random_state=42
    )
    
    mlp_search.fit(X_train, y_train)
    
    print(f"\n✓ Best params: {mlp_search.best_params_}")
    print(f"✓ Best CV F1-score (weighted): {mlp_search.best_score_:.4f}")
    
    return mlp_search.best_estimator_

def evaluate_model(model, X_test, y_test):
    """Evaluate model on test set"""
    print("\n" + "="*80)
    print("MLP EVALUATION ON TEST SET")
    print("="*80)
    
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    precision_weighted = precision_score(y_test, y_pred, average='weighted')
    recall_weighted = recall_score(y_test, y_pred, average='weighted')
    f1_weighted = f1_score(y_test, y_pred, average='weighted')
    
    print(f"\n📊 Overall Performance Metrics:")
    print(f"   Accuracy:          {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"   Balanced Accuracy: {balanced_acc:.4f} ({balanced_acc*100:.2f}%)")
    print(f"   Precision (weighted): {precision_weighted:.4f}")
    print(f"   Recall (weighted):    {recall_weighted:.4f}")
    print(f"   F1-Score (weighted):  {f1_weighted:.4f}")
    
    if accuracy >= 0.90:
        print(f"\n   ✅ TARGET ACHIEVED: {accuracy*100:.2f}% >= 90%")
    else:
        print(f"\n   ⚠️  Below target: {accuracy*100:.2f}% < 90%")
    
    print(f"\n📋 Per-Class Classification Report:")
    class_names = ['Low-Risk Legit', 'High-Risk Legit', 'Small Fraud', 'Large Fraud']
    print(classification_report(y_test, y_pred, target_names=class_names, digits=4))
    
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n📊 Confusion Matrix:")
    print(f"         Predicted")
    print(f"         Class 0  Class 1  Class 2  Class 3")
    for i in range(4):
        row_str = f"Actual {i}"
        for j in range(4):
            row_str += f"  {cm[i,j]:6d}"
        print(row_str)
    
    return {
        'accuracy': accuracy,
        'y_pred': y_pred,
        'confusion_matrix': cm,
        'metrics': {
            'balanced_accuracy': balanced_acc,
            'precision_weighted': precision_weighted,
            'recall_weighted': recall_weighted,
            'f1_weighted': f1_weighted
        }
    }

def create_visualizations(result, y_test):
    """Create visualizations"""
    print("\n" + "="*80)
    print("CREATING VISUALIZATIONS")
    print("="*80)
    
    class_names = ['Low-Risk\nLegit', 'High-Risk\nLegit', 'Small\nFraud', 'Large\nFraud']
    
    # 1. Confusion Matrix
    fig, ax = plt.subplots(figsize=(10, 8))
    cm = result['confusion_matrix']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
               xticklabels=class_names,
               yticklabels=class_names)
    ax.set_title('MLP - Multiclass Classification (4 Risk Levels)\nConfusion Matrix', 
                fontsize=14, fontweight='bold')
    ax.set_ylabel('Actual', fontsize=12)
    ax.set_xlabel('Predicted', fontsize=12)
    plt.tight_layout()
    plt.savefig('fraud_results/multiclass_confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("   ✓ Saved: fraud_results/multiclass_confusion_matrix.png")
    plt.close()
    
    # 2. Metrics Bar Chart
    metrics = result['metrics']
    fig, ax = plt.subplots(figsize=(10, 6))
    metric_names = ['Accuracy', 'Balanced Acc', 'Precision\n(weighted)', 
                   'Recall\n(weighted)', 'F1-Score\n(weighted)']
    values = [result['accuracy'], metrics['balanced_accuracy'], metrics['precision_weighted'], 
              metrics['recall_weighted'], metrics['f1_weighted']]
    
    bars = ax.bar(metric_names, values, color='#e74c3c', alpha=0.8)
    ax.axhline(y=0.90, color='green', linestyle='--', linewidth=2, label='90% Target', alpha=0.7)
    
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('MLP Multiclass Classification - Performance Metrics', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1.1])
    
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('fraud_results/multiclass_metrics.png', dpi=300, bbox_inches='tight')
    print("   ✓ Saved: fraud_results/multiclass_metrics.png")
    plt.close()
    
    # 3. Per-class Performance
    fig, ax = plt.subplots(figsize=(12, 6))
    
    precision_per_class = precision_score(y_test, result['y_pred'], average=None)
    recall_per_class = recall_score(y_test, result['y_pred'], average=None)
    f1_per_class = f1_score(y_test, result['y_pred'], average=None)
    
    x = np.arange(4)
    width = 0.25
    
    bars1 = ax.bar(x - width, precision_per_class, width, label='Precision', color='#3498db', alpha=0.8)
    bars2 = ax.bar(x, recall_per_class, width, label='Recall', color='#2ecc71', alpha=0.8)
    bars3 = ax.bar(x + width, f1_per_class, width, label='F1-Score', color='#e74c3c', alpha=0.8)
    
    ax.axhline(y=0.90, color='black', linestyle='--', linewidth=2, label='90% Target', alpha=0.5)
    
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('MLP Multiclass - Per-Class Performance', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1.1])
    
    plt.tight_layout()
    plt.savefig('fraud_results/multiclass_per_class.png', dpi=300, bbox_inches='tight')
    print("   ✓ Saved: fraud_results/multiclass_per_class.png")
    plt.close()

def main():
    """Main execution function"""
    print("\n" + "🎯"*40)
    print("CREDIT CARD FRAUD DETECTION - MULTICLASS CLASSIFICATION")
    print("Neural Network (MLP) ONLY - Target: 90%+ Accuracy")
    print("🎯"*40)
    
    # Load and create risk levels
    X, y = load_and_create_risk_levels()
    
    # Feature engineering
    X_train, X_test, y_train, y_test = feature_engineering_and_selection(X, y, n_features=20)
    
    # Apply SMOTE
    X_train_resampled, y_train_resampled = apply_smote(X_train, y_train)
    
    # Train MLP
    model = train_mlp(X_train_resampled, y_train_resampled)
    
    # Evaluate
    result = evaluate_model(model, X_test, y_test)
    
    # Visualize
    create_visualizations(result, y_test)
    
    print("\n" + "="*80)
    print("✅ MULTICLASS CLASSIFICATION COMPLETE!")
    print("="*80)
    print("\nGenerated files in fraud_results/ folder:")
    print("  - multiclass_confusion_matrix.png")
    print("  - multiclass_metrics.png")
    print("  - multiclass_per_class.png")
    
    print("\n" + "="*80)
    print("FINAL RESULT")
    print("="*80)
    acc = result['accuracy'] * 100
    status = "✅ TARGET ACHIEVED" if acc >= 90 else "⚠️  NEEDS IMPROVEMENT"
    print(f"MLP Accuracy: {acc:.2f}%  {status}")
    print("\n" + "🎯"*40 + "\n")

if __name__ == "__main__":
    main()
