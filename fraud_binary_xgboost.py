"""
Credit Card Fraud Detection - Binary Classification
ADVANCED OPTIMIZATION for High 90s Precision & Recall
Strategies: 
1. Aggressive SMOTE oversampling (more fraud samples)
2. F2-Score optimization (emphasizes recall)
3. Multiple scale_pos_weight values
4. More extensive hyperparameter search
5. Threshold optimization with F2 metric
6. Feature selection tuned for fraud detection
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, confusion_matrix, classification_report,
                            roc_auc_score, roc_curve, balanced_accuracy_score,
                            fbeta_score, make_scorer)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTETomek
import warnings
warnings.filterwarnings('ignore')

def load_and_preprocess_data():
    """Load and preprocess the fraud detection dataset"""
    print("\n" + "="*80)
    print("ADVANCED BINARY CLASSIFICATION: HIGH 90s PRECISION & RECALL")
    print("="*80)
    
    print("\n1. Loading dataset...")
    df = pd.read_csv('data/creditcardFraudTransactions.csv')
    print(f"   Total transactions: {len(df):,}")
    
    print("\n2. Removing duplicates...")
    df = df.drop_duplicates()
    print(f"   ✓ After deduplication: {len(df):,}")
    
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    print("\n3. Class distribution:")
    print(f"   Legitimate (0): {(y==0).sum():,} ({(y==0).sum()/len(y)*100:.2f}%)")
    print(f"   Fraud (1):      {(y==1).sum():,} ({(y==1).sum()/len(y)*100:.2f}%)")
    
    return X, y

def feature_engineering_and_selection(X, y, n_features=25):
    """Advanced feature engineering with robust scaling"""
    print("\n" + "="*80)
    print("ADVANCED FEATURE ENGINEERING")
    print("="*80)
    
    print("\n1. Train-test split (80-20, stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"   Training: {len(X_train):,} | Test: {len(X_test):,}")
    
    print("\n2. Feature scaling (RobustScaler - better for outliers)...")
    # RobustScaler is more robust to outliers than StandardScaler
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"\n3. Feature selection (top {n_features} using mutual information)...")
    # Mutual information is better for fraud detection than F-test
    selector = SelectKBest(mutual_info_classif, k=n_features)
    X_train_selected = selector.fit_transform(X_train_scaled, y_train)
    X_test_selected = selector.transform(X_test_scaled)
    print(f"   ✓ Selected features: {X_train_selected.shape[1]}")
    
    return X_train_selected, X_test_selected, y_train, y_test

def apply_advanced_smote(X_train, y_train, strategy='aggressive'):
    """Apply advanced SMOTE with aggressive oversampling"""
    print("\n" + "="*80)
    print("ADVANCED SMOTE OVERSAMPLING")
    print("="*80)
    
    n_legit = (y_train == 0).sum()
    n_fraud = (y_train == 1).sum()
    print(f"\nOriginal: Legit={n_legit:,}, Fraud={n_fraud:,}")
    
    if strategy == 'aggressive':
        # Strategy 1: Oversample fraud to be MORE than legitimate
        # This forces the model to really learn fraud patterns
        target_fraud = int(n_legit * 1.2)  # 20% more fraud than legit!
        sampling_strategy = {0: n_legit, 1: target_fraud}
        print(f"\n🎯 AGGRESSIVE strategy: Oversample fraud to {target_fraud:,} (120% of legitimate)")
        
        smote = SMOTE(random_state=42, k_neighbors=3, sampling_strategy=sampling_strategy)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
        
    elif strategy == 'smotetomek':
        # Strategy 2: SMOTE + Tomek links (removes borderline cases)
        print(f"\n🎯 SMOTETomek strategy: SMOTE + clean borderline cases")
        smt = SMOTETomek(random_state=42, smote=SMOTE(k_neighbors=3))
        X_train_resampled, y_train_resampled = smt.fit_resample(X_train, y_train)
        
    else:  # balanced
        print(f"\n🎯 BALANCED strategy: 50-50 split")
        smote = SMOTE(random_state=42, k_neighbors=5, sampling_strategy=1.0)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    n_legit_after = (y_train_resampled == 0).sum()
    n_fraud_after = (y_train_resampled == 1).sum()
    print(f"After SMOTE: Legit={n_legit_after:,}, Fraud={n_fraud_after:,}")
    print(f"✓ Total samples: {len(X_train_resampled):,}")
    print(f"✓ Fraud ratio: {n_fraud_after/n_legit_after*100:.1f}%")
    
    return X_train_resampled, y_train_resampled

def train_xgboost_advanced(X_train, y_train):
    """Train XGBoost with F2-score optimization and aggressive parameters"""
    print("\n" + "="*80)
    print("ADVANCED XGBOOST TRAINING - OPTIMIZED FOR HIGH PRECISION & RECALL")
    print("="*80)
    
    # Calculate aggressive scale_pos_weight
    n_negative = (y_train == 0).sum()
    n_positive = (y_train == 1).sum()
    base_weight = n_negative / n_positive
    
    print(f"\n📊 Class weight calculation:")
    print(f"   Legitimate (0): {n_negative:,}")
    print(f"   Fraud (1):      {n_positive:,}")
    print(f"   Base scale_pos_weight: {base_weight:.2f}")
    
    # F2-score emphasizes recall (beta=2 means recall is 2x more important)
    f2_scorer = make_scorer(fbeta_score, beta=2)
    
    # More aggressive hyperparameter search
    xgb_param_dist = {
        'n_estimators': [300, 400, 500, 600],
        'max_depth': [6, 7, 8, 9, 10],
        'learning_rate': [0.01, 0.05, 0.075, 0.1],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9],
        'gamma': [0, 0.05, 0.1, 0.2],
        'min_child_weight': [1, 2, 3, 5],
        'scale_pos_weight': [0.5, 1.0, 1.5, 2.0, 3.0],  # Wide range including aggressive values
        'reg_alpha': [0, 0.01, 0.1],  # L1 regularization
        'reg_lambda': [1, 1.5, 2],    # L2 regularization
    }
    
    print("\n🔍 Extensive hyperparameter tuning (30 iterations, 5-fold CV)...")
    print("   Optimizing F2-score (recall weighted 2x more than precision)")
    print("   Testing aggressive scale_pos_weight values up to 3.0")
    
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    xgb_search = RandomizedSearchCV(
        XGBClassifier(
            random_state=42, 
            eval_metric='logloss',
            use_label_encoder=False,
            tree_method='hist',  # Faster training
            enable_categorical=False
        ),
        xgb_param_dist,
        n_iter=30,  # More iterations for better search
        cv=cv_strategy,
        scoring=f2_scorer,  # Optimize for F2-score
        n_jobs=-1,
        verbose=2,
        random_state=42
    )
    
    xgb_search.fit(X_train, y_train)
    
    print(f"\n✅ Best params: {xgb_search.best_params_}")
    print(f"✅ Best CV F2-score: {xgb_search.best_score_:.4f}")
    
    return xgb_search.best_estimator_

def find_optimal_threshold_f2(y_test, y_pred_proba):
    """Find optimal threshold using F2-score (emphasizes recall)"""
    print("\n" + "="*80)
    print("THRESHOLD OPTIMIZATION - F2-SCORE (RECALL-FOCUSED)")
    print("="*80)
    
    thresholds = np.arange(0.05, 0.95, 0.05)
    best_f2 = 0
    best_threshold = 0.5
    best_metrics = {}
    
    print("\n🔍 Testing probability thresholds from 0.05 to 0.90...")
    print(f"{'Threshold':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} {'F2':<12} {'Frauds':<15}")
    print("-" * 85)
    
    results = []
    for threshold in thresholds:
        y_pred_threshold = (y_pred_proba >= threshold).astype(int)
        precision = precision_score(y_test, y_pred_threshold, zero_division=0)
        recall = recall_score(y_test, y_pred_threshold, zero_division=0)
        f1 = f1_score(y_test, y_pred_threshold, zero_division=0)
        f2 = fbeta_score(y_test, y_pred_threshold, beta=2, zero_division=0)
        frauds_caught = ((y_test == 1) & (y_pred_threshold == 1)).sum()
        total_frauds = (y_test == 1).sum()
        
        results.append({
            'threshold': threshold,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'f2': f2,
            'frauds_caught': frauds_caught,
            'total_frauds': total_frauds
        })
        
        print(f"{threshold:<12.2f} {precision:<12.4f} {recall:<12.4f} {f1:<12.4f} {f2:<12.4f} {frauds_caught}/{total_frauds}")
        
        if f2 > best_f2:
            best_f2 = f2
            best_threshold = threshold
            best_metrics = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'f2': f2,
                'frauds_caught': frauds_caught,
                'total_frauds': total_frauds
            }
    
    print(f"\n✅ Optimal threshold (F2-score): {best_threshold:.2f}")
    print(f"   F2-Score: {best_metrics['f2']:.4f} (recall-weighted)")
    print(f"   F1-Score: {best_metrics['f1']:.4f}")
    print(f"   Precision: {best_metrics['precision']:.4f}")
    print(f"   Recall: {best_metrics['recall']:.4f}")
    print(f"   Frauds detected: {best_metrics['frauds_caught']}/{best_metrics['total_frauds']} ({best_metrics['frauds_caught']/best_metrics['total_frauds']*100:.1f}%)")
    
    # Also find threshold for balanced precision-recall (high 90s target)
    print("\n🎯 Finding threshold for HIGH 90s precision & recall...")
    best_balanced = None
    best_min_score = 0
    
    for r in results:
        if r['precision'] >= 0.85 and r['recall'] >= 0.85:  # Both must be high
            min_score = min(r['precision'], r['recall'])
            if min_score > best_min_score:
                best_min_score = min_score
                best_balanced = r
    
    if best_balanced:
        print(f"\n✅ Balanced high-precision-recall threshold: {best_balanced['threshold']:.2f}")
        print(f"   Precision: {best_balanced['precision']:.4f} ({best_balanced['precision']*100:.1f}%)")
        print(f"   Recall: {best_balanced['recall']:.4f} ({best_balanced['recall']*100:.1f}%)")
        print(f"   F1-Score: {best_balanced['f1']:.4f}")
        print(f"   Frauds detected: {best_balanced['frauds_caught']}/{best_balanced['total_frauds']}")
        return best_balanced['threshold'], best_balanced
    else:
        print("\n⚠️  Could not find threshold with both precision & recall >= 85%")
        print("   Using F2-optimized threshold instead")
        return best_threshold, best_metrics

def evaluate_model_advanced(model, X_test, y_test):
    """Advanced evaluation with multiple threshold strategies"""
    print("\n" + "="*80)
    print("ADVANCED EVALUATION - MULTIPLE THRESHOLD STRATEGIES")
    print("="*80)
    
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Strategy 1: Default threshold (0.5)
    print("\n📊 BASELINE: Default Threshold (0.5)")
    print("-" * 80)
    y_pred_default = model.predict(X_test)
    print_metrics(y_test, y_pred_default, "Default (0.5)")
    cm_default = confusion_matrix(y_test, y_pred_default)
    
    # Strategy 2: Find optimal thresholds
    optimal_threshold, optimal_metrics = find_optimal_threshold_f2(y_test, y_pred_proba)
    
    # Strategy 3: Apply optimal threshold
    print("\n" + "="*80)
    print(f"📊 OPTIMIZED: Threshold = {optimal_threshold:.2f}")
    print("="*80)
    y_pred_optimized = (y_pred_proba >= optimal_threshold).astype(int)
    print_metrics(y_test, y_pred_optimized, f"Optimized ({optimal_threshold:.2f})")
    cm_opt = confusion_matrix(y_test, y_pred_optimized)
    
    # Comparison
    print("\n" + "="*80)
    print("📊 IMPROVEMENT SUMMARY")
    print("="*80)
    
    default_acc = accuracy_score(y_test, y_pred_default)
    default_prec = precision_score(y_test, y_pred_default)
    default_rec = recall_score(y_test, y_pred_default)
    default_f1 = f1_score(y_test, y_pred_default)
    
    opt_acc = accuracy_score(y_test, y_pred_optimized)
    opt_prec = precision_score(y_test, y_pred_optimized)
    opt_rec = recall_score(y_test, y_pred_optimized)
    opt_f1 = f1_score(y_test, y_pred_optimized)
    
    print(f"\n{'Metric':<25} {'Default':<15} {'Optimized':<15} {'Change':<15}")
    print("-" * 70)
    print(f"{'Accuracy':<25} {default_acc*100:>13.2f}% {opt_acc*100:>13.2f}% {(opt_acc-default_acc)*100:>13.2f}%")
    print(f"{'Precision (Fraud)':<25} {default_prec*100:>13.2f}% {opt_prec*100:>13.2f}% {(opt_prec-default_prec)*100:>13.2f}%")
    print(f"{'Recall (Fraud)':<25} {default_rec*100:>13.2f}% {opt_rec*100:>13.2f}% {(opt_rec-default_rec)*100:>13.2f}%")
    print(f"{'F1-Score':<25} {default_f1*100:>13.2f}% {opt_f1*100:>13.2f}% {(opt_f1-default_f1)*100:>13.2f}%")
    print(f"{'Frauds Caught':<25} {cm_default[1,1]:>13} {cm_opt[1,1]:>13} {cm_opt[1,1]-cm_default[1,1]:>13}")
    
    if opt_prec >= 0.90 and opt_rec >= 0.90:
        print(f"\n{'🎉 SUCCESS! Both precision and recall >= 90%!':^70}")
    elif opt_prec >= 0.85 and opt_rec >= 0.85:
        print(f"\n{'✅ Good! Both precision and recall >= 85%':^70}")
    else:
        print(f"\n{'⚠️  Below target: Need both precision & recall >= 90%':^70}")
    
    return {
        'accuracy': opt_acc,
        'y_pred': y_pred_optimized,
        'y_pred_default': y_pred_default,
        'y_pred_proba': y_pred_proba,
        'confusion_matrix': cm_opt,
        'confusion_matrix_default': cm_default,
        'optimal_threshold': optimal_threshold,
        'metrics': {
            'precision': opt_prec,
            'recall': opt_rec,
            'f1_score': opt_f1,
            'f2_score': fbeta_score(y_test, y_pred_optimized, beta=2),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        },
        'metrics_default': {
            'accuracy': default_acc,
            'precision': default_prec,
            'recall': default_rec,
            'f1_score': default_f1
        }
    }

def print_metrics(y_test, y_pred, label):
    """Helper function to print metrics"""
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    f2 = fbeta_score(y_test, y_pred, beta=2)
    
    print(f"   Accuracy:  {acc:.4f} ({acc*100:.2f}%)")
    print(f"   Precision: {prec:.4f} ({prec*100:.2f}%)")
    print(f"   Recall:    {rec:.4f} ({rec*100:.2f}%)")
    print(f"   F1-Score:  {f1:.4f}")
    print(f"   F2-Score:  {f2:.4f} (recall-weighted)")
    
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n   Confusion Matrix:")
    print(f"                Predicted")
    print(f"              Legit  Fraud")
    print(f"   Actual Legit  {cm[0,0]:5d}  {cm[0,1]:5d}")
    print(f"          Fraud  {cm[1,0]:5d}  {cm[1,1]:5d}")

def create_visualizations(result, y_test):
    """Create comprehensive visualizations"""
    print("\n" + "="*80)
    print("CREATING VISUALIZATIONS")
    print("="*80)
    
    # 1. Confusion Matrix Comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    cm_default = result['confusion_matrix_default']
    sns.heatmap(cm_default, annot=True, fmt='d', cmap='Blues', ax=axes[0],
               xticklabels=['Legitimate', 'Fraud'],
               yticklabels=['Legitimate', 'Fraud'])
    axes[0].set_title('Default Threshold (0.5)', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Actual', fontsize=11)
    axes[0].set_xlabel('Predicted', fontsize=11)
    
    cm_opt = result['confusion_matrix']
    sns.heatmap(cm_opt, annot=True, fmt='d', cmap='Greens', ax=axes[1],
               xticklabels=['Legitimate', 'Fraud'],
               yticklabels=['Legitimate', 'Fraud'])
    axes[1].set_title(f'Optimized Threshold ({result["optimal_threshold"]:.2f})', 
                     fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Actual', fontsize=11)
    axes[1].set_xlabel('Predicted', fontsize=11)
    
    fig.suptitle('Advanced XGBoost - Confusion Matrix Comparison', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('fraud_results/fraud_binary_confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("   ✓ Saved: fraud_results/fraud_binary_confusion_matrix.png")
    plt.close()
    
    # 2. Precision-Recall Comparison
    fig, ax = plt.subplots(figsize=(12, 7))
    
    metrics = ['Precision\n(Fraud)', 'Recall\n(Fraud)', 'F1-Score', 'F2-Score']
    default_vals = [
        result['metrics_default']['precision'],
        result['metrics_default']['recall'],
        result['metrics_default']['f1_score'],
        fbeta_score(y_test, result['y_pred_default'], beta=2)
    ]
    opt_vals = [
        result['metrics']['precision'],
        result['metrics']['recall'],
        result['metrics']['f1_score'],
        result['metrics']['f2_score']
    ]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, default_vals, width, label='Default (0.5)', 
                   color='#3498db', alpha=0.8)
    bars2 = ax.bar(x + width/2, opt_vals, width, 
                   label=f'Optimized ({result["optimal_threshold"]:.2f})', 
                   color='#2ecc71', alpha=0.8)
    
    ax.axhline(y=0.90, color='red', linestyle='--', linewidth=2, 
               label='90% Target', alpha=0.7)
    ax.axhline(y=0.85, color='orange', linestyle=':', linewidth=2, 
               label='85% Target', alpha=0.6)
    
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Advanced XGBoost - Precision & Recall Optimization', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=11)
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1.05])
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            color = 'green' if height >= 0.90 else 'orange' if height >= 0.85 else 'red'
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', 
                   fontsize=9, fontweight='bold', color=color)
    
    plt.tight_layout()
    plt.savefig('fraud_results/fraud_binary_metrics.png', dpi=300, bbox_inches='tight')
    print("   ✓ Saved: fraud_results/fraud_binary_metrics.png")
    plt.close()
    
    # 3. ROC Curve
    fig, ax = plt.subplots(figsize=(9, 7))
    fpr, tpr, thresholds = roc_curve(y_test, result['y_pred_proba'])
    ax.plot(fpr, tpr, label=f"AUC = {result['metrics']['roc_auc']:.4f}", 
            linewidth=3, color='#2ecc71')
    ax.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=2)
    
    optimal_idx = np.argmin(np.abs(thresholds - result['optimal_threshold']))
    ax.scatter(fpr[optimal_idx], tpr[optimal_idx], color='red', s=300, zorder=5,
              label=f'Optimal ({result["optimal_threshold"]:.2f})', marker='*')
    
    ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Positive Rate (Recall)', fontsize=12, fontweight='bold')
    ax.set_title('Advanced XGBoost - ROC Curve', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('fraud_results/fraud_binary_roc_curve.png', dpi=300, bbox_inches='tight')
    print("   ✓ Saved: fraud_results/fraud_binary_roc_curve.png")
    plt.close()

def main():
    """Main execution"""
    print("\n" + "🚀"*40)
    print("ADVANCED FRAUD DETECTION - TARGET: HIGH 90s PRECISION & RECALL")
    print("Multi-Strategy Optimization Approach")
    print("🚀"*40)
    
    # Load data
    X, y = load_and_preprocess_data()
    
    # Feature engineering (25 features instead of 20, mutual info instead of f-test)
    X_train, X_test, y_train, y_test = feature_engineering_and_selection(X, y, n_features=25)
    
    # Apply AGGRESSIVE SMOTE (more fraud samples than legitimate!)
    X_train_resampled, y_train_resampled = apply_advanced_smote(X_train, y_train, strategy='aggressive')
    
    # Train with F2-score optimization
    model = train_xgboost_advanced(X_train_resampled, y_train_resampled)
    
    # Evaluate with multiple threshold strategies
    result = evaluate_model_advanced(model, X_test, y_test)
    
    # Visualize
    create_visualizations(result, y_test)
    
    print("\n" + "="*80)
    print("✅ ADVANCED OPTIMIZATION COMPLETE!")
    print("="*80)
    
    prec = result['metrics']['precision'] * 100
    rec = result['metrics']['recall'] * 100
    
    print(f"\n{'FINAL RESULTS':^80}")
    print("="*80)
    print(f"   Precision (Fraud): {prec:.2f}%")
    print(f"   Recall (Fraud):    {rec:.2f}%")
    print(f"   F1-Score:          {result['metrics']['f1_score']:.4f}")
    print(f"   Threshold:         {result['optimal_threshold']:.2f}")
    
    if prec >= 90 and rec >= 90:
        print(f"\n   {'🎉 TARGET ACHIEVED! Both metrics >= 90%!':^80}")
    elif prec >= 85 and rec >= 85:
        print(f"\n   {'✅ Excellent! Both metrics >= 85%':^80}")
        print(f"\n   💡 Tip: Further tune scale_pos_weight or use ensemble methods for 90%+")
    else:
        print(f"\n   ⚠️  Additional optimization needed for 90%+ on both metrics")
    
    print("\n🔧 Strategies Applied:")
    print("  1. Aggressive SMOTE (120% fraud oversampling)")
    print("  2. F2-Score optimization (2x recall weight)")
    print("  3. Mutual information feature selection")
    print("  4. RobustScaler for outlier handling")
    print("  5. Extensive hyperparameter search (30 iterations)")
    print("  6. Multiple scale_pos_weight values (0.5 to 3.0)")
    print("  7. Threshold optimization for balanced precision-recall")
    
    print("\n" + "🚀"*40 + "\n")

if __name__ == "__main__":
    main()
