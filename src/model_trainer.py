import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class ModelTrainer:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.cv_results = {}
        
    def initialize_models(self):
        """Initialize multiple ML algorithms"""
        models = {
            'LGBM_Baseline': LGBMClassifier(
                random_state=self.random_state, verbose=-1, n_estimators=150,
                max_depth=6, learning_rate=0.1, class_weight='balanced'
            ),
            'LGBM_Culture_Aware': LGBMClassifier(
                random_state=self.random_state, verbose=-1, n_estimators=150,
                max_depth=6, learning_rate=0.1, class_weight='balanced'
            ),
            'RF_Baseline': RandomForestClassifier(
                random_state=self.random_state, n_estimators=100,
                max_depth=6, class_weight='balanced'
            ),
            'RF_Culture_Aware': RandomForestClassifier(
                random_state=self.random_state, n_estimators=100,
                max_depth=6, class_weight='balanced'
            ),
            'LR_Baseline': LogisticRegression(
                random_state=self.random_state, max_iter=1000,
                class_weight='balanced', penalty='l2', C=1.0
            ),
            'LR_Culture_Aware': LogisticRegression(
                random_state=self.random_state, max_iter=1000,
                class_weight='balanced', penalty='l2', C=1.0
            )
        }
        
        print("🤖 MODELS INITIALIZED:")
        for name, model in models.items():
            print(f"  • {name}: {type(model).__name__}")
            
        return models
    
    def cross_validate_models(self, models, feature_sets, X, y):
        """Perform comprehensive cross-validation"""
        cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        cv_results = {}
        
        print("🔄 CROSS-VALIDATION RESULTS (F1 Macro):")
        print("-" * 50)
        
        for model_name, model in models.items():
            # Determine which feature set to use
            if 'Baseline' in model_name:
                X_data = X[feature_sets['A_standard']]
            else:  # Culture-aware models
                X_data = X[feature_sets['B_culture_aware']]
            
            cv_scores = cross_val_score(model, X_data, y, cv=cv_strategy,
                                      scoring='f1_macro', n_jobs=-1)
            
            cv_results[model_name] = {
                'mean_f1': cv_scores.mean(),
                'std_f1': cv_scores.std(),
                'all_scores': cv_scores
            }
            
            print(f"{model_name:20} : {cv_scores.mean():.3f} ± {cv_scores.std() * 2:.3f}")
        
        # Create comparison dataframe
        cv_comparison = pd.DataFrame({
            'Model': list(cv_results.keys()),
            'CV_Mean_F1': [results['mean_f1'] for results in cv_results.values()],
            'CV_Std_F1': [results['std_f1'] for results in cv_results.values()]
        }).sort_values('CV_Mean_F1', ascending=False)
        
        print("\n📊 CROSS-VALIDATION RANKING:")
        print(cv_comparison.round(4))
        
        self.cv_results = cv_results
        return cv_comparison
    
    def train_selected_models(self, models, feature_sets, X, y, test_size=0.3):
        """Train final models on train-test split"""
        # Prepare feature sets
        X_A = X[feature_sets['A_standard']]
        X_B = X[feature_sets['B_culture_aware']]
        
        # Split data
        X_train_A, X_test_A, y_train, y_test = train_test_split(
            X_A, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        # Get same indices for culture-aware features
        train_indices = X_train_A.index
        test_indices = X_test_A.index
        X_train_B = X_B.loc[train_indices]
        X_test_B = X_B.loc[test_indices]
        
        print(f"📊 DATA SPLITS:")
        print(f"  Training set: {X_train_A.shape[0]} samples")
        print(f"  Test set: {X_test_A.shape[0]} samples")
        
        # Select best models based on CV
        baseline_models = {k: v for k, v in models.items() if 'Baseline' in k}
        culture_models = {k: v for k, v in models.items() if 'Culture_Aware' in k}
        
        best_baseline_name = max(baseline_models.keys(), 
                               key=lambda x: self.cv_results[x]['mean_f1'])
        best_culture_name = max(culture_models.keys(), 
                              key=lambda x: self.cv_results[x]['mean_f1'])
        
        print(f"🎯 SELECTED MODELS:")
        print(f"  Baseline: {best_baseline_name}")
        print(f"  Culture-Aware: {best_culture_name}")
        
        # Train models
        model_baseline = models[best_baseline_name]
        model_culture = models[best_culture_name]
        
        model_baseline.fit(X_train_A, y_train)
        model_culture.fit(X_train_B, y_train)
        
        # Make predictions
        y_pred_A = model_baseline.predict(X_test_A)
        y_prob_A = model_baseline.predict_proba(X_test_A)[:, 1]
        y_pred_B = model_culture.predict(X_test_B)
        y_prob_B = model_culture.predict_proba(X_test_B)[:, 1]
        
        results = {
            'models': {
                'baseline': model_baseline,
                'culture_aware': model_culture
            },
            'predictions': {
                'baseline': {'y_pred': y_pred_A, 'y_prob': y_prob_A},
                'culture_aware': {'y_pred': y_pred_B, 'y_prob': y_prob_B}
            },
            'data_splits': {
                'X_test_A': X_test_A, 'X_test_B': X_test_B, 'y_test': y_test,
                'test_indices': test_indices
            }
        }
        
        return results
    
    def evaluate_models(self, results):
        """Comprehensive model evaluation"""
        y_test = results['data_splits']['y_test']
        pred_baseline = results['predictions']['baseline']
        pred_culture = results['predictions']['culture_aware']
        
        def calculate_metrics(y_true, y_pred, y_prob, model_name):
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            auc_roc = roc_auc_score(y_true, y_prob)
            
            cm = confusion_matrix(y_true, y_pred)
            tn, fp, fn, tp = cm.ravel()
            
            return {
                'Model': model_name,
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1-Score': f1,
                'AUC-ROC': auc_roc,
                'True_Positive': tp,
                'True_Negative': tn,
                'False_Positive': fp,
                'False_Negative': fn
            }
        
        metrics_A = calculate_metrics(y_test, pred_baseline['y_pred'], 
                                    pred_baseline['y_prob'], "Baseline")
        metrics_B = calculate_metrics(y_test, pred_culture['y_pred'], 
                                    pred_culture['y_prob'], "Culture-Aware")
        
        performance_df = pd.DataFrame([metrics_A, metrics_B])
        
        print("📈 MODEL PERFORMANCE COMPARISON:")
        print(performance_df.round(4))
        
        return performance_df
