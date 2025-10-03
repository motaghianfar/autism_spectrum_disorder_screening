import pandas as pd
import numpy as np
from scipy import stats
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import warnings
warnings.filterwarnings('ignore')

class BiasAnalyzer:
    def __init__(self):
        self.bias_metrics = {}
        
    def analyze_cultural_bias(self, results, cultural_clusters, test_indices):
        """Analyze model performance across cultural clusters"""
        y_test = results['data_splits']['y_test']
        pred_baseline = results['predictions']['baseline']
        pred_culture = results['predictions']['culture_aware']
        
        # Get cultural clusters for test set
        test_cultural_clusters = cultural_clusters.loc[test_indices]
        
        # Analyze performance by cluster
        cluster_performance = []
        min_cluster_size = 5
        valid_clusters = test_cultural_clusters.value_counts()
        valid_clusters = valid_clusters[valid_clusters >= min_cluster_size].index
        
        print(f"🔍 ANALYZING {len(valid_clusters)} CULTURAL CLUSTERS")
        
        for cluster in valid_clusters:
            cluster_mask = test_cultural_clusters == cluster
            cluster_size = cluster_mask.sum()
            
            y_test_cluster = y_test[cluster_mask]
            y_pred_A_cluster = pred_baseline['y_pred'][cluster_mask]
            y_pred_B_cluster = pred_culture['y_pred'][cluster_mask]
            
            # Calculate metrics
            acc_A = accuracy_score(y_test_cluster, y_pred_A_cluster)
            acc_B = accuracy_score(y_test_cluster, y_pred_B_cluster)
            f1_A = f1_score(y_test_cluster, y_pred_A_cluster, zero_division=0)
            f1_B = f1_score(y_test_cluster, y_pred_B_cluster, zero_division=0)
            
            cluster_performance.append({
                'Cluster': cluster,
                'Samples': cluster_size,
                'Acc_A': acc_A,
                'Acc_B': acc_B,
                'F1_A': f1_A,
                'F1_B': f1_B,
                'Acc_Improvement': acc_B - acc_A,
                'F1_Improvement': f1_B - f1_A
            })
        
        performance_df = pd.DataFrame(cluster_performance)
        performance_df = performance_df.sort_values('Samples', ascending=False)
        
        print("📊 PERFORMANCE BY CULTURAL CLUSTER:")
        print(performance_df.round(4))
        
        return performance_df
    
    def calculate_bias_metrics(self, performance_df):
        """Calculate comprehensive bias metrics"""
        # Performance variance across clusters
        acc_variance_A = performance_df['Acc_A'].var()
        acc_variance_B = performance_df['Acc_B'].var()
        
        f1_variance_A = performance_df['F1_A'].var()
        f1_variance_B = performance_df['F1_B'].var()
        
        # Maximum performance disparity
        max_acc_disparity_A = performance_df['Acc_A'].max() - performance_df['Acc_A'].min()
        max_acc_disparity_B = performance_df['Acc_B'].max() - performance_df['Acc_B'].min()
        
        # Average improvement
        avg_acc_improvement = performance_df['Acc_Improvement'].mean()
        avg_f1_improvement = performance_df['F1_Improvement'].mean()
        
        # Number of clusters with improvement
        clusters_improved_acc = (performance_df['Acc_Improvement'] > 0).sum()
        clusters_improved_f1 = (performance_df['F1_Improvement'] > 0).sum()
        
        # Fairness ratio
        fairness_ratio_A = performance_df['Acc_A'].min() / performance_df['Acc_A'].max() if performance_df['Acc_A'].max() > 0 else 0
        fairness_ratio_B = performance_df['Acc_B'].min() / performance_df['Acc_B'].max() if performance_df['Acc_B'].max() > 0 else 0
        
        bias_metrics = {
            'Accuracy_Variance_A': acc_variance_A,
            'Accuracy_Variance_B': acc_variance_B,
            'Accuracy_Variance_Reduction_Pct': ((acc_variance_A - acc_variance_B) / acc_variance_A) * 100 if acc_variance_A > 0 else 0,
            'F1_Variance_A': f1_variance_A,
            'F1_Variance_B': f1_variance_B,
            'Max_Accuracy_Disparity_A': max_acc_disparity_A,
            'Max_Accuracy_Disparity_B': max_acc_disparity_B,
            'Avg_Accuracy_Improvement': avg_acc_improvement,
            'Avg_F1_Improvement': avg_f1_improvement,
            'Clusters_Improved_Accuracy': clusters_improved_acc,
            'Clusters_Improved_F1': clusters_improved_f1,
            'Fairness_Ratio_A': fairness_ratio_A,
            'Fairness_Ratio_B': fairness_ratio_B
        }
        
        print("🎯 BIAS REDUCTION ANALYSIS:")
        for metric, value in bias_metrics.items():
            if 'Pct' in metric:
                print(f"  {metric:35}: {value:+.1f}%")
            elif 'Improvement' in metric:
                print(f"  {metric:35}: {value:+.3f}")
            elif 'Ratio' in metric:
                print(f"  {metric:35}: {value:.3f}")
            else:
                print(f"  {metric:35}: {value:.4f}")
        
        self.bias_metrics = bias_metrics
        return bias_metrics
    
    def statistical_significance_testing(self, performance_df):
        """Perform statistical tests for bias reduction"""
        print("📊 STATISTICAL SIGNIFICANCE TESTING:")
        
        # Levene's test for variance equality
        if len(performance_df) >= 2:
            levene_stat, levene_p = stats.levene(performance_df['Acc_A'], performance_df['Acc_B'])
            print(f"  Levene's test for variance equality: W = {levene_stat:.3f}, p = {levene_p:.4f}")
        else:
            levene_stat, levene_p = np.nan, np.nan
            print("  Insufficient clusters for Levene's test")
        
        # Paired t-test for accuracy improvement
        if len(performance_df) >= 2:
            t_stat, p_value = stats.ttest_1samp(performance_df['Acc_Improvement'].dropna(), 0)
            print(f"  Paired t-test for accuracy improvement: t = {t_stat:.3f}, p = {p_value:.4f}")
            
            if p_value < 0.05:
                print("  ✅ Statistically significant overall improvement")
            else:
                print("  ⚠️  No statistically significant improvement")
        else:
            print("  Insufficient clusters for paired t-test")
        
        return {
            'levene_stat': levene_stat,
            'levene_p': levene_p,
            't_stat_improvement': t_stat if 't_stat' in locals() else np.nan,
            'p_value_improvement': p_value if 'p_value' in locals() else np.nan
        }
