import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib.ticker import PercentFormatter

class Visualizer:
    def __init__(self, style='seaborn-v0_8-whitegrid'):
        plt.style.use(style)
        sns.set_palette("husl")
        
        # Set publication-quality settings
        plt.rcParams.update({
            'font.family': 'serif',
            'font.serif': ['Times New Roman'],
            'font.size': 12,
            'axes.labelsize': 14,
            'axes.titlesize': 16,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'legend.fontsize': 11,
            'figure.titlesize': 18,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight',
            'savefig.format': 'png'
        })
    
    def plot_asd_prevalence(self, df, save_path='figures/figure1_asd_prevalence.png'):
        """Plot ASD prevalence by cultural cluster"""
        if 'cultural_cluster' not in df.columns or 'Class/ASD' not in df.columns:
            print("❌ Required columns not found for prevalence plot")
            return
        
        cluster_stats = df.groupby('cultural_cluster').agg({
            'Class/ASD': ['count', 'mean']
        }).round(4)
        cluster_stats.columns = ['count', 'prevalence']
        cluster_stats = cluster_stats.sort_values('prevalence', ascending=True)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bars = ax.barh(range(len(cluster_stats)), cluster_stats['prevalence'] * 100,
                      alpha=0.7, color='steelblue', edgecolor='navy', linewidth=1)
        
        ax.set_yticks(range(len(cluster_stats)))
        ax.set_yticklabels(cluster_stats.index)
        ax.set_xlabel('ASD Prevalence (%)', fontweight='bold')
        ax.set_title('ASD Prevalence by Cultural Cluster', fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        # Add sample sizes
        for i, (bar, n) in enumerate(zip(bars, cluster_stats['count'])):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_y() + bar.get_height() + 1,
                   f'n={n}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ Saved prevalence plot: {save_path}")
    
    def plot_model_performance(self, performance_df, bias_metrics, save_path='figures/figure2_model_performance.png'):
        """Plot model performance comparison and bias reduction"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Panel A: Performance comparison
        models = ['Baseline', 'Culture-Aware']
        accuracy = performance_df['Accuracy'].values
        f1_scores = performance_df['F1-Score'].values
        
        x = np.arange(len(models))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, accuracy, width, label='Accuracy',
                       color='lightcoral', alpha=0.8, edgecolor='darkred')
        bars2 = ax1.bar(x + width/2, f1_scores, width, label='F1-Score',
                       color='lightseagreen', alpha=0.8, edgecolor='darkgreen')
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
        
        ax1.set_xlabel('Model Type', fontweight='bold')
        ax1.set_ylabel('Score', fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(models)
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        ax1.set_ylim(0, 1.0)
        
        # Panel B: Bias reduction
        metrics = ['Variance\nReduction', 'Fairness\nImprovement']
        values = [
            bias_metrics['Accuracy_Variance_Reduction_Pct'],
            (bias_metrics['Fairness_Ratio_B'] - bias_metrics['Fairness_Ratio_A']) * 100
        ]
        
        colors = ['gold', 'lightblue']
        bars = ax2.bar(metrics, values, color=colors, alpha=0.8, edgecolor='black')
        
        ax2.set_ylabel('Improvement (%)', fontweight='bold')
        ax2.set_title('Bias Reduction Metrics', fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{value:+.1f}%', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ Saved performance plot: {save_path}")
    
    def plot_cluster_performance(self, cluster_performance_df, save_path='figures/figure3_cluster_performance.png'):
        """Plot performance across cultural clusters"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Sort by sample size
        cluster_df_sorted = cluster_performance_df.sort_values('Samples', ascending=True)
        
        # Panel A: Accuracy comparison
        y_pos = np.arange(len(cluster_df_sorted))
        width = 0.35
        
        bars1 = ax1.barh(y_pos - width/2, cluster_df_sorted['Acc_A'], width,
                        label='Baseline', alpha=0.7, color='lightblue')
        bars2 = ax1.barh(y_pos + width/2, cluster_df_sorted['Acc_B'], width,
                        label='Culture-Aware', alpha=0.7, color='lightcoral')
        
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(cluster_df_sorted['Cluster'])
        ax1.set_xlabel('Accuracy', fontweight='bold')
        ax1.set_title('Accuracy by Cultural Cluster', fontweight='bold')
        ax1.legend()
        ax1.grid(axis='x', alpha=0.3)
        
        # Panel B: Improvement
        colors = ['green' if x > 0 else 'red' for x in cluster_df_sorted['Acc_Improvement']]
        bars = ax2.barh(y_pos, cluster_df_sorted['Acc_Improvement'], color=colors, alpha=0.7)
        
        ax2.axvline(x=0, color='black', linestyle='-', alpha=0.8)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(cluster_df_sorted['Cluster'])
        ax2.set_xlabel('Accuracy Improvement', fontweight='bold')
        ax2.set_title('Culture-Aware Model Improvement', fontweight='bold')
        ax2.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for bar, improvement in zip(bars, cluster_df_sorted['Acc_Improvement']):
            width = bar.get_width()
            if abs(width) > 0.01:  # Only label significant improvements
                ax2.text(width + (0.01 if width > 0 else -0.01), bar.get_y() + bar.get_height()/2,
                        f'{width:+.3f}', va='center', 
                        ha='left' if width > 0 else 'right', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ Saved cluster performance plot: {save_path}")
