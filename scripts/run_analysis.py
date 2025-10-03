#!/usr/bin/env python3
"""
Main script to run the complete culture-aware autism screening analysis pipeline.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data_loader import DataLoader
from src.preprocessor import DataPreprocessor
from src.model_trainer import ModelTrainer
from src.bias_analyzer import BiasAnalyzer
from src.visualization import Visualizer

def main():
    print("🎯 CULTURE-AWARE AUTISM SCREENING ANALYSIS")
    print("=" * 60)
    
    # Initialize components
    data_loader = DataLoader(random_state=42)
    preprocessor = DataPreprocessor()
    model_trainer = ModelTrainer(random_state=42)
    bias_analyzer = BiasAnalyzer()
    visualizer = Visualizer()
    
    # Step 1: Load and validate data
    print("\n📥 STEP 1: LOADING DATA")
    df = data_loader.load_data('data/train.csv')
    quality_report = data_loader.validate_data_quality(df)
    diversity_metrics = data_loader.analyze_cultural_diversity(df)
    
    # Step 2: Preprocess data
    print("\n🔧 STEP 2: PREPROCESSING DATA")
    df_processed = preprocessor.remove_constant_features(df)
    df_processed = preprocessor.create_cultural_clusters(df_processed)
    df_encoded = preprocessor.encode_categorical_features(df_processed)
    feature_sets = preprocessor.create_feature_sets(df_encoded)
    
    # Step 3: Train models
    print("\n🤖 STEP 3: TRAINING MODELS")
    models = model_trainer.initialize_models()
    
    # Prepare features and target
    X = df_encoded
    y = df_encoded['Class/ASD'] if 'Class/ASD' in df_encoded.columns else None
    
    if y is None:
        print("❌ Target variable 'Class/ASD' not found")
        return
    
    cv_results = model_trainer.cross_validate_models(models, feature_sets, X, y)
    training_results = model_trainer.train_selected_models(models, feature_sets, X, y)
    performance_df = model_trainer.evaluate_models(training_results)
    
    # Step 4: Bias analysis
    print("\n⚖️  STEP 4: BIAS ANALYSIS")
    cultural_clusters = df_encoded['cultural_cluster']
    test_indices = training_results['data_splits']['test_indices']
    
    cluster_performance = bias_analyzer.analyze_cultural_bias(
        training_results, cultural_clusters, test_indices
    )
    bias_metrics = bias_analyzer.calculate_bias_metrics(cluster_performance)
    stats_results = bias_analyzer.statistical_significance_testing(cluster_performance)
    
    # Step 5: Generate visualizations
    print("\n📊 STEP 5: GENERATING VISUALIZATIONS")
    os.makedirs('figures', exist_ok=True)
    
    visualizer.plot_asd_prevalence(df_encoded)
    visualizer.plot_model_performance(performance_df, bias_metrics)
    visualizer.plot_cluster_performance(cluster_performance)
    
    print("\n✅ ANALYSIS COMPLETED SUCCESSFULLY!")
    print(f"📁 Results saved in: figures/")
    
    return {
        'data_quality': quality_report,
        'diversity_metrics': diversity_metrics,
        'performance': performance_df,
        'bias_metrics': bias_metrics,
        'cluster_performance': cluster_performance
    }

if __name__ == "__main__":
    results = main()
