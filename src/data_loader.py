import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import chi2_contingency
import warnings
warnings.filterwarnings('ignore')

class DataLoader:
    def __init__(self, random_state=42):
        self.random_state = random_state
        np.random.seed(random_state)
        
    def load_data(self, file_path='data/train.csv'):
        """Load and validate dataset"""
        try:
            df = pd.read_csv(file_path)
            print("✅ Dataset loaded successfully")
            return df
        except FileNotFoundError:
            print(f"❌ Dataset file not found: {file_path}")
            raise
            
    def validate_data_quality(self, df):
        """Comprehensive data quality assessment"""
        print("📊 DATA QUALITY ASSESSMENT")
        print("=" * 50)
        
        # Basic info
        print(f"Dataset shape: {df.shape}")
        print(f"Samples: {df.shape[0]}, Features: {df.shape[1]}")
        
        # Duplicates
        duplicates = df.duplicated().sum()
        print(f"Duplicate entries: {duplicates}")
        
        # Constant features
        constant_features = [col for col in df.columns if df[col].nunique() == 1]
        print(f"Constant features: {constant_features}")
        
        # Missing values
        missing_summary = pd.DataFrame({
            'Missing_Count': df.isnull().sum(),
            'Missing_Percentage': (df.isnull().sum() / len(df)) * 100,
            'Data_Type': df.dtypes
        })
        
        missing_data = missing_summary[missing_summary['Missing_Count'] > 0]
        print(f"Features with missing values: {len(missing_data)}")
        
        # Target analysis
        if 'Class/ASD' in df.columns:
            target_dist = df['Class/ASD'].value_counts()
            print(f"Target distribution:\n{target_dist}")
            print(f"ASD Prevalence: {df['Class/ASD'].mean():.2%}")
        
        return {
            'duplicates': duplicates,
            'constant_features': constant_features,
            'missing_summary': missing_summary,
            'target_distribution': target_dist if 'Class/ASD' in df.columns else None
        }
    
    def analyze_cultural_diversity(self, df):
        """Analyze cultural diversity in dataset"""
        diversity_metrics = {}
        
        if 'ethnicity' in df.columns:
            ethnic_diversity = df['ethnicity'].nunique()
            ethnic_counts = df['ethnicity'].value_counts()
            proportions = ethnic_counts / ethnic_counts.sum()
            shannon_diversity = -np.sum(proportions * np.log(proportions))
            
            diversity_metrics['ethnic_groups'] = ethnic_diversity
            diversity_metrics['ethnic_shannon'] = shannon_diversity
            
        if 'contry_of_res' in df.columns:
            country_diversity = df['contry_of_res'].nunique()
            diversity_metrics['countries'] = country_diversity
            
        print("🌍 CULTURAL DIVERSITY METRICS")
        print("=" * 50)
        for metric, value in diversity_metrics.items():
            print(f"{metric.replace('_', ' ').title()}: {value:.3f}")
            
        return diversity_metrics
