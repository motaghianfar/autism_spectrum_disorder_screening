import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

class DataPreprocessor:
    def __init__(self):
        self.label_encoders = {}
        self.cultural_mappings = {}
        
    def remove_constant_features(self, df):
        """Remove constant features identified during EDA"""
        constant_features = [col for col in df.columns if df[col].nunique() == 1]
        if 'age_desc' in df.columns:
            constant_features.append('age_desc')
            
        df_processed = df.drop(constant_features, axis=1, errors='ignore')
        print(f"✅ Removed constant features: {constant_features}")
        return df_processed
    
    def create_cultural_clusters(self, df):
        """Enhanced cultural clustering based on Hofstede's dimensions"""
        df_processed = df.copy()
        
        # Country to region mapping
        def map_country_to_region(country):
            country = str(country).lower().strip()
            
            # Cultural clusters definition
            clusters = {
                'Anglo': ['united states', 'united kingdom', 'canada', 'australia', 'new zealand'],
                'European': ['austria', 'france', 'germany', 'netherlands', 'italy', 'spain'],
                'Latin American': ['brazil', 'argentina', 'mexico', 'chile', 'colombia'],
                'South Asian': ['india', 'sri lanka', 'pakistan', 'bangladesh'],
                'Middle Eastern': ['jordan', 'uae', 'saudi arabia', 'iran', 'iraq'],
                'East Asian': ['china', 'japan', 'south korea', 'taiwan', 'singapore'],
                'African': ['south africa', 'nigeria', 'kenya', 'ghana']
            }
            
            for cluster_name, countries in clusters.items():
                for country_name in countries:
                    if country_name in country:
                        return cluster_name
            return 'Other Region'
        
        # Apply region mapping
        if 'contry_of_res' in df_processed.columns:
            df_processed['cultural_region'] = df_processed['contry_of_res'].apply(map_country_to_region)
        
        # Enhanced cultural clustering
        def create_cultural_cluster(row):
            ethnicity = str(row.get('ethnicity', '')).strip()
            region = row.get('cultural_region', 'Other Region')
            
            meaningful_ethnicities = ['white-european', 'middle eastern', 'asian', 'black', 
                                    'south asian', 'pasifika', 'latino', 'hispanic', 'turkish']
            
            ethnicity_lower = ethnicity.lower()
            if ethnicity_lower in meaningful_ethnicities:
                return ethnicity_lower.title()
            else:
                return region
        
        if 'ethnicity' in df_processed.columns and 'cultural_region' in df_processed.columns:
            df_processed['cultural_cluster'] = df_processed.apply(create_cultural_cluster, axis=1)
            
            # Consolidate small clusters
            min_cluster_size = 20
            cluster_sizes = df_processed['cultural_cluster'].value_counts()
            small_clusters = cluster_sizes[cluster_sizes < min_cluster_size].index
            df_processed['cultural_cluster'] = df_processed['cultural_cluster'].apply(
                lambda x: 'Diverse' if x in small_clusters else x
            )
        
        return df_processed
    
    def encode_categorical_features(self, df, categorical_columns=None):
        """Encode categorical features using LabelEncoder"""
        if categorical_columns is None:
            categorical_columns = ['gender', 'jaundice', 'austim', 'cultural_cluster', 'cultural_region']
            categorical_columns = [col for col in categorical_columns if col in df.columns]
        
        df_encoded = df.copy()
        
        for col in categorical_columns:
            if col in df_encoded.columns:
                le = LabelEncoder()
                encoded_col = col + '_encoded'
                df_encoded[encoded_col] = le.fit_transform(df_encoded[col].astype(str))
                self.label_encoders[col] = le
                
                print(f"  {col}: {df_encoded[col].nunique()} categories → {df_encoded[encoded_col].nunique()} encoded values")
        
        return df_encoded
    
    def create_feature_sets(self, df):
        """Create different feature sets for experimentation"""
        # Base features
        feature_columns_A = ['A1_Score', 'A2_Score', 'A3_Score', 'A4_Score', 'A5_Score',
                           'A6_Score', 'A7_Score', 'A8_Score', 'A9_Score', 'A10_Score',
                           'age', 'gender', 'jaundice', 'austim']
        
        # Culture-aware features
        feature_columns_B = feature_columns_A + ['cultural_cluster'] if 'cultural_cluster' in df.columns else feature_columns_A
        
        # Comprehensive cultural features
        feature_columns_C = feature_columns_B + ['cultural_region'] if 'cultural_region' in df.columns else feature_columns_B
        
        # Convert to encoded versions
        categorical_columns = ['gender', 'jaundice', 'austim', 'cultural_cluster', 'cultural_region']
        
        features_A_encoded = [col for col in feature_columns_A if col not in categorical_columns] + \
                           [col + '_encoded' for col in feature_columns_A if col in categorical_columns and col + '_encoded' in df.columns]
        
        features_B_encoded = [col for col in feature_columns_B if col not in categorical_columns] + \
                           [col + '_encoded' for col in feature_columns_B if col in categorical_columns and col + '_encoded' in df.columns]
        
        features_C_encoded = [col for col in feature_columns_C if col not in categorical_columns] + \
                           [col + '_encoded' for col in feature_columns_C if col in categorical_columns and col + '_encoded' in df.columns]
        
        feature_sets = {
            'A_standard': features_A_encoded,
            'B_culture_aware': features_B_encoded,
            'C_comprehensive': features_C_encoded
        }
        
        print("🔧 FEATURE SETS CREATED:")
        for set_name, features in feature_sets.items():
            print(f"  {set_name}: {len(features)} features")
        
        return feature_sets
