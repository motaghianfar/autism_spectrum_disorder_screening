"""
Culture-Aware Autism Screening: A modular framework for detecting and mitigating cultural bias in ASD screening tools.
"""

__version__ = "1.0.0"
__author__ = "Research Team"
__email__ = "research@example.com"

from .data_loader import DataLoader
from .preprocessor import DataPreprocessor
from .model_trainer import ModelTrainer
from .bias_analyzer import BiasAnalyzer
from .visualization import Visualizer

__all__ = [
    'DataLoader',
    'DataPreprocessor', 
    'ModelTrainer',
    'BiasAnalyzer',
    'Visualizer'
]
