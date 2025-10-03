import unittest
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_loader import DataLoader
from preprocessor import DataPreprocessor

class TestBasicFunctionality(unittest.TestCase):
    
    def test_data_loader_initialization(self):
        loader = DataLoader()
        self.assertEqual(loader.random_state, 42)
    
    def test_preprocessor_initialization(self):
        preprocessor = DataPreprocessor()
        self.assertIsInstance(preprocessor.label_encoders, dict)
    
    # Add more tests as needed

if __name__ == '__main__':
    unittest.main()
