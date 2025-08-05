"""
Unit tests for alpha generation module
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from alpha_generation import AlphaGenerator


class TestAlphaGenerator:
    """Test cases for AlphaGenerator class"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample market data for testing"""
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        data = pd.DataFrame({
            'open': np.random.uniform(100, 200, 100),
            'high': np.random.uniform(100, 200, 100),
            'low': np.random.uniform(100, 200, 100),
            'close': np.random.uniform(100, 200, 100),
            'volume': np.random.randint(1000, 10000, 100)
        }, index=dates)
        
        # Ensure high >= close >= low
        data['high'] = np.maximum(data['high'], data['close'])
        data['low'] = np.minimum(data['low'], data['close'])
        
        return data
    
    @pytest.fixture
    def alpha_generator(self):
        """Create AlphaGenerator instance for testing"""
        return AlphaGenerator()
    
    def test_initialization(self, alpha_generator):
        """Test AlphaGenerator initialization"""
        assert alpha_generator.scaler is not None
        assert alpha_generator.pca is None
        assert alpha_generator.elastic_net is None
        assert alpha_generator.lstm_model is None
        assert alpha_generator.feature_names == []
        assert alpha_generator.is_fitted is False
    
    def test_prepare_features(self, alpha_generator, sample_data):
        """Test feature preparation"""
        features_df = alpha_generator.prepare_features(sample_data)
        
        # Check that features were created
        assert len(features_df) > 0
        assert 'returns' in features_df.columns
        assert 'log_returns' in features_df.columns
        assert 'volatility' in features_df.columns
        assert 'sma_20' in features_df.columns
        assert 'rsi' in features_df.columns
        
        # Check that NaN values were removed
        assert not features_df.isnull().any().any()
    
    def test_calculate_rsi(self, alpha_generator):
        """Test RSI calculation"""
        prices = pd.Series([100, 101, 102, 101, 100, 99, 98, 97, 98, 99, 100])
        rsi = alpha_generator._calculate_rsi(prices, period=5)
        
        assert len(rsi) == len(prices)
        assert not rsi.isnull().all()  # Should have some non-null values
        assert (rsi >= 0).all() and (rsi <= 100).all()  # RSI should be between 0 and 100
    
    def test_calculate_macd(self, alpha_generator):
        """Test MACD calculation"""
        prices = pd.Series([100, 101, 102, 101, 100, 99, 98, 97, 98, 99, 100])
        macd = alpha_generator._calculate_macd(prices)
        
        assert len(macd) == len(prices)
        assert not macd.isnull().all()
    
    def test_calculate_bollinger_bands(self, alpha_generator):
        """Test Bollinger Bands calculation"""
        prices = pd.Series([100, 101, 102, 101, 100, 99, 98, 97, 98, 99, 100])
        upper, lower = alpha_generator._calculate_bollinger_bands(prices)
        
        assert len(upper) == len(prices)
        assert len(lower) == len(prices)
        assert (upper >= lower).all()  # Upper band should be >= lower band
    
    @patch('alpha_generation.PCA')
    @patch('alpha_generation.ElasticNet')
    @patch('alpha_generation.tf.keras.models.Sequential')
    def test_fit_method(self, mock_lstm, mock_elastic_net, mock_pca, alpha_generator, sample_data):
        """Test the fit method with mocked dependencies"""
        # Mock the dependencies
        mock_pca_instance = MagicMock()
        mock_pca_instance.fit_transform.return_value = np.random.randn(50, 10)
        mock_pca_instance.explained_variance_ratio_ = np.array([0.3, 0.2, 0.1, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05])
        mock_pca.return_value = mock_pca_instance
        
        mock_elastic_net_instance = MagicMock()
        mock_elastic_net_instance.fit.return_value = None
        mock_elastic_net_instance.predict.return_value = np.random.randn(50)
        mock_elastic_net.return_value = mock_elastic_net_instance
        
        mock_lstm_instance = MagicMock()
        mock_lstm_instance.fit.return_value = MagicMock()
        mock_lstm_instance.predict.return_value = np.random.randn(50, 1)
        mock_lstm.return_value = mock_lstm_instance
        
        # Test the fit method
        result = alpha_generator.fit(sample_data, target_col='close')
        
        assert result is alpha_generator
        assert alpha_generator.is_fitted is True
    
    def test_predict_without_fitting(self, alpha_generator, sample_data):
        """Test that predict raises error when not fitted"""
        with pytest.raises(ValueError, match="Model must be fitted before making predictions"):
            alpha_generator.predict(sample_data)
    
    @patch('alpha_generation.joblib.dump')
    def test_save_model(self, mock_dump, alpha_generator):
        """Test model saving"""
        alpha_generator.save_model("test_model.pkl")
        mock_dump.assert_called_once()
    
    @patch('alpha_generation.joblib.load')
    def test_load_model(self, mock_load, alpha_generator):
        """Test model loading"""
        mock_data = {
            'scaler': MagicMock(),
            'pca': MagicMock(),
            'elastic_net': MagicMock(),
            'lstm_model': MagicMock(),
            'feature_names': ['feature1', 'feature2'],
            'config': {},
            'performance_metrics': {},
            'is_fitted': True
        }
        mock_load.return_value = mock_data
        
        alpha_generator.load_model("test_model.pkl")
        mock_load.assert_called_once_with("test_model.pkl")


if __name__ == "__main__":
    pytest.main([__file__]) 