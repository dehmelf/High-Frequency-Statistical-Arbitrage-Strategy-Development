"""
High-Frequency Statistical Arbitrage Alpha Generation Framework

This module implements a comprehensive alpha generation system combining:
- PCA for dimensionality reduction and factor modeling
- Elastic Net regression for sparse feature selection
- LSTM neural networks for time series forecasting
- Ensemble methods for robust signal generation
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Attention
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import joblib
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from loguru import logger
import warnings
warnings.filterwarnings('ignore')

class AlphaGenerator:
    """
    Main alpha generation class that orchestrates the entire pipeline
    """
    
    def __init__(self, config_path: str = "config/alpha_config.yaml"):
        """
        Initialize the alpha generator with configuration
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.scaler = StandardScaler()
        self.pca = None
        self.elastic_net = None
        self.lstm_model = None
        self.feature_names = []
        self.is_fitted = False
        
        # Performance tracking
        self.performance_metrics = {
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'total_return': 0.0,
            'volatility': 0.0,
            'win_rate': 0.0
        }
        
        logger.info("Alpha Generator initialized")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
            logger.info(f"Configuration loaded from {config_path}")
            return config
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found, using defaults")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Get default configuration"""
        return {
            'pca': {
                'n_components': 10,
                'explained_variance_threshold': 0.95
            },
            'elastic_net': {
                'alpha': 0.01,
                'l1_ratio': 0.5,
                'max_iter': 2000,
                'random_state': 42
            },
            'lstm': {
                'sequence_length': 60,
                'hidden_units': 128,
                'dropout_rate': 0.2,
                'learning_rate': 0.001,
                'batch_size': 32,
                'epochs': 100,
                'validation_split': 0.2
            },
            'ensemble': {
                'weights': [0.4, 0.3, 0.3]  # PCA, Elastic Net, LSTM weights
            },
            'features': {
                'price_features': ['open', 'high', 'low', 'close', 'volume'],
                'technical_indicators': ['sma_20', 'sma_50', 'rsi', 'macd', 'bollinger_upper', 'bollinger_lower'],
                'market_microstructure': ['bid_ask_spread', 'order_imbalance', 'volume_imbalance']
            }
        }
    
    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare and engineer features for alpha generation
        
        Args:
            data: Raw market data DataFrame
            
        Returns:
            DataFrame with engineered features
        """
        logger.info("Preparing features for alpha generation")
        
        features_df = data.copy()
        
        # Price-based features
        features_df['returns'] = features_df['close'].pct_change()
        features_df['log_returns'] = np.log(features_df['close'] / features_df['close'].shift(1))
        features_df['volatility'] = features_df['returns'].rolling(window=20).std()
        
        # Technical indicators
        features_df['sma_20'] = features_df['close'].rolling(window=20).mean()
        features_df['sma_50'] = features_df['close'].rolling(window=50).mean()
        features_df['rsi'] = self._calculate_rsi(features_df['close'])
        features_df['macd'] = self._calculate_macd(features_df['close'])
        
        # Bollinger Bands
        bb_upper, bb_lower = self._calculate_bollinger_bands(features_df['close'])
        features_df['bollinger_upper'] = bb_upper
        features_df['bollinger_lower'] = bb_lower
        features_df['bollinger_position'] = (features_df['close'] - bb_lower) / (bb_upper - bb_lower)
        
        # Market microstructure features
        if 'bid' in features_df.columns and 'ask' in features_df.columns:
            features_df['bid_ask_spread'] = features_df['ask'] - features_df['bid']
            features_df['spread_pct'] = features_df['bid_ask_spread'] / features_df['close']
        
        if 'bid_volume' in features_df.columns and 'ask_volume' in features_df.columns:
            features_df['order_imbalance'] = (features_df['bid_volume'] - features_df['ask_volume']) / \
                                           (features_df['bid_volume'] + features_df['ask_volume'])
        
        # Lagged features
        for lag in [1, 5, 10, 20]:
            features_df[f'returns_lag_{lag}'] = features_df['returns'].shift(lag)
            features_df[f'volume_lag_{lag}'] = features_df['volume'].shift(lag)
        
        # Rolling statistics
        for window in [5, 10, 20]:
            features_df[f'volatility_{window}'] = features_df['returns'].rolling(window=window).std()
            features_df[f'volume_ma_{window}'] = features_df['volume'].rolling(window=window).mean()
        
        # Remove NaN values
        features_df = features_df.dropna()
        
        self.feature_names = [col for col in features_df.columns if col not in ['timestamp', 'symbol']]
        logger.info(f"Generated {len(self.feature_names)} features")
        
        return features_df
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26) -> pd.Series:
        """Calculate MACD"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        return macd
    
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: int = 2) -> Tuple[pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        return upper_band, lower_band
    
    def fit_pca(self, X: np.ndarray) -> np.ndarray:
        """
        Fit PCA model for dimensionality reduction
        
        Args:
            X: Feature matrix
            
        Returns:
            Transformed features
        """
        logger.info("Fitting PCA model")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit PCA
        n_components = self.config['pca']['n_components']
        explained_variance_threshold = self.config['pca']['explained_variance_threshold']
        
        self.pca = PCA(n_components=n_components)
        X_pca = self.pca.fit_transform(X_scaled)
        
        # Check explained variance
        explained_variance = np.cumsum(self.pca.explained_variance_ratio_)
        logger.info(f"PCA explained variance: {explained_variance[-1]:.4f}")
        
        if explained_variance[-1] < explained_variance_threshold:
            logger.warning(f"PCA explained variance {explained_variance[-1]:.4f} below threshold {explained_variance_threshold}")
        
        return X_pca
    
    def fit_elastic_net(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Fit Elastic Net regression model
        
        Args:
            X: Feature matrix
            y: Target variable
            
        Returns:
            Predicted values
        """
        logger.info("Fitting Elastic Net model")
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Fit Elastic Net
        self.elastic_net = ElasticNet(
            alpha=self.config['elastic_net']['alpha'],
            l1_ratio=self.config['elastic_net']['l1_ratio'],
            max_iter=self.config['elastic_net']['max_iter'],
            random_state=self.config['elastic_net']['random_state']
        )
        
        self.elastic_net.fit(X_scaled, y)
        
        # Get predictions
        y_pred = self.elastic_net.predict(X_scaled)
        
        # Feature importance
        feature_importance = np.abs(self.elastic_net.coef_)
        important_features = np.where(feature_importance > 0)[0]
        logger.info(f"Elastic Net selected {len(important_features)} features out of {X.shape[1]}")
        
        return y_pred
    
    def build_lstm_model(self, input_shape: Tuple[int, int]) -> tf.keras.Model:
        """
        Build LSTM model for time series forecasting
        
        Args:
            input_shape: Shape of input data (sequence_length, n_features)
            
        Returns:
            Compiled LSTM model
        """
        model = Sequential([
            LSTM(
                units=self.config['lstm']['hidden_units'],
                return_sequences=True,
                input_shape=input_shape
            ),
            Dropout(self.config['lstm']['dropout_rate']),
            
            LSTM(
                units=self.config['lstm']['hidden_units'] // 2,
                return_sequences=False
            ),
            Dropout(self.config['lstm']['dropout_rate']),
            
            Dense(64, activation='relu'),
            Dropout(self.config['lstm']['dropout_rate']),
            
            Dense(1, activation='linear')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=self.config['lstm']['learning_rate']),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def prepare_lstm_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for LSTM model
        
        Args:
            X: Feature matrix
            y: Target variable
            
        Returns:
            Tuple of (X_sequences, y_targets)
        """
        sequence_length = self.config['lstm']['sequence_length']
        X_sequences = []
        y_targets = []
        
        for i in range(sequence_length, len(X)):
            X_sequences.append(X[i-sequence_length:i])
            y_targets.append(y[i])
        
        return np.array(X_sequences), np.array(y_targets)
    
    def fit_lstm(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Fit LSTM model
        
        Args:
            X: Feature matrix
            y: Target variable
            
        Returns:
            Predicted values
        """
        logger.info("Fitting LSTM model")
        
        # Prepare sequences
        X_sequences, y_targets = self.prepare_lstm_data(X, y)
        
        # Split data
        split_idx = int(len(X_sequences) * (1 - self.config['lstm']['validation_split']))
        X_train, X_val = X_sequences[:split_idx], X_sequences[split_idx:]
        y_train, y_val = y_targets[:split_idx], y_targets[split_idx:]
        
        # Build and fit model
        self.lstm_model = self.build_lstm_model((X_train.shape[1], X_train.shape[2]))
        
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.5, patience=5)
        ]
        
        history = self.lstm_model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.config['lstm']['epochs'],
            batch_size=self.config['lstm']['batch_size'],
            callbacks=callbacks,
            verbose=1
        )
        
        # Get predictions
        y_pred_sequences = self.lstm_model.predict(X_sequences)
        y_pred = np.full(len(y), np.nan)
        y_pred[self.config['lstm']['sequence_length']:] = y_pred_sequences.flatten()
        
        return y_pred
    
    def fit(self, data: pd.DataFrame, target_col: str = 'returns') -> 'AlphaGenerator':
        """
        Fit all models in the alpha generation pipeline
        
        Args:
            data: Market data DataFrame
            target_col: Target variable column name
            
        Returns:
            Self for method chaining
        """
        logger.info("Starting alpha generation pipeline fitting")
        
        # Prepare features
        features_df = self.prepare_features(data)
        
        # Prepare target
        y = features_df[target_col].values
        X = features_df[self.feature_names].values
        
        # Remove rows with NaN target
        valid_mask = ~np.isnan(y)
        X = X[valid_mask]
        y = y[valid_mask]
        
        logger.info(f"Training on {len(X)} samples with {X.shape[1]} features")
        
        # Fit PCA
        X_pca = self.fit_pca(X)
        
        # Fit Elastic Net
        y_pred_elastic = self.fit_elastic_net(X, y)
        
        # Fit LSTM
        y_pred_lstm = self.fit_lstm(X, y)
        
        # Calculate ensemble predictions
        weights = self.config['ensemble']['weights']
        y_pred_ensemble = (
            weights[0] * X_pca[:, 0] +  # Use first PCA component as proxy
            weights[1] * y_pred_elastic +
            weights[2] * y_pred_lstm
        )
        
        # Calculate performance metrics
        self._calculate_performance_metrics(y, y_pred_ensemble)
        
        self.is_fitted = True
        logger.info("Alpha generation pipeline fitting completed")
        
        return self
    
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """
        Generate alpha signals for new data
        
        Args:
            data: Market data DataFrame
            
        Returns:
            Alpha signals
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Prepare features
        features_df = self.prepare_features(data)
        X = features_df[self.feature_names].values
        
        # Generate predictions from each model
        X_scaled = self.scaler.transform(X)
        
        # PCA predictions
        X_pca = self.pca.transform(X_scaled)
        
        # Elastic Net predictions
        y_pred_elastic = self.elastic_net.predict(X_scaled)
        
        # LSTM predictions
        X_sequences, _ = self.prepare_lstm_data(X, np.zeros(len(X)))
        y_pred_lstm = np.full(len(X), np.nan)
        if len(X_sequences) > 0:
            y_pred_lstm_sequences = self.lstm_model.predict(X_sequences)
            y_pred_lstm[self.config['lstm']['sequence_length']:] = y_pred_lstm_sequences.flatten()
        
        # Ensemble predictions
        weights = self.config['ensemble']['weights']
        alpha_signals = (
            weights[0] * X_pca[:, 0] +
            weights[1] * y_pred_elastic +
            weights[2] * y_pred_lstm
        )
        
        return alpha_signals
    
    def _calculate_performance_metrics(self, y_true: np.ndarray, y_pred: np.ndarray):
        """Calculate performance metrics"""
        # Remove NaN values
        valid_mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true_clean = y_true[valid_mask]
        y_pred_clean = y_pred[valid_mask]
        
        if len(y_true_clean) == 0:
            logger.warning("No valid predictions for performance calculation")
            return
        
        # Calculate metrics
        returns = y_true_clean
        signal_returns = y_pred_clean * returns  # Assuming long-only strategy
        
        # Sharpe ratio
        if np.std(signal_returns) > 0:
            self.performance_metrics['sharpe_ratio'] = np.mean(signal_returns) / np.std(signal_returns) * np.sqrt(252)
        
        # Total return
        self.performance_metrics['total_return'] = np.sum(signal_returns)
        
        # Volatility
        self.performance_metrics['volatility'] = np.std(signal_returns) * np.sqrt(252)
        
        # Win rate
        self.performance_metrics['win_rate'] = np.mean(signal_returns > 0)
        
        # Maximum drawdown
        cumulative_returns = np.cumsum(signal_returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = cumulative_returns - running_max
        self.performance_metrics['max_drawdown'] = np.min(drawdown)
        
        logger.info(f"Performance Metrics:")
        logger.info(f"  Sharpe Ratio: {self.performance_metrics['sharpe_ratio']:.4f}")
        logger.info(f"  Total Return: {self.performance_metrics['total_return']:.4f}")
        logger.info(f"  Volatility: {self.performance_metrics['volatility']:.4f}")
        logger.info(f"  Win Rate: {self.performance_metrics['win_rate']:.4f}")
        logger.info(f"  Max Drawdown: {self.performance_metrics['max_drawdown']:.4f}")
    
    def save_model(self, path: str):
        """Save the fitted model"""
        model_data = {
            'scaler': self.scaler,
            'pca': self.pca,
            'elastic_net': self.elastic_net,
            'lstm_model': self.lstm_model,
            'feature_names': self.feature_names,
            'config': self.config,
            'performance_metrics': self.performance_metrics,
            'is_fitted': self.is_fitted
        }
        
        joblib.dump(model_data, path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load a fitted model"""
        model_data = joblib.load(path)
        
        self.scaler = model_data['scaler']
        self.pca = model_data['pca']
        self.elastic_net = model_data['elastic_net']
        self.lstm_model = model_data['lstm_model']
        self.feature_names = model_data['feature_names']
        self.config = model_data['config']
        self.performance_metrics = model_data['performance_metrics']
        self.is_fitted = model_data['is_fitted']
        
        logger.info(f"Model loaded from {path}")


def main():
    """Main function for testing the alpha generator"""
    import yfinance as yf
    
    # Download sample data
    logger.info("Downloading sample data")
    ticker = "AAPL"
    data = yf.download(ticker, start="2022-01-01", end="2023-12-31", interval="1d")
    
    # Initialize alpha generator
    alpha_gen = AlphaGenerator()
    
    # Fit the model
    alpha_gen.fit(data, target_col='Close')
    
    # Generate predictions
    alpha_signals = alpha_gen.predict(data)
    
    logger.info(f"Generated alpha signals for {len(alpha_signals)} time points")
    logger.info(f"Signal statistics: mean={np.mean(alpha_signals):.6f}, std={np.std(alpha_signals):.6f}")
    
    # Save model
    alpha_gen.save_model("models/alpha_generator.pkl")


if __name__ == "__main__":
    main() 