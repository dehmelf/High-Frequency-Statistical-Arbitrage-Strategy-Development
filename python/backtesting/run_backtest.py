"""
High-Frequency Statistical Arbitrage Backtesting Framework

This module provides a comprehensive backtesting system that:
- Evaluates strategy performance with realistic constraints
- Calculates key performance metrics (Sharpe ratio, drawdown, etc.)
- Implements transaction costs and market impact
- Provides detailed performance analysis and visualization
"""

import numpy as np
import pandas as pd
import yaml
import click
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from loguru import logger
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import strategy components
import sys
sys.path.append('../src')
from alpha_generation import AlphaGenerator
from deep_q_execution import DQNAgent, ExecutionEnvironment, OrderParams

@dataclass
class BacktestConfig:
    """Backtesting configuration"""
    start_date: str
    end_date: str
    symbols: List[str]
    initial_capital: float
    transaction_cost_bps: float  # Basis points
    market_impact_bps: float
    max_position_size: float
    rebalance_frequency: str  # 'daily', 'hourly', 'minute'
    risk_free_rate: float
    benchmark_symbol: str

@dataclass
class Trade:
    """Trade record"""
    timestamp: pd.Timestamp
    symbol: str
    side: str  # 'buy' or 'sell'
    quantity: int
    price: float
    cost: float
    alpha_signal: float

@dataclass
class Position:
    """Position record"""
    symbol: str
    quantity: int
    avg_price: float
    current_value: float
    unrealized_pnl: float

class BacktestEngine:
    """
    Main backtesting engine
    """
    
    def __init__(self, config: BacktestConfig):
        """
        Initialize backtesting engine
        
        Args:
            config: Backtesting configuration
        """
        self.config = config
        self.alpha_generator = AlphaGenerator()
        self.dqn_agent = None
        
        # Portfolio state
        self.cash = config.initial_capital
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        
        # Performance tracking
        self.portfolio_values = []
        self.returns = []
        self.alpha_signals = {}
        self.execution_costs = []
        
        # Market data
        self.market_data = {}
        self.current_prices = {}
        
        logger.info("Backtesting engine initialized")
    
    def load_market_data(self):
        """Load market data for all symbols"""
        import yfinance as yf
        
        logger.info("Loading market data")
        
        for symbol in self.config.symbols + [self.config.benchmark_symbol]:
            try:
                data = yf.download(
                    symbol, 
                    start=self.config.start_date, 
                    end=self.config.end_date,
                    interval='1h'  # Use hourly data for HF trading
                )
                
                if len(data) > 0:
                    self.market_data[symbol] = data
                    logger.info(f"Loaded {len(data)} records for {symbol}")
                else:
                    logger.warning(f"No data found for {symbol}")
                    
            except Exception as e:
                logger.error(f"Failed to load data for {symbol}: {e}")
        
        # Align all data to common timestamps
        self._align_data()
    
    def _align_data(self):
        """Align all market data to common timestamps"""
        if not self.market_data:
            return
        
        # Find common timestamps
        common_timestamps = None
        for symbol, data in self.market_data.items():
            if common_timestamps is None:
                common_timestamps = set(data.index)
            else:
                common_timestamps = common_timestamps.intersection(set(data.index))
        
        if not common_timestamps:
            logger.error("No common timestamps found across symbols")
            return
        
        # Filter data to common timestamps
        common_timestamps = sorted(list(common_timestamps))
        for symbol in self.market_data:
            self.market_data[symbol] = self.market_data[symbol].loc[common_timestamps]
        
        logger.info(f"Aligned data to {len(common_timestamps)} common timestamps")
    
    def train_models(self):
        """Train alpha generation and execution models"""
        logger.info("Training models")
        
        # Train alpha generator on historical data
        if self.config.symbols:
            training_data = self.market_data[self.config.symbols[0]]
            self.alpha_generator.fit(training_data, target_col='Close')
            logger.info("Alpha generator trained")
        
        # Train DQN execution agent
        if self.config.symbols:
            symbol = self.config.symbols[0]
            order_params = OrderParams(
                symbol=symbol,
                side="buy",
                quantity=1000,
                target_price=training_data['Close'].iloc[-1],
                urgency=0.5,
                max_slippage=0.02,
                time_horizon=24  # 24 hours
            )
            
            env = ExecutionEnvironment(self.market_data[symbol], order_params)
            self.dqn_agent = DQNAgent(state_size=12, action_size=1)
            self.dqn_agent.train(env, episodes=200)  # Reduced for demo
            logger.info("DQN execution agent trained")
    
    def calculate_alpha_signals(self, timestamp: pd.Timestamp) -> Dict[str, float]:
        """
        Calculate alpha signals for all symbols at given timestamp
        
        Args:
            timestamp: Current timestamp
            
        Returns:
            Dictionary of alpha signals by symbol
        """
        signals = {}
        
        for symbol in self.config.symbols:
            if symbol in self.market_data:
                # Get historical data up to current timestamp
                historical_data = self.market_data[symbol][:timestamp]
                
                if len(historical_data) > 100:  # Need sufficient history
                    try:
                        # Generate alpha signal
                        signal = self.alpha_generator.predict(historical_data)
                        if len(signal) > 0 and not np.isnan(signal[-1]):
                            signals[symbol] = signal[-1]
                    except Exception as e:
                        logger.warning(f"Failed to generate alpha signal for {symbol}: {e}")
        
        return signals
    
    def calculate_position_sizes(self, alpha_signals: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate position sizes based on alpha signals and risk constraints
        
        Args:
            alpha_signals: Alpha signals by symbol
            
        Returns:
            Position sizes as fraction of portfolio
        """
        position_sizes = {}
        
        if not alpha_signals:
            return position_sizes
        
        # Normalize signals
        signal_values = list(alpha_signals.values())
        signal_abs = np.abs(signal_values)
        
        if np.sum(signal_abs) > 0:
            # Risk parity allocation
            weights = signal_abs / np.sum(signal_abs)
            
            # Apply maximum position size constraint
            max_weight = self.config.max_position_size
            weights = np.clip(weights, -max_weight, max_weight)
            
            # Apply sign from original signals
            for i, symbol in enumerate(alpha_signals.keys()):
                position_sizes[symbol] = weights[i] * np.sign(alpha_signals[symbol])
        
        return position_sizes
    
    def execute_trades(self, timestamp: pd.Timestamp, target_positions: Dict[str, float]):
        """
        Execute trades to reach target positions
        
        Args:
            timestamp: Current timestamp
            target_positions: Target position sizes by symbol
        """
        current_portfolio_value = self._calculate_portfolio_value(timestamp)
        
        for symbol, target_weight in target_positions.items():
            if symbol not in self.market_data:
                continue
            
            current_price = self.market_data[symbol].loc[timestamp, 'Close']
            current_position = self.positions.get(symbol, Position(symbol, 0, 0, 0, 0))
            
            # Calculate target quantity
            target_value = target_weight * current_portfolio_value
            target_quantity = int(target_value / current_price)
            
            # Calculate trade quantity
            trade_quantity = target_quantity - current_position.quantity
            
            if trade_quantity != 0:
                # Execute trade
                side = "buy" if trade_quantity > 0 else "sell"
                quantity = abs(trade_quantity)
                
                # Calculate execution price with market impact
                market_impact = self.config.market_impact_bps / 10000 * quantity / 1000
                execution_price = current_price * (1 + market_impact) if side == "buy" else current_price * (1 - market_impact)
                
                # Calculate transaction cost
                transaction_cost = execution_price * quantity * self.config.transaction_cost_bps / 10000
                total_cost = execution_price * quantity + transaction_cost
                
                # Check if we have enough cash
                if side == "buy" and total_cost > self.cash:
                    # Scale down trade
                    max_quantity = int(self.cash / (execution_price * (1 + self.config.transaction_cost_bps / 10000)))
                    quantity = min(quantity, max_quantity)
                    total_cost = execution_price * quantity + execution_price * quantity * self.config.transaction_cost_bps / 10000
                
                if quantity > 0:
                    # Execute trade
                    if side == "buy":
                        self.cash -= total_cost
                    else:
                        self.cash += execution_price * quantity - transaction_cost
                    
                    # Update position
                    if symbol not in self.positions:
                        self.positions[symbol] = Position(symbol, 0, 0, 0, 0)
                    
                    position = self.positions[symbol]
                    if side == "buy":
                        # Add to position
                        total_quantity = position.quantity + quantity
                        total_cost_basis = position.avg_price * position.quantity + execution_price * quantity
                        position.quantity = total_quantity
                        position.avg_price = total_cost_basis / total_quantity
                    else:
                        # Reduce position
                        position.quantity -= quantity
                        if position.quantity == 0:
                            position.avg_price = 0
                    
                    # Record trade
                    trade = Trade(
                        timestamp=timestamp,
                        symbol=symbol,
                        side=side,
                        quantity=quantity,
                        price=execution_price,
                        cost=total_cost,
                        alpha_signal=target_weight
                    )
                    self.trades.append(trade)
                    
                    self.execution_costs.append(transaction_cost)
    
    def _calculate_portfolio_value(self, timestamp: pd.Timestamp) -> float:
        """Calculate current portfolio value"""
        portfolio_value = self.cash
        
        for symbol, position in self.positions.items():
            if symbol in self.market_data and timestamp in self.market_data[symbol].index:
                current_price = self.market_data[symbol].loc[timestamp, 'Close']
                position.current_value = position.quantity * current_price
                position.unrealized_pnl = position.current_value - (position.quantity * position.avg_price)
                portfolio_value += position.current_value
        
        return portfolio_value
    
    def run_backtest(self) -> Dict[str, float]:
        """
        Run the complete backtest
        
        Returns:
            Performance metrics
        """
        logger.info("Starting backtest")
        
        # Get all timestamps
        if not self.market_data:
            logger.error("No market data available")
            return {}
        
        timestamps = list(self.market_data[self.config.symbols[0]].index)
        
        # Initialize portfolio tracking
        initial_value = self.config.initial_capital
        self.portfolio_values = [initial_value]
        
        for i, timestamp in enumerate(timestamps):
            if i % 100 == 0:
                logger.info(f"Processing timestamp {i+1}/{len(timestamps)}: {timestamp}")
            
            # Calculate alpha signals
            alpha_signals = self.calculate_alpha_signals(timestamp)
            self.alpha_signals[timestamp] = alpha_signals
            
            # Calculate target positions
            target_positions = self.calculate_position_sizes(alpha_signals)
            
            # Execute trades
            self.execute_trades(timestamp, target_positions)
            
            # Calculate portfolio value
            portfolio_value = self._calculate_portfolio_value(timestamp)
            self.portfolio_values.append(portfolio_value)
            
            # Calculate return
            if i > 0:
                return_rate = (portfolio_value - self.portfolio_values[-2]) / self.portfolio_values[-2]
                self.returns.append(return_rate)
        
        # Calculate performance metrics
        performance_metrics = self._calculate_performance_metrics()
        
        logger.info("Backtest completed")
        return performance_metrics
    
    def _calculate_performance_metrics(self) -> Dict[str, float]:
        """Calculate comprehensive performance metrics"""
        if len(self.returns) == 0:
            return {}
        
        returns = np.array(self.returns)
        portfolio_values = np.array(self.portfolio_values)
        
        # Basic metrics
        total_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]
        annualized_return = total_return * 252 / len(returns)
        volatility = np.std(returns) * np.sqrt(252)
        
        # Sharpe ratio
        risk_free_return = self.config.risk_free_rate / 252
        excess_returns = returns - risk_free_return
        sharpe_ratio = np.mean(excess_returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
        
        # Maximum drawdown
        cumulative_returns = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = np.min(drawdown)
        
        # Win rate
        win_rate = np.mean(returns > 0)
        
        # Calmar ratio
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Sortino ratio
        downside_returns = returns[returns < 0]
        downside_deviation = np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino_ratio = np.mean(excess_returns) / downside_deviation if downside_deviation > 0 else 0
        
        # Transaction costs
        total_transaction_costs = np.sum(self.execution_costs)
        transaction_cost_ratio = total_transaction_costs / portfolio_values[0]
        
        # Benchmark comparison
        benchmark_return = self._calculate_benchmark_return()
        excess_return_vs_benchmark = total_return - benchmark_return
        
        metrics = {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'calmar_ratio': calmar_ratio,
            'sortino_ratio': sortino_ratio,
            'transaction_cost_ratio': transaction_cost_ratio,
            'total_transaction_costs': total_transaction_costs,
            'excess_return_vs_benchmark': excess_return_vs_benchmark,
            'total_trades': len(self.trades)
        }
        
        return metrics
    
    def _calculate_benchmark_return(self) -> float:
        """Calculate benchmark return"""
        if self.config.benchmark_symbol not in self.market_data:
            return 0.0
        
        benchmark_data = self.market_data[self.config.benchmark_symbol]
        if len(benchmark_data) < 2:
            return 0.0
        
        initial_price = benchmark_data.iloc[0]['Close']
        final_price = benchmark_data.iloc[-1]['Close']
        return (final_price - initial_price) / initial_price
    
    def generate_report(self, output_dir: str = "data"):
        """Generate comprehensive backtest report"""
        logger.info("Generating backtest report")
        
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Performance metrics
        metrics = self._calculate_performance_metrics()
        
        # Save metrics to file
        metrics_df = pd.DataFrame([metrics])
        metrics_df.to_csv(f"{output_dir}/backtest_metrics.csv", index=False)
        
        # Save trades to file
        trades_df = pd.DataFrame([vars(trade) for trade in self.trades])
        if len(trades_df) > 0:
            trades_df.to_csv(f"{output_dir}/backtest_trades.csv", index=False)
        
        # Create performance plots
        self._create_performance_plots(output_dir)
        
        # Print summary
        logger.info("\n" + "="*50)
        logger.info("BACKTEST RESULTS SUMMARY")
        logger.info("="*50)
        logger.info(f"Total Return: {metrics.get('total_return', 0):.4f} ({metrics.get('total_return', 0)*100:.2f}%)")
        logger.info(f"Annualized Return: {metrics.get('annualized_return', 0):.4f} ({metrics.get('annualized_return', 0)*100:.2f}%)")
        logger.info(f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.4f}")
        logger.info(f"Max Drawdown: {metrics.get('max_drawdown', 0):.4f} ({metrics.get('max_drawdown', 0)*100:.2f}%)")
        logger.info(f"Volatility: {metrics.get('volatility', 0):.4f} ({metrics.get('volatility', 0)*100:.2f}%)")
        logger.info(f"Win Rate: {metrics.get('win_rate', 0):.4f} ({metrics.get('win_rate', 0)*100:.2f}%)")
        logger.info(f"Total Trades: {metrics.get('total_trades', 0)}")
        logger.info(f"Transaction Costs: ${metrics.get('total_transaction_costs', 0):,.2f}")
        logger.info("="*50)
    
    def _create_performance_plots(self, output_dir: str):
        """Create performance visualization plots"""
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Portfolio value over time
        axes[0, 0].plot(self.portfolio_values)
        axes[0, 0].set_title('Portfolio Value Over Time')
        axes[0, 0].set_xlabel('Time Step')
        axes[0, 0].set_ylabel('Portfolio Value ($)')
        axes[0, 0].grid(True)
        
        # Returns distribution
        if self.returns:
            axes[0, 1].hist(self.returns, bins=50, alpha=0.7, edgecolor='black')
            axes[0, 1].set_title('Returns Distribution')
            axes[0, 1].set_xlabel('Return')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].grid(True)
        
        # Cumulative returns
        if self.returns:
            cumulative_returns = np.cumprod(1 + np.array(self.returns))
            axes[1, 0].plot(cumulative_returns)
            axes[1, 0].set_title('Cumulative Returns')
            axes[1, 0].set_xlabel('Time Step')
            axes[1, 0].set_ylabel('Cumulative Return')
            axes[1, 0].grid(True)
        
        # Drawdown
        if self.returns:
            cumulative_returns = np.cumprod(1 + np.array(self.returns))
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdown = (cumulative_returns - running_max) / running_max
            axes[1, 1].fill_between(range(len(drawdown)), drawdown, 0, alpha=0.3, color='red')
            axes[1, 1].set_title('Drawdown')
            axes[1, 1].set_xlabel('Time Step')
            axes[1, 1].set_ylabel('Drawdown')
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/backtest_performance.png", dpi=300, bbox_inches='tight')
        plt.close()


@click.command()
@click.option('--config', default='config/backtest_config.yaml', help='Configuration file path')
@click.option('--output', default='data', help='Output directory')
def main(config: str, output: str):
    """Run backtest with configuration file"""
    
    # Load configuration
    try:
        with open(config, 'r') as file:
            config_data = yaml.safe_load(file)
        
        backtest_config = BacktestConfig(
            start_date=config_data['start_date'],
            end_date=config_data['end_date'],
            symbols=config_data['symbols'],
            initial_capital=config_data['initial_capital'],
            transaction_cost_bps=config_data['transaction_cost_bps'],
            market_impact_bps=config_data['market_impact_bps'],
            max_position_size=config_data['max_position_size'],
            rebalance_frequency=config_data['rebalance_frequency'],
            risk_free_rate=config_data['risk_free_rate'],
            benchmark_symbol=config_data['benchmark_symbol']
        )
        
    except FileNotFoundError:
        logger.warning(f"Config file {config} not found, using defaults")
        backtest_config = BacktestConfig(
            start_date="2023-01-01",
            end_date="2023-12-31",
            symbols=["AAPL", "MSFT", "GOOGL"],
            initial_capital=1000000.0,
            transaction_cost_bps=1.0,
            market_impact_bps=2.0,
            max_position_size=0.2,
            rebalance_frequency="hourly",
            risk_free_rate=0.05,
            benchmark_symbol="SPY"
        )
    
    # Initialize and run backtest
    engine = BacktestEngine(backtest_config)
    engine.load_market_data()
    engine.train_models()
    
    # Run backtest
    performance_metrics = engine.run_backtest()
    
    # Generate report
    engine.generate_report(output)
    
    logger.info("Backtest completed successfully")


if __name__ == "__main__":
    main() 