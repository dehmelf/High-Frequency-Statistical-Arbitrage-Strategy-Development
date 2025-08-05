"""
GPU-Accelerated Monte Carlo Simulation for Exotic Options Pricing

This module implements high-performance Monte Carlo simulations for:
- Heston Model: Stochastic volatility option pricing
- SABR Model: Stochastic alpha beta rho model
- GPU acceleration using CUDA via CuPy
- Parallel path generation and pricing
"""

import numpy as np
import cupy as cp
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import time
from dataclasses import dataclass
from loguru import logger
import matplotlib.pyplot as plt
import seaborn as sns

@dataclass
class OptionParams:
    """Option parameters"""
    S0: float  # Initial stock price
    K: float   # Strike price
    T: float   # Time to maturity
    r: float   # Risk-free rate
    sigma: float  # Initial volatility
    option_type: str = "call"  # "call" or "put"
    
    def __post_init__(self):
        if self.option_type not in ["call", "put"]:
            raise ValueError("option_type must be 'call' or 'put'")

@dataclass
class HestonParams:
    """Heston model parameters"""
    kappa: float  # Mean reversion speed
    theta: float  # Long-term volatility
    eta: float    # Volatility of volatility
    rho: float    # Correlation between asset and volatility
    v0: float     # Initial volatility

@dataclass
class SABRParams:
    """SABR model parameters"""
    alpha: float  # Initial volatility
    beta: float   # CEV parameter
    rho: float    # Correlation
    nu: float     # Volatility of volatility

class GPUMonteCarlo:
    """
    GPU-accelerated Monte Carlo simulation engine
    """
    
    def __init__(self, device_id: int = 0):
        """
        Initialize GPU Monte Carlo engine
        
        Args:
            device_id: CUDA device ID to use
        """
        try:
            cp.cuda.Device(device_id).use()
            self.device_id = device_id
            logger.info(f"Using CUDA device {device_id}: {cp.cuda.Device(device_id).name}")
        except Exception as e:
            logger.error(f"Failed to initialize CUDA device {device_id}: {e}")
            raise
        
        # Performance tracking
        self.performance_stats = {
            'total_paths': 0,
            'total_time': 0.0,
            'paths_per_second': 0.0,
            'gpu_memory_used': 0.0
        }
    
    def generate_normal_random_numbers(self, shape: Tuple[int, ...], seed: Optional[int] = None) -> cp.ndarray:
        """
        Generate normal random numbers on GPU
        
        Args:
            shape: Shape of the random number array
            seed: Random seed for reproducibility
            
        Returns:
            GPU array of normal random numbers
        """
        if seed is not None:
            cp.random.seed(seed)
        
        return cp.random.normal(0, 1, shape)
    
    def black_scholes_price(self, params: OptionParams, n_paths: int = 1000000) -> Dict[str, float]:
        """
        Price European options using Black-Scholes Monte Carlo
        
        Args:
            params: Option parameters
            n_paths: Number of Monte Carlo paths
            
        Returns:
            Dictionary with price and confidence interval
        """
        logger.info(f"Pricing {params.option_type} option using Black-Scholes MC with {n_paths:,} paths")
        
        start_time = time.time()
        
        # Generate random numbers
        Z = self.generate_normal_random_numbers((n_paths,))
        
        # Calculate stock price paths
        drift = (params.r - 0.5 * params.sigma**2) * params.T
        diffusion = params.sigma * cp.sqrt(params.T) * Z
        ST = params.S0 * cp.exp(drift + diffusion)
        
        # Calculate option payoffs
        if params.option_type == "call":
            payoffs = cp.maximum(ST - params.K, 0)
        else:
            payoffs = cp.maximum(params.K - ST, 0)
        
        # Discount and calculate statistics
        price = cp.exp(-params.r * params.T) * cp.mean(payoffs)
        std_error = cp.exp(-params.r * params.T) * cp.std(payoffs) / cp.sqrt(n_paths)
        
        # Move to CPU for final calculations
        price_cpu = float(price)
        std_error_cpu = float(std_error)
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        # Update performance stats
        self.performance_stats['total_paths'] += n_paths
        self.performance_stats['total_time'] += elapsed_time
        self.performance_stats['paths_per_second'] = n_paths / elapsed_time
        
        result = {
            'price': price_cpu,
            'std_error': std_error_cpu,
            'confidence_interval_95': (price_cpu - 1.96 * std_error_cpu, price_cpu + 1.96 * std_error_cpu),
            'computation_time': elapsed_time,
            'paths_per_second': n_paths / elapsed_time
        }
        
        logger.info(f"Black-Scholes price: {price_cpu:.6f} ± {std_error_cpu:.6f}")
        logger.info(f"Computation time: {elapsed_time:.3f}s ({n_paths/elapsed_time:,.0f} paths/sec)")
        
        return result
    
    def heston_price(self, option_params: OptionParams, heston_params: HestonParams, 
                    n_paths: int = 1000000, n_steps: int = 252) -> Dict[str, float]:
        """
        Price European options using Heston model Monte Carlo
        
        Args:
            option_params: Option parameters
            heston_params: Heston model parameters
            n_paths: Number of Monte Carlo paths
            n_steps: Number of time steps
            
        Returns:
            Dictionary with price and confidence interval
        """
        logger.info(f"Pricing {option_params.option_type} option using Heston MC with {n_paths:,} paths")
        
        start_time = time.time()
        
        # Time discretization
        dt = option_params.T / n_steps
        sqrt_dt = cp.sqrt(dt)
        
        # Initialize arrays on GPU
        S = cp.full((n_paths, n_steps + 1), option_params.S0)
        v = cp.full((n_paths, n_steps + 1), heston_params.v0)
        
        # Generate correlated random numbers
        Z1 = self.generate_normal_random_numbers((n_paths, n_steps))
        Z2 = self.generate_normal_random_numbers((n_paths, n_steps))
        Z_v = heston_params.rho * Z1 + cp.sqrt(1 - heston_params.rho**2) * Z2
        
        # Euler-Maruyama scheme for Heston model
        for i in range(n_steps):
            # Volatility process
            v_sqrt = cp.sqrt(cp.maximum(v[:, i], 0))  # Ensure positive volatility
            v[:, i + 1] = v[:, i] + heston_params.kappa * (heston_params.theta - v[:, i]) * dt + \
                         heston_params.eta * v_sqrt * sqrt_dt * Z_v[:, i]
            
            # Stock price process
            S_sqrt_v = cp.sqrt(cp.maximum(v[:, i], 0))
            S[:, i + 1] = S[:, i] * cp.exp((option_params.r - 0.5 * v[:, i]) * dt + \
                                         S_sqrt_v * sqrt_dt * Z1[:, i])
        
        # Calculate option payoffs
        ST = S[:, -1]
        if option_params.option_type == "call":
            payoffs = cp.maximum(ST - option_params.K, 0)
        else:
            payoffs = cp.maximum(option_params.K - ST, 0)
        
        # Discount and calculate statistics
        price = cp.exp(-option_params.r * option_params.T) * cp.mean(payoffs)
        std_error = cp.exp(-option_params.r * option_params.T) * cp.std(payoffs) / cp.sqrt(n_paths)
        
        # Move to CPU for final calculations
        price_cpu = float(price)
        std_error_cpu = float(std_error)
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        # Update performance stats
        self.performance_stats['total_paths'] += n_paths
        self.performance_stats['total_time'] += elapsed_time
        self.performance_stats['paths_per_second'] = n_paths / elapsed_time
        
        result = {
            'price': price_cpu,
            'std_error': std_error_cpu,
            'confidence_interval_95': (price_cpu - 1.96 * std_error_cpu, price_cpu + 1.96 * std_error_cpu),
            'computation_time': elapsed_time,
            'paths_per_second': n_paths / elapsed_time,
            'final_volatility_mean': float(cp.mean(v[:, -1])),
            'final_volatility_std': float(cp.std(v[:, -1]))
        }
        
        logger.info(f"Heston price: {price_cpu:.6f} ± {std_error_cpu:.6f}")
        logger.info(f"Final volatility: {result['final_volatility_mean']:.4f} ± {result['final_volatility_std']:.4f}")
        logger.info(f"Computation time: {elapsed_time:.3f}s ({n_paths/elapsed_time:,.0f} paths/sec)")
        
        return result
    
    def sabr_price(self, option_params: OptionParams, sabr_params: SABRParams,
                  n_paths: int = 1000000, n_steps: int = 252) -> Dict[str, float]:
        """
        Price European options using SABR model Monte Carlo
        
        Args:
            option_params: Option parameters
            sabr_params: SABR model parameters
            n_paths: Number of Monte Carlo paths
            n_steps: Number of time steps
            
        Returns:
            Dictionary with price and confidence interval
        """
        logger.info(f"Pricing {option_params.option_type} option using SABR MC with {n_paths:,} paths")
        
        start_time = time.time()
        
        # Time discretization
        dt = option_params.T / n_steps
        sqrt_dt = cp.sqrt(dt)
        
        # Initialize arrays on GPU
        S = cp.full((n_paths, n_steps + 1), option_params.S0)
        alpha = cp.full((n_paths, n_steps + 1), sabr_params.alpha)
        
        # Generate correlated random numbers
        Z1 = self.generate_normal_random_numbers((n_paths, n_steps))
        Z2 = self.generate_normal_random_numbers((n_paths, n_steps))
        Z_alpha = sabr_params.rho * Z1 + cp.sqrt(1 - sabr_params.rho**2) * Z2
        
        # Euler-Maruyama scheme for SABR model
        for i in range(n_steps):
            # Volatility process
            alpha[:, i + 1] = alpha[:, i] + sabr_params.nu * alpha[:, i] * sqrt_dt * Z_alpha[:, i]
            
            # Stock price process
            S_beta = S[:, i] ** sabr_params.beta
            S[:, i + 1] = S[:, i] + alpha[:, i] * S_beta * sqrt_dt * Z1[:, i]
        
        # Calculate option payoffs
        ST = S[:, -1]
        if option_params.option_type == "call":
            payoffs = cp.maximum(ST - option_params.K, 0)
        else:
            payoffs = cp.maximum(option_params.K - ST, 0)
        
        # Discount and calculate statistics
        price = cp.exp(-option_params.r * option_params.T) * cp.mean(payoffs)
        std_error = cp.exp(-option_params.r * option_params.T) * cp.std(payoffs) / cp.sqrt(n_paths)
        
        # Move to CPU for final calculations
        price_cpu = float(price)
        std_error_cpu = float(std_error)
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        # Update performance stats
        self.performance_stats['total_paths'] += n_paths
        self.performance_stats['total_time'] += elapsed_time
        self.performance_stats['paths_per_second'] = n_paths / elapsed_time
        
        result = {
            'price': price_cpu,
            'std_error': std_error_cpu,
            'confidence_interval_95': (price_cpu - 1.96 * std_error_cpu, price_cpu + 1.96 * std_error_cpu),
            'computation_time': elapsed_time,
            'paths_per_second': n_paths / elapsed_time,
            'final_alpha_mean': float(cp.mean(alpha[:, -1])),
            'final_alpha_std': float(cp.std(alpha[:, -1]))
        }
        
        logger.info(f"SABR price: {price_cpu:.6f} ± {std_error_cpu:.6f}")
        logger.info(f"Final alpha: {result['final_alpha_mean']:.4f} ± {result['final_alpha_std']:.4f}")
        logger.info(f"Computation time: {elapsed_time:.3f}s ({n_paths/elapsed_time:,.0f} paths/sec)")
        
        return result
    
    def asian_option_price(self, option_params: OptionParams, heston_params: HestonParams,
                          n_paths: int = 1000000, n_steps: int = 252) -> Dict[str, float]:
        """
        Price Asian options using Heston model
        
        Args:
            option_params: Option parameters
            heston_params: Heston model parameters
            n_paths: Number of Monte Carlo paths
            n_steps: Number of time steps
            
        Returns:
            Dictionary with price and confidence interval
        """
        logger.info(f"Pricing Asian {option_params.option_type} option using Heston MC with {n_paths:,} paths")
        
        start_time = time.time()
        
        # Time discretization
        dt = option_params.T / n_steps
        sqrt_dt = cp.sqrt(dt)
        
        # Initialize arrays on GPU
        S = cp.full((n_paths, n_steps + 1), option_params.S0)
        v = cp.full((n_paths, n_steps + 1), heston_params.v0)
        
        # Generate correlated random numbers
        Z1 = self.generate_normal_random_numbers((n_paths, n_steps))
        Z2 = self.generate_normal_random_numbers((n_paths, n_steps))
        Z_v = heston_params.rho * Z1 + cp.sqrt(1 - heston_params.rho**2) * Z2
        
        # Euler-Maruyama scheme for Heston model
        for i in range(n_steps):
            # Volatility process
            v_sqrt = cp.sqrt(cp.maximum(v[:, i], 0))
            v[:, i + 1] = v[:, i] + heston_params.kappa * (heston_params.theta - v[:, i]) * dt + \
                         heston_params.eta * v_sqrt * sqrt_dt * Z_v[:, i]
            
            # Stock price process
            S_sqrt_v = cp.sqrt(cp.maximum(v[:, i], 0))
            S[:, i + 1] = S[:, i] * cp.exp((option_params.r - 0.5 * v[:, i]) * dt + \
                                         S_sqrt_v * sqrt_dt * Z1[:, i])
        
        # Calculate Asian option payoffs (arithmetic average)
        S_avg = cp.mean(S, axis=1)
        
        if option_params.option_type == "call":
            payoffs = cp.maximum(S_avg - option_params.K, 0)
        else:
            payoffs = cp.maximum(option_params.K - S_avg, 0)
        
        # Discount and calculate statistics
        price = cp.exp(-option_params.r * option_params.T) * cp.mean(payoffs)
        std_error = cp.exp(-option_params.r * option_params.T) * cp.std(payoffs) / cp.sqrt(n_paths)
        
        # Move to CPU for final calculations
        price_cpu = float(price)
        std_error_cpu = float(std_error)
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        result = {
            'price': price_cpu,
            'std_error': std_error_cpu,
            'confidence_interval_95': (price_cpu - 1.96 * std_error_cpu, price_cpu + 1.96 * std_error_cpu),
            'computation_time': elapsed_time,
            'paths_per_second': n_paths / elapsed_time
        }
        
        logger.info(f"Asian option price: {price_cpu:.6f} ± {std_error_cpu:.6f}")
        logger.info(f"Computation time: {elapsed_time:.3f}s ({n_paths/elapsed_time:,.0f} paths/sec)")
        
        return result
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get performance statistics"""
        return self.performance_stats.copy()
    
    def benchmark_models(self, option_params: OptionParams, heston_params: HestonParams,
                        sabr_params: SABRParams, n_paths: int = 1000000) -> pd.DataFrame:
        """
        Benchmark different pricing models
        
        Args:
            option_params: Option parameters
            heston_params: Heston model parameters
            sabr_params: SABR model parameters
            n_paths: Number of Monte Carlo paths
            
        Returns:
            DataFrame with benchmark results
        """
        logger.info("Starting model benchmark")
        
        results = []
        
        # Black-Scholes
        bs_result = self.black_scholes_price(option_params, n_paths)
        results.append({
            'model': 'Black-Scholes',
            'price': bs_result['price'],
            'std_error': bs_result['std_error'],
            'computation_time': bs_result['computation_time'],
            'paths_per_second': bs_result['paths_per_second']
        })
        
        # Heston
        heston_result = self.heston_price(option_params, heston_params, n_paths)
        results.append({
            'model': 'Heston',
            'price': heston_result['price'],
            'std_error': heston_result['std_error'],
            'computation_time': heston_result['computation_time'],
            'paths_per_second': heston_result['paths_per_second']
        })
        
        # SABR
        sabr_result = self.sabr_price(option_params, sabr_params, n_paths)
        results.append({
            'model': 'SABR',
            'price': sabr_result['price'],
            'std_error': sabr_result['std_error'],
            'computation_time': sabr_result['computation_time'],
            'paths_per_second': sabr_result['paths_per_second']
        })
        
        # Asian option with Heston
        asian_result = self.asian_option_price(option_params, heston_params, n_paths)
        results.append({
            'model': 'Asian (Heston)',
            'price': asian_result['price'],
            'std_error': asian_result['std_error'],
            'computation_time': asian_result['computation_time'],
            'paths_per_second': asian_result['paths_per_second']
        })
        
        df = pd.DataFrame(results)
        logger.info("\nBenchmark Results:")
        logger.info(df.to_string(index=False))
        
        return df


def main():
    """Main function for testing the GPU Monte Carlo engine"""
    
    # Example parameters
    option_params = OptionParams(
        S0=100.0,      # Initial stock price
        K=100.0,       # Strike price
        T=1.0,         # Time to maturity (1 year)
        r=0.05,        # Risk-free rate
        sigma=0.2,     # Initial volatility
        option_type="call"
    )
    
    heston_params = HestonParams(
        kappa=2.0,     # Mean reversion speed
        theta=0.04,    # Long-term volatility
        eta=0.3,       # Volatility of volatility
        rho=-0.7,      # Correlation
        v0=0.04        # Initial volatility
    )
    
    sabr_params = SABRParams(
        alpha=0.2,     # Initial volatility
        beta=0.5,      # CEV parameter
        rho=-0.7,      # Correlation
        nu=0.3         # Volatility of volatility
    )
    
    # Initialize GPU Monte Carlo engine
    mc_engine = GPUMonteCarlo(device_id=0)
    
    # Run benchmark
    benchmark_results = mc_engine.benchmark_models(
        option_params, heston_params, sabr_params, n_paths=1000000
    )
    
    # Save results
    benchmark_results.to_csv("data/monte_carlo_benchmark.csv", index=False)
    logger.info("Benchmark results saved to data/monte_carlo_benchmark.csv")
    
    # Print performance stats
    perf_stats = mc_engine.get_performance_stats()
    logger.info(f"\nPerformance Statistics:")
    logger.info(f"Total paths: {perf_stats['total_paths']:,}")
    logger.info(f"Total time: {perf_stats['total_time']:.3f}s")
    logger.info(f"Average paths/sec: {perf_stats['total_paths']/perf_stats['total_time']:,.0f}")


if __name__ == "__main__":
    main() 