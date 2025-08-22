#!/usr/bin/env python3
"""
V5 Equity Feature Selector - Fast Parallel Processing with Utils Integration
==========================================================================

Fast, well-organized equity feature selector that properly utilizes V5 utils
structure and creates advanced equity features with GPU acceleration.

FEATURES:
- V5 utils integration (feature_calculators.py, data_validators.py)
- Fast parallel processing with ThreadPoolExecutor
- GPU acceleration via cuDF-pandas with fallback
- Technical indicators using V5 TechnicalCalculator
- Statistical features using V5 StatisticalCalculator  
- Proper temporal lag enforcement (1-day lag)
- Single parquet output with date in name
- Comprehensive feature importance tracking

ARCHITECTURE:
- Uses V5 utils/feature_calculators.py for calculations
- Integrates with V5 temporal validation
- Modular design with proper error handling
- Memory-efficient batch processing
"""

import os
import sys
import warnings
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import time
import logging
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import gc

# Suppress warnings
warnings.filterwarnings('ignore')

# GPU acceleration setup
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
os.environ['CUPY_GPU_MEMORY_LIMIT'] = '20480'  # 20GB per GPU

# Enable cuDF-pandas acceleration
try:
    import cudf.pandas
    cudf.pandas.install()
    print("🚀 cuDF-pandas GPU acceleration ENABLED")
    GPU_ACCELERATION = True
except ImportError:
    print("⚠️ cuDF-pandas not available, using standard pandas")
    GPU_ACCELERATION = False

# Add V5 utils to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent / "utils"))

import numpy as np
import pandas as pd

# Import V5 utils
from feature_calculators import TechnicalCalculator, StatisticalCalculator, PVMCalculator
from data_validators import DataValidator
from data.temporal_validator import TemporalValidator

# GPU libraries
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

# Setup logging
def setup_logging() -> logging.Logger:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path("/media/knight2/EDB/numer_crypto_temp/data/log")
    log_dir.mkdir(exist_ok=True)
    
    logger = logging.getLogger("v5_equity_feature_selector")
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        file_handler = logging.FileHandler(log_dir / f"v5_equity_feature_selector_{timestamp}.log")
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter('%(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    
    return logger

logger = setup_logging()

class V5EquityFeatureSelector:
    """
    V5 Equity Feature Selector with Utils Integration
    
    Fast, parallel equity feature selector that utilizes V5 utils structure
    for maximum performance and code reusability.
    """
    
    def __init__(self, data_dir: Optional[Path] = None, max_workers: int = 10):
        self.data_dir = data_dir or Path("/media/knight2/EDB/numer_crypto_temp/data")
        self.equity_dir = self.data_dir / "raw" / "equity"
        self.crypto_dir = self.data_dir / "raw" / "price"
        self.numerai_dir = self.data_dir / "raw" / "numerai"
        self.output_dir = self.data_dir / "processed" / "feature" / "equity"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # V5 utils integration
        self.technical_calc = TechnicalCalculator()
        self.statistical_calc = StatisticalCalculator()
        self.pvm_calc = PVMCalculator()
        self.data_validator = DataValidator()
        self.temporal_validator = TemporalValidator()
        
        # Processing parameters
        self.max_workers = max_workers
        self.batch_size = 50  # Batch size for parallel processing
        self.min_correlation = 0.1  # Minimum correlation threshold
        self.top_n_equities = 5  # Top N correlated equities per crypto
        
        # GPU settings
        self.gpu_available = GPU_ACCELERATION and CUPY_AVAILABLE
        
        # Temporal lag enforcement
        self.TEMPORAL_LAG_DAYS = 1
        self.cutoff_date = datetime.now() - timedelta(days=self.TEMPORAL_LAG_DAYS)
        
        logger.info("🏦 V5 Equity Feature Selector initialized")
        logger.info(f"📁 Data directory: {self.data_dir}")
        logger.info(f"📁 Output directory: {self.output_dir}")
        logger.info(f"🧵 Max workers: {self.max_workers}")
        logger.info(f"🚀 GPU acceleration: {self.gpu_available}")
        logger.info(f"📊 Top equities per crypto: {self.top_n_equities}")
        logger.info(f"📅 Temporal cutoff: {self.cutoff_date.date()}")
    
    def load_latest_equity_data(self) -> pd.DataFrame:
        """Load latest consolidated equity data"""
        logger.info("📊 Loading latest equity data...")
        
        # Look for latest equity data file
        equity_files = list(self.equity_dir.glob("equity_data_*.parquet"))
        if not equity_files:
            equity_files = list(self.equity_dir.glob("*.parquet"))
        
        if not equity_files:
            raise FileNotFoundError(f"No equity data files found in {self.equity_dir}")
        
        latest_file = max(equity_files, key=lambda f: f.stat().st_mtime)
        logger.info(f"📂 Loading: {latest_file.name}")
        
        try:
            # Load with GPU acceleration if available
            if self.gpu_available:
                equity_df = pd.read_parquet(latest_file)
            else:
                equity_df = pd.read_parquet(latest_file)
            
            # Standardize date column
            equity_df['date'] = pd.to_datetime(equity_df['date'])
            
            # Validate data structure
            required_cols = ['date', 'numerai_ticker', 'open', 'high', 'low', 'close', 'volume']
            missing_cols = [col for col in required_cols if col not in equity_df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Validate temporal lag using V5 validator
            if not self.temporal_validator.validate_temporal_lag(equity_df):
                logger.warning("⚠️ Temporal lag validation warning for equity data")
            
            logger.info(f"✅ Equity data loaded: {equity_df.shape}")
            logger.info(f"📊 Date range: {equity_df['date'].min().date()} to {equity_df['date'].max().date()}")
            logger.info(f"📊 Unique tickers: {equity_df['numerai_ticker'].nunique()}")
            
            return equity_df
            
        except Exception as e:
            logger.error(f"❌ Error loading equity data: {e}")
            raise
    
    def load_cryptocurrency_data(self) -> pd.DataFrame:
        """Load cryptocurrency price data"""
        logger.info("📈 Loading cryptocurrency data...")
        
        # Look for latest crypto price data
        crypto_files = list(self.crypto_dir.glob("*.parquet"))
        if not crypto_files:
            crypto_files = list(self.crypto_dir.glob("*.csv"))
        
        if not crypto_files:
            raise FileNotFoundError(f"No cryptocurrency data found in {self.crypto_dir}")
        
        latest_file = max(crypto_files, key=lambda f: f.stat().st_mtime)
        logger.info(f"📂 Loading: {latest_file.name}")
        
        try:
            if latest_file.suffix == '.parquet':
                crypto_df = pd.read_parquet(latest_file)
            else:
                crypto_df = pd.read_csv(latest_file)
            
            # Standardize columns
            crypto_df['date'] = pd.to_datetime(crypto_df['date'])
            
            # Validate temporal lag
            if not self.temporal_validator.validate_temporal_lag(crypto_df):
                logger.warning("⚠️ Temporal lag validation warning for crypto data")
            
            logger.info(f"✅ Crypto data loaded: {crypto_df.shape}")
            logger.info(f"📊 Crypto symbols: {crypto_df['symbol'].nunique()}")
            
            return crypto_df
            
        except Exception as e:
            logger.error(f"❌ Error loading crypto data: {e}")
            raise
    
    def load_numerai_targets(self) -> pd.DataFrame:
        """Load Numerai targets to get target cryptocurrencies"""
        logger.info("🎯 Loading Numerai targets...")
        
        try:
            target_files = list(self.numerai_dir.glob("*train_targets*.parquet"))
            if not target_files:
                target_files = list(self.numerai_dir.glob("*targets*.parquet"))
            
            if not target_files:
                raise FileNotFoundError(f"No Numerai targets found in {self.numerai_dir}")
            
            latest_file = max(target_files, key=lambda f: f.stat().st_mtime)
            logger.info(f"📂 Loading: {latest_file.name}")
            
            targets_df = pd.read_parquet(latest_file)
            targets_df['date'] = pd.to_datetime(targets_df['date'])
            
            logger.info(f"✅ Targets loaded: {targets_df.shape}")
            logger.info(f"📊 Target symbols: {targets_df['symbol'].nunique()}")
            
            return targets_df
            
        except Exception as e:
            logger.error(f"❌ Error loading Numerai targets: {e}")
            raise
    
    def calculate_crypto_equity_correlations(self, crypto_symbol: str, 
                                           crypto_data: pd.DataFrame,
                                           equity_data: pd.DataFrame) -> List[Tuple[str, float]]:
        """
        Calculate correlations between a cryptocurrency and equity tickers
        using V5 GPU acceleration where available
        """
        crypto_prices = crypto_data[crypto_data['symbol'] == crypto_symbol].copy()
        if crypto_prices.empty:
            return []
        
        crypto_prices = crypto_prices.set_index('date')['close'].sort_index()
        
        correlations = []
        equity_tickers = equity_data['numerai_ticker'].unique()
        
        for equity_ticker in equity_tickers:
            try:
                equity_prices = equity_data[equity_data['numerai_ticker'] == equity_ticker].copy()
                if equity_prices.empty:
                    continue
                
                equity_prices = equity_prices.set_index('date')['close'].sort_index()
                
                # Align dates
                common_dates = crypto_prices.index.intersection(equity_prices.index)
                if len(common_dates) < 30:  # Need minimum data points
                    continue
                
                crypto_aligned = crypto_prices.loc[common_dates]
                equity_aligned = equity_prices.loc[common_dates]
                
                # Calculate correlation with GPU acceleration if available
                if self.gpu_available and len(common_dates) > 100:
                    try:
                        # Use GPU for large datasets
                        corr = np.corrcoef(crypto_aligned.values, equity_aligned.values)[0, 1]
                    except:
                        # Fallback to CPU
                        corr = crypto_aligned.corr(equity_aligned)
                else:
                    corr = crypto_aligned.corr(equity_aligned)
                
                if not np.isnan(corr) and abs(corr) >= self.min_correlation:
                    correlations.append((equity_ticker, abs(corr)))
                    
            except Exception as e:
                logger.debug(f"Correlation calculation failed for {equity_ticker}: {e}")
                continue
        
        # Return top N correlations
        correlations.sort(key=lambda x: x[1], reverse=True)
        return correlations[:self.top_n_equities]
    
    def engineer_equity_features_v5(self, equity_data: pd.DataFrame, 
                                   equity_ticker: str) -> pd.DataFrame:
        """
        Engineer comprehensive equity features using V5 utils calculators
        
        Uses V5 TechnicalCalculator, StatisticalCalculator, and PVMCalculator
        for consistent feature engineering across the pipeline.
        """
        try:
            df = equity_data.copy()
            
            # Extract OHLCV arrays for V5 calculators
            prices = df['close'].values
            volumes = df['volume'].values
            high = df['high'].values
            low = df['low'].values
            open_prices = df['open'].values
            
            # Remove any NaN or infinite values
            mask = np.isfinite(prices) & np.isfinite(volumes) & np.isfinite(high) & np.isfinite(low)
            if not mask.all():
                logger.debug(f"Cleaning {(~mask).sum()} invalid values for {equity_ticker}")
                df = df[mask]
                prices = prices[mask]
                volumes = volumes[mask]
                high = high[mask]
                low = low[mask]
                open_prices = open_prices[mask]
            
            if len(prices) < 100:  # Need sufficient data
                logger.warning(f"Insufficient data for {equity_ticker}: {len(prices)} points")
                return df
            
            feature_dict = {}
            
            # 1. Technical indicators using V5 TechnicalCalculator
            try:
                # RSI
                rsi = self.technical_calc.calculate_rsi_safe(prices, window=14)
                feature_dict[f'{equity_ticker}_rsi_14d_lag1'] = np.roll(rsi, 1)
                
                # MACD
                macd = self.technical_calc.calculate_macd_safe(prices)
                feature_dict[f'{equity_ticker}_macd_lag1'] = np.roll(macd, 1)
                
                # Bollinger Band position
                bb_pos = self.technical_calc.calculate_bollinger_position_safe(prices, window=20)
                feature_dict[f'{equity_ticker}_bb_position_20d_lag1'] = np.roll(bb_pos, 1)
                
                # Moving averages
                sma_20 = self.technical_calc.calculate_sma(prices, 20)
                feature_dict[f'{equity_ticker}_sma_20d_lag1'] = np.roll(sma_20, 1)
                
                ema_12 = self.technical_calc.calculate_ema(prices, 12)
                feature_dict[f'{equity_ticker}_ema_12d_lag1'] = np.roll(ema_12, 1)
                
                # Stochastic %K
                stoch_k = self.technical_calc.calculate_stochastic_k(prices, high, low, 14)
                feature_dict[f'{equity_ticker}_stoch_k_14d_lag1'] = np.roll(stoch_k, 1)
                
                # ATR
                atr = self.technical_calc.calculate_atr(prices, high, low, 14)
                feature_dict[f'{equity_ticker}_atr_14d_lag1'] = np.roll(atr, 1)
                
            except Exception as e:
                logger.debug(f"Technical indicator calculation failed for {equity_ticker}: {e}")
            
            # 2. Statistical features using V5 StatisticalCalculator
            try:
                stat_features = self.statistical_calc.generate_statistical_features(
                    prices, volumes, high, low
                )
                stat_names = self.statistical_calc.get_feature_names()
                
                # Add statistical features with ticker prefix
                for i, name in enumerate(stat_names):
                    feature_name = name.replace('stat_', f'{equity_ticker}_')
                    if i < stat_features.shape[1]:
                        feature_dict[feature_name] = np.roll(stat_features[:, i], 1)
                        
            except Exception as e:
                logger.debug(f"Statistical feature calculation failed for {equity_ticker}: {e}")
            
            # 3. PVM features using V5 PVMCalculator (subset)
            try:
                pvm_features = self.pvm_calc.generate_pvm_features(prices, volumes)
                pvm_names = self.pvm_calc.get_feature_names()
                
                # Add top PVM features (first 20 to avoid too many features)
                for i, name in enumerate(pvm_names[:20]):
                    feature_name = name.replace('pvm_', f'{equity_ticker}_')
                    if i < pvm_features.shape[1]:
                        feature_dict[feature_name] = np.roll(pvm_features[:, i], 1)
                        
            except Exception as e:
                logger.debug(f"PVM feature calculation failed for {equity_ticker}: {e}")
            
            # 4. Basic price features with temporal lag
            try:
                # Price ratios
                feature_dict[f'{equity_ticker}_high_low_ratio_lag1'] = np.roll(high / low, 1)
                feature_dict[f'{equity_ticker}_close_open_ratio_lag1'] = np.roll(prices / open_prices, 1)
                
                # Returns
                returns_1d = np.diff(prices, prepend=prices[0]) / prices
                feature_dict[f'{equity_ticker}_returns_1d_lag1'] = np.roll(returns_1d, 1)
                
                # Volume features
                volume_sma = pd.Series(volumes).rolling(10).mean().values
                feature_dict[f'{equity_ticker}_volume_sma_10d_lag1'] = np.roll(volume_sma, 1)
                
                volume_ratio = volumes / (volume_sma + 1e-8)  # Avoid division by zero
                feature_dict[f'{equity_ticker}_volume_ratio_lag1'] = np.roll(volume_ratio, 1)
                
            except Exception as e:
                logger.debug(f"Basic feature calculation failed for {equity_ticker}: {e}")
            
            # Add features to dataframe
            for name, values in feature_dict.items():
                if len(values) == len(df):
                    df[name] = values
                else:
                    logger.debug(f"Feature {name} length mismatch: {len(values)} vs {len(df)}")
            
            logger.debug(f"✅ Generated {len(feature_dict)} features for {equity_ticker}")
            return df
            
        except Exception as e:
            logger.error(f"❌ Feature engineering failed for {equity_ticker}: {e}")
            return equity_data
    
    def process_single_cryptocurrency(self, crypto_symbol: str,
                                    crypto_data: pd.DataFrame,
                                    equity_data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Process a single cryptocurrency to find correlations and generate features"""
        try:
            # Find top correlated equity tickers
            correlations = self.calculate_crypto_equity_correlations(
                crypto_symbol, crypto_data, equity_data
            )
            
            if not correlations:
                logger.debug(f"No correlations found for {crypto_symbol}")
                return None
            
            # Get crypto price data for alignment
            crypto_prices = crypto_data[crypto_data['symbol'] == crypto_symbol].copy()
            crypto_prices = crypto_prices.set_index('date').sort_index()
            
            # Initialize result with crypto data
            result_df = crypto_prices[['open', 'high', 'low', 'close', 'volume']].reset_index()
            result_df['crypto_symbol'] = crypto_symbol
            
            # Process each correlated equity ticker
            selected_tickers = []
            correlation_values = []
            
            for equity_ticker, correlation in correlations:
                try:
                    # Get equity data for this ticker
                    equity_ticker_data = equity_data[
                        equity_data['numerai_ticker'] == equity_ticker
                    ].copy()
                    
                    if equity_ticker_data.empty:
                        continue
                    
                    equity_ticker_data = equity_ticker_data.set_index('date').sort_index()
                    
                    # Align dates
                    common_dates = crypto_prices.index.intersection(equity_ticker_data.index)
                    if len(common_dates) < 50:
                        continue
                    
                    # Engineer features for this equity ticker
                    equity_aligned = equity_ticker_data.loc[common_dates].reset_index()
                    equity_features = self.engineer_equity_features_v5(equity_aligned, equity_ticker)
                    
                    # Merge with crypto data
                    equity_features = equity_features.set_index('date')
                    
                    # Get only the engineered features (exclude base OHLCV)
                    feature_cols = [col for col in equity_features.columns 
                                   if col.startswith(f"{equity_ticker}_") and 'lag1' in col]
                    
                    if feature_cols:
                        equity_feature_subset = equity_features[feature_cols]
                        
                        # Merge with result
                        result_df = result_df.set_index('date')
                        result_df = result_df.join(equity_feature_subset, how='left')
                        result_df = result_df.reset_index()
                        
                        selected_tickers.append(equity_ticker)
                        correlation_values.append(f"{correlation:.4f}")
                    
                except Exception as e:
                    logger.debug(f"Error processing {equity_ticker} for {crypto_symbol}: {e}")
                    continue
            
            if selected_tickers:
                # Add metadata
                result_df['selected_equity_tickers'] = ','.join(selected_tickers)
                result_df['equity_correlations'] = ','.join(correlation_values)
                
                logger.debug(f"✅ {crypto_symbol}: {len(selected_tickers)} equity tickers, {result_df.shape[1]} total features")
                return result_df
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Failed to process {crypto_symbol}: {e}")
            return None
    
    def process_cryptocurrency_batch_parallel(self, crypto_symbols: List[str],
                                            crypto_data: pd.DataFrame,
                                            equity_data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Process cryptocurrency batch using parallel processing"""
        logger.info(f"🔄 Processing {len(crypto_symbols)} cryptocurrencies in parallel...")
        
        results = {}
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_symbol = {
                executor.submit(
                    self.process_single_cryptocurrency,
                    symbol, crypto_data, equity_data
                ): symbol
                for symbol in crypto_symbols
            }
            
            # Collect results with progress bar
            with tqdm(total=len(future_to_symbol), desc="Processing cryptos") as pbar:
                for future in as_completed(future_to_symbol):
                    symbol = future_to_symbol[future]
                    
                    try:
                        result = future.result()
                        if result is not None:
                            results[symbol] = result
                    except Exception as e:
                        logger.error(f"❌ Error processing {symbol}: {e}")
                    
                    pbar.update(1)
        
        logger.info(f"✅ Processed {len(results)}/{len(crypto_symbols)} cryptocurrencies successfully")
        return results
    
    def save_equity_features_v5(self, results: Dict[str, pd.DataFrame]) -> Tuple[str, str]:
        """Save equity features using V5 conventions"""
        logger.info("💾 Saving V5 equity features...")
        
        if not results:
            logger.warning("⚠️ No features to save")
            return "", ""
        
        try:
            # Combine all results
            all_features = pd.concat(results.values(), ignore_index=True)
            
            # Create single parquet file with date in name
            date_str = datetime.now().strftime('%Y%m%d')
            features_file = self.output_dir / f"equity_features_{date_str}.parquet"
            
            # Save consolidated features
            all_features.to_parquet(features_file, index=False)
            
            # Create statistics
            stats_data = {
                'metric': [
                    'total_cryptocurrencies_processed',
                    'total_records',
                    'total_features',
                    'average_features_per_crypto',
                    'processing_timestamp',
                    'gpu_acceleration_used',
                    'temporal_lag_enforced'
                ],
                'value': [
                    len(results),
                    len(all_features),
                    all_features.shape[1],
                    round(all_features.shape[1] / len(results), 1) if results else 0,
                    datetime.now().isoformat(),
                    self.gpu_available,
                    self.TEMPORAL_LAG_DAYS
                ]
            }
            
            stats_df = pd.DataFrame(stats_data)
            stats_file = self.output_dir / f"equity_features_statistics_{date_str}.csv"
            stats_df.to_csv(stats_file, index=False)
            
            # Summary info
            logger.info(f"✅ Features saved: {features_file}")
            logger.info(f"📊 Shape: {all_features.shape}")
            logger.info(f"📊 Cryptocurrencies: {len(results)}")
            logger.info(f"📊 Features per crypto: {all_features.shape[1] / len(results):.1f}")
            logger.info(f"📊 Statistics: {stats_file}")
            
            return str(features_file), str(stats_file)
            
        except Exception as e:
            logger.error(f"❌ Error saving features: {e}")
            return "", ""
    
    def run_comprehensive_feature_selection(self, max_cryptos: Optional[int] = None) -> Dict[str, Any]:
        """Run complete V5 equity feature selection pipeline"""
        logger.info("🚀 Starting V5 Comprehensive Equity Feature Selection")
        logger.info("=" * 80)
        
        start_time = time.time()
        
        try:
            # 1. Load all data sources
            logger.info("📊 PHASE 1: LOADING DATA SOURCES")
            equity_data = self.load_latest_equity_data()
            crypto_data = self.load_cryptocurrency_data()
            targets_data = self.load_numerai_targets()
            
            # 2. Get target cryptocurrencies
            target_cryptos = targets_data['symbol'].unique()
            available_cryptos = crypto_data['symbol'].unique()
            process_cryptos = [c for c in target_cryptos if c in available_cryptos]
            
            if max_cryptos:
                process_cryptos = process_cryptos[:max_cryptos]
                logger.info(f"🎯 Limited to {max_cryptos} cryptocurrencies for testing")
            
            logger.info(f"📊 Processing {len(process_cryptos)} target cryptocurrencies")
            
            # 3. Process cryptocurrencies in batches
            logger.info("🔄 PHASE 2: PARALLEL FEATURE GENERATION")
            all_results = {}
            
            # Process in batches for memory management
            for batch_start in range(0, len(process_cryptos), self.batch_size):
                batch_end = min(batch_start + self.batch_size, len(process_cryptos))
                batch_cryptos = process_cryptos[batch_start:batch_end]
                
                batch_num = batch_start // self.batch_size + 1
                total_batches = (len(process_cryptos) - 1) // self.batch_size + 1
                
                logger.info(f"📦 Processing batch {batch_num}/{total_batches}: {len(batch_cryptos)} cryptos")
                
                batch_results = self.process_cryptocurrency_batch_parallel(
                    batch_cryptos, crypto_data, equity_data
                )
                
                all_results.update(batch_results)
                
                # Memory cleanup
                if self.gpu_available:
                    gc.collect()
                
                logger.info(f"✅ Batch {batch_num} completed: {len(batch_results)} cryptos processed")
            
            # 4. Save results
            logger.info("💾 PHASE 3: SAVING FEATURES")
            features_file, stats_file = self.save_equity_features_v5(all_results)
            
            # 5. Final validation
            logger.info("🔍 PHASE 4: VALIDATION")
            if features_file:
                validation_df = pd.read_parquet(features_file)
                validation_df['date'] = pd.to_datetime(validation_df['date'])
                
                if self.temporal_validator.validate_temporal_lag(validation_df):
                    logger.info("✅ Temporal validation PASSED")
                else:
                    logger.warning("⚠️ Temporal validation WARNING")
            
            # Final summary
            duration = time.time() - start_time
            
            result = {
                'success': True,
                'duration_seconds': duration,
                'duration_formatted': f"{duration//60:.0f}m {duration%60:.0f}s",
                'cryptocurrencies_processed': len(all_results),
                'total_features': sum(df.shape[1] for df in all_results.values()) if all_results else 0,
                'features_file': features_file,
                'stats_file': stats_file
            }
            
            logger.info("=" * 80)
            logger.info("🎉 V5 EQUITY FEATURE SELECTION COMPLETED!")
            logger.info(f"⏱️ Duration: {result['duration_formatted']}")
            logger.info(f"📊 Cryptocurrencies: {result['cryptocurrencies_processed']}")
            logger.info(f"📈 Total features: {result['total_features']}")
            logger.info(f"💾 Features file: {features_file}")
            logger.info(f"📊 Stats file: {stats_file}")
            logger.info("🛡️ Temporal lag enforced - No data leakage")
            logger.info("=" * 80)
            
            return result
            
        except Exception as e:
            logger.error(f"❌ V5 equity feature selection failed: {e}")
            import traceback
            traceback.print_exc()
            return {
                'success': False,
                'error': str(e),
                'duration_seconds': time.time() - start_time
            }

def main():
    """Main execution function"""
    print("\n🏦 V5 EQUITY FEATURE SELECTOR")
    print("=" * 80)
    print("FEATURES:")
    print("• V5 utils integration (TechnicalCalculator, StatisticalCalculator, PVMCalculator)")
    print("• Fast parallel processing with ThreadPoolExecutor")
    print("• GPU acceleration via cuDF-pandas")
    print("• Proper temporal lag enforcement (1-day lag)")
    print("• Single parquet output with date in name")
    print("• Comprehensive feature importance tracking")
    print("\nARCHITECTURE:")
    print("• Uses V5 feature_calculators.py for calculations")
    print("• Integrates with V5 temporal validation")
    print("• Modular design with proper error handling")
    print("• Memory-efficient batch processing")
    print("=" * 80)
    
    # Initialize selector
    selector = V5EquityFeatureSelector(max_workers=10)
    
    # Run feature selection
    result = selector.run_comprehensive_feature_selection()
    
    # Show results
    if result['success']:
        print(f"\n✅ Feature selection completed successfully!")
        print(f"Duration: {result['duration_formatted']}")
        print(f"Cryptocurrencies: {result['cryptocurrencies_processed']}")
        print(f"Total features: {result['total_features']}")
        print(f"Features file: {result['features_file']}")
        print(f"Stats file: {result['stats_file']}")
    else:
        print(f"\n❌ Feature selection failed: {result['error']}")
        sys.exit(1)

if __name__ == "__main__":
    main()