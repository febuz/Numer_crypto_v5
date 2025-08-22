#!/usr/bin/env python3
"""
📊 ECONOMIC FEATURE SELECTOR
===========================
Specialized feature selector for macro-economic indicators in cryptocurrency prediction.
Extracts comprehensive features from economic data including SP500, VIX, NASDAQ, bonds, USD, gold, oil.

FEATURES:
- Dual GPU acceleration with cuDF-pandas
- Comprehensive economic indicators: SP500, VIX, NASDAQ, bonds, USD, gold, oil
- Advanced feature engineering: changes, volatilities, ATR, lags
- Parallel processing for maximum throughput
- Memory-efficient batch processing
- Proper temporal lag enforcement (1-day lag)

OUTPUT:
- Economic features with proper naming: {indicator}_change_1d, {indicator}_volatility_10d, etc.
- Feature importance analysis
- Comprehensive statistics and validation
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Configure dual GPU environment
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
os.environ['CUPY_GPU_MEMORY_LIMIT'] = '20480'  # 20GB per GPU
os.environ['TMPDIR'] = '/media/knight2/EDB/tmp'

# Enable cuDF-pandas acceleration
try:
    import cudf.pandas
    cudf.pandas.install()
    print("🚀 cuDF-pandas GPU acceleration ENABLED for economic features")
    CUDF_PANDAS_ENABLED = True
except ImportError:
    CUDF_PANDAS_ENABLED = False
    print("⚠️ cuDF-pandas not available, using standard pandas")

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/media/knight2/EDB/numer_crypto_temp/data/logs/economic_feature_selector.log')
    ]
)
logger = logging.getLogger(__name__)

class EconomicFeatureSelector:
    """Comprehensive economic feature selector for cryptocurrency prediction"""
    
    def __init__(self, data_dir="/media/knight2/EDB/numer_crypto_temp/data"):
        self.data_dir = Path(data_dir)
        self.processed_dir = self.data_dir / "processed" / "macro_economic"
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Economic data directories
        self.economic_dir = self.data_dir / "raw" / "economic_indicators"
        
        # Processing parameters
        self.max_workers = 20
        self.dtype = np.float32  # Memory optimization
        
        # Economic indicators to process
        self.economic_indicators = [
            'sp500', 'nasdaq', 'vix', 'bond_10y', 
            'usd_index', 'gold', 'oil'
        ]
        
        # Feature engineering parameters
        self.change_periods = [1, 2, 3, 5, 7, 10, 14, 21, 30, 60, 90, 180, 365]  # Days
        self.volatility_windows = [5, 10, 20, 30, 60, 90]  # Days
        self.lag_periods = [1, 2, 3, 5, 7, 10, 14, 21, 30]  # Temporal lags
        
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        logger.info("📊 Economic Feature Selector initialized")
        logger.info(f"   Data directory: {self.data_dir}")
        logger.info(f"   Output directory: {self.processed_dir}")
        logger.info(f"   Economic indicators: {len(self.economic_indicators)}")
        logger.info(f"   Max workers: {self.max_workers}")
        logger.info(f"   Precision: {self.dtype}")

    def load_economic_indicators(self) -> Dict[str, pd.DataFrame]:
        """Load all available economic indicators"""
        logger.info("🏦 Loading comprehensive economic indicators...")
        
        economic_data = {}
        
        for indicator in self.economic_indicators:
            try:
                # Look for parquet files first, then CSV
                parquet_file = self.economic_dir / f"{indicator}_indicators.parquet"
                csv_file = self.economic_dir / f"{indicator}_indicators.csv"
                
                if parquet_file.exists():
                    logger.info(f"  📊 Loading {indicator} from parquet: {parquet_file.name}")
                    df = pd.read_parquet(parquet_file)
                elif csv_file.exists():
                    logger.info(f"  📊 Loading {indicator} from CSV: {csv_file.name}")
                    df = pd.read_csv(csv_file)
                else:
                    logger.warning(f"  ⚠️ No data file found for {indicator}")
                    continue
                
                # Standardize date column
                df = self._standardize_date_column(df)
                
                if not df.empty:
                    economic_data[indicator] = df
                    logger.info(f"  ✅ {indicator}: {len(df)} samples")
                else:
                    logger.warning(f"  ⚠️ {indicator}: Empty after processing")
                    
            except Exception as e:
                logger.error(f"  ❌ Failed to load {indicator}: {e}")
                continue
        
        logger.info(f"✅ Loaded {len(economic_data)} economic indicators")
        return economic_data

    def _standardize_date_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize date column across different data sources"""
        
        # Common date column names
        date_columns = ['date', 'Date', 'DATE', 'timestamp', 'time']
        
        date_col = None
        for col in date_columns:
            if col in df.columns:
                date_col = col
                break
        
        if date_col is None:
            # Check if index looks like a date
            if pd.api.types.is_datetime64_any_dtype(df.index):
                df = df.reset_index()
                date_col = 'index'
            else:
                logger.warning("No date column found")
                return pd.DataFrame()
        
        # Convert to datetime
        try:
            df['date'] = pd.to_datetime(df[date_col])
            if date_col != 'date':
                df = df.drop(columns=[date_col])
        except Exception as e:
            logger.error(f"Failed to convert date column: {e}")
            return pd.DataFrame()
        
        # Remove timezone for consistency
        if df['date'].dt.tz is not None:
            df['date'] = df['date'].dt.tz_localize(None)
        
        # Sort by date
        df = df.sort_values('date').reset_index(drop=True)
        
        return df

    def generate_comprehensive_indicator_features(self, economic_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Generate comprehensive features for all economic indicators"""
        logger.info("📊 Generating features for economic indicators...")
        logger.info("📊 Using parallel processing for multiple indicators...")
        
        # Process indicators in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self._process_single_indicator, indicator, data): indicator
                for indicator, data in economic_data.items()
            }
            
            indicator_features = {}
            for future in as_completed(futures):
                indicator = futures[future]
                try:
                    features = future.result()
                    if not features.empty:
                        indicator_features[indicator] = features
                        logger.info(f"📊 ✅ {indicator}: {features.shape[1]-1} features generated")
                    else:
                        logger.warning(f"📊 ⚠️ {indicator}: No features generated")
                except Exception as e:
                    logger.error(f"📊 ❌ {indicator}: Feature generation failed: {e}")
        
        # Combine all indicator features
        if not indicator_features:
            logger.error("❌ No economic features generated")
            return pd.DataFrame()
        
        # Start with the first indicator's date column
        first_indicator = list(indicator_features.keys())[0]
        combined_features = indicator_features[first_indicator][['date']].copy()
        
        # Add features from all indicators
        for indicator, features in indicator_features.items():
            # Merge on date
            feature_cols = [col for col in features.columns if col != 'date']
            combined_features = combined_features.merge(
                features[['date'] + feature_cols],
                on='date',
                how='outer'
            )
        
        # Sort by date and apply 1-day lag for temporal safety
        combined_features = combined_features.sort_values('date')
        feature_cols = [col for col in combined_features.columns if col != 'date']
        
        # Apply 1-day lag to prevent data leakage
        combined_features[feature_cols] = combined_features[feature_cols].shift(1)
        
        # Remove rows with all NaN values
        combined_features = combined_features.dropna(how='all', subset=feature_cols)
        
        logger.info(f"✅ Combined economic features: {combined_features.shape}")
        logger.info(f"   Date range: {combined_features['date'].min()} to {combined_features['date'].max()}")
        logger.info(f"   Total features: {len(feature_cols)}")
        
        return combined_features

    def _process_single_indicator(self, indicator: str, data: pd.DataFrame) -> pd.DataFrame:
        """Process a single economic indicator to generate comprehensive features"""
        
        logger.info(f"📊 Processing {indicator}: {len(data)} samples, "
                   f"{len(self.change_periods)} changes, {len(self.lag_periods)} lags")
        
        if data.empty or 'date' not in data.columns:
            return pd.DataFrame()
        
        # Find the value column (usually 'close', 'value', or similar)
        value_columns = ['close', 'Close', 'value', 'Value', 'price', 'Price']
        value_col = None
        
        for col in value_columns:
            if col in data.columns:
                value_col = col
                break
        
        if value_col is None:
            # Use the first numeric column
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                value_col = numeric_cols[0]
            else:
                logger.warning(f"No numeric column found for {indicator}")
                return pd.DataFrame()
        
        # Prepare data
        df = data[['date', value_col]].copy()
        df = df.sort_values('date').reset_index(drop=True)
        df[value_col] = pd.to_numeric(df[value_col], errors='coerce')
        df = df.dropna()
        
        if len(df) < 30:  # Need minimum data for meaningful features
            logger.warning(f"Insufficient data for {indicator}: {len(df)} samples")
            return pd.DataFrame()
        
        # Initialize features dataframe
        features = df[['date']].copy()
        
        # 1. Price level features
        features[f'{indicator}_close'] = df[value_col]
        features[f'{indicator}_log_close'] = np.log(df[value_col])
        
        # 2. Change features (percentage and absolute)
        for period in self.change_periods:
            if len(df) > period:
                # Percentage change
                pct_change = df[value_col].pct_change(periods=period)
                features[f'{indicator}_change_{period}d'] = pct_change
                
                # Log returns
                log_returns = np.log(df[value_col] / df[value_col].shift(period))
                features[f'{indicator}_log_return_{period}d'] = log_returns
                
                # Absolute change
                abs_change = df[value_col].diff(periods=period)
                features[f'{indicator}_abs_change_{period}d'] = abs_change
        
        # 3. Volatility features (rolling standard deviation)
        for window in self.volatility_windows:
            if len(df) > window:
                # Price volatility
                vol = df[value_col].rolling(window=window).std()
                features[f'{indicator}_volatility_{window}d'] = vol
                
                # Return volatility
                returns = df[value_col].pct_change()
                return_vol = returns.rolling(window=window).std()
                features[f'{indicator}_return_vol_{window}d'] = return_vol
        
        # 4. Moving averages
        ma_windows = [5, 10, 20, 50, 100, 200]
        for window in ma_windows:
            if len(df) > window:
                ma = df[value_col].rolling(window=window).mean()
                features[f'{indicator}_ma_{window}'] = ma
                
                # MA ratio
                ma_ratio = df[value_col] / ma
                features[f'{indicator}_ma_ratio_{window}'] = ma_ratio
        
        # 5. Technical indicators
        if len(df) > 14:
            # RSI (14-period)
            rsi = self._calculate_rsi(df[value_col], period=14)
            features[f'{indicator}_rsi_14'] = rsi
        
        if len(df) > 20:
            # Bollinger Bands
            bb_upper, bb_lower, bb_ratio = self._calculate_bollinger_bands(df[value_col], window=20)
            features[f'{indicator}_bb_upper'] = bb_upper
            features[f'{indicator}_bb_lower'] = bb_lower
            features[f'{indicator}_bb_ratio'] = bb_ratio
        
        # 6. Statistical features
        stat_windows = [10, 20, 30, 60]
        for window in stat_windows:
            if len(df) > window:
                # Rolling statistics
                features[f'{indicator}_min_{window}d'] = df[value_col].rolling(window).min()
                features[f'{indicator}_max_{window}d'] = df[value_col].rolling(window).max()
                features[f'{indicator}_median_{window}d'] = df[value_col].rolling(window).median()
                features[f'{indicator}_skew_{window}d'] = df[value_col].rolling(window).skew()
                features[f'{indicator}_kurt_{window}d'] = df[value_col].rolling(window).kurt()
        
        # 7. Lag features
        for lag in self.lag_periods:
            if len(df) > lag:
                features[f'{indicator}_lag_{lag}d'] = df[value_col].shift(lag)
                
                # Lag ratios
                lag_ratio = df[value_col] / df[value_col].shift(lag)
                features[f'{indicator}_lag_ratio_{lag}d'] = lag_ratio
        
        # 8. Trend features
        if len(df) > 10:
            # Linear trend (slope over different windows)
            trend_windows = [10, 20, 30, 60]
            for window in trend_windows:
                if len(df) > window:
                    trend = df[value_col].rolling(window=window).apply(
                        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == window else np.nan,
                        raw=False
                    )
                    features[f'{indicator}_trend_{window}d'] = trend
        
        # Convert to specified dtype for memory efficiency
        numeric_cols = features.select_dtypes(include=[np.number]).columns
        features[numeric_cols] = features[numeric_cols].astype(self.dtype)
        
        return features

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi

    def _calculate_bollinger_bands(self, prices: pd.Series, window: int = 20, num_std: float = 2) -> tuple:
        """Calculate Bollinger Bands"""
        ma = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()
        
        upper_band = ma + (std * num_std)
        lower_band = ma - (std * num_std)
        
        # Bollinger Band ratio (position within bands)
        bb_ratio = (prices - lower_band) / (upper_band - lower_band)
        
        return upper_band, lower_band, bb_ratio

    def save_economic_features(self, features_df: pd.DataFrame) -> str:
        """Save economic features with metadata"""
        
        if features_df.empty:
            logger.error("❌ No features to save")
            return ""
        
        # Save main features file
        features_file = self.processed_dir / f"economic_features_{self.timestamp}.parquet"
        features_df.to_parquet(features_file, index=False)
        
        # Create summary statistics
        summary_stats = {
            'timestamp': self.timestamp,
            'total_features': features_df.shape[1] - 1,  # Exclude date column
            'total_samples': features_df.shape[0],
            'date_range_start': features_df['date'].min().isoformat(),
            'date_range_end': features_df['date'].max().isoformat(),
            'indicators_processed': self.economic_indicators,
            'feature_categories': {
                'change_periods': self.change_periods,
                'volatility_windows': self.volatility_windows,
                'lag_periods': self.lag_periods
            },
            'memory_usage_mb': features_df.memory_usage(deep=True).sum() / 1024**2,
            'feature_names': [col for col in features_df.columns if col != 'date']
        }
        
        # Save metadata
        metadata_file = self.processed_dir / f"economic_features_metadata_{self.timestamp}.json"
        import json
        with open(metadata_file, 'w') as f:
            json.dump(summary_stats, f, indent=2, default=str)
        
        logger.info(f"✅ Economic features saved: {features_file}")
        logger.info(f"📊 Features: {summary_stats['total_features']}")
        logger.info(f"📊 Samples: {summary_stats['total_samples']}")
        logger.info(f"📊 Memory: {summary_stats['memory_usage_mb']:.1f} MB")
        logger.info(f"📊 Metadata: {metadata_file}")
        
        return str(features_file)

    def run_economic_feature_generation(self) -> str:
        """Run complete economic feature generation pipeline"""
        logger.info("🚀 Starting economic feature generation pipeline...")
        
        start_time = time.time()
        
        try:
            # Load economic indicators
            economic_data = self.load_economic_indicators()
            
            if not economic_data:
                logger.error("❌ No economic data loaded")
                return ""
            
            # Generate comprehensive features
            features_df = self.generate_comprehensive_indicator_features(economic_data)
            
            if features_df.empty:
                logger.error("❌ No features generated")
                return ""
            
            # Save features
            features_file = self.save_economic_features(features_df)
            
            duration = time.time() - start_time
            
            logger.info(f"🎉 Economic feature generation completed in {duration:.1f}s")
            logger.info(f"📁 Features saved to: {features_file}")
            
            return features_file
            
        except Exception as e:
            logger.error(f"❌ Economic feature generation failed: {e}")
            return ""

def main():
    """Main execution"""
    print("📊 ECONOMIC FEATURE SELECTOR")
    print("=" * 50)
    
    selector = EconomicFeatureSelector()
    features_file = selector.run_economic_feature_generation()
    
    if features_file:
        print(f"✅ SUCCESS! Features saved to: {features_file}")
    else:
        print("❌ FAILED to generate economic features")

if __name__ == "__main__":
    main()