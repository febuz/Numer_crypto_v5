#!/usr/bin/env python3
"""
V5 Technical Feature Selector - Essential for Weekly Training & Daily Inference
Based on analysis: rsi_30 (485.02), close_ma_30d (421.69), macd_30d (338.93)

ESSENTIAL TECHNICAL INDICATORS ONLY
"""

import os
import sys
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import numba as nb

warnings.filterwarnings('ignore')

def setup_logging() -> logging.Logger:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path("/media/knight2/EDB/numer_crypto_temp/data/log")
    log_dir.mkdir(exist_ok=True)
    
    logger = logging.getLogger("v5_technical")
    logger.setLevel(logging.INFO)
    
    file_handler = logging.FileHandler(log_dir / f"technical_selector_{timestamp}.log")
    console_handler = logging.StreamHandler()
    
    for handler in [file_handler, console_handler]:
        handler.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
        logger.addHandler(handler)
    
    return logger

logger = setup_logging()

@nb.jit(nopython=True, fastmath=True)
def calculate_rsi(prices: np.ndarray, window: int = 14) -> np.ndarray:
    """Calculate RSI with Numba optimization"""
    n = len(prices)
    rsi = np.empty(n)
    rsi[:window] = 50.0  # Default RSI for insufficient data
    
    for i in range(window, n):
        price_slice = prices[i-window+1:i+1]
        changes = np.diff(price_slice)
        
        gains = np.where(changes > 0, changes, 0.0)
        losses = np.where(changes < 0, -changes, 0.0)
        
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        
        if avg_loss == 0:
            rsi[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi

@nb.jit(nopython=True, fastmath=True)
def calculate_macd(prices: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> np.ndarray:
    """Calculate MACD with Numba optimization"""
    n = len(prices)
    macd = np.empty(n)
    
    # Calculate EMA
    alpha_fast = 2.0 / (fast + 1.0)
    alpha_slow = 2.0 / (slow + 1.0)
    
    ema_fast = np.empty(n)
    ema_slow = np.empty(n)
    
    ema_fast[0] = prices[0]
    ema_slow[0] = prices[0]
    
    for i in range(1, n):
        ema_fast[i] = alpha_fast * prices[i] + (1 - alpha_fast) * ema_fast[i-1]
        ema_slow[i] = alpha_slow * prices[i] + (1 - alpha_slow) * ema_slow[i-1]
    
    macd = ema_fast - ema_slow
    return macd

@nb.jit(nopython=True, fastmath=True)
def calculate_bollinger_bands(prices: np.ndarray, window: int = 20, std_mult: float = 2.0) -> np.ndarray:
    """Calculate Bollinger Band position"""
    n = len(prices)
    bb_position = np.empty(n)
    
    for i in range(n):
        start_idx = max(0, i - window + 1)
        price_window = prices[start_idx:i+1]
        
        if len(price_window) >= 2:
            mean_price = np.mean(price_window)
            std_price = np.std(price_window)
            
            if std_price > 0:
                upper_band = mean_price + std_mult * std_price
                lower_band = mean_price - std_mult * std_price
                bb_position[i] = (prices[i] - lower_band) / (upper_band - lower_band)
            else:
                bb_position[i] = 0.5
        else:
            bb_position[i] = 0.5
    
    return bb_position

class V5TechnicalFeatureSelector:
    """Essential Technical Indicators for V5 - Weekly Training & Daily Inference"""
    
    def __init__(self):
        self.data_dir = Path("/media/knight2/EDB/numer_crypto_temp/data")
        self.processed_dir = self.data_dir / "processed" / "technical"
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Essential parameters based on top performers
        self.sma_windows = [5, 10, 20, 30, 50]
        self.ema_windows = [5, 10, 20, 30]
        self.rsi_windows = [14, 21, 30]
        self.bb_windows = [10, 20, 30]
        
        logger.info("🚀 V5 Technical Feature Selector initialized")

    def generate_technical_features(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """Generate essential technical indicators with 1-day lag"""
        logger.info("🔧 Generating essential technical indicators...")
        
        all_features = []
        symbols = price_data['symbol'].unique()
        
        for symbol in symbols:
            symbol_data = price_data[price_data['symbol'] == symbol].copy()
            
            if len(symbol_data) < 60:  # Need sufficient data for technical indicators
                continue
            
            prices = symbol_data['close'].values.astype(np.float64)
            high = symbol_data.get('high', symbol_data['close']).values.astype(np.float64)
            low = symbol_data.get('low', symbol_data['close']).values.astype(np.float64)
            volume = symbol_data['volume'].values.astype(np.float64)
            
            # Handle missing values
            prices = np.nan_to_num(prices, nan=np.median(prices[prices > 0]))
            high = np.nan_to_num(high, nan=np.median(high[high > 0]))
            low = np.nan_to_num(low, nan=np.median(low[low > 0]))
            volume = np.nan_to_num(volume, nan=np.median(volume[volume > 0]))
            
            feature_dict = {
                'symbol': symbol,
                'date': symbol_data['date'].values
            }
            
            # 1. Simple Moving Averages (SMA) - Essential for trend following
            for window in self.sma_windows:
                sma = pd.Series(prices).rolling(window=window, min_periods=1).mean().values
                feature_dict[f'sma_{window}d_lag1'] = sma
                
                # SMA ratio (price relative to SMA)
                sma_ratio = prices / np.where(sma > 0, sma, 1.0)
                feature_dict[f'sma_ratio_{window}d_lag1'] = sma_ratio
            
            # 2. Exponential Moving Averages (EMA) - More responsive to recent changes
            for window in self.ema_windows:
                alpha = 2.0 / (window + 1.0)
                ema = np.empty_like(prices)
                ema[0] = prices[0]
                for i in range(1, len(prices)):
                    ema[i] = alpha * prices[i] + (1 - alpha) * ema[i-1]
                feature_dict[f'ema_{window}d_lag1'] = ema
            
            # 3. RSI - Essential momentum oscillator (top performer: rsi_30)
            for window in self.rsi_windows:
                rsi = calculate_rsi(prices, window)
                feature_dict[f'rsi_{window}d_lag1'] = rsi
            
            # 4. MACD - Trend and momentum indicator (top performer: macd_30d)
            macd = calculate_macd(prices, fast=12, slow=26, signal=9)
            feature_dict['macd_lag1'] = macd
            
            # MACD with different parameters for 30-day analysis
            macd_30 = calculate_macd(prices, fast=8, slow=21, signal=9)
            feature_dict['macd_30d_lag1'] = macd_30
            
            # 5. Bollinger Bands - Volatility and mean reversion
            for window in self.bb_windows:
                bb_pos = calculate_bollinger_bands(prices, window)
                feature_dict[f'bb_position_{window}d_lag1'] = bb_pos
            
            # 6. Price position features
            for window in [10, 20, 50]:
                rolling_max = pd.Series(high).rolling(window=window, min_periods=1).max().values
                rolling_min = pd.Series(low).rolling(window=window, min_periods=1).min().values
                
                # Stochastic %K
                stoch_k = np.where(
                    rolling_max > rolling_min,
                    (prices - rolling_min) / (rolling_max - rolling_min) * 100,
                    50.0
                )
                feature_dict[f'stoch_k_{window}d_lag1'] = stoch_k
            
            # 7. Volume-based indicators
            if np.sum(volume) > 0:
                # Volume moving average
                vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=1).mean().values
                feature_dict['volume_ma_20d_lag1'] = vol_ma_20
                
                # Volume ratio
                vol_ratio = volume / np.where(vol_ma_20 > 0, vol_ma_20, 1.0)
                feature_dict['volume_ratio_lag1'] = vol_ratio
            else:
                feature_dict['volume_ma_20d_lag1'] = np.zeros_like(prices)
                feature_dict['volume_ratio_lag1'] = np.ones_like(prices)
            
            # 8. Volatility indicators
            # True Range and ATR
            tr = np.maximum(high - low, 
                           np.maximum(np.abs(high - np.roll(prices, 1)), 
                                    np.abs(low - np.roll(prices, 1))))
            atr = pd.Series(tr).rolling(window=14, min_periods=1).mean().values
            feature_dict['atr_14d_lag1'] = atr
            
            # Create DataFrame
            feature_df = pd.DataFrame(feature_dict)
            
            # Apply 1-day lag to ALL technical features (CRITICAL for no data leakage)
            feature_cols = [col for col in feature_df.columns if col.endswith('_lag1')]
            for col in feature_cols:
                feature_df[col] = feature_df[col].shift(1)
            
            # Remove first row (NaN due to lag)
            feature_df = feature_df.iloc[1:].copy()
            all_features.append(feature_df)
        
        if not all_features:
            raise ValueError("No technical features generated")
        
        combined_features = pd.concat(all_features, ignore_index=True)
        
        # Clean data
        numeric_cols = combined_features.select_dtypes(include=[np.number]).columns
        combined_features[numeric_cols] = combined_features[numeric_cols].replace([np.inf, -np.inf], np.nan)
        combined_features[numeric_cols] = combined_features[numeric_cols].fillna(0)
        
        logger.info(f"✅ Technical features generated: {len(combined_features)} rows")
        logger.info(f"🔢 Features: {len([col for col in combined_features.columns if col.endswith('_lag1')])}")
        
        return combined_features

    def run_technical_feature_generation(self) -> str:
        """Run essential technical feature generation"""
        logger.info("🚀 Starting V5 Technical Feature Generation")
        
        try:
            # Load price data
            price_files = list(self.data_dir.glob("raw/price/**/crypto_price_data_*.parquet"))
            if not price_files:
                raise FileNotFoundError("No price data found")
            
            latest_file = max(price_files, key=lambda x: x.stat().st_mtime)
            price_data = pd.read_parquet(latest_file)
            price_data['date'] = pd.to_datetime(price_data['date'])
            
            # Apply temporal lag
            cutoff_date = datetime.now() - timedelta(days=1)
            price_data = price_data[price_data['date'] <= cutoff_date]
            price_data = price_data.sort_values(['symbol', 'date']).reset_index(drop=True)
            
            # Generate features
            features_df = self.generate_technical_features(price_data)
            
            # Save features
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            feature_file = self.processed_dir / f"technical_features_{timestamp}.parquet"
            features_df.to_parquet(feature_file, index=False)
            
            # Save metadata
            metadata = {
                'timestamp': timestamp,
                'feature_count': len([col for col in features_df.columns if col.endswith('_lag1')]),
                'records': len(features_df),
                'symbols': features_df['symbol'].nunique(),
                'temporal_lag_days': 1,
                'essential_indicators': ['SMA', 'EMA', 'RSI', 'MACD', 'Bollinger_Bands', 'Stochastic', 'ATR'],
                'essential_for': ['weekly_training', 'daily_inference']
            }
            
            import json
            metadata_file = self.processed_dir / f"technical_metadata_{timestamp}.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            logger.info(f"✅ Technical features saved: {feature_file}")
            return str(feature_file)
            
        except Exception as e:
            logger.error(f"❌ Technical feature generation failed: {e}")
            raise

def main():
    selector = V5TechnicalFeatureSelector()
    feature_file = selector.run_technical_feature_generation()
    print(f"🎉 Technical features generated: {feature_file}")

if __name__ == "__main__":
    main()