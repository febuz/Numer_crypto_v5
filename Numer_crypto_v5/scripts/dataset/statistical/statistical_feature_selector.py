#!/usr/bin/env python3
"""
V5 Statistical Feature Selector - Essential for Weekly Training & Daily Inference
Based on analysis: volume_volatility_30d (746.83), price_volatility_30d (626.91)

ESSENTIAL FEATURES ONLY - NO ABANDONED SCRIPTS
"""

import os
import sys
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import numba as nb

warnings.filterwarnings('ignore')

def setup_logging() -> logging.Logger:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path("/media/knight2/EDB/numer_crypto_temp/data/log")
    log_dir.mkdir(exist_ok=True)
    
    logger = logging.getLogger("v5_statistical")
    logger.setLevel(logging.INFO)
    
    file_handler = logging.FileHandler(log_dir / f"statistical_selector_{timestamp}.log")
    console_handler = logging.StreamHandler()
    
    for handler in [file_handler, console_handler]:
        handler.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
        logger.addHandler(handler)
    
    return logger

logger = setup_logging()

def calculate_statistical_features_safe(prices: np.ndarray, volumes: np.ndarray, 
                                       high: np.ndarray, low: np.ndarray,
                                       windows: np.ndarray) -> np.ndarray:
    """Safe statistical feature calculation without Numba JIT to avoid segfaults"""
    n = len(prices)
    n_features = len(windows) * 8  # 8 statistical features per window
    features = np.zeros((n, n_features))
    
    for i in range(n):
        for w_idx, window in enumerate(windows):
            start_idx = max(0, i - window + 1)
            
            price_window = prices[start_idx:i+1]
            vol_window = volumes[start_idx:i+1]
            high_window = high[start_idx:i+1]
            low_window = low[start_idx:i+1]
            
            base_idx = w_idx * 8
            
            if len(price_window) > 1 and not np.isnan(price_window).all():
                # Top performing statistical features from analysis
                price_volatility = np.nanstd(price_window) if len(price_window) > 1 else 0.0
                volume_volatility = np.nanstd(vol_window) if len(vol_window) > 1 else 0.0
                price_mean = np.nanmean(price_window)
                volume_mean = np.nanmean(vol_window)
                
                # Z-score calculation with safety
                if np.nanstd(price_window) > 0 and not np.isnan(price_mean):
                    z_score = (prices[i] - price_mean) / np.nanstd(price_window)
                else:
                    z_score = 0.0
                
                # Range ratio calculation with safety
                if len(high_window) > 0 and len(low_window) > 0 and price_mean > 0:
                    range_ratio = (np.nanmax(high_window) - np.nanmin(low_window)) / price_mean
                else:
                    range_ratio = 0.0
                
                # High-Low volatility
                if len(high_window) > 1 and len(low_window) > 1:
                    hl_volatility = np.nanstd(high_window - low_window)
                else:
                    hl_volatility = 0.0
                
                # Price-Volume correlation with safety
                if len(price_window) > 2 and len(vol_window) > 2:
                    try:
                        corr_matrix = np.corrcoef(price_window, vol_window)
                        pv_correlation = corr_matrix[0, 1] if not np.isnan(corr_matrix[0, 1]) else 0.0
                    except:
                        pv_correlation = 0.0
                else:
                    pv_correlation = 0.0
                
                # Store with NaN safety
                features[i, base_idx] = price_volatility if not np.isnan(price_volatility) else 0.0
                features[i, base_idx + 1] = volume_volatility if not np.isnan(volume_volatility) else 0.0
                features[i, base_idx + 2] = price_mean if not np.isnan(price_mean) else 0.0
                features[i, base_idx + 3] = volume_mean if not np.isnan(volume_mean) else 0.0
                features[i, base_idx + 4] = z_score if not np.isnan(z_score) else 0.0
                features[i, base_idx + 5] = range_ratio if not np.isnan(range_ratio) else 0.0
                features[i, base_idx + 6] = hl_volatility if not np.isnan(hl_volatility) else 0.0
                features[i, base_idx + 7] = pv_correlation if not np.isnan(pv_correlation) else 0.0
            else:
                features[i, base_idx:base_idx + 8] = 0.0
    
    return features

class V5StatisticalFeatureSelector:
    """Essential Statistical Features for V5 - Weekly Training & Daily Inference"""
    
    def __init__(self):
        self.data_dir = Path("/media/knight2/EDB/numer_crypto_temp/data")
        self.processed_dir = self.data_dir / "processed" / "statistical"
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Essential windows based on top performers
        self.windows = np.array([5, 10, 14, 20, 30], dtype=np.int32)
        self.feature_names = self._generate_feature_names()
        
        logger.info("🚀 V5 Statistical Feature Selector initialized")

    def _generate_feature_names(self) -> List[str]:
        """Generate feature names for essential statistical features"""
        names = []
        base_features = [
            'price_volatility', 'volume_volatility', 'price_mean', 'volume_mean',
            'price_zscore', 'price_range_ratio', 'hl_volatility', 'pv_correlation'
        ]
        
        for window in self.windows:
            for feature in base_features:
                names.append(f"stat_{feature}_{window}d_lag1")
        
        return names

    def generate_statistical_features(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """Generate essential statistical features with 1-day lag"""
        logger.info("🔧 Generating essential statistical features...")
        
        all_features = []
        symbols = price_data['symbol'].unique()
        
        for symbol in symbols:
            symbol_data = price_data[price_data['symbol'] == symbol].copy()
            
            if len(symbol_data) < 10:
                continue
            
            # Extract arrays
            prices = symbol_data['close'].values.astype(np.float64)
            volumes = symbol_data['volume'].values.astype(np.float64)
            high = symbol_data.get('high', symbol_data['close']).values.astype(np.float64)
            low = symbol_data.get('low', symbol_data['close']).values.astype(np.float64)
            
            # Handle missing values
            prices = np.nan_to_num(prices, nan=np.median(prices[prices > 0]))
            volumes = np.nan_to_num(volumes, nan=np.median(volumes[volumes > 0]))
            high = np.nan_to_num(high, nan=np.median(high[high > 0]))
            low = np.nan_to_num(low, nan=np.median(low[low > 0]))
            
            # Generate features
            stat_features = calculate_statistical_features_safe(prices, volumes, high, low, self.windows)
            
            # Create DataFrame
            feature_df = pd.DataFrame(stat_features, columns=self.feature_names, index=symbol_data.index)
            feature_df['symbol'] = symbol
            feature_df['date'] = symbol_data['date'].values
            
            # Apply 1-day lag (CRITICAL for no data leakage)
            feature_cols = [col for col in feature_df.columns if col.startswith('stat_')]
            for col in feature_cols:
                feature_df[col] = feature_df[col].shift(1)
            
            feature_df = feature_df.iloc[1:].copy()  # Remove first row (NaN due to lag)
            all_features.append(feature_df)
        
        combined_features = pd.concat(all_features, ignore_index=True)
        
        # Clean data
        numeric_cols = combined_features.select_dtypes(include=[np.number]).columns
        combined_features[numeric_cols] = combined_features[numeric_cols].replace([np.inf, -np.inf], np.nan)
        combined_features[numeric_cols] = combined_features[numeric_cols].fillna(0)
        
        logger.info(f"✅ Statistical features generated: {len(combined_features)} rows")
        return combined_features

    def run_statistical_feature_generation(self) -> str:
        """Run essential statistical feature generation"""
        logger.info("🚀 Starting V5 Statistical Feature Generation")
        
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
            features_df = self.generate_statistical_features(price_data)
            
            # Save features
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            feature_file = self.processed_dir / f"statistical_features_{timestamp}.parquet"
            features_df.to_parquet(feature_file, index=False)
            
            # Save metadata
            metadata = {
                'timestamp': timestamp,
                'feature_count': len([col for col in features_df.columns if col.startswith('stat_')]),
                'records': len(features_df),
                'symbols': features_df['symbol'].nunique(),
                'temporal_lag_days': 1,
                'essential_for': ['weekly_training', 'daily_inference']
            }
            
            import json
            metadata_file = self.processed_dir / f"statistical_metadata_{timestamp}.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            logger.info(f"✅ Statistical features saved: {feature_file}")
            return str(feature_file)
            
        except Exception as e:
            logger.error(f"❌ Statistical feature generation failed: {e}")
            raise

def main():
    selector = V5StatisticalFeatureSelector()
    feature_file = selector.run_statistical_feature_generation()
    print(f"🎉 Statistical features generated: {feature_file}")

if __name__ == "__main__":
    main()