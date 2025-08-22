#!/usr/bin/env python3
"""
🚀 ADVANCED FEATURE ENGINEERING V5
================================
Ultra-advanced feature engineering techniques to achieve RMSE < 0.08

ADVANCED TECHNIQUES:
- Regime detection features (bull/bear/sideways markets)
- Cross-asset momentum and mean reversion
- Volatility clustering and GARCH-like features  
- Fractal and chaos theory indicators
- Information theory features (entropy, mutual information)
- Dynamic time warping features
- Market microstructure features
- Behavioral finance indicators
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy import signal
from scipy.stats import entropy, skew, kurtosis
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import talib

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AdvancedFeatureEngineering:
    """Ultra-advanced feature engineering for cryptocurrency prediction"""
    
    def __init__(self):
        self.scaler = RobustScaler()
        self.pca = PCA(n_components=0.95)  # Retain 95% variance
        self.regime_detector = None
        
        logger.info("🚀 Advanced Feature Engineering V5 initialized")
    
    def detect_market_regimes(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """Detect market regimes using advanced statistical methods"""
        logger.info("📊 Detecting market regimes...")
        
        regime_features = price_data.copy()
        
        for symbol in price_data['symbol'].unique():
            symbol_data = price_data[price_data['symbol'] == symbol].copy()
            if len(symbol_data) < 100:
                continue
                
            symbol_data = symbol_data.sort_values('date')
            # Use lagged close price (leak-free)
            if 'close_lag1' in symbol_data.columns:
                prices = symbol_data['close_lag1'].dropna().values
            elif 'open' in symbol_data.columns:
                prices = symbol_data['open'].values  # Fallback to open price
            else:
                logger.warning(f"⚠️ No price data available for {symbol}")
                continue
            
            # 1. Volatility regime detection
            returns = np.diff(np.log(prices))
            rolling_vol = pd.Series(returns).rolling(window=30).std()
            vol_regime = pd.cut(rolling_vol, bins=3, labels=['low_vol', 'medium_vol', 'high_vol'])
            
            # 2. Trend regime detection (using multiple timeframes)
            short_ma = pd.Series(prices).rolling(10).mean()
            medium_ma = pd.Series(prices).rolling(30).mean()
            long_ma = pd.Series(prices).rolling(90).mean()
            
            trend_signal = np.where(
                (short_ma > medium_ma) & (medium_ma > long_ma), 2,  # Strong bullish
                np.where((short_ma > medium_ma) & (medium_ma <= long_ma), 1,  # Weak bullish
                         np.where((short_ma <= medium_ma) & (medium_ma > long_ma), -1,  # Weak bearish
                                  -2))  # Strong bearish
            )
            
            # 3. Momentum regime with proper shape handling
            if len(prices) >= 30:  # Ensure we have enough data
                roc_10 = (prices[10:] - prices[:-10]) / prices[:-10] * 100
                roc_30 = (prices[30:] - prices[:-30]) / prices[:-30] * 100
                
                # Align arrays to same length (roc_30 is shorter)
                roc_10_aligned = roc_10[20:]  # Skip first 20 to match roc_30 start
                
                momentum_regime_short = np.where(
                    (roc_10_aligned > 5) & (roc_30 > 10), 2,  # Strong momentum
                    np.where((roc_10_aligned > 0) & (roc_30 > 0), 1,  # Positive momentum
                             np.where((roc_10_aligned < -5) & (roc_30 < -10), -2,  # Strong negative
                                      -1))  # Negative momentum
                )
                
                # Pad to match original data length safely
                pad_start = 30  # Account for the 30-day ROC calculation
                pad_end = len(prices) - len(momentum_regime_short) - pad_start
                
                if pad_end < 0:
                    # If still too long, truncate
                    momentum_regime_short = momentum_regime_short[:len(prices)-pad_start]
                    pad_end = 0
                
                momentum_regime = np.pad(momentum_regime_short, (pad_start, pad_end), 
                                       mode='constant', constant_values=0)
                
                # Ensure exact length match
                momentum_regime = momentum_regime[:len(prices)]
                
            else:
                # Not enough data, fill with neutral values
                momentum_regime = np.zeros(len(prices))
            
            # Store regime features with proper length checking
            mask = regime_features['symbol'] == symbol
            mask_length = np.sum(mask)
            
            # Ensure all arrays match the mask length
            vol_regime_safe = vol_regime.reindex(regime_features[mask].index, fill_value='medium_vol')
            trend_signal_safe = pd.Series(trend_signal).reindex(regime_features[mask].index, fill_value=0)
            
            # Handle momentum regime length mismatch
            if len(momentum_regime) != mask_length:
                if len(momentum_regime) > mask_length:
                    momentum_regime = momentum_regime[:mask_length]
                else:
                    momentum_regime = np.pad(momentum_regime, (0, mask_length - len(momentum_regime)), 
                                           mode='constant', constant_values=0)
            
            regime_features.loc[mask, 'volatility_regime'] = vol_regime_safe
            regime_features.loc[mask, 'trend_regime'] = trend_signal_safe
            regime_features.loc[mask, 'momentum_regime'] = momentum_regime
        
        # Convert categorical to numeric
        regime_features['vol_regime_low'] = (regime_features['volatility_regime'] == 'low_vol').astype(int)
        regime_features['vol_regime_medium'] = (regime_features['volatility_regime'] == 'medium_vol').astype(int)
        regime_features['vol_regime_high'] = (regime_features['volatility_regime'] == 'high_vol').astype(int)
        
        logger.info("✅ Market regime detection completed")
        return regime_features
    
    def generate_volatility_clustering_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate GARCH-like volatility clustering features"""
        logger.info("📈 Generating volatility clustering features...")
        
        vol_features = data.copy()
        
        for symbol in data['symbol'].unique():
            symbol_data = data[data['symbol'] == symbol].copy()
            if len(symbol_data) < 50:
                continue
                
            symbol_data = symbol_data.sort_values('date')
            # Use lagged close price (leak-free)
            if 'close_lag1' in symbol_data.columns:
                prices = symbol_data['close_lag1'].dropna().values
            elif 'open' in symbol_data.columns:
                prices = symbol_data['open'].values  # Fallback to open price
            else:
                logger.warning(f"⚠️ No price data available for {symbol}")
                continue
            returns = np.diff(np.log(prices))
            
            # 1. EWMA volatility
            alpha = 0.06  # Decay factor
            ewma_var = np.zeros(len(returns))
            ewma_var[0] = returns[0]**2
            
            for i in range(1, len(returns)):
                ewma_var[i] = alpha * returns[i-1]**2 + (1-alpha) * ewma_var[i-1]
            
            ewma_vol = np.sqrt(ewma_var)
            
            # 2. Realized volatility (5-day, 20-day)
            rv_5 = pd.Series(returns**2).rolling(5).sum()
            rv_20 = pd.Series(returns**2).rolling(20).sum()
            
            # 3. Volatility of volatility
            vol_of_vol = pd.Series(ewma_vol).rolling(10).std()
            
            # 4. Volatility persistence (autocorrelation)
            vol_persistence = pd.Series(ewma_vol).rolling(20).apply(
                lambda x: np.corrcoef(x[:-1], x[1:])[0,1] if len(x) > 1 else 0
            )
            
            # Store features (pad to match original length)
            mask = vol_features['symbol'] == symbol
            ewma_vol_padded = np.pad(ewma_vol, (1, 0), mode='edge')
            vol_features.loc[mask, 'ewma_volatility'] = ewma_vol_padded[:len(symbol_data)]
            vol_features.loc[mask, 'realized_vol_5d'] = rv_5.reindex(vol_features[mask].index, fill_value=0)
            vol_features.loc[mask, 'realized_vol_20d'] = rv_20.reindex(vol_features[mask].index, fill_value=0)
            vol_features.loc[mask, 'vol_of_vol'] = vol_of_vol.reindex(vol_features[mask].index, fill_value=0)
            vol_features.loc[mask, 'vol_persistence'] = vol_persistence.reindex(vol_features[mask].index, fill_value=0)
        
        logger.info("✅ Volatility clustering features completed")
        return vol_features
    
    def generate_cross_asset_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate sophisticated cross-asset momentum and correlation features"""
        logger.info("🔗 Generating cross-asset features...")
        
        cross_features = data.copy()
        
        # Calculate daily cross-sectional statistics
        daily_stats = data.groupby('date').agg({
            'close': ['mean', 'median', 'std', 'min', 'max'],
            'volume': ['mean', 'median', 'std'] if 'volume' in data.columns else []
        })
        daily_stats.columns = ['_'.join(col).strip() for col in daily_stats.columns]
        
        # Market-wide indicators
        cross_features = cross_features.merge(daily_stats, left_on='date', right_index=True, how='left')
        
        # Individual asset vs market
        # Use leak-free price columns
        price_col = 'close_lag1' if 'close_lag1' in cross_features.columns else 'open'
        if price_col in cross_features.columns:
            cross_features['price_vs_market_mean'] = cross_features[price_col] / cross_features['close_mean']
            cross_features['price_vs_market_median'] = cross_features[price_col] / cross_features['close_median']
            cross_features['price_market_zscore'] = (cross_features[price_col] - cross_features['close_mean']) / cross_features['close_std']
        
        # Relative strength features
        cross_features['market_position'] = cross_features.groupby('date')['close'].rank(pct=True)
        cross_features['volume_position'] = cross_features.groupby('date')['volume'].rank(pct=True) if 'volume' in data.columns else 0
        
        # Cross-asset momentum
        for window in [5, 10, 20]:
            # Rolling cross-sectional rank
            cross_features[f'rank_momentum_{window}d'] = cross_features.groupby('symbol')['market_position'].rolling(window).mean().reset_index(0, drop=True)
            
            # Market dispersion
            cross_features[f'market_dispersion_{window}d'] = cross_features.groupby('date')['close'].rolling(window).apply(
                lambda x: (x.max() - x.min()) / x.mean() if len(x) > 0 else 0
            ).reset_index(0, drop=True)
        
        # Cross-asset correlations (rolling)
        symbol_list = data['symbol'].unique()[:10]  # Top 10 symbols for speed
        for symbol1 in symbol_list:
            for symbol2 in symbol_list:
                if symbol1 != symbol2:
                    s1_data = data[data['symbol'] == symbol1].set_index('date')['close']
                    s2_data = data[data['symbol'] == symbol2].set_index('date')['close']
                    
                    # Rolling correlation
                    rolling_corr = s1_data.rolling(30).corr(s2_data)
                    cross_features.loc[cross_features['symbol'] == symbol1, f'corr_{symbol2}'] = rolling_corr.reindex(
                        cross_features[cross_features['symbol'] == symbol1]['date']
                    ).fillna(0).values
        
        logger.info("✅ Cross-asset features completed")
        return cross_features
    
    def generate_fractal_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate fractal dimension and chaos theory features"""
        logger.info("🌀 Generating fractal and chaos features...")
        
        fractal_features = data.copy()
        
        for symbol in data['symbol'].unique():
            symbol_data = data[data['symbol'] == symbol].copy()
            if len(symbol_data) < 100:
                continue
                
            symbol_data = symbol_data.sort_values('date')
            # Use lagged close price (leak-free)
            if 'close_lag1' in symbol_data.columns:
                prices = symbol_data['close_lag1'].dropna().values
            elif 'open' in symbol_data.columns:
                prices = symbol_data['open'].values  # Fallback to open price
            else:
                logger.warning(f"⚠️ No price data available for {symbol}")
                continue
            returns = np.diff(np.log(prices))
            
            # 1. Hurst Exponent (persistence measure)
            def hurst_exponent(ts, max_lag=20):
                lags = range(2, max_lag)
                tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]
                poly = np.polyfit(np.log(lags), np.log(tau), 1)
                return poly[0] * 2.0
            
            # Calculate Hurst for different windows
            hurst_30 = pd.Series(prices).rolling(60).apply(lambda x: hurst_exponent(x.values) if len(x) >= 30 else 0.5)
            
            # 2. Fractal dimension (Higuchi method approximation)
            def fractal_dimension(ts, k_max=10):
                N = len(ts)
                L = []
                for k in range(1, k_max):
                    Lk = []
                    for m in range(k):
                        Lmk = 0
                        for i in range(1, int((N-m)/k)):
                            Lmk += abs(ts[m+i*k] - ts[m+(i-1)*k])
                        Lmk = Lmk * (N-1) / (((N-m)/k) * k)
                        Lk.append(Lmk)
                    L.append(np.log(np.mean(Lk)))
                
                return -np.polyfit(range(1, k_max), L, 1)[0] if len(L) > 1 else 1.5
            
            fractal_dim = pd.Series(prices).rolling(50).apply(lambda x: fractal_dimension(x.values) if len(x) >= 20 else 1.5)
            
            # 3. Lyapunov exponent approximation
            def lyapunov_approx(ts, delay=1):
                embedded = np.array([ts[i:i+delay+1] for i in range(len(ts)-delay)])
                if len(embedded) < 10:
                    return 0
                    
                # Find nearest neighbors and track divergence
                divergences = []
                for i in range(len(embedded)-10):
                    distances = [np.linalg.norm(embedded[i] - embedded[j]) for j in range(len(embedded)) if abs(i-j) > 10]
                    if distances:
                        min_dist = min(distances)
                        min_idx = np.argmin([np.linalg.norm(embedded[i] - embedded[j]) for j in range(len(embedded)) if abs(i-j) > 10])
                        # Track divergence over next few steps
                        if i+5 < len(embedded) and min_idx+5 < len(embedded):
                            future_dist = np.linalg.norm(embedded[i+5] - embedded[min_idx+5])
                            if min_dist > 0 and future_dist > 0:
                                divergences.append(np.log(future_dist / min_dist) / 5)
                
                return np.mean(divergences) if divergences else 0
            
            lyapunov = pd.Series(returns).rolling(100).apply(lambda x: lyapunov_approx(x.values) if len(x) >= 50 else 0)
            
            # Store features
            mask = fractal_features['symbol'] == symbol
            fractal_features.loc[mask, 'hurst_exponent'] = hurst_30.reindex(fractal_features[mask].index, fill_value=0.5)
            fractal_features.loc[mask, 'fractal_dimension'] = fractal_dim.reindex(fractal_features[mask].index, fill_value=1.5)
            fractal_features.loc[mask, 'lyapunov_exponent'] = lyapunov.reindex(fractal_features[mask].index, fill_value=0)
            
        logger.info("✅ Fractal and chaos features completed")
        return fractal_features
    
    def generate_information_theory_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate entropy and information theory features"""
        logger.info("📊 Generating information theory features...")
        
        info_features = data.copy()
        
        for symbol in data['symbol'].unique():
            symbol_data = data[data['symbol'] == symbol].copy()
            if len(symbol_data) < 50:
                continue
                
            symbol_data = symbol_data.sort_values('date')
            # Use lagged close price (leak-free)
            if 'close_lag1' in symbol_data.columns:
                prices = symbol_data['close_lag1'].dropna().values
            elif 'open' in symbol_data.columns:
                prices = symbol_data['open'].values  # Fallback to open price
            else:
                logger.warning(f"⚠️ No price data available for {symbol}")
                continue
            returns = np.diff(np.log(prices))
            
            # 1. Shannon Entropy of returns
            def shannon_entropy(ts, bins=20):
                hist, _ = np.histogram(ts, bins=bins)
                hist = hist[hist > 0]
                return entropy(hist)
            
            entropy_10 = pd.Series(returns).rolling(30).apply(lambda x: shannon_entropy(x.values) if len(x) >= 10 else 0)
            entropy_30 = pd.Series(returns).rolling(60).apply(lambda x: shannon_entropy(x.values) if len(x) >= 30 else 0)
            
            # 2. Approximate Entropy
            def approximate_entropy(ts, m=2, r=0.2):
                def _maxdist(ts, i, j, m):
                    return max([abs(ua - va) for ua, va in zip(ts[i:i+m], ts[j:j+m])])
                
                N = len(ts)
                patterns = np.array([ts[i:i+m] for i in range(N-m+1)])
                
                C = np.zeros(N-m+1)
                for i in range(N-m+1):
                    template_i = patterns[i]
                    matches = 0
                    for j in range(N-m+1):
                        if _maxdist(ts, i, j, m) <= r * np.std(ts):
                            matches += 1
                    C[i] = matches / (N-m+1)
                
                phi_m = np.mean([np.log(c) for c in C if c > 0])
                
                # Repeat for m+1
                patterns = np.array([ts[i:i+m+1] for i in range(N-m)])
                C = np.zeros(N-m)
                for i in range(N-m):
                    matches = 0
                    for j in range(N-m):
                        if _maxdist(ts, i, j, m+1) <= r * np.std(ts):
                            matches += 1
                    C[i] = matches / (N-m)
                
                phi_m1 = np.mean([np.log(c) for c in C if c > 0])
                
                return phi_m - phi_m1
            
            app_entropy = pd.Series(returns).rolling(50).apply(lambda x: approximate_entropy(x.values) if len(x) >= 20 else 0)
            
            # 3. Sample Entropy (simplified)
            def sample_entropy(ts, m=2, r=0.2):
                N = len(ts)
                if N < m + 1:
                    return 0
                    
                templates = [ts[i:i+m] for i in range(N-m+1)]
                A = 0
                B = 0
                
                for i in range(len(templates)):
                    for j in range(len(templates)):
                        if i != j:
                            if max(abs(a-b) for a, b in zip(templates[i], templates[j])) <= r * np.std(ts):
                                B += 1
                                if max(abs(a-b) for a, b in zip(ts[i:i+m+1], ts[j:j+m+1])) <= r * np.std(ts):
                                    A += 1
                
                return -np.log(A/B) if B > 0 and A > 0 else 0
            
            sample_ent = pd.Series(returns).rolling(40).apply(lambda x: sample_entropy(x.values) if len(x) >= 15 else 0)
            
            # Store features
            mask = info_features['symbol'] == symbol
            info_features.loc[mask, 'shannon_entropy_10'] = entropy_10.reindex(info_features[mask].index, fill_value=0)
            info_features.loc[mask, 'shannon_entropy_30'] = entropy_30.reindex(info_features[mask].index, fill_value=0)
            info_features.loc[mask, 'approximate_entropy'] = app_entropy.reindex(info_features[mask].index, fill_value=0)
            info_features.loc[mask, 'sample_entropy'] = sample_ent.reindex(info_features[mask].index, fill_value=0)
        
        logger.info("✅ Information theory features completed")
        return info_features
    
    def generate_microstructure_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate market microstructure features"""
        logger.info("🏪 Generating market microstructure features...")
        
        micro_features = data.copy()
        
        # Price impact and efficiency measures
        if 'volume' in data.columns:
            # Volume-weighted features
            micro_features['vwap_5'] = micro_features.groupby('symbol').apply(
                lambda x: (x['close'] * x['volume']).rolling(5).sum() / x['volume'].rolling(5).sum()
            ).reset_index(0, drop=True)
            
            # Price-volume correlation
            micro_features['price_volume_corr_20'] = micro_features.groupby('symbol').apply(
                lambda x: x['close'].rolling(20).corr(x['volume'])
            ).reset_index(0, drop=True)
            
            # Amihud illiquidity measure approximation
            micro_features['amihud_illiquidity'] = micro_features.groupby('symbol').apply(
                lambda x: (abs(x['close'].pct_change()) / (x['volume'] * x['close'])).rolling(20).mean()
            ).reset_index(0, drop=True)
        
        # Bid-ask spread proxy (using high-low)
        if 'high' in data.columns and 'low' in data.columns:
            micro_features['spread_proxy'] = (micro_features['high'] - micro_features['low']) / micro_features['close']
            micro_features['spread_ma_20'] = micro_features.groupby('symbol')['spread_proxy'].rolling(20).mean().reset_index(0, drop=True)
        
        # Market efficiency measures
        micro_features['variance_ratio_2'] = micro_features.groupby('symbol').apply(
            lambda x: x['close'].pct_change().rolling(20).var() / (x['close'].pct_change().rolling(10).var() * 2)
        ).reset_index(0, drop=True)
        
        logger.info("✅ Market microstructure features completed")
        return micro_features
    
    def generate_behavioral_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate behavioral finance features"""
        logger.info("🧠 Generating behavioral finance features...")
        
        behavioral_features = data.copy()
        
        for symbol in data['symbol'].unique():
            symbol_data = data[data['symbol'] == symbol].copy()
            if len(symbol_data) < 50:
                continue
                
            symbol_data = symbol_data.sort_values('date')
            # Use lagged close price (leak-free)
            if 'close_lag1' in symbol_data.columns:
                prices = symbol_data['close_lag1'].dropna().values
            elif 'open' in symbol_data.columns:
                prices = symbol_data['open'].values  # Fallback to open price
            else:
                logger.warning(f"⚠️ No price data available for {symbol}")
                continue
            returns = np.diff(np.log(prices))
            
            # 1. Momentum and contrarian indicators
            # Price momentum
            momentum_10 = (prices[10:] - prices[:-10]) / prices[:-10]
            momentum_20 = (prices[20:] - prices[:-20]) / prices[:-20]
            
            # Mean reversion indicators
            mean_20 = pd.Series(prices).rolling(20).mean()
            mean_reversion = (prices - mean_20) / mean_20
            
            # 2. Overreaction indicators
            # Large price moves followed by reversals
            large_moves = abs(pd.Series(returns)) > 2 * pd.Series(returns).std()
            subsequent_returns = pd.Series(returns).shift(-1)
            overreaction_reversal = large_moves & (pd.Series(returns) * subsequent_returns < 0)
            
            # 3. Herding behavior proxy
            # Correlation with market during extreme moves
            market_returns = behavioral_features.groupby('date')['close'].apply(lambda x: x.pct_change().mean())
            symbol_returns = pd.Series(returns)
            
            # Rolling correlation during high volatility periods
            high_vol_periods = symbol_returns.rolling(20).std() > symbol_returns.std()
            herding_correlation = symbol_returns.rolling(30).corr(
                market_returns.reindex(symbol_data['date'][1:].values)
            )
            
            # Store features (with proper alignment)
            mask = behavioral_features['symbol'] == symbol
            
            # Pad momentum features
            momentum_10_padded = np.pad(momentum_10, (10, len(symbol_data)-len(momentum_10)-10), mode='edge')
            momentum_20_padded = np.pad(momentum_20, (20, len(symbol_data)-len(momentum_20)-20), mode='edge')
            
            behavioral_features.loc[mask, 'momentum_10d'] = momentum_10_padded[:len(symbol_data)]
            behavioral_features.loc[mask, 'momentum_20d'] = momentum_20_padded[:len(symbol_data)]
            behavioral_features.loc[mask, 'mean_reversion'] = mean_reversion.reindex(behavioral_features[mask].index, fill_value=0)
            behavioral_features.loc[mask, 'overreaction_indicator'] = overreaction_reversal.reindex(behavioral_features[mask].index, fill_value=False).astype(int)
            behavioral_features.loc[mask, 'herding_correlation'] = herding_correlation.reindex(behavioral_features[mask].index, fill_value=0)
        
        logger.info("✅ Behavioral finance features completed")
        return behavioral_features
    
    def apply_dimensionality_reduction(self, features_df: pd.DataFrame, exclude_cols: List[str] = None) -> pd.DataFrame:
        """Apply PCA for dimensionality reduction while preserving key features"""
        logger.info("🔍 Applying dimensionality reduction...")
        
        if exclude_cols is None:
            exclude_cols = ['symbol', 'date', 'target']
        
        # Separate features for PCA
        feature_cols = [col for col in features_df.columns if col not in exclude_cols and not col.startswith('target')]
        numeric_features = features_df[feature_cols].select_dtypes(include=[np.number])
        
        if len(numeric_features.columns) == 0:
            return features_df
        
        # Comprehensive NaN and infinite value handling
        logger.info("🔧 Applying comprehensive data cleaning for dimensionality reduction...")
        
        # Check for problematic values
        nan_count = numeric_features.isna().sum().sum()
        inf_count = np.isinf(numeric_features.select_dtypes(include=[np.number])).sum().sum()
        
        logger.info(f"   Initial NaN count: {nan_count}")
        logger.info(f"   Initial infinite count: {inf_count}")
        
        # Handle infinite values first
        numeric_features_cleaned = numeric_features.replace([np.inf, -np.inf], np.nan)
        
        # Handle NaN values with multiple strategies
        numeric_features_filled = numeric_features_cleaned.copy()
        
        # For each column, choose appropriate imputation strategy
        for col in numeric_features_filled.columns:
            col_data = numeric_features_filled[col]
            nan_ratio = col_data.isna().mean()
            
            if nan_ratio > 0.8:
                # If more than 80% NaN, fill with zeros
                numeric_features_filled[col] = col_data.fillna(0)
            elif nan_ratio > 0.3:
                # If 30-80% NaN, use forward fill then median
                numeric_features_filled[col] = col_data.fillna(method='ffill').fillna(col_data.median())
            else:
                # If less than 30% NaN, use median
                numeric_features_filled[col] = col_data.fillna(col_data.median())
        
        # Final safety check - replace any remaining NaN with 0
        numeric_features_filled = numeric_features_filled.fillna(0)
        
        # Verify no NaN or infinite values remain
        final_nan_count = numeric_features_filled.isna().sum().sum()
        final_inf_count = np.isinf(numeric_features_filled).sum().sum()
        
        logger.info(f"✅ After cleaning - NaN: {final_nan_count}, Infinite: {final_inf_count}")
        
        if final_nan_count > 0 or final_inf_count > 0:
            logger.error(f"❌ Data cleaning failed! Still have NaN: {final_nan_count}, Inf: {final_inf_count}")
            # Force clean any remaining problematic values
            numeric_features_filled = numeric_features_filled.replace([np.nan, np.inf, -np.inf], 0)
        
        # Apply PCA to reduce highly correlated features
        try:
            # Additional check for constant columns (zero variance)
            feature_variances = numeric_features_filled.var()
            non_constant_cols = feature_variances[feature_variances > 1e-10].index
            
            if len(non_constant_cols) == 0:
                logger.warning("⚠️ All features have zero variance, skipping PCA")
                return features_df
            
            numeric_features_for_pca = numeric_features_filled[non_constant_cols]
            logger.info(f"Using {len(non_constant_cols)} non-constant features for PCA")
            
            pca_features = self.pca.fit_transform(self.scaler.fit_transform(numeric_features_for_pca))
            
            # Create PCA feature names
            pca_cols = [f'pca_component_{i}' for i in range(pca_features.shape[1])]
            pca_df = pd.DataFrame(pca_features, columns=pca_cols, index=features_df.index)
            
            # Combine with non-PCA columns
            non_feature_cols = [col for col in features_df.columns if col in exclude_cols]
            result_df = pd.concat([features_df[non_feature_cols], pca_df], axis=1)
            
            logger.info(f"✅ PCA reduced {len(feature_cols)} features to {pca_features.shape[1]} components")
            logger.info(f"   Explained variance ratio: {sum(self.pca.explained_variance_ratio_):.3f}")
            
            return result_df
            
        except Exception as e:
            logger.warning(f"PCA failed: {e}, returning original features with cleaning")
            # Return cleaned features without PCA if PCA fails
            non_feature_cols = [col for col in features_df.columns if col in exclude_cols]
            result_df = pd.concat([features_df[non_feature_cols], numeric_features_filled], axis=1)
            return result_df
    
    def run_advanced_feature_engineering(self, data: pd.DataFrame) -> pd.DataFrame:
        """Run complete advanced feature engineering pipeline"""
        logger.info("🚀 Starting advanced feature engineering pipeline...")
        
        start_time = datetime.now()
        
        try:
            # Start with input data
            enhanced_data = data.copy()
            
            # 1. Market regime detection
            # enhanced_data = self.detect_market_regimes(enhanced_data)
            
            # 2. Volatility clustering features  
            enhanced_data = self.generate_volatility_clustering_features(enhanced_data)
            
            # 3. Cross-asset features
            # enhanced_data = self.generate_cross_asset_features(enhanced_data)
            
            # 4. Fractal and chaos features
            enhanced_data = self.generate_fractal_features(enhanced_data)
            
            # 5. Information theory features
            enhanced_data = self.generate_information_theory_features(enhanced_data)
            
            # 6. Market microstructure features
            # # # enhanced_data = self.generate_microstructure_features(enhanced_data)
            
            # 7. Behavioral finance features
            enhanced_data = self.generate_behavioral_features(enhanced_data)
            
            # 8. Apply dimensionality reduction
            enhanced_data = self.apply_dimensionality_reduction(enhanced_data)
            
            duration = datetime.now() - start_time
            
            logger.info("✅ Advanced feature engineering completed")
            logger.info(f"   Duration: {duration.total_seconds():.1f} seconds")
            logger.info(f"   Final shape: {enhanced_data.shape}")
            logger.info(f"   Features added: {enhanced_data.shape[1] - data.shape[1]}")
            
            return enhanced_data
            
        except Exception as e:
            logger.error(f"❌ Advanced feature engineering failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return data  # Return original data if processing fails

def test_advanced_features():
    """Test advanced feature engineering with sample data"""
    print("🧪 Testing Advanced Feature Engineering...")
    
    # Create sample data
    dates = pd.date_range('2020-01-01', '2024-01-01', freq='D')
    symbols = ['BTC', 'ETH', 'ADA']
    
    sample_data = []
    for symbol in symbols:
        for date in dates:
            price = 100 * (1 + np.random.normal(0, 0.02))  # Random walk
            sample_data.append({
                'date': date,
                'symbol': symbol,
                'close': price,
                'high': price * 1.02,
                'low': price * 0.98,
                'open': price * (1 + np.random.normal(0, 0.01)),
                'volume': np.random.lognormal(10, 1),
                'target': np.random.normal(0.5, 0.1)
            })
    
    df = pd.DataFrame(sample_data)
    
    # Test feature engineering
    engineer = AdvancedFeatureEngineering()
    enhanced_df = engineer.run_advanced_feature_engineering(df)
    
    print(f"✅ Test completed: {df.shape} → {enhanced_df.shape}")
    print(f"New features: {list(enhanced_df.columns[len(df.columns):])[:10]}...")  # Show first 10 new features

if __name__ == "__main__":
    test_advanced_features()