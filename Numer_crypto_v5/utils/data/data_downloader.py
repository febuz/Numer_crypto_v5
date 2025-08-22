#!/usr/bin/env python3
"""
V5 Data Downloader Utilities - Extracted Common Functions

Contains reusable data downloading functionality extracted from
scripts/dataset/data_downloader_v5.py for better code organization.

CRITICAL RULES:
- 1-day temporal lag mandatory
- NO synthetic data generation
- Real market data only
"""

import os
import sys
import logging
import time
import requests
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import yfinance as yf

class DataDownloader:
    """
    Reusable data downloading utilities for V5 pipeline
    
    Extracted from data_downloader_v5.py to provide common
    functionality across multiple dataset processing scripts.
    """
    
    def __init__(self, data_dir: Optional[Path] = None, temporal_lag_days: int = 1):
        self.data_dir = data_dir or Path("/media/knight2/EDB/numer_crypto_temp/data")
        self.raw_dir = self.data_dir / "raw"
        self.TEMPORAL_LAG_DAYS = temporal_lag_days
        self.cutoff_date = datetime.now() - timedelta(days=temporal_lag_days)
        
        self.logger = logging.getLogger(__name__)
    
    def load_crypto_symbols(self, max_symbols: Optional[int] = None) -> List[str]:
        """
        Load cryptocurrency symbols from existing Numerai data
        
        Args:
            max_symbols: Maximum number of symbols to return (None for all)
            
        Returns:
            List of cryptocurrency symbols
        """
        try:
            # Load from latest Numerai crypto targets file
            numerai_files = list(self.raw_dir.glob("**/train_targets_*.parquet"))
            if numerai_files:
                latest_file = max(numerai_files, key=lambda x: x.stat().st_mtime)
                targets_df = pd.read_parquet(latest_file)
                
                if 'symbol' in targets_df.columns:
                    symbols = sorted(targets_df['symbol'].unique().tolist())
                    
                    # Filter out problematic symbols
                    filtered_symbols = []
                    for symbol in symbols:
                        # Remove symbols with special characters
                        if not any(char in symbol for char in ['$', '/']):
                            filtered_symbols.append(symbol)
                    
                    if max_symbols:
                        filtered_symbols = filtered_symbols[:max_symbols]
                    
                    self.logger.info(f"✅ Loaded {len(filtered_symbols)} symbols from {latest_file.name}")
                    return filtered_symbols
                    
        except Exception as e:
            self.logger.warning(f"Could not load symbols from existing data: {e}")
        
        # Fallback to expanded default crypto symbols
        default_symbols = [
            'BTC', 'ETH', 'BNB', 'XRP', 'ADA', 'DOGE', 'MATIC', 'SOL', 'DOT', 'LTC',
            'AVAX', 'SHIB', 'TRX', 'ETC', 'UNI', 'LINK', 'XLM', 'ALGO', 'VET', 'ICP',
            'FIL', 'THETA', 'HBAR', 'MANA', 'AXS', 'SAND', 'CRV', 'AAVE', 'MKR', 'COMP',
            'ATOM', 'NEAR', 'GRT', 'ENJ', 'BAT', 'ZEC', 'DASH', 'DCR', 'QTUM', 'ONT',
            'ZIL', 'ICX', 'NANO', 'RVN', 'WAVES', 'SC', 'DGB', 'BTG', 'XEM', 'LSK',
            'LUNA', 'FTM', 'ONE', 'ROSE', 'CELO', 'KAVA', 'BAND', 'REN', 'STORJ', 'NMR'
        ]
        
        if max_symbols:
            default_symbols = default_symbols[:max_symbols]
            
        self.logger.info(f"📊 Using {len(default_symbols)} default symbols")
        return default_symbols
    
    def download_single_symbol(self, symbol: str, days_history: int = 2000) -> Optional[pd.DataFrame]:
        """
        Download price data for a single cryptocurrency symbol
        
        Args:
            symbol: Cryptocurrency symbol (e.g., 'BTC')
            days_history: Number of days of historical data to fetch
            
        Returns:
            DataFrame with OHLCV data or None if failed
        """
        try:
            # Map symbol to Yahoo Finance format
            yf_symbol = f"{symbol}-USD"
            
            # Download data with sufficient history
            end_date = self.cutoff_date
            start_date = end_date - timedelta(days=days_history)
            
            ticker = yf.Ticker(yf_symbol)
            hist = ticker.history(start=start_date, end=end_date, auto_adjust=True)
            
            if hist.empty:
                self.logger.warning(f"⚠️ No data for {symbol}")
                return None
            
            # Process data
            df = hist.reset_index()
            df.columns = df.columns.str.lower()
            df['symbol'] = symbol
            df['date'] = pd.to_datetime(df['date']).dt.date
            
            # Ensure no future data leakage
            df = df[pd.to_datetime(df['date']) <= self.cutoff_date]
            
            # Data quality checks
            if len(df) < 30:  # Minimum 30 days of data
                self.logger.warning(f"⚠️ Insufficient data for {symbol}: {len(df)} days")
                return None
            
            # Remove extreme outliers
            price_cols = ['open', 'high', 'low', 'close']
            for col in price_cols:
                if col in df.columns:
                    median_price = df[col].median()
                    df = df[(df[col] >= median_price * 0.1) & (df[col] <= median_price * 10)]
            
            return df
            
        except Exception as e:
            self.logger.error(f"❌ Failed to download {symbol}: {e}")
            return None
    
    def download_symbols_batch(self, symbols: List[str], batch_size: int = 50, 
                              max_workers: int = 10) -> List[pd.DataFrame]:
        """
        Download cryptocurrency data in batches with parallel processing
        
        Args:
            symbols: List of cryptocurrency symbols
            batch_size: Number of symbols to process in each batch
            max_workers: Maximum number of parallel workers
            
        Returns:
            List of DataFrames with price data
        """
        all_data = []
        
        self.logger.info(f"📊 Processing {len(symbols)} symbols in batches of {batch_size}")
        
        for batch_start in range(0, len(symbols), batch_size):
            batch_symbols = symbols[batch_start:batch_start + batch_size]
            batch_num = batch_start // batch_size + 1
            total_batches = (len(symbols) - 1) // batch_size + 1
            
            self.logger.info(f"🔄 Processing batch {batch_num}/{total_batches}: {len(batch_symbols)} symbols")
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_symbol = {
                    executor.submit(self.download_single_symbol, symbol): symbol
                    for symbol in batch_symbols
                }
                
                for future in as_completed(future_to_symbol):
                    symbol = future_to_symbol[future]
                    try:
                        result = future.result()
                        if result is not None:
                            all_data.append(result)
                    except Exception as e:
                        self.logger.error(f"❌ Error processing {symbol}: {e}")
        
        return all_data
    
    def download_economic_indicator(self, symbol: str, days_history: int = 2000) -> Optional[pd.DataFrame]:
        """
        Download economic indicator data
        
        Args:
            symbol: Economic indicator symbol (e.g., 'SPY', 'VIX')
            days_history: Number of days of historical data
            
        Returns:
            DataFrame with indicator data or None if failed
        """
        try:
            end_date = self.cutoff_date
            start_date = end_date - timedelta(days=days_history)
            
            ticker = yf.Ticker(symbol)
            hist = ticker.history(start=start_date, end=end_date, auto_adjust=True)
            
            if hist.empty:
                return None
            
            df = hist.reset_index()
            df.columns = df.columns.str.lower()
            df['indicator'] = symbol
            df['date'] = pd.to_datetime(df['date']).dt.date
            
            # Apply temporal lag
            df = df[pd.to_datetime(df['date']) <= self.cutoff_date]
            
            return df[['date', 'indicator', 'close', 'volume']]
            
        except Exception as e:
            self.logger.error(f"❌ Failed to download {symbol}: {e}")
            return None
    
    def download_numerai_data(self, url: str) -> Optional[pd.DataFrame]:
        """
        Download Numerai tournament data with error handling
        
        Args:
            url: URL to Numerai data
            
        Returns:
            DataFrame with tournament data or None if failed
        """
        try:
            response = requests.get(url, timeout=300)
            response.raise_for_status()
            
            # Create temporary file
            timestamp = datetime.now().strftime("%Y%m%d")
            temp_file = self.raw_dir / "numerai" / f"temp_tournament_data_{timestamp}.parquet"
            temp_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(temp_file, 'wb') as f:
                f.write(response.content)
            
            # Load and process
            df = pd.read_parquet(temp_file)
            
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                # Apply temporal lag
                df = df[df['date'] <= self.cutoff_date]
            
            # Clean up temp file
            temp_file.unlink()
            
            return df
            
        except Exception as e:
            self.logger.error(f"❌ Numerai download failed: {e}")
            return None
    
    def create_mock_numerai_data(self, symbols: List[str], days: int = 365) -> pd.DataFrame:
        """
        Create mock Numerai data for development/testing
        
        Args:
            symbols: List of symbols to create mock data for
            days: Number of days of mock data
            
        Returns:
            DataFrame with mock tournament structure
        """
        dates = pd.date_range(
            start=datetime.now() - timedelta(days=days),
            end=self.cutoff_date,
            freq='D'
        )
        
        mock_data = []
        for symbol in symbols:
            for date in dates:
                mock_data.append({
                    'symbol': symbol,
                    'date': date,
                    'target': np.random.normal(0, 0.1)  # Mock target values
                })
        
        df = pd.DataFrame(mock_data)
        self.logger.info(f"✅ Created mock Numerai data: {len(df)} rows")
        return df
    
    def validate_temporal_lag(self, df: pd.DataFrame) -> bool:
        """
        Validate that data respects temporal lag requirements
        
        Args:
            df: DataFrame with 'date' column
            
        Returns:
            True if temporal lag is respected, False otherwise
        """
        if 'date' not in df.columns:
            return True  # No date column to validate
        
        df_dates = pd.to_datetime(df['date'])
        max_date = df_dates.max()
        
        if max_date > pd.Timestamp(self.cutoff_date):
            self.logger.error(f"❌ DATA LEAKAGE: max_date={max_date} > cutoff={self.cutoff_date}")
            return False
        
        return True
    
    def save_with_metadata(self, df: pd.DataFrame, file_path: Path, 
                          metadata: Optional[Dict] = None) -> bool:
        """
        Save DataFrame with metadata tracking
        
        Args:
            df: DataFrame to save
            file_path: Path to save the file
            metadata: Optional metadata dictionary
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Ensure directory exists
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save main data
            df.to_parquet(file_path, index=False)
            
            # Save metadata if provided
            if metadata:
                metadata_file = file_path.with_suffix('.json')
                import json
                with open(metadata_file, 'w') as f:
                    json.dump(metadata, f, indent=2, default=str)
            
            self.logger.info(f"✅ Data saved: {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to save {file_path}: {e}")
            return False