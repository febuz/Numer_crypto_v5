#!/usr/bin/env python3
"""
V5 Equity Data Downloader - Simplified Working Version
=====================================================

Simplified, working version of equity data downloader that bypasses
cuDF-pandas compatibility issues and focuses on core functionality.

FEATURES:
- Multi-threaded parallel processing
- EDS ticker identifier support
- Single parquet output with date in name
- 1-day temporal lag enforcement
- Comprehensive statistics

OUTPUT:
- equity_data_YYYYMMDD.parquet
- equity_statistics_YYYYMMDD.csv
"""

import os
import sys
import warnings
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Suppress warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import yfinance as yf

# Setup logging
def setup_logging() -> logging.Logger:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path("/media/knight2/EDB/numer_crypto_temp/data/log")
    log_dir.mkdir(exist_ok=True)
    
    logger = logging.getLogger("v5_equity_downloader_simple")
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        file_handler = logging.FileHandler(log_dir / f"v5_equity_downloader_simple_{timestamp}.log")
        console_handler = logging.StreamHandler()
        
        formatter = logging.Formatter('%(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    
    return logger

logger = setup_logging()

class V5EquityDataDownloaderSimple:
    """Simplified V5 Equity Data Downloader"""
    
    def __init__(self, max_workers: int = 10):
        self.data_dir = Path("/media/knight2/EDB/numer_crypto_temp/data")
        self.raw_dir = self.data_dir / "raw" / "equity"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_workers = max_workers
        self.eds_file = self.data_dir / "raw" / "identifier" / "EDS_20241223.csv"
        
        # Temporal lag
        self.cutoff_date = datetime.now() - timedelta(days=1)
        
        logger.info("🚀 V5 Simple Equity Data Downloader initialized")
        logger.info(f"📁 Output directory: {self.raw_dir}")
        logger.info(f"🧵 Max workers: {self.max_workers}")
        logger.info(f"📅 Cutoff date: {self.cutoff_date.date()}")
    
    def load_eds_tickers(self, max_tickers: Optional[int] = None) -> List[Tuple[str, str]]:
        """Load ticker mappings from EDS file"""
        logger.info("📊 Loading EDS ticker mappings...")
        
        if not self.eds_file.exists():
            raise FileNotFoundError(f"EDS file not found: {self.eds_file}")
        
        df = pd.read_csv(self.eds_file)
        
        # Load all valid tickers (US and international)
        all_tickers = []
        for _, row in df.iterrows():
            try:
                yahoo_ticker = str(row.get('yahoo_ticker', '')).strip()
                numerai_ticker = str(row.get('numerai_ticker', '')).strip()
                
                # Filter for valid ticker format (allow dots for international symbols)
                if (yahoo_ticker and numerai_ticker and 
                    yahoo_ticker not in ['nan', ''] and numerai_ticker not in ['nan', ''] and
                    len(yahoo_ticker) <= 12):  # Extended length for international symbols
                    all_tickers.append((numerai_ticker, yahoo_ticker))
                    
            except Exception:
                continue
        
        if max_tickers:
            all_tickers = all_tickers[:max_tickers]
        
        logger.info(f"✅ Loaded {len(all_tickers)} valid ticker mappings (international symbols included)")
        return all_tickers
    
    def download_single_ticker(self, ticker_info: Tuple[str, str]) -> Optional[pd.DataFrame]:
        """Download data for a single ticker"""
        numerai_ticker, yahoo_ticker = ticker_info
        
        try:
            # Simple yfinance call
            end_date = self.cutoff_date
            start_date = end_date - timedelta(days=3000)  # ~9 years
            
            ticker = yf.Ticker(yahoo_ticker)
            hist = ticker.history(start=start_date, end=end_date)
            
            if hist.empty or len(hist) < 50:
                return None
            
            # Process data
            df = hist.reset_index()
            df.columns = df.columns.str.lower()
            df['numerai_ticker'] = numerai_ticker
            df['yahoo_ticker'] = yahoo_ticker
            df['date'] = pd.to_datetime(df['date']).dt.date
            
            # Clean dividend data to handle string values like "0.14 USD"
            if 'dividends' in df.columns:
                df['dividends'] = df['dividends'].astype(str)
                # Extract numeric part from strings like "0.14 USD"
                df['dividends'] = df['dividends'].str.extract(r'([0-9]*\.?[0-9]+)').fillna(0)
                df['dividends'] = pd.to_numeric(df['dividends'], errors='coerce').fillna(0)
            
            # Clean other potential string columns
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Handle stock splits column if it exists
            if 'stock splits' in df.columns:
                df['stock splits'] = pd.to_numeric(df['stock splits'], errors='coerce').fillna(0)
            
            # Simple temporal check
            max_date = pd.to_datetime(df['date']).max()
            if max_date > pd.Timestamp(self.cutoff_date):
                df = df[pd.to_datetime(df['date']) <= pd.Timestamp(self.cutoff_date)]
            
            return df
            
        except Exception as e:
            logger.debug(f"Failed to download {yahoo_ticker}: {e}")
            return None
    
    def download_parallel(self, ticker_list: List[Tuple[str, str]]) -> Tuple[List[pd.DataFrame], int]:
        """Download tickers in parallel"""
        logger.info(f"📥 Downloading {len(ticker_list)} tickers in parallel...")
        
        successful_data = []
        failed_count = 0
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self.download_single_ticker, ticker): ticker 
                      for ticker in ticker_list}
            
            with tqdm(total=len(futures), desc="Downloading") as pbar:
                for future in as_completed(futures):
                    result = future.result()
                    if result is not None:
                        successful_data.append(result)
                    else:
                        failed_count += 1
                    pbar.update(1)
        
        logger.info(f"✅ Downloaded: {len(successful_data)}, Failed: {failed_count}")
        return successful_data, failed_count
    
    def save_consolidated_data(self, data_list: List[pd.DataFrame]) -> str:
        """Save consolidated equity data"""
        if not data_list:
            return ""
        
        logger.info("💾 Saving consolidated equity data...")
        
        # Combine all data
        combined_df = pd.concat(data_list, ignore_index=True)
        combined_df = combined_df.sort_values(['numerai_ticker', 'date'])
        
        # Save with date in filename
        date_str = datetime.now().strftime('%Y%m%d')
        filename = f"equity_data_{date_str}.parquet"
        filepath = self.raw_dir / filename
        
        combined_df.to_parquet(filepath, index=False)
        
        logger.info(f"✅ Saved: {filepath}")
        logger.info(f"📊 Shape: {combined_df.shape}")
        logger.info(f"📊 Unique tickers: {combined_df['numerai_ticker'].nunique()}")
        
        return str(filepath)
    
    def create_statistics(self, data_list: List[pd.DataFrame], failed_count: int) -> str:
        """Create statistics file"""
        logger.info("📊 Creating statistics...")
        
        if not data_list:
            return ""
        
        combined_df = pd.concat(data_list, ignore_index=True)
        
        # Calculate statistics
        total_attempted = len(data_list) + failed_count
        successful = len(data_list)
        
        date_str = datetime.now().strftime('%Y%m%d')
        stats_file = self.raw_dir / f"equity_statistics_{date_str}.csv"
        
        stats_data = {
            'metric': [
                'total_attempted',
                'successful_downloads',
                'failed_downloads', 
                'success_rate_percent',
                'total_records',
                'unique_tickers',
                'date_range_start',
                'date_range_end'
            ],
            'value': [
                total_attempted,
                successful,
                failed_count,
                round(successful / total_attempted * 100, 1),
                len(combined_df),
                combined_df['numerai_ticker'].nunique(),
                combined_df['date'].min(),
                combined_df['date'].max()
            ]
        }
        
        pd.DataFrame(stats_data).to_csv(stats_file, index=False)
        
        logger.info(f"✅ Statistics saved: {stats_file}")
        return str(stats_file)
    
    def run_download(self, max_tickers: Optional[int] = None) -> Dict[str, Any]:
        """Run complete download pipeline"""
        logger.info("🚀 Starting V5 Simple Equity Download")
        start_time = time.time()
        
        try:
            # Load tickers
            ticker_list = self.load_eds_tickers(max_tickers)
            
            if not ticker_list:
                return {'success': False, 'error': 'No tickers loaded'}
            
            # Download data
            data_list, failed_count = self.download_parallel(ticker_list)
            
            if not data_list:
                return {'success': False, 'error': 'No data downloaded'}
            
            # Save results
            data_file = self.save_consolidated_data(data_list)
            stats_file = self.create_statistics(data_list, failed_count)
            
            duration = time.time() - start_time
            
            result = {
                'success': True,
                'duration_seconds': duration,
                'tickers_attempted': len(ticker_list),
                'tickers_downloaded': len(data_list),
                'data_file': data_file,
                'stats_file': stats_file
            }
            
            logger.info(f"🎉 Download completed in {duration:.1f}s")
            logger.info(f"📊 Success rate: {len(data_list)/len(ticker_list)*100:.1f}%")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Download failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'duration_seconds': time.time() - start_time
            }

def main():
    """Main execution"""
    print("🏦 V5 SIMPLE EQUITY DATA DOWNLOADER")
    print("=" * 50)
    
    downloader = V5EquityDataDownloaderSimple(max_workers=10)
    result = downloader.run_download()  #max_tickers=50000
    
    if result['success']:
        print(f"✅ SUCCESS!")
        print(f"Duration: {result['duration_seconds']:.1f}s")
        print(f"Downloaded: {result['tickers_downloaded']}/{result['tickers_attempted']}")
        print(f"Data file: {result['data_file']}")
        print(f"Stats file: {result['stats_file']}")
    else:
        print(f"❌ FAILED: {result['error']}")

if __name__ == "__main__":
    main()
