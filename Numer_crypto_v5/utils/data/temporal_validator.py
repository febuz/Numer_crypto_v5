#!/usr/bin/env python3
"""
V5 Temporal Validator - GPU-Accelerated with cudf-pandas fallback

High-performance temporal validation utilities with GPU acceleration
and smart fallbacks for data leakage prevention.

ACCELERATION STACK:
- Primary: cudf (GPU pandas)
- Fallback: Polars (fast CPU processing)
- Alternative: PySpark (distributed processing)
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
import warnings

# GPU-accelerated pandas with intelligent fallback
try:
    import cudf.pandas
    cudf.pandas.install()
    import pandas as pd
    GPU_PANDAS_AVAILABLE = True
    BACKEND = "cudf-pandas"
except ImportError:
    try:
        import polars as pl
        import pandas as pd
        GPU_PANDAS_AVAILABLE = False
        POLARS_AVAILABLE = True
        BACKEND = "polars"
    except ImportError:
        import pandas as pd
        GPU_PANDAS_AVAILABLE = False
        POLARS_AVAILABLE = False
        BACKEND = "pandas"

# PySpark for distributed processing
try:
    from pyspark.sql import SparkSession, DataFrame as SparkDataFrame
    from pyspark.sql.functions import col, max as spark_max, min as spark_min
    PYSPARK_AVAILABLE = True
except ImportError:
    PYSPARK_AVAILABLE = False

import numpy as np

# Handle polars import for type hints
try:
    import polars as pl
except ImportError:
    pl = None

warnings.filterwarnings('ignore')

class TemporalValidator:
    """
    GPU-accelerated temporal validation for V5 pipeline
    
    Provides high-performance validation of temporal lag requirements
    with automatic GPU/CPU optimization and smart backend selection.
    """
    
    def __init__(self, temporal_lag_days: int = 1, use_gpu: bool = True, 
                 use_distributed: bool = False):
        self.TEMPORAL_LAG_DAYS = temporal_lag_days
        self.cutoff_date = datetime.now() - timedelta(days=temporal_lag_days)
        self.use_gpu = use_gpu and GPU_PANDAS_AVAILABLE
        self.use_distributed = use_distributed and PYSPARK_AVAILABLE
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize Spark if requested and available
        if self.use_distributed:
            self.spark = self._initialize_spark()
        else:
            self.spark = None
        
        self.logger.info(f"🚀 TemporalValidator initialized - Backend: {BACKEND}")
        if self.use_gpu:
            self.logger.info("⚡ GPU acceleration enabled (cudf-pandas)")
        elif POLARS_AVAILABLE:
            self.logger.info("🏃 Fast CPU processing (Polars)")
        else:
            self.logger.info("🐼 Standard pandas processing")
    
    def _initialize_spark(self) -> Optional[SparkSession]:
        """Initialize optimized Spark session for distributed processing"""
        try:
            spark = SparkSession.builder \
                .appName("V5_TemporalValidator") \
                .config("spark.sql.adaptive.enabled", "true") \
                .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
                .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
                .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
                .getOrCreate()
            
            spark.sparkContext.setLogLevel("WARN")
            self.logger.info("⚡ PySpark distributed processing enabled")
            return spark
            
        except Exception as e:
            self.logger.warning(f"Failed to initialize Spark: {e}")
            return None
    
    def validate_temporal_lag_pandas(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        GPU-accelerated temporal validation using cudf-pandas
        
        Args:
            df: DataFrame with 'date' column
            
        Returns:
            Validation results dictionary
        """
        try:
            if 'date' not in df.columns:
                return {
                    'valid': True,
                    'reason': 'no_date_column',
                    'backend': BACKEND
                }
            
            # Convert to datetime if needed (GPU-accelerated)
            if not pd.api.types.is_datetime64_any_dtype(df['date']):
                df['date'] = pd.to_datetime(df['date'])
            
            # Vectorized temporal validation
            max_date = df['date'].max()
            min_date = df['date'].min()
            cutoff_timestamp = pd.Timestamp(self.cutoff_date)
            
            # Check for future data leakage
            has_leakage = max_date > cutoff_timestamp
            future_records = (df['date'] > cutoff_timestamp).sum() if has_leakage else 0
            
            return {
                'valid': not has_leakage,
                'max_date': max_date,
                'min_date': min_date,
                'cutoff_date': cutoff_timestamp,
                'future_records': int(future_records),
                'total_records': len(df),
                'backend': BACKEND,
                'reason': 'data_leakage_detected' if has_leakage else 'temporal_lag_ok'
            }
            
        except Exception as e:
            self.logger.error(f"❌ Pandas temporal validation failed: {e}")
            return {
                'valid': False,
                'error': str(e),
                'backend': BACKEND
            }
    
    def validate_temporal_lag_polars(self, df: Union[pd.DataFrame, Any]) -> Dict[str, Any]:
        """
        High-performance temporal validation using Polars
        
        Args:
            df: DataFrame (pandas or polars)
            
        Returns:
            Validation results dictionary
        """
        try:
            # Convert pandas to polars if needed
            if isinstance(df, pd.DataFrame):
                if 'date' not in df.columns:
                    return {
                        'valid': True,
                        'reason': 'no_date_column',
                        'backend': 'polars'
                    }
                pl_df = pl.from_pandas(df[['date']].copy())
            else:
                pl_df = df
            
            # Ensure datetime type
            if not pl_df['date'].dtype.is_temporal():
                pl_df = pl_df.with_columns(pl.col('date').str.strptime(pl.Datetime))
            
            # Vectorized aggregations (Polars is very fast here)
            stats = pl_df.select([
                pl.col('date').max().alias('max_date'),
                pl.col('date').min().alias('min_date'),
                pl.col('date').count().alias('total_records')
            ]).collect()
            
            max_date = stats['max_date'][0]
            min_date = stats['min_date'][0]
            total_records = stats['total_records'][0]
            
            cutoff_timestamp = datetime.now() - timedelta(days=self.TEMPORAL_LAG_DAYS)
            
            # Check for leakage
            has_leakage = max_date > cutoff_timestamp
            
            if has_leakage:
                future_count = pl_df.filter(
                    pl.col('date') > cutoff_timestamp
                ).height
            else:
                future_count = 0
            
            return {
                'valid': not has_leakage,
                'max_date': max_date,
                'min_date': min_date,
                'cutoff_date': cutoff_timestamp,
                'future_records': future_count,
                'total_records': total_records,
                'backend': 'polars',
                'reason': 'data_leakage_detected' if has_leakage else 'temporal_lag_ok'
            }
            
        except Exception as e:
            self.logger.error(f"❌ Polars temporal validation failed: {e}")
            return {
                'valid': False,
                'error': str(e),
                'backend': 'polars'
            }
    
    def validate_temporal_lag_spark(self, spark_df: SparkDataFrame) -> Dict[str, Any]:
        """
        Distributed temporal validation using PySpark
        
        Args:
            spark_df: Spark DataFrame
            
        Returns:
            Validation results dictionary
        """
        try:
            if 'date' not in spark_df.columns:
                return {
                    'valid': True,
                    'reason': 'no_date_column',
                    'backend': 'pyspark'
                }
            
            # Distributed aggregations
            date_stats = spark_df.agg(
                spark_max(col('date')).alias('max_date'),
                spark_min(col('date')).alias('min_date')
            ).collect()[0]
            
            max_date = date_stats['max_date']
            min_date = date_stats['min_date']
            total_records = spark_df.count()
            
            cutoff_timestamp = datetime.now() - timedelta(days=self.TEMPORAL_LAG_DAYS)
            
            # Check for leakage
            has_leakage = max_date > cutoff_timestamp
            
            if has_leakage:
                future_count = spark_df.filter(
                    col('date') > cutoff_timestamp
                ).count()
            else:
                future_count = 0
            
            return {
                'valid': not has_leakage,
                'max_date': max_date,
                'min_date': min_date,
                'cutoff_date': cutoff_timestamp,
                'future_records': future_count,
                'total_records': total_records,
                'backend': 'pyspark',
                'reason': 'data_leakage_detected' if has_leakage else 'temporal_lag_ok'
            }
            
        except Exception as e:
            self.logger.error(f"❌ PySpark temporal validation failed: {e}")
            return {
                'valid': False,
                'error': str(e),
                'backend': 'pyspark'
            }
    
    def validate_temporal_lag(self, df: Union[pd.DataFrame, pl.DataFrame, SparkDataFrame]) -> Dict[str, Any]:
        """
        Smart temporal validation with automatic backend selection
        
        Args:
            df: DataFrame (pandas, polars, or spark)
            
        Returns:
            Validation results dictionary
        """
        # Determine optimal backend based on data size and availability
        if hasattr(df, 'count'):  # Spark DataFrame
            if self.use_distributed and self.spark is not None:
                return self.validate_temporal_lag_spark(df)
        
        if isinstance(df, pl.DataFrame) and POLARS_AVAILABLE:
            return self.validate_temporal_lag_polars(df)
        
        # Convert Polars to pandas if needed
        if isinstance(df, pl.DataFrame):
            df = df.to_pandas()
        
        # Default to GPU-accelerated pandas
        return self.validate_temporal_lag_pandas(df)
    
    def batch_validate_files(self, file_patterns: List[str], 
                           base_dir: Optional[Path] = None) -> Dict[str, Dict[str, Any]]:
        """
        Batch validation of multiple files with parallel processing
        
        Args:
            file_patterns: List of glob patterns for files to validate
            base_dir: Base directory for file search
            
        Returns:
            Dictionary mapping file paths to validation results
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        base_dir = base_dir or Path("/media/knight2/EDB/numer_crypto_temp/data")
        results = {}
        
        # Collect all files
        all_files = []
        for pattern in file_patterns:
            files = list(base_dir.glob(pattern))
            all_files.extend(files)
        
        def validate_single_file(file_path):
            try:
                # Smart loading based on backend
                if POLARS_AVAILABLE and not self.use_gpu:
                    df = pl.read_parquet(file_path)
                else:
                    df = pd.read_parquet(file_path)
                
                result = self.validate_temporal_lag(df)
                result['file_path'] = str(file_path)
                result['file_size_mb'] = file_path.stat().st_size / (1024 * 1024)
                
                return str(file_path), result
                
            except Exception as e:
                return str(file_path), {
                    'valid': False,
                    'error': str(e),
                    'file_path': str(file_path)
                }
        
        # Parallel validation
        with ThreadPoolExecutor(max_workers=8) as executor:
            future_to_file = {
                executor.submit(validate_single_file, file_path): file_path
                for file_path in all_files
            }
            
            for future in as_completed(future_to_file):
                file_path, result = future.result()
                results[file_path] = result
                
                if result['valid']:
                    self.logger.info(f"✅ {Path(file_path).name}: temporal lag OK")
                else:
                    self.logger.error(f"❌ {Path(file_path).name}: {result.get('reason', 'validation failed')}")
        
        return results
    
    def apply_temporal_lag_filter(self, df: Union[pd.DataFrame, pl.DataFrame]) -> Union[pd.DataFrame, pl.DataFrame]:
        """
        Apply temporal lag filter with GPU acceleration
        
        Args:
            df: Input DataFrame
            
        Returns:
            Filtered DataFrame with temporal lag applied
        """
        if 'date' not in df.columns:
            return df
        
        cutoff_timestamp = datetime.now() - timedelta(days=self.TEMPORAL_LAG_DAYS)
        
        if isinstance(df, pl.DataFrame):
            # Polars vectorized filtering
            return df.filter(pl.col('date') <= cutoff_timestamp)
        else:
            # GPU-accelerated pandas filtering
            df['date'] = pd.to_datetime(df['date'])
            return df[df['date'] <= cutoff_timestamp].copy()
    
    def generate_temporal_report(self, results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate comprehensive temporal validation report
        
        Args:
            results: Validation results from batch_validate_files
            
        Returns:
            Summary report dictionary
        """
        total_files = len(results)
        valid_files = sum(1 for r in results.values() if r.get('valid', False))
        invalid_files = total_files - valid_files
        
        # Aggregate statistics
        total_records = sum(r.get('total_records', 0) for r in results.values())
        total_future_records = sum(r.get('future_records', 0) for r in results.values())
        
        # Backend usage
        backend_count = {}
        for result in results.values():
            backend = result.get('backend', 'unknown')
            backend_count[backend] = backend_count.get(backend, 0) + 1
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'validation_summary': {
                'total_files': total_files,
                'valid_files': valid_files,
                'invalid_files': invalid_files,
                'validation_rate': valid_files / total_files if total_files > 0 else 0
            },
            'data_summary': {
                'total_records': total_records,
                'future_records': total_future_records,
                'leakage_rate': total_future_records / total_records if total_records > 0 else 0
            },
            'backend_usage': backend_count,
            'temporal_settings': {
                'lag_days': self.TEMPORAL_LAG_DAYS,
                'cutoff_date': self.cutoff_date.isoformat(),
                'gpu_enabled': self.use_gpu,
                'distributed_enabled': self.use_distributed
            },
            'performance': {
                'backend_primary': BACKEND,
                'gpu_pandas_available': GPU_PANDAS_AVAILABLE,
                'polars_available': POLARS_AVAILABLE,
                'pyspark_available': PYSPARK_AVAILABLE
            }
        }
        
        return report
    
    def __del__(self):
        """Cleanup resources"""
        if hasattr(self, 'spark') and self.spark is not None:
            try:
                self.spark.stop()
            except:
                pass