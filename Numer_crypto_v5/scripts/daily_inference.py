#!/usr/bin/env python3
"""
V5 Daily Inference Script - Essential for Production
Generates daily predictions using trained V5 models

ESSENTIAL SCRIPT - Used for daily production inference
NO DATA LEAKAGE - NO SYNTHETIC DATA - 1-day lag enforced
"""

import os
import sys
import logging
import warnings
import time
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import traceback

# Suppress warnings for production
warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Headless mode for server deployment
import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

# GPU support if available
try:
    import cudf
    import cuml
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False

# Configure logging
def setup_logging() -> logging.Logger:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path("log")
    log_dir.mkdir(exist_ok=True)
    
    logger = logging.getLogger("v5_daily_inference")
    logger.setLevel(logging.INFO)
    
    file_handler = logging.FileHandler(log_dir / f"daily_inference_{timestamp}.log")
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter('%(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

logger = setup_logging()

class V5DailyInference:
    """
    V5 Daily Inference Engine - Production-Ready
    
    ESSENTIAL FEATURES:
    - Loads latest trained models
    - Generates fresh features with 1-day lag
    - Creates daily predictions
    - Validates for data leakage
    - Outputs Numerai-format submissions
    """
    
    def __init__(self):
        self.data_dir = Path("/media/knight2/EDB/numer_crypto_temp/data")
        self.models_dir = Path("models")
        self.processed_dir = self.data_dir / "processed" / "v5"
        self.submission_dir = self.data_dir / "submission" / "v5"
        
        # Ensure directories exist
        for dir_path in [self.models_dir, self.submission_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Temporal lag for production safety
        self.TEMPORAL_LAG_DAYS = 1
        self.inference_cutoff = datetime.now() - timedelta(days=self.TEMPORAL_LAG_DAYS)
        
        # Model and scaler objects
        self.models = {}
        self.scaler = None
        self.ensemble_weights = {}
        self.selected_features = []
        
        logger.info("🚀 V5 Daily Inference Engine initialized")
        logger.info(f"📅 Inference cutoff: {self.inference_cutoff.date()} (1-day lag)")

    def load_latest_models(self) -> bool:
        """Load latest trained V5 models and metadata"""
        logger.info("📥 Loading latest trained models...")
        
        try:
            # Find latest model files
            model_files = list(self.models_dir.glob("*_model_*.pkl"))
            if not model_files:
                logger.error("❌ No trained models found")
                return False
            
            # Group by timestamp to get latest set
            model_timestamps = {}
            for file in model_files:
                timestamp = file.stem.split('_')[-1]
                if timestamp not in model_timestamps:
                    model_timestamps[timestamp] = []
                model_timestamps[timestamp].append(file)
            
            # Get latest timestamp
            latest_timestamp = max(model_timestamps.keys())
            latest_model_files = model_timestamps[latest_timestamp]
            
            logger.info(f"🕒 Loading models from timestamp: {latest_timestamp}")
            
            # Load models
            for model_file in latest_model_files:
                model_name = model_file.stem.replace(f'_model_{latest_timestamp}', '')
                try:
                    model = joblib.load(model_file)
                    self.models[model_name] = model
                    logger.info(f"✅ Loaded {model_name}")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to load {model_name}: {e}")
            
            if not self.models:
                logger.error("❌ No models loaded successfully")
                return False
            
            # Load scaler
            scaler_files = list(self.models_dir.glob(f"scaler_{latest_timestamp}.pkl"))
            if scaler_files:
                self.scaler = joblib.load(scaler_files[0])
                logger.info("✅ Scaler loaded")
            else:
                logger.warning("⚠️ No scaler found, using default StandardScaler")
                self.scaler = StandardScaler()
            
            # Load ensemble weights and features
            ensemble_files = list(self.models_dir.glob(f"ensemble_results_{latest_timestamp}.json"))
            if ensemble_files:
                with open(ensemble_files[0], 'r') as f:
                    ensemble_data = json.load(f)
                    self.ensemble_weights = ensemble_data.get('model_weights', {})
                    self.selected_features = ensemble_data.get('selected_features', [])
                logger.info(f"✅ Ensemble config loaded: {len(self.selected_features)} features")
            else:
                logger.warning("⚠️ No ensemble config found, using equal weights")
                self.ensemble_weights = {name: 1.0/len(self.models) for name in self.models.keys()}
            
            logger.info(f"✅ Models loaded: {list(self.models.keys())}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Model loading failed: {e}")
            return False

    def generate_fresh_features(self) -> pd.DataFrame:
        """Generate fresh features for inference with strict temporal controls"""
        logger.info("🔧 Generating fresh features for inference...")
        
        try:
            # Load latest price data
            price_files = list(self.data_dir.glob("raw/price/**/crypto_price_data_*.parquet"))
            if not price_files:
                raise FileNotFoundError("No price data found for inference")
            
            latest_file = max(price_files, key=lambda x: x.stat().st_mtime)
            price_data = pd.read_parquet(latest_file)
            price_data['date'] = pd.to_datetime(price_data['date'])
            
            # Apply temporal lag (CRITICAL for production)
            original_count = len(price_data)
            price_data = price_data[price_data['date'] <= self.inference_cutoff]
            
            logger.info(f"⏰ Applied production lag: {original_count} → {len(price_data)} rows")
            logger.info(f"📅 Latest data date: {price_data['date'].max().date()}")
            
            # Generate features using existing selectors
            all_features = []
            
            # Import feature selectors
            sys.path.insert(0, str(Path(__file__).parent / "dataset"))
            
            # PVM Features
            try:
                from pvm.pvm_feature_selector import V5PVMFeatureSelector
                pvm_selector = V5PVMFeatureSelector()
                pvm_features = pvm_selector.generate_pvm_features(price_data)
                logger.info(f"✅ PVM features: {len([col for col in pvm_features.columns if col.startswith('pvm_')])} features")
                all_features.append(pvm_features)
            except Exception as e:
                logger.warning(f"⚠️ PVM features failed: {e}")
            
            # Statistical Features
            try:
                from statistical.statistical_feature_selector import V5StatisticalFeatureSelector
                stat_selector = V5StatisticalFeatureSelector()
                stat_features = stat_selector.generate_statistical_features(price_data)
                logger.info(f"✅ Statistical features: {len([col for col in stat_features.columns if col.startswith('stat_')])} features")
                all_features.append(stat_features)
            except Exception as e:
                logger.warning(f"⚠️ Statistical features failed: {e}")
            
            # Technical Features
            try:
                from technical.technical_feature_selector import V5TechnicalFeatureSelector
                tech_selector = V5TechnicalFeatureSelector()
                tech_features = tech_selector.generate_technical_features(price_data)
                logger.info(f"✅ Technical features: {len([col for col in tech_features.columns if col.endswith('_lag1')])} features")
                all_features.append(tech_features)
            except Exception as e:
                logger.warning(f"⚠️ Technical features failed: {e}")
            
            if not all_features:
                raise ValueError("No features generated for inference")
            
            # Merge all features
            merged_features = all_features[0]
            for feature_df in all_features[1:]:
                merged_features = pd.merge(
                    merged_features, 
                    feature_df, 
                    on=['symbol', 'date'], 
                    how='outer'
                )
            
            # Get most recent data for each symbol (for daily inference)
            latest_features = merged_features.groupby('symbol').last().reset_index()
            
            logger.info(f"✅ Fresh features generated: {len(latest_features)} symbols")
            logger.info(f"🔢 Total feature columns: {len([col for col in merged_features.columns if col not in ['symbol', 'date']])}")
            
            return latest_features
            
        except Exception as e:
            logger.error(f"❌ Fresh feature generation failed: {e}")
            raise

    def validate_inference_data(self, features_df: pd.DataFrame) -> bool:
        """Validate inference data for production safety"""
        logger.info("🔍 Validating inference data...")
        
        try:
            # Check 1: Temporal validation (no future data)
            if 'date' in features_df.columns:
                max_date = pd.to_datetime(features_df['date']).max()
                if max_date > self.inference_cutoff:
                    logger.error(f"❌ DATA LEAKAGE: max_date={max_date} > cutoff={self.inference_cutoff}")
                    return False
            
            # Check 2: Required features validation
            if self.selected_features:
                available_features = set(features_df.columns)
                missing_features = set(self.selected_features) - available_features
                if missing_features:
                    logger.error(f"❌ Missing required features: {len(missing_features)} features")
                    logger.error(f"Sample missing: {list(missing_features)[:5]}")
                    return False
            
            # Check 3: Data quality validation
            if features_df.empty:
                logger.error("❌ Empty feature dataset")
                return False
            
            if 'symbol' not in features_df.columns:
                logger.error("❌ Missing symbol column")
                return False
            
            # Check 4: Realistic value ranges
            numeric_cols = features_df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if features_df[col].isna().all():
                    logger.warning(f"⚠️ All NaN values in {col}")
                
                # Check for extreme outliers
                q99 = features_df[col].quantile(0.99)
                q01 = features_df[col].quantile(0.01)
                
                if abs(q99) > 1e6 or abs(q01) > 1e6:
                    logger.warning(f"⚠️ Extreme values in {col}: Q1={q01}, Q99={q99}")
            
            logger.info("✅ Inference data validation PASSED")
            return True
            
        except Exception as e:
            logger.error(f"❌ Inference validation failed: {e}")
            return False

    def generate_predictions(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Generate predictions using ensemble of trained models"""
        logger.info("🤖 Generating daily predictions...")
        
        try:
            if not self.models:
                raise ValueError("No models loaded for prediction")
            
            # Prepare feature matrix
            if self.selected_features:
                # Use selected features from training
                available_features = [col for col in self.selected_features if col in features_df.columns]
                if len(available_features) < len(self.selected_features) * 0.8:  # Need at least 80% of features
                    logger.warning(f"⚠️ Only {len(available_features)}/{len(self.selected_features)} features available")
                X = features_df[available_features].copy()
            else:
                # Use all numeric features
                exclude_cols = ['symbol', 'date']
                numeric_cols = features_df.select_dtypes(include=[np.number]).columns
                feature_cols = [col for col in numeric_cols if col not in exclude_cols]
                X = features_df[feature_cols].copy()
            
            logger.info(f"🔢 Using {X.shape[1]} features for prediction")
            
            # Handle missing values
            X = X.fillna(X.median())
            
            # Scale features
            if self.scaler is not None:
                try:
                    X_scaled = self.scaler.transform(X)
                    X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
                except Exception as e:
                    logger.warning(f"⚠️ Scaling failed, using original features: {e}")
                    X_scaled = X
            else:
                X_scaled = X
            
            # Generate predictions from each model
            predictions = {}
            for model_name, model in self.models.items():
                try:
                    pred = model.predict(X_scaled)
                    predictions[model_name] = pred
                    logger.info(f"✅ {model_name} predictions generated")
                except Exception as e:
                    logger.warning(f"⚠️ {model_name} prediction failed: {e}")
            
            if not predictions:
                raise ValueError("No model predictions generated")
            
            # Create ensemble prediction
            ensemble_pred = np.zeros(len(X_scaled))
            total_weight = 0
            
            for model_name, pred in predictions.items():
                weight = self.ensemble_weights.get(model_name, 1.0)
                ensemble_pred += weight * pred
                total_weight += weight
            
            # Normalize ensemble
            if total_weight > 0:
                ensemble_pred /= total_weight
            
            # Create prediction DataFrame
            prediction_df = pd.DataFrame({
                'symbol': features_df['symbol'].values,
                'signal': ensemble_pred
            })
            
            # Add individual model predictions for analysis
            for model_name, pred in predictions.items():
                prediction_df[f'{model_name}_signal'] = pred
            
            logger.info(f"✅ Predictions generated for {len(prediction_df)} symbols")
            logger.info(f"📊 Signal range: [{prediction_df['signal'].min():.6f}, {prediction_df['signal'].max():.6f}]")
            
            return prediction_df
            
        except Exception as e:
            logger.error(f"❌ Prediction generation failed: {e}")
            raise

    def create_submission_file(self, predictions_df: pd.DataFrame) -> str:
        """Create Numerai-format submission file"""
        logger.info("📄 Creating daily submission file...")
        
        try:
            # Normalize signals to [0.001, 0.999] range (Numerai requirement)
            signals = predictions_df['signal'].copy()
            signal_min, signal_max = signals.min(), signals.max()
            
            if signal_max > signal_min:
                normalized_signals = 0.001 + 0.998 * (signals - signal_min) / (signal_max - signal_min)
            else:
                normalized_signals = pd.Series(0.5, index=signals.index)
            
            # Create submission DataFrame
            submission_df = pd.DataFrame({
                'symbol': predictions_df['symbol'].values,
                'signal': normalized_signals
            })
            
            # Sort by symbol for consistency
            submission_df = submission_df.sort_values('symbol').reset_index(drop=True)
            
            # Save submission file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            submission_file = self.submission_dir / f"daily_submission_{timestamp}.csv"
            submission_df.to_csv(submission_file, index=False)
            
            # Save detailed predictions for analysis
            analysis_file = self.submission_dir / f"daily_predictions_detailed_{timestamp}.csv"
            predictions_df.to_csv(analysis_file, index=False)
            
            # Create submission metadata
            metadata = {
                'timestamp': timestamp,
                'submission_type': 'daily_inference',
                'symbols_count': len(submission_df),
                'signal_stats': {
                    'min': float(submission_df['signal'].min()),
                    'max': float(submission_df['signal'].max()),
                    'mean': float(submission_df['signal'].mean()),
                    'std': float(submission_df['signal'].std())
                },
                'models_used': list(self.models.keys()),
                'temporal_cutoff': self.inference_cutoff.isoformat(),
                'data_leakage_check': 'PASSED',
                'production_ready': True
            }
            
            metadata_file = self.submission_dir / f"daily_submission_metadata_{timestamp}.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            logger.info(f"✅ Submission created: {submission_file}")
            logger.info(f"📊 Symbols: {len(submission_df)}")
            logger.info(f"📊 Signal range: [{submission_df['signal'].min():.6f}, {submission_df['signal'].max():.6f}]")
            
            return str(submission_file)
            
        except Exception as e:
            logger.error(f"❌ Submission creation failed: {e}")
            raise

    def run_daily_inference(self) -> Dict[str, Any]:
        """Run complete daily inference pipeline"""
        logger.info("🚀 Starting V5 Daily Inference Pipeline")
        logger.info("=" * 60)
        
        start_time = time.time()
        results = {}
        
        try:
            # Phase 1: Load Models
            logger.info("📥 PHASE 1: LOADING TRAINED MODELS")
            if not self.load_latest_models():
                raise ValueError("Failed to load trained models")
            results['models_loaded'] = list(self.models.keys())
            
            # Phase 2: Generate Fresh Features
            logger.info("🔧 PHASE 2: GENERATING FRESH FEATURES")
            features_df = self.generate_fresh_features()
            results['features_generated'] = features_df.shape
            
            # Phase 3: Validate Data
            logger.info("🔍 PHASE 3: VALIDATING INFERENCE DATA")
            if not self.validate_inference_data(features_df):
                raise ValueError("Inference data validation failed - CRITICAL")
            
            # Phase 4: Generate Predictions
            logger.info("🤖 PHASE 4: GENERATING PREDICTIONS")
            predictions_df = self.generate_predictions(features_df)
            results['predictions_generated'] = len(predictions_df)
            
            # Phase 5: Create Submission
            logger.info("📄 PHASE 5: CREATING SUBMISSION FILE")
            submission_file = self.create_submission_file(predictions_df)
            results['submission_file'] = submission_file
            
            # Calculate performance metrics
            total_time = time.time() - start_time
            results['runtime_seconds'] = total_time
            results['runtime_minutes'] = total_time / 60
            
            # Final summary
            logger.info("=" * 60)
            logger.info("✅ V5 DAILY INFERENCE COMPLETED SUCCESSFULLY")
            logger.info("=" * 60)
            logger.info(f"🕒 Total Runtime: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
            logger.info(f"🤖 Models Used: {len(self.models)} ({', '.join(self.models.keys())})")
            logger.info(f"📊 Symbols Processed: {len(predictions_df)}")
            logger.info(f"📄 Submission File: {submission_file}")
            logger.info(f"⏰ Data Cutoff: {self.inference_cutoff.date()} (1-day lag)")
            logger.info("🛡️ No data leakage - Production safe")
            logger.info("🚫 No synthetic data used")
            logger.info("=" * 60)
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Daily inference failed: {e}")
            logger.error(traceback.format_exc())
            raise

def main():
    """Main entry point for daily inference"""
    inference_engine = V5DailyInference()
    
    try:
        results = inference_engine.run_daily_inference()
        print(f"🎉 Daily inference completed successfully!")
        print(f"📄 Submission: {results['submission_file']}")
        print(f"📊 Symbols: {results['predictions_generated']}")
        print(f"🕒 Runtime: {results['runtime_minutes']:.1f} minutes")
        
    except KeyboardInterrupt:
        logger.info("⏹️ Daily inference interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"❌ Daily inference failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
