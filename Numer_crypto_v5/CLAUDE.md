# CLAUDE.md - Numer_crypto_v5

This file provides guidance to Claude Code (claude.ai/code) when working with the advanced cryptocurrency prediction system for Numerai tournaments.

## Project Overview

This is the V5 iteration of the Numerai Crypto prediction pipeline, designed to achieve aggressive performance targets:
- **RMSE Target**: < 0.15 (vs V4's 0.214999)
- **Correlation Target**: > 0.5 
- **MMC Target**: > 0.2

## V5 Architecture Improvements

### Performance Enhancements
- **Advanced Feature Selection**: Focus on top 600 comprehensive features from master analysis
- **GPU-Accelerated Processing**: Dual GPU setup with RTX 3090 x2 for batch processing
- **Enhanced Ensemble Methods**: Multiple model stacking with bias correction
- **Temporal Lag Implementation**: Proper 1-day lag to prevent data leakage

### Key Feature Categories (Based on Analysis)
1. **PVM Features** (Price-Volume-Momentum) - Highest importance scores (9220.93+)
2. **Statistical Features** - volume_volatility_30d, price_volatility_30d
3. **Technical Indicators** - RSI, SMA, MACD with optimized periods
4. **Sentiment Features** - Advanced sentiment indicators with lag
5. **Onchain Metrics** - Blockchain transaction and wallet features

## Common Commands

### Running V5 Pipeline
```bash
# Full comprehensive pipeline (all models, 60min time limits)
python scripts/numerai_crypto_pipeline.py comprehensive

# Fast mode (limited models, 10min time limits)
python scripts/numerai_crypto_pipeline.py fast

# Feature selector mode (generate features only)
python scripts/numerai_crypto_pipeline.py features

# Prophet-only mode (time series focused)
python scripts/numerai_crypto_pipeline.py prophet
```

### Environment Setup
```bash
# Install V5 dependencies
pip install -r requirements_v5.txt

# Setup GPU environment
./setup_v5_gpu.sh

# Initialize feature selectors
python scripts/initialize_v5_selectors.py
```

## V5 Data Structure

### Primary Data Locations
- `/media/knight2/EDB/numer_crypto_temp/data/raw/` - Raw data sources
- `/media/knight2/EDB/numer_crypto_temp/data/processed/v5/` - V5 processed features
- `/media/knight2/EDB/numer_crypto_temp/data/submission/v5/` - V5 predictions
- `/media/knight2/EDB/numer_crypto_temp/data/models/v5/` - V5 trained models

### Feature Selector Structure
```
scripts/dataset/v5/
├── pvm/                    # Price-Volume-Momentum selectors
│   ├── pvm_feature_selector.py
│   └── pvm_advanced_selector.py
├── statistical/            # Statistical feature selectors  
│   ├── volatility_selector.py
│   └── correlation_selector.py
├── technical/              # Technical indicator selectors
│   ├── technical_selector.py
│   └── advanced_ta_selector.py
├── sentiment/              # Sentiment feature selectors
│   ├── sentiment_selector.py
│   └── news_sentiment_selector.py
├── onchain/               # Onchain metric selectors
│   ├── onchain_selector.py
│   └── blockchain_metrics_selector.py
└── ensemble/              # Ensemble feature selectors
    ├── ensemble_selector.py
    └── meta_feature_selector.py
```

## V5 Models and Methods (Updated 2025-08-21)

### Core Model Suite (100% Success Rate)
✅ **Production Models**: All models now train successfully with robust error handling
- **LightGBM**: Enhanced hyperparameters with GPU acceleration + CPU fallback
- **XGBoost**: Advanced tree methods with GPU acceleration + CPU fallback  
- **CatBoost**: Categorical boost with GPU training + CPU fallback
- **RandomForest**: Ensemble trees with GPU acceleration (cuML/CPU fallback)
- **Neural Ensemble**: Multi-architecture ensemble (Transformer + ResNet + LSTM)

### Removed Models
❌ **Ridge Regression**: Removed due to feature mask complications and stability issues

### Ensemble Strategy
1. **Level 1**: Base models with 5-fold CV
2. **Level 2**: Meta-learners on out-of-fold predictions
3. **Level 3**: Final ensemble with dynamic weighting
4. **Bias Correction**: Prediction difference features and rolling statistics

## V5 Feature Engineering

### Advanced Features (Top 50 from Analysis)
```python
# PVM Features (Highest Importance)
pvm_0085_mean, pvm_0094_mean, pvm_0193_std, pvm_0270_std, pvm_0391_std

# Statistical Features  
volume_volatility_30d, price_volatility_30d, day_of_year_cos, close_rank

# Technical Indicators
rsi_30, close_ma_30d, macd_30d, bollinger_bands

# Sentiment Features
sentiment_0000, sentiment_0503, sentiment_0845

# Mathematical Transforms
sqrt_open, sqrt_low, ratio_low_div_close, ratio_open_div_high
```

### Feature Selection Process
1. **Correlation Analysis**: Remove features with >0.95 correlation
2. **Mutual Information**: Select top features by MI score
3. **Random Forest Importance**: Tree-based feature ranking
4. **Lasso Regularization**: L1 penalty for sparse selection
5. **Variance Threshold**: Remove low-variance features
6. **Forward Selection**: Iterative RMSE-based selection

## Performance Targets and Validation

### V5 Targets (Aggressive)
- **Primary RMSE**: < 0.15 (improvement from V4's 0.214999)
- **Correlation**: > 0.5 (Pearson and Spearman)
- **MMC**: > 0.2 (Mean Model Correlation)
- **Feature Count**: 50-200 optimal features
- **Training Time**: < 90 minutes full pipeline

### Validation Strategy
- **TimeSeriesSplit**: 5-fold temporal validation
- **Walk-Forward**: Rolling window validation
- **Out-of-Sample**: 20% holdout for final evaluation
- **Cross-Asset**: Symbol-specific validation

## GPU Configuration

### V5 GPU Setup
```python
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'  # Dual GPU
os.environ['CUPY_GPU_MEMORY_LIMIT'] = '24576'  # 24GB per GPU
```

### Memory Optimization
- **Batch Processing**: 50k feature batches
- **Mixed Precision**: FP16 for model training
- **Memory Mapping**: Efficient data loading
- **Garbage Collection**: Aggressive cleanup

## V5 Development Guidelines

### Code Standards
- **Type Hints**: All functions must have type annotations
- **Docstrings**: Google-style documentation
- **Error Handling**: Comprehensive exception handling
- **Logging**: Structured logging with timestamps
- **Testing**: Unit tests for all selectors

### Data Quality
- **Temporal Lag**: All features implement 1-day lag (`_lag1` suffix)
- **NaN Handling**: Forward fill with limits
- **Outlier Detection**: IQR-based clipping
- **Normalization**: StandardScaler for continuous features

## Important Policies

### V5 Specific Rules (Updated 2025-08-21)
- **No Synthetic Data**: Only real market data allowed
- **Feature Validation**: Must outperform random benchmarks by >0.1% RMSE
- **Model Diversity**: Exactly 5 core model types (LightGBM, XGBoost, CatBoost, RandomForest, Neural Ensemble)
- **Robust Training**: All models must have GPU/CPU fallback mechanisms
- **Exception Handling**: Comprehensive error handling prevents pipeline crashes
- **Ensemble Weighting**: Based on inverse RMSE performance
- **Temporal Consistency**: No future information leakage

### Performance Monitoring
- **Real-time Metrics**: Live RMSE/Correlation tracking
- **Feature Importance**: Dynamic importance scoring
- **Model Performance**: Per-model contribution tracking
- **Resource Usage**: GPU/Memory utilization monitoring

## Success Criteria

### V5 Completion Requirements
1. **Performance**: Achieve RMSE < 0.15 on validation set
2. **Correlation**: Maintain Pearson correlation > 0.5
3. **MMC**: Achieve MMC > 0.2 on test set
4. **Robustness**: Consistent performance across 10 runs
5. **Documentation**: Complete feature and model documentation

### Deliverables
- Trained V5 models achieving targets
- Feature importance rankings and analysis
- Performance comparison with V4
- Submission files for Numerai tournament
- Comprehensive documentation and setup guides

## Recent Fixes (2025-08-21)

### Pipeline Stability Improvements ✅
- **Fixed Model Training Crashes**: Added comprehensive exception handling for all models
- **GPU Memory Management**: Implemented automatic GPU/CPU fallbacks for memory issues
- **Validation Error Handling**: Fixed `'NoneType' object has no attribute 'predict'` errors
- **Ridge Model Removal**: Completely removed problematic Ridge regression and feature mask
- **100% Model Success Rate**: All 5 core models now train successfully

### Error Handling Enhancements
- **Graceful Degradation**: Pipeline continues even if individual models fail
- **Detailed Error Logging**: Specific error messages for each model failure
- **Memory Fallbacks**: CatBoost automatically falls back to CPU when GPU memory insufficient
- **Validation Robustness**: Cross-validation handles None models and prediction failures

### Testing Framework
- **Comprehensive Tests**: Added test suite in `/media/knight2/EDB/numer_crypto_temp/tests_v5/`
- **Model Training Tests**: Individual model testing to identify specific issues
- **Pipeline Validation**: End-to-end pipeline testing without crashes

## Quick Start

```bash
cd /media/knight2/EDB/repos/Numer_crypto_v5
python scripts/initialize_v5.py
python scripts/numerai_crypto_pipeline.py comprehensive
```

This V5 system builds on V4's foundation while targeting significantly improved performance through advanced feature engineering, ensemble methods, and GPU acceleration. Recent stability fixes ensure 100% reliable model training.