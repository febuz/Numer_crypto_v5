#!/usr/bin/env python3
"""
📊 ADVANCED VALIDATION FRAMEWORK V5
==================================
Comprehensive validation techniques for ultra-reliable RMSE measurement

ADVANCED TECHNIQUES:
- Walk-forward analysis with expanding/rolling windows
- Time-aware cross-validation with gap handling
- Regime-aware validation (bull/bear/sideways markets)
- Bootstrap confidence intervals for metrics
- Temporal stability analysis
- Out-of-sample performance monitoring
- Model degradation detection
- Risk-adjusted performance metrics
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Callable
import logging
from datetime import datetime, timedelta
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr, spearmanr, bootstrap
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AdvancedValidationFramework:
    """Advanced validation framework for cryptocurrency prediction models"""
    
    def __init__(self, min_train_size: int = 252, gap_days: int = 1):
        self.min_train_size = min_train_size  # Minimum training samples
        self.gap_days = gap_days  # Gap between train and test to prevent leakage
        self.validation_results = {}
        self.regime_detector = None
        
        logger.info(f"📊 Advanced Validation Framework initialized")
        logger.info(f"   Min train size: {min_train_size}")
        logger.info(f"   Gap days: {gap_days}")
    
    def walk_forward_validation(self, X: pd.DataFrame, y: pd.Series, 
                               model_func: Callable, 
                               window_type: str = 'expanding',
                               test_size: int = 30,
                               step_size: int = 7) -> Dict[str, Any]:
        """Walk-forward analysis with expanding or rolling windows"""
        logger.info(f"🚶 Starting walk-forward validation ({window_type} window)...")
        
        results = {
            'predictions': [],
            'targets': [],
            'dates': [],
            'fold_metrics': [],
            'model_parameters': []
        }
        
        n_samples = len(X)
        current_pos = self.min_train_size
        fold = 0
        
        try:
            while current_pos + test_size + self.gap_days <= n_samples:
                fold += 1
                
                # Define training window
                if window_type == 'expanding':
                    train_start = 0
                    train_end = current_pos
                elif window_type == 'rolling':
                    train_start = max(0, current_pos - self.min_train_size)
                    train_end = current_pos
                else:
                    raise ValueError(f"Unknown window type: {window_type}")
                
                # Define test window with gap
                test_start = current_pos + self.gap_days
                test_end = min(test_start + test_size, n_samples)
                
                # Extract data
                X_train = X.iloc[train_start:train_end]
                y_train = y.iloc[train_start:train_end]
                X_test = X.iloc[test_start:test_end]
                y_test = y.iloc[test_start:test_end]
                
                if len(X_test) < test_size // 2:  # Skip if test set too small
                    break
                
                logger.debug(f"Fold {fold}: Train [{train_start}:{train_end}], Test [{test_start}:{test_end}]")
                
                # Train model
                model = model_func(X_train, y_train)
                
                # Predict
                y_pred = model.predict(X_test)
                
                # Calculate metrics
                fold_metrics = self._calculate_comprehensive_metrics(y_test, y_pred)
                fold_metrics['fold'] = fold
                fold_metrics['train_size'] = len(X_train)
                fold_metrics['test_size'] = len(X_test)
                fold_metrics['train_period'] = f"{train_start}-{train_end}"
                fold_metrics['test_period'] = f"{test_start}-{test_end}"
                
                # Store results
                results['predictions'].extend(y_pred)
                results['targets'].extend(y_test.values)
                results['dates'].extend(X_test.index if hasattr(X_test, 'index') else range(test_start, test_end))
                results['fold_metrics'].append(fold_metrics)
                
                # Store model parameters if available
                if hasattr(model, 'get_params'):
                    results['model_parameters'].append(model.get_params())
                
                # Move to next position
                current_pos += step_size
            
            # Calculate overall metrics
            overall_metrics = self._calculate_comprehensive_metrics(
                np.array(results['targets']), 
                np.array(results['predictions'])
            )
            
            results['overall_metrics'] = overall_metrics
            results['n_folds'] = fold
            results['window_type'] = window_type
            
            logger.info(f"✅ Walk-forward validation completed: {fold} folds")
            logger.info(f"   Overall RMSE: {overall_metrics['rmse']:.6f}")
            logger.info(f"   Overall Correlation: {overall_metrics['correlation']:.6f}")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Walk-forward validation failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise
    
    def regime_aware_validation(self, X: pd.DataFrame, y: pd.Series,
                               model_func: Callable,
                               regime_column: str = 'market_regime') -> Dict[str, Any]:
        """Validation across different market regimes"""
        logger.info("📈 Starting regime-aware validation...")
        
        if regime_column not in X.columns:
            logger.warning(f"Regime column '{regime_column}' not found, creating artificial regimes")
            X = self._create_artificial_regimes(X, y)
            regime_column = 'artificial_regime'
        
        regimes = X[regime_column].unique()
        regime_results = {}
        
        try:
            for regime in regimes:
                if pd.isna(regime):
                    continue
                    
                logger.info(f"📊 Validating regime: {regime}")
                
                # Get regime data
                regime_mask = X[regime_column] == regime
                X_regime = X[regime_mask]
                y_regime = y[regime_mask]
                
                if len(X_regime) < self.min_train_size:
                    logger.warning(f"Insufficient data for regime {regime}: {len(X_regime)} samples")
                    continue
                
                # Perform time series split within regime
                tscv = TimeSeriesSplit(n_splits=3)
                regime_metrics = []
                
                for train_idx, test_idx in tscv.split(X_regime):
                    X_train = X_regime.iloc[train_idx]
                    y_train = y_regime.iloc[train_idx]
                    X_test = X_regime.iloc[test_idx]
                    y_test = y_regime.iloc[test_idx]
                    
                    # Train model
                    model = model_func(X_train, y_train)
                    y_pred = model.predict(X_test)
                    
                    # Calculate metrics
                    fold_metrics = self._calculate_comprehensive_metrics(y_test, y_pred)
                    regime_metrics.append(fold_metrics)
                
                # Aggregate regime metrics
                regime_results[regime] = {
                    'n_samples': len(X_regime),
                    'n_folds': len(regime_metrics),
                    'avg_rmse': np.mean([m['rmse'] for m in regime_metrics]),
                    'std_rmse': np.std([m['rmse'] for m in regime_metrics]),
                    'avg_correlation': np.mean([m['correlation'] for m in regime_metrics]),
                    'std_correlation': np.std([m['correlation'] for m in regime_metrics]),
                    'fold_metrics': regime_metrics
                }
                
                logger.info(f"   {regime}: RMSE={regime_results[regime]['avg_rmse']:.6f} ± {regime_results[regime]['std_rmse']:.6f}")
            
            logger.info(f"✅ Regime-aware validation completed for {len(regime_results)} regimes")
            return regime_results
            
        except Exception as e:
            logger.error(f"❌ Regime-aware validation failed: {e}")
            raise
    
    def temporal_stability_analysis(self, predictions: np.ndarray, 
                                   targets: np.ndarray,
                                   dates: pd.Series,
                                   window_size: int = 30) -> Dict[str, Any]:
        """Analyze temporal stability of model performance"""
        logger.info(f"📅 Analyzing temporal stability (window size: {window_size})...")
        
        try:
            stability_metrics = {
                'rolling_rmse': [],
                'rolling_correlation': [],
                'rolling_mae': [],
                'rolling_sharpe': [],
                'dates': [],
                'trend_analysis': {},
                'stability_score': 0
            }
            
            # Calculate rolling metrics
            for i in range(window_size, len(predictions)):
                window_start = i - window_size
                window_end = i
                
                window_targets = targets[window_start:window_end]
                window_preds = predictions[window_start:window_end]
                
                # Calculate metrics for this window
                rmse = np.sqrt(mean_squared_error(window_targets, window_preds))
                corr, _ = pearsonr(window_targets, window_preds)
                mae = mean_absolute_error(window_targets, window_preds)
                
                # Sharpe-like ratio
                returns = np.diff(window_preds)
                if len(returns) > 1 and np.std(returns) > 0:
                    sharpe = np.mean(returns) / np.std(returns)
                else:
                    sharpe = 0
                
                stability_metrics['rolling_rmse'].append(rmse)
                stability_metrics['rolling_correlation'].append(corr if not np.isnan(corr) else 0)
                stability_metrics['rolling_mae'].append(mae)
                stability_metrics['rolling_sharpe'].append(sharpe)
                stability_metrics['dates'].append(dates.iloc[i] if len(dates) > i else i)
            
            # Trend analysis
            rolling_rmse = np.array(stability_metrics['rolling_rmse'])
            rolling_corr = np.array(stability_metrics['rolling_correlation'])
            
            # Linear trend in RMSE (should be stable/decreasing)
            x = np.arange(len(rolling_rmse))
            rmse_trend, _ = stats.linregress(x, rolling_rmse)[:2]
            corr_trend, _ = stats.linregress(x, rolling_corr)[:2]
            
            # Volatility of metrics (lower is better)
            rmse_volatility = np.std(rolling_rmse) / np.mean(rolling_rmse)
            corr_volatility = np.std(rolling_corr) / max(np.mean(rolling_corr), 1e-8)
            
            stability_metrics['trend_analysis'] = {
                'rmse_trend': rmse_trend,
                'correlation_trend': corr_trend,
                'rmse_volatility': rmse_volatility,
                'correlation_volatility': corr_volatility,
                'rmse_mean': np.mean(rolling_rmse),
                'rmse_std': np.std(rolling_rmse),
                'correlation_mean': np.mean(rolling_corr),
                'correlation_std': np.std(rolling_corr)
            }
            
            # Calculate stability score (0-100, higher is better)
            stability_score = 100 * (
                0.4 * max(0, 1 - rmse_volatility) +  # Stable RMSE
                0.3 * max(0, 1 - abs(rmse_trend)) +  # No upward RMSE trend
                0.2 * max(0, 1 - corr_volatility) +  # Stable correlation
                0.1 * max(0, corr_trend)             # Positive correlation trend
            )
            
            stability_metrics['stability_score'] = min(100, max(0, stability_score))
            
            logger.info(f"✅ Temporal stability analysis completed")
            logger.info(f"   Stability score: {stability_score:.1f}/100")
            logger.info(f"   RMSE trend: {rmse_trend:.6f} (negative is good)")
            logger.info(f"   RMSE volatility: {rmse_volatility:.3f} (lower is better)")
            
            return stability_metrics
            
        except Exception as e:
            logger.error(f"❌ Temporal stability analysis failed: {e}")
            raise
    
    def bootstrap_confidence_intervals(self, predictions: np.ndarray,
                                     targets: np.ndarray,
                                     n_bootstrap: int = 1000,
                                     confidence_level: float = 0.95) -> Dict[str, Any]:
        """Calculate bootstrap confidence intervals for metrics"""
        logger.info(f"🔄 Calculating bootstrap confidence intervals (n={n_bootstrap})...")
        
        try:
            def calculate_metrics(preds, targs):
                rmse = np.sqrt(mean_squared_error(targs, preds))
                corr, _ = pearsonr(targs, preds)
                mae = mean_absolute_error(targs, preds)
                return {'rmse': rmse, 'correlation': corr if not np.isnan(corr) else 0, 'mae': mae}
            
            # Original metrics
            original_metrics = calculate_metrics(predictions, targets)
            
            # Bootstrap sampling
            bootstrap_metrics = {'rmse': [], 'correlation': [], 'mae': []}
            
            n_samples = len(predictions)
            
            for _ in range(n_bootstrap):
                # Bootstrap sample with replacement
                indices = np.random.choice(n_samples, n_samples, replace=True)
                boot_preds = predictions[indices]
                boot_targets = targets[indices]
                
                # Calculate metrics for bootstrap sample
                boot_metrics = calculate_metrics(boot_preds, boot_targets)
                
                bootstrap_metrics['rmse'].append(boot_metrics['rmse'])
                bootstrap_metrics['correlation'].append(boot_metrics['correlation'])
                bootstrap_metrics['mae'].append(boot_metrics['mae'])
            
            # Calculate confidence intervals
            alpha = 1 - confidence_level
            lower_percentile = (alpha / 2) * 100
            upper_percentile = (1 - alpha / 2) * 100
            
            confidence_intervals = {}
            for metric in ['rmse', 'correlation', 'mae']:
                values = np.array(bootstrap_metrics[metric])
                ci_lower = np.percentile(values, lower_percentile)
                ci_upper = np.percentile(values, upper_percentile)
                
                confidence_intervals[metric] = {
                    'original': original_metrics[metric],
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'ci_lower': ci_lower,
                    'ci_upper': ci_upper,
                    'ci_width': ci_upper - ci_lower
                }
            
            logger.info(f"✅ Bootstrap confidence intervals calculated")
            for metric in ['rmse', 'correlation', 'mae']:
                ci = confidence_intervals[metric]
                logger.info(f"   {metric.upper()}: {ci['original']:.6f} [{ci['ci_lower']:.6f}, {ci['ci_upper']:.6f}]")
            
            return confidence_intervals
            
        except Exception as e:
            logger.error(f"❌ Bootstrap confidence intervals failed: {e}")
            raise
    
    def risk_adjusted_metrics(self, predictions: np.ndarray,
                             targets: np.ndarray,
                             benchmark_return: float = 0.0) -> Dict[str, float]:
        """Calculate risk-adjusted performance metrics"""
        logger.info("📊 Calculating risk-adjusted metrics...")
        
        try:
            # Basic metrics
            rmse = np.sqrt(mean_squared_error(targets, predictions))
            corr, _ = pearsonr(targets, predictions)
            mae = mean_absolute_error(targets, predictions)
            
            # Returns-based metrics
            pred_returns = np.diff(predictions)
            target_returns = np.diff(targets)
            
            # Sharpe ratio (using prediction returns)
            if len(pred_returns) > 1 and np.std(pred_returns) > 0:
                sharpe_ratio = (np.mean(pred_returns) - benchmark_return) / np.std(pred_returns)
            else:
                sharpe_ratio = 0
            
            # Information ratio (tracking error-adjusted performance)
            tracking_error = np.std(pred_returns - target_returns) if len(pred_returns) > 1 else 0
            if tracking_error > 0:
                information_ratio = (np.mean(pred_returns) - np.mean(target_returns)) / tracking_error
            else:
                information_ratio = 0
            
            # Calmar ratio (return/max drawdown)
            cumulative_returns = np.cumsum(pred_returns)
            peak = np.maximum.accumulate(cumulative_returns)
            drawdown = peak - cumulative_returns
            max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0
            
            if max_drawdown > 0:
                calmar_ratio = np.mean(pred_returns) / max_drawdown
            else:
                calmar_ratio = 0
            
            # Sortino ratio (downside deviation)
            downside_returns = pred_returns[pred_returns < benchmark_return]
            if len(downside_returns) > 1:
                downside_deviation = np.std(downside_returns)
                sortino_ratio = (np.mean(pred_returns) - benchmark_return) / downside_deviation
            else:
                sortino_ratio = 0
            
            # Hit rate (percentage of correct directional predictions)
            if len(target_returns) > 0:
                directional_accuracy = np.mean(np.sign(pred_returns) == np.sign(target_returns))
            else:
                directional_accuracy = 0
            
            risk_metrics = {
                'rmse': rmse,
                'correlation': corr if not np.isnan(corr) else 0,
                'mae': mae,
                'sharpe_ratio': sharpe_ratio,
                'information_ratio': information_ratio,
                'calmar_ratio': calmar_ratio,
                'sortino_ratio': sortino_ratio,
                'max_drawdown': max_drawdown,
                'hit_rate': directional_accuracy,
                'tracking_error': tracking_error
            }
            
            logger.info(f"✅ Risk-adjusted metrics calculated")
            logger.info(f"   Sharpe ratio: {sharpe_ratio:.3f}")
            logger.info(f"   Information ratio: {information_ratio:.3f}")
            logger.info(f"   Hit rate: {directional_accuracy:.1%}")
            
            return risk_metrics
            
        except Exception as e:
            logger.error(f"❌ Risk-adjusted metrics calculation failed: {e}")
            raise
    
    def _calculate_comprehensive_metrics(self, targets: np.ndarray, predictions: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive set of metrics"""
        try:
            metrics = {}
            
            # Basic metrics
            metrics['rmse'] = np.sqrt(mean_squared_error(targets, predictions))
            metrics['mae'] = mean_absolute_error(targets, predictions)
            metrics['mse'] = mean_squared_error(targets, predictions)
            
            # Correlation metrics
            corr, p_value = pearsonr(targets, predictions)
            metrics['correlation'] = corr if not np.isnan(corr) else 0
            metrics['correlation_pvalue'] = p_value if not np.isnan(p_value) else 1
            
            spearman_corr, spearman_p = spearmanr(targets, predictions)
            metrics['spearman_correlation'] = spearman_corr if not np.isnan(spearman_corr) else 0
            
            # R-squared
            ss_res = np.sum((targets - predictions) ** 2)
            ss_tot = np.sum((targets - np.mean(targets)) ** 2)
            metrics['r_squared'] = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            # Additional metrics
            metrics['mean_target'] = np.mean(targets)
            metrics['mean_prediction'] = np.mean(predictions)
            metrics['std_target'] = np.std(targets)
            metrics['std_prediction'] = np.std(predictions)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
            return {'rmse': np.inf, 'correlation': 0, 'mae': np.inf}
    
    def _create_artificial_regimes(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Create artificial market regimes based on target volatility"""
        X_regime = X.copy()
        
        # Calculate rolling volatility of targets
        rolling_vol = y.rolling(window=30).std()
        
        # Define regime thresholds
        vol_25 = rolling_vol.quantile(0.33)
        vol_75 = rolling_vol.quantile(0.67)
        
        # Assign regimes
        regime = np.where(rolling_vol <= vol_25, 'low_volatility',
                         np.where(rolling_vol >= vol_75, 'high_volatility', 'medium_volatility'))
        
        X_regime['artificial_regime'] = regime
        
        return X_regime
    
    def run_comprehensive_validation(self, X: pd.DataFrame, y: pd.Series,
                                   model_func: Callable,
                                   validation_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Run comprehensive validation pipeline"""
        logger.info("🚀 Starting comprehensive validation pipeline...")
        
        if validation_config is None:
            validation_config = {
                'walk_forward': True,
                'regime_aware': True,
                'temporal_stability': True,
                'bootstrap_ci': True,
                'risk_adjusted': True
            }
        
        comprehensive_results = {}
        
        try:
            # 1. Walk-forward validation
            if validation_config.get('walk_forward', True):
                logger.info("📈 Running walk-forward validation...")
                wf_results = self.walk_forward_validation(X, y, model_func)
                comprehensive_results['walk_forward'] = wf_results
                
                # Extract predictions for further analysis
                predictions = np.array(wf_results['predictions'])
                targets = np.array(wf_results['targets'])
                dates = pd.Series(wf_results['dates'])
            
            # 2. Regime-aware validation
            if validation_config.get('regime_aware', True):
                logger.info("📊 Running regime-aware validation...")
                regime_results = self.regime_aware_validation(X, y, model_func)
                comprehensive_results['regime_aware'] = regime_results
            
            # 3. Temporal stability analysis
            if validation_config.get('temporal_stability', True) and 'walk_forward' in comprehensive_results:
                logger.info("📅 Running temporal stability analysis...")
                stability_results = self.temporal_stability_analysis(predictions, targets, dates)
                comprehensive_results['temporal_stability'] = stability_results
            
            # 4. Bootstrap confidence intervals
            if validation_config.get('bootstrap_ci', True) and 'walk_forward' in comprehensive_results:
                logger.info("🔄 Calculating bootstrap confidence intervals...")
                bootstrap_results = self.bootstrap_confidence_intervals(predictions, targets)
                comprehensive_results['bootstrap_ci'] = bootstrap_results
            
            # 5. Risk-adjusted metrics
            if validation_config.get('risk_adjusted', True) and 'walk_forward' in comprehensive_results:
                logger.info("📊 Calculating risk-adjusted metrics...")
                risk_results = self.risk_adjusted_metrics(predictions, targets)
                comprehensive_results['risk_adjusted'] = risk_results
            
            # Summary
            if 'walk_forward' in comprehensive_results:
                overall_metrics = comprehensive_results['walk_forward']['overall_metrics']
                logger.info("✅ Comprehensive validation completed")
                logger.info(f"   Overall RMSE: {overall_metrics['rmse']:.6f}")
                logger.info(f"   Overall Correlation: {overall_metrics['correlation']:.6f}")
                
                if 'temporal_stability' in comprehensive_results:
                    stability_score = comprehensive_results['temporal_stability']['stability_score']
                    logger.info(f"   Stability Score: {stability_score:.1f}/100")
            
            self.validation_results = comprehensive_results
            return comprehensive_results
            
        except Exception as e:
            logger.error(f"❌ Comprehensive validation failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise

def test_advanced_validation():
    """Test advanced validation framework"""
    print("🧪 Testing Advanced Validation Framework...")
    
    # Create sample data
    n_samples = 500
    dates = pd.date_range('2020-01-01', periods=n_samples, freq='D')
    
    # Generate synthetic features and targets
    np.random.seed(42)
    X = pd.DataFrame({
        'feature1': np.random.normal(0, 1, n_samples),
        'feature2': np.random.normal(0, 1, n_samples),
        'feature3': np.random.normal(0, 1, n_samples),
    }, index=dates)
    
    # Target with some predictable pattern
    y = pd.Series(0.3 * X['feature1'] + 0.2 * X['feature2'] + np.random.normal(0, 0.1, n_samples), index=dates)
    
    # Simple model function for testing
    def simple_model_func(X_train, y_train):
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        model.fit(X_train, y_train)
        return model
    
    # Test validation framework
    validator = AdvancedValidationFramework(min_train_size=100)
    
    # Run comprehensive validation
    results = validator.run_comprehensive_validation(X, y, simple_model_func)
    
    print(f"✅ Validation testing completed!")
    print(f"   Walk-forward RMSE: {results['walk_forward']['overall_metrics']['rmse']:.6f}")
    print(f"   Stability score: {results['temporal_stability']['stability_score']:.1f}/100")
    
    if 'bootstrap_ci' in results:
        rmse_ci = results['bootstrap_ci']['rmse']
        print(f"   RMSE 95% CI: [{rmse_ci['ci_lower']:.6f}, {rmse_ci['ci_upper']:.6f}]")

if __name__ == "__main__":
    test_advanced_validation()