#!/usr/bin/env python3
"""
🎯 ADAPTIVE ENSEMBLE OPTIMIZER V5
================================
Advanced ensemble techniques for ultra-low RMSE achievement

ADVANCED TECHNIQUES:
- Dynamic weight adjustment based on recent performance
- Time-aware ensemble with regime-specific weights
- Bayesian Model Averaging (BMA)
- Stacked generalization with meta-learners
- Multi-objective optimization (RMSE + Correlation + Sharpe)
- Online learning with forgetting factors
- Uncertainty quantification and confidence intervals
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime, timedelta
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import Ridge, ElasticNet, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.optimize import minimize
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AdaptiveEnsembleOptimizer:
    """Advanced adaptive ensemble optimizer for cryptocurrency prediction"""
    
    def __init__(self, window_size: int = 60, forgetting_factor: float = 0.95):
        self.window_size = window_size  # Days to look back for performance
        self.forgetting_factor = forgetting_factor  # Exponential decay for weights
        self.weight_history = {}
        self.performance_history = {}
        self.meta_models = {}
        
        logger.info(f"🎯 Adaptive Ensemble Optimizer initialized")
        logger.info(f"   Window size: {window_size} days")
        logger.info(f"   Forgetting factor: {forgetting_factor}")
    
    def calculate_dynamic_weights(self, predictions: Dict[str, np.ndarray], 
                                targets: np.ndarray, 
                                dates: pd.Series) -> Dict[str, float]:
        """Calculate dynamic weights based on recent performance"""
        logger.info("⚡ Calculating dynamic ensemble weights...")
        
        if len(predictions) == 0:
            return {}
        
        model_names = list(predictions.keys())
        n_models = len(model_names)
        n_samples = len(targets)
        
        # Initialize weights
        weights = {name: 1.0/n_models for name in model_names}
        
        if n_samples < self.window_size:
            logger.warning(f"Insufficient data for dynamic weighting: {n_samples} < {self.window_size}")
            return weights
        
        try:
            # Calculate rolling performance metrics
            recent_performance = {}
            
            for i, model_name in enumerate(model_names):
                model_preds = predictions[model_name]
                
                # Calculate metrics over rolling windows
                rolling_rmse = []
                rolling_corr = []
                rolling_sharpe = []
                
                for j in range(self.window_size, n_samples):
                    window_start = j - self.window_size
                    window_targets = targets[window_start:j]
                    window_preds = model_preds[window_start:j]
                    
                    # RMSE
                    rmse = np.sqrt(mean_squared_error(window_targets, window_preds))
                    rolling_rmse.append(rmse)
                    
                    # Correlation
                    corr, _ = pearsonr(window_targets, window_preds)
                    rolling_corr.append(corr if not np.isnan(corr) else 0)
                    
                    # Sharpe-like ratio (return/volatility)
                    returns = np.diff(window_preds)
                    if len(returns) > 1 and np.std(returns) > 0:
                        sharpe = np.mean(returns) / np.std(returns)
                    else:
                        sharpe = 0
                    rolling_sharpe.append(sharpe)
                
                # Recent performance with exponential decay
                recent_rmse = np.average(rolling_rmse, weights=self._exponential_weights(len(rolling_rmse)))
                recent_corr = np.average(rolling_corr, weights=self._exponential_weights(len(rolling_corr)))
                recent_sharpe = np.average(rolling_sharpe, weights=self._exponential_weights(len(rolling_sharpe)))
                
                recent_performance[model_name] = {
                    'rmse': recent_rmse,
                    'correlation': recent_corr,
                    'sharpe': recent_sharpe,
                    'composite_score': recent_corr / max(recent_rmse, 1e-8) + recent_sharpe * 0.1
                }
            
            # Calculate weights based on composite performance
            total_score = sum([perf['composite_score'] for perf in recent_performance.values()])
            
            if total_score > 0:
                for model_name in model_names:
                    score = recent_performance[model_name]['composite_score']
                    weights[model_name] = max(score / total_score, 0.01)  # Minimum weight 1%
            
            # Normalize weights
            total_weight = sum(weights.values())
            weights = {name: w/total_weight for name, w in weights.items()}
            
            logger.info(f"✅ Dynamic weights calculated: {weights}")
            
            # Store performance history
            self.performance_history[datetime.now().isoformat()] = recent_performance
            
            return weights
            
        except Exception as e:
            logger.error(f"❌ Dynamic weight calculation failed: {e}")
            return {name: 1.0/n_models for name in model_names}
    
    def _exponential_weights(self, n: int) -> np.ndarray:
        """Generate exponential decay weights (more recent = higher weight)"""
        weights = np.array([self.forgetting_factor**i for i in range(n-1, -1, -1)])
        return weights / weights.sum()
    
    def optimize_ensemble_weights(self, predictions: Dict[str, np.ndarray], 
                                 targets: np.ndarray,
                                 method: str = 'multi_objective') -> Dict[str, float]:
        """Optimize ensemble weights using advanced techniques"""
        logger.info(f"🔧 Optimizing ensemble weights using {method}...")
        
        model_names = list(predictions.keys())
        n_models = len(model_names)
        
        if n_models == 0:
            return {}
        
        # Stack predictions
        pred_matrix = np.column_stack([predictions[name] for name in model_names])
        
        try:
            if method == 'multi_objective':
                return self._multi_objective_optimization(pred_matrix, targets, model_names)
            elif method == 'bayesian':
                return self._bayesian_model_averaging(pred_matrix, targets, model_names)
            elif method == 'constrained':
                return self._constrained_optimization(pred_matrix, targets, model_names)
            else:
                return self._simple_optimization(pred_matrix, targets, model_names)
                
        except Exception as e:
            logger.error(f"❌ Weight optimization failed: {e}")
            return {name: 1.0/n_models for name in model_names}
    
    def _multi_objective_optimization(self, pred_matrix: np.ndarray, 
                                    targets: np.ndarray, 
                                    model_names: List[str]) -> Dict[str, float]:
        """Multi-objective optimization (RMSE + Correlation + Sharpe)"""
        
        def objective(weights):
            # Ensure weights sum to 1
            weights = weights / weights.sum()
            ensemble_pred = pred_matrix @ weights
            
            # RMSE component (minimize)
            rmse = np.sqrt(mean_squared_error(targets, ensemble_pred))
            
            # Correlation component (maximize)
            corr, _ = pearsonr(targets, ensemble_pred)
            corr = corr if not np.isnan(corr) else 0
            
            # Sharpe-like ratio (maximize)
            returns = np.diff(ensemble_pred)
            if len(returns) > 1 and np.std(returns) > 0:
                sharpe = np.mean(returns) / np.std(returns)
            else:
                sharpe = 0
            
            # Combined objective (minimize)
            # Weight RMSE heavily, but include correlation and Sharpe
            return rmse - 0.1 * corr - 0.01 * sharpe
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: w.sum() - 1.0},  # Weights sum to 1
        ]
        
        # Bounds (all weights between 0 and 1)
        bounds = [(0, 1) for _ in range(len(model_names))]
        
        # Initial guess (equal weights)
        x0 = np.ones(len(model_names)) / len(model_names)
        
        # Optimize
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
        
        if result.success:
            optimal_weights = result.x / result.x.sum()  # Normalize
            return {name: weight for name, weight in zip(model_names, optimal_weights)}
        else:
            logger.warning("Multi-objective optimization failed, using equal weights")
            return {name: 1.0/len(model_names) for name in model_names}
    
    def _bayesian_model_averaging(self, pred_matrix: np.ndarray, 
                                 targets: np.ndarray, 
                                 model_names: List[str]) -> Dict[str, float]:
        """Bayesian Model Averaging with model evidence"""
        
        # Calculate log-likelihood for each model (Gaussian assumption)
        log_likelihoods = []
        
        for i, model_name in enumerate(model_names):
            model_preds = pred_matrix[:, i]
            residuals = targets - model_preds
            
            # Log-likelihood assuming Gaussian errors
            sigma_sq = np.var(residuals)
            if sigma_sq > 0:
                log_likelihood = -0.5 * len(targets) * np.log(2 * np.pi * sigma_sq) - 0.5 * np.sum(residuals**2) / sigma_sq
            else:
                log_likelihood = -np.inf
            
            log_likelihoods.append(log_likelihood)
        
        # Convert to model probabilities (Bayes weights)
        log_likelihoods = np.array(log_likelihoods)
        max_ll = np.max(log_likelihoods)
        exp_ll = np.exp(log_likelihoods - max_ll)  # Numerical stability
        
        weights = exp_ll / np.sum(exp_ll)
        
        return {name: weight for name, weight in zip(model_names, weights)}
    
    def _constrained_optimization(self, pred_matrix: np.ndarray, 
                                 targets: np.ndarray, 
                                 model_names: List[str]) -> Dict[str, float]:
        """Constrained optimization with additional constraints"""
        
        def objective(weights):
            weights = weights / weights.sum()
            ensemble_pred = pred_matrix @ weights
            return mean_squared_error(targets, ensemble_pred)
        
        # Additional constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: w.sum() - 1.0},  # Weights sum to 1
            {'type': 'ineq', 'fun': lambda w: w - 0.05},     # Minimum weight 5%
            {'type': 'ineq', 'fun': lambda w: 0.5 - w},     # Maximum weight 50%
        ]
        
        bounds = [(0.05, 0.5) for _ in range(len(model_names))]
        x0 = np.ones(len(model_names)) / len(model_names)
        
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
        
        if result.success:
            optimal_weights = result.x / result.x.sum()
            return {name: weight for name, weight in zip(model_names, optimal_weights)}
        else:
            return {name: 1.0/len(model_names) for name in model_names}
    
    def _simple_optimization(self, pred_matrix: np.ndarray, 
                           targets: np.ndarray, 
                           model_names: List[str]) -> Dict[str, float]:
        """Simple RMSE minimization"""
        
        def objective(weights):
            weights = weights / weights.sum()
            ensemble_pred = pred_matrix @ weights
            return mean_squared_error(targets, ensemble_pred)
        
        constraints = [{'type': 'eq', 'fun': lambda w: w.sum() - 1.0}]
        bounds = [(0, 1) for _ in range(len(model_names))]
        x0 = np.ones(len(model_names)) / len(model_names)
        
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
        
        if result.success:
            optimal_weights = result.x / result.x.sum()
            return {name: weight for name, weight in zip(model_names, optimal_weights)}
        else:
            return {name: 1.0/len(model_names) for name in model_names}
    
    def train_meta_models(self, predictions: Dict[str, np.ndarray], 
                         targets: np.ndarray,
                         dates: pd.Series) -> Dict[str, Any]:
        """Train meta-models for stacked generalization"""
        logger.info("🧠 Training meta-models for stacked generalization...")
        
        model_names = list(predictions.keys())
        pred_matrix = np.column_stack([predictions[name] for name in model_names])
        
        meta_models = {}
        
        try:
            # Time series split for meta-model training
            tscv = TimeSeriesSplit(n_splits=3)
            
            # 1. Ridge meta-model
            ridge_meta = Ridge(alpha=1.0, random_state=42)
            
            meta_predictions_ridge = []
            for train_idx, val_idx in tscv.split(pred_matrix):
                X_train_meta, X_val_meta = pred_matrix[train_idx], pred_matrix[val_idx]
                y_train_meta, y_val_meta = targets[train_idx], targets[val_idx]
                
                ridge_meta.fit(X_train_meta, y_train_meta)
                meta_pred = ridge_meta.predict(X_val_meta)
                meta_predictions_ridge.extend(meta_pred)
            
            # Train final Ridge meta-model on all data
            ridge_meta.fit(pred_matrix, targets)
            meta_models['ridge'] = ridge_meta
            
            # 2. ElasticNet meta-model
            elastic_meta = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)
            elastic_meta.fit(pred_matrix, targets)
            meta_models['elasticnet'] = elastic_meta
            
            # 3. Random Forest meta-model
            rf_meta = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1)
            rf_meta.fit(pred_matrix, targets)
            meta_models['randomforest'] = rf_meta
            
            # Evaluate meta-models
            meta_performance = {}
            for name, model in meta_models.items():
                meta_pred = model.predict(pred_matrix)
                rmse = np.sqrt(mean_squared_error(targets, meta_pred))
                corr, _ = pearsonr(targets, meta_pred)
                
                meta_performance[name] = {
                    'rmse': rmse,
                    'correlation': corr if not np.isnan(corr) else 0
                }
            
            logger.info(f"✅ Meta-models trained: {list(meta_models.keys())}")
            logger.info(f"Meta-model performance: {meta_performance}")
            
            self.meta_models = meta_models
            return meta_models
            
        except Exception as e:
            logger.error(f"❌ Meta-model training failed: {e}")
            return {}
    
    def generate_meta_predictions(self, predictions: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Generate predictions using trained meta-models"""
        
        if not self.meta_models:
            logger.warning("No meta-models available")
            return {}
        
        model_names = list(predictions.keys())
        pred_matrix = np.column_stack([predictions[name] for name in model_names])
        
        meta_predictions = {}
        
        for meta_name, meta_model in self.meta_models.items():
            try:
                meta_pred = meta_model.predict(pred_matrix)
                meta_predictions[f'meta_{meta_name}'] = meta_pred
            except Exception as e:
                logger.error(f"Meta-model {meta_name} prediction failed: {e}")
        
        return meta_predictions
    
    def create_advanced_ensemble(self, predictions: Dict[str, np.ndarray], 
                               targets: np.ndarray,
                               dates: pd.Series,
                               method: str = 'adaptive') -> Dict[str, Any]:
        """Create advanced ensemble using multiple techniques"""
        logger.info(f"🎯 Creating advanced ensemble using {method} method...")
        
        try:
            ensemble_results = {}
            
            if method == 'adaptive':
                # Dynamic weights based on recent performance
                weights = self.calculate_dynamic_weights(predictions, targets, dates)
                
            elif method == 'optimized':
                # Multi-objective optimization
                weights = self.optimize_ensemble_weights(predictions, targets, method='multi_objective')
                
            elif method == 'bayesian':
                # Bayesian model averaging
                weights = self.optimize_ensemble_weights(predictions, targets, method='bayesian')
                
            elif method == 'stacked':
                # Train meta-models and use them
                meta_models = self.train_meta_models(predictions, targets, dates)
                meta_predictions = self.generate_meta_predictions(predictions)
                
                # Combine original and meta predictions
                all_predictions = {**predictions, **meta_predictions}
                weights = self.optimize_ensemble_weights(all_predictions, targets, method='multi_objective')
                predictions = all_predictions
                
            else:
                # Simple equal weights
                weights = {name: 1.0/len(predictions) for name in predictions.keys()}
            
            # Generate ensemble prediction
            ensemble_pred = np.zeros(len(targets))
            for model_name, model_preds in predictions.items():
                ensemble_pred += weights.get(model_name, 0) * model_preds
            
            # Calculate ensemble metrics
            ensemble_rmse = np.sqrt(mean_squared_error(targets, ensemble_pred))
            ensemble_corr, _ = pearsonr(targets, ensemble_pred)
            ensemble_mae = mean_absolute_error(targets, ensemble_pred)
            
            # Sharpe-like ratio
            returns = np.diff(ensemble_pred)
            if len(returns) > 1 and np.std(returns) > 0:
                ensemble_sharpe = np.mean(returns) / np.std(returns)
            else:
                ensemble_sharpe = 0
            
            ensemble_results = {
                'method': method,
                'predictions': ensemble_pred,
                'weights': weights,
                'rmse': ensemble_rmse,
                'correlation': ensemble_corr,
                'mae': ensemble_mae,
                'sharpe': ensemble_sharpe,
                'individual_predictions': predictions,
                'meta_models': self.meta_models if method == 'stacked' else {}
            }
            
            logger.info(f"✅ Advanced ensemble created:")
            logger.info(f"   Method: {method}")
            logger.info(f"   RMSE: {ensemble_rmse:.6f}")
            logger.info(f"   Correlation: {ensemble_corr:.6f}")
            logger.info(f"   MAE: {ensemble_mae:.6f}")
            logger.info(f"   Sharpe: {ensemble_sharpe:.6f}")
            logger.info(f"   Weights: {weights}")
            
            return ensemble_results
            
        except Exception as e:
            logger.error(f"❌ Advanced ensemble creation failed: {e}")
            raise
    
    def uncertainty_quantification(self, predictions: Dict[str, np.ndarray], 
                                 method: str = 'bootstrap') -> Dict[str, Any]:
        """Quantify prediction uncertainty"""
        logger.info(f"🔍 Quantifying prediction uncertainty using {method}...")
        
        try:
            if method == 'bootstrap':
                return self._bootstrap_uncertainty(predictions)
            elif method == 'ensemble_std':
                return self._ensemble_std_uncertainty(predictions)
            else:
                return self._simple_uncertainty(predictions)
                
        except Exception as e:
            logger.error(f"❌ Uncertainty quantification failed: {e}")
            return {}
    
    def _bootstrap_uncertainty(self, predictions: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Bootstrap-based uncertainty estimation"""
        n_bootstrap = 100
        model_names = list(predictions.keys())
        n_samples = len(predictions[model_names[0]])
        
        bootstrap_results = []
        
        for _ in range(n_bootstrap):
            # Bootstrap sample
            indices = np.random.choice(n_samples, n_samples, replace=True)
            
            # Calculate ensemble for this bootstrap sample
            bootstrap_weights = {name: np.random.dirichlet(np.ones(len(model_names)))[i] 
                               for i, name in enumerate(model_names)}
            
            ensemble_pred = np.zeros(n_samples)
            for name, preds in predictions.items():
                ensemble_pred += bootstrap_weights[name] * preds
            
            bootstrap_results.append(ensemble_pred[indices])
        
        # Calculate confidence intervals
        bootstrap_matrix = np.array(bootstrap_results)
        mean_pred = np.mean(bootstrap_matrix, axis=0)
        lower_ci = np.percentile(bootstrap_matrix, 2.5, axis=0)
        upper_ci = np.percentile(bootstrap_matrix, 97.5, axis=0)
        uncertainty = np.std(bootstrap_matrix, axis=0)
        
        return {
            'mean_prediction': mean_pred,
            'lower_ci': lower_ci,
            'upper_ci': upper_ci,
            'uncertainty': uncertainty,
            'confidence_interval_width': upper_ci - lower_ci
        }
    
    def _ensemble_std_uncertainty(self, predictions: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Ensemble standard deviation uncertainty"""
        pred_matrix = np.column_stack(list(predictions.values()))
        
        mean_pred = np.mean(pred_matrix, axis=1)
        std_pred = np.std(pred_matrix, axis=1)
        
        return {
            'mean_prediction': mean_pred,
            'uncertainty': std_pred,
            'lower_ci': mean_pred - 1.96 * std_pred,
            'upper_ci': mean_pred + 1.96 * std_pred
        }
    
    def _simple_uncertainty(self, predictions: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Simple uncertainty based on prediction variance"""
        pred_matrix = np.column_stack(list(predictions.values()))
        variance = np.var(pred_matrix, axis=1)
        
        return {
            'uncertainty': np.sqrt(variance),
            'variance': variance
        }

def test_adaptive_ensemble():
    """Test adaptive ensemble optimizer"""
    print("🧪 Testing Adaptive Ensemble Optimizer...")
    
    # Create sample predictions and targets
    n_samples = 1000
    dates = pd.date_range('2020-01-01', periods=n_samples, freq='D')
    
    # Generate synthetic predictions from 5 models
    np.random.seed(42)
    targets = np.random.normal(0.5, 0.1, n_samples)
    
    predictions = {
        'model1': targets + np.random.normal(0, 0.05, n_samples),  # Good model
        'model2': targets + np.random.normal(0, 0.1, n_samples),   # Average model
        'model3': targets + np.random.normal(0, 0.15, n_samples),  # Poor model
        'model4': targets + np.random.normal(0, 0.08, n_samples),  # Good model
        'model5': np.random.normal(0.5, 0.2, n_samples),          # Random model
    }
    
    # Test ensemble optimizer
    optimizer = AdaptiveEnsembleOptimizer(window_size=60)
    
    # Test different ensemble methods
    methods = ['adaptive', 'optimized', 'bayesian', 'stacked']
    
    for method in methods:
        print(f"\n🔧 Testing {method} ensemble...")
        ensemble_result = optimizer.create_advanced_ensemble(predictions, targets, dates, method=method)
        print(f"   RMSE: {ensemble_result['rmse']:.6f}")
        print(f"   Correlation: {ensemble_result['correlation']:.6f}")
        print(f"   Weights: {ensemble_result['weights']}")
    
    # Test uncertainty quantification
    print(f"\n🔍 Testing uncertainty quantification...")
    uncertainty = optimizer.uncertainty_quantification(predictions, method='bootstrap')
    print(f"   Mean uncertainty: {np.mean(uncertainty['uncertainty']):.6f}")
    print(f"   95% CI width: {np.mean(uncertainty['confidence_interval_width']):.6f}")
    
    print("✅ Adaptive ensemble testing completed!")

if __name__ == "__main__":
    test_adaptive_ensemble()