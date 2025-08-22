#!/usr/bin/env python3
"""
V5 Utilities Package - Reusable Feature Calculation Functions

This package contains essential utility classes for feature calculation,
data processing, and model validation used across V5 pipeline components.

Structure:
- feature_calculators.py: Core feature calculation functions
- data_validators.py: Data quality and temporal validation utilities  
- model_evaluators.py: Model performance evaluation utilities
- gpu_accelerators.py: GPU-accelerated computation utilities

CRITICAL RULES:
- ALL functions must enforce 1-day temporal lag
- NO synthetic data generation allowed
- Real market data processing only
- Strict data leakage prevention
"""

from .feature_calculators import (
    PVMCalculator,
    StatisticalCalculator,
    TechnicalCalculator
)

from .data_validators import (
    TemporalValidator,
    DataQualityValidator,
    FeatureValidator
)

from .model_evaluators import (
    PerformanceEvaluator,
    ValidationMetrics
)

__version__ = "5.0.0"
__all__ = [
    "PVMCalculator",
    "StatisticalCalculator", 
    "TechnicalCalculator",
    "TemporalValidator",
    "DataQualityValidator",
    "FeatureValidator",
    "PerformanceEvaluator",
    "ValidationMetrics"
]