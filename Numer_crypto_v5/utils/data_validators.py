#!/usr/bin/env python3
"""
V5 Data Validators - Comprehensive Data Quality and Validation
============================================================

Provides comprehensive data validation functionality for V5 pipeline.
Ensures data quality, temporal compliance, and feature validity.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import logging

class DataValidator:
    """Comprehensive data validator for V5 pipeline"""
    
    def __init__(self, temporal_lag_days: int = 1):
        self.temporal_lag_days = temporal_lag_days
        self.cutoff_date = datetime.now() - timedelta(days=temporal_lag_days)
        self.logger = logging.getLogger(__name__)
    
    def validate_temporal_compliance(self, df: pd.DataFrame, 
                                   date_column: str = 'date') -> Tuple[bool, Dict[str, Any]]:
        """Validate temporal compliance for data leakage prevention"""
        validation_result = {
            'test_name': 'temporal_compliance',
            'passed': True,
            'details': {},
            'errors': []
        }
        
        try:
            if date_column not in df.columns:
                validation_result['passed'] = False
                validation_result['errors'].append(f"Date column '{date_column}' not found")
                return False, validation_result
            
            dates = pd.to_datetime(df[date_column])
            max_date = dates.max()
            
            if max_date > pd.Timestamp(self.cutoff_date):
                validation_result['passed'] = False
                validation_result['errors'].append(
                    f"Future data detected: max_date={max_date} > cutoff={self.cutoff_date}"
                )
            
            validation_result['details'] = {
                'max_date': max_date,
                'cutoff_date': self.cutoff_date,
                'temporal_lag_enforced': max_date <= pd.Timestamp(self.cutoff_date)
            }
            
            return validation_result['passed'], validation_result
            
        except Exception as e:
            validation_result['passed'] = False
            validation_result['errors'].append(f"Validation error: {str(e)}")
            return False, validation_result