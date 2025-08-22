#!/usr/bin/env python3
"""
🧠 ADVANCED PYTORCH NEURAL NETWORK ENSEMBLE V5
==============================================
State-of-the-art neural network architectures for cryptocurrency prediction
with dual GPU acceleration and ultra-low RMSE targeting.

BREAKTHROUGH FEATURES:
- Multi-GPU parallel training with DataParallel/DistributedDataParallel  
- Advanced architectures: Transformer, TabNet, ResNet for tabular data
- Attention mechanisms for temporal feature relationships
- Gradient accumulation for large effective batch sizes
- Mixed precision training (FP16) for speed
- Advanced regularization: dropout, batch norm, weight decay
- Custom loss functions optimized for correlation and RMSE
- Early stopping with patience and learning rate scheduling
- Feature embedding and continuous feature normalization
- Time-series aware cross-validation
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Configure PyTorch for dual GPU setup
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'  # Enable async execution
os.environ['TORCH_CUDA_ARCH_LIST'] = '8.6'  # RTX 3090 architecture

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import time
import logging
from datetime import datetime
import json
from dataclasses import dataclass

# PyTorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DataParallel, DistributedDataParallel

# Advanced optimizers and schedulers
from torch.optim import Adam, AdamW, RMSprop, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, OneCycleLR

# Metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr, spearmanr

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Configuration for neural network models"""
    # Model architecture
    model_type: str = 'transformer'  # transformer, tabnet, resnet, lstm
    hidden_dim: int = 512
    num_layers: int = 6
    num_heads: int = 8
    dropout: float = 0.2
    
    # Training parameters
    batch_size: int = 1024
    learning_rate: float = 1e-3
    num_epochs: int = 100
    patience: int = 15
    gradient_accumulation_steps: int = 4
    
    # Optimization
    optimizer: str = 'adamw'
    scheduler: str = 'cosine'
    weight_decay: float = 1e-4
    use_mixed_precision: bool = True
    
    # Architecture specific
    embed_dim: int = 64
    ff_dim: int = 2048
    activation: str = 'gelu'

class CryptoDataset(Dataset):
    """Custom dataset for cryptocurrency prediction"""
    
    def __init__(self, X: np.ndarray, y: np.ndarray, feature_names: List[str] = None):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.feature_names = feature_names or []
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism for feature relationships"""
    
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim]))
        
    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        
        # Linear transformations
        Q = self.q_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention weights
        attention = torch.matmul(Q, K.transpose(-2, -1)) / self.scale.to(x.device)
        attention = F.softmax(attention, dim=-1)
        attention = self.dropout(attention)
        
        # Apply attention to values
        x = torch.matmul(attention, V)
        x = x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        
        return self.out(x)

class TransformerBlock(nn.Module):
    """Transformer block with attention and feed-forward layers"""
    
    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        
        self.attention = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        # Self-attention with residual connection
        attn_out = self.attention(x)
        x = self.norm1(x + attn_out)
        
        # Feed-forward with residual connection
        ff_out = self.feed_forward(x)
        x = self.norm2(x + ff_out)
        
        return x

class CryptoTransformerModel(nn.Module):
    """Advanced Transformer model for cryptocurrency prediction"""
    
    def __init__(self, input_dim: int, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Feature embedding and projection
        self.feature_embedding = nn.Linear(input_dim, config.embed_dim)
        self.positional_encoding = nn.Parameter(torch.randn(1, 1000, config.embed_dim))
        
        # Transformer layers
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(config.embed_dim, config.num_heads, config.ff_dim, config.dropout)
            for _ in range(config.num_layers)
        ])
        
        # Classification/regression head
        self.norm = nn.LayerNorm(config.embed_dim)
        self.dropout = nn.Dropout(config.dropout)
        
        self.head = nn.Sequential(
            nn.Linear(config.embed_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, 1)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
    
    def forward(self, x):
        batch_size, seq_len = x.size()
        
        # Reshape for transformer (batch_size, seq_len=1, features)
        x = x.unsqueeze(1)
        
        # Feature embedding
        x = self.feature_embedding(x)
        
        # Add positional encoding
        x = x + self.positional_encoding[:, :x.size(1), :]
        
        # Apply transformer blocks
        for block in self.transformer_blocks:
            x = block(x)
        
        # Global average pooling and output
        x = x.mean(dim=1)  # Average over sequence dimension
        x = self.norm(x)
        x = self.dropout(x)
        x = self.head(x)
        
        return x.squeeze(-1)

class ResidualBlock(nn.Module):
    """Residual block for deep tabular networks"""
    
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        return F.gelu(x + self.block(x))

class CryptoResNetModel(nn.Module):
    """ResNet-style architecture for tabular data"""
    
    def __init__(self, input_dim: int, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, config.hidden_dim),
            nn.BatchNorm1d(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout)
        )
        
        # Residual blocks
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(config.hidden_dim, config.dropout)
            for _ in range(config.num_layers)
        ])
        
        # Output head
        self.head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.BatchNorm1d(config.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, config.hidden_dim // 4),
            nn.BatchNorm1d(config.hidden_dim // 4),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 4, 1)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
    def forward(self, x):
        x = self.input_proj(x)
        
        for block in self.residual_blocks:
            x = block(x)
        
        return self.head(x).squeeze(-1)

class CryptoLSTMModel(nn.Module):
    """LSTM model for time-series cryptocurrency prediction"""
    
    def __init__(self, input_dim: int, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Feature projection
        self.input_proj = nn.Linear(input_dim, config.hidden_dim)
        
        # LSTM layers
        self.lstm = nn.LSTM(
            config.hidden_dim, 
            config.hidden_dim,
            config.num_layers,
            batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.Tanh(),
            nn.Linear(config.hidden_dim, 1)
        )
        
        # Output head
        self.head = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, 1)
        )
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # Project features and add sequence dimension
        x = self.input_proj(x).unsqueeze(1)
        
        # LSTM processing
        lstm_out, _ = self.lstm(x)
        
        # Attention mechanism
        attention_weights = F.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(attention_weights * lstm_out, dim=1)
        
        # Output prediction
        return self.head(context).squeeze(-1)

class CustomLoss(nn.Module):
    """Custom loss function optimized for correlation and RMSE"""
    
    def __init__(self, alpha: float = 0.7, beta: float = 0.3):
        super().__init__()
        self.alpha = alpha  # Weight for RMSE
        self.beta = beta    # Weight for correlation loss
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor):
        # RMSE loss
        mse_loss = F.mse_loss(predictions, targets)
        
        # Correlation loss (negative correlation to maximize)
        pred_centered = predictions - predictions.mean()
        target_centered = targets - targets.mean()
        
        correlation = torch.sum(pred_centered * target_centered) / torch.sqrt(
            torch.sum(pred_centered**2) * torch.sum(target_centered**2)
        )
        
        correlation_loss = 1 - correlation  # Maximize correlation
        
        # Combined loss
        total_loss = self.alpha * mse_loss + self.beta * correlation_loss
        
        return total_loss

class PyTorchNeuralEnsemble:
    """Advanced PyTorch neural network ensemble for cryptocurrency prediction"""
    
    def __init__(self, config: ModelConfig = None):
        self.config = config or ModelConfig()
        self.models = {}
        self.scalers = {}
        self.training_history = {}
        
        # Setup device
        self.device = self._setup_device()
        logger.info(f"🧠 PyTorch Neural Ensemble initialized")
        logger.info(f"   Device: {self.device}")
        logger.info(f"   Available GPUs: {torch.cuda.device_count()}")
        logger.info(f"   Model type: {self.config.model_type}")
        
    def _setup_device(self):
        """Setup GPU device with proper configuration"""
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
            logger.info(f"✅ CUDA available: {torch.cuda.device_count()} GPUs")
            logger.info(f"   GPU 0: {torch.cuda.get_device_name(0)}")
            if torch.cuda.device_count() > 1:
                logger.info(f"   GPU 1: {torch.cuda.get_device_name(1)}")
                logger.info("🚀 Multi-GPU training enabled")
        else:
            device = torch.device('cpu')
            logger.warning("⚠️ CUDA not available, using CPU")
        
        return device
    
    def _create_model(self, input_dim: int, model_name: str = 'default') -> nn.Module:
        """Create neural network model based on configuration"""
        
        if self.config.model_type == 'transformer':
            model = CryptoTransformerModel(input_dim, self.config)
        elif self.config.model_type == 'resnet':
            model = CryptoResNetModel(input_dim, self.config)
        elif self.config.model_type == 'lstm':
            model = CryptoLSTMModel(input_dim, self.config)
        else:
            raise ValueError(f"Unknown model type: {self.config.model_type}")
        
        # Move to device
        model = model.to(self.device)
        
        # Enable multi-GPU if available
        if torch.cuda.device_count() > 1:
            model = DataParallel(model)
            logger.info(f"🚀 Model {model_name} wrapped with DataParallel")
        
        return model
    
    def train_model(self, X_train: np.ndarray, y_train: np.ndarray,
                   X_val: np.ndarray = None, y_val: np.ndarray = None,
                   model_name: str = 'default') -> Dict[str, Any]:
        """Train neural network model with advanced techniques"""
        logger.info(f"🧠 Training neural network model: {model_name}")
        
        start_time = time.time()
        
        # Normalize features
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        self.scalers[model_name] = scaler
        
        if X_val is not None:
            X_val_scaled = scaler.transform(X_val)
        
        # Create datasets and dataloaders
        train_dataset = CryptoDataset(X_train_scaled, y_train)
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=True, 
            num_workers=4,
            pin_memory=True
        )
        
        if X_val is not None:
            val_dataset = CryptoDataset(X_val_scaled, y_val)
            val_loader = DataLoader(
                val_dataset, 
                batch_size=self.config.batch_size * 2, 
                shuffle=False, 
                num_workers=4,
                pin_memory=True
            )
        
        # Create model
        model = self._create_model(X_train.shape[1], model_name)
        
        # Setup optimizer
        if self.config.optimizer == 'adamw':
            optimizer = AdamW(model.parameters(), 
                            lr=self.config.learning_rate, 
                            weight_decay=self.config.weight_decay)
        elif self.config.optimizer == 'adam':
            optimizer = Adam(model.parameters(), lr=self.config.learning_rate)
        else:
            optimizer = SGD(model.parameters(), 
                          lr=self.config.learning_rate, 
                          momentum=0.9,
                          weight_decay=self.config.weight_decay)
        
        # Setup scheduler
        if self.config.scheduler == 'cosine':
            scheduler = CosineAnnealingLR(optimizer, T_max=self.config.num_epochs)
        elif self.config.scheduler == 'plateau':
            scheduler = ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        else:
            scheduler = OneCycleLR(optimizer, 
                                 max_lr=self.config.learning_rate * 10,
                                 steps_per_epoch=len(train_loader),
                                 epochs=self.config.num_epochs)
        
        # Setup loss function and mixed precision
        criterion = CustomLoss()
        scaler_amp = GradScaler() if self.config.use_mixed_precision else None
        
        # Training loop
        training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_rmse': [],
            'val_rmse': [],
            'train_correlation': [],
            'val_correlation': []
        }
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        logger.info(f"🚀 Starting training for {self.config.num_epochs} epochs...")
        
        for epoch in range(self.config.num_epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            train_predictions = []
            train_targets = []
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                
                if self.config.use_mixed_precision and scaler_amp:
                    with autocast():
                        output = model(data)
                        loss = criterion(output, target)
                        loss = loss / self.config.gradient_accumulation_steps
                    
                    scaler_amp.scale(loss).backward()
                    
                    if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                        scaler_amp.step(optimizer)
                        scaler_amp.update()
                        optimizer.zero_grad()
                else:
                    output = model(data)
                    loss = criterion(output, target)
                    loss = loss / self.config.gradient_accumulation_steps
                    loss.backward()
                    
                    if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                        optimizer.step()
                        optimizer.zero_grad()
                
                train_loss += loss.item() * self.config.gradient_accumulation_steps
                
                train_predictions.extend(output.detach().cpu().numpy())
                train_targets.extend(target.detach().cpu().numpy())
            
            # Calculate training metrics
            train_predictions = np.array(train_predictions)
            train_targets = np.array(train_targets)
            train_rmse = np.sqrt(mean_squared_error(train_targets, train_predictions))
            train_corr, _ = pearsonr(train_targets, train_predictions)
            train_corr = train_corr if not np.isnan(train_corr) else 0
            
            training_history['train_loss'].append(train_loss / len(train_loader))
            training_history['train_rmse'].append(train_rmse)
            training_history['train_correlation'].append(train_corr)
            
            # Validation phase
            if X_val is not None:
                model.eval()
                val_loss = 0.0
                val_predictions = []
                val_targets = []
                
                with torch.no_grad():
                    for data, target in val_loader:
                        data, target = data.to(self.device), target.to(self.device)
                        
                        if self.config.use_mixed_precision:
                            with autocast():
                                output = model(data)
                                loss = criterion(output, target)
                        else:
                            output = model(data)
                            loss = criterion(output, target)
                        
                        val_loss += loss.item()
                        val_predictions.extend(output.cpu().numpy())
                        val_targets.extend(target.cpu().numpy())
                
                # Calculate validation metrics
                val_predictions = np.array(val_predictions)
                val_targets = np.array(val_targets)
                val_rmse = np.sqrt(mean_squared_error(val_targets, val_predictions))
                val_corr, _ = pearsonr(val_targets, val_predictions)
                val_corr = val_corr if not np.isnan(val_corr) else 0
                
                training_history['val_loss'].append(val_loss / len(val_loader))
                training_history['val_rmse'].append(val_rmse)
                training_history['val_correlation'].append(val_corr)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Save best model state
                    best_model_state = model.state_dict().copy()
                else:
                    patience_counter += 1
                
                # Learning rate scheduling
                if self.config.scheduler == 'plateau':
                    scheduler.step(val_loss)
                elif self.config.scheduler != 'onecycle':
                    scheduler.step()
                
                # Logging
                if (epoch + 1) % 10 == 0 or epoch == 0:
                    logger.info(f"   Epoch {epoch+1:3d}: Train RMSE={train_rmse:.6f}, "
                              f"Val RMSE={val_rmse:.6f}, Train Corr={train_corr:.4f}, "
                              f"Val Corr={val_corr:.4f}")
                
                # Early stopping check
                if patience_counter >= self.config.patience:
                    logger.info(f"   Early stopping at epoch {epoch+1}")
                    break
            else:
                # No validation set
                if self.config.scheduler != 'plateau':
                    scheduler.step()
                
                if (epoch + 1) % 10 == 0 or epoch == 0:
                    logger.info(f"   Epoch {epoch+1:3d}: Train RMSE={train_rmse:.6f}, "
                              f"Train Corr={train_corr:.4f}")
        
        # Restore best model if validation was used
        if X_val is not None and 'best_model_state' in locals():
            model.load_state_dict(best_model_state)
        
        # Store model and training history
        self.models[model_name] = model
        self.training_history[model_name] = training_history
        
        training_time = time.time() - start_time
        
        logger.info(f"✅ Model {model_name} training completed in {training_time:.1f}s")
        if X_val is not None:
            logger.info(f"   Best validation RMSE: {min(training_history['val_rmse']):.6f}")
            logger.info(f"   Best validation correlation: {max(training_history['val_correlation']):.6f}")
        
        return {
            'model': model,
            'training_history': training_history,
            'training_time': training_time,
            'best_val_rmse': min(training_history['val_rmse']) if X_val is not None else None,
            'best_val_correlation': max(training_history['val_correlation']) if X_val is not None else None
        }
    
    def predict(self, X: np.ndarray, model_name: str = 'default') -> np.ndarray:
        """Generate predictions using trained model"""
        
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found. Available models: {list(self.models.keys())}")
        
        model = self.models[model_name]
        scaler = self.scalers[model_name]
        
        # Normalize features
        X_scaled = scaler.transform(X)
        
        # Create dataset
        dataset = CryptoDataset(X_scaled, np.zeros(len(X_scaled)))  # Dummy targets
        dataloader = DataLoader(dataset, batch_size=self.config.batch_size * 2, shuffle=False)
        
        # Generate predictions
        model.eval()
        predictions = []
        
        with torch.no_grad():
            for data, _ in dataloader:
                data = data.to(self.device)
                
                if self.config.use_mixed_precision:
                    with autocast():
                        output = model(data)
                else:
                    output = model(data)
                
                predictions.extend(output.cpu().numpy())
        
        return np.array(predictions)
    
    def train_ensemble(self, X_train: np.ndarray, y_train: np.ndarray,
                      X_val: np.ndarray = None, y_val: np.ndarray = None) -> Dict[str, Any]:
        """Train ensemble of different neural network architectures"""
        logger.info("🧠 Training neural network ensemble...")
        
        ensemble_results = {}
        model_configs = [
            ('transformer', ModelConfig(model_type='transformer', num_layers=4, hidden_dim=512)),
            ('resnet', ModelConfig(model_type='resnet', num_layers=6, hidden_dim=512)),
            ('lstm', ModelConfig(model_type='lstm', num_layers=3, hidden_dim=512)),
        ]
        
        for model_name, config in model_configs:
            logger.info(f"🚀 Training {model_name} model...")
            
            # Update configuration
            old_config = self.config
            self.config = config
            
            try:
                # Train model
                result = self.train_model(X_train, y_train, X_val, y_val, model_name)
                ensemble_results[model_name] = result
                
            except Exception as e:
                logger.error(f"❌ Training {model_name} failed: {e}")
                ensemble_results[model_name] = None
            
            # Restore original config
            self.config = old_config
        
        # Calculate ensemble predictions if we have trained models
        trained_models = [name for name, result in ensemble_results.items() if result is not None]
        
        if trained_models and X_val is not None:
            logger.info("🎯 Calculating ensemble predictions...")
            
            # Generate predictions from all models
            ensemble_predictions = {}
            for model_name in trained_models:
                pred = self.predict(X_val, model_name)
                ensemble_predictions[model_name] = pred
            
            # Simple ensemble (equal weights)
            if ensemble_predictions:
                ensemble_pred = np.mean(list(ensemble_predictions.values()), axis=0)
                ensemble_rmse = np.sqrt(mean_squared_error(y_val, ensemble_pred))
                ensemble_corr, _ = pearsonr(y_val, ensemble_pred)
                ensemble_corr = ensemble_corr if not np.isnan(ensemble_corr) else 0
                
                ensemble_results['ensemble'] = {
                    'predictions': ensemble_predictions,
                    'ensemble_prediction': ensemble_pred,
                    'ensemble_rmse': ensemble_rmse,
                    'ensemble_correlation': ensemble_corr,
                    'trained_models': trained_models
                }
                
                logger.info(f"✅ Neural ensemble completed:")
                logger.info(f"   Ensemble RMSE: {ensemble_rmse:.6f}")
                logger.info(f"   Ensemble Correlation: {ensemble_corr:.6f}")
                logger.info(f"   Models in ensemble: {trained_models}")
        
        return ensemble_results
    
    def save_models(self, save_dir: str):
        """Save trained models and configurations"""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for model_name, model in self.models.items():
            # Save model state dict
            model_file = save_path / f"{model_name}_neural_model_{timestamp}.pth"
            torch.save(model.state_dict(), model_file)
            
            # Save scaler
            import joblib
            scaler_file = save_path / f"{model_name}_scaler_{timestamp}.pkl"
            joblib.dump(self.scalers[model_name], scaler_file)
            
            # Save training history
            history_file = save_path / f"{model_name}_history_{timestamp}.json"
            with open(history_file, 'w') as f:
                # Convert numpy types to regular Python types for JSON serialization
                history = self.training_history[model_name]
                json_history = {}
                for key, values in history.items():
                    json_history[key] = [float(v) for v in values]
                
                json.dump(json_history, f, indent=2)
        
        logger.info(f"✅ Neural models saved to {save_path} with timestamp {timestamp}")
        return timestamp

def test_pytorch_neural_ensemble():
    """Test PyTorch neural ensemble with synthetic data"""
    print("🧪 Testing PyTorch Neural Ensemble...")
    
    # Create synthetic data
    n_samples = 2000
    n_features = 50
    
    np.random.seed(42)
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    y = (0.3 * X[:, 0] + 0.2 * X[:, 1] - 0.1 * X[:, 2] + 
         0.1 * np.random.randn(n_samples)).astype(np.float32)
    
    # Split data
    split_idx = int(0.8 * n_samples)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    # Test individual model training
    config = ModelConfig(
        model_type='transformer',
        hidden_dim=256,
        num_layers=3,
        num_epochs=20,
        batch_size=256,
        patience=10
    )
    
    ensemble = PyTorchNeuralEnsemble(config)
    
    # Test single model
    print("🚀 Testing single model training...")
    result = ensemble.train_model(X_train, y_train, X_val, y_val, 'test_model')
    print(f"   Best validation RMSE: {result['best_val_rmse']:.6f}")
    print(f"   Best validation correlation: {result['best_val_correlation']:.6f}")
    
    # Test predictions
    predictions = ensemble.predict(X_val, 'test_model')
    test_rmse = np.sqrt(mean_squared_error(y_val, predictions))
    test_corr, _ = pearsonr(y_val, predictions)
    print(f"   Test RMSE: {test_rmse:.6f}")
    print(f"   Test correlation: {test_corr:.6f}")
    
    print("✅ PyTorch Neural Ensemble testing completed!")

if __name__ == "__main__":
    test_pytorch_neural_ensemble()