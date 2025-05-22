# Predictive Analytics Modules for Mars Habitat Resource Management

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import json
import sys
from tqdm import tqdm
import pickle
from datetime import datetime, timedelta

# Add project directories to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from simulations.rl_environment import MarsHabitatRLEnvironment
from simulations.rl_formulation import MarsHabitatRLFormulation

# Set random seeds for reproducibility
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

class TimeSeriesDataset(Dataset):
    """
    Dataset for time series forecasting
    """
    
    def __init__(self, X, y, seq_length=24):
        """
        Initialize the dataset
        
        Args:
            X (numpy.ndarray): Input features
            y (numpy.ndarray): Target values
            seq_length (int): Sequence length for time series
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.seq_length = seq_length
        
    def __len__(self):
        """
        Get dataset length
        
        Returns:
            int: Dataset length
        """
        return len(self.X) - self.seq_length
    
    def __getitem__(self, idx):
        """
        Get item at index
        
        Args:
            idx (int): Index
            
        Returns:
            tuple: (X_seq, y_seq)
        """
        X_seq = self.X[idx:idx+self.seq_length]
        y_seq = self.y[idx+1:idx+self.seq_length+1]
        return X_seq, y_seq

class LSTMForecaster(nn.Module):
    """
    LSTM-based forecaster for time series prediction
    """
    
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.2):
        """
        Initialize the forecaster
        
        Args:
            input_dim (int): Input dimension
            hidden_dim (int): Hidden dimension
            output_dim (int): Output dimension
            num_layers (int): Number of LSTM layers
            dropout (float): Dropout rate
        """
        super(LSTMForecaster, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Output layer
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, input_dim)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_length, output_dim)
        """
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        
        # Apply output layer to each time step
        output = self.fc(lstm_out)
        
        return output

class TransformerForecaster(nn.Module):
    """
    Transformer-based forecaster for time series prediction
    """
    
    def __init__(self, input_dim, d_model, nhead, num_layers, output_dim, dropout=0.1):
        """
        Initialize the forecaster
        
        Args:
            input_dim (int): Input dimension
            d_model (int): Model dimension
            nhead (int): Number of attention heads
            num_layers (int): Number of transformer layers
            output_dim (int): Output dimension
            dropout (float): Dropout rate
        """
        super(TransformerForecaster, self).__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        self.output_dim = output_dim
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=4*d_model, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        # Output layer
        self.output_layer = nn.Linear(d_model, output_dim)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, input_dim)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_length, output_dim)
        """
        # Project input to d_model dimension
        x = self.input_projection(x)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer expects shape (seq_length, batch_size, d_model)
        x = x.permute(1, 0, 2)
        
        # Apply transformer encoder
        x = self.transformer_encoder(x)
        
        # Restore shape (batch_size, seq_length, d_model)
        x = x.permute(1, 0, 2)
        
        # Apply output layer
        output = self.output_layer(x)
        
        return output

class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer models
    """
    
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        """
        Initialize the positional encoding
        
        Args:
            d_model (int): Model dimension
            dropout (float): Dropout rate
            max_len (int): Maximum sequence length
        """
        super(PositionalEncoding, self).__init__()
        
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        # Register buffer (not a parameter, but should be saved and restored)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, d_model)
            
        Returns:
            torch.Tensor: Output tensor with positional encoding added
        """
        x = x + self.pe[:x.size(1), :]
        return self.dropout(x)

class ResourceForecaster:
    """
    Resource forecaster for Mars habitat resource management
    """
    
    def __init__(self, resource_name, model_type='lstm', device='cpu'):
        """
        Initialize the resource forecaster
        
        Args:
            resource_name (str): Name of the resource to forecast
            model_type (str): Type of model to use ('lstm' or 'transformer')
            device (str): Device to run the model on ('cpu' or 'cuda')
        """
        self.resource_name = resource_name
        self.model_type = model_type
        self.device = device
        
        # Model parameters
        self.input_dim = None
        self.output_dim = 1  # Single resource forecast
        self.seq_length = 24  # 24-hour sequence
        
        # Data preprocessing
        self.feature_scaler = None
        self.target_scaler = None
        
        # Model
        self.model = None
        
        # Training parameters
        self.batch_size = 32
        self.learning_rate = 1e-3
        self.num_epochs = 100
        
        print(f"Resource Forecaster initialized for {resource_name} using {model_type} model")
    
    def preprocess_data(self, data):
        """
        Preprocess data for training
        
        Args:
            data (pandas.DataFrame): Data to preprocess
            
        Returns:
            tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        # Extract features and target
        features = data.drop(columns=[self.resource_name])
        target = data[[self.resource_name]]
        
        # Scale features
        self.feature_scaler = StandardScaler()
        features_scaled = self.feature_scaler.fit_transform(features)
        
        # Scale target
        self.target_scaler = StandardScaler()
        target_scaled = self.target_scaler.fit_transform(target)
        
        # Split data into train, validation, and test sets
        X_train, X_temp, y_train, y_temp = train_test_split(
            features_scaled, target_scaled, test_size=0.3, random_state=RANDOM_SEED
        )
        
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=RANDOM_SEED
        )
        
        # Set input dimension
        self.input_dim = X_train.shape[1]
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def create_model(self):
        """
        Create forecasting model
        
        Returns:
            torch.nn.Module: Forecasting model
        """
        if self.model_type == 'lstm':
            model = LSTMForecaster(
                input_dim=self.input_dim,
                hidden_dim=128,
                output_dim=self.output_dim,
                num_layers=2,
                dropout=0.2
            ).to(self.device)
        elif self.model_type == 'transformer':
            model = TransformerForecaster(
                input_dim=self.input_dim,
                d_model=128,
                nhead=8,
                num_layers=3,
                output_dim=self.output_dim,
                dropout=0.1
            ).to(self.device)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        return model
    
    def train(self, X_train, y_train, X_val, y_val):
        """
        Train the forecasting model
        
        Args:
            X_train (numpy.ndarray): Training features
            y_train (numpy.ndarray): Training targets
            X_val (numpy.ndarray): Validation features
            y_val (numpy.ndarray): Validation targets
            
        Returns:
            dict: Training history
        """
        # Create model
        self.model = self.create_model()
        
        # Create datasets
        train_dataset = TimeSeriesDataset(X_train, y_train, self.seq_length)
        val_dataset = TimeSeriesDataset(X_val, y_val, self.seq_length)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size)
        
        # Loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        # Training history
        history = {
            'train_loss': [],
            'val_loss': []
        }
        
        # Training loop
        best_val_loss = float('inf')
        best_model_state = None
        
        for epoch in tqdm(range(self.num_epochs), desc=f"Training {self.resource_name} forecaster"):
            # Training
            self.model.train()
            train_loss = 0.0
            
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                # Forward pass
                y_pred = self.model(X_batch)
                loss = criterion(y_pred, y_batch)
                
                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * X_batch.size(0)
            
            train_loss /= len(train_loader.dataset)
            history['train_loss'].append(train_loss)
            
            # Validation
            self.model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    
                    # Forward pass
                    y_pred = self.model(X_batch)
                    loss = criterion(y_pred, y_batch)
                    
                    val_loss += loss.item() * X_batch.size(0)
            
            val_loss /= len(val_loader.dataset)
            history['val_loss'].append(val_loss)
            
            # Update learning rate
            scheduler.step(val_loss)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = self.model.state_dict()
            
            # Print progress
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{self.num_epochs} - "
                      f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Load best model
        self.model.load_state_dict(best_model_state)
        
        return history
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the forecasting model
        
        Args:
            X_test (numpy.ndarray): Test features
            y_test (numpy.ndarray): Test targets
            
        Returns:
            dict: Evaluation metrics
        """
        # Create dataset and data loader
        test_dataset = TimeSeriesDataset(X_test, y_test, self.seq_length)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size)
        
        # Evaluation
        self.model.eval()
        y_true_list = []
        y_pred_list = []
        
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(self.device)
                
                # Forward pass
                y_pred = self.model(X_batch)
                
                # Store predictions and targets
                y_true_list.append(y_batch.cpu().numpy())
                y_pred_list.append(y_pred.cpu().numpy())
        
        # Concatenate batches
        y_true = np.concatenate(y_true_list)
        y_pred = np.concatenate(y_pred_list)
        
        # Reshape to 2D
        y_true = y_true.reshape(-1, self.output_dim)
        y_pred = y_pred.reshape(-1, self.output_dim)
        
        # Inverse transform
        y_true = self.target_scaler.inverse_transform(y_true)
        y_pred = self.target_scaler.inverse_transform(y_pred)
        
        # Calculate metrics
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # Calculate MAPE
        mask = y_true != 0
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        
        metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'mape': mape
        }
        
        return metrics
    
    def forecast(self, X, steps=24):
        """
        Generate forecast
        
        Args:
            X (numpy.ndarray): Input features
            steps (int): Number of steps to forecast
            
        Returns:
            numpy.ndarray: Forecasted values
        """
        # Ensure model is in evaluation mode
        self.model.eval()
        
        # Scale input
        X_scaled = self.feature_scaler.transform(X)
        
        # Convert to tensor
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # Initialize forecast
        forecast = []
        
        # Generate forecast
        with torch.no_grad():
            for _ in range(steps):
                # Get latest sequence
                if len(forecast) == 0:
                    # First step, use input sequence
                    input_seq = X_tensor[:, -self.seq_length:, :]
                else:
                    # Use previous forecasts
                    input_seq = torch.cat([
                        X_tensor[:, -(self.seq_length-len(forecast)):, :],
                        torch.tensor(forecast, dtype=torch.float32).unsqueeze(0).to(self.device)
                    ], dim=1)
                
                # Generate prediction
                pred = self.model(input_seq)
                
                # Add prediction to forecast
                forecast.append(pred[:, -1, :].cpu().numpy())
        
        # Concatenate forecasts
        forecast = np.concatenate(forecast)
        
        # Inverse transform
        forecast = self.target_scaler.inverse_transform(forecast)
        
        return forecast
    
    def plot_forecast(self, X, y, steps=24, save_path=None):
        """
        Plot forecast against actual values
        
        Args:
            X (numpy.ndarray): Input features
            y (numpy.ndarray): Actual values
            steps (int): Number of steps to forecast
            save_path (str): Path to save the plot
            
        Returns:
            tuple: (figure, axes)
        """
        # Generate forecast
        forecast = self.forecast(X, steps)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot actual values
        ax.plot(y[:steps], label='Actual', marker='o')
        
        # Plot forecast
        ax.plot(forecast, label='Forecast', marker='x')
        
        # Add labels and title
        ax.set_xlabel('Time Step')
        ax.set_ylabel(self.resource_name)
        ax.set_title(f'{self.resource_name} Forecast')
        ax.legend()
        ax.grid(True)
        
        # Save figure
        if save_path:
            plt.savefig(save_path, dpi=300)
        
        return fig, ax
    
    def save(self, path):
        """
        Save the forecaster
        
        Args:
            path (str): Path to save the forecaster
            
        Returns:
            str: Path to saved forecaster
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save model
        torch.save(self.model.state_dict(), f"{path}_model.pth")
        
        # Save scalers
        with open(f"{path}_scalers.pkl", 'wb') as f:
            pickle.dump({
                'feature_scaler': self.feature_scaler,
                'target_scaler': self.target_scaler
            }, f)
        
        # Save metadata
        metadata = {
            'resource_name': self.resource_name,
            'model_type': self.model_type,
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'seq_length': self.seq_length
        }
        
        with open(f"{path}_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Forecaster saved to {path}")
        return path
    
    @classmethod
    def load(cls, path, device='cpu'):
        """
        Load a forecaster
        
        Args:
            path (str): Path to load the forecaster from
            device (str): Device to run the model on ('cpu' or 'cuda')
            
        Returns:
            ResourceForecaster: Loaded forecaster
        """
        # Load metadata
        with open(f"{path}_metadata.json", 'r') as f:
            metadata = json.load(f)
        
        # Create forecaster
        forecaster = cls(
            resource_name=metadata['resource_name'],
            model_type=metadata['model_type'],
            device=device
        )
        
        # Set parameters
        forecaster.input_dim = metadata['input_dim']
        forecaster.output_dim = metadata['output_dim']
        forecaster.seq_length = metadata['seq_length']
        
        # Load scalers
        with open(f"{path}_scalers.pkl", 'rb') as f:
            scalers = pickle.load(f)
            forecaster.feature_scaler = scalers['feature_scaler']
            forecaster.target_scaler = scalers['target_scaler']
        
        # Create and load model
        forecaster.model = forecaster.create_model()
        forecaster.model.load_state_dict(torch.load(f"{path}_model.pth", map_location=device))
        forecaster.model.eval()
        
        print(f"Forecaster loaded from {path}")
        return forecaster

class AnomalyDetector:
    """
    Anomaly detector for Mars habitat resource management
    """
    
    def __init__(self, resource_name, method='isolation_forest', device='cpu'):
        """
        Initialize the anomaly detector
        
        Args:
            resource_name (str): Name of the resource to detect anomalies for
            method (str): Method to use for anomaly detection
            device (str): Device to run the model on ('cpu' or 'cuda')
        """
        self.resource_name = resource_name
        self.method = method
        self.device = device
        
        # Model
        self.model = None
        
        # Data preprocessing
        self.scaler = None
        
        # Threshold for anomaly detection
        self.threshold = None
        
        print(f"Anomaly Detector initialized for {resource_name} using {method} method")
    
    def train(self, data):
        """
        Train the anomaly detector
        
        Args:
            data (pandas.DataFrame): Data to train on
            
        Returns:
            AnomalyDetector: Trained anomaly detector
        """
        # Extract resource data
        resource_data = data[[self.resource_name]].values
        
        # Scale data
        self.scaler = StandardScaler()
        resource_data_scaled = self.scaler.fit_transform(resource_data)
        
        # Train model based on method
        if self.method == 'isolation_forest':
            from sklearn.ensemble import IsolationForest
            
            self.model = IsolationForest(
                n_estimators=100,
                contamination=0.05,
                random_state=RANDOM_SEED
            )
            
            self.model.fit(resource_data_scaled)
            
            # Compute anomaly scores
            scores = -self.model.score_samples(resource_data_scaled)
            
            # Set threshold (95th percentile)
            self.threshold = np.percentile(scores, 95)
        
        elif self.method == 'autoencoder':
            # Create and train autoencoder
            self.model = Autoencoder(
                input_dim=1,
                hidden_dims=[16, 8, 16],
                dropout=0.2
            ).to(self.device)
            
            # Create dataset and data loader
            tensor_data = torch.tensor(resource_data_scaled, dtype=torch.float32).to(self.device)
            dataset = torch.utils.data.TensorDataset(tensor_data, tensor_data)
            dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
            
            # Train autoencoder
            criterion = nn.MSELoss()
            optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
            
            self.model.train()
            for epoch in tqdm(range(100), desc=f"Training {self.resource_name} anomaly detector"):
                for X, _ in dataloader:
                    # Forward pass
                    X_recon = self.model(X)
                    loss = criterion(X_recon, X)
                    
                    # Backward pass and optimize
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            
            # Compute reconstruction errors
            self.model.eval()
            with torch.no_grad():
                recon_errors = []
                for X, _ in dataloader:
                    X_recon = self.model(X)
                    error = torch.mean((X - X_recon) ** 2, dim=1)
                    recon_errors.append(error)
                
                recon_errors = torch.cat(recon_errors).cpu().numpy()
            
            # Set threshold (95th percentile)
            self.threshold = np.percentile(recon_errors, 95)
        
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        return self
    
    def detect(self, data):
        """
        Detect anomalies in data
        
        Args:
            data (pandas.DataFrame): Data to detect anomalies in
            
        Returns:
            numpy.ndarray: Anomaly scores
        """
        # Extract resource data
        resource_data = data[[self.resource_name]].values
        
        # Scale data
        resource_data_scaled = self.scaler.transform(resource_data)
        
        # Detect anomalies based on method
        if self.method == 'isolation_forest':
            # Compute anomaly scores
            scores = -self.model.score_samples(resource_data_scaled)
            
            # Detect anomalies
            anomalies = scores > self.threshold
        
        elif self.method == 'autoencoder':
            # Convert to tensor
            tensor_data = torch.tensor(resource_data_scaled, dtype=torch.float32).to(self.device)
            
            # Compute reconstruction errors
            self.model.eval()
            with torch.no_grad():
                recon = self.model(tensor_data)
                errors = torch.mean((tensor_data - recon) ** 2, dim=1).cpu().numpy()
            
            # Detect anomalies
            anomalies = errors > self.threshold
            scores = errors
        
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        return anomalies, scores
    
    def plot_anomalies(self, data, save_path=None):
        """
        Plot anomalies in data
        
        Args:
            data (pandas.DataFrame): Data to plot anomalies for
            save_path (str): Path to save the plot
            
        Returns:
            tuple: (figure, axes)
        """
        # Detect anomalies
        anomalies, scores = self.detect(data)
        
        # Extract resource data
        resource_data = data[[self.resource_name]].values
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        # Plot resource data
        ax1.plot(resource_data, label=self.resource_name)
        ax1.scatter(np.where(anomalies)[0], resource_data[anomalies], color='red', label='Anomalies')
        ax1.set_ylabel(self.resource_name)
        ax1.set_title(f'{self.resource_name} Anomalies')
        ax1.legend()
        ax1.grid(True)
        
        # Plot anomaly scores
        ax2.plot(scores, label='Anomaly Score')
        ax2.axhline(y=self.threshold, color='r', linestyle='--', label='Threshold')
        ax2.set_xlabel('Time Step')
        ax2.set_ylabel('Anomaly Score')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        # Save figure
        if save_path:
            plt.savefig(save_path, dpi=300)
        
        return fig, (ax1, ax2)
    
    def save(self, path):
        """
        Save the anomaly detector
        
        Args:
            path (str): Path to save the anomaly detector
            
        Returns:
            str: Path to saved anomaly detector
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save model
        if self.method == 'isolation_forest':
            with open(f"{path}_model.pkl", 'wb') as f:
                pickle.dump(self.model, f)
        elif self.method == 'autoencoder':
            torch.save(self.model.state_dict(), f"{path}_model.pth")
        
        # Save scaler
        with open(f"{path}_scaler.pkl", 'wb') as f:
            pickle.dump(self.scaler, f)
        
        # Save metadata
        metadata = {
            'resource_name': self.resource_name,
            'method': self.method,
            'threshold': float(self.threshold)
        }
        
        with open(f"{path}_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Anomaly Detector saved to {path}")
        return path
    
    @classmethod
    def load(cls, path, device='cpu'):
        """
        Load an anomaly detector
        
        Args:
            path (str): Path to load the anomaly detector from
            device (str): Device to run the model on ('cpu' or 'cuda')
            
        Returns:
            AnomalyDetector: Loaded anomaly detector
        """
        # Load metadata
        with open(f"{path}_metadata.json", 'r') as f:
            metadata = json.load(f)
        
        # Create anomaly detector
        detector = cls(
            resource_name=metadata['resource_name'],
            method=metadata['method'],
            device=device
        )
        
        # Set threshold
        detector.threshold = metadata['threshold']
        
        # Load scaler
        with open(f"{path}_scaler.pkl", 'rb') as f:
            detector.scaler = pickle.load(f)
        
        # Load model
        if detector.method == 'isolation_forest':
            with open(f"{path}_model.pkl", 'rb') as f:
                detector.model = pickle.load(f)
        elif detector.method == 'autoencoder':
            # Create model
            detector.model = Autoencoder(
                input_dim=1,
                hidden_dims=[16, 8, 16],
                dropout=0.2
            ).to(device)
            
            # Load weights
            detector.model.load_state_dict(torch.load(f"{path}_model.pth", map_location=device))
            detector.model.eval()
        
        print(f"Anomaly Detector loaded from {path}")
        return detector

class Autoencoder(nn.Module):
    """
    Autoencoder for anomaly detection
    """
    
    def __init__(self, input_dim, hidden_dims, dropout=0.2):
        """
        Initialize the autoencoder
        
        Args:
            input_dim (int): Input dimension
            hidden_dims (list): List of hidden dimensions
            dropout (float): Dropout rate
        """
        super(Autoencoder, self).__init__()
        
        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims[:len(hidden_dims)//2 + 1]:
            encoder_layers.append(nn.Linear(prev_dim, hidden_dim))
            encoder_layers.append(nn.ReLU())
            encoder_layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder
        decoder_layers = []
        for hidden_dim in hidden_dims[len(hidden_dims)//2 + 1:]:
            decoder_layers.append(nn.Linear(prev_dim, hidden_dim))
            decoder_layers.append(nn.ReLU())
            decoder_layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        
        self.decoder = nn.Sequential(*decoder_layers)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Reconstructed tensor
        """
        # Encode
        encoded = self.encoder(x)
        
        # Decode
        decoded = self.decoder(encoded)
        
        return decoded

class ResourceOptimizer:
    """
    Resource optimizer for Mars habitat resource management
    """
    
    def __init__(self, resource_names, constraints=None, device='cpu'):
        """
        Initialize the resource optimizer
        
        Args:
            resource_names (list): List of resource names
            constraints (dict): Dictionary of constraints
            device (str): Device to run the model on ('cpu' or 'cuda')
        """
        self.resource_names = resource_names
        self.constraints = constraints or {}
        self.device = device
        
        # Forecasters
        self.forecasters = {}
        
        # Optimization parameters
        self.horizon = 24  # 24-hour horizon
        self.num_scenarios = 10  # Number of scenarios for stochastic optimization
        
        print(f"Resource Optimizer initialized for {len(resource_names)} resources")
    
    def add_forecaster(self, forecaster):
        """
        Add a forecaster
        
        Args:
            forecaster (ResourceForecaster): Forecaster to add
            
        Returns:
            ResourceOptimizer: Self
        """
        self.forecasters[forecaster.resource_name] = forecaster
        return self
    
    def optimize(self, current_state, horizon=None):
        """
        Optimize resource allocation
        
        Args:
            current_state (dict): Current state of the habitat
            horizon (int): Optimization horizon
            
        Returns:
            dict: Optimized resource allocation
        """
        # Set horizon
        horizon = horizon or self.horizon
        
        # Generate forecasts for each resource
        forecasts = {}
        for resource_name, forecaster in self.forecasters.items():
            # Prepare input for forecaster
            X = self._prepare_forecaster_input(current_state)
            
            # Generate forecast
            forecast = forecaster.forecast(X, steps=horizon)
            forecasts[resource_name] = forecast
        
        # Optimize resource allocation based on forecasts
        # This is a simplified optimization that prioritizes critical resources
        # In a real implementation, this would use a more sophisticated optimization algorithm
        
        # Define resource priorities
        priorities = {
            'power': 1.0,
            'oxygen': 0.9,
            'water': 0.8,
            'food': 0.7
        }
        
        # Define minimum required levels
        min_levels = {
            'power': 50.0,  # kWh
            'oxygen': 100.0,  # kg
            'water': 200.0,  # liters
            'food': 50.0  # kg
        }
        
        # Initialize allocation
        allocation = {
            'power_allocation': {
                'life_support': 0.0,
                'isru': 0.0,
                'thermal_control': 0.0
            },
            'isru_mode': 'off',
            'maintenance_target': None
        }
        
        # Check forecasted resource levels
        critical_resources = []
        for resource_name, forecast in forecasts.items():
            if resource_name in min_levels:
                # Check if resource will fall below minimum level
                if np.min(forecast) < min_levels[resource_name]:
                    critical_resources.append((resource_name, priorities.get(resource_name, 0.5)))
        
        # Sort critical resources by priority
        critical_resources.sort(key=lambda x: x[1], reverse=True)
        
        # Allocate resources based on critical needs
        if critical_resources:
            # Prioritize life support for oxygen and water
            if any(resource[0] in ['oxygen', 'water'] for resource in critical_resources):
                allocation['power_allocation']['life_support'] = 5.0
                allocation['isru_mode'] = 'both'
            
            # Allocate remaining power
            remaining_power = 10.0 - allocation['power_allocation']['life_support']
            
            # Allocate to thermal control
            allocation['power_allocation']['thermal_control'] = min(2.0, remaining_power)
            remaining_power -= allocation['power_allocation']['thermal_control']
            
            # Allocate to ISRU
            allocation['power_allocation']['isru'] = remaining_power
        else:
            # No critical resources, balanced allocation
            allocation['power_allocation']['life_support'] = 4.0
            allocation['power_allocation']['isru'] = 3.0
            allocation['power_allocation']['thermal_control'] = 3.0
            allocation['isru_mode'] = 'both'
        
        # Determine maintenance target
        # In a real implementation, this would be based on system health forecasts
        # For simplicity, we'll randomly select a maintenance target
        maintenance_targets = ['power_system', 'life_support', 'isru', 'thermal_control', None]
        allocation['maintenance_target'] = np.random.choice(maintenance_targets)
        
        return allocation
    
    def _prepare_forecaster_input(self, current_state):
        """
        Prepare input for forecaster
        
        Args:
            current_state (dict): Current state of the habitat
            
        Returns:
            numpy.ndarray: Input for forecaster
        """
        # Extract relevant features from current state
        # This is a simplified implementation
        # In a real implementation, this would extract and preprocess all relevant features
        
        # Create feature vector
        features = []
        
        # Add time features
        features.append(current_state.get('time', {}).get('sol', 0))
        features.append(current_state.get('time', {}).get('hour', 0))
        
        # Add environmental features
        features.append(current_state.get('environment', {}).get('temperature', 0))
        features.append(current_state.get('environment', {}).get('solar_irradiance', 0))
        features.append(current_state.get('environment', {}).get('dust_opacity', 0))
        
        # Add habitat features
        for resource in self.resource_names:
            features.append(current_state.get('habitat', {}).get(resource, 0))
        
        # Add system status features
        for system in ['power_system', 'life_support', 'isru', 'thermal_control']:
            status = current_state.get('subsystems', {}).get(system, {}).get('status', 'operational')
            features.append(1.0 if status == 'operational' else 0.0)
        
        # Convert to numpy array
        X = np.array(features).reshape(1, -1)
        
        return X
    
    def save(self, path):
        """
        Save the resource optimizer
        
        Args:
            path (str): Path to save the resource optimizer
            
        Returns:
            str: Path to saved resource optimizer
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save metadata
        metadata = {
            'resource_names': self.resource_names,
            'constraints': self.constraints,
            'horizon': self.horizon,
            'num_scenarios': self.num_scenarios
        }
        
        with open(f"{path}_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Resource Optimizer saved to {path}")
        return path
    
    @classmethod
    def load(cls, path, forecasters=None, device='cpu'):
        """
        Load a resource optimizer
        
        Args:
            path (str): Path to load the resource optimizer from
            forecasters (dict): Dictionary of forecasters
            device (str): Device to run the model on ('cpu' or 'cuda')
            
        Returns:
            ResourceOptimizer: Loaded resource optimizer
        """
        # Load metadata
        with open(f"{path}_metadata.json", 'r') as f:
            metadata = json.load(f)
        
        # Create optimizer
        optimizer = cls(
            resource_names=metadata['resource_names'],
            constraints=metadata['constraints'],
            device=device
        )
        
        # Set parameters
        optimizer.horizon = metadata['horizon']
        optimizer.num_scenarios = metadata['num_scenarios']
        
        # Add forecasters
        if forecasters:
            for resource_name, forecaster in forecasters.items():
                optimizer.add_forecaster(forecaster)
        
        print(f"Resource Optimizer loaded from {path}")
        return optimizer

class MarsHabitatPredictiveAnalytics:
    """
    Predictive analytics for Mars habitat resource management
    """
    
    def __init__(self, data_dir):
        """
        Initialize the predictive analytics
        
        Args:
            data_dir (str): Directory containing data and configuration files
        """
        self.data_dir = data_dir
        self.analytics_dir = os.path.join(data_dir, "analytics")
        os.makedirs(self.analytics_dir, exist_ok=True)
        
        # Create RL formulation
        self.formulation = MarsHabitatRLFormulation(data_dir)
        
        # Device configuration
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Resource names
        self.resource_names = ['power', 'water', 'oxygen', 'food']
        
        # Forecasters
        self.forecasters = {}
        
        # Anomaly detectors
        self.anomaly_detectors = {}
        
        # Resource optimizer
        self.optimizer = None
        
        print(f"Mars Habitat Predictive Analytics initialized")
    
    def generate_simulation_data(self, num_episodes=10, max_steps=500):
        """
        Generate simulation data for training
        
        Args:
            num_episodes (int): Number of episodes to simulate
            max_steps (int): Maximum steps per episode
            
        Returns:
            pandas.DataFrame: Simulation data
        """
        print(f"Generating simulation data...")
        
        # Create environment
        env = self.formulation.create_environment(difficulty='normal')
        
        # Initialize data collection
        data = []
        
        # Simulate episodes
        for i_episode in tqdm(range(num_episodes), desc="Episodes"):
            # Reset environment
            state = env.reset()
            
            for t in range(max_steps):
                # Take random action
                action = {
                    'power_allocation': {
                        'life_support': np.random.uniform(0, 10),
                        'isru': np.random.uniform(0, 10),
                        'thermal_control': np.random.uniform(0, 10)
                    },
                    'isru_mode': np.random.choice(['water', 'oxygen', 'both', 'off']),
                    'maintenance_target': np.random.choice(['power_system', 'life_support', 'isru', 'thermal_control', None])
                }
                
                # Take action
                next_state, reward, done, _ = env.step(action)
                
                # Extract features
                features = self._extract_features(state, action)
                
                # Add to data
                data.append(features)
                
                # Update state
                state = next_state
                
                if done:
                    break
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Save data
        df.to_csv(os.path.join(self.analytics_dir, "simulation_data.csv"), index=False)
        
        print(f"Generated {len(df)} data points")
        return df
    
    def _extract_features(self, state, action):
        """
        Extract features from state and action
        
        Args:
            state (dict): State dictionary
            action (dict): Action dictionary
            
        Returns:
            dict: Features
        """
        features = {}
        
        # Add time features
        features['sol'] = state['time'][0]
        features['hour'] = state['time'][1]
        
        # Add environmental features
        features['temperature'] = state['environment'][0]
        features['pressure'] = state['environment'][1]
        features['wind_speed'] = state['environment'][2]
        features['dust_opacity'] = state['environment'][3]
        features['solar_irradiance'] = state['environment'][4]
        
        # Add habitat features
        features['power'] = state['habitat'][0]
        features['water'] = state['habitat'][1]
        features['oxygen'] = state['habitat'][2]
        features['food'] = state['habitat'][3]
        features['spare_parts'] = state['habitat'][4]
        features['internal_temperature'] = state['habitat'][5]
        features['internal_pressure'] = state['habitat'][6]
        features['internal_humidity'] = state['habitat'][7]
        features['co2_level'] = state['habitat'][8]
        
        # Add subsystem features
        features['power_system_status'] = state['subsystems'][0]
        features['life_support_status'] = state['subsystems'][1]
        features['isru_status'] = state['subsystems'][2]
        features['thermal_control_status'] = state['subsystems'][3]
        features['power_system_maintenance'] = state['subsystems'][4]
        features['life_support_maintenance'] = state['subsystems'][5]
        features['isru_maintenance'] = state['subsystems'][6]
        features['thermal_control_maintenance'] = state['subsystems'][7]
        features['battery_charge'] = state['subsystems'][8]
        features['power_generation'] = state['subsystems'][9]
        features['power_consumption'] = state['subsystems'][10]
        features['heating_power'] = state['subsystems'][11]
        
        # Add action features
        features['power_allocation_life_support'] = action['power_allocation']['life_support']
        features['power_allocation_isru'] = action['power_allocation']['isru']
        features['power_allocation_thermal_control'] = action['power_allocation']['thermal_control']
        features['isru_mode'] = action['isru_mode']
        features['maintenance_target'] = action['maintenance_target']
        
        return features
    
    def train_forecasters(self, data=None, model_type='lstm'):
        """
        Train forecasters for each resource
        
        Args:
            data (pandas.DataFrame): Data to train on
            model_type (str): Type of model to use ('lstm' or 'transformer')
            
        Returns:
            dict: Dictionary of trained forecasters
        """
        # Load data if not provided
        if data is None:
            data_path = os.path.join(self.analytics_dir, "simulation_data.csv")
            if os.path.exists(data_path):
                data = pd.read_csv(data_path)
            else:
                data = self.generate_simulation_data()
        
        # Train forecasters for each resource
        for resource_name in self.resource_names:
            print(f"Training forecaster for {resource_name}...")
            
            # Create forecaster
            forecaster = ResourceForecaster(
                resource_name=resource_name,
                model_type=model_type,
                device=self.device
            )
            
            # Preprocess data
            X_train, X_val, X_test, y_train, y_val, y_test = forecaster.preprocess_data(data)
            
            # Train forecaster
            history = forecaster.train(X_train, y_train, X_val, y_val)
            
            # Evaluate forecaster
            metrics = forecaster.evaluate(X_test, y_test)
            print(f"Evaluation metrics for {resource_name}:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value:.4f}")
            
            # Plot forecast
            forecaster.plot_forecast(
                X_test[:24],
                y_test[:24],
                save_path=os.path.join(self.analytics_dir, f"{resource_name}_forecast.png")
            )
            
            # Save forecaster
            forecaster.save(os.path.join(self.analytics_dir, f"{resource_name}_forecaster"))
            
            # Add to forecasters
            self.forecasters[resource_name] = forecaster
        
        return self.forecasters
    
    def train_anomaly_detectors(self, data=None, method='isolation_forest'):
        """
        Train anomaly detectors for each resource
        
        Args:
            data (pandas.DataFrame): Data to train on
            method (str): Method to use for anomaly detection
            
        Returns:
            dict: Dictionary of trained anomaly detectors
        """
        # Load data if not provided
        if data is None:
            data_path = os.path.join(self.analytics_dir, "simulation_data.csv")
            if os.path.exists(data_path):
                data = pd.read_csv(data_path)
            else:
                data = self.generate_simulation_data()
        
        # Train anomaly detectors for each resource
        for resource_name in self.resource_names:
            print(f"Training anomaly detector for {resource_name}...")
            
            # Create anomaly detector
            detector = AnomalyDetector(
                resource_name=resource_name,
                method=method,
                device=self.device
            )
            
            # Train detector
            detector.train(data)
            
            # Plot anomalies
            detector.plot_anomalies(
                data,
                save_path=os.path.join(self.analytics_dir, f"{resource_name}_anomalies.png")
            )
            
            # Save detector
            detector.save(os.path.join(self.analytics_dir, f"{resource_name}_detector"))
            
            # Add to detectors
            self.anomaly_detectors[resource_name] = detector
        
        return self.anomaly_detectors
    
    def create_resource_optimizer(self):
        """
        Create resource optimizer
        
        Returns:
            ResourceOptimizer: Resource optimizer
        """
        print(f"Creating resource optimizer...")
        
        # Create optimizer
        self.optimizer = ResourceOptimizer(
            resource_names=self.resource_names,
            device=self.device
        )
        
        # Add forecasters
        for resource_name, forecaster in self.forecasters.items():
            self.optimizer.add_forecaster(forecaster)
        
        # Save optimizer
        self.optimizer.save(os.path.join(self.analytics_dir, "resource_optimizer"))
        
        return self.optimizer
    
    def load_models(self):
        """
        Load trained models
        
        Returns:
            tuple: (forecasters, anomaly_detectors, optimizer)
        """
        print(f"Loading trained models...")
        
        # Load forecasters
        for resource_name in self.resource_names:
            path = os.path.join(self.analytics_dir, f"{resource_name}_forecaster")
            if os.path.exists(f"{path}_metadata.json"):
                self.forecasters[resource_name] = ResourceForecaster.load(path, self.device)
        
        # Load anomaly detectors
        for resource_name in self.resource_names:
            path = os.path.join(self.analytics_dir, f"{resource_name}_detector")
            if os.path.exists(f"{path}_metadata.json"):
                self.anomaly_detectors[resource_name] = AnomalyDetector.load(path, self.device)
        
        # Load optimizer
        path = os.path.join(self.analytics_dir, "resource_optimizer")
        if os.path.exists(f"{path}_metadata.json"):
            self.optimizer = ResourceOptimizer.load(path, self.forecasters, self.device)
        
        return self.forecasters, self.anomaly_detectors, self.optimizer
    
    def run_analytics_pipeline(self, current_state):
        """
        Run the analytics pipeline
        
        Args:
            current_state (dict): Current state of the habitat
            
        Returns:
            dict: Analytics results
        """
        print(f"Running analytics pipeline...")
        
        # Initialize results
        results = {
            'forecasts': {},
            'anomalies': {},
            'optimization': None
        }
        
        # Generate forecasts
        for resource_name, forecaster in self.forecasters.items():
            # Prepare input
            X = self._prepare_forecaster_input(current_state)
            
            # Generate forecast
            forecast = forecaster.forecast(X)
            results['forecasts'][resource_name] = forecast
        
        # Detect anomalies
        for resource_name, detector in self.anomaly_detectors.items():
            # Prepare input
            data = self._prepare_detector_input(current_state)
            
            # Detect anomalies
            anomalies, scores = detector.detect(data)
            results['anomalies'][resource_name] = {
                'anomalies': anomalies,
                'scores': scores
            }
        
        # Optimize resource allocation
        if self.optimizer:
            allocation = self.optimizer.optimize(current_state)
            results['optimization'] = allocation
        
        return results
    
    def _prepare_forecaster_input(self, current_state):
        """
        Prepare input for forecaster
        
        Args:
            current_state (dict): Current state of the habitat
            
        Returns:
            numpy.ndarray: Input for forecaster
        """
        # This is a simplified implementation
        # In a real implementation, this would extract and preprocess all relevant features
        
        # Create feature vector
        features = []
        
        # Add time features
        features.append(current_state.get('time', {}).get('sol', 0))
        features.append(current_state.get('time', {}).get('hour', 0))
        
        # Add environmental features
        features.append(current_state.get('environment', {}).get('temperature', 0))
        features.append(current_state.get('environment', {}).get('solar_irradiance', 0))
        features.append(current_state.get('environment', {}).get('dust_opacity', 0))
        
        # Add habitat features
        for resource in self.resource_names:
            features.append(current_state.get('habitat', {}).get(resource, 0))
        
        # Add system status features
        for system in ['power_system', 'life_support', 'isru', 'thermal_control']:
            status = current_state.get('subsystems', {}).get(system, {}).get('status', 'operational')
            features.append(1.0 if status == 'operational' else 0.0)
        
        # Convert to numpy array
        X = np.array(features).reshape(1, -1)
        
        return X
    
    def _prepare_detector_input(self, current_state):
        """
        Prepare input for anomaly detector
        
        Args:
            current_state (dict): Current state of the habitat
            
        Returns:
            pandas.DataFrame: Input for anomaly detector
        """
        # This is a simplified implementation
        # In a real implementation, this would extract and preprocess all relevant features
        
        # Create feature dictionary
        features = {}
        
        # Add resource values
        for resource in self.resource_names:
            features[resource] = [current_state.get('habitat', {}).get(resource, 0)]
        
        # Convert to DataFrame
        df = pd.DataFrame(features)
        
        return df
    
    def generate_analytics_report(self, current_state, save_path=None):
        """
        Generate analytics report
        
        Args:
            current_state (dict): Current state of the habitat
            save_path (str): Path to save the report
            
        Returns:
            str: Report content
        """
        print(f"Generating analytics report...")
        
        # Run analytics pipeline
        results = self.run_analytics_pipeline(current_state)
        
        # Generate report
        report = "# Mars Habitat Resource Analytics Report\n\n"
        report += f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        # Add current state summary
        report += "## Current State\n\n"
        report += "### Resources\n\n"
        report += "| Resource | Value | Status |\n"
        report += "|----------|-------|--------|\n"
        
        for resource in self.resource_names:
            value = current_state.get('habitat', {}).get(resource, 0)
            
            # Determine status based on anomaly detection
            status = "Normal"
            if resource in results['anomalies']:
                if results['anomalies'][resource]['anomalies'][0]:
                    status = "Anomaly Detected"
            
            report += f"| {resource.capitalize()} | {value:.2f} | {status} |\n"
        
        # Add forecasts
        report += "\n## Resource Forecasts (24-hour)\n\n"
        
        for resource, forecast in results['forecasts'].items():
            report += f"### {resource.capitalize()}\n\n"
            report += "| Hour | Forecasted Value |\n"
            report += "|------|------------------|\n"
            
            for i, value in enumerate(forecast[:24]):
                report += f"| +{i+1} | {value[0]:.2f} |\n"
            
            report += "\n"
        
        # Add anomaly detection
        report += "## Anomaly Detection\n\n"
        report += "| Resource | Anomaly Detected | Anomaly Score | Threshold |\n"
        report += "|----------|------------------|---------------|----------|\n"
        
        for resource, anomaly_data in results['anomalies'].items():
            anomaly = "Yes" if anomaly_data['anomalies'][0] else "No"
            score = anomaly_data['scores'][0]
            threshold = self.anomaly_detectors[resource].threshold
            
            report += f"| {resource.capitalize()} | {anomaly} | {score:.4f} | {threshold:.4f} |\n"
        
        # Add resource optimization
        if results['optimization']:
            report += "\n## Resource Optimization\n\n"
            report += "### Power Allocation\n\n"
            report += "| Subsystem | Allocation (kW) |\n"
            report += "|-----------|----------------|\n"
            
            for subsystem, allocation in results['optimization']['power_allocation'].items():
                report += f"| {subsystem.replace('_', ' ').capitalize()} | {allocation:.2f} |\n"
            
            report += "\n### ISRU Mode\n\n"
            report += f"Recommended ISRU mode: **{results['optimization']['isru_mode']}**\n\n"
            
            report += "### Maintenance\n\n"
            target = results['optimization']['maintenance_target']
            if target:
                report += f"Recommended maintenance target: **{target.replace('_', ' ')}**\n"
            else:
                report += "No maintenance recommended at this time.\n"
        
        # Save report
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
            print(f"Report saved to {save_path}")
        
        return report
    
    def generate_dashboard(self, current_state, save_path=None):
        """
        Generate dashboard visualization
        
        Args:
            current_state (dict): Current state of the habitat
            save_path (str): Path to save the dashboard
            
        Returns:
            tuple: (figure, axes)
        """
        print(f"Generating dashboard...")
        
        # Run analytics pipeline
        results = self.run_analytics_pipeline(current_state)
        
        # Create figure
        fig = plt.figure(figsize=(15, 10))
        
        # Create grid
        gs = fig.add_gridspec(3, 4)
        
        # Current resources
        ax_current = fig.add_subplot(gs[0, :2])
        resource_values = [current_state.get('habitat', {}).get(resource, 0) for resource in self.resource_names]
        ax_current.bar(self.resource_names, resource_values)
        ax_current.set_title('Current Resources')
        ax_current.set_ylabel('Value')
        ax_current.grid(axis='y')
        
        # Anomaly detection
        ax_anomaly = fig.add_subplot(gs[0, 2:])
        anomaly_scores = [results['anomalies'][resource]['scores'][0] for resource in self.resource_names]
        thresholds = [self.anomaly_detectors[resource].threshold for resource in self.resource_names]
        
        x = np.arange(len(self.resource_names))
        width = 0.35
        
        ax_anomaly.bar(x - width/2, anomaly_scores, width, label='Anomaly Score')
        ax_anomaly.bar(x + width/2, thresholds, width, label='Threshold')
        
        ax_anomaly.set_xticks(x)
        ax_anomaly.set_xticklabels(self.resource_names)
        ax_anomaly.set_title('Anomaly Detection')
        ax_anomaly.set_ylabel('Score')
        ax_anomaly.legend()
        ax_anomaly.grid(axis='y')
        
        # Resource forecasts
        for i, resource in enumerate(self.resource_names):
            ax_forecast = fig.add_subplot(gs[1, i])
            
            forecast = results['forecasts'][resource]
            hours = np.arange(1, len(forecast) + 1)
            
            ax_forecast.plot(hours, forecast, marker='o')
            ax_forecast.set_title(f'{resource.capitalize()} Forecast')
            ax_forecast.set_xlabel('Hour')
            ax_forecast.set_ylabel('Value')
            ax_forecast.grid(True)
        
        # Power allocation
        if results['optimization']:
            ax_power = fig.add_subplot(gs[2, :2])
            
            subsystems = list(results['optimization']['power_allocation'].keys())
            allocations = list(results['optimization']['power_allocation'].values())
            
            ax_power.bar(subsystems, allocations)
            ax_power.set_title('Power Allocation')
            ax_power.set_ylabel('Allocation (kW)')
            ax_power.grid(axis='y')
            
            # ISRU mode and maintenance
            ax_isru = fig.add_subplot(gs[2, 2:])
            
            isru_mode = results['optimization']['isru_mode']
            maintenance_target = results['optimization']['maintenance_target'] or 'None'
            
            ax_isru.text(0.5, 0.7, f"ISRU Mode: {isru_mode}", ha='center', va='center', fontsize=12)
            ax_isru.text(0.5, 0.3, f"Maintenance: {maintenance_target}", ha='center', va='center', fontsize=12)
            ax_isru.set_title('Recommendations')
            ax_isru.axis('off')
        
        plt.tight_layout()
        
        # Save dashboard
        if save_path:
            plt.savefig(save_path, dpi=300)
            print(f"Dashboard saved to {save_path}")
        
        return fig
    
    def save_analytics_config(self):
        """
        Save analytics configuration
        
        Returns:
            str: Path to saved configuration
        """
        config = {
            'resource_names': self.resource_names,
            'device': str(self.device)
        }
        
        config_path = os.path.join(self.analytics_dir, "analytics_config.json")
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"Analytics configuration saved to {config_path}")
        return config_path

# Example usage
if __name__ == "__main__":
    # Create predictive analytics
    analytics = MarsHabitatPredictiveAnalytics("/home/ubuntu/martian_habitat_pathfinder/data")
    
    # Save analytics configuration
    analytics.save_analytics_config()
    
    # Generate simulation data
    data = analytics.generate_simulation_data(num_episodes=5, max_steps=200)
    
    # Train forecasters
    forecasters = analytics.train_forecasters(data, model_type='lstm')
    
    # Train anomaly detectors
    detectors = analytics.train_anomaly_detectors(data, method='isolation_forest')
    
    # Create resource optimizer
    optimizer = analytics.create_resource_optimizer()
    
    # Generate sample current state
    current_state = {
        'time': {'sol': 10, 'hour': 12},
        'environment': {
            'temperature': -60.0,
            'pressure': 600.0,
            'wind_speed': 5.0,
            'dust_opacity': 0.3,
            'solar_irradiance': 500.0
        },
        'habitat': {
            'power': 120.0,
            'water': 800.0,
            'oxygen': 400.0,
            'food': 600.0,
            'spare_parts': 50.0,
            'internal_temperature': 22.0,
            'internal_pressure': 101000.0,
            'internal_humidity': 40.0,
            'co2_level': 0.1
        },
        'subsystems': {
            'power_system': {'status': 'operational', 'maintenance_needed': 0.1},
            'life_support': {'status': 'operational', 'maintenance_needed': 0.2},
            'isru': {'status': 'operational', 'maintenance_needed': 0.3},
            'thermal_control': {'status': 'operational', 'maintenance_needed': 0.1}
        }
    }
    
    # Generate analytics report
    report = analytics.generate_analytics_report(
        current_state,
        save_path="/home/ubuntu/martian_habitat_pathfinder/analytics/analytics_report.md"
    )
    
    # Generate dashboard
    dashboard = analytics.generate_dashboard(
        current_state,
        save_path="/home/ubuntu/martian_habitat_pathfinder/analytics/dashboard.png"
    )
