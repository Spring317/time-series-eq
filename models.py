"""
Time-series models for seismic classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalCNN(nn.Module):
    """
    1D CNN for temporal seismic data classification
    Efficient for long sequences with multiple channels
    """
    
    def __init__(
        self,
        num_channels: int,
        num_classes: int = 2,
        hidden_dim: int = 128,
        num_layers: int = 4,
        dropout: float = 0.3
    ):
        super(TemporalCNN, self).__init__()
        
        self.num_channels = num_channels
        self.num_classes = num_classes
        
        # Initial projection to reduce channel dimension
        self.channel_proj = nn.Conv1d(num_channels, hidden_dim, kernel_size=1)
        
        # Temporal convolution blocks
        layers = []
        in_dim = hidden_dim
        for i in range(num_layers):
            out_dim = hidden_dim * (2 ** min(i, 2))  # Gradually increase channels
            layers.append(self._make_conv_block(in_dim, out_dim, dropout))
            in_dim = out_dim
        
        self.conv_blocks = nn.ModuleList(layers)
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def _make_conv_block(self, in_channels, out_channels, dropout):
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=7, padding=3),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_channels, out_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch, channels, time)
        Returns:
            logits: (batch, num_classes)
        """
        # Project channels
        x = self.channel_proj(x)
        
        # Apply conv blocks
        for block in self.conv_blocks:
            x = block(x)
        
        # Global pooling
        x = self.global_pool(x).squeeze(-1)
        
        # Classification
        logits = self.classifier(x)
        
        return logits


class LSTMClassifier(nn.Module):
    """
    LSTM-based classifier for seismic data
    Good for capturing temporal dependencies
    """
    
    def __init__(
        self,
        num_channels: int,
        num_classes: int = 2,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True
    ):
        super(LSTMClassifier, self).__init__()
        
        self.num_channels = num_channels
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # Input projection
        self.input_proj = nn.Linear(num_channels, hidden_dim)
        
        # LSTM layers
        self.lstm = nn.LSTM(
            hidden_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Output dimension after LSTM
        lstm_out_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(lstm_out_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch, channels, time)
        Returns:
            logits: (batch, num_classes)
        """
        # Transpose to (batch, time, channels)
        x = x.transpose(1, 2)
        
        # Project input
        x = self.input_proj(x)
        
        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use the last hidden state
        if self.bidirectional:
            # Concatenate forward and backward hidden states
            h_n = torch.cat([h_n[-2], h_n[-1]], dim=1)
        else:
            h_n = h_n[-1]
        
        # Classification
        logits = self.classifier(h_n)
        
        return logits


class TransformerClassifier(nn.Module):
    """
    Transformer-based classifier for seismic data
    Good for capturing long-range dependencies
    """
    
    def __init__(
        self,
        num_channels: int,
        num_classes: int = 2,
        hidden_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.3,
        max_seq_len: int = 1024
    ):
        super(TransformerClassifier, self).__init__()
        
        self.num_channels = num_channels
        self.hidden_dim = hidden_dim
        
        # Input projection
        self.input_proj = nn.Linear(num_channels, hidden_dim)
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(
            torch.randn(1, max_seq_len, hidden_dim) * 0.02
        )
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Classification token
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch, channels, time)
        Returns:
            logits: (batch, num_classes)
        """
        batch_size = x.size(0)
        
        # Transpose to (batch, time, channels)
        x = x.transpose(1, 2)
        
        # Project input
        x = self.input_proj(x)
        
        # Add positional encoding
        seq_len = x.size(1)
        x = x + self.pos_encoding[:, :seq_len, :]
        
        # Prepend CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Transformer
        x = self.transformer(x)
        
        # Use CLS token for classification
        cls_output = x[:, 0, :]
        
        # Classification
        logits = self.classifier(cls_output)
        
        return logits


def create_model(config: dict, num_channels: int) -> nn.Module:
    """
    Factory function to create model based on config
    """
    model_type = config['model']['type']
    hidden_dim = config['model']['hidden_dim']
    num_layers = config['model']['num_layers']
    dropout = config['model']['dropout']
    
    if model_type == 'temporal_cnn':
        model = TemporalCNN(
            num_channels=num_channels,
            num_classes=2,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout
        )
    elif model_type == 'lstm':
        model = LSTMClassifier(
            num_channels=num_channels,
            num_classes=2,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=True
        )
    elif model_type == 'transformer':
        model = TransformerClassifier(
            num_channels=num_channels,
            num_classes=2,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=8,
            dropout=dropout,
            max_seq_len=config['data']['window_size']
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model
