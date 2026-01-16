"""
LightEQ-style CNN+LSTM Model for Earthquake Detection (PyTorch)

Based on the LightEQ architecture:
- CNN residual blocks for feature extraction from STFT spectrograms
- LSTM for temporal modeling
- TimeDistributed Dense for output

Reference:
LightEQ: On-Device Earthquake Detection with Embedded Machine Learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class ResidualBlock(nn.Module):
    """
    Residual block with two convolutional layers
    """
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class LightEQCNNLSTM(nn.Module):
    """
    LightEQ-style CNN+LSTM model for earthquake classification
    
    Input: STFT spectrogram (batch, 3, 151, 41)
    Output: Class probabilities (batch, num_classes)
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 2,
        cnn_channels: list = [16, 32, 64, 128],
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        dropout: float = 0.3
    ):
        """
        Args:
            in_channels: Number of input channels (3 for STEAD E/N/Z)
            num_classes: Number of output classes
            cnn_channels: Number of channels in each CNN block
            lstm_hidden: LSTM hidden size
            lstm_layers: Number of LSTM layers
            dropout: Dropout rate
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.num_classes = num_classes
        
        # Initial convolution
        self.conv_initial = nn.Sequential(
            nn.Conv2d(in_channels, cnn_channels[0], kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(cnn_channels[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # Residual blocks
        self.res_blocks = nn.Sequential(
            ResidualBlock(cnn_channels[0], cnn_channels[0]),
            ResidualBlock(cnn_channels[0], cnn_channels[1], stride=2),
            ResidualBlock(cnn_channels[1], cnn_channels[1]),
            ResidualBlock(cnn_channels[1], cnn_channels[2], stride=2),
            ResidualBlock(cnn_channels[2], cnn_channels[2]),
            ResidualBlock(cnn_channels[2], cnn_channels[3], stride=2),
            ResidualBlock(cnn_channels[3], cnn_channels[3])
        )
        
        # Global average pooling along frequency dimension
        self.global_pool = nn.AdaptiveAvgPool2d((None, 1))
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=cnn_channels[3],
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden * 2, 64),  # *2 for bidirectional
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input spectrogram (batch, channels, time, freq) = (B, 3, 151, 41)
        
        Returns:
            Class logits (batch, num_classes)
        """
        # CNN feature extraction
        x = self.conv_initial(x)  # (B, 16, 38, 10)
        x = self.res_blocks(x)     # (B, 128, 2, 1)
        
        # Pool frequency dimension
        x = self.global_pool(x)    # (B, 128, T, 1)
        x = x.squeeze(-1)          # (B, 128, T)
        
        # Prepare for LSTM: (batch, time, features)
        x = x.permute(0, 2, 1)     # (B, T, 128)
        
        # LSTM
        x, _ = self.lstm(x)        # (B, T, 256)
        
        # Take the last time step
        x = x[:, -1, :]            # (B, 256)
        
        # Classification
        x = self.classifier(x)     # (B, num_classes)
        
        return x


class LightEQModel0(nn.Module):
    """
    Simplified LightEQ model (model_lighteq_model0 style)
    
    Lighter architecture for faster inference:
    - Fewer CNN layers
    - Smaller feature maps
    - Single-direction LSTM
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 2,
        dropout: float = 0.2
    ):
        super().__init__()
        
        # CNN backbone
        self.cnn = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout),
            
            # Block 2
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout),
            
            # Block 3
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout),
        )
        
        # LSTM
        self.lstm = nn.LSTM(
            input_size=32 * 5,  # freq_bins after pooling
            hidden_size=64,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 3, 151, 41)
        Returns:
            (batch, num_classes)
        """
        # CNN: (B, 3, 151, 41) -> (B, 32, 18, 5)
        x = self.cnn(x)
        
        # Reshape for LSTM: (B, 32, 18, 5) -> (B, 18, 32*5)
        batch_size = x.size(0)
        x = x.permute(0, 2, 1, 3)  # (B, 18, 32, 5)
        x = x.reshape(batch_size, x.size(1), -1)  # (B, 18, 160)
        
        # LSTM
        x, _ = self.lstm(x)  # (B, 18, 64)
        x = x[:, -1, :]  # Last time step: (B, 64)
        
        # Classify
        x = self.classifier(x)
        
        return x


class LightEQModel1(nn.Module):
    """
    Medium LightEQ model with attention (model_lighteq_model1 style)
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 2,
        dropout: float = 0.25
    ):
        super().__init__()
        
        # CNN with residual connections
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        
        self.pool1 = nn.MaxPool2d(2)
        self.drop1 = nn.Dropout2d(dropout)
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        self.pool2 = nn.MaxPool2d(2)
        self.drop2 = nn.Dropout2d(dropout)
        
        self.conv5 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.pool3 = nn.MaxPool2d(2)
        self.drop3 = nn.Dropout2d(dropout)
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=64 * 5,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 1, bias=False)
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 3, 151, 41)
        """
        # CNN with residual
        x = self.conv1(x)
        x = x + self.conv2(x)  # Residual
        x = self.drop1(self.pool1(x))
        
        x = self.conv3(x)
        x = x + self.conv4(x)  # Residual
        x = self.drop2(self.pool2(x))
        
        x = self.conv5(x)
        x = self.drop3(self.pool3(x))  # (B, 64, 18, 5)
        
        # Reshape for LSTM
        batch_size = x.size(0)
        x = x.permute(0, 2, 1, 3)  # (B, 18, 64, 5)
        x = x.reshape(batch_size, x.size(1), -1)  # (B, 18, 320)
        
        # LSTM
        lstm_out, _ = self.lstm(x)  # (B, 18, 128)
        
        # Attention
        attention_weights = self.attention(lstm_out)  # (B, 18, 1)
        attention_weights = F.softmax(attention_weights, dim=1)
        x = torch.sum(attention_weights * lstm_out, dim=1)  # (B, 128)
        
        # Classify
        x = self.classifier(x)
        
        return x


class LightEQDetector(nn.Module):
    """
    LightEQ-style detection model (outputs detection probability per time step)
    
    This follows the original LightEQ output format: (batch, time_steps, 1)
    For earthquake detection where you want to know WHEN in the signal the earthquake occurs
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        output_length: int = 151,
        dropout: float = 0.2
    ):
        super().__init__()
        
        # CNN encoder
        self.encoder = nn.Sequential(
            # Block 1: (B, 3, 151, 41) -> (B, 16, 75, 20)
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Block 2: -> (B, 32, 37, 10)
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Block 3: -> (B, 64, 18, 5)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=64 * 5,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )
        
        # Upsample to original time resolution
        self.upsample = nn.Linear(128, 128)
        
        # Output head (TimeDistributed Dense in Keras)
        self.output_head = nn.Sequential(
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        self.output_length = output_length
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 3, 151, 41)
        Returns:
            (batch, output_length, 1) - detection probability per time step
        """
        # Encode
        x = self.encoder(x)  # (B, 64, 18, 5)
        
        # Reshape for LSTM
        batch_size = x.size(0)
        x = x.permute(0, 2, 1, 3)  # (B, 18, 64, 5)
        x = x.reshape(batch_size, x.size(1), -1)  # (B, 18, 320)
        
        # LSTM
        x, _ = self.lstm(x)  # (B, 18, 128)
        
        # Upsample to output_length
        x = F.interpolate(x.permute(0, 2, 1), size=self.output_length, mode='linear', align_corners=False)
        x = x.permute(0, 2, 1)  # (B, 151, 128)
        
        # Apply output head to each time step
        x = self.output_head(x)  # (B, 151, 1)
        
        return x


def get_lighteq_model(model_type: str = 'medium', num_classes: int = 2, **kwargs) -> nn.Module:
    """
    Factory function to get LightEQ model
    
    Args:
        model_type: 'light' (Model0), 'medium' (Model1), 'full' (CNNLSTM), 'detector'
        num_classes: Number of classes for classification
        **kwargs: Additional model arguments
    
    Returns:
        LightEQ model
    """
    models = {
        'light': LightEQModel0,
        'medium': LightEQModel1,
        'full': LightEQCNNLSTM,
        'detector': LightEQDetector
    }
    
    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}. Choose from {list(models.keys())}")
    
    model_class = models[model_type]
    
    if model_type == 'detector':
        return model_class(**kwargs)
    else:
        return model_class(num_classes=num_classes, **kwargs)


if __name__ == '__main__':
    # Test models
    print("Testing LightEQ models...")
    
    # Test input shape (batch, channels, time_bins, freq_bins)
    x = torch.randn(4, 3, 151, 41)
    
    print(f"Input shape: {x.shape}")
    
    # Test each model
    for model_type in ['light', 'medium', 'full']:
        model = get_lighteq_model(model_type, num_classes=2)
        out = model(x)
        print(f"{model_type.capitalize()} model output: {out.shape}")
        
        # Count parameters
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Parameters: {params:,}")
    
    # Test detector
    detector = get_lighteq_model('detector', output_length=151)
    out = detector(x)
    print(f"Detector output: {out.shape}")
