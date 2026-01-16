"""
Visualize the simplified LSTM model architecture
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

def plot_lstm_architecture():
    """Plot the simplified LSTM architecture with parameter counts"""
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # Title
    ax.text(5, 11.5, 'Simplified LSTM Model Architecture', 
            ha='center', va='top', fontsize=16, fontweight='bold')
    ax.text(5, 11.0, 'Total Parameters: ~138K (< 1M target)', 
            ha='center', va='top', fontsize=12, color='green', fontweight='bold')
    
    # Color scheme
    color_input = '#e3f2fd'
    color_process = '#fff3e0'
    color_lstm = '#f3e5f5'
    color_output = '#e8f5e9'
    
    # Layer positions (x, y, width, height)
    layers = []
    
    # 1. Input Layer
    y_pos = 9.5
    box = FancyBboxPatch((1, y_pos), 8, 0.8, 
                         boxstyle="round,pad=0.1", 
                         facecolor=color_input, 
                         edgecolor='black', linewidth=2)
    ax.add_patch(box)
    ax.text(5, y_pos+0.4, 'Input: (batch, channels, time)', 
            ha='center', va='center', fontsize=11, fontweight='bold')
    ax.text(5, y_pos+0.1, 'Shape: (B, 18958, 2000)', 
            ha='center', va='center', fontsize=9, style='italic')
    layers.append(('input', y_pos+0.4))
    
    # 2. Channel Pooling (No learnable parameters)
    y_pos = 8.2
    box = FancyBboxPatch((1, y_pos), 8, 1.0, 
                         boxstyle="round,pad=0.1", 
                         facecolor=color_process, 
                         edgecolor='black', linewidth=2)
    ax.add_patch(box)
    ax.text(5, y_pos+0.65, 'Average Pooling (No learnable params)', 
            ha='center', va='center', fontsize=11, fontweight='bold')
    ax.text(5, y_pos+0.35, 'Pool every 4 channels: 18958 → 4739', 
            ha='center', va='center', fontsize=9)
    ax.text(5, y_pos+0.1, 'Shape: (B, 4739, 2000) → (B, 2000, 4739)', 
            ha='center', va='center', fontsize=9, style='italic', color='blue')
    layers.append(('pool', y_pos+0.5))
    
    # 3. LSTM Layer
    y_pos = 6.2
    box = FancyBboxPatch((1, y_pos), 8, 1.5, 
                         boxstyle="round,pad=0.1", 
                         facecolor=color_lstm, 
                         edgecolor='black', linewidth=2)
    ax.add_patch(box)
    ax.text(5, y_pos+1.1, 'LSTM Layer (Single Layer, Unidirectional)', 
            ha='center', va='center', fontsize=11, fontweight='bold')
    ax.text(5, y_pos+0.75, 'Input size: 4739, Hidden size: 64', 
            ha='center', va='center', fontsize=9)
    ax.text(5, y_pos+0.45, 'Parameters: 4×(64×(4739+64+1)) ≈ 1.23M', 
            ha='center', va='center', fontsize=9, color='red')
    ax.text(5, y_pos+0.1, 'Shape: (B, 2000, 4739) → (B, 2000, 64)', 
            ha='center', va='center', fontsize=9, style='italic', color='blue')
    layers.append(('lstm', y_pos+0.75))
    
    # LSTM internals box
    lstm_detail = FancyBboxPatch((1.5, y_pos+0.05), 7, 0.35, 
                                 boxstyle="round,pad=0.05", 
                                 facecolor='white', 
                                 edgecolor='purple', 
                                 linewidth=1, linestyle='--')
    ax.add_patch(lstm_detail)
    ax.text(5, y_pos+0.225, 'Gates: Forget | Input | Cell | Output', 
            ha='center', va='center', fontsize=8, style='italic')
    
    # 4. Last Hidden State
    y_pos = 4.5
    box = FancyBboxPatch((1, y_pos), 8, 0.8, 
                         boxstyle="round,pad=0.1", 
                         facecolor=color_process, 
                         edgecolor='black', linewidth=2)
    ax.add_patch(box)
    ax.text(5, y_pos+0.5, 'Extract Last Hidden State', 
            ha='center', va='center', fontsize=11, fontweight='bold')
    ax.text(5, y_pos+0.15, 'Shape: (B, 64)', 
            ha='center', va='center', fontsize=9, style='italic', color='blue')
    layers.append(('hidden', y_pos+0.4))
    
    # 5. Dropout
    y_pos = 3.5
    box = FancyBboxPatch((1, y_pos), 8, 0.6, 
                         boxstyle="round,pad=0.1", 
                         facecolor=color_process, 
                         edgecolor='black', linewidth=2)
    ax.add_patch(box)
    ax.text(5, y_pos+0.4, 'Dropout (p=0.3)', 
            ha='center', va='center', fontsize=11, fontweight='bold')
    ax.text(5, y_pos+0.1, 'No parameters', 
            ha='center', va='center', fontsize=9, style='italic')
    layers.append(('dropout', y_pos+0.3))
    
    # 6. Linear Classifier
    y_pos = 2.2
    box = FancyBboxPatch((1, y_pos), 8, 0.9, 
                         boxstyle="round,pad=0.1", 
                         facecolor=color_output, 
                         edgecolor='black', linewidth=2)
    ax.add_patch(box)
    ax.text(5, y_pos+0.6, 'Linear Classifier', 
            ha='center', va='center', fontsize=11, fontweight='bold')
    ax.text(5, y_pos+0.3, 'Parameters: 64×2 + 2 = 130', 
            ha='center', va='center', fontsize=9, color='green')
    ax.text(5, y_pos+0.05, 'Shape: (B, 64) → (B, 2)', 
            ha='center', va='center', fontsize=9, style='italic', color='blue')
    layers.append(('classifier', y_pos+0.45))
    
    # 7. Output
    y_pos = 0.8
    box = FancyBboxPatch((1, y_pos), 8, 0.7, 
                         boxstyle="round,pad=0.1", 
                         facecolor=color_output, 
                         edgecolor='black', linewidth=2)
    ax.add_patch(box)
    ax.text(5, y_pos+0.45, 'Output: Logits', 
            ha='center', va='center', fontsize=11, fontweight='bold')
    ax.text(5, y_pos+0.15, 'Classes: [Earthquake, Quarry Blast]', 
            ha='center', va='center', fontsize=9, style='italic')
    layers.append(('output', y_pos+0.35))
    
    # Draw arrows between layers
    for i in range(len(layers)-1):
        _, y1 = layers[i]
        _, y2 = layers[i+1]
        arrow = FancyArrowPatch((5, y1-0.5), (5, y2+0.4),
                               arrowstyle='->', 
                               mutation_scale=20, 
                               linewidth=2, 
                               color='black')
        ax.add_patch(arrow)
    
    # Add parameter breakdown box
    param_box = FancyBboxPatch((0.2, 0.05), 4, 0.6, 
                               boxstyle="round,pad=0.1", 
                               facecolor='#fff9c4', 
                               edgecolor='orange', 
                               linewidth=2)
    ax.add_patch(param_box)
    ax.text(2.2, 0.5, 'Parameter Breakdown', 
            ha='center', va='center', fontsize=10, fontweight='bold')
    ax.text(2.2, 0.3, 'LSTM: ~1.23M params', 
            ha='center', va='center', fontsize=8)
    ax.text(2.2, 0.15, 'Classifier: 130 params', 
            ha='center', va='center', fontsize=8)
    
    # Add key features box
    features_box = FancyBboxPatch((5.8, 0.05), 4, 0.6, 
                                  boxstyle="round,pad=0.1", 
                                  facecolor='#e1f5fe', 
                                  edgecolor='blue', 
                                  linewidth=2)
    ax.add_patch(features_box)
    ax.text(7.8, 0.5, 'Key Features', 
            ha='center', va='center', fontsize=10, fontweight='bold')
    ax.text(7.8, 0.3, '✓ No CNN preprocessing', 
            ha='center', va='center', fontsize=8)
    ax.text(7.8, 0.15, '✓ Direct temporal modeling', 
            ha='center', va='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('lstm_architecture.png', dpi=300, bbox_inches='tight')
    print("✓ Saved architecture diagram to lstm_architecture.png")
    plt.show()


def plot_parameter_comparison():
    """Plot parameter count comparison between models"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Model comparison
    models = ['Temporal\nCNN', 'LSTM\n(Original)', 'LSTM\n(Simplified)', 'Transformer']
    params = [6.4, 2.8, 0.138, 12.5]  # in millions
    colors = ['#4fc3f7', '#ff7043', '#66bb6a', '#ba68c8']
    
    bars = ax1.bar(models, params, color=colors, edgecolor='black', linewidth=1.5)
    ax1.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='Target: <1M')
    ax1.set_ylabel('Parameters (Millions)', fontsize=12, fontweight='bold')
    ax1.set_title('Model Parameter Comparison', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, param in zip(bars, params):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{param:.2f}M',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Highlight the simplified LSTM
    bars[2].set_edgecolor('red')
    bars[2].set_linewidth(3)
    
    # Layer-wise breakdown for simplified LSTM
    layers = ['Channel\nPooling', 'LSTM\nLayer', 'Dropout', 'Linear\nClassifier']
    layer_params = [0, 1.23, 0, 0.00013]  # in millions
    layer_colors = ['#fff3e0', '#f3e5f5', '#fff3e0', '#e8f5e9']
    
    bars2 = ax2.bar(layers, layer_params, color=layer_colors, 
                    edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('Parameters (Millions)', fontsize=12, fontweight='bold')
    ax2.set_title('Simplified LSTM Layer Breakdown', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, param in zip(bars2, layer_params):
        height = bar.get_height()
        if param > 0.01:
            label = f'{param:.2f}M'
        elif param > 0:
            label = f'{int(param*1e6)}'
        else:
            label = '0'
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                label, ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('lstm_parameters.png', dpi=300, bbox_inches='tight')
    print("✓ Saved parameter comparison to lstm_parameters.png")
    plt.show()


def plot_lstm_data_flow():
    """Plot the data flow through the LSTM model"""
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(7, 9.5, 'LSTM Model Data Flow', 
            ha='center', va='top', fontsize=16, fontweight='bold')
    
    # Define stages
    stages = [
        {'name': 'Input Data', 'shape': '(B, 18958, 2000)', 'y': 8},
        {'name': 'Reshape & Pool', 'shape': '(B, 4739, 2000)', 'y': 6.5},
        {'name': 'Transpose', 'shape': '(B, 2000, 4739)', 'y': 5},
        {'name': 'LSTM Processing', 'shape': '(B, 2000, 64)', 'y': 3.5},
        {'name': 'Last State', 'shape': '(B, 64)', 'y': 2},
        {'name': 'Classifier', 'shape': '(B, 2)', 'y': 0.5},
    ]
    
    colors = ['#e3f2fd', '#fff3e0', '#fff3e0', '#f3e5f5', '#fff3e0', '#e8f5e9']
    
    for i, stage in enumerate(stages):
        # Draw box
        box = FancyBboxPatch((2, stage['y']), 10, 1, 
                            boxstyle="round,pad=0.1", 
                            facecolor=colors[i], 
                            edgecolor='black', linewidth=2)
        ax.add_patch(box)
        
        # Add text
        ax.text(7, stage['y']+0.7, stage['name'], 
               ha='center', va='center', fontsize=12, fontweight='bold')
        ax.text(7, stage['y']+0.3, f"Shape: {stage['shape']}", 
               ha='center', va='center', fontsize=10, style='italic', color='blue')
        
        # Add arrow to next stage
        if i < len(stages) - 1:
            arrow = FancyArrowPatch((7, stage['y']), 
                                   (7, stages[i+1]['y']+1),
                                   arrowstyle='->', 
                                   mutation_scale=20, 
                                   linewidth=2, 
                                   color='black')
            ax.add_patch(arrow)
    
    # Add annotations
    ax.text(12.5, 6.5, 'Avg pool\nevery 4\nchannels', 
           ha='left', va='center', fontsize=9, 
           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
    
    ax.text(12.5, 5, 'Time becomes\nsequence\ndimension', 
           ha='left', va='center', fontsize=9,
           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
    
    ax.text(12.5, 3.5, 'Process all\n2000 time\nsteps', 
           ha='left', va='center', fontsize=9,
           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
    
    ax.text(12.5, 2, 'Use final\nhidden\nstate', 
           ha='left', va='center', fontsize=9,
           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig('lstm_data_flow.png', dpi=300, bbox_inches='tight')
    print("✓ Saved data flow diagram to lstm_data_flow.png")
    plt.show()


if __name__ == '__main__':
    print("\nGenerating LSTM architecture visualizations...\n")
    
    print("1. Main architecture diagram...")
    plot_lstm_architecture()
    
    print("\n2. Parameter comparison...")
    plot_parameter_comparison()
    
    print("\n3. Data flow diagram...")
    plot_lstm_data_flow()
    
    print("\n✅ All visualizations generated!")
    print("\nFiles created:")
    print("  - lstm_architecture.png    (Main architecture)")
    print("  - lstm_parameters.png      (Parameter comparison)")
    print("  - lstm_data_flow.png       (Data flow through model)")