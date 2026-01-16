"""
Visualize the simplified LSTM model architecture
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

def plot_lstm_architecture():
    """Plot the simplified LSTM architecture with parameter counts"""
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # Title
    ax.text(5, 11.5, 'Simplified LSTM Model Architecture', 
            ha='center', va='top', fontsize=16, fontweight='bold')
    ax.text(5, 11.0, 'Total Parameters: ~138K', 
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
    
    plt.tight_layout()
    plt.savefig('lstm_architecture.png', dpi=300, bbox_inches='tight')
    print("✓ Saved architecture diagram to lstm_architecture.png")
    plt.show()


if __name__ == '__main__':
    print("\nGenerating LSTM architecture visualization...\n")
    plot_lstm_architecture()
    print("\n✅ Visualization generated!")
    print("\nFile created: lstm_architecture.png")
