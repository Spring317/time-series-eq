"""
Generate visualization of STEAD → DAS training pipeline
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

def create_pipeline_diagram():
    """Create visual diagram of the training pipeline"""
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # Title
    ax.text(5, 11.5, 'STEAD → DAS Transfer Learning Pipeline', 
            fontsize=20, fontweight='bold', ha='center')
    
    # Colors
    color_stead = '#3498db'  # Blue
    color_adapt = '#9b59b6'  # Purple
    color_model = '#e74c3c'  # Red
    color_das = '#27ae60'    # Green
    color_results = '#f39c12'  # Orange
    
    # 1. STEAD Dataset
    box1 = FancyBboxPatch((0.5, 9), 3, 1.5, 
                          boxstyle="round,pad=0.1", 
                          edgecolor=color_stead, facecolor=color_stead, alpha=0.3, linewidth=2)
    ax.add_patch(box1)
    ax.text(2, 10.2, 'STEAD Dataset', fontsize=12, fontweight='bold', ha='center')
    ax.text(2, 9.8, '• 1.2M waveforms', fontsize=9, ha='center')
    ax.text(2, 9.5, '• 3 channels (E,N,Z)', fontsize=9, ha='center')
    ax.text(2, 9.2, '• 6000 samples @ 100Hz', fontsize=9, ha='center')
    
    # 2. Channel Adaptation
    box2 = FancyBboxPatch((4.5, 9), 2, 1.5, 
                          boxstyle="round,pad=0.1", 
                          edgecolor=color_adapt, facecolor=color_adapt, alpha=0.3, linewidth=2)
    ax.add_patch(box2)
    ax.text(5.5, 10.2, 'Channel', fontsize=12, fontweight='bold', ha='center')
    ax.text(5.5, 9.9, 'Adaptation', fontsize=12, fontweight='bold', ha='center')
    ax.text(5.5, 9.5, '3 → N channels', fontsize=9, ha='center')
    ax.text(5.5, 9.2, 'Interpolation', fontsize=9, ha='center')
    
    # 3. Training Data
    box3 = FancyBboxPatch((7.5, 9), 2, 1.5, 
                          boxstyle="round,pad=0.1", 
                          edgecolor=color_stead, facecolor=color_stead, alpha=0.3, linewidth=2)
    ax.add_patch(box3)
    ax.text(8.5, 10.2, 'Training Data', fontsize=12, fontweight='bold', ha='center')
    ax.text(8.5, 9.8, '• 70% train', fontsize=9, ha='center')
    ax.text(8.5, 9.5, '• 15% validation', fontsize=9, ha='center')
    ax.text(8.5, 9.2, '• 15% STEAD test', fontsize=9, ha='center')
    
    # 4. LSTM Model
    box4 = FancyBboxPatch((3, 6.5), 4, 1.5, 
                          boxstyle="round,pad=0.1", 
                          edgecolor=color_model, facecolor=color_model, alpha=0.3, linewidth=2)
    ax.add_patch(box4)
    ax.text(5, 7.6, 'LSTM Model', fontsize=14, fontweight='bold', ha='center')
    ax.text(5, 7.2, '• Hidden dim: 64', fontsize=9, ha='center')
    ax.text(5, 6.9, '• 1 bidirectional layer', fontsize=9, ha='center')
    ax.text(5, 6.6, '• <1M parameters', fontsize=9, ha='center')
    
    # 5. Training Process
    box5 = FancyBboxPatch((0.5, 4.5), 4, 1.5, 
                          boxstyle="round,pad=0.1", 
                          edgecolor=color_model, facecolor=color_model, alpha=0.3, linewidth=2)
    ax.add_patch(box5)
    ax.text(2.5, 5.6, 'Training', fontsize=12, fontweight='bold', ha='center')
    ax.text(2.5, 5.2, '• 50 epochs', fontsize=9, ha='center')
    ax.text(2.5, 4.9, '• Early stopping', fontsize=9, ha='center')
    ax.text(2.5, 4.6, '• Class weighting', fontsize=9, ha='center')
    
    # 6. Best Model
    box6 = FancyBboxPatch((5.5, 4.5), 4, 1.5, 
                          boxstyle="round,pad=0.1", 
                          edgecolor=color_model, facecolor=color_model, alpha=0.3, linewidth=2)
    ax.add_patch(box6)
    ax.text(7.5, 5.6, 'Best Model', fontsize=12, fontweight='bold', ha='center')
    ax.text(7.5, 5.2, '✓ Highest val accuracy', fontsize=9, ha='center')
    ax.text(7.5, 4.9, '✓ Saved checkpoint', fontsize=9, ha='center')
    ax.text(7.5, 4.6, '✓ Ready for testing', fontsize=9, ha='center')
    
    # 7. DAS Test Data
    box7 = FancyBboxPatch((1, 2), 3.5, 1.5, 
                          boxstyle="round,pad=0.1", 
                          edgecolor=color_das, facecolor=color_das, alpha=0.3, linewidth=2)
    ax.add_patch(box7)
    ax.text(2.75, 3.1, 'DAS Test Dataset', fontsize=12, fontweight='bold', ha='center')
    ax.text(2.75, 2.7, '• 19 files', fontsize=9, ha='center')
    ax.text(2.75, 2.4, '• ~2000 channels', fontsize=9, ha='center')
    ax.text(2.75, 2.1, '• Real deployment', fontsize=9, ha='center')
    
    # 8. Evaluation
    box8 = FancyBboxPatch((5.5, 2), 4, 1.5, 
                          boxstyle="round,pad=0.1", 
                          edgecolor=color_results, facecolor=color_results, alpha=0.3, linewidth=2)
    ax.add_patch(box8)
    ax.text(7.5, 3.1, 'Results', fontsize=12, fontweight='bold', ha='center')
    ax.text(7.5, 2.7, '• Accuracy, F1-score', fontsize=9, ha='center')
    ax.text(7.5, 2.4, '• Confusion matrix', fontsize=9, ha='center')
    ax.text(7.5, 2.1, '• ROC-AUC', fontsize=9, ha='center')
    
    # 9. Output Files
    box9 = FancyBboxPatch((2, 0.2), 6, 1.2, 
                          boxstyle="round,pad=0.1", 
                          edgecolor=color_results, facecolor=color_results, alpha=0.3, linewidth=2)
    ax.add_patch(box9)
    ax.text(5, 1.1, 'Output Files', fontsize=12, fontweight='bold', ha='center')
    ax.text(5, 0.7, 'best_model_stead.pth  |  das_test_results.json  |  confusion_matrix.png', 
            fontsize=9, ha='center')
    ax.text(5, 0.4, 'TensorBoard logs in logs/', fontsize=9, ha='center', style='italic')
    
    # Arrows
    arrow_props = dict(arrowstyle='->', lw=2, color='black')
    
    # STEAD → Adaptation
    ax.annotate('', xy=(4.5, 9.75), xytext=(3.5, 9.75),
                arrowprops=arrow_props)
    
    # Adaptation → Training Data
    ax.annotate('', xy=(7.5, 9.75), xytext=(6.5, 9.75),
                arrowprops=arrow_props)
    
    # Training Data → Model
    ax.annotate('', xy=(5, 8), xytext=(8.5, 9),
                arrowprops=dict(arrowstyle='->', lw=2, color='black',
                              connectionstyle="arc3,rad=0.3"))
    
    # Model → Training
    ax.annotate('', xy=(2.5, 6), xytext=(3.5, 6.5),
                arrowprops=dict(arrowstyle='->', lw=2, color='black',
                              connectionstyle="arc3,rad=0.3"))
    
    # Training → Best Model
    ax.annotate('', xy=(7.5, 6), xytext=(5.5, 6.5),
                arrowprops=dict(arrowstyle='->', lw=2, color='black',
                              connectionstyle="arc3,rad=-0.3"))
    
    # Best Model → Testing
    ax.annotate('', xy=(5, 3.5), xytext=(7.5, 4.5),
                arrowprops=dict(arrowstyle='->', lw=2, color='black',
                              connectionstyle="arc3,rad=0.3"))
    
    # DAS Data → Testing
    ax.annotate('', xy=(5, 2.75), xytext=(4.5, 2.75),
                arrowprops=arrow_props)
    
    # Testing → Results
    ax.annotate('', xy=(5.5, 2.75), xytext=(5, 2.75),
                arrowprops=arrow_props)
    
    # Results → Output
    ax.annotate('', xy=(5, 1.4), xytext=(7.5, 2),
                arrowprops=dict(arrowstyle='->', lw=2, color='black',
                              connectionstyle="arc3,rad=0.3"))
    
    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=color_stead, alpha=0.3, edgecolor=color_stead, 
                      label='STEAD Data', linewidth=2),
        mpatches.Patch(facecolor=color_adapt, alpha=0.3, edgecolor=color_adapt, 
                      label='Processing', linewidth=2),
        mpatches.Patch(facecolor=color_model, alpha=0.3, edgecolor=color_model, 
                      label='Model/Training', linewidth=2),
        mpatches.Patch(facecolor=color_das, alpha=0.3, edgecolor=color_das, 
                      label='DAS Data', linewidth=2),
        mpatches.Patch(facecolor=color_results, alpha=0.3, edgecolor=color_results, 
                      label='Results', linewidth=2)
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('visualizations/stead_das_pipeline.png', dpi=300, bbox_inches='tight')
    print("✓ Saved pipeline diagram to: visualizations/stead_das_pipeline.png")
    plt.close()
    
    # Create comparison chart
    create_comparison_chart()


def create_comparison_chart():
    """Create comparison chart between STEAD and DAS"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Dataset comparison
    categories = ['Samples', 'Channels', 'Time\nSamples', 'Sampling\nRate (Hz)']
    stead_values = [1200000, 3, 6000, 100]
    das_values = [19, 2000, 2000, 200]
    
    x = np.arange(len(categories))
    width = 0.35
    
    # Normalize for visualization (log scale for samples)
    stead_norm = [np.log10(v) if v > 1000 else v for v in stead_values]
    das_norm = [np.log10(v) if v > 1000 else v for v in das_values]
    
    ax1.bar(x - width/2, stead_norm, width, label='STEAD', color='#3498db', alpha=0.7)
    ax1.bar(x + width/2, das_norm, width, label='DAS', color='#27ae60', alpha=0.7)
    
    ax1.set_ylabel('Value (log scale for large numbers)', fontsize=12)
    ax1.set_title('Dataset Comparison', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories, fontsize=10)
    ax1.legend(fontsize=11)
    ax1.grid(axis='y', alpha=0.3)
    
    # Add actual values as text
    for i, (s, d) in enumerate(zip(stead_values, das_values)):
        ax1.text(i - width/2, stead_norm[i] + 0.1, f'{s:,}' if s < 1000 else f'{s:,}', 
                ha='center', va='bottom', fontsize=8, rotation=0)
        ax1.text(i + width/2, das_norm[i] + 0.1, f'{d:,}' if d < 1000 else f'{d:,}', 
                ha='center', va='bottom', fontsize=8, rotation=0)
    
    # Characteristics comparison
    characteristics = [
        ('Coverage', 'Global', 'Local'),
        ('Instrument', 'Seismometer', 'Fiber DAS'),
        ('Duration', '60s', '10-30s'),
        ('Labels', 'Eq/Noise', 'Eq/Blast'),
        ('Purpose', 'Training', 'Testing')
    ]
    
    ax2.axis('off')
    ax2.set_title('Key Characteristics', fontsize=14, fontweight='bold', pad=20)
    
    # Create table
    table_data = [['Aspect', 'STEAD', 'DAS-BIGORRE']]
    colors = [['#ecf0f1', '#ecf0f1', '#ecf0f1']]
    
    for char, stead, das in characteristics:
        table_data.append([char, stead, das])
        colors.append(['white', '#ebf5fb', '#e8f8f5'])
    
    table = ax2.table(cellText=table_data, cellLoc='center', loc='center',
                     cellColours=colors, bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)
    
    # Bold header
    for i in range(3):
        table[(0, i)].set_text_props(weight='bold')
    
    plt.tight_layout()
    plt.savefig('visualizations/stead_das_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Saved comparison chart to: visualizations/stead_das_comparison.png")
    plt.close()


if __name__ == '__main__':
    import os
    os.makedirs('visualizations', exist_ok=True)
    
    print("\nGenerating STEAD → DAS pipeline visualizations...\n")
    create_pipeline_diagram()
    print("\n✓ All visualizations created!")
    print("\nView the diagrams:")
    print("  - visualizations/stead_das_pipeline.png")
    print("  - visualizations/stead_das_comparison.png")
