import numpy as np
import matplotlib.pyplot as plt

def plot_confusion_matrix(cm, class_names, output_filename="confusion_matrix.png"):
    """Renders and saves a clean Matplotlib heatmap for the Confusion Matrix."""
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        title="Validation Confusion Matrix",
        ylabel="True Label",
        xlabel="Predicted Label"
    )

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, f"{cm[i, j]:,}",
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=11, fontweight="bold"
            )

    plt.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close('all')
    print(f"💾 Saved confusion matrix heatmap to '{output_filename}'")