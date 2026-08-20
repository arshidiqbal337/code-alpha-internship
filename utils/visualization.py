"""
Visualization utilities for model training curves, prediction results,
and segmented character bounding boxes.
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch


def plot_training_history(train_losses: list, val_losses: list, val_accuracies: list, save_path: str = None):
    """
    Plots training loss, validation loss, and validation accuracy curves.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(train_losses) + 1)

    # Loss plot
    ax1.plot(epochs, train_losses, 'b-o', label='Train Loss')
    ax1.plot(epochs, val_losses, 'r-s', label='Val Loss')
    ax1.set_title('Training & Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend()

    # Accuracy plot
    ax2.plot(epochs, val_accuracies, 'g-^', label='Val Accuracy (%)')
    ax2.set_title('Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend()

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Training history saved to {save_path}")
    else:
        plt.show()


def draw_segmented_boxes(original_image: np.ndarray, segmented_chars: list, output_path: str = None) -> np.ndarray:
    """
    Draws bounding boxes around segmented characters with predicted label text overlays.
    """
    if len(original_image.shape) == 2:
        img_color = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
    else:
        img_color = original_image.copy()

    for item in segmented_chars:
        x, y, w, h = item['box']
        pred_label = item.get('label', '')
        confidence = item.get('confidence', None)

        # Draw green bounding box
        cv2.rectangle(img_color, (x, y), (x + w, y + h), (0, 255, 0), 2)

        if pred_label:
            text = f"{pred_label}"
            if confidence is not None:
                text += f" ({confidence*100:.0f}%)"

            (txt_w, txt_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img_color, (x, y - txt_h - 4), (x + txt_w, y), (0, 255, 0), -1)
            cv2.putText(img_color, text, (x, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, img_color)
        print(f"Annotated segmentation image saved to {output_path}")

    return img_color


def plot_predictions(images: list, true_labels: list, pred_labels: list, confidences: list = None, save_path: str = None):
    """
    Plots a grid of sample character images with ground truth and predicted labels.
    """
    num_samples = len(images)
    cols = min(5, num_samples)
    rows = (num_samples + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5))
    if num_samples == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i in range(num_samples):
        img = images[i]
        if isinstance(img, torch.Tensor):
            img = img.squeeze().numpy()

        axes[i].imshow(img, cmap='gray')
        axes[i].axis('off')

        title = f"Pred: {pred_labels[i]}"
        if true_labels and i < len(true_labels):
            title += f"\nTrue: {true_labels[i]}"
        if confidences and i < len(confidences):
            title += f" ({confidences[i]*100:.1f}%)"

        color = 'green' if true_labels and pred_labels[i] == true_labels[i] else 'red'
        axes[i].set_title(title, fontsize=10, color=color if true_labels else 'black')

    for j in range(num_samples, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Predictions plot saved to {save_path}")
    else:
        plt.show()
