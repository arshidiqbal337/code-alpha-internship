"""
Self-contained End-to-End Demonstration Script for Handwritten Character & Word Recognition.
Generates sample handwritten text images, trains/evaluates CNN and CRNN models,
and executes character segmentation and prediction pipelines instantly.
"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import config
from utils.dataset import SyntheticCharacterDataset, get_dataloader
from utils.preprocessing import load_and_preprocess_image, segment_characters, prepare_tensor
from utils.visualization import draw_segmented_boxes, plot_predictions
from models.cnn_model import CharacterCNN
from models.crnn_model import WordCRNN
from predict import HandwrittenRecognizer


def create_synthetic_handwritten_image(text: str, image_path: str) -> np.ndarray:
    """
    Creates a synthetic handwritten-style image containing a multi-character word or string.
    Saves to file and returns the image array.
    """
    # Create black background canvas
    canvas = np.zeros((80, 40 * len(text) + 40), dtype=np.uint8)

    # Draw text using Hershey handwriting fonts with slight variation
    x_offset = 20
    for char in text:
        font = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX
        scale = 1.6
        thickness = 2
        y_jitter = np.random.randint(-3, 4)
        cv2.putText(canvas, char, (x_offset, 55 + y_jitter), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
        x_offset += 38

    # Apply light Gaussian blur to mimic natural ink stroke smoothing
    blurred = cv2.GaussianBlur(canvas, (3, 3), 0)

    os.makedirs(os.path.dirname(image_path), exist_ok=True)
    cv2.imwrite(image_path, blurred)
    print(f"Created sample test image for word '{text}' at: {image_path}")
    return blurred


def run_demo():
    print("=" * 70)
    print("      HANDWRITTEN CHARACTER & WORD RECOGNITION DEMO")
    print("=" * 70)
    print(f"Compute Device: {config.DEVICE}")

    # 1. Fast Training of CNN model on Character dataset (1 fast epoch)
    print("\n[Step 1] Initializing Character Dataset & CharacterCNN model...")
    train_ds = SyntheticCharacterDataset(num_samples_per_class=100)
    test_ds = SyntheticCharacterDataset(num_samples_per_class=20)

    train_loader = get_dataloader(train_ds, batch_size=64, shuffle=True)
    test_loader = get_dataloader(test_ds, batch_size=64, shuffle=False)

    model = CharacterCNN(num_classes=config.NUM_CLASSES_EMNIST).to(config.DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)

    print("Training CNN model for 1 fast demo epoch...")
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(config.DEVICE), labels.to(config.DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # Save model state
    demo_checkpoint = os.path.join(config.CHECKPOINT_DIR, "cnn_emnist.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'num_classes': config.NUM_CLASSES_EMNIST
    }, demo_checkpoint)
    print(f"Model trained and saved to {demo_checkpoint}")

    # 2. Evaluate on sample test batch
    print("\n[Step 2] Evaluating sample character predictions...")
    model.eval()
    sample_images, sample_labels, sample_preds, sample_confs = [], [], [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(config.DEVICE)
            probs = model.predict_proba(images)
            confs, preds = torch.max(probs, dim=1)

            for i in range(min(10, len(images))):
                sample_images.append(images[i].cpu())
                true_char = config.EMNIST_BALANCED_MAPPING[labels[i].item()]
                pred_char = config.EMNIST_BALANCED_MAPPING[preds[i].item()]
                sample_labels.append(true_char)
                sample_preds.append(pred_char)
                sample_confs.append(confs[i].item())
            break

    plot_out = os.path.join(config.OUTPUT_DIR, "sample_character_predictions.png")
    plot_predictions(sample_images, sample_labels, sample_preds, sample_confs, save_path=plot_out)
    print(f"Sample prediction visualization saved to {plot_out}")

    # 3. Test Full Word Recognition & Character Segmentation Pipeline
    print("\n[Step 3] Testing Word Recognition & Contour Segmentation...")
    test_words = ["CODE", "DATA", "2026"]
    recognizer = HandwrittenRecognizer(cnn_checkpoint_path=demo_checkpoint)

    for word in test_words:
        sample_img_path = os.path.join(config.OUTPUT_DIR, f"sample_word_{word}.png")
        annotated_out_path = os.path.join(config.OUTPUT_DIR, f"annotated_word_{word}.png")

        create_synthetic_handwritten_image(word, sample_img_path)
        recognized_text, segmented = recognizer.predict_word_segmented(sample_img_path, save_annotated_path=annotated_out_path)

        print(f"Target Word: '{word}' -> Recognized Word: '{recognized_text}'")
        print(f"Character segmentation boxes saved to {annotated_out_path}")

    # 4. Demonstrate CRNN Sequence Forward Pass
    print("\n[Step 4] Verifying CRNN Sequence Model Architecture...")
    crnn_model = WordCRNN(num_classes=config.NUM_CLASSES_EMNIST).to(config.DEVICE)
    dummy_word_tensor = torch.randn(1, 1, 32, 128).to(config.DEVICE)
    crnn_out = crnn_model(dummy_word_tensor)
    decoded_dummy = crnn_model.decode_greedy(crnn_out)[0]
    print(f"CRNN output shape: {crnn_out.shape} | Initial Greedy Decoding: '{decoded_dummy}'")

    print("\n" + "=" * 70)
    print(" DEMO COMPLETED SUCCESSFULLY! All output artifacts saved to 'outputs/'")
    print("=" * 70)


if __name__ == "__main__":
    run_demo()
