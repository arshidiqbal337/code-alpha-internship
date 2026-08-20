"""
Inference module for Handwritten Character and Word Recognition.
Provides clean functions and CLI to predict single characters, segmented words, and CRNN sequence images.
"""

import os
import argparse
import numpy as np
import torch

import config
from utils.preprocessing import load_and_preprocess_image, prepare_tensor, segment_characters
from utils.visualization import draw_segmented_boxes, plot_predictions
from models.cnn_model import CharacterCNN
from models.crnn_model import WordCRNN


class HandwrittenRecognizer:
    """
    Unified Recognition Engine for single characters and full handwritten words.
    """

    def __init__(self, cnn_checkpoint_path: str = None, crnn_checkpoint_path: str = None):
        self.device = config.DEVICE

        # Load CNN Model if checkpoint exists or initialize
        self.cnn_model = CharacterCNN(num_classes=config.NUM_CLASSES_EMNIST).to(self.device)
        if cnn_checkpoint_path and os.path.exists(cnn_checkpoint_path):
            checkpoint = torch.load(cnn_checkpoint_path, map_location=self.device)
            self.cnn_model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded trained CNN model from {cnn_checkpoint_path}")
        self.cnn_model.eval()

        # Load CRNN Model if checkpoint exists
        self.crnn_model = WordCRNN(num_classes=config.NUM_CLASSES_EMNIST).to(self.device)
        if crnn_checkpoint_path and os.path.exists(crnn_checkpoint_path):
            checkpoint = torch.load(crnn_checkpoint_path, map_location=self.device)
            self.crnn_model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded trained CRNN model from {crnn_checkpoint_path}")
        self.crnn_model.eval()

    def predict_single_character(self, image_path_or_array) -> tuple:
        """
        Recognizes a single isolated character or digit image.
        
        Returns:
            tuple: (predicted_character_string, confidence_probability_float)
        """
        # Preprocess to 28x28 grayscale padded numpy array
        prep_img = load_and_preprocess_image(image_path_or_array, target_size=(28, 28))
        tensor = prepare_tensor(prep_img).to(self.device)

        with torch.no_grad():
            probs = self.cnn_model.predict_proba(tensor)
            conf, pred_class = torch.max(probs, dim=1)

        char_label = config.EMNIST_BALANCED_MAPPING.get(pred_class.item(), '?')
        confidence = conf.item()

        return char_label, confidence

    def predict_word_segmented(self, image_path_or_array, save_annotated_path: str = None) -> tuple:
        """
        Recognizes a word or text line image by segmenting individual character bounding boxes,
        classifying each box with CNN, and assembling the full word string.

        Returns:
            tuple: (recognized_word_string, list_of_segmented_items)
        """
        segmented = segment_characters(image_path_or_array)

        if not segmented:
            return "", []

        word_chars = []
        for item in segmented:
            crop_28x28 = item['crop']
            tensor = prepare_tensor(crop_28x28).to(self.device)

            with torch.no_grad():
                probs = self.cnn_model.predict_proba(tensor)
                conf, pred_class = torch.max(probs, dim=1)

            char_label = config.EMNIST_BALANCED_MAPPING.get(pred_class.item(), '?')
            confidence = conf.item()

            item['label'] = char_label
            item['confidence'] = confidence
            word_chars.append(char_label)

        recognized_word = "".join(word_chars)

        if save_annotated_path:
            import cv2
            if isinstance(image_path_or_array, str):
                orig = cv2.imread(image_path_or_array)
            else:
                orig = image_path_or_array
            draw_segmented_boxes(orig, segmented, output_path=save_annotated_path)

        return recognized_word, segmented

    def predict_word_crnn(self, image_path_or_array) -> str:
        """
        Recognizes a full continuous word image using the CRNN sequence model.
        """
        from utils.preprocessing import load_and_preprocess_image
        import torchvision.transforms as transforms
        from PIL import Image

        prep_img = load_and_preprocess_image(image_path_or_array, target_size=(32, 128))
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1751,), (0.3267,))
        ])
        tensor = transform(Image.fromarray(prep_img)).unsqueeze(0).to(self.device)

        with torch.no_grad():
            log_probs = self.crnn_model(tensor)
            decoded_text = self.crnn_model.decode_greedy(log_probs)[0]

        return decoded_text


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Handwritten Character and Word Recognition Predictor")
    parser.add_argument("--image", type=str, required=True, help="Path to input image file")
    parser.add_argument("--mode", type=str, choices=["single", "word", "crnn"], default="word",
                        help="Mode: 'single' for isolated char, 'word' for segmented word, 'crnn' for sequence model")
    parser.add_argument("--output", type=str, default=os.path.join(config.OUTPUT_DIR, "annotated_prediction.png"),
                        help="Output path for annotated segmentation image")

    args = parser.parse_args()

    cnn_ckpt = os.path.join(config.CHECKPOINT_DIR, "cnn_emnist.pth")
    crnn_ckpt = os.path.join(config.CHECKPOINT_DIR, "crnn_word.pth")

    recognizer = HandwrittenRecognizer(cnn_checkpoint_path=cnn_ckpt, crnn_checkpoint_path=crnn_ckpt)

    print(f"\n--- Processing Image: {args.image} (Mode: {args.mode.upper()}) ---")

    if args.mode == "single":
        label, conf = recognizer.predict_single_character(args.image)
        print(f"Predicted Character: {label} (Confidence: {conf*100:.2f}%)")
    elif args.mode == "word":
        word, segmented = recognizer.predict_word_segmented(args.image, save_annotated_path=args.output)
        print(f"Recognized Word: '{word}'")
        print(f"Detected {len(segmented)} characters. Bounding boxes annotated image saved to: {args.output}")
    elif args.mode == "crnn":
        word = recognizer.predict_word_crnn(args.image)
        print(f"CRNN Sequence Recognized Word: '{word}'")
