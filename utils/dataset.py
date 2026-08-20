"""
Dataset handling module for MNIST, EMNIST, and synthetic word sequence datasets.
Handles data transformations, rotation corrections for EMNIST, SSL context bypass,
and synthetic sequence generation.
"""

import os
import random
import ssl
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

import config

# Bypass SSL certificate verification for dataset downloads (handles expired server certs)
ssl._create_default_https_context = ssl._create_unverified_context


class EMNISTRotateTransform:
    """
    Fixes the EMNIST rotation issue.
    EMNIST images in torchvision are stored rotated by 90 deg and flipped by default.
    """
    def __call__(self, img: Image.Image) -> Image.Image:
        return TF.rotate(TF.hflip(img), -90)


class SyntheticCharacterDataset(Dataset):
    """
    Synthetic fallback character dataset generated on-the-fly if downloading MNIST/EMNIST fails or offline.
    Renders clean 28x28 grayscale character images for all 47 EMNIST balanced classes.
    """
    def __init__(self, num_samples_per_class: int = 100):
        self.samples = []
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1751,), (0.3267,))
        ])

        for class_idx in range(config.NUM_CLASSES_EMNIST):
            char_str = config.EMNIST_BALANCED_MAPPING[class_idx]
            for _ in range(num_samples_per_class):
                # Create 28x28 black canvas
                img = Image.new('L', (28, 28), color=0)
                draw = ImageDraw.Draw(img)

                # Draw character centered with slight random offsets & sizes
                font_size = random.randint(18, 22)
                try:
                    font = ImageFont.truetype("arial.ttf", font_size)
                except IOError:
                    font = ImageFont.load_default()

                x_off = random.randint(4, 8)
                y_off = random.randint(2, 6)
                draw.text((x_off, y_off), char_str, fill=255, font=font)

                self.samples.append((img, class_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img, label = self.samples[idx]
        return self.transform(img), label


def get_mnist_datasets(data_dir: str = config.DATA_DIR):
    """
    Returns train and test DataLoaders for the standard MNIST (digits) dataset.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    try:
        train_dataset = torchvision.datasets.MNIST(
            root=data_dir, train=True, download=True, transform=transform
        )
        test_dataset = torchvision.datasets.MNIST(
            root=data_dir, train=False, download=True, transform=transform
        )
    except Exception as e:
        print(f"Warning: Could not download MNIST ({e}). Falling back to synthetic character dataset.")
        train_dataset = SyntheticCharacterDataset(num_samples_per_class=100)
        test_dataset = SyntheticCharacterDataset(num_samples_per_class=20)

    return train_dataset, test_dataset


def get_emnist_datasets(data_dir: str = config.DATA_DIR, split: str = "balanced"):
    """
    Returns train and test datasets for EMNIST (e.g. 'balanced', 'letters', 'digits').
    Fixes the orientation rotation of raw EMNIST data.
    """
    transform = transforms.Compose([
        EMNISTRotateTransform(),
        transforms.ToTensor(),
        transforms.Normalize((0.1751,), (0.3267,))
    ])

    try:
        train_dataset = torchvision.datasets.EMNIST(
            root=data_dir, split=split, train=True, download=True, transform=transform
        )
        test_dataset = torchvision.datasets.EMNIST(
            root=data_dir, split=split, train=False, download=True, transform=transform
        )
    except Exception as e:
        print(f"Warning: Could not download EMNIST ({e}). Falling back to synthetic character dataset.")
        train_dataset = SyntheticCharacterDataset(num_samples_per_class=100)
        test_dataset = SyntheticCharacterDataset(num_samples_per_class=20)

    return train_dataset, test_dataset


def get_dataloader(dataset: Dataset, batch_size: int = config.BATCH_SIZE, shuffle: bool = True, num_workers: int = 0):
    """
    Wraps a PyTorch Dataset into a DataLoader.
    """
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )


class SyntheticWordDataset(Dataset):
    """
    Generates synthetic multi-character word images dynamically from an isolated character dataset (e.g. EMNIST)
    for sequence training with CRNN (CNN + BiLSTM + CTC Loss).
    """

    def __init__(self, character_dataset: Dataset, word_list: list = None, num_samples: int = 10000, img_height: int = 32, max_width: int = 128):
        super().__init__()
        self.char_dataset = character_dataset
        self.num_samples = num_samples
        self.img_height = img_height
        self.max_width = max_width

        # Group character dataset indices by target class label
        self.label_to_indices = {}
        for idx in range(len(character_dataset)):
            _, label = character_dataset[idx]
            if isinstance(label, torch.Tensor):
                label = label.item()
            if label not in self.label_to_indices:
                self.label_to_indices[label] = []
            self.label_to_indices[label].append(idx)

        # Default word vocabulary if none provided
        if word_list is None:
            self.word_list = [
                "CAT", "DOG", "FOX", "BOY", "MAN", "PEN", "CAR", "BUS", "SKY", "SUN",
                "BOOK", "CODE", "DATA", "TEST", "AI", "ML", "CNN", "RNN", "TEXT", "READ",
                "123", "456", "789", "2026", "PYTORCH", "DEEP", "LEARN", "IMAGE", "MODEL"
            ]
        else:
            self.word_list = word_list

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int):
        word = random.choice(self.word_list)

        char_imgs = []
        target_labels = []

        for char in word:
            class_idx = config.CHAR_TO_INDEX.get(char, None)
            if class_idx is None or class_idx not in self.label_to_indices:
                class_idx = random.choice(list(self.label_to_indices.keys()))

            rand_sample_idx = random.choice(self.label_to_indices[class_idx])
            img_tensor, _ = self.char_dataset[rand_sample_idx]  # Shape: (1, 28, 28)

            img_np = img_tensor.squeeze(0).numpy()
            char_imgs.append(img_np)
            target_labels.append(class_idx)

        word_img_np = np.hstack(char_imgs)  # Shape: (28, 28 * len(word))

        pil_img = Image.fromarray((word_img_np * 255).astype(np.uint8))
        orig_w, orig_h = pil_img.size
        new_w = int(orig_w * (self.img_height / float(orig_h)))
        pil_img = pil_img.resize((new_w, self.img_height), Image.Resampling.BILINEAR)

        canvas = Image.new("L", (self.max_width, self.img_height), color=0)
        canvas.paste(pil_img, (0, 0))

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1751,), (0.3267,))
        ])
        canvas_tensor = transform(canvas)

        target_tensor = torch.tensor(target_labels, dtype=torch.long)
        target_length = torch.tensor(len(target_labels), dtype=torch.long)

        return canvas_tensor, target_tensor, target_length
