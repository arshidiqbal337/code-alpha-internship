"""
Training and evaluation module for Handwritten Character & Word Recognition models.
Supports training CNN on MNIST/EMNIST datasets and CRNN on word sequences with CTC Loss.
"""

import os
import time
import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim

import config
from utils.dataset import get_mnist_datasets, get_emnist_datasets, get_dataloader, SyntheticWordDataset
from utils.visualization import plot_training_history
from models.cnn_model import CharacterCNN
from models.crnn_model import WordCRNN


def train_cnn_epoch(model, dataloader, criterion, optimizer, device):
    """
    Trains the CNN model for one epoch.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(dataloader, desc="Training Batch", leave=False):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = (correct / total) * 100.0
    return epoch_loss, epoch_acc


def evaluate_cnn(model, dataloader, criterion, device):
    """
    Evaluates the CNN model on validation/test set.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating Batch", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    loss = running_loss / total
    acc = (correct / total) * 100.0
    return loss, acc


def train_cnn_pipeline(dataset_name="emnist", epochs=config.EPOCHS_CNN, batch_size=config.BATCH_SIZE, lr=config.LEARNING_RATE):
    """
    Full training pipeline for character classification CNN model.
    """
    print(f"=== Starting CNN Training Pipeline (Dataset: {dataset_name.upper()}) ===")
    device = config.DEVICE
    print(f"Using compute device: {device}")

    # Load dataset
    if dataset_name.lower() == "mnist":
        train_ds, test_ds = get_mnist_datasets()
        num_classes = 10
    else:
        train_ds, test_ds = get_emnist_datasets(split="balanced")
        num_classes = config.NUM_CLASSES_EMNIST

    train_loader = get_dataloader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = get_dataloader(test_ds, batch_size=batch_size, shuffle=False)

    print(f"Train samples: {len(train_ds)} | Test samples: {len(test_ds)} | Classes: {num_classes}")

    # Initialize model, loss criterion, optimizer & scheduler
    model = CharacterCNN(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    best_acc = 0.0
    train_losses, val_losses, val_accs = [], [], []

    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, f"cnn_{dataset_name.lower()}.pth")

    start_time = time.time()
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_cnn_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate_cnn(model, test_loader, criterion, device)
        scheduler.step(val_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(f"Epoch [{epoch:02d}/{epochs:02d}] | Train Loss: {train_loss:.4f} (Acc: {train_acc:.2f}%) | "
              f"Val Loss: {val_loss:.4f} (Acc: {val_acc:.2f}%)")

        # Save best checkpoint
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'num_classes': num_classes,
                'dataset': dataset_name
            }, checkpoint_path)
            print(f"  -> Saved new best model checkpoint to {checkpoint_path} (Acc: {val_acc:.2f}%)")

    elapsed = time.time() - start_time
    print(f"=== CNN Training Completed in {elapsed/60:.2f} minutes | Best Val Acc: {best_acc:.2f}% ===")

    # Plot training history
    plot_path = os.path.join(config.OUTPUT_DIR, f"training_history_{dataset_name.lower()}.png")
    plot_training_history(train_losses, val_losses, val_accs, save_path=plot_path)

    return model, best_acc


def train_crnn_pipeline(epochs=config.EPOCHS_CRNN, batch_size=32, lr=1e-3, num_samples=5000):
    """
    Training pipeline for Word CRNN model using CTC Loss.
    """
    print(f"=== Starting CRNN Word Recognition Training Pipeline ===")
    device = config.DEVICE
    print(f"Using compute device: {device}")

    # Create synthetic word dataset from EMNIST
    train_emnist, test_emnist = get_emnist_datasets(split="balanced")
    train_word_ds = SyntheticWordDataset(train_emnist, num_samples=num_samples)
    train_word_loader = DataLoader(train_word_ds, batch_size=batch_size, shuffle=True)

    model = WordCRNN(num_classes=config.NUM_CLASSES_EMNIST).to(device)
    ctc_loss_fn = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "crnn_word.pth")

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        total_samples = 0

        for images, targets, target_lengths in tqdm(train_word_loader, desc=f"Epoch {epoch}/{epochs}", leave=False):
            images = images.to(device)  # (B, 1, 32, 128)

            optimizer.zero_grad()
            log_probs = model(images)  # (W_seq, B, num_classes_ctc)

            # Input lengths for CTC: length of sequence output (W_seq for all samples in batch)
            w_seq = log_probs.size(0)
            input_lengths = torch.full((images.size(0),), w_seq, dtype=torch.long, device=device)

            # Targets flattening for PyTorch CTCLoss compatibility
            # Concatenate non-zero character label targets across batch
            targets_flat = targets.to(device)
            target_lengths = target_lengths.to(device)

            loss = ctc_loss_fn(log_probs, targets_flat, input_lengths, target_lengths)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            total_samples += images.size(0)

        epoch_loss = running_loss / total_samples
        print(f"CRNN Epoch [{epoch:02d}/{epochs:02d}] | CTC Loss: {epoch_loss:.4f}")

    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'num_classes': config.NUM_CLASSES_EMNIST
    }, checkpoint_path)
    print(f"Saved trained CRNN model to {checkpoint_path}")

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Handwritten Character / Word Recognition Model")
    parser.add_argument("--model", type=str, choices=["cnn", "crnn"], default="cnn", help="Model type to train (cnn or crnn)")
    parser.add_argument("--dataset", type=str, choices=["emnist", "mnist"], default="emnist", help="Dataset split for CNN")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")

    args = parser.parse_args()

    if args.model == "cnn":
        train_cnn_pipeline(dataset_name=args.dataset, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
    else:
        train_crnn_pipeline(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
