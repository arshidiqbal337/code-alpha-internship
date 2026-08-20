"""
Deep Learning models package containing CNN and CRNN architectures for character and sequence recognition.
"""

from .cnn_model import CharacterCNN
from .crnn_model import WordCRNN

__all__ = ['CharacterCNN', 'WordCRNN']
