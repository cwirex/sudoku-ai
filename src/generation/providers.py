import os
import cv2
import numpy as np
from PIL import ImageFont
import random
import urllib.request
import tensorflow as tf

class MNISTDigitProvider:
    """Provides handwritten digits from MNIST dataset."""

    _instance = None

    @classmethod
    def get_instance(cls):
        """Get or create the singleton instance."""
        if cls._instance is None:
            cls._instance = MNISTDigitProvider()
        return cls._instance

    def __init__(self):
        """Initialize and load MNIST dataset."""
        self.mnist_digits = None
        self.load_mnist()

    def load_mnist(self):
        """Load MNIST dataset and organize digits by class."""
        try:
            print("Loading MNIST dataset...")
            (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
            # Create dictionary of digits by class (1-9)
            self.mnist_digits = {
                i: x_train[y_train == i] for i in range(1, 10)
            }
        except Exception as e:
            print(f"Error loading MNIST: {e}")
            # Create dummy data if MNIST fails to load
            self.mnist_digits = {}
            for i in range(1, 10):
                # Create a simple digit pattern
                img = np.zeros((28, 28), dtype=np.uint8)
                if i % 3 == 0:  # Horizontal line for 3, 6, 9
                    img[14:16, 5:23] = 255
                if i % 3 == 1:  # Vertical line for 1, 4, 7
                    img[5:23, 14:16] = 255
                if i % 3 == 2:  # Cross for 2, 5, 8
                    img[14:16, 5:23] = 255
                    img[5:23, 14:16] = 255
                # Store 10 copies of each pattern
                self.mnist_digits[i] = np.stack([img] * 10)

    def get_random_digit(self, digit_value):
        """Get a random MNIST digit image for a specific value."""
        if self.mnist_digits is None:
            self.load_mnist()

        try:
            digit_samples = self.mnist_digits[digit_value]
            idx = random.randint(0, len(digit_samples) - 1)
            return digit_samples[idx]
        except (KeyError, IndexError):
            # Fallback if there's an issue
            digit_img = np.ones((28, 28), dtype=np.uint8) * 255
            cv2.putText(digit_img, str(digit_value), (9, 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, 0, 2)
            return digit_img


class FontProvider:
    """Manages fonts for printed digits using Google Fonts."""

    _instance = None

    @classmethod
    def get_instance(cls):
        """Get or create the singleton instance."""
        if cls._instance is None:
            cls._instance = FontProvider()
        return cls._instance

    def __init__(self):
        """Initialize and setup fonts."""
        self.fonts_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'fonts')
        self.font_files = []
        self.font_names = {}  # Map font file paths to readable names
        self.font_size_multipliers = {}  # Size multipliers for visual consistency
        self.font_vertical_offsets = {}  # Vertical offset for each font
        self.setup_fonts()

    def setup_fonts(self):
        """Download and setup a variety of fonts from Google Fonts."""
        os.makedirs(self.fonts_dir, exist_ok=True)

        # Dictionary of Google Fonts with direct download URLs, display names, and adjustments
        # Format: 'font_id': {'url': url, 'name': readable_name, 'size_multiplier': multiplier, 'vertical_offset': offset}
        # - size_multiplier affects the font size
        # - vertical_offset is pixels to shift upward (positive = higher in cell)
        google_fonts = {
            'open_sans': {
                'url': 'https://fonts.gstatic.com/s/opensans/v35/memSYaGs126MiZpBA-UvWbX2vVnXBbObj2OVZyOOSr4dVJWUgsjZ0B4gaVc.ttf',
                'name': 'Open Sans',
                'size_multiplier': 0.9,
                'vertical_offset': 16
            },
            'lato': {
                'url': 'https://fonts.gstatic.com/s/lato/v24/S6uyw4BMUTPHjx4wWw.ttf',
                'name': 'Lato',
                'size_multiplier': .8,
                'vertical_offset': 10
            },
            'montserrat': {
                'url': 'https://fonts.gstatic.com/s/montserrat/v26/JTUHjIg1_i6t8kCHKm4532VJOt5-QNFgpCtr6Hw5aX8.ttf',
                'name': 'Montserrat',
                'size_multiplier': 1.0,
                'vertical_offset': 14
            },
            'oswald': {
                'url': 'https://fonts.gstatic.com/s/oswald/v53/TK3_WkUHHAIjg75cFRf3bXL8LICs1_FvsUZiYA.ttf',
                'name': 'Oswald',
                'size_multiplier': 0.85,
                'vertical_offset': 17
            }
        }

        arial_url = 'https://github.com/matomo-org/travis-scripts/raw/master/fonts/Arial.ttf'
        arial_info = {
            'url': arial_url,
            'name': 'Arial',
            'size_multiplier': 1.0,
            'vertical_offset': 9  # Standard offset
        }
        google_fonts['arial'] = arial_info

        # Download each font
        successful_downloads = 0
        for font_id, font_info in google_fonts.items():
            font_path = os.path.join(self.fonts_dir, f"{font_id}.ttf")

            # Download if not exists
            if not os.path.exists(font_path):
                try:
                    print(f"Downloading {font_info['name']} font...")
                    urllib.request.urlretrieve(font_info['url'], font_path)
                    print(f"✓ Downloaded {font_info['name']}")
                except Exception as e:
                    print(f"✗ Failed to download {font_info['name']}: {e}")
                    continue

            # Verify the font file works by trying to load it
            try:
                test_font = ImageFont.truetype(font_path, size=50)
                # If we get here, the font works
                self.font_files.append(font_path)
                self.font_names[font_path] = font_info['name']
                self.font_size_multipliers[font_path] = font_info['size_multiplier']
                self.font_vertical_offsets[font_path] = font_info['vertical_offset']
                successful_downloads += 1
                print(f"✓ Verified {font_info['name']} font")
            except Exception as e:
                print(f"✗ Font file for {font_info['name']} is invalid: {e}")
                # Delete the invalid font file so we can try again next time
                if os.path.exists(font_path):
                    try:
                        os.remove(font_path)
                    except:
                        pass

        # If we couldn't download any fonts, add a fallback
        if not self.font_files:
            default_font = 'default'
            self.font_files.append(default_font)
            self.font_names[default_font] = 'Default'
            self.font_size_multipliers[default_font] = 1.0
            self.font_vertical_offsets[default_font] = 15
            print("⚠ Using default font as fallback")
        else:
            print(f"Successfully set up {successful_downloads} fonts")

    def get_random_font(self):
        """Get a random font from available fonts."""
        if not self.font_files:
            self.setup_fonts()
        return random.choice(self.font_files)

    def get_font_name(self, font_path):
        """Get the readable name for a font path."""
        if font_path in self.font_names:
            return self.font_names[font_path]
        return os.path.basename(font_path)

    def get_size_multiplier(self, font_path):
        """Get the size multiplier for a font path."""
        if font_path in self.font_size_multipliers:
            return self.font_size_multipliers[font_path]
        return 1.0  # Default multiplier

    def get_vertical_offset(self, font_path):
        """Get the vertical offset for a font path."""
        if font_path in self.font_vertical_offsets:
            return self.font_vertical_offsets[font_path]
        return 15  # Default offset

    def get_all_fonts(self):
        """Get all available fonts."""
        if not self.font_files:
            self.setup_fonts()
        return self.font_files