"""
Sudoku puzzle generator and dataset creation.
"""

import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import random
import urllib.request
import json
import uuid
import zipfile
from datetime import datetime
from tqdm import tqdm  # Using standard tqdm, not notebook version
from dataclasses import dataclass
import tensorflow as tf


class BaseConfig:
    """Base configuration class that other configs inherit from."""
    def __init__(self, **kwargs):
        # Set values from kwargs
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                print(f"Warning: Unknown parameter '{key}'")


class DigitConfig(BaseConfig):
    """Configuration for digit rendering."""
    def __init__(self, **kwargs):
        # Default values for digit rendering
        self.cell_size = 80
        self.printed_font_size_range = (45, 60)
        self.handwritten_size_range = (55, 68)
        # Added a separate parameter for vertical alignment adjustment
        self.vertical_alignment_offset = 5  # Positive value shifts up, negative shifts down
        self.text_position_jitter = (-2, 2)
        self.printed_colors = [
            (13, 13, 13),         # Black (most common)
            (13, 13, 13),         # (do not use 0,0,0 because of the render issues!)
            (33, 33, 33),      # Dark gray
            (0, 0, 60),        # Dark blue
        ]
        self.handwritten_colors = [
            (13, 13, 13),         # Pure black
            (13, 13, 13),         # Black (duplicate to increase probability)
            (25, 25, 25),      # Almost black
            (50, 50, 50),      # Dark grey
            (80, 80, 80),      # Medium grey
            (0, 0, 150),       # Blue
            (0, 0, 110),       # Deep blue
            (20, 20, 80),      # Dark blue-grey
            (150, 0, 0),       # Red (less common)
            (0, 100, 0),       # Green (less common)
            (75, 0, 130),       # Indigo/purple (less common)
        ]
        super().__init__(**kwargs)


class GridConfig(BaseConfig):
    """Configuration for grid line rendering."""
    def __init__(self, **kwargs):
        # Default values for grid lines
        self.thin_line_widths = [1, 1, 2]
        self.thin_line_colors = [
            (160, 160, 160),  # Light gray
            (120, 120, 120),  # Medium gray
            (180, 180, 180),  # Very light gray
            (100, 100, 100),   # Darker gray
            (40, 40, 90),    # Dark blue
        ]
        self.thick_line_widths = [2, 2, 3, 4]
        self.thick_line_colors = [
            (10, 10, 10),       # Black
            (30, 30, 30),    # Near black
            (60, 60, 60),    # Dark gray
            (10, 10, 50),    # Dark blue
        ]
        self.same_color_thin_thick_probability = 0.2
        super().__init__(**kwargs)


class BackgroundConfig(BaseConfig):
    """Configuration for background generation."""
    def __init__(self, **kwargs):
        # Default values for background
        self.background_options = [
            # (color, name, texture_intensity)
            (np.array([252, 248, 240], dtype=np.uint8), "cream", 0.08),
            (np.array([240, 248, 255], dtype=np.uint8), "blue", 0.07),
            (np.array([255, 250, 205], dtype=np.uint8), "yellow", 0.06),
            (np.array([240, 255, 240], dtype=np.uint8), "green", 0.05),
            (np.array([245, 245, 245], dtype=np.uint8), "gray", 0.04)
        ]
        self.background_noise_level = 2.0
        self.vignette_strength = 0.3
        self.gradient_intensity = 0.1
        self.texture_detail_level = 50
        self.texture_contrast = 10
        super().__init__(**kwargs)


class AugmentationConfig(BaseConfig):
    """Configuration for image augmentations."""
    def __init__(self, **kwargs):
        # Difficulty levels for augmentations
        self.difficulties = {
            'easy': {
                'rotation_range': (0, 0),  # No rotation
                'scale_range': (1.0, 1.0),  # No scaling
                'perspective_strength': 0,  # No perspective
                'noise_level': 0.001,  # Minimal noise
                'blur_levels': [0],  # No blur
                'brightness_range': (-1, 1),  # Minimal brightness change
                'contrast_range': (0.99, 1.01)  # Almost no contrast change
            },
            'medium': {
                'rotation_range': (-2, 2),
                'scale_range': (0.98, 1.02),
                'perspective_strength': 0.007,
                'noise_level': 0.008,
                'blur_levels': [0, 1],
                'brightness_range': (-3, 3),
                'contrast_range': (0.97, 1.03)
            },
            'hard': {
                'rotation_range': (-9, 9),  # More rotation
                'scale_range': (0.94, 1.06),  # More scaling
                'perspective_strength': 0.04,  # Significantly more perspective distortion
                'noise_level': 0.016,  # More noise
                'blur_levels': [0, 1, 2],  # More blur options including stronger blur
                'brightness_range': (-6, 6),  # More brightness variation
                'contrast_range': (0.92, 1.1)  # More contrast variation
            }
        }
        super().__init__(**kwargs)

    def get_params(self, difficulty):
        """Get parameters for the specified difficulty level."""
        if difficulty in self.difficulties:
            return self.difficulties[difficulty]
        return self.difficulties['easy']  # Default to easy


class SudokuGeneratorConfig(BaseConfig):
    """Main configuration for Sudoku image generation."""
    def __init__(self, **kwargs):
        # Main configuration values
        self.cell_size = 80
        self.grid_padding = 40  # Padding around the grid
        self.background_padding = 120  # Extra padding for perspective

        # Puzzle generation
        self.mask_rates = [0.1, 0.3, 0.5, 0.7]  # Empty cell ratios

        # Mixed mode settings
        self.mixed_mode_handwritten_ratio = (0.5, 0.8)  # Min/max percentage of cells to be handwritten
        self.handwritten_num_colors_weights = [0.5, 0.3, 0.2]  # Weights for 1, 2, or 3 colors

        # Initialize sub-configurations
        self.digit_config = DigitConfig()
        self.grid_config = GridConfig()
        self.background_config = BackgroundConfig()
        self.augmentation_config = AugmentationConfig()

        # Process kwargs and override defaults
        super().__init__(**kwargs)

        # Update sub-configurations if provided in kwargs
        if 'digit_config' in kwargs and isinstance(kwargs['digit_config'], dict):
            self.digit_config = DigitConfig(**kwargs['digit_config'])
        if 'grid_config' in kwargs and isinstance(kwargs['grid_config'], dict):
            self.grid_config = GridConfig(**kwargs['grid_config'])
        if 'background_config' in kwargs and isinstance(kwargs['background_config'], dict):
            self.background_config = BackgroundConfig(**kwargs['background_config'])
        if 'augmentation_config' in kwargs and isinstance(kwargs['augmentation_config'], dict):
            self.augmentation_config = AugmentationConfig(**kwargs['augmentation_config'])


class SudokuPuzzleGenerator:
    """Generates valid Sudoku puzzles and solutions."""

    def __init__(self, config=None):
        """Initialize with configuration."""
        self.config = config or SudokuGeneratorConfig()

    def generate(self, mask_rate=None):
        """
        Generate a Sudoku puzzle and its solution.

        Args:
            mask_rate: Percentage of cells to leave empty (0.0 to 1.0)
                      If None, a random value will be chosen from config

        Returns:
            grid: 9x9 numpy array with the puzzle (0 for empty cells)
            solution: 9x9 numpy array with the full solution
        """
        if mask_rate is None:
            mask_rate = random.choice(self.config.mask_rates)

        # Create a base pattern (this is a valid Sudoku solution)
        base = [
            [5, 3, 4, 6, 7, 8, 9, 1, 2],
            [6, 7, 2, 1, 9, 5, 3, 4, 8],
            [1, 9, 8, 3, 4, 2, 5, 6, 7],
            [8, 5, 9, 7, 6, 1, 4, 2, 3],
            [4, 2, 6, 8, 5, 3, 7, 9, 1],
            [7, 1, 3, 9, 2, 4, 8, 5, 6],
            [9, 6, 1, 5, 3, 7, 2, 8, 4],
            [2, 8, 7, 4, 1, 9, 6, 3, 5],
            [3, 4, 5, 2, 8, 6, 1, 7, 9]
        ]

        # Randomize the solution by shuffling rows and columns within blocks
        solution = np.array(base, dtype=int)

        # Shuffle rows within each block
        for block in range(3):
            rows = range(block * 3, (block + 1) * 3)
            perm = list(rows)
            random.shuffle(perm)
            solution[[rows[0], rows[1], rows[2]]] = solution[[perm[0], perm[1], perm[2]]]

        # Shuffle columns within each block
        for block in range(3):
            cols = range(block * 3, (block + 1) * 3)
            perm = list(cols)
            random.shuffle(perm)
            solution[:, [cols[0], cols[1], cols[2]]] = solution[:, [perm[0], perm[1], perm[2]]]

        # Create a masked version as the puzzle grid
        grid = solution.copy()
        mask = np.random.random((9, 9)) < mask_rate
        grid[mask] = 0

        return grid, solution


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


class DigitRenderer:
    """Renders individual digits in cells."""

    def __init__(self, config=None):
        """Initialize with configuration."""
        self.config = config or DigitConfig()
        self.mnist_provider = MNISTDigitProvider.get_instance()
        self.font_provider = FontProvider.get_instance()

    def render_cell(self, digit, cell_size=None, mode="printed",
           font_path=None, font_size=None, text_color=None):
        """
        Draw a single Sudoku cell with the given digit.

        Args:
            digit: The digit to draw (0 for empty)
            cell_size: Size of the cell in pixels (uses config if None)
            mode: "printed" or "handwritten"
            font_path: Path to the font file (random if None)
            font_size: Fixed font size (random if None)
            text_color: Color to use for the digit (random if None)

        Returns:
            PIL Image object of the cell
        """
        # Use config values if parameters are None
        cell_size = cell_size or self.config.cell_size

        # Create a white cell
        img = Image.new('RGB', (cell_size, cell_size), color='white')
        draw = ImageDraw.Draw(img)

        # Draw a light border
        border_color = (220, 220, 220)
        draw.rectangle([(0, 0), (cell_size-1, cell_size-1)], outline=border_color)

        # Skip empty cells
        if digit == 0:
            return img

        if mode == "printed":
            # Get font provider
            font_provider = FontProvider.get_instance()

            # Much larger base font size - Sudoku digits should be prominent
            base_font_size = font_size if font_size else random.randint(65, 75)

            # Apply font-specific size adjustment
            size_multiplier = font_provider.get_size_multiplier(font_path)
            adjusted_size = int(base_font_size * size_multiplier)

            # Get font-specific vertical offset - restore to previous values if needed
            # Default is 5 (original value) plus any font-specific adjustment
            base_offset = 5  # This was the original default value that worked well
            font_specific_adjustment = 0  # Additional font-specific adjustment

            if font_path in font_provider.font_vertical_offsets:
                font_specific_adjustment = font_provider.font_vertical_offsets.get(font_path, 0)

            vertical_offset = base_offset + font_specific_adjustment

            # Load font
            try:
                if font_path == 'default' or not os.path.exists(font_path):
                    font = ImageFont.load_default()
                    # Default font is small, we need to compensate
                    scale_factor = 3.0
                    adjusted_size = int(adjusted_size * scale_factor)
                else:
                    font = ImageFont.truetype(font_path, size=adjusted_size)
            except Exception as e:
                print(f"Error loading font, using fallback: {e}")
                font = ImageFont.load_default()
                # Default font is small, we need to compensate
                scale_factor = 3.0
                adjusted_size = int(adjusted_size * scale_factor)

            # Convert digit to string
            text = str(digit)

            # Get text dimensions for centering
            try:
                # Newer PIL versions
                left, top, right, bottom = font.getbbox(text)
                text_width = right - left
                text_height = bottom - top
            except AttributeError:
                # Older PIL versions
                text_width, text_height = draw.textsize(text, font=font)

            # Calculate position (centered with small random offset)
            jitter_range = self.config.text_position_jitter
            x = (cell_size - text_width) / 2 + random.randint(*jitter_range)

            # RESTORE ORIGINAL APPROACH: Center vertically but apply offset
            y = (cell_size - text_height) / 2 + random.randint(*jitter_range) - vertical_offset

            # Use provided color or random from config
            color = text_color if text_color else random.choice(self.config.printed_colors)

            # Draw the text
            draw.text((x, y), text, font=font, fill=color)

        elif mode == "handwritten":
            # Get a random MNIST digit
            mnist_digit = self.mnist_provider.get_random_digit(digit)

            # Also increase handwritten digit size for consistency with printed digits
            digit_size = random.randint(60, 70)

            # Calculate safe padding ranges
            padding_x = random.randint(5, cell_size - digit_size - 5)

            # FIX THE ERROR-PRONE PART: Ensure minimum value is always less than maximum
            min_padding_y = 5  # Minimum padding from top
            max_possible = cell_size - digit_size - 5  # Maximum possible padding
            max_padding_y = min(max_possible // 2, 20)  # Use at most 20px or half available space

            # Ensure max is not less than min
            if max_padding_y < min_padding_y:
                max_padding_y = min_padding_y

            # Now this will never throw an error
            padding_y = random.randint(min_padding_y, max_padding_y)

            # Convert MNIST digit to PIL image and resize
            mnist_pil = Image.fromarray(mnist_digit).convert('L')
            mnist_pil = mnist_pil.resize((digit_size, digit_size))

            # Invert colors (MNIST is white on black, we want black on white)
            mnist_pil = Image.eval(mnist_pil, lambda x: 255 - x)

            # Use provided color or default to random from config
            color = text_color if text_color else random.choice(self.config.handwritten_colors)

            # Create an RGB image with white background
            digit_rgb = Image.new("RGB", mnist_pil.size, (255, 255, 255))

            # For each pixel, if it's dark in the original (a digit pixel), color it
            digit_array = np.array(mnist_pil)
            rgb_array = np.array(digit_rgb)

            # Create mask where digit pixels are (darker pixels)
            mask = digit_array < 128

            # Apply the color only to digit pixels, keep background white
            for c in range(3):
                channel = rgb_array[:,:,c]
                channel[mask] = color[c]

            # Convert back to PIL image
            colored_digit = Image.fromarray(rgb_array)

            # Paste onto cell with proper alpha blending
            img.paste(colored_digit, (padding_x, padding_y))

        return img


class BackgroundGenerator:
    """Generates backgrounds for Sudoku puzzles."""

    def __init__(self, config=None):
        """Initialize with configuration."""
        self.config = config or BackgroundConfig()

        # Define paper-like colors (RGB values)
        self.paper_colors = [
            # Classic white paper
            (255, 255, 255),
            # Off-white/cream colors
            (252, 250, 242),
            (250, 249, 246),
            (253, 252, 245),
            # Very light beige/cream
            (249, 246, 238),
            (245, 242, 232),
            # Light gray paper
            (245, 245, 245),
            (248, 248, 248),
            # Light blue tint (recycled paper look)
            (248, 250, 252),
            # Very light yellow tint
            (252, 252, 242),
        ]

    def create_paper_color_background(self, width, height):
        """
        Creates a simple paper-colored background with subtle variations.

        Args:
            width: Width of the background
            height: Height of the background

        Returns:
            Background image as numpy array
        """
        # Select a random paper color
        base_color = random.choice(self.paper_colors)

        # Create base background
        background = np.ones((height, width, 3), dtype=np.uint8)
        for i in range(3):
            background[:,:,i] = base_color[i]

        # Add very subtle noise to create a slight paper texture effect
        # The noise is so minimal that it appears as a solid color at normal viewing distances
        noise_level = random.uniform(0.5, 2.0)
        noise = np.random.normal(0, noise_level, background.shape).astype(np.float32)

        # Apply the noise
        background_float = background.astype(np.float32)
        background_float += noise

        # Optional: Add very subtle vignette (darkening at corners)
        if random.random() < 0.3:  # 30% chance of having vignette
            center_x, center_y = width // 2, height // 2
            Y, X = np.mgrid[:height, :width]

            # Create radial gradient (distance from center)
            dist_from_center = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
            max_dist = np.sqrt(center_x**2 + center_y**2)

            # Create very subtle vignette (almost imperceptible)
            vignette_strength = random.uniform(0.02, 0.05)
            vignette = 1.0 - vignette_strength * (dist_from_center / max_dist) ** 2
            vignette = np.clip(vignette, 0, 1)

            # Apply vignette
            for i in range(3):
                background_float[:,:,i] *= vignette

        # Clip and convert back to uint8
        background = np.clip(background_float, 0, 255).astype(np.uint8)

        return background

    def create_background(self, width, height, style="paper_color"):
        """
        Creates a background with the specified style.

        Args:
            width: Width of the background
            height: Height of the background
            style: "texture" for the original enhanced background with texture,
                  "paper_color" for simple paper-colored background

        Returns:
            Background image as numpy array
        """
        if style == "paper_color":
            return self.create_paper_color_background(width, height)
        else:  # "texture" or any other value defaults to original texture
            return self.create_textured_background(width, height)

    def create_textured_background(self, width, height):
        """
        Creates a natural-looking paper background with gradients and texture.
        This is the original background generation method.

        Args:
            width: Width of the background
            height: Height of the background

        Returns:
            Background image as numpy array
        """
        # Select a random background color and texture intensity
        bg_color, bg_name, texture_intensity = random.choice(self.config.background_options)

        # Create base texture with float32 for calculations
        background = np.ones((height, width, 3), dtype=np.float32)

        # Apply a gradient effect from corners to center
        center_x, center_y = width // 2, height // 2
        Y, X = np.mgrid[:height, :width]

        # Create radial gradient (brighter in middle, darker at edges)
        dist_from_center = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
        max_dist = np.sqrt(center_x**2 + center_y**2)
        gradient = 1.0 - self.config.gradient_intensity * (dist_from_center / max_dist)
        gradient = np.clip(gradient, 0, 1)

        # Add slight directional gradient (top-left to bottom-right or other combinations)
        direction = random.choice(['tl_br', 'tr_bl', 'horizontal', 'vertical'])

        if direction == 'tl_br':
            # Top-left to bottom-right
            dir_gradient = (X / width + Y / height) / 2.0
        elif direction == 'tr_bl':
            # Top-right to bottom-left
            dir_gradient = ((width - X) / width + Y / height) / 2.0
        elif direction == 'horizontal':
            # Left to right
            dir_gradient = X / width
        else:  # vertical
            # Top to bottom
            dir_gradient = Y / height

        # Normalize and scale directional gradient
        dir_gradient = dir_gradient * 0.05 + 0.975  # Small effect

        # Combine gradients with base color
        for i in range(3):
            background[:,:,i] = bg_color[i] * gradient * dir_gradient

        # Add random light source (subtle brightening in one area)
        light_x = random.randint(0, width-1)
        light_y = random.randint(0, height-1)
        light_radius = random.randint(width//3, width//2)
        light_intensity = random.uniform(0.03, 0.08)

        # Create light mask
        light_mask = np.zeros((height, width), dtype=np.float32)
        cv2.circle(light_mask, (light_x, light_y), light_radius, 1.0, -1)
        light_mask = cv2.GaussianBlur(light_mask, (light_radius//2*2+1, light_radius//2*2+1), 0)

        # Apply light
        for i in range(3):
            background[:,:,i] += light_intensity * 255 * light_mask

        # Add texture detail with higher frequency noise
        detail_scale = self.config.texture_detail_level
        x_coords = np.linspace(0, detail_scale, width)
        y_coords = np.linspace(0, detail_scale, height)
        xx, yy = np.meshgrid(x_coords, y_coords)

        # Create Perlin-like noise (simplified)
        noise = np.sin(xx) * np.sin(yy) + np.sin(2.5*xx) * np.sin(2.5*yy)
        noise = (noise - noise.min()) / (noise.max() - noise.min())  # Normalize to [0,1]

        # Scale noise to subtle values and add to background
        noise = (noise - 0.5) * self.config.texture_contrast
        for i in range(3):
            background[:,:,i] += noise

        # Add vignetting effect
        vignette = 1.0 - self.config.vignette_strength * (dist_from_center / max_dist) ** 2
        vignette = np.clip(vignette, 0, 1)
        for i in range(3):
            background[:,:,i] *= vignette

        # Add fine grain noise
        grain = np.random.normal(0, self.config.background_noise_level, background.shape)
        background += grain

        # Clip and convert to uint8
        background = np.clip(background, 0, 255).astype(np.uint8)

        return background


class GridRenderer:
    """Renders Sudoku grid lines."""

    def __init__(self, config=None):
        """Initialize with configuration."""
        self.config = config or GridConfig()

    def draw_grid_lines(self, image, grid_top_left, grid_size, cell_size):
        """
        Draws grid lines on the image.

        Args:
            image: Input image as numpy array
            grid_top_left: (x, y) coordinates of the top-left of the grid
            grid_size: Size of the grid in pixels
            cell_size: Size of each cell in pixels

        Returns:
            Image with grid lines drawn
        """
        # Make a copy of the image
        result = image.copy()

        # Define grid line parameters for this board (consistent within the board)
        grid_params = {
            # Thin lines style (random from several options)
            "thin_line_width": random.choice(self.config.thin_line_widths),
            "thin_line_color": random.choice(self.config.thin_line_colors),

            # Thick lines style (random from several options)
            "thick_line_width": random.choice(self.config.thick_line_widths),
            "thick_line_color": random.choice(self.config.thick_line_colors)
        }

        # For some boards, make thin and thick lines the same color
        if random.random() < self.config.same_color_thin_thick_probability:
            grid_params["thin_line_color"] = grid_params["thick_line_color"]

        # Calculate the grid corners
        grid_bottom_right = (grid_top_left[0] + grid_size, grid_top_left[1] + grid_size)

        # Thin lines between cells
        for i in range(1, 9):
            # Skip lines that will be drawn thicker (box borders)
            if i % 3 != 0:
                pos_x = grid_top_left[0] + i * cell_size
                pos_y = grid_top_left[1] + i * cell_size
                # Horizontal line
                cv2.line(
                    result,
                    (grid_top_left[0], pos_y),
                    (grid_bottom_right[0], pos_y),
                    grid_params["thin_line_color"],
                    grid_params["thin_line_width"]
                )
                # Vertical line
                cv2.line(
                    result,
                    (pos_x, grid_top_left[1]),
                    (pos_x, grid_bottom_right[1]),
                    grid_params["thin_line_color"],
                    grid_params["thin_line_width"]
                )

        # Thick lines between 3x3 boxes and outline
        for i in range(0, 10, 3):
            pos_x = grid_top_left[0] + i * cell_size
            pos_y = grid_top_left[1] + i * cell_size
            # Horizontal line
            cv2.line(
                result,
                (grid_top_left[0], pos_y),
                (grid_bottom_right[0], pos_y),
                grid_params["thick_line_color"],
                grid_params["thick_line_width"]
            )
            # Vertical line
            cv2.line(
                result,
                (pos_x, grid_top_left[1]),
                (pos_x, grid_bottom_right[1]),
                grid_params["thick_line_color"],
                grid_params["thick_line_width"]
            )

        return result


class ImageAugmentor:
    """Applies augmentations to images based on difficulty."""

    def __init__(self, config=None):
        """Initialize with configuration."""
        self.config = config or AugmentationConfig()

    def apply_augmentations(self, img, difficulty='easy'):
        """
        Apply augmentations to an image based on difficulty.

        Args:
            img: Input image as numpy array (RGB)
            difficulty: 'easy', 'medium', or 'hard'

        Returns:
            Augmented image
        """
        # Get parameters for the specified difficulty
        params = self.config.get_params(difficulty)

        # Skip all transformations for 'easy' difficulty to ensure clean images
        if difficulty == 'easy':
            # For easy, just apply minimal processing
            result = img.copy()

            # Add very slight noise (almost imperceptible)
            if params['noise_level'] > 0:
                noise = np.random.normal(0, params['noise_level'] * 255, img.shape).astype(np.float32)
                result = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

            return result

        # For medium and hard, we'll do full transformations

        # Make a copy to avoid modifying the original
        original_img = img.copy()
        h, w = original_img.shape[:2]

        # Determine background color from the corner pixels (for proper filling)
        # Take the average of the four corners to get a good estimate
        corners = [
            original_img[0, 0],      # top-left
            original_img[0, w-1],    # top-right
            original_img[h-1, 0],    # bottom-left
            original_img[h-1, w-1]   # bottom-right
        ]
        background_color = np.mean(corners, axis=0).astype(np.uint8)

        # SOLUTION: Add padding to prevent black borders after transformation
        # Calculate padding size based on image dimensions and difficulty
        padding_percent = 0.1 if difficulty == 'medium' else 0.2  # More padding for hard
        padding_x = int(w * padding_percent)
        padding_y = int(h * padding_percent)

        # Create padded image with background color
        padded_img = np.ones((h + 2*padding_y, w + 2*padding_x, 3), dtype=np.uint8) * background_color
        padded_img[padding_y:padding_y+h, padding_x:padding_x+w] = original_img

        # Update dimensions for the padded image
        h_padded, w_padded = padded_img.shape[:2]
        center = (w_padded // 2, h_padded // 2)

        # Apply transformations on padded image
        result = padded_img.copy()

        # Get random rotation angle
        angle = random.uniform(*params['rotation_range'])

        # Get random scale
        scale = random.uniform(*params['scale_range'])

        # Apply affine transformation
        M = cv2.getRotationMatrix2D(center, angle, scale)
        result = cv2.warpAffine(result, M, (w_padded, h_padded),
                               borderMode=cv2.BORDER_CONSTANT,
                               borderValue=background_color.tolist())

        # Perspective transform (if strength > 0)
        if params['perspective_strength'] > 0:
            # Calculate offset based on perspective strength
            offset = int(w_padded * params['perspective_strength'])

            # Enhanced perspective transformation for hard difficulty
            if difficulty == 'hard':
                # Create source points (corners of the image)
                src_points = np.float32([
                    [0, 0],                  # top-left
                    [w_padded-1, 0],         # top-right
                    [w_padded-1, h_padded-1],# bottom-right
                    [0, h_padded-1]          # bottom-left
                ])

                # For hard difficulty, create more extreme distortions
                # Random corner adjustments with more variation
                dst_points = np.float32([
                    [random.randint(0, offset*2), random.randint(0, offset*2)],  # top-left
                    [w_padded-1-random.randint(0, offset*2), random.randint(0, offset)],  # top-right
                    [w_padded-1-random.randint(0, offset), h_padded-1-random.randint(0, offset)],  # bottom-right
                    [random.randint(0, offset), h_padded-1-random.randint(0, offset*2)]  # bottom-left
                ])

                # Add some non-corner distortions for more diverse transformations
                if random.random() < 0.5:  # 50% chance of additional wave distortion
                    # Apply slight wave distortion to the image
                    rows, cols = result.shape[:2]
                    img_output = np.zeros(result.shape, dtype=result.dtype)

                    # Random wave amplitude
                    amplitude = random.uniform(2.0, 5.0)

                    # Random wave frequency
                    frequency = random.uniform(0.1, 0.3)

                    # Apply sine wave distortion
                    for i in range(rows):
                        for j in range(cols):
                            offset_x = int(amplitude * np.sin(2 * np.pi * frequency * i))
                            offset_y = int(amplitude * np.cos(2 * np.pi * frequency * j))

                            if 0 <= j + offset_x < cols and 0 <= i + offset_y < rows:
                                img_output[i, j] = result[i + offset_y, j + offset_x]
                            else:
                                img_output[i, j] = result[i, j]

                    result = img_output
            else:
                # Standard perspective for medium difficulty
                src_points = np.float32([
                    [0, 0],                  # top-left
                    [w_padded-1, 0],         # top-right
                    [w_padded-1, h_padded-1],# bottom-right
                    [0, h_padded-1]          # bottom-left
                ])

                # Add random offsets to the corners
                dst_points = np.float32([
                    [random.randint(0, offset), random.randint(0, offset)],
                    [w_padded-1-random.randint(0, offset), random.randint(0, offset)],
                    [w_padded-1-random.randint(0, offset), h_padded-1-random.randint(0, offset)],
                    [random.randint(0, offset), h_padded-1-random.randint(0, offset)]
                ])

            # Apply perspective transform with proper border filling
            M = cv2.getPerspectiveTransform(src_points, dst_points)
            result = cv2.warpPerspective(result, M, (w_padded, h_padded),
                                       borderMode=cv2.BORDER_CONSTANT,
                                       borderValue=background_color.tolist())

        # Now crop back to the original size from the center of the padded image
        y_start = (h_padded - h) // 2
        y_end = y_start + h
        x_start = (w_padded - w) // 2
        x_end = x_start + w

        result = result[y_start:y_end, x_start:x_end]

        # Continue with other effects

        # Add noise
        result_float = result.astype(np.float32)
        noise = np.random.normal(0, params['noise_level'] * 255, result.shape).astype(np.float32)
        result_float = np.clip(result_float + noise, 0, 255)

        # Apply blur if needed
        blur_level = random.choice(params['blur_levels'])
        if blur_level > 0:
            result_float = cv2.GaussianBlur(result_float, (blur_level*2+1, blur_level*2+1), 0)

        # Apply brightness and contrast adjustments
        brightness = random.randint(*params['brightness_range'])
        contrast = random.uniform(*params['contrast_range'])

        # Apply contrast first, then brightness
        result_float = contrast * result_float + brightness

        # For hard difficulty, add some lighting effects
        if difficulty == 'hard' and random.random() < 0.7:  # 70% chance
            # Create a random light source effect
            light_x = random.randint(0, w-1)
            light_y = random.randint(0, h-1)
            light_radius = random.randint(w//3, w//2)
            light_intensity = random.uniform(0.05, 0.15)  # Stronger lighting effect

            # Create light mask
            light_mask = np.zeros((h, w), dtype=np.float32)
            cv2.circle(light_mask, (light_x, light_y), light_radius, 1.0, -1)
            light_mask = cv2.GaussianBlur(light_mask, (light_radius//2*2+1, light_radius//2*2+1), 0)

            # Apply light to the image
            for i in range(3):
                result_float[:,:,i] += light_intensity * 255 * light_mask

        # Clip and convert back to uint8
        result = np.clip(result_float, 0, 255).astype(np.uint8)

        # Final sanity check to fill any remaining black pixels
        mask = np.all(result < 10, axis=2)  # Find very dark (potentially unfilled) pixels
        if np.any(mask):
            result[mask] = background_color

        return result


class SudokuImageGenerator:
    """Main class for generating Sudoku puzzle images."""

    def __init__(self, config=None):
        """Initialize the generator with configuration."""
        self.config = config or SudokuGeneratorConfig()

        # Create component objects
        self.puzzle_generator = SudokuPuzzleGenerator(self.config)
        self.digit_renderer = DigitRenderer(self.config.digit_config)
        self.background_generator = BackgroundGenerator(self.config.background_config)
        self.grid_renderer = GridRenderer(self.config.grid_config)
        self.image_augmentor = ImageAugmentor(self.config.augmentation_config)

        # Ensure font provider and MNIST are initialized
        self.font_provider = FontProvider.get_instance()
        if "handwritten" in ["handwritten", "mixed"]:
            MNISTDigitProvider.get_instance()

    def generate_image(self, grid=None, mode="printed", difficulty="easy",
                      background_style="paper_color", handwritten_positions=None):
        """
        Generate a complete Sudoku grid image.

        Args:
            grid: 9x9 numpy array with digits (0 for empty cells)
                If None, a new puzzle will be generated
            mode: "printed", "handwritten", or "mixed"
            difficulty: "easy", "medium", or "hard"
            background_style:
                - "none": Just the grid with white background (no texture)
                - "paper_color": Simple paper-colored background (DEFAULT)
                - "texture": Original textured paper background
                - "unified": Single paper texture for entire image
                - "default": Standard white grid on paper background
            handwritten_positions: Set of (i,j) tuples indicating which cells should be handwritten
                                (only used for mixed mode)

        Returns:
            tuple: (numpy array (RGB image), dict of generation info)
        """
        # Generate a puzzle if none provided
        if grid is None:
            grid, _ = self.puzzle_generator.generate()

        cell_size = self.config.cell_size
        grid_size = cell_size * 9

        # Add padding around the grid for a background
        padding = self.config.grid_padding

        # Handle different background styles (code unchanged)
        if background_style == "none":
            extra_padding = 0  # No extra padding needed for simple background
            full_size = grid_size + 2 * padding
            background = np.ones((full_size, full_size, 3), dtype=np.uint8) * 255
        elif background_style == "paper_color":
            extra_padding = 0 if difficulty == "easy" else self.config.background_padding
            full_size = grid_size + 2 * (padding + extra_padding)
            background = self.background_generator.create_paper_color_background(full_size, full_size)
            grid_area = np.ones((grid_size + 2*padding, grid_size + 2*padding, 3), dtype=np.uint8) * 255
            y_start = extra_padding
            y_end = extra_padding + grid_size + 2*padding
            x_start = extra_padding
            x_end = extra_padding + grid_size + 2*padding
            background[y_start:y_end, x_start:x_end] = grid_area
        elif background_style == "unified":
            extra_padding = 0 if difficulty == "easy" else self.config.background_padding
            full_size = grid_size + 2 * (padding + extra_padding)
            background = self.background_generator.create_paper_color_background(full_size, full_size)
        elif background_style == "texture":
            extra_padding = 0 if difficulty == "easy" else self.config.background_padding
            full_size = grid_size + 2 * (padding + extra_padding)
            background = self.background_generator.create_textured_background(full_size, full_size)
            if background_style != "unified":
                grid_area = np.ones((grid_size + 2*padding, grid_size + 2*padding, 3), dtype=np.uint8) * 255
                y_start = extra_padding
                y_end = extra_padding + grid_size + 2*padding
                x_start = extra_padding
                x_end = extra_padding + grid_size + 2*padding
                background[y_start:y_end, x_start:x_end] = grid_area
        else:  # "default" style - just for backward compatibility
            extra_padding = 0 if difficulty == "easy" else self.config.background_padding
            full_size = grid_size + 2 * (padding + extra_padding)
            background = self.background_generator.create_textured_background(full_size, full_size)
            grid_area = np.ones((grid_size + 2*padding, grid_size + 2*padding, 3), dtype=np.uint8) * 255
            y_start = extra_padding
            y_end = extra_padding + grid_size + 2*padding
            x_start = extra_padding
            x_end = extra_padding + grid_size + 2*padding
            background[y_start:y_end, x_start:x_end] = grid_area

        # Handle handwritten positions based on mode
        if handwritten_positions is None:
            print("Warning: handwritten positions not provided. Generating them instead...")
            if mode == "handwritten":
                # All non-empty cells are handwritten
                handwritten_positions = {(i, j) for i in range(9) for j in range(9) if grid[i, j] != 0}
            elif mode == "mixed":
                # Calculate handwritten positions if not provided
                non_empty_positions = [(i, j) for i in range(9) for j in range(9) if grid[i, j] != 0]
                min_ratio, max_ratio = self.config.mixed_mode_handwritten_ratio
                ratio = random.uniform(min_ratio, max_ratio)
                num_handwritten = max(1, int(len(non_empty_positions) * ratio))
                handwritten_positions = set(random.sample(non_empty_positions, num_handwritten))
            else:
                handwritten_positions = set()  # Empty set for printed mode

        # Initialize generation info dictionary
        generation_info = {
            "font_name": None,
            "printed_color": None,
            "handwritten_colors": None,
            "handwritten_positions": list(handwritten_positions) if handwritten_positions else []
        }

        # For printed mode, select ONE font and ONE size for the whole board
        if mode in ["printed", "mixed"]:
            # Select a random font for the whole board
            board_font_path = self.font_provider.get_random_font()
            # Get the font name for the log
            board_font_name = self.font_provider.get_font_name(board_font_path)
            # Select a consistent size - using a larger base size range
            board_font_size = random.randint(65, 75)  # Increased size for better visibility
            # Select one color for all printed digits
            board_printed_color = random.choice(self.config.digit_config.printed_colors)

            # Store font information
            generation_info["font_name"] = board_font_name
            generation_info["printed_color"] = board_printed_color

        # For handwritten mode or mixed mode, select consistent colors for handwritten digits
        if mode in ["handwritten", "mixed"]:
            # Decide how many colors to use (1-3)
            num_colors = random.choices([1, 2, 3],
                                      weights=self.config.handwritten_num_colors_weights)[0]

            # Shuffle and pick the first num_colors
            available_colors = self.config.digit_config.handwritten_colors.copy()
            random.shuffle(available_colors)
            board_handwritten_colors = available_colors[:num_colors]

            # Store handwritten colors information
            generation_info["handwritten_colors"] = board_handwritten_colors

        # Draw each cell
        for i in range(9):
            for j in range(9):
                digit = grid[i, j]

                # Determine cell mode
                cell_mode = mode
                if mode == "mixed":
                    cell_mode = "handwritten" if (i, j) in handwritten_positions else "printed"

                # For printed cells, pass the consistent font and color
                if cell_mode == "printed":
                    cell_img = self.digit_renderer.render_cell(
                        digit, cell_size, "printed",
                        font_path=board_font_path,
                        font_size=board_font_size,
                        text_color=board_printed_color
                    )
                # For handwritten cells, pass the board's color scheme
                else:
                    # Select one of the board's handwritten colors
                    cell_color = random.choice(board_handwritten_colors)
                    cell_img = self.digit_renderer.render_cell(
                        digit, cell_size, "handwritten",
                        text_color=cell_color
                    )

                # Paste onto the background
                # Include both padding and extra_padding
                x = extra_padding + padding + j * cell_size
                y = extra_padding + padding + i * cell_size
                cell_array = np.array(cell_img)
                background[y:y+cell_size, x:x+cell_size] = cell_array

        # Draw grid lines
        grid_top_left = (extra_padding + padding, extra_padding + padding)
        background = self.grid_renderer.draw_grid_lines(
            background, grid_top_left, grid_size, cell_size
        )

        # Apply augmentations based on difficulty
        result = self.image_augmentor.apply_augmentations(background, difficulty)

        return result, generation_info


class SudokuDatasetGenerator:
    """Class for generating Sudoku puzzle datasets."""

    def __init__(self, generator=None, config=None, output_dir=None):
        """
        Initialize the dataset generator.

        Args:
            generator: SudokuImageGenerator instance (creates one if None)
            config: Configuration for the generator (uses default if None)
            output_dir: Directory to save the generated dataset
        """
        self.config = config or SudokuGeneratorConfig()
        self.generator = generator or SudokuImageGenerator(self.config)
        
        if output_dir is None:
            from .utils import setup_project_paths
            paths = setup_project_paths()
            output_dir = os.path.join(paths['data_dir'], 'generated_datasets')
            
        self.output_dir = output_dir

        # Create output directories
        self.images_dir = os.path.join(output_dir, "images")
        self.labels_dir = os.path.join(output_dir, "labels")
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.labels_dir, exist_ok=True)

        # Mapping for mode and difficulty in labels
        self.mode_map = {"printed": "P", "mixed": "M", "handwritten": "H"}
        self.difficulty_map = {"easy": "1", "medium": "2", "hard": "3"}
        self.texture_map = {
            "none": "N",
            "paper_color": "P",
            "texture": "T",
            "unified": "U",
            "default": "D"
        }

    def generate_dataset(self, num_samples,
                 modes=None,
                 difficulties=None,
                 background_styles=None,
                 mask_rates=None,
                 save_images=True,
                 save_labels=True,
                 display_samples=0,
                 seed=None):
        """
        Generate a dataset of Sudoku puzzles.

        Args:
            num_samples: Number of samples to generate
            modes: List of modes to use (if None, uses all available)
            difficulties: List of difficulties to use (if None, uses all available)
            background_styles: List of background styles (if None, uses all available)
            mask_rates: List of mask rates to use (if None, uses default config)
            save_images: Whether to save generated images
            save_labels: Whether to save label information
            display_samples: Number of samples to display (0 to disable)
            seed: Random seed for reproducibility

        Returns:
            List of dictionaries with sample information
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Default options if not specified
        modes = modes or ["printed", "mixed", "handwritten"]
        difficulties = difficulties or ["easy", "medium", "hard"]
        background_styles = background_styles or ["none", "paper_color", "texture", "unified"]
        mask_rates = mask_rates or self.config.mask_rates

        # Generate samples
        samples = []

        print(f"Generating {num_samples} Sudoku puzzle samples...")
        for i in tqdm(range(num_samples)):
            # Generate unique sample ID
            sample_id = str(uuid.uuid4())[:8]

            # Randomly select parameters
            mode = random.choice(modes)
            difficulty = random.choice(difficulties)
            background_style = random.choice(background_styles)
            mask_rate = random.choice(mask_rates)

            # Generate puzzle
            grid, solution = self.generator.puzzle_generator.generate(mask_rate=mask_rate)

            # Calculate handwritten positions before generating the image
            handwritten_positions = set()
            if mode == "mixed":
                # For mixed mode, determine which cells will be handwritten
                non_empty_positions = [(i, j) for i in range(9) for j in range(9) if grid[i, j] != 0]
                min_ratio, max_ratio = self.config.mixed_mode_handwritten_ratio
                ratio = random.uniform(min_ratio, max_ratio)
                num_handwritten = max(1, int(len(non_empty_positions) * ratio))
                handwritten_positions = set(random.sample(non_empty_positions, num_handwritten))
            elif mode == "handwritten":
                # All non-empty cells are handwritten
                handwritten_positions = {(i, j) for i in range(9) for j in range(9) if grid[i, j] != 0}

            # Generate image and pass the handwritten positions
            image, generation_info = self.generator.generate_image(
                grid=grid,
                mode=mode,
                difficulty=difficulty,
                background_style=background_style,
                handwritten_positions=handwritten_positions
            )

            # Create label string
            mode_code = self.mode_map[mode]
            difficulty_code = self.difficulty_map[difficulty]
            texture_code = self.texture_map[background_style]

            # Add font code to the label if available
            font_code = ""
            if generation_info["font_name"] and mode in ["printed", "mixed"]:
                # Create a short font code from the first letter of the font name
                font_code = generation_info["font_name"][0].upper()
                label = f"{mode_code}{difficulty_code}{font_code}_{sample_id}"
            else:
                label = f"{mode_code}{difficulty_code}_{sample_id}"

            # Create handwritten mask (2D array of 0s and 1s)
            handwritten_mask = np.zeros((9, 9), dtype=int)
            for pos in handwritten_positions:
                handwritten_mask[pos[0], pos[1]] = 1

            # Create sample info
            sample_info = {
                "id": sample_id,
                "label": label,
                "mode": mode,
                "difficulty": difficulty,
                "background_style": background_style,
                "mask_rate": mask_rate,
                "grid": grid.tolist(),
                "solution": solution.tolist(),
                "handwritten_mask": handwritten_mask.tolist(),  # 2D array mask of handwritten positions
                "image_path": os.path.join("images", f"{label}.png") if save_images else None,
                "font_name": generation_info["font_name"],
                "printed_color": generation_info["printed_color"],
                "handwritten_colors": generation_info["handwritten_colors"]
            }

            samples.append(sample_info)

            # Save image
            if save_images:
                image_path = os.path.join(self.images_dir, f"{label}.png")
                cv2.imwrite(image_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

            # Save individual label
            if save_labels:
                label_path = os.path.join(self.labels_dir, f"{label}.json")
                with open(label_path, 'w') as f:
                    json.dump(sample_info, f, indent=2)

        # Save complete dataset info
        if save_labels:
            dataset_info = {
                "dataset_size": num_samples,
                "creation_date": datetime.now().isoformat(),
                "config": {
                    "modes": modes,
                    "difficulties": difficulties,
                    "background_styles": background_styles,
                    "mask_rates": mask_rates,
                },
                "samples": samples
            }

            with open(os.path.join(self.output_dir, "dataset_info.json"), 'w') as f:
                json.dump(dataset_info, f, indent=2)

        # Display samples if requested
        if display_samples > 0:
            self.display_samples(samples[:min(display_samples, len(samples))])

        return samples

    def display_samples(self, samples):
        """Display sample images with labels."""
        num_samples = len(samples)
        cols = min(num_samples, 4)
        rows = (num_samples + cols - 1) // cols

        plt.figure(figsize=(15, rows * 4))

        for i, sample in enumerate(samples):
            image_path = os.path.join(self.output_dir, sample["image_path"])
            if os.path.exists(image_path):
                img = cv2.imread(image_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                plt.subplot(rows, cols, i + 1)
                plt.imshow(img)
                plt.title(f"Label: {sample['label']}")
                plt.axis('off')

        plt.tight_layout()
        plt.show()

    def export_as_archive(self, archive_path=None):
        """
        Export the dataset as a zip archive.

        Args:
            archive_path: Path to save the archive (defaults to output_dir.zip)

        Returns:
            Path to the created archive
        """
        if archive_path is None:
            archive_path = f"{self.output_dir}.zip"

        print(f"Creating archive: {archive_path}")
        with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, _, files in os.walk(self.output_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    zipf.write(
                        file_path,
                        os.path.relpath(file_path, os.path.dirname(self.output_dir))
                    )

        print(f"Archive created: {archive_path}")
        return archive_path


def generate_sample_dataset(num_samples=10, output_dir=None, display_samples=5):
    """
    Generate a small sample dataset for testing.
    
    Args:
        num_samples: Number of samples to generate
        output_dir: Directory to save the dataset (uses default if None)
        display_samples: Number of samples to display (0 to disable)
        
    Returns:
        Dataset generator and samples
    """
    # Create dataset generator with custom config options
    config = SudokuGeneratorConfig(
        digit_config={
            "vertical_alignment_offset": 5,  # Apply the centering fix
        }
    )

    generator = SudokuImageGenerator(config)
    dataset_generator = SudokuDatasetGenerator(generator, config, output_dir)

    # Generate a small dataset with diverse parameters
    samples = dataset_generator.generate_dataset(
        num_samples=num_samples,
        modes=["printed", "mixed"],
        difficulties=["easy", "medium", "hard"],
        background_styles=["none", "unified"],
        display_samples=display_samples,
    )

    # Print information about each generated sample
    print("\nGenerated Samples:")
    for i, sample in enumerate(samples):
        print(f"{i+1}. Label: {sample['label']} - Mode: {sample['mode']}, " +
              f"Difficulty: {sample['difficulty']}")

    print(f"\nDataset generated successfully!")
    print(f"- {len(samples)} samples created")
    print(f"- Dataset saved at: {os.path.abspath(dataset_generator.output_dir)}")

    return dataset_generator, samples