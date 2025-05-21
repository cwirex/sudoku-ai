import numpy as np


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
