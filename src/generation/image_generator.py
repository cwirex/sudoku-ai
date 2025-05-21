import numpy as np
import random

from src.config.generator_config import SudokuGeneratorConfig
from src.generation.providers import FontProvider, MNISTDigitProvider
from src.generation.puzzle_generator import SudokuPuzzleGenerator
from src.generation.renderers import BackgroundGenerator, DigitRenderer, GridRenderer, ImageAugmentor



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
