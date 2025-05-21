import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import random

from src.config.generator_config import AugmentationConfig, BackgroundConfig, DigitConfig, GridConfig
from src.generation.providers import FontProvider, MNISTDigitProvider

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
