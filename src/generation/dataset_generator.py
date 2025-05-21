import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import json
import uuid
import zipfile
from datetime import datetime
from tqdm import tqdm

from src.config.generator_config import SudokuGeneratorConfig
from src.generation.image_generator import SudokuImageGenerator  # Using standard tqdm, not notebook version



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
            from utils import setup_project_paths
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