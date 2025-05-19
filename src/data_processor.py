"""
Data processing and dataset management for Sudoku AI.
"""

import os
import json
import random
import numpy as np
import cv2
from typing import List, Tuple, Dict, Optional, Any, Union
import zipfile
from tqdm import tqdm  # Use standard tqdm, not notebook version

from .board_detection import BoardExtractor, ProcessingParams
from .utils import setup_project_paths

class SudokuDataset:
    """Class for loading and managing Sudoku datasets."""

    def __init__(self, base_path: Optional[str] = None):
        """
        Initialize the dataset manager.

        Args:
            base_path: Base directory for datasets
        """
        if base_path is None:
            paths = setup_project_paths()
            base_path = paths['data_dir']
            
        self.base_path = base_path
        self.extracted_path = None
        self.dataset_dir = None
        self.dataset_info = None

    def load_dataset(self,
                    file_name: str,
                    zip_path: Optional[str] = None,
                    extract_path: Optional[str] = None,
                    force_extract: bool = False):
        """
        Load a dataset from a zip file.

        Args:
            file_name: Name of the dataset file
            zip_path: Path to the zip file (if None, will use base_path)
            extract_path: Path where to extract the dataset
            force_extract: Whether to force extraction even if already extracted

        Returns:
            bool: True if successful, False otherwise
        """
        # Set up paths
        if zip_path is None:
            zip_file_path = os.path.join(self.base_path, f'{file_name}.zip')
        else:
            zip_file_path = zip_path

        if extract_path is None:
            extract_path = os.path.join(self.base_path, 'extracted')
        
        self.extracted_path = extract_path
        self.dataset_dir = os.path.join(extract_path, file_name)

        # Extract dataset
        try:
            # Check if already extracted and not forcing re-extraction
            if os.path.exists(self.dataset_dir) and not force_extract:
                print(f"Dataset already extracted at {self.dataset_dir}")
            else:
                # Create extraction directory if it doesn't exist
                os.makedirs(extract_path, exist_ok=True)

                # Extract zip file
                with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_path)
                print(f"Successfully extracted contents of '{zip_file_path}' to '{extract_path}'")

            # Load dataset info
            return self._load_dataset_info()

        except FileNotFoundError:
            print(f"Error: File '{zip_file_path}' not found.")
            return False
        except zipfile.BadZipFile:
            print(f"Error: '{zip_file_path}' is not a valid zip file.")
            return False
        except Exception as e:
            print(f"Error extracting dataset: {e}")
            return False

    def _load_dataset_info(self):
        """
        Load the dataset information from dataset_info.json.

        Returns:
            bool: True if successful, False otherwise
        """
        dataset_info_path = os.path.join(self.dataset_dir, 'dataset_info.json')

        try:
            with open(dataset_info_path, 'r') as f:
                self.dataset_info = json.load(f)

            # Validate dataset
            dataset_size = self.dataset_info['dataset_size']
            num_images = len(os.listdir(os.path.join(self.dataset_dir, 'images')))
            num_labels = len(os.listdir(os.path.join(self.dataset_dir, 'labels')))

            assert dataset_size == num_images == num_labels, "Dataset size does not match number of images and/or labels."
            print(f"Dataset size: {dataset_size}")

            return True

        except FileNotFoundError:
            print(f"Error: File '{dataset_info_path}' not found.")
            return False
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON format in '{dataset_info_path}'.")
            return False
        except KeyError as e:
            print(f"Error: Key '{e}' not found in '{dataset_info_path}'.")
            return False
        except Exception as e:
            print(f"Error loading dataset info: {e}")
            return False

    def print_dataset_info(self, detailed: bool = False):
        """
        Print information about the loaded dataset.

        Args:
            detailed: Whether to print detailed sample information
        """
        if self.dataset_info is None:
            print("No dataset loaded. Call load_dataset() first.")
            return

        print(f"Dataset size: {self.dataset_info['dataset_size']}")
        print(f"Config: {self.dataset_info['config']}")

        if detailed and 'samples' in self.dataset_info:
            print(f"Example: {self.dataset_info['samples'][0]}")
            sample = self.dataset_info['samples'][0]
            print(f"Grid: {sample['grid']}")
            print(f"Solution: {sample['solution']}")
            print(f"Handwritten mask: {sample['handwritten_mask']}")

    def load_samples(self,
                    max_samples: Optional[int] = None,
                    difficulties: Optional[List[str]] = None,
                    modes: Optional[List[str]] = None,
                    random_seed: int = 42):
        """
        Load specific samples from the dataset.

        Args:
            max_samples: Maximum number of samples to load (None for all)
            difficulties: List of difficulties to include (None for all)
            modes: List of modes to include (None for all)
            random_seed: Random seed for sample selection

        Returns:
            List of loaded samples
        """
        if self.dataset_info is None:
            print("No dataset loaded. Call load_dataset() first.")
            return []

        # Filter samples based on criteria
        filtered_samples = self.dataset_info['samples']

        if difficulties:
            filtered_samples = [s for s in filtered_samples if s['difficulty'] in difficulties]

        if modes:
            filtered_samples = [s for s in filtered_samples if s['mode'] in modes]

        # Limit number of samples if specified
        if max_samples and max_samples < len(filtered_samples):
            random.seed(random_seed)
            filtered_samples = random.sample(filtered_samples, max_samples)

        # Load the samples
        loaded_samples = []
        for sample in tqdm(filtered_samples, desc="Loading samples"):
            img_path = os.path.join(self.dataset_dir, sample['image_path'])
            image = cv2.imread(img_path)

            if image is None:
                print(f"Warning: Could not load image at {img_path}")
                continue

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            loaded_samples.append({
                'id': sample['id'],
                'image': image,
                'grid': np.array(sample['grid']),
                'solution': np.array(sample['solution']),
                'handwritten_mask': np.array(sample['handwritten_mask']),
                'mode': sample['mode'],
                'difficulty': sample['difficulty'],
                'background_style': sample['background_style']
            })

        print(f"Loaded {len(loaded_samples)} samples")
        return loaded_samples

    def load_samples_by_difficulty(self,
                                  difficulties: List[str] = ['easy', 'medium', 'hard'],
                                  num_per_difficulty: int = 3,
                                  random_seed: int = 42):
        """
        Load a specified number of samples for each difficulty level.

        Args:
            difficulties: List of difficulty levels to include
            num_per_difficulty: Number of samples to load per difficulty
            random_seed: Random seed for sample selection

        Returns:
            Dict mapping difficulties to sample lists
        """
        if self.dataset_info is None:
            print("No dataset loaded. Call load_dataset() first.")
            return {}

        random.seed(random_seed)

        # Group samples by difficulty
        samples_by_difficulty = {d: [] for d in difficulties}
        for sample in self.dataset_info['samples']:
            if sample['difficulty'] in difficulties:
                samples_by_difficulty[sample['difficulty']].append(sample)

        # Select random samples for each difficulty
        selected_samples = {d: [] for d in difficulties}

        for difficulty in difficulties:
            available_samples = samples_by_difficulty[difficulty]

            # Determine how many samples to select
            count = min(num_per_difficulty, len(available_samples))
            if count < num_per_difficulty:
                print(f"Warning: Not enough samples for difficulty '{difficulty}'. "
                      f"Requested {num_per_difficulty}, but only {count} available.")

            # Choose random samples
            if count > 0:
                chosen = random.sample(available_samples, count)

                # Load images for selected samples
                for sample in chosen:
                    img_path = os.path.join(self.dataset_dir, sample['image_path'])
                    image = cv2.imread(img_path)

                    if image is None:
                        print(f"Warning: Could not load image at {img_path}")
                        continue

                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                    selected_samples[difficulty].append({
                        'id': sample['id'],
                        'image': image,
                        'grid': np.array(sample['grid']),
                        'solution': np.array(sample['solution']),
                        'handwritten_mask': np.array(sample['handwritten_mask']),
                        'mode': sample['mode'],
                        'difficulty': sample['difficulty'],
                        'background_style': sample['background_style']
                    })

        # Print statistics
        total_samples = sum(len(samples) for samples in selected_samples.values())
        print(f"Loaded {total_samples} samples total")
        for difficulty, samples in selected_samples.items():
            print(f"  - {difficulty}: {len(samples)} samples")

        return selected_samples


class SudokuDataProcessor:
    """Process Sudoku boards to prepare data for AI training"""

    def __init__(self, dataset_dir: str, target_size: Tuple[int, int] = (28, 28)):
        """
        Initialize the data processor.

        Args:
            dataset_dir: Path to the dataset directory
            target_size: Size to resize each cell to
        """
        self.dataset_dir = dataset_dir
        self.target_size = target_size
        self.board_extractor = BoardExtractor(ProcessingParams())
        self.samples = None
        self.digits_data = None
        self.cell_types_data = None
        self.dataset_info = None

    def load_dataset_info(self) -> bool:
        """
        Load the dataset information from dataset_info.json.

        Returns:
            bool: Success status
        """
        info_path = os.path.join(self.dataset_dir, 'dataset_info.json')
        try:
            with open(info_path, 'r') as f:
                self.dataset_info = json.load(f)
            print(f"Loaded dataset with {self.dataset_info['dataset_size']} samples")
            return True
        except Exception as e:
            print(f"Error loading dataset info: {e}")
            return False

    def load_samples(self,
                    max_samples: Optional[int] = None,
                    difficulties: Optional[List[str]] = None,
                    modes: Optional[List[str]] = None) -> List[Dict]:
        """
        Load samples from the dataset.

        Args:
            max_samples: Maximum number of samples to load (None for all)
            difficulties: List of difficulties to include (None for all)
            modes: List of modes to include (None for all)

        Returns:
            List of loaded samples
        """
        if not hasattr(self, 'dataset_info') or self.dataset_info is None:
            if not self.load_dataset_info():
                return []

        # Filter samples based on criteria
        filtered_samples = self.dataset_info['samples']

        if difficulties:
            filtered_samples = [s for s in filtered_samples if s['difficulty'] in difficulties]

        if modes:
            filtered_samples = [s for s in filtered_samples if s['mode'] in modes]

        # Limit number of samples if specified
        if max_samples and max_samples < len(filtered_samples):
            filtered_samples = random.sample(filtered_samples, max_samples)

        # Load the samples
        loaded_samples = []
        for sample in tqdm(filtered_samples, desc="Loading samples"):
            img_path = os.path.join(self.dataset_dir, sample['image_path'])
            image = cv2.imread(img_path)

            if image is None:
                print(f"Warning: Could not load image at {img_path}")
                continue

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            loaded_samples.append({
                'id': sample['id'],
                'image': image,
                'grid': np.array(sample['grid']),
                'solution': np.array(sample['solution']),
                'handwritten_mask': np.array(sample['handwritten_mask']),
                'mode': sample['mode'],
                'difficulty': sample['difficulty'],
                'background_style': sample['background_style']
            })

        self.samples = loaded_samples
        print(f"Loaded {len(loaded_samples)} samples")
        return loaded_samples

    def extract_cells_from_samples(self, display_progress: bool = True) -> Optional[Dict]:
        """
        Extract cells from all loaded samples.

        Args:
            display_progress: Whether to display a progress bar

        Returns:
            Dictionary with extracted cells data or None if extraction failed
        """
        if not self.samples:
            print("No samples loaded. Call load_samples() first.")
            return None

        # Containers for cell data
        digit_images = []  # Cell images
        digit_labels = []  # Digit values (0-9)
        cell_type_labels = []  # Cell types (0=empty, 1=printed, 2=handwritten)
        metadata = []  # Additional info about each cell

        # Process each sample
        samples_iterator = tqdm(self.samples) if display_progress else self.samples
        for sample in samples_iterator:
            # Extract board and cells
            warped, cells = self.board_extractor.extract_board(sample['image'], display_steps=False)

            if warped is None or cells is None:
                continue

            # Process each cell
            for cell_img, (row, col) in cells:
                # Resize cell image
                resized_cell = cv2.resize(cell_img, self.target_size)

                # Convert to grayscale
                if len(resized_cell.shape) == 3:
                    gray_cell = cv2.cvtColor(resized_cell, cv2.COLOR_RGB2GRAY)
                else:
                    gray_cell = resized_cell

                # Get digit value from grid
                digit = sample['grid'][row, col]

                # Determine cell type
                if digit == 0:
                    cell_type = 0  # Empty
                elif sample['handwritten_mask'][row, col] == 1:
                    cell_type = 2  # Handwritten
                elif sample['handwritten_mask'][row, col] == 0:
                    cell_type = 1  # Printed

                # Store the data
                digit_images.append(gray_cell)
                digit_labels.append(digit)
                cell_type_labels.append(cell_type)
                metadata.append({
                    'sample_id': sample['id'],
                    'row': row,
                    'col': col,
                    'difficulty': sample['difficulty'],
                    'mode': sample['mode']
                })

        # Convert lists to numpy arrays
        digit_images = np.array(digit_images)
        digit_labels = np.array(digit_labels)
        cell_type_labels = np.array(cell_type_labels)

        # Normalize images to [0, 1]
        digit_images = digit_images / 255.0

        # Store the processed data
        self.digits_data = {
            'images': digit_images,
            'labels': digit_labels,
            'cell_types': cell_type_labels,
            'metadata': metadata
        }

        # Print some statistics
        print(f"Processed {len(digit_images)} cells")
        print(f"Digit distribution: {np.bincount(digit_labels)}")
        print(f"Cell type distribution: {np.bincount(cell_type_labels)}")

        return self.digits_data

    def prepare_training_data(self,
                             test_size: float = 0.2,
                             validation_size: float = 0.1,
                             random_state: int = 42) -> Optional[Dict]:
        """
        Prepare training, validation, and test datasets.

        Args:
            test_size: Proportion of data to use for testing
            validation_size: Proportion of training data to use for validation
            random_state: Random seed for reproducibility

        Returns:
            Dictionary with prepared datasets or None if preparation failed
        """
        if self.digits_data is None:
            print("No cell data available. Call extract_cells_from_samples() first.")
            return None

        from sklearn.model_selection import train_test_split
        from tensorflow.keras.utils import to_categorical

        # Prepare data for digit recognition
        X = self.digits_data['images']
        y_digits = self.digits_data['labels']
        y_cell_types = self.digits_data['cell_types']

        # Split into train and test sets
        X_train, X_test, y_digits_train, y_digits_test, y_cell_types_train, y_cell_types_test = train_test_split(
            X, y_digits, y_cell_types, test_size=test_size, random_state=random_state
        )

        # Split training data to create validation set
        X_train, X_val, y_digits_train, y_digits_val, y_cell_types_train, y_cell_types_val = train_test_split(
            X_train, y_digits_train, y_cell_types_train,
            test_size=validation_size/(1-test_size),  # Adjust validation size
            random_state=random_state
        )

        # Reshape images for CNN input [samples, height, width, channels]
        X_train = X_train.reshape(-1, self.target_size[0], self.target_size[1], 1)
        X_val = X_val.reshape(-1, self.target_size[0], self.target_size[1], 1)
        X_test = X_test.reshape(-1, self.target_size[0], self.target_size[1], 1)

        # Convert labels to categorical
        y_digits_train_cat = to_categorical(y_digits_train, num_classes=10)
        y_digits_val_cat = to_categorical(y_digits_val, num_classes=10)
        y_digits_test_cat = to_categorical(y_digits_test, num_classes=10)

        y_cell_types_train_cat = to_categorical(y_cell_types_train, num_classes=3)
        y_cell_types_val_cat = to_categorical(y_cell_types_val, num_classes=3)
        y_cell_types_test_cat = to_categorical(y_cell_types_test, num_classes=3)

        # Create datasets for digit recognition
        digit_recognition = {
            'train': (X_train, y_digits_train_cat),
            'val': (X_val, y_digits_val_cat),
            'test': (X_test, y_digits_test_cat),
            'train_raw': (X_train, y_digits_train),
            'val_raw': (X_val, y_digits_val),
            'test_raw': (X_test, y_digits_test)
        }

        # Create datasets for cell type classification
        cell_type_classification = {
            'train': (X_train, y_cell_types_train_cat),
            'val': (X_val, y_cell_types_val_cat),
            'test': (X_test, y_cell_types_test_cat),
            'train_raw': (X_train, y_cell_types_train),
            'val_raw': (X_val, y_cell_types_val),
            'test_raw': (X_test, y_cell_types_test)
        }

        # Print dataset shapes
        print("Dataset shapes:")
        print(f"Train: {X_train.shape}, Validation: {X_val.shape}, Test: {X_test.shape}")

        return {
            'digit_recognition': digit_recognition,
            'cell_type_classification': cell_type_classification
        }

    def visualize_samples(self, num_samples: int = 10):
        """
        Visualize random samples from the processed data.

        Args:
            num_samples: Number of samples to visualize
        """
        import matplotlib.pyplot as plt
        import random
        
        if self.digits_data is None:
            print("No cell data available. Call extract_cells_from_samples() first.")
            return

        # Select random samples
        indices = random.sample(range(len(self.digits_data['images'])), min(num_samples, len(self.digits_data['images'])))

        # Create figure
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        axes = axes.flatten()

        for i, idx in enumerate(indices):
            if i >= len(axes):
                break

            # Get image and labels
            img = self.digits_data['images'][idx]
            digit = self.digits_data['labels'][idx]
            cell_type = self.digits_data['cell_types'][idx]

            # Map cell type to string
            cell_type_str = {0: 'Empty', 1: 'Printed', 2: 'Handwritten'}[cell_type]

            # Display image
            axes[i].imshow(img, cmap='gray')
            axes[i].set_title(f"Digit: {digit}\nType: {cell_type_str}")
            axes[i].axis('off')

        plt.tight_layout()
        plt.show()

    def visualize_digit_distribution(self):
        """Visualize the distribution of digits in the dataset"""
        import matplotlib.pyplot as plt
        
        if self.digits_data is None:
            print("No cell data available. Call extract_cells_from_samples() first.")
            return

        # Count occurrences of each digit
        digit_counts = np.bincount(self.digits_data['labels'])

        # Create bar plot
        plt.figure(figsize=(10, 5))
        plt.bar(range(len(digit_counts)), digit_counts)
        plt.title('Digit Distribution in Dataset')
        plt.xlabel('Digit Value')
        plt.ylabel('Count')
        plt.xticks(range(len(digit_counts)))
        plt.grid(axis='y', alpha=0.3)

        # Add count labels on top of bars
        for i, count in enumerate(digit_counts):
            plt.text(i, count + 50, str(count), ha='center')

        plt.tight_layout()
        plt.show()

    def visualize_cell_types(self):
        """Visualize the distribution of cell types in the dataset"""
        import matplotlib.pyplot as plt
        
        if self.digits_data is None:
            print("No cell data available. Call extract_cells_from_samples() first.")
            return

        # Count occurrences of each cell type
        cell_type_counts = np.bincount(self.digits_data['cell_types'])

        # Create bar plot
        plt.figure(figsize=(8, 5))
        plt.bar(['Empty', 'Printed', 'Handwritten'], cell_type_counts)
        plt.title('Cell Type Distribution in Dataset')
        plt.xlabel('Cell Type')
        plt.ylabel('Count')
        plt.grid(axis='y', alpha=0.3)

        # Add count labels on top of bars
        for i, count in enumerate(cell_type_counts):
            plt.text(i, count + 50, str(count), ha='center')

        plt.tight_layout()
        plt.show()