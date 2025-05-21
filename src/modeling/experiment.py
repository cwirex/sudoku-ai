"""End-to-end experiment management."""
from typing import List, Tuple, Optional

from src.modeling.evaluation import ModelEvaluator
from src.modeling.models import SudokuModels
        
class SudokuExperiment:
    """Class for managing Sudoku experiments end-to-end"""

    def __init__(self, base_path: str = None):
        """
        Initialize experiment manager.

        Args:
            base_path: Base directory for datasets and results
        """
        from processing.data_processor import SudokuDataset, SudokuDataProcessor
        from utils import setup_project_paths
        
        if base_path is None:
            paths = setup_project_paths()
            base_path = paths['data_dir']
            
        self.base_path = base_path
        self.dataset_manager = SudokuDataset(base_path)
        self.data_processor = None
        self.model_builder = None
        self.model_evaluator = None
        self.datasets = None
        self.results = {}

    def setup_dataset(self, file_name: str):
        """
        Set up the dataset for the experiment.

        Args:
            file_name: Name of the dataset file

        Returns:
            Bool indicating success
        """
        # Load the dataset
        success = self.dataset_manager.load_dataset(file_name)
        if success:
            # Initialize data processor with dataset directory
            from processing.data_processor import SudokuDataProcessor
            self.data_processor = SudokuDataProcessor(self.dataset_manager.dataset_dir)
            return True
        return False

    def load_samples(self, max_samples: Optional[int] = None, difficulties: Optional[List[str]] = None):
        """
        Load samples from the dataset.

        Args:
            max_samples: Maximum number of samples to load
            difficulties: List of difficulty levels to include

        Returns:
            List of loaded samples
        """
        if self.data_processor is None:
            print("Dataset not set up. Call setup_dataset() first.")
            return []

        return self.data_processor.load_samples(max_samples=max_samples, difficulties=difficulties)

    def prepare_data(self):
        """
        Extract cells and prepare training data.

        Returns:
            Dictionary with prepared datasets
        """
        if self.data_processor is None:
            print("Dataset not set up. Call setup_dataset() first.")
            return None

        # Extract cells from samples
        self.data_processor.extract_cells_from_samples()

        # Prepare training data
        self.datasets = self.data_processor.prepare_training_data()
        return self.datasets

    def visualize_data(self):
        """Visualize the processed data"""
        if self.data_processor is None or self.data_processor.digits_data is None:
            print("No data to visualize. Call prepare_data() first.")
            return

        # Show sample images
        self.data_processor.visualize_samples(10)

        # Show digit distribution
        self.data_processor.visualize_digit_distribution()

        # Show cell type distribution
        self.data_processor.visualize_cell_types()

    def build_models(self, input_shape: Tuple[int, int, int] = (28, 28, 1), learning_rate: float = 0.001):
        """
        Initialize model builder.

        Args:
            input_shape: Shape of input images

        Returns:
            SudokuModels instance
        """
        self.model_builder = SudokuModels(input_shape, learning_rate=learning_rate)
        self.model_evaluator = ModelEvaluator()
        return self.model_builder

    def train_digit_models(self, epochs: int = 10, batch_size: int = 32, patience: int = 5):
        """
        Train digit recognition models.

        Args:
            epochs: Maximum number of epochs to train
            batch_size: Batch size for training

        Returns:
            Dictionary with trained models
        """
        if self.model_builder is None:
            print("Models not built. Call build_models() first.")
            return {}

        if self.datasets is None:
            print("Data not prepared. Call prepare_data() first.")
            return {}

        digit_data = self.datasets['digit_recognition']

        # Train simple CNN
        simple_cnn = self.model_builder.simple_cnn(num_classes=10, name="simple_cnn_digits")
        self.model_evaluator.train_model(
            simple_cnn,
            digit_data['train'],
            digit_data['val'],
            epochs=epochs,
            batch_size=batch_size,
            patience=patience
        )
        self.model_evaluator.evaluate_model(simple_cnn, digit_data['test'])

        # Train deeper CNN
        deeper_cnn = self.model_builder.deeper_cnn(num_classes=10, name="deeper_cnn_digits")
        self.model_evaluator.train_model(
            deeper_cnn,
            digit_data['train'],
            digit_data['val'],
            epochs=epochs,
            batch_size=batch_size,
            patience=patience
        )
        self.model_evaluator.evaluate_model(deeper_cnn, digit_data['test'])

        # Train ResNet-like mini
        resnet_mini = self.model_builder.resnet_like_mini(num_classes=10, name="resnet_mini_digits")
        self.model_evaluator.train_model(
            resnet_mini,
            digit_data['train'],
            digit_data['val'],
            epochs=epochs,
            batch_size=batch_size,
            patience=patience
        )
        self.model_evaluator.evaluate_model(resnet_mini, digit_data['test'])

        # Store results
        self.results['digit_models'] = {
            'simple_cnn': simple_cnn,
            'deeper_cnn': deeper_cnn,
            'resnet_mini': resnet_mini
        }

        return self.results['digit_models']

    def train_cell_type_models(self, epochs: int = 10, batch_size: int = 32, patience: int = 5):
        """
        Train cell type classification models.

        Args:
            epochs: Maximum number of epochs to train
            batch_size: Batch size for training

        Returns:
            Dictionary with trained models
        """
        if self.model_builder is None:
            print("Models not built. Call build_models() first.")
            return {}

        if self.datasets is None:
            print("Data not prepared. Call prepare_data() first.")
            return {}

        cell_type_data = self.datasets['cell_type_classification']

        # Train simple cell classifier
        cell_classifier = self.model_builder.cell_type_classifier(num_classes=3, name="simple_cell_classifier")
        self.model_evaluator.train_model(
            cell_classifier,
            cell_type_data['train'],
            cell_type_data['val'],
            epochs=epochs,
            batch_size=batch_size,
            patience=patience
        )
        self.model_evaluator.evaluate_model(cell_classifier, cell_type_data['test'])

        # Train deeper cell classifier
        deeper_cell = self.model_builder.deeper_cnn(num_classes=3, name="deeper_cell_classifier")
        self.model_evaluator.train_model(
            deeper_cell,
            cell_type_data['train'],
            cell_type_data['val'],
            epochs=epochs,
            batch_size=batch_size,
            patience=patience
        )
        self.model_evaluator.evaluate_model(deeper_cell, cell_type_data['test'])

        # Store results
        self.results['cell_type_models'] = {
            'simple_cell_classifier': cell_classifier,
            'deeper_cell_classifier': deeper_cell
        }

        return self.results['cell_type_models']

    def evaluate_digit_models(self):
        """
        Evaluate and compare digit recognition models.

        Returns:
            DataFrame with comparison results
        """
        if 'digit_models' not in self.results:
            print("Digit models not trained. Call train_digit_models() first.")
            return None

        # Compare digit recognition models
        print("\nDigit Recognition Model Comparison:")
        digit_models = ["simple_cnn_digits", "deeper_cnn_digits", "resnet_mini_digits"]
        self.model_evaluator.plot_training_history(digit_models)

        digit_comparison = self.model_evaluator.compare_models()
        print(digit_comparison)

        self.model_evaluator.plot_comparison_chart('Test Accuracy')

        # Plot confusion matrices for digit models
        for model_name in digit_models:
            self.model_evaluator.plot_confusion_matrix(
                model_name,
                class_names=[str(i) for i in range(10)],
                normalize=False
            )

        return digit_comparison

    def evaluate_cell_type_models(self):
        """
        Evaluate and compare cell type classification models.

        Returns:
            DataFrame with comparison results
        """
        if 'cell_type_models' not in self.results:
            print("Cell type models not trained. Call train_cell_type_models() first.")
            return None

        # Compare cell type classification models
        print("\nCell Type Classification Model Comparison:")
        cell_models = ["simple_cell_classifier", "deeper_cell_classifier"]
        self.model_evaluator.plot_training_history(cell_models)

        cell_comparison = self.model_evaluator.compare_models()
        print(cell_comparison)

        # Plot confusion matrices for cell type models
        cell_class_names = ['Empty', 'Printed', 'Handwritten']
        for model_name in cell_models:
            self.model_evaluator.plot_confusion_matrix(
                model_name,
                class_names=cell_class_names,
                normalize=False
            )

        return cell_comparison