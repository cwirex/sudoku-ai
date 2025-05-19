"""
Models for Sudoku digit recognition and cell classification.
"""

import numpy as np
import itertools
from typing import List, Tuple, Dict, Optional, Any, Union
import tensorflow as tf
from keras import layers, models, optimizers
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns


class SudokuModels:
    """Class containing different model architectures for Sudoku recognition tasks"""

    def __init__(self, input_shape: Tuple[int, int, int] = (28, 28, 1)):
        """
        Initialize models with input shape.

        Args:
            input_shape: Shape of input images (height, width, channels)
        """
        self.input_shape = input_shape

    def simple_cnn(self, num_classes: int = 10, name: str = "simple_cnn"):
        """
        A simple CNN architecture suitable for digit recognition.

        Args:
            num_classes: Number of output classes (10 for digits 0-9)
            name: Name of the model

        Returns:
            Compiled Keras model
        """
        # Use functional API to avoid input_shape warning
        inputs = layers.Input(shape=self.input_shape)

        # First convolutional block
        x = layers.Conv2D(32, (3, 3), activation='relu')(inputs)
        x = layers.MaxPooling2D((2, 2))(x)

        # Second convolutional block
        x = layers.Conv2D(64, (3, 3), activation='relu')(x)
        x = layers.MaxPooling2D((2, 2))(x)

        # Flatten and dense layers
        x = layers.Flatten()(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(num_classes, activation='softmax')(x)

        # Create model
        model = models.Model(inputs=inputs, outputs=outputs, name=name)

        # Compile model
        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        return model

    def deeper_cnn(self, num_classes: int = 10, name: str = "deeper_cnn"):
        """
        A deeper CNN with more filters and dropout for better generalization.

        Args:
            num_classes: Number of output classes
            name: Name of the model

        Returns:
            Compiled Keras model
        """
        # Use functional API to avoid input_shape warning
        inputs = layers.Input(shape=self.input_shape)

        # First convolutional block
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)

        # Second convolutional block
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)

        # Third convolutional block
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)

        # Flatten and dense layers
        x = layers.Flatten()(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(num_classes, activation='softmax')(x)

        # Create model
        model = models.Model(inputs=inputs, outputs=outputs, name=name)

        # Compile model
        model.compile(optimizer=optimizers.Adam(learning_rate=0.001),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        return model

    def resnet_like_mini(self, num_classes: int = 10, name: str = "resnet_mini"):
        """
        A simplified ResNet-like architecture with residual connections.

        Args:
            num_classes: Number of output classes
            name: Name of the model

        Returns:
            Compiled Keras model
        """
        inputs = layers.Input(shape=self.input_shape)

        # First conv layer
        x = layers.Conv2D(32, (3, 3), padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)

        # First residual block
        shortcut = x
        x = layers.Conv2D(32, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(32, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.add([shortcut, x])
        x = layers.Activation('relu')(x)

        # Second block with downsampling
        shortcut = layers.Conv2D(64, (1, 1), strides=(2, 2), padding='same')(x)
        x = layers.Conv2D(64, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.add([shortcut, x])
        x = layers.Activation('relu')(x)

        # Third block with downsampling
        shortcut = layers.Conv2D(128, (1, 1), strides=(2, 2), padding='same')(x)
        x = layers.Conv2D(128, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.add([shortcut, x])
        x = layers.Activation('relu')(x)

        # Global average pooling and output
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(num_classes, activation='softmax')(x)

        model = models.Model(inputs=inputs, outputs=outputs, name=name)

        # Compile model
        model.compile(optimizer=optimizers.Adam(learning_rate=0.001),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        return model

    def cell_type_classifier(self, num_classes: int = 3, name: str = "cell_type_classifier"):
        """
        Model for classifying cell types (empty, printed, handwritten).

        Args:
            num_classes: Number of cell types (3)
            name: Name of the model

        Returns:
            Compiled Keras model
        """
        # Use functional API to avoid input_shape warning
        inputs = layers.Input(shape=self.input_shape)

        # Convolutional layers
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2))(x)

        # Flatten and dense layers
        x = layers.Flatten()(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(num_classes, activation='softmax')(x)

        # Create model
        model = models.Model(inputs=inputs, outputs=outputs, name=name)

        # Compile model
        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        return model


class ModelEvaluator:
    """Class for training, evaluating and comparing different models"""

    def __init__(self):
        """Initialize evaluator"""
        self.history = {}
        self.evaluation_results = {}
        self.models = {}

    def train_model(self,
                   model,
                   train_data: Tuple,
                   val_data: Tuple,
                   epochs: int = 15,
                   batch_size: int = 32,
                   model_name: Optional[str] = None,
                   patience: int = 5,
                   lr_patience: int = 3,
                   min_lr: float = 0.00001):
        """
        Train a model and save its history.

        Args:
            model: Keras model to train
            train_data: Tuple of (x_train, y_train)
            val_data: Tuple of (x_val, y_val)
            epochs: Maximum number of epochs to train
            batch_size: Batch size for training
            model_name: Name to identify the model (uses model.name if None)
            patience: Early stopping patience
            lr_patience: Learning rate reduction patience
            min_lr: Minimum learning rate

        Returns:
            Trained model
        """
        x_train, y_train = train_data
        x_val, y_val = val_data

        # Set model name
        if model_name is None:
            model_name = model.name

        # Define callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.5, patience=lr_patience, min_lr=min_lr)
        ]

        # Train the model
        history = model.fit(
            x_train, y_train,
            validation_data=(x_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )

        # Save results
        self.history[model_name] = history.history
        self.models[model_name] = model

        return model

    def evaluate_model(self, model, test_data: Tuple, model_name: Optional[str] = None):
        """
        Evaluate a model on test data.

        Args:
            model: Keras model to evaluate
            test_data: Tuple of (x_test, y_test)
            model_name: Name to identify the model (uses model.name if None)

        Returns:
            Dictionary with evaluation metrics
        """
        x_test, y_test = test_data

        # Set model name
        if model_name is None:
            model_name = model.name

        # Evaluate model
        test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)

        # Get predictions
        y_pred = model.predict(x_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_test, axis=1)

        # Calculate confusion matrix
        cm = confusion_matrix(y_true_classes, y_pred_classes)

        # Generate classification report
        report = classification_report(y_true_classes, y_pred_classes, output_dict=True)

        # Store results
        self.evaluation_results[model_name] = {
            'test_loss': test_loss,
            'test_accuracy': test_acc,
            'confusion_matrix': cm,
            'classification_report': report,
            'y_true': y_true_classes,
            'y_pred': y_pred_classes
        }

        return self.evaluation_results[model_name]

    def plot_training_history(self, model_names: Optional[List[str]] = None, figsize: Tuple[int, int] = (12, 5)):
        """
        Plot training and validation accuracy/loss for specified models.

        Args:
            model_names: List of model names to plot (None for all)
            figsize: Figure size
        """
        if model_names is None:
            model_names = list(self.history.keys())

        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Plot accuracy
        ax1.set_title('Model Accuracy')
        ax1.set_ylabel('Accuracy')
        ax1.set_xlabel('Epoch')

        # Plot loss
        ax2.set_title('Model Loss')
        ax2.set_ylabel('Loss')
        ax2.set_xlabel('Epoch')

        # Add data for each model
        for name in model_names:
            history = self.history[name]

            # Plot accuracy
            ax1.plot(history['accuracy'], label=f'{name} (train)')
            ax1.plot(history['val_accuracy'], linestyle='--', label=f'{name} (val)')

            # Plot loss
            ax2.plot(history['loss'], label=f'{name} (train)')
            ax2.plot(history['val_loss'], linestyle='--', label=f'{name} (val)')

        # Add legends
        ax1.legend()
        ax2.legend()

        # Show grid
        ax1.grid(True, alpha=0.3)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def plot_confusion_matrix(self,
                             model_name: str,
                             class_names: Optional[List[str]] = None,
                             normalize: bool = False,
                             figsize: Tuple[int, int] = (8, 6),
                             cmap=plt.cm.Blues):
        """
        Plot confusion matrix for a model.

        Args:
            model_name: Name of the model to plot
            class_names: List of class names (digits, cell types)
            normalize: Whether to normalize the confusion matrix
            figsize: Figure size
            cmap: Color map for the plot
        """
        if model_name not in self.evaluation_results:
            print(f"No evaluation results found for model '{model_name}'")
            return

        # Get confusion matrix
        cm = self.evaluation_results[model_name]['confusion_matrix']

        # Generate default class names if not provided
        if class_names is None:
            class_names = [str(i) for i in range(cm.shape[0])]

        # Normalize if requested
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
            title = f'Normalized Confusion Matrix - {model_name}'
        else:
            fmt = 'd'
            title = f'Confusion Matrix - {model_name}'

        # Create figure
        plt.figure(figsize=figsize)
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()

        # Add class labels
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)

        # Add count labels
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")

        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        plt.show()

    def compare_models(self):
        """
        Compare evaluation results for all models.

        Returns:
            DataFrame with model comparison
        """
        import pandas as pd

        # Collect results
        results = []
        for name, eval_result in self.evaluation_results.items():
            results.append({
                'Model': name,
                'Test Accuracy': eval_result['test_accuracy'],
                'Test Loss': eval_result['test_loss'],
                'F1 Score (macro)': eval_result['classification_report']['macro avg']['f1-score'],
                'Precision (macro)': eval_result['classification_report']['macro avg']['precision'],
                'Recall (macro)': eval_result['classification_report']['macro avg']['recall']
            })

        # Create DataFrame
        df = pd.DataFrame(results)

        # Sort by test accuracy
        df = df.sort_values('Test Accuracy', ascending=False).reset_index(drop=True)

        return df

    def plot_comparison_chart(self, metric: str = 'Test Accuracy'):
        """
        Plot a bar chart comparing models on a specific metric.

        Args:
            metric: Metric to compare ('Test Accuracy', 'Test Loss', etc.)
        """
        # Get comparison DataFrame
        df = self.compare_models()

        if metric not in df.columns:
            print(f"Metric '{metric}' not found. Available metrics: {list(df.columns)}")
            return

        # Create plot
        plt.figure(figsize=(10, 6))

        # Sort by the selected metric
        if metric == 'Test Loss':
            df = df.sort_values(metric, ascending=True)
        else:
            df = df.sort_values(metric, ascending=False)

        # Create bar chart
        bars = plt.bar(df['Model'], df[metric])

        # Add labels
        plt.title(f'Model Comparison by {metric}')
        plt.xlabel('Model')
        plt.ylabel(metric)
        plt.xticks(rotation=45, ha='right')

        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                    f'{height:.4f}',
                    ha='center', va='bottom', rotation=0)

        plt.ylim(0, max(df[metric]) * 1.15)  # Add some space for labels
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        
class SudokuExperiment:
    """Class for managing Sudoku experiments end-to-end"""

    def __init__(self, base_path: str = None):
        """
        Initialize experiment manager.

        Args:
            base_path: Base directory for datasets and results
        """
        from .data_processor import SudokuDataset, SudokuDataProcessor
        from .utils import setup_project_paths
        
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
            from .data_processor import SudokuDataProcessor
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

    def build_models(self, input_shape: Tuple[int, int, int] = (28, 28, 1)):
        """
        Initialize model builder.

        Args:
            input_shape: Shape of input images

        Returns:
            SudokuModels instance
        """
        self.model_builder = SudokuModels(input_shape)
        self.model_evaluator = ModelEvaluator()
        return self.model_builder

    def train_digit_models(self, epochs: int = 10, batch_size: int = 32):
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
            batch_size=batch_size
        )
        self.model_evaluator.evaluate_model(simple_cnn, digit_data['test'])

        # Train deeper CNN
        deeper_cnn = self.model_builder.deeper_cnn(num_classes=10, name="deeper_cnn_digits")
        self.model_evaluator.train_model(
            deeper_cnn,
            digit_data['train'],
            digit_data['val'],
            epochs=epochs,
            batch_size=batch_size
        )
        self.model_evaluator.evaluate_model(deeper_cnn, digit_data['test'])

        # Train ResNet-like mini
        resnet_mini = self.model_builder.resnet_like_mini(num_classes=10, name="resnet_mini_digits")
        self.model_evaluator.train_model(
            resnet_mini,
            digit_data['train'],
            digit_data['val'],
            epochs=epochs,
            batch_size=batch_size
        )
        self.model_evaluator.evaluate_model(resnet_mini, digit_data['test'])

        # Store results
        self.results['digit_models'] = {
            'simple_cnn': simple_cnn,
            'deeper_cnn': deeper_cnn,
            'resnet_mini': resnet_mini
        }

        return self.results['digit_models']

    def train_cell_type_models(self, epochs: int = 10, batch_size: int = 32):
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
            batch_size=batch_size
        )
        self.model_evaluator.evaluate_model(cell_classifier, cell_type_data['test'])

        # Train deeper cell classifier
        deeper_cell = self.model_builder.deeper_cnn(num_classes=3, name="deeper_cell_classifier")
        self.model_evaluator.train_model(
            deeper_cell,
            cell_type_data['train'],
            cell_type_data['val'],
            epochs=epochs,
            batch_size=batch_size
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