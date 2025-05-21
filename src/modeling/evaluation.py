"""Model evaluation utilities."""
import numpy as np
import itertools
from typing import List, Tuple, Optional
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

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
                   batch_size: int = 64,
                   model_name: Optional[str] = None,
                   patience: int = 8,
                   lr_patience: int = 4,
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
    