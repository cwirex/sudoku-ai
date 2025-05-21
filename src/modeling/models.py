from typing import Tuple
from keras import layers, models, optimizers


class SudokuModels:
    """Class containing different model architectures for Sudoku recognition tasks"""

    def __init__(self, input_shape: Tuple[int, int, int] = (28, 28, 1), learning_rate: float = 0.001):
        """
        Initialize models with input shape.

        Args:
            input_shape: Shape of input images (height, width, channels)
        """
        self.input_shape = input_shape
        self.learning_rate = learning_rate

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
        model.compile(optimizer=optimizers.Adam(learning_rate=self.learning_rate),
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
        model.compile(optimizer=optimizers.Adam(learning_rate=self.learning_rate),
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

