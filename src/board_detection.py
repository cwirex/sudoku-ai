"""
Sudoku board detection and cell extraction functionality.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Any, Union

@dataclass
class ProcessingParams:
    """Configuration parameters for image processing"""
    # Board extraction parameters
    corner_offset: float = 1.0
    target_size: int = 480
    cell_size_divisor: int = 9
    blur_kernel_size: int = 5
    threshold_block_size: int = 11
    threshold_c: int = 5
    plot_size: tuple = (5, 5)
    subplot_size: tuple = (15, 5)


class BoardExtractor:
    """Class for extracting Sudoku board and cells from images"""

    def __init__(self, params: Optional[ProcessingParams] = None):
        """
        Initialize the board extractor.

        Args:
            params: Configuration parameters for image processing
        """
        self.params = params or ProcessingParams()

    @staticmethod
    def add_offset_to_corners(corners: List[np.ndarray],
                            img_shape: tuple,
                            offset: float) -> List[np.ndarray]:
        """
        Adds an offset to each corner point, moving them towards the center.

        Args:
            corners: List of corner coordinates
            img_shape: Shape of the image (height, width)
            offset: Offset value in pixels

        Returns:
            List of adjusted corner coordinates
        """
        img_center = np.array([img_shape[1] / 2, img_shape[0] / 2])
        adjusted_corners = []

        for corner in corners:
            vector_to_center = img_center - corner
            norm_vector = vector_to_center / np.linalg.norm(vector_to_center)
            adjusted_corner = corner + norm_vector * offset
            adjusted_corners.append(adjusted_corner.astype(int))

        return adjusted_corners

    def preprocess_image(self, image: np.ndarray, show_steps: bool = False) -> np.ndarray:
        """
        Preprocess image for board detection.

        Args:
            image: Input image
            show_steps: Whether to visualize intermediate steps

        Returns:
            Preprocessed image
        """
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(gray,
                               (self.params.blur_kernel_size, self.params.blur_kernel_size),
                               0)
        thresh = cv2.adaptiveThreshold(blur, 255,
                                     cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY,
                                     self.params.threshold_block_size,
                                     self.params.threshold_c)
        inverted = cv2.bitwise_not(thresh)
        kernel = np.array([[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]], np.uint8)
        dilated = cv2.dilate(inverted, kernel)

        if show_steps:
            fig, axes = plt.subplots(1, 5, figsize=self.params.subplot_size)
            steps = [('Grayscale', gray),
                    ('Blur', blur),
                    ('Threshold', thresh),
                    ('Inverted', inverted),
                    ('Dilated', dilated)]
            for idx, (title, img) in enumerate(steps):
                axes[idx].imshow(img, cmap='gray')
                axes[idx].set_title(title)
                axes[idx].axis('off')
            plt.tight_layout()
            plt.show()

        return dilated

    def find_board_corners(self,
                          processed_image: np.ndarray,
                          original_image: Optional[np.ndarray] = None,
                          show_steps: bool = False) -> Optional[List[np.ndarray]]:
        """
        Find board corners in the processed image.

        Args:
            processed_image: Preprocessed image
            original_image: Original image (for visualization)
            show_steps: Whether to visualize the corner detection process

        Returns:
            List of corner coordinates or None if detection failed
        """
        contours, _ = cv2.findContours(processed_image.copy(),
                                     cv2.RETR_EXTERNAL,
                                     cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None

        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        points = contours[0].reshape(-1, 2)

        # Find corner points
        original_corners = [
            points[np.argmin(points.sum(axis=1))],     # Top-left
            points[np.argmin(points[:, 1] - points[:, 0])],  # Top-right
            points[np.argmax(points.sum(axis=1))],     # Bottom-right
            points[np.argmax(points[:, 1] - points[:, 0])]   # Bottom-left
        ]

        # Apply offset to corners
        adjusted_corners = self.add_offset_to_corners(
            original_corners,
            processed_image.shape,
            self.params.corner_offset
        )

        if show_steps and original_image is not None:
            self._visualize_corners(original_image, contours,
                                  original_corners, adjusted_corners)

        return adjusted_corners

    def _visualize_corners(self, original_image, contours, original_corners, adjusted_corners):
        """
        Helper method to visualize corner detection.

        Args:
            original_image: Original input image
            contours: Detected contours
            original_corners: Original corner coordinates
            adjusted_corners: Adjusted corner coordinates
        """
        img_original = original_image.copy()
        img_adjusted = original_image.copy()

        cv2.drawContours(img_original, contours, 0, (0, 255, 0), 2)
        adjusted_contour = np.array([adjusted_corners], dtype=np.int32)
        cv2.drawContours(img_adjusted, adjusted_contour, -1, (0, 255, 0), 2)

        for corner in original_corners:
            cv2.circle(img_original, tuple(corner), 5, (255, 0, 0), -1)

        for corner in adjusted_corners:
            cv2.circle(img_adjusted, tuple(corner), 5, (0, 0, 255), -1)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.params.subplot_size)
        ax1.imshow(cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB))
        ax1.set_title('Original Corners (Blue)')
        ax1.axis('off')

        ax2.imshow(cv2.cvtColor(img_adjusted, cv2.COLOR_BGR2RGB))
        ax2.set_title(f'Adjusted Corners (Red)\nOffset: {self.params.corner_offset}px')
        ax2.axis('off')

        plt.tight_layout()
        plt.show()

    def transform_perspective(self,
                            image: np.ndarray,
                            corners: List[np.ndarray],
                            show_result: bool = False) -> np.ndarray:
        """
        Apply perspective transform to extract the board.

        Args:
            image: Input image
            corners: Corner coordinates
            show_result: Whether to visualize the result

        Returns:
            Transformed image
        """
        src = np.float32(corners)
        dst = np.float32([[0, 0],
                         [self.params.target_size, 0],
                         [self.params.target_size, self.params.target_size],
                         [0, self.params.target_size]])

        matrix = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(image, matrix,
                                   (self.params.target_size, self.params.target_size))

        if show_result:
            plt.figure(figsize=self.params.plot_size)
            plt.imshow(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
            plt.title('Perspective Transform Result')
            plt.axis('off')
            plt.show()

        return warped

    def extract_cells(self,
                     board_image: np.ndarray,
                     show_grid: bool = False) -> List[Tuple[np.ndarray, Tuple[int, int]]]:
        """
        Extract individual cells from the board.

        Args:
            board_image: Transformed board image
            show_grid: Whether to visualize the grid

        Returns:
            List of (cell_image, (row, col)) tuples
        """
        cell_size = board_image.shape[0] // self.params.cell_size_divisor
        cells = []

        if show_grid:
            self._visualize_grid(board_image, cell_size)

        for row in range(self.params.cell_size_divisor):
            for col in range(self.params.cell_size_divisor):
                top = row * cell_size
                left = col * cell_size
                cell_img = board_image[top:top+cell_size, left:left+cell_size]
                cells.append((cell_img, (row, col)))

        return cells

    def _visualize_grid(self, board_image: np.ndarray, cell_size: int):
        """
        Helper method to visualize grid overlay.

        Args:
            board_image: Board image
            cell_size: Size of each cell in pixels
        """
        grid_image = board_image.copy()
        for i in range(self.params.cell_size_divisor + 1):
            thickness = 3 if i % 3 == 0 else 1
            color = (0, 255, 0) if i % 3 == 0 else (0, 200, 0)
            cv2.line(grid_image, (0, i * cell_size),
                    (board_image.shape[1], i * cell_size), color, thickness)
            cv2.line(grid_image, (i * cell_size, 0),
                    (i * cell_size, board_image.shape[0]), color, thickness)

        plt.figure(figsize=self.params.plot_size)
        plt.imshow(cv2.cvtColor(grid_image, cv2.COLOR_BGR2RGB))
        plt.title('Grid Overlay')
        plt.axis('off')
        plt.show()

    def extract_board(self,
                     image: np.ndarray,
                     display_steps: bool = False) -> Tuple[Optional[np.ndarray], Optional[List]]:
        """
        Extract the Sudoku board and cells from an image.

        Args:
            image: Input image
            display_steps: Whether to visualize intermediate steps

        Returns:
            Tuple of (transformed_board, cells) or (None, None) if extraction failed
        """
        if display_steps:
            plt.figure(figsize=self.params.plot_size)
            plt.imshow(image)
            plt.title('Original Image')
            plt.axis('off')
            plt.show()

        processed = self.preprocess_image(image, show_steps=display_steps)
        corners = self.find_board_corners(processed, image, show_steps=display_steps)

        if corners is not None:
            warped = self.transform_perspective(image, corners, show_result=display_steps)
            cells = self.extract_cells(warped, show_grid=display_steps)

            return warped, cells

        return None, None

    def visualize_ground_truth(self, grid: np.ndarray):
        """
        Helper method to display ground truth board.

        Args:
            grid: Ground truth grid values
        """
        plt.figure(figsize=self.params.plot_size)
        plt.fill([0, 9, 9, 0], [0, 0, 9, 9], 'white')

        # Draw grid lines
        for i in range(10):
            lw = 2 if i % 3 == 0 else 0.5
            plt.axhline(y=i, color='black', linewidth=lw)
            plt.axvline(x=i, color='black', linewidth=lw)

        # Add numbers
        for i in range(9):
            for j in range(9):
                if grid[i, j] != 0:
                    plt.text(j + 0.5, i + 0.5, str(grid[i, j]),
                           ha='center', va='center',
                           color='black',
                           fontsize=12,
                           fontweight='bold')

        plt.xlim(0, 9)
        plt.ylim(9, 0)
        plt.title('Ground Truth Labels')
        plt.axis('equal')
        plt.axis('off')
        plt.show()