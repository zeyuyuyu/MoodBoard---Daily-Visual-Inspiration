from PIL import Image
import numpy as np
from collections import Counter
from typing import List, Tuple
import colorsys

class ImageProcessor:
    def __init__(self):
        self.supported_formats = {'.jpg', '.jpeg', '.png'}

    def is_supported_format(self, filename: str) -> bool:
        return any(filename.lower().endswith(fmt) for fmt in self.supported_formats)

    def extract_color_palette(self, image_path: str, num_colors: int = 5) -> List[Tuple[int, int, int]]:
        """
        Extract the dominant color palette from an image using color quantization
        and perceptual color distance.
        
        Args:
            image_path: Path to the image file
            num_colors: Number of colors to extract (default: 5)
            
        Returns:
            List of RGB tuples representing the dominant colors
        """
        try:
            # Open and resize image for faster processing
            img = Image.open(image_path)
            img = img.resize((150, 150))
            
            # Convert to RGB if necessary
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Get pixel data
            pixels = np.float32(img).reshape(-1, 3)
            
            # Use k-means clustering to find dominant colors
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
            flags = cv2.KMEANS_RANDOM_CENTERS
            _, labels, palette = cv2.kmeans(pixels, num_colors, None, criteria, 10, flags)
            
            # Convert back to integer RGB values
            palette = palette.astype(np.uint8)
            
            # Sort colors by frequency
            unique_labels, label_counts = np.unique(labels, return_counts=True)
            palette = [tuple(color) for color in palette[unique_labels[np.argsort(-label_counts)]]]
            
            return palette

        except Exception as e:
            print(f'Error processing image {image_path}: {str(e)}')
            return []

    def calculate_color_harmony(self, colors: List[Tuple[int, int, int]]) -> float:
        """
        Calculate a harmony score for a color palette based on color theory principles.
        
        Args:
            colors: List of RGB color tuples
            
        Returns:
            Harmony score between 0 and 1
        """
        if not colors:
            return 0.0
            
        # Convert RGB to HSV for better color relationship analysis
        hsv_colors = []
        for rgb in colors:
            hsv = colorsys.rgb_to_hsv(rgb[0]/255, rgb[1]/255, rgb[2]/255)
            hsv_colors.append(hsv)
            
        # Calculate hue differences
        hue_diffs = []
        for i in range(len(hsv_colors)):
            for j in range(i + 1, len(hsv_colors)):
                diff = abs(hsv_colors[i][0] - hsv_colors[j][0])
                hue_diffs.append(min(diff, 1 - diff))
                
        # Analyze hue spacing
        if not hue_diffs:
            return 0.0
            
        avg_hue_diff = sum(hue_diffs) / len(hue_diffs)
        harmony_score = 1.0 - (abs(avg_hue_diff - 0.33) * 3)
        
        return max(0.0, min(1.0, harmony_score))

    def resize_image(self, image_path: str, max_size: Tuple[int, int]) -> Image.Image:
        """
        Resize an image while maintaining aspect ratio.
        
        Args:
            image_path: Path to the image file
            max_size: Tuple of (width, height) maximum dimensions
            
        Returns:
            Resized PIL Image object
        """
        try:
            img = Image.open(image_path)
            img.thumbnail(max_size, Image.LANCZOS)
            return img
        except Exception as e:
            print(f'Error resizing image {image_path}: {str(e)}')
            return None
