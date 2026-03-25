import os
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from colormath.color_objects import sRGBColor, LabColor
from colormath.color_diff import delta_e_cie2000

class ImageProcessor:
    def __init__(self, image_dir='./images'):
        self.image_dir = image_dir

    def analyze_images(self):
        image_paths = [os.path.join(self.image_dir, f) for f in os.listdir(self.image_dir)]
        image_data = []
        for path in image_paths:
            image = Image.open(path)
            pixels = np.array(image).reshape(-1, 3)
            image_data.append(pixels)

        self.palette, self.palette_counts = self.generate_palette(image_data)
        self.dominant_colors = self.get_dominant_colors(self.palette, self.palette_counts)
        self.color_harmony = self.analyze_color_harmony(self.dominant_colors)
        self.style_suggestions = self.generate_style_suggestions(self.dominant_colors, self.color_harmony)

        return {
            'palette': self.palette.tolist(),
            'palette_counts': self.palette_counts.tolist(),
            'dominant_colors': self.dominant_colors,
            'color_harmony': self.color_harmony,
            'style_suggestions': self.style_suggestions
        }

    def generate_palette(self, image_data, n_colors=16):
        all_pixels = np.concatenate(image_data, axis=0)
        kmeans = KMeans(n_clusters=n_colors, random_state=42)
        kmeans.fit(all_pixels)
        palette = kmeans.cluster_centers_.astype(np.uint8)
        palette_counts = np.bincount(kmeans.labels_)
        return palette, palette_counts

    def get_dominant_colors(self, palette, palette_counts, num_colors=5):
        sorted_indices = np.argsort(palette_counts)[::-1][:num_colors]
        dominant_colors = [palette[i].tolist() for i in sorted_indices]
        return dominant_colors

    def analyze_color_harmony(self, dominant_colors):
        color_harmony = {}
        for i, color1 in enumerate(dominant_colors):
            color_harmony[i] = {}
            for j, color2 in enumerate(dominant_colors):
                if i != j:
                    lab1 = LabColor(rgb1=sRGBColor(color1[0]/255, color1[1]/255, color1[2]/255))
                    lab2 = LabColor(rgb1=sRGBColor(color2[0]/255, color2[1]/255, color2[2]/255))
                    delta_e = delta_e_cie2000(lab1, lab2)
                    color_harmony[i][j] = delta_e
        return color_harmony

    def generate_style_suggestions(self, dominant_colors, color_harmony):
        style_suggestions = []
        for i, color1 in enumerate(dominant_colors):
            for j, color2 in enumerate(dominant_colors):
                if i != j:
                    delta_e = color_harmony[i][j]
                    if delta_e < 10:
                        style_suggestions.append({
                            'primary_color': color1,
                            'secondary_color': color2,
                            'style': 'Monochromatic'
                        })
                    elif 10 <= delta_e < 20:
                        style_suggestions.append({
                            'primary_color': color1,
                            'secondary_color': color2,
                            'style': 'Analogous'
                        })
                    elif 20 <= delta_e < 40:
                        style_suggestions.append({
                            'primary_color': color1,
                            'secondary_color': color2,
                            'style': 'Complementary'
                        })
                    else:
                        style_suggestions.append({
                            'primary_color': color1,
                            'secondary_color': color2,
                            'style': 'Triadic'
                        })
        return style_suggestions
