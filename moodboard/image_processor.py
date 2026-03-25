import numpy as np
from sklearn.cluster import KMeans
from PIL import Image
import colorsys

class ColorPaletteExtractor:
    def __init__(self, n_colors=5):
        self.n_colors = n_colors
        self.kmeans = KMeans(n_clusters=n_colors, random_state=42)
        
    def extract_colors(self, image_path):
        '''Extract dominant colors from an image using K-means clustering'''
        # Load and preprocess image
        img = Image.open(image_path)
        img = img.convert('RGB')
        pixels = np.float32(img).reshape(-1, 3)
        
        # Fit K-means
        self.kmeans.fit(pixels)
        colors = self.kmeans.cluster_centers_
        
        # Convert to hex and sort by prominence
        hex_colors = []
        counts = np.bincount(self.kmeans.labels_)
        percentages = counts / len(self.kmeans.labels_)
        
        for color, pct in zip(colors, percentages):
            rgb = np.round(color).astype(int)
            hex_color = '#{:02x}{:02x}{:02x}'.format(rgb[0], rgb[1], rgb[2])
            hex_colors.append({
                'hex': hex_color,
                'percentage': float(pct),
                'rgb': rgb.tolist()
            })
            
        # Sort by percentage
        hex_colors.sort(key=lambda x: x['percentage'], reverse=True)
        
        return hex_colors
    
    def get_color_harmony(self, hex_color):
        '''Generate harmonious color combinations'''
        # Convert hex to HSV
        hex_color = hex_color.lstrip('#')
        rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        hsv = colorsys.rgb_to_hsv(rgb[0]/255, rgb[1]/255, rgb[2]/255)
        
        harmonies = {
            'complementary': [(hsv[0] + 0.5) % 1],
            'triadic': [(hsv[0] + 1/3) % 1, (hsv[0] + 2/3) % 1],
            'analogous': [(hsv[0] + 0.1) % 1, (hsv[0] - 0.1) % 1]
        }
        
        result = {}
        for harmony_type, h_values in harmonies.items():
            colors = []
            for h in h_values:
                rgb = colorsys.hsv_to_rgb(h, hsv[1], hsv[2])
                rgb_int = tuple(int(x * 255) for x in rgb)
                hex_value = '#{:02x}{:02x}{:02x}'.format(*rgb_int)
                colors.append(hex_value)
            result[harmony_type] = colors
            
        return result

def process_image(image_path):
    '''Process an image to extract color information'''
    extractor = ColorPaletteExtractor()
    colors = extractor.extract_colors(image_path)
    
    # Get harmonious colors for the dominant color
    harmonies = extractor.get_color_harmony(colors[0]['hex'])
    
    return {
        'dominant_colors': colors,
        'color_harmonies': harmonies
    }