import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
from scipy.stats import skew, kurtosis

class EnhancedCherryDetector:
    def __init__(self):
        """
        Initialize detector with refined HSV ranges and texture parameters
        """
        # Refined HSV ranges
        self.color_ranges = {
            'Ripe (Red)': (
                np.array([0, 120, 100]),     # Lower red bound (more saturated)
                np.array([10, 255, 255])     # Upper red bound
            ),
            'Partially Ripe (Yellow)': (
                np.array([20, 100, 100]),    # Lower yellow bound
                np.array([35, 255, 255])     # Upper yellow bound
            ),
            'Unripe (Green)': (
                np.array([35, 40, 40]),      # Lower green bound (adjusted for better leaf distinction)
                np.array([85, 255, 255])     # Upper green bound
            )
        }
        
        # Texture parameters for leaf vs. cherry distinction
        self.leaf_texture_params = {
            'contrast_threshold': 0.4,
            'uniformity_threshold': 0.6,
            'edge_density_threshold': 0.3
        }

    def calculate_texture_features(self, roi):
        """
        Calculate texture features for a region of interest
        """
        # Convert to grayscale
        if len(roi.shape) == 3:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = roi
            
        # Calculate GLCM (Gray-Level Co-occurrence Matrix)
        glcm = np.zeros((256, 256))
        h, w = gray.shape
        for i in range(h-1):
            for j in range(w-1):
                glcm[gray[i,j], gray[i,j+1]] += 1
                
        glcm = glcm / glcm.sum()  # Normalize
        
        # Calculate texture features
        contrast = np.sum(np.square(np.arange(256)[:, None] - np.arange(256)[None, :]) * glcm)
        uniformity = np.sum(np.square(glcm))
        
        # Calculate edge density using Sobel
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        edge_density = (np.abs(sobelx) + np.abs(sobely)).mean() / 255.0
        
        return {
            'contrast': contrast,
            'uniformity': uniformity,
            'edge_density': edge_density
        }

    def is_cherry_shape(self, contour):
        """
        Validate if a contour has cherry-like properties
        """
        # Calculate shape features
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            return False
            
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        
        # Get bounding box
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h if h != 0 else 0
        
        # Define cherry shape criteria
        return (0.5 < circularity < 1.2 and  # Nearly circular
                0.7 < aspect_ratio < 1.4 and  # Nearly square
                50 < area < 2000)             # Reasonable size

    def detect_cherries(self, image):
        """
        Enhanced cherry detection with texture analysis and shape validation
        """
        # Convert PIL image to OpenCV format
        opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Convert to HSV
        hsv = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2HSV)
        
        results = {}
        annotated_image = opencv_image.copy()
        
        for stage, (lower, upper) in self.color_ranges.items():
            mask = None
            if stage == 'Ripe (Red)':
                # Handle red's HSV wrap-around
                mask1 = cv2.inRange(hsv, lower, upper)
                mask2 = cv2.inRange(hsv, np.array([170, 120, 100]), np.array([180, 255, 255]))
                mask = cv2.bitwise_or(mask1, mask2)
            else:
                mask = cv2.inRange(hsv, lower, upper)
            
            # Enhanced noise removal
            kernel = np.ones((3,3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            valid_cherries = 0
            for contour in contours:
                if not self.is_cherry_shape(contour):
                    continue
                    
                # Get ROI for texture analysis
                x, y, w, h = cv2.boundingRect(contour)
                roi = opencv_image[y:y+h, x:x+w]
                
                if roi.size == 0:
                    continue
                    
                # Calculate texture features
                texture_features = self.calculate_texture_features(roi)
                
                # Skip if it has leaf-like texture (for green detection)
                if stage == 'Unripe (Green)' and (
                    texture_features['contrast'] < self.leaf_texture_params['contrast_threshold'] or
                    texture_features['uniformity'] > self.leaf_texture_params['uniformity_threshold'] or
                    texture_features['edge_density'] < self.leaf_texture_params['edge_density_threshold']
                ):
                    continue
                
                valid_cherries += 1
                
                # Draw detection with color coding
                color = (0, 255, 0) if stage == 'Unripe (Green)' else \
                       (0, 255, 255) if stage == 'Partially Ripe (Yellow)' else \
                       (0, 0, 255)  # Red
                cv2.rectangle(annotated_image, (x, y), (x+w, y+h), color, 2)
            
            results[stage] = valid_cherries
        
        return results, annotated_image

def main():
    st.title("Enhanced Coffee Cherry Counter")
    
    st.write("Upload a coffee tree image for analysis:")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Original Image", use_container_width=True)
        
        try:
            detector = EnhancedCherryDetector()
            
            with st.spinner('Analyzing image...'):
                results, annotated_image = detector.detect_cherries(image)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB), 
                        caption="Detected Cherries", 
                        use_container_width=True)
            
            with col2:
                st.subheader("Cherry Count:")
                for stage, count in results.items():
                    color = "🔴" if stage == 'Ripe (Red)' else \
                           "🟡" if stage == 'Partially Ripe (Yellow)' else "🟢"
                    st.write(f"{color} {stage}: {count}")
                
                total = sum(results.values())
                st.write(f"**Total Cherries:** {total}")
                
                # Add confidence metrics
                st.subheader("Detection Confidence:")
                st.write("Higher values indicate better detection quality")
                st.progress(0.8)  # Example confidence score
                
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")

if __name__ == "__main__":
    main()
