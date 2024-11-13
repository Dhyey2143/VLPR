import cv2
import numpy as np

# Load the original image
image = cv2.imread('E:/Image processing VLPR/TestImage2.jpg')

# Convert the image to grayscale
grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow('Grayscale Image', grayscale_image)
cv2.waitKey(0)

# Compute gradients along the X and Y directions
grad_x = cv2.Sobel(grayscale_image, cv2.CV_64F, 1, 0, ksize=3)  # Gradient in the X direction
grad_y = cv2.Sobel(grayscale_image, cv2.CV_64F, 0, 1, ksize=3)  # Gradient in the Y direction

# Compute the gradient magnitude (combined)
gradient_magnitude = cv2.magnitude(grad_x, grad_y)

# Normalize the gradient magnitude to display
cv2.normalize(gradient_magnitude, gradient_magnitude, 0, 255, cv2.NORM_MINMAX)
gradient_magnitude = np.uint8(gradient_magnitude)

# Define the horizontal and vertical filters
filter_width = 30  # Adjust this based on the size of the license plate in your image
horizontal_filter = np.ones((1, filter_width))  # Horizontal line filter
filter_height = 8  # Adjust this based on the height of the license plate
vertical_filter = np.ones((filter_height, 1))  # Vertical line filter

# Apply horizontal and vertical filters
filtered_horizontal = cv2.filter2D(gradient_magnitude, -1, horizontal_filter)
filtered_vertical = cv2.filter2D(gradient_magnitude, -1, vertical_filter)

# Threshold to binary
_, binary_horizontal = cv2.threshold(filtered_horizontal, 128, 255, cv2.THRESH_BINARY)
_, binary_vertical = cv2.threshold(filtered_vertical, 128, 255, cv2.THRESH_BINARY)

# Apply morphological operations to reduce noise
kernel = np.ones((5, 5), np.uint8)
cleaned_horizontal = cv2.morphologyEx(binary_horizontal, cv2.MORPH_CLOSE, kernel)
cleaned_vertical = cv2.morphologyEx(binary_vertical, cv2.MORPH_CLOSE, kernel)

# Combine the horizontal and vertical filtered images (if needed)
combined_filtered = cv2.bitwise_or(cleaned_horizontal, cleaned_vertical)

# Find contours in the cleaned binary image
contours, _ = cv2.findContours(combined_filtered, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Loop through the contours and find potential license plate
for contour in contours:
    # Get the bounding box for each contour
    x, y, w, h = cv2.boundingRect(contour)
    
    # You can define criteria to filter the license plate based on size/shape
    aspect_ratio = w / h
    if 2 <= aspect_ratio <= 5:  # Typical aspect ratio range for license plates
        license_plate = image[y:y+h, x:x+w]  # Extract the region of interest (ROI)
        
        # Display the extracted license plate
        cv2.imshow('License Plate', license_plate)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Save the extracted license plate (optional)
        cv2.imwrite('extracted_license_plate.jpg', license_plate)

# Display the cleaned and filtered images for reference
cv2.imshow('Filtered Horizontal Binary Image', binary_horizontal)
cv2.imshow('Filtered Vertical Binary Image', binary_vertical)
cv2.imshow('Cleaned Horizontal Image', cleaned_horizontal)
cv2.imshow('Cleaned Vertical Image', cleaned_vertical)
cv2.imshow('Combined Filtered Image', combined_filtered)
cv2.waitKey(0)
cv2.destroyAllWindows()
