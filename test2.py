import cv2
import numpy as np
import pytesseract
import sqlite3

# Configure Tesseract path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Function to detect license plate
def detect_license_plate(image_path):
    # Load the image
    image = cv2.imread(image_path)
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply bilateral filter to reduce noise while keeping edges sharp
    filtered = cv2.bilateralFilter(gray, 11, 17, 17)
    
    # Detect edges
    edged = cv2.Canny(filtered, 30, 200)
    
    # Find contours
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    
    license_plate = None
    license_plate_contour = None

    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.018 * perimeter, True)
        
        if len(approx) == 4:  # Assuming license plate is rectangular
            license_plate_contour = approx
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / float(h)
            
            # Adjust aspect ratio range based on the example image
            if 2.5 <= aspect_ratio <= 5.5:
                license_plate = gray[y:y+h, x:x+w]
                break

    if license_plate is not None:
        # Apply thresholding to preprocess the image
        _, license_plate = cv2.threshold(license_plate, 64, 255, cv2.THRESH_BINARY_INV)

        # Remove small noise
        kernel = np.ones((3,3), np.uint8)
        license_plate = cv2.morphologyEx(license_plate, cv2.MORPH_OPEN, kernel)

    return image, license_plate, license_plate_contour

# Function to extract text using OCR
def extract_text_from_plate(license_plate_image):
    # Apply OCR to extract text from the processed license plate image
    text = pytesseract.image_to_string(license_plate_image, config='--psm 8').strip()
    return text

# Function to display results
def display_results(image, license_plate, license_plate_contour):
    if license_plate_contour is not None:
        cv2.drawContours(image, [license_plate_contour], -1, (0, 255, 0), 3)
    
    cv2.imshow('Original Image', image)
    if license_plate is not None:
        cv2.imshow('License Plate', license_plate)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Function to verify license plate from the database
def verify_license_plate(plate_text):
    # Connect to the database
    conn = sqlite3.connect('society_vehicles.db')
    cursor = conn.cursor()

    # Query to check if the license plate exists in the database
    cursor.execute("SELECT * FROM vehicle_owners WHERE license_plate = ?", (plate_text,))
    result = cursor.fetchone()
    conn.close()

    if result:
        return f"Authorized Vehicle. Owner: {result[0]}, Apartment: {result[2]}"
    else:
        return "Unauthorized Vehicle"

# Main script
image_path = 'E:/Image processing VLPR/TestImage4.jpg'  # Replace with your image path
original_image, detected_plate, plate_contour = detect_license_plate(image_path)

if detected_plate is not None:
    display_results(original_image, detected_plate, plate_contour)

    # Extract text from the detected license plate
    extracted_text = extract_text_from_plate(detected_plate)
    print("Extracted License Plate Text:", extracted_text)

    # Verify the license plate against the database
    verification_result = verify_license_plate(extracted_text)
    print(verification_result)
else:
    print("No license plate detected.")
