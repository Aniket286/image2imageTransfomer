import cv2
import numpy as np
import os

# Function to process a single image
def process_image(image_path, output_folder):
    # Read the image
    frame = cv2.imread(image_path)

    # Resize the image
    new_width = 256
    new_height = 256
    frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)

    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define range of red color in HSV
    lower_red = np.array([30, 150, 50])
    upper_red = np.array([255, 255, 180])

    # Create a red HSV color boundary and threshold HSV image
    mask = cv2.inRange(hsv, lower_red, upper_red)

    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(frame, frame, mask=mask)

    # Find edges in the input image and mark them in the output map
    edges = cv2.Canny(frame, 100, 200)
    edges = cv2.bitwise_not(edges)

    # Display the original image and edges (Optional)
    # cv2.imshow('Original', frame)
    # cv2.imshow('Edges', edges)
    # cv2.waitKey(0)

    # Save the processed images
    base_name = os.path.basename(image_path)
    name, ext = os.path.splitext(base_name)
    # cv2.imwrite(os.path.join(output_folder, f"{name}_resized{ext}"), frame)
    cv2.imwrite(os.path.join(output_folder, f"{name}{ext}"), edges)

# Directory containing images
input_folder = 'tshirt/'
output_folder = 'output/'

# Create output folder if it does not exist
os.makedirs(output_folder, exist_ok=True)

# List all image files in the directory
image_files = [f for f in os.listdir(input_folder) if f.endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

# Process each image
for image_file in image_files:
    image_path = os.path.join(input_folder, image_file)
    process_image(image_path, output_folder)

# Close all OpenCV windows
cv2.destroyAllWindows()
