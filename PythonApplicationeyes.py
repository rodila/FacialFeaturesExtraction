import cv2
import numpy as np
from mtcnn import MTCNN
import webcolors

def get_main_eye_color(image_path):
    # Load an image
    img = cv2.imread(image_path)
    if img is None:
        print("Image not found or path is incorrect.")
        return None

    detector = MTCNN()
    faces = detector.detect_faces(img)

    left_eye_color = None
    right_eye_color = None

    for face in faces:
        # Detect eyes, extract regions, and find dominant colors
        for key in ['left_eye', 'right_eye']:
            eye_position = face['keypoints'][key]
            eye_region = get_iris_region(img, eye_position, 20)  # Adjust radius based on your requirement
            if eye_region.size != 0:  # Check if the eye region was correctly extracted
                # Get the pupil area, center, and bounding box
                pupil_area, pupil_center, pupil_bbox = get_pupil_area(eye_region, key == 'left_eye')
                if pupil_area > 0:
                    # Extract the pupil region from the eye
                    x, y, w, h = pupil_bbox
                    pupil_region = eye_region[y:y+h, x:x+w]
                    # Draw a green circle around the pupil area
                    cv2.circle(eye_region, pupil_center, w // 3, (0, 255, 0), 2)
                    # Create a mask to ignore the green circle
                    mask = np.zeros_like(pupil_region)
                    cv2.circle(mask, (w // 2, h // 2), w // 3, (255, 255, 255), -1)  # Circle in white
                    # Extract the dominant color within the circled area excluding the green circle
                    main_color = get_dominant_color(pupil_region, mask)
                    # Get the color name
                    color_name = get_color_name(main_color)
                    # Store the color for each eye
                    if key == 'left_eye':
                        left_eye_color = main_color
                    else:
                        right_eye_color = main_color

    # Calculate the average color between left and right eyes
    average_color = np.mean([left_eye_color, right_eye_color], axis=0)
    # Get the main eye color name
    main_eye_color = get_color_name(average_color)
    return main_eye_color

def get_dominant_color(image, mask=None, k=1):
    # Convert to RGB (OpenCV uses BGR by default)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Apply mask if provided
    if mask is not None:
        image = cv2.bitwise_and(image, mask)
    # Reshape the image to be a list of pixels
    pixels = np.float32(image.reshape(-1, 3))

    # Use k-means to find the most dominant color
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, palette = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Find the most frequent color
    dominant_color = palette[0].astype(int)
    return dominant_color

def closest_color(requested_color):
    min_colors = {}
    for key, name in webcolors.CSS3_HEX_TO_NAMES.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(key)
        rd = (r_c - requested_color[0]) ** 2
        gd = (g_c - requested_color[1]) ** 2
        bd = (b_c - requested_color[2]) ** 2
        min_colors[(rd + gd + bd)] = name
    return min_colors[min(min_colors.keys())]

def get_iris_region(img, eye_position, radius=20):
    # Extract the region around the eye
    x1, y1 = max(0, eye_position[0] - radius), max(0, eye_position[1] - radius)
    x2, y2 = eye_position[0] + radius, eye_position[1] + radius
    return img[y1:y2, x1:x2]

def get_pupil_area(eye_region, is_left_eye):
    # Convert to grayscale
    gray = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
    # Apply thresholding
    _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Get the largest contour
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        # Get the bounding rectangle
        x, y, w, h = cv2.boundingRect(largest_contour)
        # Calculate center
        center = (x + w // 2, y + h // 2)
        # Adjust position for left or right eye
        if is_left_eye:
            center = (center[0] + 2, center[1] + 3)  # Add 5 pixels to x-coordinate for left eye
        else:
            center = (center[0] - 2, center[1] + 3)  # Subtract 5 pixels from x-coordinate for right eye
        # Return the area, center, and bounding box
        return w * h, center, (x, y, w, h)
    else:
        return 0, None, None


def get_color_name(rgb):
    # Define the range of RGB values for different human eye colors
    colors = {
        "Brown": [(10, 10, 0), (190, 150, 120)],    # Brown thresholds
        "Green": [(0, 60, 0), (80, 180, 80)],       # Green thresholds
        "Blue": [(0, 40, 40), (100, 180, 180)],      # Blue thresholds
    }

    # Convert the input RGB values to a numpy array
    rgb = np.array(rgb)
    
    if rgb[2] > 35:
        return "Blue"

    # Initialize variables to store the closest color and its distance
    closest_color = None
    min_distance = float('inf')

    # Check each color range and find the closest match
    for color_name, (lower, upper) in colors.items():
        lower_bound = np.array(lower)
        upper_bound = np.array(upper)
        
        # Check if the RGB values fall within the range for the current color
        if np.all(rgb >= lower_bound) and np.all(rgb <= upper_bound):
            # Calculate the Euclidean distance between the input color and the center of the color range
            distance = np.linalg.norm((lower_bound + upper_bound) / 2 - rgb)
            
            # Update the closest color if the current distance is smaller
            if distance < min_distance:
                closest_color = color_name
                min_distance = distance

    # If no color range matches, return the color with the smallest distance
    if closest_color is None:
        closest_color = min(colors.keys(), key=lambda x: np.linalg.norm(np.mean(colors[x], axis=0) - rgb))

    return closest_color

   

