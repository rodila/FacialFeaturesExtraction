import cv2
import numpy as np
from mtcnn import MTCNN
import dlib
import webcolors

def get_main_skin_color(image_path):
    # Load the input image
    image = cv2.imread(image_path)

    # Detect faces in the image using MTCNN
    detector = MTCNN()
    faces = detector.detect_faces(image)
    face_boxes = [face['box'] for face in faces]

    # Detect and highlight skin areas within face bounding boxes
    skin_highlighted_image, skin_mask = detect_skin(image, face_boxes)

    # Get the dominant color from the skin-highlighted image within the skin mask
    dominant_color = get_dominant_color(skin_highlighted_image, skin_mask)

    # Determine the color name of the dominant color
    color_name = get_color_name(dominant_color)

    return color_name

def detect_skin(image, face_boxes):
    # Convert image to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define lower and upper bounds for skin color in HSV (excluding lips)
    lower_skin = np.array([0, 10, 60], dtype=np.uint8)
    upper_skin = np.array([15, 135, 220], dtype=np.uint8)

    # Create a binary mask for skin regions
    skin_mask = np.zeros_like(hsv_image[:, :, 0], dtype=np.uint8)
    for face_box in face_boxes:
        x, y, w, h = face_box
        face_region = hsv_image[y:y+h, x:x+w]

        # Threshold the HSV image to get a binary mask for skin regions
        skin_region_mask = cv2.inRange(face_region, lower_skin, upper_skin)

        # Perform morphological operations to refine the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        skin_region_mask = cv2.morphologyEx(skin_region_mask, cv2.MORPH_CLOSE, kernel)
        skin_region_mask = cv2.morphologyEx(skin_region_mask, cv2.MORPH_OPEN, kernel)

        # Exclude the regions corresponding to eyes, mouth, and hair
        skin_region_mask = exclude_eye_mouth_hair(skin_region_mask, image[y:y+h, x:x+w])

        # Add the skin region mask to the overall skin mask
        skin_mask[y:y+h, x:x+w] = cv2.bitwise_or(skin_mask[y:y+h, x:x+w], skin_region_mask)

    # Apply the skin mask to the original image to highlight skin areas
    skin_highlighted_image = cv2.bitwise_and(image, image, mask=skin_mask)

    return skin_highlighted_image, skin_mask

def exclude_eye_mouth_hair(mask, image):
    # Initialize Dlib's face detector and shape predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(r"C:\Users\Utilizator\OneDrive\Desktop\python\Cod_Apelator\PythonApplication\shape_predictor_68_face_landmarks.dat")

    # Detect faces in the grayscale image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        # Predict facial landmarks
        shape = predictor(gray, face)

        # Extract coordinates of eyes, mouth, and eyebrows
        left_eye = shape.parts()[36:42]  # Left eye landmarks
        right_eye = shape.parts()[42:48]  # Right eye landmarks
        mouth = shape.parts()[48:68]  # Mouth landmarks
        eyebrows = shape.parts()[17:27]  # Eyebrow landmarks

        # Create binary masks for eyes, mouth, and eyebrows regions
        eye_mask = np.zeros_like(mask)
        mouth_mask = np.zeros_like(mask)
        eyebrow_mask = np.zeros_like(mask)

        # Fill eye, mouth, and eyebrow regions with black in their respective masks
        cv2.fillPoly(eye_mask, [np.array([(pt.x, pt.y) for pt in left_eye], dtype=np.int32)], 255)
        cv2.fillPoly(eye_mask, [np.array([(pt.x, pt.y) for pt in right_eye], dtype=np.int32)], 255)
        cv2.fillPoly(mouth_mask, [np.array([(pt.x, pt.y) for pt in mouth], dtype=np.int32)], 255)
        
        # Exclude mouth region from the eyebrow mask to prevent overfitting
        cv2.fillPoly(eyebrow_mask, [np.array([(pt.x, pt.y) for pt in eyebrows], dtype=np.int32)], 255)
        cv2.fillPoly(eyebrow_mask, [np.array([(pt.x, pt.y) for pt in mouth], dtype=np.int32)], 0)

        # Exclude eye, mouth, and eyebrow regions from the mask
        mask = cv2.bitwise_and(mask, cv2.bitwise_not(eye_mask))
        mask = cv2.bitwise_and(mask, cv2.bitwise_not(mouth_mask))
        mask = cv2.bitwise_and(mask, cv2.bitwise_not(eyebrow_mask))

    return mask

def get_dominant_color(image, mask):
    # Reshape the image to a 2D array of pixels
    pixels = image[mask > 0].reshape((-1, 3))

    # Calculate the histogram of colors
    histogram = np.zeros((256, 256, 256))
    for pixel in pixels:
        histogram[pixel[0], pixel[1], pixel[2]] += 1

    max_count = np.max(histogram)
    r, g, b = np.where(histogram == max_count)

    return [r[0], g[0], b[0]]

def get_color_name(rgb): 
    if isinstance(rgb, list):
        rgb_list = rgb  # Use the provided list
    else:
        rgb_list = rgb.tolist()  # Convert NumPy array to list if needed

    if  rgb_list[0] > 230 and rgb_list[1] > 180 and rgb_list[2] > 150:
        return "Light Skin"
    elif 170 < rgb_list[0] < 220 and 90 < rgb_list[1] < 160 and 70 < rgb_list[2] < 140:
        return "Medium Skin"
    elif 90 < rgb_list[0] < 170 and 30 < rgb_list[1] < 110 and 20 < rgb_list[2] < 90:
        return "Dark Skin"
    else:
        # Determine the closest category based on Euclidean distance
        light_skin_rgb = [230, 180, 150]
        medium_skin_rgb = [195, 125, 105]
        dark_skin_rgb = [60, 35, 30]

        distances = {
            "Light Skin": sum((x - y) ** 2 for x, y in zip(rgb_list, light_skin_rgb)),
            "Medium Skin": sum((x - y) ** 2 for x, y in zip(rgb_list, medium_skin_rgb)),
            "Dark Skin": sum((x - y) ** 2 for x, y in zip(rgb_list, dark_skin_rgb))
        }

        return min(distances, key=distances.get)



