import cv2
import dlib
from mtcnn import MTCNN

def categorize_size(size, thresholds):
    if size < thresholds[0]:
        return "Small"
    elif size < thresholds[1]:
        return "Medium"
    else:
        return "Large"

def detect_face_landmarks(image):

    # Initialize MTCNN detector with adjusted min_face_size
    detector = MTCNN(min_face_size=40)  # Adjust min_face_size as needed


    # Detect faces using MTCNN
    result = detector.detect_faces(image)

    if result:
        # Get the first face
        bbox = result[0]['box']
        x, y, w, h = bbox

        # Extract face region
        face = image[y:y+h, x:x+w]

        # Convert the face region to grayscale
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        # Initialize dlib's face detector and shape predictor
        detector_dlib = dlib.get_frontal_face_detector()
        predictor_dlib = dlib.shape_predictor(r"C:\Users\Utilizator\OneDrive\Desktop\python\Cod_Apelator\PythonApplication\shape_predictor_68_face_landmarks.dat")  # Update with the path to your shape predictor

        # Detect faces in the grayscale image using dlib
        faces_dlib = detector_dlib(gray)

        if faces_dlib:
            # Assuming only one face is detected
            face_dlib = faces_dlib[0]

            # Predict facial landmarks for the detected face using dlib
            landmarks = predictor_dlib(gray, face_dlib)

            # Extract the coordinates of the lips landmarks
            lips_top = (landmarks.part(50).x, landmarks.part(50).y)  # Index 50 corresponds to the left side of the upper lip
            lips_bottom = (landmarks.part(58).x, landmarks.part(58).y)  # Index 58 corresponds to the left side of the lower lip

            # Calculate the lips size relative to face height
            lips_size = (lips_bottom[1] - lips_top[1]) / h

            # Calculate other face features (nose size, jawline size) relative to face width and height
            nose_size = w / h  # Normalized nose size
            jawline_size = (w + h) / (2 * w)  # Normalized jawline size

            return nose_size, lips_size, jawline_size
        else:
            return None, None, None
    else:
        return None, None, None

def analyze_face(image_path):
    image = cv2.imread(image_path)
    nose_size, lips_size, jawline_size = detect_face_landmarks(image)
    
    if nose_size is not None:
        # Define thresholds for size categories
        nose_thresholds = (0.7, 0.83)  # Example threshold values for normalized nose size
        lips_thresholds = (0.12, 0.15)  # Example threshold values for normalized lips size
        jawline_thresholds = (1.0, 1.15)  # Example threshold values for normalized jawline size

        # Categorize sizes
        nose_category = categorize_size(nose_size, nose_thresholds)
        lips_category = categorize_size(lips_size, lips_thresholds)
        jawline_category = categorize_size(jawline_size, jawline_thresholds)

        return nose_category, lips_category, jawline_category

# If you want to call this function from another program, you can do it like this:
# analyze_face(r"C:\Users\Utilizator\OneDrive\Desktop\imagini-test-comparare\00.png")


