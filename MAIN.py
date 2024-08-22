from PythonApplicationpar import get_main_hair_color
from PythonApplicationeyes import get_main_eye_color
from PythonApplicationpiele import get_main_skin_color
from PythonApplicationface import analyze_face
# Define the image path
image_path = r"C:\Users\Utilizator\OneDrive\Desktop\imagini-test-comparare\00.png"

# Call the functions with the same image path
main_hair_color = get_main_hair_color(image_path)
main_eye_color = get_main_eye_color(image_path)
main_skin_color = get_main_skin_color(image_path)

# Call analyze_face function
nose_category, lips_category, jawline_category, = analyze_face(image_path)

# Print the results
print("Main hair color:", main_hair_color["label"])
print("Main eye color:", main_eye_color)
print("Main skin color:", main_skin_color)
print("Nose category:", nose_category)
print("Lips category:", lips_category)
print("Jawline category:", jawline_category)

class FaceAnalysisResult:
    def __init__(self, hair_color, eye_color, skin_color, nose_category, lips_category, jawline_category):
        self.hair_color = hair_color
        self.eye_color = eye_color
        self.skin_color = skin_color
        self.nose_category = nose_category
        self.lips_category = lips_category
        self.jawline_category = jawline_category

    def to_dict(self):
        return {
            "hair_color": self.hair_color,
            "eye_color": self.eye_color,
            "skin_color": self.skin_color,
            "nose_category": self.nose_category,
            "lips_category": self.lips_category,
            "jawline_category": self.jawline_category
        }


# Create an instance of FaceAnalysisResult
result = FaceAnalysisResult(main_hair_color["label"], main_eye_color, main_skin_color, nose_category, lips_category, jawline_category)

# Convert the result to a dictionary
result_dict = result.to_dict()

# Now you can send result_dict to the server
# Example of sending the data to a server:
# send_to_server(result_dict)
