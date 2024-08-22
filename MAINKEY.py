from PythonApplicationsize import extract_size_keywords
from PythonApplicationcolor import extract_color_keywords

text_input = "She has full lips, pronunced jawline, and petite nose and also has blue eyes, white skin, and black hair."

jawline, nose, lips = extract_size_keywords(text_input)
hair, skin, eyes = extract_color_keywords(text_input)

print("Jawline size:", jawline)
print("Nose size:", nose)
print("Lips size:", lips)
print("Hair color:", hair)
print("Skin color:", skin)
print("Eye color:", eyes)

