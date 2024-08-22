import requests

API_URL = "https://api-inference.huggingface.co/models/londe33/hair_v02"
headers = {"Authorization": "Bearer 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"}

def query(filename):
    with open(filename, "rb") as f:
        data = f.read()
    response = requests.post(API_URL, headers=headers, data=data)
    return response.json()

def get_main_hair_color(filename):
    # Query Hugging Face model to get hair colors
    response = query(filename)
    
    # Find the hair color with the highest score
    main_hair_color = max(response, key=lambda x: x["score"])
    
       
       
    if main_hair_color["label"] == "blond hair":
          main_hair_color["label"] = "blonde hair"
    return main_hair_color


