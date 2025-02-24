from fastapi import FastAPI, UploadFile, File, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from io import BytesIO
from PIL import Image
import numpy as np
import tensorflow as tf
import os

# Remove gTTS and playsound imports
# Keep Wikipedia and Translator for AI info
import wikipedia
from googletrans import Translator

app = FastAPI()

# Allow CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5500"],
    allow_credentials=True,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

MODEL_PATH = "D:/Projects section/Final year project/potato-project/saved_models/3"
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

# Load the model
if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"Model not found at {MODEL_PATH}")
MODEL = tf.keras.models.load_model(MODEL_PATH)

# Hindi messages mapping
HINDI_MESSAGES = {
    "Early Blight": "आलू में प्रारंभिक झुलसा रोग का पता चला है। कृपया कृषि विशेषज्ञ से सलाह लें।",
    "Late Blight": "आलू में देर से झुलसा रोग का पता चला है। तुरंत उपचार आवश्यक है।",
    "Healthy": "पौधे स्वस्थ हैं। कोई कार्रवाई आवश्यक नहीं है।"
}

@app.get("/ping")
async def ping():
    return {"message": "Hello, I am alive"}

def read_file_as_image(data) -> np.ndarray:
    try:
        image = Image.open(BytesIO(data)).convert("RGB")
        return np.array(image)
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid image format")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Process image
        image = read_file_as_image(await file.read())
        image = tf.image.resize(image, (256, 256))
        img_batch = np.expand_dims(image, 0)

        # Prediction
        prediction = MODEL.predict(img_batch)
        predicted_class = CLASS_NAMES[np.argmax(prediction[0])]
        confidence = np.max(prediction[0])

        # Hindi message
        hindi_message = HINDI_MESSAGES.get(predicted_class, "कृपया डॉक्टर से सलाह लें।")

        # English response
        english_response = f"{predicted_class} detected ({confidence:.2f} confidence). Please consult an agricultural expert."

        return {
            "class": predicted_class,
            "confidence": float(confidence),
            "message": english_response,
            "hindi_alert": hindi_message
        }

    except Exception as e:
        print("Prediction error:", str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/aiinfo")
async def ai_info(payload: dict = Body(...)):
    try:
        disease = payload.get("disease", "")
        if not disease:
            raise HTTPException(status_code=400, detail="No disease provided")
        
        query = disease + " potato"
        wikipedia.set_lang("en")
        summary = wikipedia.summary(query, sentences=2)
        
        translator = Translator()
        translation = translator.translate(summary, dest="hi")
        hindi_text = translation.text
        
        return {"summary": hindi_text}
    except Exception as e:
        print("AI info error:", str(e))
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)