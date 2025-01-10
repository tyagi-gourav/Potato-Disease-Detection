from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from io import BytesIO
from PIL import Image
import numpy as np
import tensorflow as tf
import io

app = FastAPI()

# Allow CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5500"],  # You can restrict this to your frontend URL
    allow_credentials=True,
    allow_methods=["POST"],
    allow_headers=["*"],
)

# Load the model
MODEL = tf.keras.models.load_model("D:/Final year project/potato-project/saved_models/1")

CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

@app.get("/ping")
async def ping():
    return {"message": "Hello, I am alive"}

def read_file_as_image(data) -> np.ndarray:
    try:
        image = Image.open(BytesIO(data))
        image = np.array(image)
        if image.ndim != 3 or image.shape[2] != 3:  # Validate if the image has 3 color channels
            raise ValueError("Image must be a 3-channel RGB image.")
        return image
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid image format")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read and process the image
        image = await file.read()
        image = read_file_as_image(image)
        
        print("Image Shape:", image.shape)  # Log the image shape for debugging
        
        # Ensure the image is resized to match the model's expected input size
        image_resized = tf.image.resize(image, (256, 256))
        img_batch = np.expand_dims(image_resized, 0)

        # Make prediction
        predictions = MODEL.predict(img_batch)
        predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
        confidence = np.max(predictions[0])

        # Return response
        return {"class": predicted_class, "confidence": float(confidence)}

    except Exception as e:
        print("Error during prediction:", str(e))
        raise HTTPException(status_code=500, detail="Failed to process the image")

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)
