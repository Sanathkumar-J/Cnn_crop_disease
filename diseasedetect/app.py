import torch
from torchvision import transforms
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from model import ImprovedCNN
import io

# Initialize FastAPI app
app = FastAPI()

# Enable CORS (for frontend integration)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model
model = ImprovedCNN(num_classes=10).to(device)
model.load_state_dict(torch.load("rice_leaf_model.pth", map_location=device))
model.eval()

# Class labels (must match training)
class_labels = {
    0: 'bacterial_leaf_blight',
    1: 'brown_spot',
    2: 'healthy',
    3: 'leaf_blast',
    4: 'leaf_scald',
    5: 'narrow_brown_spot',
    6: 'neck_blast',
    7: 'rice_hispa',
    8: 'sheath_blight',
    9: 'tungro'
}

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# Prediction function
def predict_image_from_bytes(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)
        pred_idx = output.argmax(1).item()
    return class_labels[pred_idx]

# Root check
@app.get("/")
def root():
    return {"message": "Rice Leaf Disease Detection API is running."}

# Main prediction endpoint
@app.post("/model")
async def model_predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        prediction = predict_image_from_bytes(contents)
        return JSONResponse(content={"prediction": prediction})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
