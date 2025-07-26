from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import json
import io
import logging
import traceback
from typing import Dict, List
import numpy as np
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Waste Classification API",
    description="AI-powered waste sorting and recycling classification",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model architecture (same as training)
class WasteClassifier(nn.Module):
    def __init__(self, num_classes=6):
        super(WasteClassifier, self).__init__()
        # Load EfficientNet-B0 with pretrained weights
        self.backbone = models.efficientnet_b0(pretrained=True)
        # Replace the classifier
        num_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(num_features, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)
        
    def forward(self, x):
        return self.backbone(x)

# Global variables for model and metadata
model = None
device = None
class_names = []
transform = None

# Recycling recommendations for each waste type
RECYCLING_INFO = {
    "cardboard": {
        "recyclable": True,
        "bin": "Recycling Bin",
        "tips": [
            "Remove all tape and staples",
            "Flatten boxes to save space",
            "Keep dry - wet cardboard goes to compost"
        ],
        "environmental_impact": "Recycling cardboard saves 1 cubic yard of landfill space per ton"
    },
    "glass": {
        "recyclable": True,
        "bin": "Recycling Bin (Glass)",
        "tips": [
            "Remove caps and lids",
            "Rinse containers clean",
            "Separate by color if required locally"
        ],
        "environmental_impact": "Glass can be recycled infinitely without quality loss"
    },
    "metal": {
        "recyclable": True,
        "bin": "Recycling Bin",
        "tips": [
            "Clean containers thoroughly",
            "Remove labels if possible",
            "Crush cans to save space"
        ],
        "environmental_impact": "Recycling aluminum cans saves 95% of energy vs. new production"
    },
    "paper": {
        "recyclable": True,
        "bin": "Recycling Bin",
        "tips": [
            "Remove plastic windows from envelopes",
            "Keep paper dry and clean",
            "No wax-coated paper"
        ],
        "environmental_impact": "Recycling 1 ton of paper saves 17 trees and 7000 gallons of water"
    },
    "plastic": {
        "recyclable": True,
        "bin": "Recycling Bin",
        "tips": [
            "Check recycling number (1-7)",
            "Clean containers thoroughly",
            "Remove caps if required locally"
        ],
        "environmental_impact": "Only 9% of plastic ever produced has been recycled"
    },
    "trash": {
        "recyclable": False,
        "bin": "General Waste",
        "tips": [
            "Double-check if any parts can be separated and recycled",
            "Consider if item can be repaired or donated",
            "Look for special disposal programs for electronics"
        ],
        "environmental_impact": "Reducing waste is better than recycling - consider reuse options"
    }
}
# Validate file type - check both content_type and file extension
def is_image_file(filename: str, content_type: str = None) -> bool:
    if content_type and content_type.startswith('image/'):
        return True
    
    # Check file extension as fallback
    if filename:
        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff'}
        file_ext = os.path.splitext(filename.lower())[1]
        return file_ext in image_extensions
    
    return False

def load_model():
    """Load the trained model and metadata"""
    global model, device, class_names, transform
    
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")
        
        # Check if model file exists
        if not os.path.exists('waste_classifier.pth'):
            raise FileNotFoundError("Model file 'waste_classifier.pth' not found")
        
        # Load model checkpoint
        checkpoint = torch.load('waste_classifier.pth', map_location=device, weights_only=False)        
        
        # Get metadata
        metadata = checkpoint['metadata']
        class_names = metadata['class_names']
        num_classes = metadata['num_classes']
        
        # Get model architecture from checkpoint (if available)
        model_arch = checkpoint.get('model_architecture', 'efficientnet')
        
        # Initialize model
        model = WasteClassifier(num_classes=num_classes)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        # Define preprocessing transform
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        logger.info(f"Model loaded successfully with {num_classes} classes: {class_names}")
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        logger.error(traceback.format_exc())
        raise e

def preprocess_image(image: Image.Image):
    """Preprocess image for model inference"""
    try:
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Apply transforms
        image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
        return image_tensor.to(device)
    
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        raise e

def predict_waste_type(image_tensor):
    """Make prediction on preprocessed image"""
    try:
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted_class = torch.max(probabilities, 1)
            
            # Get all class probabilities
            all_probs = probabilities[0].cpu().numpy()
            class_probabilities = {
                class_names[i]: float(all_probs[i]) 
                for i in range(len(class_names))
            }
            
            predicted_label = class_names[predicted_class.item()]
            confidence_score = float(confidence.item())
            
            return predicted_label, confidence_score, class_probabilities
    
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise e

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    try:
        load_model()
        logger.info("API startup completed successfully")
    except Exception as e:
        logger.error(f"Failed to start API: {str(e)}")
        raise e

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Waste Classification API",
        "version": "1.0.0",
        "endpoints": {
            "/classify": "POST - Upload image for waste classification",
            "/health": "GET - Health check",
            "/classes": "GET - Available waste classes"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device) if device else "not_set",
        "classes": len(class_names) if class_names else 0
    }

@app.get("/classes")
async def get_classes():
    """Get available waste classes and recycling info"""
    return {
        "classes": class_names,
        "recycling_info": RECYCLING_INFO
    }

@app.post("/classify")
async def classify_waste(file: UploadFile = File(...)):
    """
    Classify uploaded waste image
    """
    try:
        logger.info(f"Received file: {file.filename}, type: {file.content_type}")
        
        # Check if model is loaded
        if model is None:
            logger.error("Model is not loaded!")
            raise HTTPException(status_code=500, detail="Model not loaded")
        
        # Validate file type
        
# Replace the existing validation with:
        if not is_image_file(file.filename, file.content_type):
            raise HTTPException(status_code=400, detail="File must be an image")
    

        # Read and open image
        logger.info("Reading image file...")
        contents = await file.read()
        logger.info(f"File size: {len(contents)} bytes")
        
        image = Image.open(io.BytesIO(contents))
        logger.info(f"Image opened: {image.size}, mode: {image.mode}")
        
        # Preprocess image
        logger.info("Preprocessing image...")
        image_tensor = preprocess_image(image)
        logger.info(f"Image tensor shape: {image_tensor.shape}")
        
        # Make prediction
        logger.info("Making prediction...")
        predicted_class, confidence, all_probabilities = predict_waste_type(image_tensor)
        logger.info(f"Prediction complete: {predicted_class} with confidence {confidence}")
        
        # Get recycling information
        recycling_info = RECYCLING_INFO.get(predicted_class, {
            "recyclable": False,
            "bin": "Unknown",
            "tips": ["Please check with local waste management"],
            "environmental_impact": "Unknown impact"
        })
        
        # Prepare response
        response = {
            "prediction": {
                "class": predicted_class,
                "confidence": round(confidence, 4),
                "all_probabilities": {k: round(v, 4) for k, v in all_probabilities.items()}
            },
            "recycling_info": recycling_info,
            "recommendations": {
                "action": f"Place in {recycling_info['bin']}",
                "confidence_level": "High" if confidence > 0.8 else "Medium" if confidence > 0.6 else "Low",
                "tips": recycling_info.get("tips", [])
            }
        }
        
        logger.info(f"Classification successful: {predicted_class} ({confidence:.4f})")
        return JSONResponse(content=response)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Classification error: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")
@app.post("/classify/batch")
async def classify_batch(files: List[UploadFile] = File(...)):
    """
    Classify multiple waste images at once
    """
    if len(files) > 10:  # Limit batch size
        raise HTTPException(status_code=400, detail="Maximum 10 images per batch")
    
    results = []
    
    for i, file in enumerate(files):
        try:
            if not file.content_type.startswith('image/'):
                results.append({
                    "filename": file.filename,
                    "error": "File must be an image"
                })
                continue
            
            contents = await file.read()
            image = Image.open(io.BytesIO(contents))
            image_tensor = preprocess_image(image)
            predicted_class, confidence, all_probabilities = predict_waste_type(image_tensor)
            
            recycling_info = RECYCLING_INFO.get(predicted_class, {})
            
            results.append({
                "filename": file.filename,
                "prediction": {
                    "class": predicted_class,
                    "confidence": round(confidence, 4)
                },
                "recycling_info": recycling_info
            })
            
        except Exception as e:
            results.append({
                "filename": file.filename,
                "error": str(e)
            })
    
    return {"results": results}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)