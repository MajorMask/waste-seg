import uvicorn
import socket
from contextlib import closing
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import json
import io
import os

app = FastAPI(title="WasteSort AI API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def find_free_port(start_port=8000, max_attempts=100):
    """Find a free port starting from start_port"""
    for port in range(start_port, start_port + max_attempts):
        try:
            with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
                sock.bind(('localhost', port))
                return port
        except OSError:
            continue
    raise RuntimeError(f"Could not find a free port in range {start_port}-{start_port + max_attempts}")

def save_port_info(port):
    """Save the port information to a file for frontend to discover"""
    port_info = {
        "port": port,
        "url": f"http://localhost:{port}",
        "status": "running"
    }
    
    # Save to multiple locations for frontend to find
    locations = [
        "backend_port.json",
        "public/backend_port.json",
        ".next/backend_port.json"
    ]
    
    for location in locations:
        try:
            os.makedirs(os.path.dirname(location), exist_ok=True) if os.path.dirname(location) else None
            with open(location, 'w') as f:
                json.dump(port_info, f)
        except:
            pass  # Ignore errors, some paths might not exist yet
    
    print(f"🌐 Backend URL saved: http://localhost:{port}")
    return port_info

# Your existing model code here (keeping it the same)
class WasteClassifier(nn.Module):
    def __init__(self, num_classes=6, model_name='efficientnet'):
        super(WasteClassifier, self).__init__()
        
        if model_name == 'efficientnet':
            import torchvision.models as models
            self.backbone = models.efficientnet_b0(weights=None)
            num_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(num_features, num_classes)
            )
        
    def forward(self, x):
        return self.backbone(x)

# Global variables
model = None
device = None
class_names = []
transform = None

RECYCLING_INFO = {
    "cardboard": {
        "recyclable": True,
        "bin": "Recycling Bin",
        "tips": ["Remove all tape and staples", "Flatten boxes to save space", "Keep dry - wet cardboard goes to compost"],
        "environmental_impact": "Recycling cardboard saves 1 cubic yard of landfill space per ton"
    },
    "glass": {
        "recyclable": True,
        "bin": "Recycling Bin (Glass)",
        "tips": ["Remove caps and lids", "Rinse containers clean", "Separate by color if required locally"],
        "environmental_impact": "Glass can be recycled infinitely without quality loss"
    },
    "metal": {
        "recyclable": True,
        "bin": "Recycling Bin",
        "tips": ["Clean containers thoroughly", "Remove labels if possible", "Crush cans to save space"],
        "environmental_impact": "Recycling aluminum cans saves 95% of energy vs. new production"
    },
    "paper": {
        "recyclable": True,
        "bin": "Recycling Bin",
        "tips": ["Remove plastic windows from envelopes", "Keep paper dry and clean", "No wax-coated paper"],
        "environmental_impact": "Recycling 1 ton of paper saves 17 trees and 7000 gallons of water"
    },
    "plastic": {
        "recyclable": True,
        "bin": "Recycling Bin",
        "tips": ["Check recycling number (1-7)", "Clean containers thoroughly", "Remove caps if required locally"],
        "environmental_impact": "Only 9% of plastic ever produced has been recycled"
    },
    "trash": {
        "recyclable": False,
        "bin": "General Waste",
        "tips": ["Double-check if any parts can be separated and recycled", "Consider if item can be repaired or donated", "Look for special disposal programs for electronics"],
        "environmental_impact": "Reducing waste is better than recycling - consider reuse options"
    }
}

def load_model():
    """Load the trained model and metadata"""
    global model, device, class_names, transform
    
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        # Load model checkpoint
        checkpoint = torch.load('waste_classifier.pth', map_location=device)
        
        # Get metadata
        metadata = checkpoint['metadata']
        class_names = metadata['class_names']
        num_classes = metadata['num_classes']
        
        # Initialize model
        model = WasteClassifier(num_classes=num_classes, model_name='efficientnet')
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
        
        print(f"Model loaded successfully with {num_classes} classes: {class_names}")
        
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise e

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    try:
        load_model()
        print("API startup completed successfully")
    except Exception as e:
        print(f"Failed to start API: {str(e)}")
        raise e

@app.get("/")
async def root():
    return {
        "message": "WasteSort AI API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "/classify": "POST - Upload image for classification",
            "/health": "GET - Health check",
            "/classes": "GET - Available waste classes"
        }
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device) if device else "not_set",
        "classes": len(class_names) if class_names else 0
    }

@app.get("/classes")
async def get_classes():
    return {
        "classes": class_names,
        "recycling_info": RECYCLING_INFO
    }

@app.post("/classify")
async def classify_waste(file: UploadFile = File(...)):
    try:
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted_class = torch.max(probabilities, 1)
            
            all_probs = probabilities[0].cpu().numpy()
            class_probabilities = {
                class_names[i]: float(all_probs[i]) 
                for i in range(len(class_names))
            }
            
            predicted_label = class_names[predicted_class.item()]
            confidence_score = float(confidence.item())
        
        recycling_info = RECYCLING_INFO.get(predicted_label, RECYCLING_INFO["trash"])
        
        response = {
            "prediction": {
                "class": predicted_label,
                "confidence": confidence_score,
                "all_probabilities": class_probabilities
            },
            "recycling_info": recycling_info,
            "recommendations": {
                "action": f"Place in {recycling_info['bin']}",
                "confidence_level": "High" if confidence_score > 0.8 else "Medium" if confidence_score > 0.6 else "Low",
                "tips": recycling_info.get("tips", [])
            }
        }
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")

if __name__ == "__main__":
    try:
        # Find an available port
        port = find_free_port(start_port=8000)
        print(f"Starting WasteSort AI API on port {port}")
        
        # Save port info for frontend discovery
        save_port_info(port)
        
        # Start the server
        uvicorn.run(app, host="0.0.0.0", port=port)
        
    except Exception as e:
        print(f"Failed to start server: {e}")
