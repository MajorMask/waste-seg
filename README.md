# 🗑️♻️ Smart Waste Classifier

An AI-powered waste sorting and recycling classification system using computer vision. Built with PyTorch, FastAPI, and Streamlit.

![Demo](https://img.shields.io/badge/Demo-Live-brightgreen) ![Python](https://img.shields.io/badge/Python-3.9+-blue) ![License](https://img.shields.io/badge/License-MIT-green)

## 🌟 Features

- **AI Classification**: EfficientNet-based model trained on TrashNet dataset
- **Real-time Processing**: Fast inference with confidence scores
- **Recycling Guidance**: Detailed recycling instructions for each waste type
- **Multiple Interfaces**: Web app, mobile PWA, and REST API
- **Batch Processing**: Handle multiple images simultaneously
- **Easy Deployment**: Docker, cloud-ready configurations

## 🎯 Supported Waste Categories

- 📦 **Cardboard** - Boxes, packaging materials
- 🍶 **Glass** - Bottles, jars, containers
- 🥫 **Metal** - Cans, aluminum, steel containers
- 📄 **Paper** - Documents, newspapers, magazines
- 🥤 **Plastic** - Bottles, containers, packaging
- 🗑️ **General Waste** - Non-recyclable items

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- 4GB+ RAM (8GB recommended)
- GPU optional (for faster training)

### 1. Clone & Setup

```bash
git clone https://github.com/yourusername/smart-waste-classifier
cd smart-waste-classifier
pip install -r requirements.txt
```

### 2. Download TrashNet Dataset

```bash
# Download from: https://github.com/garythung/trashnet
# Extract to ./trashnet-dataset/
```

### 3. Train Model

```bash
# Update data path in data_pipeline.py
python data_pipeline.py
python model_training.py
```

### 4. Run Application

```bash
# Option 1: Use run script (recommended)
chmod +x run.sh
./run.sh

# Option 2: Manual startup
python fastapi_backend.py &
streamlit run streamlit_app.py

# Option 3: Docker
docker-compose up --build
```

### 5. Access Interfaces

- 🌐 **Web App**: http://localhost:8501
- 📡 **API**: http://localhost:8000
- 📚 **API Docs**: http://localhost:8000/docs
- 📱 **Mobile PWA**: http://localhost/mobile_app.html

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| **Accuracy** | 85-92% |
| **Inference Time** | <200ms |
| **Model Size** | ~20MB |
| **Classes** | 6 waste types |

## 🛠️ API Usage

### Classify Single Image

```bash
curl -X POST "http://localhost:8000/classify" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@waste_image.jpg"
```

### Python Example

```python
import requests

url = "http://localhost:8000/classify"
files = {"file": open("waste_image.jpg", "rb")}
response = requests.post(url, files=files)
result = response.json()

print(f"Predicted: {result['prediction']['class']}")
print(f"Confidence: {result['prediction']['confidence']:.2%}")
print(f"Recyclable: {result['recycling_info']['recyclable']}")
```

### JavaScript Example

```javascript
const formData = new FormData();
formData.append('file', imageFile);

fetch('http://localhost:8000/classify', {
    method: 'POST',
    body: formData
})
.then(response => response.json())
.then(data => {
    console.log('Prediction:', data.prediction.class);
    console.log('Confidence:', data.prediction.confidence);
});
```


TO-DO: Waste density on streets, adding google maps API and cameras to find the intensity of waste outside the bin and map it in AIS with the use of ArcGIS or something.

## 🏗️ Architecture

```
┌────────────────