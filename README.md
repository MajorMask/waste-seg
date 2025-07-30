# 🗑️♻️ WasteSort AI - Smart Waste Classifier

An AI-powered waste sorting and recycling classification system using computer vision. Built with Next.js, React, TypeScript, Tailwind CSS, and FastAPI backend.

![Demo](https://img.shields.io/badge/Demo-Live-brightgreen) ![React](https://img.shields.io/badge/React-18+-blue) ![TypeScript](https://img.shields.io/badge/TypeScript-5+-blue) ![Next.js](https://img.shields.io/badge/Next.js-14+-black) ![License](https://img.shields.io/badge/License-MIT-green)

## 🌟 Features

- **AI Classification**: EfficientNet-based model trained on TrashNet dataset
- **Modern UI**: Responsive React interface with Tailwind CSS
- **Real-time Processing**: Fast inference with confidence scores and animations
- **Recycling Guidance**: Detailed recycling instructions for each waste type
- **Drag & Drop**: Intuitive file upload with camera support
- **PWA Ready**: Progressive Web App capabilities
- **REST API**: FastAPI backend for classification services
- **TypeScript**: Full type safety and better developer experience

## 🎯 Supported Waste Categories

- 📦 **Cardboard** - Boxes, packaging materials
- 🍶 **Glass** - Bottles, jars, containers
- 🥫 **Metal** - Cans, aluminum, steel containers
- 📄 **Paper** - Documents, newspapers, magazines
- 🥤 **Plastic** - Bottles, containers, packaging
- 🗑️ **General Waste** - Non-recyclable items

## 🚀 Quick Start

### Prerequisites

- Node.js 18+ and npm
- Python 3.9+ (for backend)
- 4GB+ RAM (8GB recommended)

### 1. Clone & Setup

```bash
git clone https://github.com/MajorMask/waste-seg
cd waste-seg
```

### 2. Install Dependencies

```bash
# Install frontend dependencies
npm install

# Install backend dependencies (for AI model)
pip install -r requirements.txt
```

### 3. Setup Environment

```bash
# Create .env.local file
echo "NEXT_PUBLIC_API_URL=http://localhost:8000" > .env.local
```

### 4. Start Development

```bash
# Option 1: Use the run script (starts both frontend and backend)
chmod +x scripts/run.sh
./scripts/run.sh

# Option 2: Manual startup
# Terminal 1 - Start backend
python fastapi_backend.py

# Terminal 2 - Start frontend
npm run dev

# Option 3: Docker
docker-compose -f scripts/docker-compose.yml up --build
```

### 5. Access Application

- 🌐 **Frontend**: http://localhost:3000
- 📡 **API**: http://localhost:8000
- 📚 **API Docs**: http://localhost:8000/docs

## 📊 Technology Stack

### Frontend
- **Framework**: Next.js 14 with App Router
- **Language**: TypeScript
- **Styling**: Tailwind CSS
- **Animations**: Framer Motion
- **Icons**: Heroicons
- **Build Tool**: Webpack (via Next.js)

### Backend
- **API**: FastAPI (Python)
- **ML Model**: PyTorch + EfficientNet
- **Image Processing**: PIL, OpenCV

### Performance
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

### React/TypeScript Example

```typescript
import { api } from '@/lib/api';

const classifyImage = async (file: File) => {
  try {
    const result = await api.classifyImage(file);
    
    console.log(`Predicted: ${result.prediction.class}`);
    console.log(`Confidence: ${(result.prediction.confidence * 100).toFixed(1)}%`);
    console.log(`Recyclable: ${result.recycling_info.recyclable}`);
    console.log(`Disposal: ${result.recycling_info.bin}`);
    
    return result;
  } catch (error) {
    console.error('Classification failed:', error);
  }
};
```

### Frontend Development

```bash
# Start development server
npm run dev

# Build for production
npm run build

# Start production server
npm start

# Run linting
npm run lint

# Type checking
npm run type-check
```


## 📁 Project Structure

```
waste-seg/
├── app/                    # Next.js App Router
│   ├── layout.tsx         # Root layout
│   ├── page.tsx           # Main page component
│   └── globals.css        # Global styles
├── components/            # Reusable UI components
│   └── ui/               # Base UI components
├── lib/                  # Utility functions
│   ├── api.ts           # API client
│   └── utils.ts         # Helper functions
├── types/               # TypeScript type definitions
│   └── index.ts        # Shared interfaces
├── data/               # Dataset and training data
├── analytics/          # Model performance metrics
├── scripts/           # Build and deployment scripts
├── fastapi_backend.py # AI classification API
├── train.py          # Model training script
├── requirements.txt  # Python dependencies
├── package.json     # Node.js dependencies
├── tailwind.config.js # Tailwind CSS config
├── tsconfig.json     # TypeScript config
└── next.config.js    # Next.js configuration
```

## 🔧 Configuration

### Environment Variables

Create a `.env.local` file:

```env
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_APP_ENV=development
```

### Tailwind CSS

The project uses a custom Tailwind configuration with:
- Custom color palette
- Animation utilities
- Component classes
- Responsive design tokens

## 🚀 Deployment

### Production Build

```bash
# Build the application
npm run build

# Start production server
npm start
```

### Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose -f scripts/docker-compose.yml up --build -d

# Or build individual containers
docker build -t waste-classifier-frontend .
docker build -t waste-classifier-backend -f scripts/Dockerfile.backend .
```

### Environment Setup

For production deployment, ensure you have:
- Node.js 18+ runtime
- Python 3.9+ for the backend
- Sufficient memory for the ML model
- HTTPS configuration for PWA features

## 🔮 Future Enhancements

- **Mobile App**: React Native version
- **Batch Processing**: Multiple image classification
- **User Accounts**: Save classification history
- **Map Integration**: Waste density mapping with Google Maps API
- **IoT Integration**: Smart bin sensors and cameras
- **AR Features**: Augmented reality waste identification
- **Analytics Dashboard**: Environmental impact tracking

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- TrashNet dataset by Gary Thung and Mindy Yang
- EfficientNet architecture by Google Research
- Next.js team for the excellent framework
- Tailwind CSS for the utility-first styling approach