import streamlit as st
import requests
import json
from PIL import Image
import io
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from typing import Dict, Any
import base64

# Page configuration
st.set_page_config(
    page_title="Smart Waste Classifier",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #2E8B57, #32CD32);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background: #f0f8ff;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #32CD32;
        margin: 1rem 0;
    }
    .recycling-info {
        background: #f5f5dc;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #FF6347;
        margin: 1rem 0;
    }
    .confidence-high { color: #008000; font-weight: bold; }
    .confidence-medium { color: #FF8C00; font-weight: bold; }
    .confidence-low { color: #DC143C; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# API Configuration
API_URL = "http://localhost:8002"

class WasteClassifierApp:
    def __init__(self):
        self.api_url = API_URL
        
    def check_api_health(self):
        """Check if API is running"""
        try:
            response = requests.get(f"{self.api_url}/health", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def get_classes(self):
        """Get available classes from API"""
        try:
            response = requests.get(f"{self.api_url}/classes", timeout=5)
            if response.status_code == 200:
                return response.json()
            return None
        except:
            return None
    
    def classify_image(self, image_file):
        """Send image to API for classification"""
        try:
            files = {"file": image_file}
            response = requests.post(f"{self.api_url}/classify", files=files, timeout=30)
            
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"API Error: {response.status_code}"}
        except requests.exceptions.RequestException as e:
            return {"error": f"Connection error: {str(e)}"}
    
    def render_header(self):
        """Render main header"""
        st.markdown("""
        <div class="main-header">
            <h1>♻️ Smart Waste Classifier</h1>
            <p>AI-Powered Recycling Assistant</p>
        </div>
        """, unsafe_allow_html=True)
    
    def render_sidebar(self):
        """Render sidebar with information"""
        with st.sidebar:
            st.header("ℹ️ About")
            st.markdown("""
            This AI system helps you properly sort waste for recycling using computer vision.
            
            **How it works:**
            1. Upload or capture an image
            2. AI analyzes the waste type
            3. Get recycling instructions
            
            **Supported Categories:**
            - 📦 Cardboard
            - 🍶 Glass
            - 🥫 Metal
            - 📄 Paper
            - 🥤 Plastic
            - 🗑️ General Waste
            """)
            
            st.header("🔧 API Status")
            if self.check_api_health():
                st.success("✅ API Connected")
            else:
                st.error("❌ API Disconnected")
                st.info("Make sure FastAPI server is running on localhost:8000")
    
    def render_confidence_badge(self, confidence: float):
        """Render confidence level badge"""
        if confidence > 0.8:
            css_class = "confidence-high"
            level = "High"
            emoji = "🟢"
        elif confidence > 0.6:
            css_class = "confidence-medium"
            level = "Medium"
            emoji = "🟡"
        else:
            css_class = "confidence-low"
            level = "Low"
            emoji = "🔴"
        
        return f'<span class="{css_class}">{emoji} {level} Confidence ({confidence:.1%})</span>'
    
    def render_probability_chart(self, probabilities: Dict[str, float]):
        """Render probability distribution chart"""
        df = pd.DataFrame(
            list(probabilities.items()),
            columns=['Waste Type', 'Probability']
        )
        df = df.sort_values('Probability', ascending=True)
        
        fig = px.bar(
            df, 
            x='Probability', 
            y='Waste Type',
            orientation='h',
            title="Classification Probabilities",
            color='Probability',
            color_continuous_scale='Viridis'
        )
        fig.update_layout(height=300)
        return fig
    
    def render_recycling_info(self, recycling_info: Dict[str, Any], waste_class: str):
        """Render recycling information"""
        emoji_map = {
            "cardboard": "📦",
            "glass": "🍶", 
            "metal": "🥫",
            "paper": "📄",
            "plastic": "🥤",
            "trash": "🗑️"
        }
        
        emoji = emoji_map.get(waste_class, "♻️")
        recyclable = recycling_info.get('recyclable', False)
        
        st.markdown(f"""
        <div class="recycling-info">
            <h3>{emoji} {waste_class.title()} - {"♻️ Recyclable" if recyclable else "🗑️ Not Recyclable"}</h3>
            <p><strong>📍 Disposal:</strong> {recycling_info.get('bin', 'Unknown')}</p>
        </div>
        """, unsafe_allow_html=True)
        
        if 'tips' in recycling_info:
            st.subheader("💡 Tips")
            for tip in recycling_info['tips']:
                st.write(f"• {tip}")
        
        if 'environmental_impact' in recycling_info:
            st.subheader("🌍 Environmental Impact")
            st.info(recycling_info['environmental_impact'])
    
    def run(self):
        """Main application"""
        self.render_header()
        self.render_sidebar()
        
        # Check API status
        if not self.check_api_health():
            st.error("🚨 API server is not running. Please start the FastAPI server first.")
            st.code("python fastapi_backend.py")
            return
        
        # Main content
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.header("📸 Upload Image")
            
            # Image input options
            input_option = st.radio(
                "Choose input method:",
                ["Upload File", "Take Photo", "Use Sample"]
            )
            
            uploaded_file = None
            
            if input_option == "Upload File":
                uploaded_file = st.file_uploader(
                    "Choose an image...",
                    type=['png', 'jpg', 'jpeg'],
                    help="Upload a clear image of the waste item"
                )
            
            elif input_option == "Take Photo":
                uploaded_file = st.camera_input("Take a picture of the waste item")
            
            elif input_option == "Use Sample":
                st.info("Sample images for testing (you can add sample images here)")
                # You can add sample images here for demo
            
            if uploaded_file is not None:
                # Display the image
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_container_width=True)
                
                # Classification button
                if st.button("🔍 Classify Waste", type="primary"):
                    with st.spinner("Analyzing image..."):
                        # Reset file pointer
                        uploaded_file.seek(0)
                        
                        # Get classification result
                        result = self.classify_image(uploaded_file)
                        
                        if "error" in result:
                            st.error(f"Error: {result['error']}")
                        else:
                            # Store result in session state for the second column
                            st.session_state['classification_result'] = result
        
        with col2:
            st.header("🎯 Classification Results")
            
            # Display results if available
            if 'classification_result' in st.session_state:
                result = st.session_state['classification_result']
                prediction = result['prediction']
                recycling_info = result['recycling_info']
                
                # Main prediction
                st.markdown(f"""
                <div class="prediction-box">
                    <h2>Predicted Class: {prediction['class'].title()}</h2>
                    <p>{self.render_confidence_badge(prediction['confidence'])}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Probability chart
                fig = self.render_probability_chart(prediction['all_probabilities'])
                st.plotly_chart(fig, use_container_width=True)
                
                # Recycling information
                self.render_recycling_info(recycling_info, prediction['class'])
                
                # Recommendations
                recommendations = result.get('recommendations', {})
                if recommendations:
                    st.subheader("📋 Action Required")
                    st.write(f"**Action:** {recommendations.get('action', 'Unknown')}")
                    
                    if 'tips' in recommendations:
                        for tip in recommendations['tips']:
                            st.write(f"• {tip}")
            
            else:
                st.info("👆 Upload an image to see classification results")
        
        # Additional features
        st.header("📊 Batch Processing")
        batch_files = st.file_uploader(
            "Upload multiple images for batch classification",
            type=['png', 'jpg', 'jpeg'],
            accept_multiple_files=True,
            help="Upload up to 10 images for batch processing"
        )
        
        if batch_files and len(batch_files) > 0:
            if st.button("🔄 Process Batch"):
                if len(batch_files) > 10:
                    st.error("Maximum 10 images allowed per batch")
                else:
                    with st.spinner("Processing batch..."):
                        # Process batch (implement API call)
                        st.success(f"Processing {len(batch_files)} images...")
                        # Add batch processing logic here

def main():
    app = WasteClassifierApp()
    app.run()

if __name__ == "__main__":
    main()