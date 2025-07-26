#!/bin/bash

# Smart Waste Classifier - One-Click Installer
# This script installs and sets up the complete waste classification system

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ASCII Art Logo
echo -e "${GREEN}"
cat << "EOF"
 _____ _____ _____ _____ _____   _   _ _____ _____ _____ _____ 
|   __|     |  _  | __  |_   _| | | | |  _  |   __|_   _|   __|
|__   | | | |     |    -| | |   | | | |     |__   | | | |   __|
|_____|_|_|_|__|__|__|__| |_|   |_____|__|__|_____| |_| |_____|
    
     _____ __    _____ _____ _____ _____ _____ _____ _____ 
    |     |  |  |  _  |   __|   __|     |   __|     |   __|
    |   --|  |__|     |__   |__   |-   -|   __|-   -|   __|
    |_____|_____|__|__|_____|_____|_____|__|  |_____|_____|
                                                          
EOF
echo -e "${NC}"

echo -e "${BLUE}🚀 Smart Waste Classifier Installer${NC}"
echo -e "${BLUE}=====================================${NC}"
echo ""

# Function to print status
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to detect OS
detect_os() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        echo "linux"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        echo "macos"
    elif [[ "$OSTYPE" == "cygwin" ]]; then
        echo "windows"
    elif [[ "$OSTYPE" == "msys" ]]; then
        echo "windows"
    else
        echo "unknown"
    fi
}

# System requirements check
check_requirements() {
    print_status "Checking system requirements..."
    
    OS=$(detect_os)
    print_status "Detected OS: $OS"
    
    # Check Python
    if command_exists python3; then
        PYTHON_VERSION=$(python3 --version | cut -d ' ' -f 2)
        print_success "Python $PYTHON_VERSION found"
    elif command_exists python; then
        PYTHON_VERSION=$(python --version | cut -d ' ' -f 2)
        print_success "Python $PYTHON_VERSION found"
    else
        print_error "Python not found! Please install Python 3.9+ first."
        exit 1
    fi
    
    # Check pip
    if command_exists pip3; then
        print_success "pip3 found"
        PIP_CMD="pip3"
    elif command_exists pip; then
        print_success "pip found"
        PIP_CMD="pip"
    else
        print_error "pip not found! Please install pip first."
        exit 1
    fi
    
    # Check git
    if command_exists git; then
        print_success "Git found"
    else
        print_warning "Git not found. Will download as ZIP instead."
    fi
    
    # Check available disk space (need at least 2GB)
    if command_exists df; then
        AVAILABLE_SPACE=$(df . | tail -1 | awk '{print $4}')
        if [ "$AVAILABLE_SPACE" -lt 2000000 ]; then
            print_warning "Low disk space detected. At least 2GB recommended."
        fi
    fi
    
    print_success "System requirements check completed"
    echo ""
}

# Install system dependencies
install_system_deps() {
    print_status "Installing system dependencies..."
    
    OS=$(detect_os)
    
    case $OS in
        "linux")
            if command_exists apt-get; then
                sudo apt-get update
                sudo apt-get install -y python3-pip python3-venv curl wget
            elif command_exists yum; then
                sudo yum install -y python3-pip python3-venv curl wget
            elif command_exists pacman; then
                sudo pacman -S python-pip python-virtualenv curl wget
            fi
            ;;
        "macos")
            if command_exists brew; then
                brew install python3 curl wget
            else
                print_warning "Homebrew not found. Please install Python 3.9+ manually."
            fi
            ;;
        "windows")
            print_status "Windows detected. Please ensure Python 3.9+ is installed."
            ;;
    esac
    
    print_success "System dependencies installed"
    echo ""
}

# Download and setup project
setup_project() {
    print_status "Setting up project..."
    
    PROJECT_DIR="smart-waste-classifier"
    
    # Remove existing directory if it exists
    if [ -d "$PROJECT_DIR" ]; then
        print_warning "Existing installation found. Removing..."
        rm -rf "$PROJECT_DIR"
    fi
    
    # Download project
    if command_exists git; then
        print_status "Cloning repository..."
        git clone https://github.com/yourusername/smart-waste-classifier.git
    else
        print_status "Downloading as ZIP..."
        wget -O smart-waste-classifier.zip https://github.com/yourusername/smart-waste-classifier/archive/main.zip
        unzip smart-waste-classifier.zip
        mv smart-waste-classifier-main smart-waste-classifier
        rm smart-waste-classifier.zip
    fi
    
    cd "$PROJECT_DIR"
    print_success "Project downloaded"
    echo ""
}

# Setup Python environment
setup_python_env() {
    print_status "Setting up Python environment..."
    
    # Create virtual environment
    if command_exists python3; then
        python3 -m venv venv
    else
        python -m venv venv
    fi
    
    # Activate virtual environment
    source venv/bin/activate 2>/dev/null || source venv/Scripts/activate
    
    # Upgrade pip
    pip install --upgrade pip
    
    # Install requirements
    print_status "Installing Python dependencies..."
    pip install -r requirements.txt
    
    print_success "Python environment setup completed"
    echo ""
}

# Download pre-trained model
download_model() {
    print_status "Downloading pre-trained model..."
    
    # Check if model already exists
    if [ -f "waste_classifier.pth" ]; then
        print_success "Model already exists"
        return
    fi
    
    # Download from GitHub releases or Google Drive
    MODEL_URL="https://github.com/yourusername/smart-waste-classifier/releases/download/v1.0.0/waste_classifier.pth"
    
    if command_exists wget; then
        wget -O waste_classifier.pth "$MODEL_URL"
    elif command_exists curl; then
        curl -L -o waste_classifier.pth "$MODEL_URL"
    else
        print_error "Neither wget nor curl found. Please download the model manually."
        print_status "Download URL: $MODEL_URL"
        return 1
    fi
    
    # Download metadata
    METADATA_URL="https://github.com/yourusername/smart-waste-classifier/releases/download/v1.0.0/dataset_metadata.json"
    
    if command_exists wget; then
        wget -O dataset_metadata.json "$METADATA_URL"
    elif command_exists curl; then
        curl -L -o dataset_metadata.json "$METADATA_URL"
    fi
    
    print_success "Model downloaded successfully"
    echo ""
}

# Create desktop shortcuts and menu entries
create_shortcuts() {
    print_status "Creating shortcuts..."
    
    OS=$(detect_os)
    INSTALL_DIR=$(pwd)
    
    case $OS in
        "linux")
            # Create desktop entry
            mkdir -p ~/.local/share/applications
            cat > ~/.local/share/applications/waste-classifier.desktop << EOF
[Desktop Entry]
Name=Smart Waste Classifier
Comment=AI-powered waste sorting assistant
Exec=$INSTALL_DIR/run.sh
Icon=$INSTALL_DIR/icon.png
Terminal=false
Type=Application
Categories=Utility;Science;
EOF
            chmod +x ~/.local/share/applications/waste-classifier.desktop
            ;;
            
        "macos")
            # Create app bundle (simplified)
            mkdir -p ~/Applications/SmartWasteClassifier.app/Contents/MacOS
            cat > ~/Applications/SmartWasteClassifier.app/Contents/MacOS/run << EOF
#!/bin/bash
cd "$INSTALL_DIR"
./run.sh
EOF
            chmod +x ~/Applications/SmartWasteClassifier.app/Contents/MacOS/run
            ;;
            
        "windows")
            # Create batch file for Windows
            cat > ~/Desktop/SmartWasteClassifier.bat << EOF
@echo off
cd /d "$INSTALL_DIR"
call run.sh
pause
EOF
            ;;
    esac
    
    print_success "Shortcuts created"
    echo ""
}

# Test installation
test_installation() {
    print_status "Testing installation..."
    
    # Activate virtual environment
    source venv/bin/activate 2>/dev/null || source venv/Scripts/activate
    
    # Test imports
    python -c "
import torch
import torchvision
import fastapi
import streamlit
print('✅ All imports successful')
"
    
    if [ $? -eq 0 ]; then
        print_success "Installation test passed"
    else
        print_error "Installation test failed"
        return 1
    fi
    
    echo ""
}

# Final setup and instructions
finalize_setup() {
    print_status "Finalizing setup..."
    
    # Make run script executable
    chmod +x run.sh
    
    # Create activation script
    cat > activate.sh << 'EOF'
#!/bin/bash
echo "🚀 Activating Smart Waste Classifier environment..."
cd "$(dirname "$0")"
source venv/bin/activate 2>/dev/null || source venv/Scripts/activate
echo "✅ Environment activated!"
echo "Run './run.sh' to start the application"
EOF
    chmod +x activate.sh
    
    print_success "Setup finalized"
    echo ""
}

# Display final instructions
show_instructions() {
    echo -e "${GREEN}"
    cat << "EOF"
┌─────────────────────────────────────────────────────────────┐
│                   🎉 INSTALLATION COMPLETE! 🎉              │
└─────────────────────────────────────────────────────────────┘
EOF
    echo -e "${NC}"
    
    echo -e "${BLUE}📋 Next Steps:${NC}"
    echo ""
    echo -e "1. ${YELLOW}Start the application:${NC}"
    echo -e "   ${GREEN}./run.sh${NC}"
    echo ""
    echo -e "2. ${YELLOW}Access the interfaces:${NC}"
    echo -e "   🌐 Web App: ${GREEN}http://localhost:8501${NC}"
    echo -e "   📡 API: ${GREEN}http://localhost:8000${NC}"
    echo -e "   📚 API Docs: ${GREEN}http://localhost:8000/docs${NC}"
    echo ""
    echo -e "3. ${YELLOW}Test the system:${NC}"
    echo -e "   ${GREEN}python test_api.py${NC}"
    echo ""
    echo -e "${BLUE}💡 Tips:${NC}"
    echo -e "• Upload clear images of waste items for best results"
    echo -e "• The first prediction may take longer (model loading)"
    echo -e "• Check the documentation at: ${GREEN}README.md${NC}"
    echo ""
    echo -e "${BLUE}🆘 Need help?${NC}"
    echo -e "• GitHub Issues: ${GREEN}https://github.com/yourusername/smart-waste-classifier/issues${NC}"
    echo -e "• Email: ${GREEN}support@wasteai.com${NC}"
    echo ""
    echo -e "${GREEN}Thank you for using Smart Waste Classifier!${NC}"
    echo -e "${GREEN}Together, let's make recycling smarter! ♻️${NC}"
}

# Main installation flow
main() {
    echo "Starting installation process..."
    echo ""
    
    check_requirements
    install_system_deps
    setup_project
    setup_python_env
    download_model
    create_shortcuts
    test_installation
    finalize_setup
    show_instructions
    
    print_success "Installation completed successfully!"
}

# Handle Ctrl+C
trap 'echo -e "\n${RED}Installation cancelled by user${NC}"; exit 1' INT

# Run main installation
main "$@"