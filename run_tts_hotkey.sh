#!/bin/bash

# macOS TTS Hotkey Application Launcher
# This script provides a convenient way to start the TTS hotkey application

echo "macOS TTS Hotkey Application Launcher"
echo "====================================="

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed or not in PATH"
    echo "Please install Python 3.8 or higher"
    exit 1
fi

# Check if the main script exists
if [ ! -f "macos_tts_hotkey.py" ]; then
    echo "❌ macos_tts_hotkey.py not found in current directory"
    echo "Please run this script from the project root directory"
    exit 1
fi

# Check if model files exist
if [ ! -f "kokoro-v1_0.pth" ]; then
    echo "⚠️  Warning: kokoro-v1_0.pth not found"
    echo "Please ensure the TTS model file is in the project directory"
fi

# Check if requirements file exists and offer to install dependencies
if [ -f "requirements.txt" ]; then
    echo ""
    echo "📦 Checking dependencies..."
    
    # Check if pip is available
    if command -v pip3 &> /dev/null; then
        echo "Would you like to install/update dependencies? (y/n)"
        read -r response
        if [[ "$response" =~ ^[Yy]$ ]]; then
            echo "Installing dependencies..."
            pip3 install -r requirements.txt
        fi
    else
        echo "⚠️  pip3 not found. Please install dependencies manually:"
        echo "pip3 install -r requirements.txt"
    fi
fi

echo ""
echo "🚀 Starting macOS TTS Hotkey Application..."
echo ""
echo "Instructions:"
echo "1. Select text in any application"
echo "2. Double-tap the Command key quickly"
echo "3. Listen to the TTS audio"
echo "4. Press Ctrl+C to exit"
echo ""
echo "Note: You may need to grant Accessibility permissions"
echo "      Go to System Preferences > Security & Privacy > Privacy > Accessibility"
echo ""

# Start the application
python3 macos_tts_hotkey.py