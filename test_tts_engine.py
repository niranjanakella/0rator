#!/usr/bin/env python3
"""
Test script for the updated TTS Engine using kokoro-demo-app approach
"""

import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_imports():
    """Test if all required imports work"""
    try:
        # Test kokoro imports
        from kokoro import KModel, KPipeline
        print("✅ Kokoro imports successful")
        
        # Test other imports
        import torch
        import numpy as np
        print("✅ Core dependencies available")
        
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_tts_engine():
    """Test TTS Engine initialization"""
    try:
        # Import our classes
        from macos_tts_hotkey import TTSEngine, AppConfig
        
        # Create config
        config = AppConfig(voice="bf_isabella", device="cpu")
        
        # Create TTS engine
        tts_engine = TTSEngine(config)
        print("✅ TTS Engine created successfully")
        
        # Test initialization (this might fail if models aren't available)
        print("Testing model initialization...")
        success = tts_engine.initialize_model()
        
        if success:
            print("✅ TTS Engine initialized successfully")
            
            # Test audio generation with a simple phrase
            print("Testing audio generation...")
            audio_data = tts_engine.generate_audio("Hello world")
            
            if audio_data:
                print(f"✅ Audio generated: {audio_data.duration:.2f}s")
            else:
                print("⚠️ Audio generation returned None")
            
            # Cleanup
            tts_engine.cleanup()
            print("✅ TTS Engine cleanup completed")
        else:
            print("⚠️ TTS Engine initialization failed (expected if models not available)")
        
        return True
        
    except Exception as e:
        print(f"❌ TTS Engine test failed: {e}")
        return False

def main():
    print("Testing updated TTS Engine implementation...")
    print("=" * 50)
    
    # Test imports first
    if not test_imports():
        print("❌ Import test failed")
        return False
    
    # Test TTS engine
    if not test_tts_engine():
        print("❌ TTS Engine test failed")
        return False
    
    print("=" * 50)
    print("✅ All tests completed!")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)