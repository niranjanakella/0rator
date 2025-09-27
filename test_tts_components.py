#!/usr/bin/env python3
"""
Test script for macOS TTS Hotkey components
This script tests individual components without requiring the full hotkey setup
"""

import sys
import os
import logging
from macos_tts_hotkey import AppConfig, TTSEngine, AudioPlayer, TextProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_dependencies():
    """Test if all dependencies are available"""
    print("Testing dependencies...")
    
    missing = []
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
    except ImportError:
        missing.append("torch")
        print("❌ PyTorch not found")
    
    try:
        import pygame
        print(f"✅ Pygame: {pygame.version.ver}")
    except ImportError:
        missing.append("pygame")
        print("❌ Pygame not found")
    
    try:
        import pynput
        print("✅ Pynput available")
    except ImportError:
        missing.append("pynput")
        print("❌ Pynput not found")
    
    try:
        import pyperclip
        print("✅ Pyperclip available")
    except ImportError:
        missing.append("pyperclip")
        print("❌ Pyperclip not found")
    
    try:
        import soundfile
        print("✅ Soundfile available")
    except ImportError:
        missing.append("soundfile")
        print("❌ Soundfile not found")
    
    return len(missing) == 0, missing

def test_model_files():
    """Test if model files exist"""
    print("\nTesting model files...")
    
    config = AppConfig()
    
    if os.path.exists(config.model_path):
        print(f"✅ Model file found: {config.model_path}")
        model_exists = True
    else:
        print(f"❌ Model file missing: {config.model_path}")
        model_exists = False
    
    if os.path.exists(config.voice_path):
        print(f"✅ Voice file found: {config.voice_path}")
        voice_exists = True
    else:
        print(f"❌ Voice file missing: {config.voice_path}")
        voice_exists = False
    
    return model_exists and voice_exists

def test_audio_player():
    """Test audio player component"""
    print("\nTesting Audio Player...")
    
    try:
        player = AudioPlayer()
        if player.initialize():
            print("✅ Audio player initialized")
            
            # Test notification sounds
            print("Playing test notification sounds...")
            player.play_notification("success")
            import time
            time.sleep(0.5)
            player.play_notification("error")
            time.sleep(0.5)
            player.play_notification("no_text")
            time.sleep(1)
            
            player.cleanup()
            print("✅ Audio player test completed")
            return True
        else:
            print("❌ Audio player initialization failed")
            return False
    except Exception as e:
        print(f"❌ Audio player test failed: {e}")
        return False

def test_text_processor():
    """Test text processor component"""
    print("\nTesting Text Processor...")
    
    try:
        config = AppConfig()
        processor = TextProcessor(config)
        
        # Test text validation
        test_texts = [
            "Hello, world!",
            "",
            "A" * 1000,  # Long text
            "123 test text with numbers",
            "Text with\nnewlines\tand\ttabs"
        ]
        
        for text in test_texts:
            if text:
                prepared = processor.prepare_text(text)
                valid = processor.validate_text(text) if text else False
                print(f"Text: '{text[:30]}...' -> Valid: {valid}, Prepared: {prepared is not None}")
            else:
                print(f"Empty text -> Valid: False")
        
        print("✅ Text processor test completed")
        return True
        
    except Exception as e:
        print(f"❌ Text processor test failed: {e}")
        return False

def test_tts_engine():
    """Test TTS engine component"""
    print("\nTesting TTS Engine...")
    
    try:
        config = AppConfig()
        engine = TTSEngine(config)
        
        if engine.initialize_model():
            print("✅ TTS engine initialized")
            
            # Test audio generation
            test_text = "Hello, this is a test of the text to speech system."
            print(f"Generating audio for: '{test_text}'")
            
            audio_data = engine.generate_audio(test_text)
            if audio_data:
                print(f"✅ Audio generated: {audio_data.duration:.2f}s, {len(audio_data.samples)} samples")
                
                # Test playing the generated audio
                player = AudioPlayer()
                if player.initialize():
                    print("Playing generated audio...")
                    player.play_audio(audio_data)
                    import time
                    time.sleep(audio_data.duration + 0.5)
                    player.cleanup()
                
                engine.cleanup()
                print("✅ TTS engine test completed")
                return True
            else:
                print("❌ Audio generation failed")
                engine.cleanup()
                return False
        else:
            print("❌ TTS engine initialization failed")
            return False
            
    except Exception as e:
        print(f"❌ TTS engine test failed: {e}")
        return False

def main():
    """Run all component tests"""
    print("macOS TTS Hotkey Component Tests")
    print("=" * 40)
    
    # Test dependencies
    deps_ok, missing = test_dependencies()
    if not deps_ok:
        print(f"\n❌ Missing dependencies: {missing}")
        print("Please install them with: pip install -r requirements_hotkey.txt")
        return False
    
    # Test model files
    if not test_model_files():
        print("\n❌ Model files missing. Please ensure you have:")
        print("- kokoro-v0_19.pth")
        print("- voices/bf_isabella.pt")
        return False
    
    # Test individual components
    tests = [
        ("Audio Player", test_audio_player),
        ("Text Processor", test_text_processor),
        ("TTS Engine", test_tts_engine),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 40)
    print("Test Results:")
    all_passed = True
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
        if not result:
            all_passed = False
    
    if all_passed:
        print("\n🎉 All tests passed! The application should work correctly.")
        print("\nYou can now run: python macos_tts_hotkey.py")
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)