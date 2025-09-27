#!/usr/bin/env python3
"""
macOS TTS Hotkey Application

A standalone application that monitors for double Command key taps and performs
text-to-speech on selected text using the existing TTS model.
"""

import sys
import os
import signal
import time
import threading
from dataclasses import dataclass
from typing import Optional, Callable
import logging
import numpy as np
import torch
import io
import soundfile as sf

# Audio playback
import pygame

# System integration
import pyperclip
from pynput import keyboard

# Import kokoro modules using demo app approach
from kokoro import KModel, KPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class AppConfig:
    """Application configuration"""
    voice: str = "bf_isabella"   # Default voice
    hotkey_timeout: float = 0.5  # Double-tap window
    max_text_length: int = 500   # Character limit
    device: str = "auto"         # cuda/cpu/auto

@dataclass
class AudioData:
    """Audio data container"""
    samples: np.ndarray  # Audio samples
    sample_rate: int     # Sample rate (24000 Hz)
    duration: float      # Duration in seconds

class TTSEngine:
    """TTS Engine component using kokoro-demo-app.py approach with KModel and KPipeline"""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.models = {}  # Dictionary to hold CPU and GPU models
        self.pipelines = {}  # Dictionary to hold language pipelines
        self.voice = config.voice  # Use voice from config
        self.use_gpu = False
        self._model_loaded = False
        self._generation_lock = threading.Lock()  # Prevent concurrent generation
        
    def initialize_model(self):
        """Load KModel instances and KPipeline instances"""
        try:
            logger.info("Initializing TTS model using kokoro-demo-app approach...")
            
            # Determine available devices
            cuda_available = torch.cuda.is_available()
            
            if self.config.device == "auto":
                self.use_gpu = cuda_available
            elif self.config.device == "cuda":
                self.use_gpu = cuda_available
                if not cuda_available:
                    logger.warning("CUDA requested but not available, falling back to CPU")
                    self.use_gpu = False
            else:
                self.use_gpu = False
            
            logger.info(f"CUDA available: {cuda_available}, Using GPU: {self.use_gpu}")
            
            # Initialize models like in kokoro-demo-app.py
            try:
                # Always create CPU model
                logger.info("Loading CPU model...")
                self.models[False] = KModel().to('cpu').eval()
                
                # Create GPU model if available and requested
                if self.use_gpu:
                    logger.info("Loading GPU model...")
                    self.models[True] = KModel().to('cuda').eval()
                
                logger.info("Models loaded successfully")
            except Exception as e:
                error_msg = f"""
Failed to load KModel: {e}

This could be due to:
- Missing kokoro library installation
- Corrupted model files
- Incompatible PyTorch version
- Insufficient memory

Try running: pip install kokoro-tts
"""
                logger.error(error_msg)
                raise RuntimeError(error_msg)
            
            # Initialize pipelines like in kokoro-demo-app.py
            try:
                logger.info("Initializing pipelines...")
                # Create pipelines for supported languages (a=American, b=British)
                self.pipelines = {
                    lang_code: KPipeline(lang_code=lang_code, model=False) 
                    for lang_code in 'ab'
                }
                # Add custom pronunciation for 'kokoro' like in demo app
                self.pipelines['a'].g2p.lexicon.golds['kokoro'] = 'kˈOkəɹO'
                self.pipelines['b'].g2p.lexicon.golds['kokoro'] = 'kˈQkəɹQ'
                
                logger.info("Pipelines initialized successfully")
            except Exception as e:
                error_msg = f"""
Failed to initialize KPipeline: {e}

This could be due to:
- Missing kokoro library components
- Missing phonemizer dependencies
- System configuration issues

Try running: pip install kokoro-tts phonemizer
"""
                logger.error(error_msg)
                raise RuntimeError(error_msg)
            
            # Pre-load the default voice
            try:
                logger.info(f"Pre-loading voice: {self.voice}")
                pipeline = self.pipelines[self.voice[0]]  # 'b' for bf_isabella
                pipeline.load_voice(self.voice)
                logger.info("Voice loaded successfully")
            except Exception as e:
                error_msg = f"""
Failed to load voice pack: {self.voice}

This could be due to:
- Missing voice file in voices/ directory
- Corrupted voice file
- Incompatible voice format

Check that the voices directory contains: {self.voice}.pt
"""
                logger.error(error_msg)
                raise RuntimeError(error_msg)
            
            # Pre-warm the model with a short test
            try:
                logger.info("Pre-warming TTS model...")
                test_audio = self._generate_audio_internal("Hello")
                if test_audio is not None:
                    logger.info("Model pre-warming successful")
                else:
                    logger.warning("Model pre-warming failed, but continuing...")
            except Exception as e:
                logger.warning(f"Model pre-warming failed: {e}, but continuing...")
            
            self._model_loaded = True
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize TTS model: {e}")
            return False
    
    def generate_audio(self, text: str) -> Optional[AudioData]:
        """Convert text to audio with thread safety"""
        with self._generation_lock:
            return self._generate_audio_internal(text)
    
    def _generate_audio_internal(self, text: str) -> Optional[AudioData]:
        """Internal audio generation method using kokoro-demo-app approach"""
        try:
            if not self._model_loaded or not self.models or not self.pipelines:
                logger.error("Model not initialized")
                return None
            
            # Validate and prepare text
            if not text or not text.strip():
                logger.warning("Empty text provided")
                return None
            
            # Truncate text if too long
            if len(text) > self.config.max_text_length:
                text = text[:self.config.max_text_length]
                logger.warning(f"Text truncated to {self.config.max_text_length} characters")
            
            text = text.strip()
            logger.info(f"Generating audio for: {text[:50]}...")
            
            # Use the pipeline approach like in kokoro-demo-app.py
            pipeline = self.pipelines[self.voice[0]]  # Get pipeline for voice language
            pack = pipeline.load_voice(self.voice)
            
            # Generate audio using the demo app approach
            use_gpu = self.use_gpu and self.use_gpu in self.models
            
            # Process text through pipeline (similar to generate_first function)
            for _, ps, _ in pipeline(text, self.voice, speed=1.0):
                ref_s = pack[len(ps)-1]
                
                try:
                    if use_gpu:
                        # Use GPU model
                        audio_tensor = self.models[True](ps, ref_s, 1.0)
                    else:
                        # Use CPU model
                        audio_tensor = self.models[False](ps, ref_s, 1.0)
                        
                except Exception as e:
                    if use_gpu:
                        logger.warning(f"GPU generation failed: {e}, retrying with CPU")
                        try:
                            audio_tensor = self.models[False](ps, ref_s, 1.0)
                        except Exception as cpu_e:
                            logger.error(f"CPU generation also failed: {cpu_e}")
                            return None
                    else:
                        logger.error(f"Audio generation failed: {e}")
                        return None
                
                # Convert tensor to numpy array
                if audio_tensor is not None:
                    audio_samples = audio_tensor.numpy()
                    
                    # Create AudioData object
                    audio_data = AudioData(
                        samples=audio_samples,
                        sample_rate=24000,
                        duration=len(audio_samples) / 24000
                    )
                    
                    logger.info(f"Audio generated successfully: {audio_data.duration:.2f}s")
                    
                    # Clear GPU cache after generation if using CUDA
                    if use_gpu:
                        torch.cuda.empty_cache()
                    
                    return audio_data
                else:
                    logger.error("Audio generation returned None")
                    return None
            
            # If we get here, the pipeline didn't yield any results
            logger.error("Pipeline did not generate any audio segments")
            return None
            
        except Exception as e:
            logger.error(f"Audio generation failed: {e}")
            return None
    
    def cleanup(self):
        """Free model resources"""
        try:
            if self.models:
                # Clear CUDA cache if using GPU
                if self.use_gpu:
                    torch.cuda.empty_cache()
                self.models.clear()
                logger.info("Model resources cleaned up")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

class AudioPlayer:
    """Audio playback component using pygame"""
    
    def __init__(self):
        self.mixer_initialized = False
        self.current_sound = None
        self._playback_lock = threading.Lock()  # Prevent concurrent playback
        self._audio_cache = {}  # Cache for notification sounds
        
    def initialize(self):
        """Initialize pygame mixer"""
        try:
            pygame.mixer.pre_init(frequency=24000, size=-16, channels=1, buffer=1024)
            pygame.mixer.init()
            self.mixer_initialized = True
            logger.info("Audio player initialized")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize audio player: {e}")
            return False
    
    def play_audio(self, audio_data: AudioData):
        """Play WAV audio data with thread safety"""
        with self._playback_lock:
            return self._play_audio_internal(audio_data)
    
    def _play_audio_internal(self, audio_data: AudioData):
        """Internal audio playback method"""
        try:
            if not self.mixer_initialized:
                logger.error("Audio player not initialized")
                return False
            
            # Stop any currently playing audio
            self.stop_current()
            
            # Convert numpy array to bytes for pygame
            # Ensure audio is in the right format (16-bit PCM)
            audio_samples = audio_data.samples
            
            # Optimize audio conversion
            if audio_samples.dtype != np.int16:
                # Normalize to [-1, 1] if not already
                max_val = np.max(np.abs(audio_samples))
                if max_val > 1.0:
                    audio_samples = audio_samples / max_val
                
                # Convert to 16-bit PCM more efficiently
                audio_samples = (audio_samples * 32767).astype(np.int16)
            
            # Create a BytesIO buffer with WAV data
            buffer = io.BytesIO()
            sf.write(buffer, audio_samples, audio_data.sample_rate, format='WAV')
            buffer.seek(0)
            
            # Load and play the sound
            self.current_sound = pygame.mixer.Sound(buffer)
            self.current_sound.play()
            
            logger.info(f"Playing audio: {audio_data.duration:.2f}s")
            return True
            
        except Exception as e:
            logger.error(f"Failed to play audio: {e}")
            return False
    
    def stop_current(self):
        """Stop any currently playing audio"""
        try:
            if self.mixer_initialized:
                pygame.mixer.stop()
                self.current_sound = None
        except Exception as e:
            logger.error(f"Error stopping audio: {e}")
    
    def play_notification(self, sound_type: str = "success"):
        """Play system notification sounds with caching"""
        try:
            if not self.mixer_initialized:
                return False
            
            # Check cache first
            if sound_type in self._audio_cache:
                return self.play_audio(self._audio_cache[sound_type])
            
            # Generate simple notification tones
            sample_rate = 24000
            duration = 0.2  # 200ms
            
            if sound_type == "success":
                # High-pitched beep for success
                frequency = 800
            elif sound_type == "error":
                # Lower pitched beep for error
                frequency = 400
            elif sound_type == "no_text":
                # Double beep for no text selected
                frequency = 600
            else:
                frequency = 600
            
            # Generate sine wave
            t = np.linspace(0, duration, int(sample_rate * duration))
            
            if sound_type == "no_text":
                # Double beep
                beep1 = np.sin(2 * np.pi * frequency * t[:len(t)//3])
                silence = np.zeros(len(t)//6)
                beep2 = np.sin(2 * np.pi * frequency * t[:len(t)//3])
                padding = np.zeros(len(t) - len(beep1) - len(silence) - len(beep2))
                audio_samples = np.concatenate([beep1, silence, beep2, padding])
            else:
                # Single beep
                audio_samples = np.sin(2 * np.pi * frequency * t)
            
            # Apply fade in/out to avoid clicks
            fade_samples = int(0.01 * sample_rate)  # 10ms fade
            audio_samples[:fade_samples] *= np.linspace(0, 1, fade_samples)
            audio_samples[-fade_samples:] *= np.linspace(1, 0, fade_samples)
            
            # Convert to 16-bit PCM
            audio_samples = (audio_samples * 0.3 * 32767).astype(np.int16)
            
            # Create audio data and cache it
            notification_audio = AudioData(
                samples=audio_samples,
                sample_rate=sample_rate,
                duration=duration
            )
            
            # Cache the notification sound for future use
            self._audio_cache[sound_type] = notification_audio
            
            return self.play_audio(notification_audio)
            
        except Exception as e:
            logger.error(f"Failed to play notification: {e}")
            return False
    
    def cleanup(self):
        """Clean up audio resources"""
        try:
            if self.mixer_initialized:
                self.stop_current()
                pygame.mixer.quit()
                self.mixer_initialized = False
                logger.info("Audio player cleaned up")
        except Exception as e:
            logger.error(f"Error during audio cleanup: {e}")

class TextProcessor:
    """Text processing component for clipboard text capture"""
    
    def __init__(self, config: AppConfig):
        self.config = config
        
    def get_selected_text(self) -> Optional[str]:
        """Retrieve currently selected text via clipboard"""
        try:
            # Store current clipboard content
            original_clipboard = None
            try:
                original_clipboard = pyperclip.paste()
            except Exception as e:
                logger.warning(f"Could not access clipboard: {e}")
                return None
            
            # For this implementation, we'll work with whatever is currently in the clipboard
            # In a more advanced version, you could use AppleScript to copy selected text:
            # osascript -e 'tell application "System Events" to keystroke "c" using command down'
            
            try:
                selected_text = pyperclip.paste()
                
                if selected_text and selected_text.strip():
                    logger.info(f"Retrieved text from clipboard: {selected_text[:50]}...")
                    return selected_text.strip()
                else:
                    logger.info("No text found in clipboard")
                    return None
                    
            except Exception as e:
                error_msg = f"""
Failed to access clipboard: {e}

This could be due to:
- Clipboard access permissions
- Another application blocking clipboard access
- System clipboard issues

Try:
1. Copy some text manually (Cmd+C)
2. Then trigger the hotkey
3. Check System Preferences > Security & Privacy > Privacy > Accessibility
"""
                logger.error(error_msg)
                return None
                
        except Exception as e:
            logger.error(f"Error getting selected text: {e}")
            return None
    
    def validate_text(self, text: str) -> bool:
        """Check if text is suitable for TTS"""
        if not text or not text.strip():
            return False
        
        # Check length
        if len(text) > self.config.max_text_length:
            logger.warning(f"Text too long: {len(text)} > {self.config.max_text_length}")
            # Still valid, will be truncated later
            return True
        
        # Check for valid characters (basic validation)
        # Allow letters, numbers, punctuation, and whitespace
        import re
        if not re.search(r'[a-zA-Z0-9]', text):
            logger.warning("Text contains no alphanumeric characters")
            return False
        
        return True
    
    def prepare_text(self, text: str) -> Optional[str]:
        """Clean and normalize text for processing"""
        try:
            if not text:
                return None
            
            # Basic cleaning
            text = text.strip()
            
            # Remove excessive whitespace
            import re
            text = re.sub(r'\s+', ' ', text)
            
            # Remove or replace problematic characters
            # Remove control characters except newlines and tabs
            text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
            
            # Replace newlines with spaces for better TTS flow
            text = text.replace('\n', ' ').replace('\t', ' ')
            
            # Truncate if too long
            if len(text) > self.config.max_text_length:
                text = text[:self.config.max_text_length]
                # Try to break at word boundary
                last_space = text.rfind(' ')
                if last_space > self.config.max_text_length * 0.8:  # If we can find a space in the last 20%
                    text = text[:last_space]
                logger.info(f"Text truncated to {len(text)} characters")
            
            # Final validation
            if not self.validate_text(text):
                return None
            
            return text
            
        except Exception as e:
            logger.error(f"Error preparing text: {e}")
            return None

class HotkeyMonitor:
    """Hotkey monitoring component using pynput"""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.listener = None
        self.last_cmd_press = 0
        self.cmd_pressed = False
        self.hotkey_callback = None
        self.running = False
        
    def start_monitoring(self):
        """Begin listening for global key events"""
        try:
            logger.info("Starting hotkey monitoring...")
            self.running = True
            
            # Create keyboard listener with error handling
            try:
                self.listener = keyboard.Listener(
                    on_press=self._on_key_press,
                    on_release=self._on_key_release
                )
                
                # Start listener in a separate thread
                self.listener.start()
                logger.info("Hotkey monitoring started - listening for double Option key press")
                return True
                
            except Exception as e:
                error_msg = f"""
Failed to start keyboard monitoring: {e}

This is likely due to missing accessibility permissions.

To fix this on macOS:
1. Open System Preferences/Settings
2. Go to Security & Privacy > Privacy
3. Select "Accessibility" from the left panel
4. Click the lock icon and enter your password
5. Add your terminal application (Terminal.app, iTerm2, etc.)
6. Make sure it's checked/enabled
7. Restart this application

You may also need to add Python to the accessibility list.
"""
                logger.error(error_msg)
                raise RuntimeError(error_msg)
            
        except Exception as e:
            logger.error(f"Failed to start hotkey monitoring: {e}")
            return False
    
    def stop_monitoring(self):
        """Clean up event listeners"""
        try:
            self.running = False
            if self.listener:
                self.listener.stop()
                self.listener = None
            logger.info("Hotkey monitoring stopped")
        except Exception as e:
            logger.error(f"Error stopping hotkey monitoring: {e}")
    
    def on_hotkey_detected(self, callback: Callable):
        """Register callback for hotkey events"""
        self.hotkey_callback = callback
    
    def _on_key_press(self, key):
        """Handle key press events"""
        try:
            # Check if it's an Option key (Alt key on macOS)
            if key == keyboard.Key.alt or key == keyboard.Key.alt_r:
                current_time = time.time()
                
                # If Option was already pressed recently, check for double-tap
                if self.cmd_pressed and (current_time - self.last_cmd_press) <= self.config.hotkey_timeout:
                    logger.info("Double Option key detected!")
                    if self.hotkey_callback and self.running:
                        # Run callback in a separate thread to avoid blocking the listener
                        threading.Thread(
                            target=self.hotkey_callback, 
                            daemon=True, 
                            name="HotkeyCallback"
                        ).start()
                    
                    # Reset state
                    self.cmd_pressed = False
                    self.last_cmd_press = 0
                else:
                    # First Option press
                    self.cmd_pressed = True
                    self.last_cmd_press = current_time
                    
        except Exception as e:
            logger.error(f"Error in key press handler: {e}")
            # Try to restart monitoring if there's a critical error
            if not self.running:
                return
            try:
                self.stop_monitoring()
                time.sleep(1)
                self.start_monitoring()
                logger.info("Hotkey monitoring restarted after error")
            except:
                logger.error("Failed to restart hotkey monitoring")
    
    def _on_key_release(self, key):
        """Handle key release events"""
        try:
            # Reset Option state when any other key is pressed
            if key != keyboard.Key.alt and key != keyboard.Key.alt_r:
                if self.cmd_pressed:
                    # Check if enough time has passed to reset
                    current_time = time.time()
                    if (current_time - self.last_cmd_press) > self.config.hotkey_timeout:
                        self.cmd_pressed = False
                        self.last_cmd_press = 0
                        
        except Exception as e:
            logger.error(f"Error in key release handler: {e}")
    
    def cleanup(self):
        """Clean up resources"""
        self.stop_monitoring()

class MacOSTTSApp:
    """Main application coordinator"""
    
    def __init__(self, config: AppConfig = None):
        self.config = config or AppConfig()
        self.running = False
        self.components = {}
        
    def start(self):
        """Initialize all components and start monitoring"""
        logger.info("Starting macOS TTS Hotkey application...")
        self.running = True
        
        # Initialize TTS Engine
        tts_engine = TTSEngine(self.config)
        if not tts_engine.initialize_model():
            logger.error("Failed to initialize TTS engine")
            return False
        
        # Initialize Audio Player
        audio_player = AudioPlayer()
        if not audio_player.initialize():
            logger.error("Failed to initialize audio player")
            return False
        
        # Initialize Text Processor
        text_processor = TextProcessor(self.config)
        
        # Initialize Hotkey Monitor
        hotkey_monitor = HotkeyMonitor(self.config)
        hotkey_monitor.on_hotkey_detected(self.handle_hotkey)
        
        if not hotkey_monitor.start_monitoring():
            logger.error("Failed to start hotkey monitoring")
            return False
        
        self.components['tts_engine'] = tts_engine
        self.components['audio_player'] = audio_player
        self.components['text_processor'] = text_processor
        self.components['hotkey_monitor'] = hotkey_monitor
        logger.info("Application started successfully")
        return True
        
    def stop(self):
        """Clean shutdown of all components"""
        logger.info("Stopping application...")
        self.running = False
        
        # Cleanup components
        for component in self.components.values():
            if hasattr(component, 'cleanup'):
                component.cleanup()
                
        logger.info("Application stopped")
        
    def handle_hotkey(self):
        """Main event handler for hotkey triggers with background processing"""
        # Run in background thread to avoid blocking the hotkey listener
        def process_hotkey():
            logger.info("Hotkey detected - processing...")
            
            try:
                # Get components
                text_processor = self.components.get('text_processor')
                tts_engine = self.components.get('tts_engine')
                audio_player = self.components.get('audio_player')
                
                if not all([text_processor, tts_engine, audio_player]):
                    logger.error("Not all components are available")
                    return
                
                # Immediate audio feedback
                audio_player.play_notification("success")
                
                # Step 1: Get selected text
                selected_text = text_processor.get_selected_text()
                
                if not selected_text:
                    logger.info("No text selected - playing notification")
                    audio_player.play_notification("no_text")
                    return
                
                # Step 2: Prepare text for TTS
                prepared_text = text_processor.prepare_text(selected_text)
                
                if not prepared_text:
                    logger.warning("Text preparation failed - playing error notification")
                    audio_player.play_notification("error")
                    return
                
                # Step 3: Generate audio
                logger.info("Generating TTS audio...")
                audio_data = tts_engine.generate_audio(prepared_text)
                
                if not audio_data:
                    logger.error("TTS generation failed - playing error notification")
                    audio_player.play_notification("error")
                    return
                
                # Step 4: Play audio
                logger.info("Playing generated audio...")
                if audio_player.play_audio(audio_data):
                    logger.info("TTS playback completed successfully")
                else:
                    logger.error("Audio playback failed")
                    audio_player.play_notification("error")
                    
            except Exception as e:
                logger.error(f"Error in hotkey handler: {e}")
                # Try to play error notification
                try:
                    audio_player = self.components.get('audio_player')
                    if audio_player:
                        audio_player.play_notification("error")
                except:
                    pass  # Ignore errors in error handling
        
        # Start processing in background thread
        threading.Thread(target=process_hotkey, daemon=True, name="HotkeyProcessor").start()

def signal_handler(signum, frame):
    """Handle shutdown signals"""
    logger.info("Received shutdown signal")
    if hasattr(signal_handler, 'app'):
        signal_handler.app.stop()
    sys.exit(0)

def check_dependencies():
    """Check if all required dependencies are available"""
    missing_deps = []
    
    try:
        import torch
    except ImportError:
        missing_deps.append("torch")
    
    try:
        import pygame
    except ImportError:
        missing_deps.append("pygame")
    
    try:
        import pynput
    except ImportError:
        missing_deps.append("pynput")
    
    try:
        import pyperclip
    except ImportError:
        missing_deps.append("pyperclip")
    
    try:
        import soundfile
    except ImportError:
        missing_deps.append("soundfile")
    
    try:
        from kokoro import KModel, KPipeline
    except ImportError:
        missing_deps.append("kokoro-tts")
    
    if missing_deps:
        error_msg = f"""
Missing required dependencies: {', '.join(missing_deps)}

Please install them using:
uv pip install {' '.join(missing_deps)}

Or install all dependencies with:
uv pip install -r requirements_hotkey.txt
"""
        logger.error(error_msg)
        return False
    
    return True

def load_config():
    """Load configuration from file if it exists"""
    config_file = "config_hotkey.json"
    if os.path.exists(config_file):
        try:
            import json
            with open(config_file, 'r') as f:
                config_data = json.load(f)
            
            # Create AppConfig with loaded values
            config = AppConfig(
                voice=config_data.get("voice", "bf_isabella"),
                hotkey_timeout=config_data.get("hotkey_timeout", 0.5),
                max_text_length=config_data.get("max_text_length", 500),
                device=config_data.get("device", "auto")
            )
            
            # Set log level if specified
            log_level = config_data.get("log_level", "INFO")
            logging.getLogger().setLevel(getattr(logging, log_level.upper(), logging.INFO))
            
            logger.info(f"Configuration loaded from {config_file}")
            return config
            
        except Exception as e:
            logger.warning(f"Failed to load config from {config_file}: {e}")
            logger.info("Using default configuration")
    else:
        logger.info("No config file found, using default configuration")
    
    return AppConfig()

def main():
    """Application entry point"""
    print("macOS TTS Hotkey Application")
    print("=" * 40)
    
    # Check dependencies first
    if not check_dependencies():
        sys.exit(1)
    
    # Load configuration
    config = load_config()
    
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Create and start application
    app = MacOSTTSApp(config)
    signal_handler.app = app
    
    try:
        logger.info("Starting application...")
        if not app.start():
            logger.error("Failed to start application. Check the error messages above.")
            sys.exit(1)
        
        print("\n✅ Application started successfully!")
        print("\nUsage:")
        print("1. Select any text in any application")
        print("2. Double-tap the Option key quickly (within 0.5 seconds)")
        print("3. Listen to the text being read aloud")
        print("\nTips:")
        print("- You'll hear a beep when the hotkey is detected")
        print("- Double beep means no text was selected")
        print("- Low beep means an error occurred")
        print("- Works with text from any macOS application")
        print("\nPress Ctrl+C to exit.")
        print("-" * 40)
        
        # Keep application running
        while app.running:
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
        print("\n👋 Shutting down...")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        print(f"\n❌ Unexpected error: {e}")
    finally:
        app.stop()
        print("Application stopped.")

if __name__ == "__main__":
    main()