# macOS TTS Hotkey Application

A standalone macOS application that monitors for double Command key taps and performs text-to-speech on selected text using the existing kokoro TTS model.

## Features

- 🎯 **Global Hotkey**: Double-tap Command key to trigger TTS
- 🔊 **High-Quality TTS**: Uses the kokoro TTS model for natural speech
- 🌍 **System-Wide**: Works with text selection in any macOS application
- 🔔 **Audio Feedback**: Notification sounds for different states
- ⚡ **Fast Response**: Optimized for quick audio generation
- 🛡️ **Error Handling**: Comprehensive error messages and recovery

## Requirements

### System Requirements
- macOS (tested on macOS 10.14+)
- Python 3.8 or higher
- Accessibility permissions for global hotkey monitoring

### Dependencies
- PyTorch (for TTS model)
- pygame (for audio playback)
- pynput (for global hotkey detection)
- pyperclip (for clipboard access)
- soundfile (for audio processing)
- numpy (for audio data handling)

## Installation

1. **Install Python dependencies:**
   ```bash
   pip install -r requirements_hotkey.txt
   ```

2. **Ensure you have the TTS model files:**
   - `kokoro-v0_19.pth` (main TTS model)
   - `voices/bf_isabella.pt` (voice pack)

3. **Grant Accessibility Permissions:**
   - Open System Preferences/Settings
   - Go to Security & Privacy > Privacy
   - Select "Accessibility" from the left panel
   - Click the lock icon and enter your password
   - Add your terminal application (Terminal.app, iTerm2, etc.)
   - Make sure it's checked/enabled

## Usage

1. **Start the application:**
   ```bash
   python macos_tts_hotkey.py
   ```

2. **Use the TTS hotkey:**
   - Select any text in any application
   - Double-tap the Command key quickly (within 0.5 seconds)
   - Listen to the text being read aloud

3. **Stop the application:**
   - Press `Ctrl+C` in the terminal

## Configuration

You can modify the application behavior by editing the `AppConfig` class in `macos_tts_hotkey.py`:

```python
@dataclass
class AppConfig:
    model_path: str = "kokoro-v0_19.pth"      # Path to TTS model
    voice_path: str = "voices/bf_isabella.pt"  # Path to voice pack
    hotkey_timeout: float = 0.5                # Double-tap window (seconds)
    max_text_length: int = 500                 # Maximum text length
    device: str = "auto"                       # "cuda", "cpu", or "auto"
```

## Audio Notifications

The application provides audio feedback for different states:

- **Success**: High-pitched beep when TTS starts
- **No Text**: Double beep when no text is selected
- **Error**: Low-pitched beep when an error occurs

## Troubleshooting

### Common Issues

1. **"Failed to start keyboard monitoring"**
   - Grant Accessibility permissions (see Installation step 3)
   - Restart the application after granting permissions

2. **"Model file not found"**
   - Ensure `kokoro-v0_19.pth` is in the project directory
   - Check that the file path is correct

3. **"Voice file not found"**
   - Ensure the `voices/` directory exists
   - Check that `bf_isabella.pt` is in the voices directory

4. **"No text found in clipboard"**
   - Make sure to select text before using the hotkey
   - Try copying text manually with Cmd+C first

5. **Audio playback issues**
   - Check system audio settings
   - Ensure no other applications are blocking audio

### Performance Tips

- Use GPU acceleration if available (CUDA)
- Keep text selections under 500 characters for best performance
- Close other resource-intensive applications if experiencing slowdowns

## Technical Details

### Architecture
- **HotkeyMonitor**: Detects double Command key presses using pynput
- **TextProcessor**: Captures and prepares text from clipboard
- **TTSEngine**: Generates audio using the kokoro TTS model
- **AudioPlayer**: Plays generated audio using pygame mixer

### Audio Format
- Sample Rate: 24kHz
- Format: 16-bit PCM WAV
- Channels: Mono

### Text Processing
- Automatic text normalization and cleaning
- Length limiting with smart word boundary detection
- Support for various text formats and encodings

## License

This application uses the same license as the original kokoro TTS project.

## Contributing

Feel free to submit issues and enhancement requests!

## Acknowledgments

- Built on top of the kokoro TTS model
- Uses existing text processing and model loading code
- Inspired by the need for quick, system-wide TTS functionality