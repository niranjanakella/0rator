# Design Document

## Overview

The macOS TTS Hotkey application will be implemented as a standalone Python script that runs as a background daemon. It will use macOS-specific libraries for global hotkey detection and clipboard access, while leveraging the existing TTS model infrastructure for audio generation. The application follows a simple event-driven architecture with minimal dependencies.

## Architecture

### High-Level Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Hotkey        │    │   Text           │    │   TTS           │
│   Monitor       │───▶│   Processor      │───▶│   Engine        │
│                 │    │                  │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   macOS         │    │   Clipboard      │    │   Audio         │
│   Global Events │    │   Access         │    │   Playback      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### Component Interaction Flow

1. **Hotkey Monitor** continuously listens for double Command key presses
2. **Text Processor** captures selected text via clipboard when hotkey is triggered
3. **TTS Engine** processes the text using the existing kokoro model
4. **Audio Playback** plays the generated audio through system speakers

## Components and Interfaces

### 1. Hotkey Monitor (`HotkeyMonitor`)

**Purpose:** Detect double Command key presses using macOS global event monitoring

**Key Methods:**
- `start_monitoring()`: Begin listening for global key events
- `stop_monitoring()`: Clean up event listeners
- `on_hotkey_detected(callback)`: Register callback for hotkey events

**Implementation Approach:**
- Use `pynput` library for cross-platform keyboard monitoring
- Implement double-tap detection with configurable timing (500ms window)
- Handle macOS accessibility permissions gracefully

**Dependencies:**
- `pynput` for global keyboard event monitoring
- macOS Accessibility API permissions

### 2. Text Processor (`TextProcessor`)

**Purpose:** Capture and prepare selected text for TTS processing

**Key Methods:**
- `get_selected_text()`: Retrieve currently selected text via clipboard
- `validate_text(text)`: Check if text is suitable for TTS
- `prepare_text(text)`: Clean and normalize text for processing

**Implementation Approach:**
- Use clipboard manipulation to capture selected text
- Implement text validation (length limits, character filtering)
- Reuse existing text normalization from `kokorooo.py`

**Dependencies:**
- `pyperclip` for clipboard access
- Existing `normalize_text()` function from kokorooo.py

### 3. TTS Engine (`TTSEngine`)

**Purpose:** Generate audio from text using the kokoro model with KModel and KPipeline approach

**Key Methods:**
- `initialize_model()`: Load KModel and KPipeline instances
- `generate_audio(text)`: Convert text to audio using pipeline approach
- `cleanup()`: Free model resources

**Implementation Approach:**
- Use `KModel` and `KPipeline` classes from kokoro-demo-app.py approach
- Initialize models for both CPU and GPU (if available) like the demo app
- Use pipeline-based text processing with automatic voice pack loading
- Implement streaming-capable generation that yields audio chunks
- Handle model initialization errors gracefully

**Dependencies:**
- Kokoro library with KModel and KPipeline classes
- Voice pack files in the voices/ directory
- PyTorch and related ML dependencies
- Spaces library for GPU acceleration (optional)

### 4. Audio Playback (`AudioPlayer`)

**Purpose:** Play generated audio through system speakers

**Key Methods:**
- `play_audio(audio_data)`: Play WAV audio data
- `stop_current()`: Stop any currently playing audio
- `play_notification(sound_type)`: Play system notification sounds

**Implementation Approach:**
- Use `pygame` mixer for audio playback (lightweight, cross-platform)
- Implement audio interruption for new TTS requests
- Add simple notification sounds for feedback

**Dependencies:**
- `pygame` for audio playback
- `numpy` for audio data handling

### 5. Main Application (`MacOSTTSApp`)

**Purpose:** Coordinate all components and manage application lifecycle

**Key Methods:**
- `start()`: Initialize all components and start monitoring
- `stop()`: Clean shutdown of all components
- `handle_hotkey()`: Main event handler for hotkey triggers

**Implementation Approach:**
- Implement graceful startup and shutdown
- Handle errors and provide user feedback
- Manage component lifecycle and dependencies

## Data Models

### Audio Data
```python
@dataclass
class AudioData:
    samples: np.ndarray  # Audio samples
    sample_rate: int     # Sample rate (24000 Hz)
    duration: float      # Duration in seconds
```

### Configuration
```python
@dataclass
class AppConfig:
    model_path: str = "kokoro-v0_19.pth"
    voice_path: str = "voices/bf_isabella.pt"
    hotkey_timeout: float = 0.5  # Double-tap window
    max_text_length: int = 500   # Character limit
    device: str = "auto"         # cuda/cpu/auto
```

## Error Handling

### Model Loading Errors
- **Missing model files**: Provide clear error message with expected file paths
- **CUDA/device errors**: Fallback to CPU processing with warning
- **Memory errors**: Implement model cleanup and retry logic

### System Integration Errors
- **Accessibility permissions**: Detect and prompt user for required permissions
- **Clipboard access**: Handle clipboard access failures gracefully
- **Audio device errors**: Provide fallback audio notification methods

### Runtime Errors
- **Text processing errors**: Skip invalid text with audio notification
- **Audio generation failures**: Play error sound and log details
- **Hotkey detection failures**: Restart monitoring with exponential backoff

## Testing Strategy

### Unit Testing
- **Text Processing**: Test normalization and validation functions
- **Audio Generation**: Test TTS model integration with sample texts
- **Configuration**: Test config loading and validation

### Integration Testing
- **Hotkey Detection**: Test double Command key detection accuracy
- **Clipboard Integration**: Test text capture from various applications
- **Audio Playback**: Test audio output and interruption handling

### Manual Testing
- **Cross-Application**: Test text selection in different macOS applications
- **Performance**: Test responsiveness and resource usage
- **Error Scenarios**: Test behavior with missing files, permissions, etc.

### Testing Limitations
- Automated testing of global hotkeys requires special setup
- Audio testing requires manual verification of output quality
- macOS permission testing requires manual user interaction

## Implementation Notes

### macOS-Specific Considerations
- **Accessibility Permissions**: App requires accessibility access for global hotkey monitoring
- **Clipboard Behavior**: Some apps may not update clipboard on text selection
- **Audio Session**: Proper audio session management for background playback

### Performance Optimizations
- **Model Caching**: Keep model loaded in memory for fast response
- **Audio Buffering**: Pre-allocate audio buffers for smooth playback
- **Resource Management**: Implement proper cleanup to prevent memory leaks

### Security Considerations
- **Clipboard Access**: Only access clipboard when hotkey is triggered
- **File Permissions**: Validate model file paths and permissions
- **Process Isolation**: Run with minimal required privileges