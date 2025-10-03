#  Orator TTS Engine

<div align="center">
<img src="0rator.png" height="128">



![Kokoro TTS](https://img.shields.io/badge/Kokoro-TTS-blue?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.11+-green?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red?style=for-the-badge&logo=pytorch)
![PyTorch](https://img.shields.io/badge/Espeak-1.51+-orange?style=for-the-badge)
![macOS](https://img.shields.io/badge/macOS-Compatible-black?style=for-the-badge&logo=apple)

*High-quality neural text-to-speech with 50+ voices across 9 languages*

[![Orator Demo](https://img.youtube.com/vi/4V6Yej99URc/0.jpg)](https://www.youtube.com/watch?v=4V6Yej99URc)

Click for Demo ☝️

</div>

## ✨ Features

- 🌍 **8 Languages**: American English, British English, Spanish, French, Hindi, Italian, Portuguese, Japanese, Chinese
- 🎭 **50+ Voices**: Male and female voices with unique personalities
- ⚡ **Lightning Fast**: GPU-accelerated inference with streaming audio
- 🎯 **macOS Hotkey**: Double-tap Option key ⌥ for instant TTS anywhere
- 🔊 **High Quality**: Super high quality neural audio synthesis
- 🚀 **Easy Setup**: Installation through UV package manager
- 📱 **System-Wide**: Works with any macOS application

## 🗣️ Available Voices & Languages
<details>
<summary>Voices/Languages Available</summary>

### 🇺🇸 American English (a)
**Female Voices:**
- `af_alloy`
- `af_aoede`
- `af_bella`
- `af_heart`
- `af_jessica`
- `af_kore`
- `af_nicole`
- `af_nova`
- `af_river`
- `af_sarah`
- `af_sky`

**Male Voices:**
- `am_adam`
- `am_echo`
- `am_eric`
- `am_fenrir`
- `am_liam`
- `am_michael`
- `am_onyx`
- `am_puck`
- `am_santa`

### 🇬🇧 British English (b)
**Female Voices:**
- `bf_alice`
- `bf_emma`
- `bf_isabella`
- `bf_lily`

**Male Voices:**
- `bm_daniel`
- `bm_fable`
- `bm_george`
- `bm_lewis`

### 🇪🇸 Spanish (e)
**Female Voices:**
- `ef_dora`

**Male Voices:**
- `em_alex`
- `em_santa`

### 🇫🇷 French (f)
**Female Voices:**
- `ff_siwis`

### 🇮🇳 Hindi (h)
**Female Voices:**
- `hf_alpha`
- `hf_beta`

**Male Voices:**
- `hm_omega`
- `hm_psi`

### 🇮🇹 Italian (i)
**Female Voices:**
- `if_sara`

**Male Voices:**
- `im_nicola`

### 🇯🇵 Japanese (j)
**Female Voices:**
- `jf_alpha`
- `jf_gongitsune`
- `jf_nezumi`
- `jf_tebukuro`

**Male Voices:**
- `jm_kumo`

### 🇧🇷 Portuguese (p)
**Female Voices:**
- `pf_dora`

**Male Voices:**
- `pm_alex`
- `pm_santa`

### 🇨🇳 Chinese (z)
**Female Voices:**
- `zf_xiaobei`
- `zf_xiaoni`
- `zf_xiaoxiao`
- `zf_xiaoyi`

**Male Voices:**
- `zm_yunjian`
- `zm_yunxi`
- `zm_yunxia`
- `zm_yunyang`
</details>

## 🚀 Quick Start

### Why UV? The Future of Python Package Management

We recommend **UV** for this project because it's:
- ⚡ **10-100x faster** than pip
- 🔒 **More secure** with built-in dependency resolution
- 🎯 **Zero configuration** - works out of the box
- 🔄 **Drop-in replacement** for pip/pipenv/poetry
- 🌟 **Industry standard** - used by major Python projects

### Installation

1. **Install UV (if you don't have it):**
- Assumed that python is already installed on your system.
   ```bash
   pip install uv 
   ```

2. **Clone and setup the project:**
   ```bash
   # Clone repo
   cd Orator
   
   # Create virtual environment and install dependencies
   uv venv --python=3.11
   source .venv/bin/activate  # On macOS/Linux
   uv pip install -r requirements.txt
   ```

3. **Install espeak-ng (required for phonemization):**
   ```bash
   # macOS
   brew install espeak-ng
   
   # Verify installation
   espeak-ng --version
   
   #eSpeak NG text-to-speech: 1.51  Data at: /opt/homebrew/Cellar/espeak-ng/1.51/share/espeak-ng-data
   ```

4. **Download [model](https://huggingface.co/hexgrad/Kokoro-82M) and voices** (if not included):
   ```bash
   uv pip install -U "huggingface_hub[cli]"

   # Download model
   huggingface-cli download hexgrad/Kokoro-82M --include "kokoro-v1_0.pth" --local-dir .

   # Download voices
   huggingface-cli download hexgrad/Kokoro-82M --include "voices/*" --local-dir .
   ```
5. **Language Pack**
- By default "en-core-web-sm" is installed through requirements for English, navigate and install other small language packs from [spaCy](https://spacy.io/models/en).

## 🎯 Usage

### 1. macOS Hotkey Application

**Grant Accessibility Permissions First:**
1. Open System Preferences → Security & Privacy → Privacy
2. Select "Accessibility" from the left panel
3. Click the lock icon and enter your password
4. Add your terminal application (Terminal.app, iTerm2, etc.)
5. Ensure it's checked/enabled

**Run the hotkey application:**
```bash
# Make sure your are inside the virtual environment
python3 macos_tts_hotkey.py
```

**How to use:**
- Select any text in any macOS application
- Double-tap the Option key (⌥) quickly to start TTS
- Press **Escape** key to stop TTS playback at any time
- Listen to the text being read aloud!

## ⚙️ Configuration

### Hotkey Application Config

Edit `config_hotkey.json`:
```json
{
    "model_path": "kokoro-v1_0.pth",
    "voices_dir": "voices",
    "voice": "af_bella",
    "speed": 1.0,
    "device": "auto"
}
```

### Voice Selection

Choose voices by language prefix:
- `af_*` / `am_*` - American English
- `bf_*` / `bm_*` - British English  
- `ef_*` / `em_*` - Spanish
- `ff_*` - French
- `hf_*` / `hm_*` - Hindi
- `if_*` / `im_*` - Italian
- `jf_*` / `jm_*` - Japanese
- `pf_*` / `pm_*` - Portuguese
- `zf_*` / `zm_*` - Chinese

## 🔧 Advanced Usage

### Multi-language Support

```python
# Create pipelines for different languages
en_pipeline = KPipeline(lang_code='a', model=model)  # American English
es_pipeline = KPipeline(lang_code='e', model=model)  # Spanish
ja_pipeline = KPipeline(lang_code='j', model=model)  # Japanese

# Use appropriate pipeline for each language
english_audio = list(en_pipeline("Hello world!", voice="af_bella"))[0].audio
spanish_audio = list(es_pipeline("¡Hola mundo!", voice="ef_dora"))[0].audio
japanese_audio = list(ja_pipeline("こんにちは世界！", voice="jf_alpha"))[0].audio
```

## 🛠️ Troubleshooting

### Common Issues

**"Failed to start keyboard monitoring"**
- Grant Accessibility permissions in System Preferences
- Restart the application after granting permissions

**"espeak-ng not found"**
```bash
# Install espeak-ng
brew install espeak-ng

# Verify installation
which espeak-ng
```

**"Model file not found"**
- Ensure `kokoro-v1_0.pth` is in the project root
- Check file permissions and path

**"CUDA out of memory"**
```python
# Use CPU instead
config.device = "cpu"

# Or reduce batch size for long texts
```

**"Voice file not found"**
- Ensure voice files are in the `voices/` directory
- Check that the voice name matches exactly (case-sensitive)

**"Stop functionality not working"**
- Ensure the application has focus or accessibility permissions
- Try pressing Escape key while TTS is actively playing
- Check terminal logs for any error messages

### Performance Optimization

- **GPU Usage**: Automatic CUDA detection, falls back to CPU
- **Memory Management**: Automatic cleanup after generation
- **Streaming**: Use `generate_audio_stream()` for long texts
- **Caching**: Voice packs are cached after first load

## 📁 Project Structure

```
Orator/
├── kokoro/               # Core TTS library
│   ├── __init__.py
│   ├── model.py          # KModel implementation
│   ├── pipeline.py       # KPipeline
│   └── ...
├── voices/               # Voice pack files (.pt)
│   ├── af_bella.pt
│   ├── am_adam.pt
│   └── ...
├── macos_tts_hotkey.py   # macOS hotkey application
├── kokoro-v1_0.pth       # Main TTS model
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

## 🤝 Contributing

We welcome contributions! Please feel free to:
- Report bugs and issues
- Suggest new features
- Submit pull requests
- Add new voice packs
- Improve documentation


## 🗺️ Roadmap

<div align="center">

</div>

- [x] Streaming Audio chunks for Long Formers (Controlled low latency)
- [x] Speed Controls for Audio Stream
- [ ] LLM driven Agentic AI Capabilities
- [ ] Native MacOS application/interface for UI driven audio controlls
- [ ] UI voice swap controlls

---

### 🤝 Get Involved

Want to help shape the future of Kokoro TTS? Here's how:

- 🐛 **Report Issues** - Help us identify bugs and improvements
- 💡 **Suggest Features** - Share your ideas for new functionality  
- 🔧 **Contribute Code** - Submit PRs for features or fixes
- 🎨 **Design UI/UX** - Help design the native app interface
- 📝 **Write Documentation** - Improve guides and tutorials
- 🗣️ **Add Voices** - Contribute new voice packs and languages

## 🙏 Acknowledgments

- Built on the amazing [Kokoro TTS](https://github.com/hexgrad/kokoro) model
- Powered by PyTorch and modern neural architectures
- Inspired by the need for accessible, high-quality TTS

---

<div align="center">

**Made with ❤️ for the open-source community**

[![LinkedIn](https://img.shields.io/badge/Reach_Me_Out-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/niranjanakella/)
</div>