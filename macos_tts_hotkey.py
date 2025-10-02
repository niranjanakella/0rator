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
import subprocess
import queue
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
    model_path: str = "kokoro-v1_0.pth"  # Path to local model file
    voices_dir: str = "voices"           # Path to voices directory
    voice: str = "bf_isabella"           # Voice to use (language determined by first character)
    hotkey_timeout: float = 0.5          # Double-tap window
    # No character limit (None = unlimited)
    max_text_length: int = None
    device: str = "auto"                 # cuda/cpu/auto
    speed: float = 1.0                   # TTS generation speed (0.5-2.0)


@dataclass
class AudioData:
    """Audio data container"""
    samples: np.ndarray  # Audio samples
    sample_rate: int     # Sample rate (24000 Hz)
    duration: float      # Duration in seconds


@dataclass
class AudioChunk:
    """Audio chunk data for streaming playback"""
    samples: np.ndarray  # Audio samples for this chunk
    sample_rate: int     # Sample rate (24000 Hz)
    chunk_index: int     # Order index for proper sequencing
    is_final: bool       # Whether this is the last chunk


class TTSEngine:
    """TTS Engine component using kokoro-demo-app.py approach with KModel and KPipeline"""

    def __init__(self, config: AppConfig):
        self.config = config
        self.models = {}  # Dictionary to hold CPU and GPU models
        self.pipelines = {}  # Dictionary to hold language pipelines
        self.voice = config.voice  # Use voice from config
        self.voice_pack = None  # Store loaded voice pack
        self.use_gpu = False
        self._model_loaded = False
        self._generation_lock = threading.Lock()  # Prevent concurrent generation



    def initialize_model(self):
        """Load KModel instances and KPipeline instances"""
        try:
            logger.info(
                "Initializing TTS model using kokoro-demo-app approach...")

            # Determine available devices
            cuda_available = torch.cuda.is_available()

            if self.config.device == "auto":
                self.use_gpu = cuda_available
            elif self.config.device == "cuda":
                self.use_gpu = cuda_available
                if not cuda_available:
                    logger.warning(
                        "CUDA requested but not available, falling back to CPU")
                    self.use_gpu = False
            else:
                self.use_gpu = False

            logger.info(
                f"CUDA available: {cuda_available}, Using GPU: {self.use_gpu}")

            # Initialize models with local model file
            try:
                # Check if local model file exists
                if not os.path.exists(self.config.model_path):
                    raise FileNotFoundError(
                        f"Model file not found: {self.config.model_path}")

                logger.info(f"Loading model from: {self.config.model_path}")

                # Always create CPU model with local model file
                logger.info("Loading CPU model...")
                self.models[False] = KModel(
                    repo_id='hexgrad/Kokoro-82M',
                    model=self.config.model_path
                ).to('cpu').eval()

                # Create GPU model if available and requested
                if self.use_gpu:
                    logger.info("Loading GPU model...")
                    self.models[True] = KModel(
                        repo_id='hexgrad/Kokoro-82M',
                        model=self.config.model_path
                    ).to('cuda').eval()

                logger.info("Models loaded successfully")
            except Exception as e:
                error_msg = f"""
Failed to load KModel from {self.config.model_path}: {e}

This could be due to:
- Missing model file: {self.config.model_path}
- Corrupted model file
- Incompatible PyTorch version
- Insufficient memory
- Missing kokoro library installation

Please ensure:
1. The model file exists at: {self.config.model_path}
2. You have installed kokoro-tts: pip install kokoro-tts
"""
                logger.error(error_msg)
                raise RuntimeError(error_msg)

            # Initialize pipelines like in kokoro-demo-app.py
            try:
                logger.info("Initializing pipelines...")
                
                # Initialize pipeline only for the configured voice's language
                self.pipelines = {}
                voice_lang = self.voice[0]  # First character is language code
                logger.info(f"Initializing pipeline for configured voice language: {voice_lang}")
                
                try:
                    self.pipelines[voice_lang] = KPipeline(
                        lang_code=voice_lang,
                        model=False,
                        repo_id='hexgrad/Kokoro-82M'
                    )
                    logger.info(f"Pipeline initialized successfully for language: {voice_lang}")
                except Exception as e:
                    logger.error(f"Failed to initialize pipeline for language {voice_lang}: {e}")
                    raise RuntimeError(f"Failed to initialize pipeline for configured voice language: {voice_lang}")
                
                # Add custom pronunciation for 'kokoro' if using English pipeline
                if voice_lang in ['a', 'b']:
                    if voice_lang == 'a':
                        self.pipelines[voice_lang].g2p.lexicon.golds['kokoro'] = 'kˈOkəɹO'
                    else:  # voice_lang == 'b'
                        self.pipelines[voice_lang].g2p.lexicon.golds['kokoro'] = 'kˈQkəɹQ'

                logger.info(f"Pipeline initialized successfully for language: {voice_lang}")
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

            # Validate voices directory exists
            if not os.path.exists(self.config.voices_dir):
                raise FileNotFoundError(
                    f"Voices directory not found: {self.config.voices_dir}")

            # Pre-load the default voice
            try:
                logger.info(f"Pre-loading voice: {self.voice}")
                voice_file = os.path.join(
                    self.config.voices_dir, f"{self.voice}.pt")

                if not os.path.exists(voice_file):
                    raise FileNotFoundError(
                        f"Voice file not found: {voice_file}")

                pipeline = self.pipelines[self.voice[0]]  # 'b' for bf_isabella
                # Load voice from local file path instead of downloading and store it
                self.voice_pack = pipeline.load_voice(voice_file)
                # Also cache it under the voice name for pipeline calls
                pipeline.voices[self.voice] = self.voice_pack
                logger.info("Voice loaded and cached successfully")
            except Exception as e:
                error_msg = f"""
Failed to load voice pack: {self.voice}

This could be due to:
- Missing voice file in voices/ directory: {self.config.voices_dir}/{self.voice}.pt
- Corrupted voice file
- Incompatible voice format

Check that the voices directory contains: {self.voice}.pt
"""
                logger.error(error_msg)
                raise RuntimeError(error_msg)

            # Set model as loaded before pre-warming
            self._model_loaded = True

            # Pre-warm the model with a short test
            try:
                logger.info("Pre-warming TTS model...")
                test_audio = self._generate_audio_internal("Hello")
                if test_audio is not None:
                    logger.info("Model pre-warming successful")
                else:
                    logger.warning(
                        "Model pre-warming failed, but continuing...")
            except Exception as e:
                logger.warning(
                    f"Model pre-warming failed: {e}, but continuing...")

            return True

        except Exception as e:
            logger.error(f"Failed to initialize TTS model: {e}")
            return False

    def generate_audio(self, text: str) -> Optional[AudioData]:
        """Convert text to audio with thread safety, handling long text by splitting"""
        with self._generation_lock:
            return self._generate_audio_with_splitting(text)

    def generate_audio_stream(self, text: str, chunk_size: int = 200):
        """Generate audio chunks as a stream for immediate playback"""
        with self._generation_lock:
            yield from self._generate_audio_stream_internal(text, chunk_size)

    def _generate_audio_with_splitting(self, text: str) -> Optional[AudioData]:
        """Generate audio for long text by splitting into chunks and concatenating"""
        try:
            if not self._model_loaded or not self.models or not self.pipelines:
                logger.error("Model not initialized")
                return None

            # Validate and prepare text
            if not text or not text.strip():
                logger.warning("Empty text provided")
                return None

            text = text.strip()
            logger.info(f"Generating audio for text: {len(text)} characters")

            # Use the configured voice and its pipeline
            voice_lang = self.voice[0]  # First character is language code
            pipeline = self.pipelines[voice_lang]
            voice_to_use = self.voice
            
            logger.info(f"Using pipeline: {voice_lang}, voice: {voice_to_use}")
            
            # Use pre-loaded voice pack
            pack = self.voice_pack

            # Generate audio using the demo app approach
            use_gpu = self.use_gpu and self.use_gpu in self.models

            # Collect all audio segments
            audio_segments = []
            total_duration = 0

            # Process text through pipeline (similar to generate_all function)
            for segment_idx, (_, ps, _) in enumerate(pipeline(text, voice_to_use, speed=self.config.speed)):
                ref_s = pack[len(ps)-1]

                try:
                    if use_gpu:
                        # Use GPU model
                        audio_tensor = self.models[True](
                            ps, ref_s, self.config.speed)
                    else:
                        # Use CPU model
                        audio_tensor = self.models[False](
                            ps, ref_s, self.config.speed)

                except Exception as e:
                    if use_gpu:
                        logger.warning(
                            f"GPU generation failed for segment {segment_idx}: {e}, retrying with CPU")
                        try:
                            audio_tensor = self.models[False](
                                ps, ref_s, self.config.speed)
                        except Exception as cpu_e:
                            logger.error(
                                f"CPU generation also failed for segment {segment_idx}: {cpu_e}")
                            continue  # Skip this segment but continue with others
                    else:
                        logger.error(
                            f"Audio generation failed for segment {segment_idx}: {e}")
                        continue  # Skip this segment but continue with others

                # Convert tensor to numpy array and collect
                if audio_tensor is not None:
                    audio_samples = audio_tensor.numpy()
                    audio_segments.append(audio_samples)
                    segment_duration = len(audio_samples) / 24000
                    total_duration += segment_duration
                    logger.info(
                        f"Generated segment {segment_idx + 1}: {segment_duration:.2f}s")

                    # Clear GPU cache after each segment if using CUDA
                    if use_gpu:
                        torch.cuda.empty_cache()
                else:
                    logger.warning(
                        f"Segment {segment_idx} returned None, skipping")

            # Check if we got any audio segments
            if not audio_segments:
                logger.error("No audio segments were generated successfully")
                return None

            # Concatenate all audio segments
            if len(audio_segments) == 1:
                # Single segment, no concatenation needed
                final_audio = audio_segments[0]
            else:
                # Multiple segments, concatenate them
                logger.info(
                    f"Concatenating {len(audio_segments)} audio segments")
                final_audio = np.concatenate(audio_segments)

            # Create final AudioData object
            audio_data = AudioData(
                samples=final_audio,
                sample_rate=24000,
                duration=total_duration
            )

            logger.info(
                f"Audio generation completed: {len(audio_segments)} segments, {total_duration:.2f}s total")
            return audio_data

        except Exception as e:
            logger.error(f"Audio generation with splitting failed: {e}")
            return None

    def _generate_audio_internal(self, text: str) -> Optional[AudioData]:
        """Internal audio generation method for single segments (used for testing)"""
        try:
            if not self._model_loaded or not self.models or not self.pipelines:
                logger.error("Model not initialized")
                return None

            # Validate and prepare text
            if not text or not text.strip():
                logger.warning("Empty text provided")
                return None

            text = text.strip()
            logger.info(f"Generating audio for: {text[:50]}...")

            # Use the pipeline approach like in kokoro-demo-app.py
            # Get pipeline for voice language
            pipeline = self.pipelines[self.voice[0]]
            # Use pre-loaded voice pack
            pack = self.voice_pack

            # Generate audio using the demo app approach
            use_gpu = self.use_gpu and self.use_gpu in self.models

            # Process text through pipeline (similar to generate_first function)
            for _, ps, _ in pipeline(text, self.voice, speed=self.config.speed):
                ref_s = pack[len(ps)-1]

                try:
                    if use_gpu:
                        # Use GPU model
                        audio_tensor = self.models[True](
                            ps, ref_s, self.config.speed)
                    else:
                        # Use CPU model
                        audio_tensor = self.models[False](
                            ps, ref_s, self.config.speed)

                except Exception as e:
                    if use_gpu:
                        logger.warning(
                            f"GPU generation failed: {e}, retrying with CPU")
                        try:
                            audio_tensor = self.models[False](
                                ps, ref_s, self.config.speed)
                        except Exception as cpu_e:
                            logger.error(
                                f"CPU generation also failed: {cpu_e}")
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

                    logger.info(
                        f"Audio generated successfully: {audio_data.duration:.2f}s")

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

    def _split_text_simple(self, text: str, chunk_size: int = 200) -> list[str]:
        """Simple text splitting method for streaming"""
        try:
            if not text or len(text) <= chunk_size:
                return [text] if text else []

            import re

            # Split text into sentences using common sentence endings
            sentences = re.split(r'(?<=[.!?])\s+', text)

            chunks = []
            current_chunk = ""

            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue

                # If adding this sentence would exceed chunk size
                if current_chunk and len(current_chunk) + len(sentence) + 1 > chunk_size:
                    # Save current chunk and start new one
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = sentence
                else:
                    # Add sentence to current chunk
                    if current_chunk:
                        current_chunk += " " + sentence
                    else:
                        current_chunk = sentence

            # Add the last chunk if it has content
            if current_chunk:
                chunks.append(current_chunk.strip())

            # If no sentence boundaries were found, split by character count
            if not chunks and text:
                chunks = [text[i:i+chunk_size]
                          for i in range(0, len(text), chunk_size)]

            return chunks

        except Exception as e:
            logger.error(f"Error splitting text into chunks: {e}")
            return [text] if text else []

    def _generate_audio_stream_internal(self, text: str, chunk_size: int = 200):
        """Internal method to generate audio chunks as a stream"""
        try:
            if not self._model_loaded or not self.models or not self.pipelines:
                logger.error("Model not initialized")
                return

            # Validate and prepare text
            if not text or not text.strip():
                logger.warning("Empty text provided")
                return

            text = text.strip()
            logger.info(
                f"Generating streaming audio for text: {len(text)} characters")

            # Split text into chunks using sentence boundaries
            text_chunks = self._split_text_simple(text, chunk_size)

            if not text_chunks:
                logger.warning("No text chunks to process")
                return

            # Use the configured voice and its pipeline
            voice_lang = self.voice[0]  # First character is language code
            pipeline = self.pipelines[voice_lang]
            voice_to_use = self.voice
            
            logger.info(f"Using streaming pipeline: {voice_lang}, voice: {voice_to_use}")
            
            # Use pre-loaded voice pack
            pack = self.voice_pack
                
            use_gpu = self.use_gpu and self.use_gpu in self.models

            # Process each chunk and yield audio as soon as it's ready
            for chunk_index, text_chunk in enumerate(text_chunks):
                try:
                    logger.info(
                        f"Processing chunk {chunk_index + 1}/{len(text_chunks)}: {len(text_chunk)} chars")

                    # Collect audio segments for this chunk
                    chunk_audio_segments = []

                    # Process text chunk through pipeline
                    for segment_idx, (_, ps, _) in enumerate(pipeline(text_chunk, voice_to_use, speed=self.config.speed)):
                        ref_s = pack[len(ps)-1]

                        try:
                            if use_gpu:
                                audio_tensor = self.models[True](
                                    ps, ref_s, self.config.speed)
                            else:
                                audio_tensor = self.models[False](
                                    ps, ref_s, self.config.speed)

                        except Exception as e:
                            if use_gpu:
                                logger.warning(
                                    f"GPU generation failed for chunk {chunk_index}, segment {segment_idx}: {e}, retrying with CPU")
                                try:
                                    audio_tensor = self.models[False](
                                        ps, ref_s, self.config.speed)
                                except Exception as cpu_e:
                                    logger.error(
                                        f"CPU generation also failed for chunk {chunk_index}, segment {segment_idx}: {cpu_e}")
                                    continue
                            else:
                                logger.error(
                                    f"Audio generation failed for chunk {chunk_index}, segment {segment_idx}: {e}")
                                continue

                        # Convert tensor to numpy array and collect
                        if audio_tensor is not None:
                            audio_samples = audio_tensor.numpy()
                            chunk_audio_segments.append(audio_samples)

                            # Clear GPU cache after each segment if using CUDA
                            if use_gpu:
                                torch.cuda.empty_cache()
                        else:
                            logger.warning(
                                f"Chunk {chunk_index}, segment {segment_idx} returned None, skipping")

                    # Concatenate segments for this chunk
                    if chunk_audio_segments:
                        if len(chunk_audio_segments) == 1:
                            chunk_audio = chunk_audio_segments[0]
                        else:
                            chunk_audio = np.concatenate(chunk_audio_segments)

                        # Create AudioChunk and yield it immediately
                        audio_chunk = AudioChunk(
                            samples=chunk_audio,
                            sample_rate=24000,
                            chunk_index=chunk_index,
                            is_final=(chunk_index == len(text_chunks) - 1)
                        )

                        logger.info(
                            f"Yielding chunk {chunk_index + 1}/{len(text_chunks)}: {len(chunk_audio)/24000:.2f}s")
                        yield audio_chunk
                    else:
                        logger.warning(
                            f"No audio generated for chunk {chunk_index}")

                except Exception as e:
                    logger.error(f"Error processing chunk {chunk_index}: {e}")
                    continue

            logger.info("Streaming audio generation completed")

        except Exception as e:
            logger.error(f"Streaming audio generation failed: {e}")

    def cleanup(self):
        """Free model resources"""
        try:
            if self.models:
                # Clear CUDA cache if using GPU
                if self.use_gpu:
                    torch.cuda.empty_cache()
                self.models.clear()

            # Clear voice pack
            self.voice_pack = None

            # Clear pipeline voice caches
            for pipeline in self.pipelines.values():
                pipeline.voices.clear()

            logger.info("Model and voice resources cleaned up")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


class AudioPlayer:
    """Audio playback component using pygame"""

    def __init__(self):
        self.mixer_initialized = False
        self.current_sound = None
        self._playback_lock = threading.Lock()  # Prevent concurrent playback
        self._audio_cache = {}  # Cache for notification sounds
        self._audio_queue = None  # Queue for streaming audio chunks
        self._streaming_active = False  # Flag for streaming playback
        self._streaming_thread = None  # Thread for streaming playback

    def initialize(self):
        """Initialize pygame mixer"""
        try:
            pygame.mixer.pre_init(frequency=24000, size=-
                                  16, channels=1, buffer=1024)
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
            sf.write(buffer, audio_samples,
                     audio_data.sample_rate, format='WAV')
            buffer.seek(0)

            # Load and play the sound
            self.current_sound = pygame.mixer.Sound(buffer)
            self.current_sound.play()

            logger.info(f"Playing audio: {audio_data.duration:.2f}s")
            return True

        except Exception as e:
            logger.error(f"Failed to play audio: {e}")
            return False

    def _play_audio_chunk_internal(self, audio_data: AudioData):
        """Internal audio playback method for streaming chunks (doesn't stop current stream)"""
        try:
            if not self.mixer_initialized:
                logger.error("Audio player not initialized")
                return False

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
            sf.write(buffer, audio_samples,
                     audio_data.sample_rate, format='WAV')
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
        """Stop any currently playing audio and clear streaming queue"""
        try:
            # Stop streaming playback
            self._streaming_active = False

            # Clear the audio queue if it exists
            if self._audio_queue:
                try:
                    while not self._audio_queue.empty():
                        self._audio_queue.get_nowait()
                except queue.Empty:
                    pass

            if self.mixer_initialized:
                pygame.mixer.stop()
                self.current_sound = None

            # Wait for streaming thread to finish, but avoid joining current thread
            if (self._streaming_thread and
                self._streaming_thread.is_alive() and
                    self._streaming_thread != threading.current_thread()):
                self._streaming_thread.join(timeout=1.0)

        except Exception as e:
            logger.error(f"Error stopping audio: {e}")

    def play_audio_stream(self, audio_chunks_generator):
        """Play streaming audio chunks with parallel processing"""
        try:
            if not self.mixer_initialized:
                logger.error("Audio player not initialized")
                return False

            # Stop any current non-streaming playback
            if self.mixer_initialized:
                pygame.mixer.stop()
                self.current_sound = None

            # Create a queue for audio chunks
            self._audio_queue = queue.Queue(maxsize=5)  # Buffer up to 5 chunks
            self._streaming_active = True

            # Start producer thread (TTS generation)
            producer_thread = threading.Thread(
                target=self._audio_producer_worker,
                args=(audio_chunks_generator,),
                daemon=True,
                name="AudioProducer"
            )

            # Start consumer thread (audio playback)
            self._streaming_thread = threading.Thread(
                target=self._audio_consumer_worker,
                daemon=True,
                name="AudioConsumer"
            )

            producer_thread.start()
            self._streaming_thread.start()

            logger.info(
                "Started streaming audio playback with parallel processing")
            return True

        except Exception as e:
            logger.error(f"Failed to start streaming playback: {e}")
            return False

    def _audio_producer_worker(self, audio_chunks_generator):
        """Producer thread that generates audio chunks and puts them in queue"""
        try:
            for audio_chunk in audio_chunks_generator:
                if not self._streaming_active:
                    logger.info("Audio production interrupted")
                    break

                try:
                    # Convert AudioChunk to AudioData for playback
                    audio_data = AudioData(
                        samples=audio_chunk.samples,
                        sample_rate=audio_chunk.sample_rate,
                        duration=len(audio_chunk.samples) /
                        audio_chunk.sample_rate
                    )

                    # Put the audio data in the queue (this will block if queue is full)
                    self._audio_queue.put(
                        (audio_data, audio_chunk.chunk_index, audio_chunk.is_final), timeout=10)
                    logger.info(
                        f"Queued chunk {audio_chunk.chunk_index + 1} for playback")

                except Exception as e:
                    logger.error(
                        f"Error processing audio chunk {audio_chunk.chunk_index}: {e}")
                    continue

            # Signal end of production
            if self._streaming_active:
                self._audio_queue.put(
                    (None, -1, True), timeout=5)  # Sentinel value

        except Exception as e:
            logger.error(f"Audio producer worker error: {e}")
        finally:
            logger.info("Audio production completed")

    def _audio_consumer_worker(self):
        """Consumer thread that plays audio chunks from the queue"""
        try:
            first_chunk = True

            while self._streaming_active:
                try:
                    # Get audio data from queue (blocks until available)
                    item = self._audio_queue.get(timeout=1.0)

                    if item is None or item[0] is None:  # Sentinel value
                        logger.info("Received end-of-stream signal")
                        break

                    audio_data, chunk_index, is_final = item

                    if first_chunk:
                        logger.info("Playing first audio chunk immediately")
                        first_chunk = False

                    # Play this chunk
                    self._play_audio_chunk_internal(audio_data)

                    # Wait for this chunk to finish playing before getting the next
                    if self.current_sound:
                        while pygame.mixer.get_busy() and self._streaming_active:
                            # Small sleep to avoid busy waiting
                            time.sleep(0.01)

                    logger.info(f"Completed chunk {chunk_index + 1}")

                    if is_final:
                        logger.info("Completed final audio chunk")
                        break

                except queue.Empty:
                    # Timeout waiting for next chunk, continue checking
                    continue
                except Exception as e:
                    logger.error(f"Error in audio consumer: {e}")
                    continue

            self._streaming_active = False
            logger.info("Streaming playback completed")

        except Exception as e:
            logger.error(f"Audio consumer worker error: {e}")
            self._streaming_active = False

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
                padding = np.zeros(len(t) - len(beep1) -
                                   len(silence) - len(beep2))
                audio_samples = np.concatenate(
                    [beep1, silence, beep2, padding])
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
        """Retrieve currently selected text using native macOS Accessibility API"""
        try:
            # Try native Accessibility API first (fastest)
            selected_text = self._get_selected_text_native()
            if selected_text:
                return selected_text

            # Fallback to clipboard method if native fails
            return self._get_selected_text_clipboard()

        except Exception as e:
            logger.error(f"Error getting selected text: {e}")
            return None

    def _get_selected_text_native(self) -> Optional[str]:
        """Get selected text using native macOS Accessibility API"""
        try:
            import Cocoa
            import Quartz
            from AppKit import NSWorkspace

            # Get the currently focused application
            workspace = NSWorkspace.sharedWorkspace()
            active_app = workspace.frontmostApplication()

            if not active_app:
                logger.debug("No active application found")
                return None

            # Get the application's process ID
            pid = active_app.processIdentifier()

            # Create accessibility object for the application
            app_ref = Quartz.AXUIElementCreateApplication(pid)
            if not app_ref:
                logger.debug("Could not create accessibility reference")
                return None

            # Get the focused UI element
            focused_result = Quartz.AXUIElementCopyAttributeValue(
                app_ref, Quartz.kAXFocusedUIElementAttribute, None
            )

            if focused_result[0] != 0 or not focused_result[1]:
                logger.debug("No focused UI element found")
                return None

            focused_element = focused_result[1]

            # Try to get selected text directly
            try:
                selected_text_result = Quartz.AXUIElementCopyAttributeValue(
                    focused_element, Quartz.kAXSelectedTextAttribute, None
                )

                if selected_text_result[0] == 0 and selected_text_result[1]:
                    selected_text = str(selected_text_result[1])
                    if selected_text and selected_text.strip():
                        logger.info(
                            f"Retrieved selected text via Accessibility API: {selected_text[:50]}...")
                        return selected_text.strip()

            except Exception as e:
                logger.debug(f"Direct selected text access failed: {e}")

            # Try alternative: get text value and selection range
            try:
                # Get the full text value
                text_result = Quartz.AXUIElementCopyAttributeValue(
                    focused_element, Quartz.kAXValueAttribute, None
                )

                # Get selection range
                range_result = Quartz.AXUIElementCopyAttributeValue(
                    focused_element, Quartz.kAXSelectedTextRangeAttribute, None
                )

                if (text_result[0] == 0 and text_result[1] and
                        range_result[0] == 0 and range_result[1]):

                    full_text = str(text_result[1])
                    selection_range = range_result[1]

                    # Extract location and length from CFRange
                    location = selection_range.location
                    length = selection_range.length

                    if length > 0 and location >= 0 and location + length <= len(full_text):
                        selected_text = full_text[location:location + length]
                        if selected_text and selected_text.strip():
                            logger.info(
                                f"Retrieved selected text via range: {selected_text[:50]}...")
                            return selected_text.strip()

            except Exception as e:
                logger.debug(f"Range-based text access failed: {e}")

            return None

        except ImportError:
            logger.debug(
                "PyObjC not available, falling back to clipboard method")
            return None
        except Exception as e:
            logger.debug(f"Native text access failed: {e}")
            return None

    def _get_selected_text_clipboard(self) -> Optional[str]:
        """Fallback method using clipboard with Core Graphics events"""
        try:
            # Store current clipboard content to restore later
            original_clipboard = None
            try:
                original_clipboard = pyperclip.paste()
            except Exception as e:
                logger.warning(f"Could not access original clipboard: {e}")

            # Use Core Graphics to send Cmd+C event (faster than subprocess)
            try:
                import Quartz

                # Create Cmd+C key event using Core Graphics
                cmd_down = Quartz.CGEventCreateKeyboardEvent(
                    None, 0x37, True)  # Cmd key down
                c_down = Quartz.CGEventCreateKeyboardEvent(
                    None, 0x08, True)   # C key down
                c_up = Quartz.CGEventCreateKeyboardEvent(
                    None, 0x08, False)    # C key up
                cmd_up = Quartz.CGEventCreateKeyboardEvent(
                    None, 0x37, False)  # Cmd key up

                # Set Cmd modifier for C key events
                Quartz.CGEventSetFlags(c_down, Quartz.kCGEventFlagMaskCommand)
                Quartz.CGEventSetFlags(c_up, Quartz.kCGEventFlagMaskCommand)

                # Post the events
                Quartz.CGEventPost(Quartz.kCGHIDEventTap, cmd_down)
                Quartz.CGEventPost(Quartz.kCGHIDEventTap, c_down)
                Quartz.CGEventPost(Quartz.kCGHIDEventTap, c_up)
                Quartz.CGEventPost(Quartz.kCGHIDEventTap, cmd_up)

                # Small delay to ensure clipboard is updated
                time.sleep(0.05)  # Reduced delay for speed

                # Get the copied text from clipboard
                selected_text = pyperclip.paste()

                # Check if we got new text (different from original clipboard)
                if selected_text and selected_text.strip():
                    # If the text is the same as original clipboard, no text was selected
                    if original_clipboard and selected_text == original_clipboard:
                        logger.info(
                            "No new text was selected (clipboard unchanged)")
                        return None

                    logger.info(
                        f"Retrieved selected text via clipboard: {selected_text[:50]}...")

                    # Restore original clipboard content if it was different
                    try:
                        if original_clipboard and original_clipboard != selected_text:
                            # Small delay before restoring
                            time.sleep(0.05)
                            pyperclip.copy(original_clipboard)
                    except Exception as e:
                        logger.warning(
                            f"Could not restore original clipboard: {e}")

                    return selected_text.strip()
                else:
                    logger.info("No text was selected or copied")
                    return None

            except ImportError:
                logger.warning("Quartz not available, using pynput fallback")
                # Final fallback to pynput if Quartz is not available
                from pynput.keyboard import Key, Controller

                keyboard_controller = Controller()
                keyboard_controller.press(Key.cmd)
                keyboard_controller.press('c')
                keyboard_controller.release('c')
                keyboard_controller.release(Key.cmd)

                time.sleep(0.1)
                selected_text = pyperclip.paste()

                if selected_text and selected_text.strip() and selected_text != original_clipboard:
                    if original_clipboard:
                        time.sleep(0.1)
                        pyperclip.copy(original_clipboard)
                    return selected_text.strip()
                return None

            except Exception as e:
                logger.error(f"Failed to simulate copy keystroke: {e}")
                return None

        except Exception as e:
            logger.error(f"Clipboard fallback failed: {e}")
            return None

    def validate_text(self, text: str) -> bool:
        """Check if text is suitable for TTS"""
        if not text or not text.strip():
            return False

        # No length restrictions - accept text of any length
        logger.info(f"Validating text: {len(text)} characters")

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

            # Handle newlines by adding periods where needed before converting to spaces
            import re

            # Process each newline: add period if the character before newline is not already punctuation
            def process_newlines(text):
                lines = text.split('\n')
                processed_lines = []

                for i, line in enumerate(lines):
                    line = line.strip()
                    if line:  # Only process non-empty lines
                        # Check if this line needs a period at the end
                        if i < len(lines) - 1:  # Not the last line
                            # Check if line ends with punctuation
                            if line and not line[-1] in '.!?;:':
                                line += '.'
                        processed_lines.append(line)

                return ' '.join(processed_lines)

            # Apply newline processing
            text = process_newlines(text)

            # Remove excessive whitespace (after newline processing)
            text = re.sub(r'\s+', ' ', text)

            # Remove or replace problematic characters
            # Remove control characters except newlines and tabs (newlines already processed)
            text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)

            # Replace any remaining tabs with spaces
            text = text.replace('\t', ' ')

            # Final cleanup of excessive whitespace
            text = re.sub(r'\s+', ' ', text).strip()

            # No truncation - process text of any length
            logger.info(
                f"Text prepared for processing: {len(text)} characters")

            # Final validation
            if not self.validate_text(text):
                return None

            return text

        except Exception as e:
            logger.error(f"Error preparing text: {e}")
            return None

    def split_text_into_chunks(self, text: str, chunk_size: int = 200) -> list[str]:
        """Split long text into chunks at sentence boundaries for streaming"""
        try:
            if not text or len(text) <= chunk_size:
                return [text] if text else []

            import re

            # Split text into sentences using common sentence endings
            sentences = re.split(r'(?<=[.!?])\s+', text)

            chunks = []
            current_chunk = ""

            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue

                # If adding this sentence would exceed chunk size
                if current_chunk and len(current_chunk) + len(sentence) + 1 > chunk_size:
                    # Save current chunk and start new one
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = sentence
                else:
                    # Add sentence to current chunk
                    if current_chunk:
                        current_chunk += " " + sentence
                    else:
                        current_chunk = sentence

            # Add the last chunk if it has content
            if current_chunk:
                chunks.append(current_chunk.strip())

            # If no sentence boundaries were found, split by character count
            if not chunks and text:
                chunks = [text[i:i+chunk_size]
                          for i in range(0, len(text), chunk_size)]

            logger.info(
                f"Split text into {len(chunks)} chunks (chunk_size: {chunk_size})")
            return chunks

        except Exception as e:
            logger.error(f"Error splitting text into chunks: {e}")
            return [text] if text else []


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
                logger.info(
                    "Hotkey monitoring started - listening for double Option key press")
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

    def process_streaming_tts(self, text: str):
        """Process TTS with streaming audio playback"""
        try:
            # Get components
            tts_engine = self.components.get('tts_engine')
            audio_player = self.components.get('audio_player')

            if not all([tts_engine, audio_player]):
                logger.error("TTS engine or audio player not available")
                return False

            # Print the text that will be processed
            print("\n" + "="*60)
            print("TEXT TO BE PROCESSED (STREAMING):")
            print("="*60)
            print(text)
            print("="*60)
            print(f"Text length: {len(text)} characters")
            print("="*60 + "\n")

            # Use streaming generation and playback
            logger.info("Starting streaming TTS generation and playback...")

            # Generate audio stream with default chunk size (200 characters)
            audio_chunks_generator = tts_engine.generate_audio_stream(text)

            # Start streaming playback
            if audio_player.play_audio_stream(audio_chunks_generator):
                logger.info("Streaming TTS playback started successfully")
                return True
            else:
                logger.error("Failed to start streaming playback")
                return False

        except Exception as e:
            logger.error(f"Error in streaming TTS processing: {e}")
            return False

    def handle_hotkey(self):
        """Main event handler for hotkey triggers with streaming support"""
        # Run in background thread to avoid blocking the hotkey listener
        def process_hotkey():
            logger.info("Hotkey detected - processing...")

            try:
                # Get components
                text_processor = self.components.get('text_processor')
                audio_player = self.components.get('audio_player')

                if not all([text_processor, audio_player]):
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
                    logger.warning(
                        "Text preparation failed - playing error notification")
                    audio_player.play_notification("error")
                    return

                # Step 3: Use streaming TTS for all text processing
                logger.info("Processing TTS with streaming...")
                if not self.process_streaming_tts(prepared_text):
                    logger.error(
                        "Streaming TTS failed - playing error notification")
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
        threading.Thread(target=process_hotkey, daemon=True,
                         name="HotkeyProcessor").start()


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

            # Load and validate speed parameter
            speed = config_data.get("speed", 1.0)

            # Validate and clamp speed to 0.5-2.0 range
            if not isinstance(speed, (int, float)):
                logger.warning(
                    f"Invalid speed type: {type(speed)}, using default 1.0")
                speed = 1.0
            elif speed < 0.5:
                logger.warning(f"Speed {speed} too low, clamping to 0.5")
                speed = 0.5
            elif speed > 2.0:
                logger.warning(f"Speed {speed} too high, clamping to 2.0")
                speed = 2.0

            # Create AppConfig with loaded values
            config = AppConfig(
                voice=config_data.get("voice", "bf_isabella"),
                hotkey_timeout=config_data.get("hotkey_timeout", 0.5),
                max_text_length=config_data.get("max_text_length", None),
                device=config_data.get("device", "auto"),
                speed=speed
            )

            # Set log level if specified
            log_level = config_data.get("log_level", "INFO")
            logging.getLogger().setLevel(getattr(logging, log_level.upper(), logging.INFO))

            logger.info(
                f"Configuration loaded from {config_file} (speed: {speed})")
            return config

        except Exception as e:
            logger.warning(f"Failed to load config from {config_file}: {e}")
            logger.info("Using default configuration")
    else:
        logger.info("No config file found, using default configuration")

    return AppConfig()


def main():
    """Application entry point"""
    logger.info("macOS TTS Hotkey Application")
    logger.info("=" * 40)

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
            logger.error(
                "Failed to start application. Check the error messages above.")
            sys.exit(1)

        logger.info("\n✅ Application started successfully!")
        logger.info("\nUsage:")
        logger.info("1. Select any text in any application")
        logger.info("2. Double-tap the Option key quickly (within 0.5 seconds)")
        logger.info("3. Listen to the text being read aloud")
        logger.info("\nTips:")
        logger.info("- You'll hear a beep when the hotkey is detected")
        logger.info("- Double beep means no text was selected")
        logger.info("- Low beep means an error occurred")
        logger.info("- Works with text from any macOS application")
        logger.info(
            "- Supports arbitrarily long text (automatically split into segments)")
        logger.info("Press Ctrl+C to exit.")

        # Keep application running
        while app.running:
            time.sleep(0.1)

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
        logger.info("\n👋 Shutting down...")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        logger.error(f"\n❌ Unexpected error: {e}")
    finally:
        app.stop()
        logger.info("Application stopped.")


if __name__ == "__main__":
    main()
