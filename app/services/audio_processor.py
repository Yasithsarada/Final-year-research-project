import logging
import os
import tempfile
from pathlib import Path
from typing import Tuple, List, Dict, Any
from app.core.config import settings
from app.core.exceptions import ExtractionException

logger = logging.getLogger(__name__)

class AudioProcessor:
    """Wrapper around faster-whisper library to handle local transcription.
    
    Supports WAV, MP3, and M4A. Automatically handles model loading and caching.
    """
    _model_instance = None

    @classmethod
    def _get_model(cls):
        """Lazy loading of the Whisper model to avoid overhead if not used."""
        if cls._model_instance is None:
            model_size = settings.WHISPER_MODEL
            device = settings.WHISPER_DEVICE
            compute_type = settings.WHISPER_COMPUTE_TYPE
            
            logger.info(f"Loading Whisper model '{model_size}' on device '{device}' with compute type '{compute_type}'...")
            try:
                from faster_whisper import WhisperModel
                cls._model_instance = WhisperModel(
                    model_size,
                    device=device,
                    compute_type=compute_type
                )
                logger.info("Whisper model loaded successfully.")
            except ImportError:
                logger.error("faster-whisper is not installed or torch is misconfigured.")
                raise ExtractionException("faster-whisper dependency missing or fails to load.")
            except Exception as e:
                logger.error(f"Failed to initialize Whisper model: {str(e)}")
                raise ExtractionException(f"Failed to initialize Whisper model: {str(e)}")
        return cls._model_instance

    def transcribe_audio(self, file_path: Path) -> Tuple[str, List[Dict[str, Any]]]:
        """Transcribes the given audio file using faster-whisper.
        
        Returns:
            Tuple[str, List[Dict[str, Any]]]:
                - The full concatenated transcript string.
                - A list of segment metadata (start, end, text).
        """
        if not file_path.exists():
            raise ExtractionException(f"Audio file path does not exist: {file_path}")

        try:
            model = self._get_model()
            logger.info(f"Starting transcription of {file_path.name}...")
            
            # Run transcription (beam_size=5 is standard for good accuracy)
            segments, info = model.transcribe(str(file_path), beam_size=5)
            
            full_text_list = []
            detailed_segments = []
            
            for segment in segments:
                text = segment.text.strip()
                if not text:
                    continue
                full_text_list.append(text)
                detailed_segments.append({
                    "start": segment.start,
                    "end": segment.end,
                    "text": text
                })
                
            full_transcript = " ".join(full_text_list)
            logger.info(f"Transcription finished. Extracted {len(detailed_segments)} segments.")
            return full_transcript, detailed_segments

        except Exception as e:
            logger.error(f"Error during audio transcription: {str(e)}")
            raise ExtractionException(f"Transcription failed: {str(e)}")
