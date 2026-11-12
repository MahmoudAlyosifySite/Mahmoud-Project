"""BA: personalised voice cloning (Egyptian Arabic, English, French) on VoxCPM2."""
from .engine import BA, on_zerogpu
from .reference import ReferenceAudioError
from .text import normalize
from .voice import Voice
from .watermark import detect as detect_watermark

__version__ = "0.1.0"
__all__ = ["BA", "Voice", "ReferenceAudioError", "normalize", "detect_watermark", "on_zerogpu", "__version__"]
