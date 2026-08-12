import os
import requests
import whisper

from dotenv import load_dotenv
from pydub import AudioSegment


load_dotenv()

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

SARVAM_PIECE_SECONDS = 25

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")

SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")

SARVAM_STT_TRANSLATE_URL = (
    "https://api.sarvam.ai/speech-to-text-translate"
)

SARVAM_MODEL = os.getenv(
    "SARVAM_STT_MODEL",
    "saaras:v2.5"
)


# Whisper model is loaded lazily.
_model = None

# ─────────────────────────────────────────────
# Whisper
# ─────────────────────────────────────────────

def load_model():
    """Load Whisper model once and reuse it."""

    global _model

    if _model is None:
        print(f"Loading Whisper model: {WHISPER_MODEL} ...")

        _model = whisper.load_model(WHISPER_MODEL)

        print("Whisper model loaded.")

    return _model


def transcribe_chunk_whisper(chunk_path: str) -> str:
    """Transcribe one audio chunk using local Whisper."""

    model = load_model()

    result = model.transcribe(
        chunk_path,
        task="transcribe"
    )

    return result["text"].strip()


# ─────────────────────────────────────────────
# Sarvam
# ─────────────────────────────────────────────

def _send_to_sarvam(piece_path: str) -> str:
    """Send one short WAV file to Sarvam."""

    if not SARVAM_API_KEY:
        raise RuntimeError(
            "SARVAM_API_KEY is not set in .env"
        )

    headers = {
        "api-subscription-key": SARVAM_API_KEY
    }

    with open(piece_path, "rb") as f:

        files = {
            "file": (
                os.path.basename(piece_path),
                f,
                "audio/wav"
            )
        }

        data = {
            "model": SARVAM_MODEL,
            "with_diarization": "false"
        }

        response = requests.post(
            SARVAM_STT_TRANSLATE_URL,
            headers=headers,
            files=files,
            data=data,
            timeout=120,
        )

    if not response.ok:
        print(
            f"\nSarvam returned "
            f"{response.status_code}"
        )

        print(
            f"Response body: "
            f"{response.text}\n"
        )

        response.raise_for_status()

    result = response.json()

    return result.get("transcript", "").strip()


def transcribe_chunk_sarvam(chunk_path: str) -> str:
    """
    Split a long audio chunk into short pieces
    and send each piece to Sarvam.
    """

    if not SARVAM_API_KEY:
        raise RuntimeError(
            "SARVAM_API_KEY is not set in .env"
        )

    audio = AudioSegment.from_wav(chunk_path)

    piece_ms = SARVAM_PIECE_SECONDS * 1000

    full_text = []

    total_pieces = (
        len(audio) + piece_ms - 1
    ) // piece_ms

    for i, start in enumerate(
        range(0, len(audio), piece_ms)
    ):

        piece = audio[
            start:start + piece_ms
        ]

        piece_path = (
            f"{chunk_path}_sv_{i}.wav"
        )

        piece.export(
            piece_path,
            format="wav"
        )

        try:

            print(
                f"  → Sarvam piece "
                f"{i + 1}/{total_pieces}..."
            )

            text = _send_to_sarvam(
                piece_path
            )

            if text:
                full_text.append(text)

        finally:

            if os.path.exists(piece_path):
                os.remove(piece_path)

    return " ".join(full_text).strip()


# ─────────────────────────────────────────────
# Routing
# ─────────────────────────────────────────────

def transcribe_chunk(
    chunk_path: str,
    language: str = "english"
) -> str:

    """
    Route audio to the appropriate engine.

    english  → local Whisper
    hinglish → Sarvam translation
    """

    language = language.lower()

    if language == "hinglish":
        return transcribe_chunk_sarvam(
            chunk_path
        )

    return transcribe_chunk_whisper(
        chunk_path
    )


# ─────────────────────────────────────────────
# Transcribe all chunks
# ─────────────────────────────────────────────

def transcribe_all(
    chunks: list,
    language: str = "english"
) -> str:

    """Transcribe all audio chunks."""

    full_transcript = []

    engine = (
        "Sarvam AI"
        if language.lower() == "hinglish"
        else "Whisper"
    )

    print(
        f"Using {engine} for transcription."
    )

    for i, chunk in enumerate(chunks):

        print(
            f"Transcribing chunk "
            f"{i + 1}/{len(chunks)}..."
        )

        text = transcribe_chunk(
            chunk,
            language=language
        )

        if text:
            full_transcript.append(text)

    print("Transcription complete.")

    return "\n\n".join(full_transcript)