import os
import yt_dlp
from pydub import AudioSegment


DOWNLOAD_DIR = "downloads"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)


# --------------------------------------------------
# FFmpeg
# --------------------------------------------------

FFMPEG_DIR = (
    r"C:\Users\ssidd\AppData\Local\Microsoft\WinGet\Packages"
    r"\Gyan.FFmpeg.Shared_Microsoft.Winget.Source_8wekyb3d8bbwe"
    r"\ffmpeg-9.0.1-full_build-shared\bin"
)


# --------------------------------------------------
# Deno
# --------------------------------------------------

DENO_DIR = (
    r"C:\Users\ssidd\AppData\Local\Microsoft\WinGet\Packages"
    r"\DenoLand.Deno_Microsoft.Winget.Source_8wekyb3d8bbwe"
)


# Add FFmpeg + Deno to PATH for this Python process
os.environ["PATH"] = (
    FFMPEG_DIR
    + os.pathsep
    + DENO_DIR
    + os.pathsep
    + os.environ.get("PATH", "")
)


# Tell pydub exactly where FFmpeg is
AudioSegment.converter = os.path.join(
    FFMPEG_DIR,
    "ffmpeg.exe"
)

AudioSegment.ffprobe = os.path.join(
    FFMPEG_DIR,
    "ffprobe.exe"
)


def download_youtube_audio(url: str) -> str:
    """Download YouTube audio and convert it to WAV."""

    print("Downloading YouTube audio...")

    ydl_opts = {
    "format": "251",
    "outtmpl": os.path.join(
        DOWNLOAD_DIR,
        "%(title)s.%(ext)s"
    ),
    "postprocessors": [
        {
            "key": "FFmpegExtractAudio",
            "preferredcodec": "wav",
        }
    ],
    "quiet": False,
}



    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)

        # Original downloaded filename
        filename = ydl.prepare_filename(info)

    # FFmpegExtractAudio changes extension to .wav
    wav_path = os.path.splitext(filename)[0] + ".wav"

    if not os.path.exists(wav_path):
        raise FileNotFoundError(
            f"Expected WAV file was not created: {wav_path}"
        )

    return wav_path


def convert_to_wav(input_path: str) -> str:
    """Convert any audio/video file to WAV format."""

    output_path = os.path.splitext(input_path)[0] + "_converted.wav"

    # Make sure pydub also knows where FFmpeg is
    AudioSegment.converter = os.path.join(
        FFMPEG_DIR,
        "ffmpeg.exe"
    )

    audio = AudioSegment.from_file(input_path)

    # Whisper works nicely with mono 16 kHz audio
    audio = audio.set_channels(1).set_frame_rate(16000)

    audio.export(output_path, format="wav")

    return output_path


def chunk_audio(wav_path: str, chunk_minutes: int = 10) -> list:
    """Split WAV audio into chunks."""

    audio = AudioSegment.from_wav(wav_path)

    chunk_ms = chunk_minutes * 60 * 1000

    chunks = []

    for i, start in enumerate(range(0, len(audio), chunk_ms)):
        chunk = audio[start:start + chunk_ms]

        chunk_path = f"{wav_path}_chunk_{i}.wav"

        chunk.export(chunk_path, format="wav")

        chunks.append(chunk_path)

    return chunks


def process_input(source: str) -> list:
    """Download/convert input and split it into audio chunks."""

    if source.startswith(("http://", "https://")):

        print("Detected YouTube URL. Downloading audio...")

        wav_path = download_youtube_audio(source)

    else:

        print("Detected local file. Converting to WAV...")

        wav_path = convert_to_wav(source)

    print(f"Audio file: {wav_path}")

    print("Chunking audio...")

    chunks = chunk_audio(wav_path)

    print(f"Audio ready — {len(chunks)} chunk(s) created.")

    return chunks


if __name__ == "__main__":

    url = "https://www.youtube.com/watch?v=Ty8gcCKuwNI"

    chunks = process_input(url)

    print("\nCreated chunks:")

    for chunk in chunks:
        print(chunk)