from utils.audio_processor import process_input
from core.transcriber import transcribe_all

URL = "https://www.youtube.com/watch?v=Ty8gcCKuwNI"

chunks = process_input(URL)

transcript = transcribe_all(
    chunks,
    language="english"
)

with open("transcript.txt", "w", encoding="utf-8") as f:
    f.write(transcript)

print("\n" + "=" * 70)
print("TRANSCRIPTION COMPLETE")
print("=" * 70)
print(transcript[:2000])
print("\nFull transcript saved to transcript.txt")