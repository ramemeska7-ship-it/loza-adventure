"""Generate British-English learning audio and Arabic/English Loza clips.

Requirements:
    pip install edge-tts pydub
    ffmpeg must be installed and available on PATH when merging multi-part Loza clips.

Usage:
    python generate_kids_audio_corrected.py --sample
    python generate_kids_audio_corrected.py
    python generate_kids_audio_corrected.py --skip-existing
    python generate_kids_audio_corrected.py --only words
    python generate_kids_audio_corrected.py --force-r

This program produces letter *names*, not isolated phonics sounds.
Review generated letter names and vocabulary clips by ear before publication.
"""

import argparse
import asyncio
from io import BytesIO
import os
from pathlib import Path
import shutil

import edge_tts
from pydub import AudioSegment

# Voice availability is subject to the external TTS service.
AR_LOZA_VOICE = os.getenv("AR_LOZA_VOICE", "ar-SY-AmanyNeural")
EN_LOZA_VOICE = os.getenv("EN_LOZA_VOICE", "en-GB-LibbyNeural")
EN_TEACHER_VOICE = os.getenv("EN_TEACHER_VOICE", "en-GB-SoniaNeural")

AUDIO_DIR = Path(__file__).resolve().parent / "audio"
PAUSE_MS = 350
MAX_RETRIES = 3

# 19 Loza recordings. Some combine separate Arabic and British-English voices.
LOZA_SCRIPTS = [
    ("welcome_1", [("ar", "مَرْحَباً يَا صَدِيقِي! أَنَا لَوْزَة!")]),
    ("welcome_2", [("ar", "هَيَّا نَتَعَلَّمُ مَعاً!")]),
    ("welcome_3", [("ar", "هَلْ أَنْتَ مُسْتَعِدٌّ لِمُغَامَرَةٍ جَدِيدَة؟")]),
    ("learn_1", [("ar", "اِسْتَمِعْ جَيِّداً إِلَى الْحَرْفِ")]),
    ("learn_2", [("ar", "هَيَّا، رَدِّدْ مَعِي!")]),
    ("learn_3", [("ar", "رَائِعٌ! وَالْآنَ لِنَتَعَلَّمْ كَلِمَةً جَدِيدَةً")]),
    ("learn_4", [("ar", "اِسْتَمِعْ إِلَى الْكَلِمَةِ وَحَاوِلْ تَكْرَارَهَا")]),
    ("correct_1", [("ar", "مُمْتَازٌ يَا بَطَلُ!"), ("en", "Excellent!")]),
    ("correct_2", [("ar", "أَحْسَنْتَ!"), ("en", "Good job!")]),
    ("correct_3", [("ar", "رَائِعٌ جِدّاً!"), ("en", "Very good!")]),
    ("correct_4", [("ar", "إِجَابَةٌ صَحِيحَةٌ!"), ("en", "Nice!")]),
    ("correct_5", [("ar", "أَنَا فَخُورَةٌ بِكَ!"), ("en", "Amazing!")]),
    ("try_1", [("ar", "لَا بَأْسَ، حَاوِلْ مَرَّةً أُخْرَى!"), ("en", "Try again!")]),
    ("try_2", [("ar", "أَنْتَ تَسْتَطِيعُ!"), ("en", "You can do it!")]),
    ("try_3", [("ar", "اِقْتَرَبْتَ مِنَ الْإِجَابَةِ الصَّحِيحَةِ!")]),
    ("try_4", [("ar", "حَاوِلْ مَرَّةً أُخْرَى، أَنَا أُشَجِّعُكَ!")]),
    ("complete_1", [("ar", "يَا لَكَ مِنْ بَطَلٍ رَائِعٍ!")]),
    ("complete_2", [("ar", "لَقَدْ حَصَلْتَ عَلَى ثَلَاثِ نُجُومٍ!")]),
    ("complete_3", [("ar", "أَحْسَنْتَ! هَيَّا نَخْتَرْ حَرْفاً جَدِيداً!")]),
]

# Names of the letters, NOT phonics sounds. Single-letter / clean forms reduce
# double-reading or repeated vowel elongation in standalone TTS.
LETTER_NAMES = {
    "A": "A",
    "B": "Bee",
    "C": "See",
    "D": "Dee",
    "E": "E",
    "F": "Eff",
    "G": "Gee",
    "H": "Aitch",
    "I": "Eye",
    "J": "Jay",
    "K": "Kay",
    "L": "L",
    "M": "M",
    "N": "N",
    "O": "Oh",
    "P": "Pee",
    "Q": "Cue",
    "R": "Are",
    "S": "S",
    "T": "Tee",
    "U": "You",
    "V": "Vee",
    "W": "Double you",
    "X": "X",
    "Y": "Why",
    "Z": "Zed",
}

CURRICULUM_WORDS = {
    "A": ["Apple", "Cat", "Pizza"],
    "B": ["Ball", "Rabbit", "Crab"],
    "C": ["Cat", "Ice", "Music"],
    "D": ["Dog", "Candy", "Bed"],
    "E": ["Egg", "Pen", "Cake"],
    "F": ["Fish", "Coffee", "Leaf"],
    "G": ["Goat", "Tiger", "Frog"],
    "H": ["Hat", "Chair", "Fish"],
    "I": ["Ice", "Fish", "Kiwi"],
    "J": ["Jam", "Ninja", "DJ"],
    "K": ["Kite", "Monkey", "Book"],
    "L": ["Lion", "Balloon", "Ball"],
    "M": ["Moon", "Lemon", "Jam"],
    "N": ["Nest", "Banana", "Sun"],
    "O": ["Orange", "Robot", "Mango"],
    "P": ["Pig", "Apple", "Cup"],
    "Q": ["Queen", "Squid", "Iraq"],
    "R": ["Rabbit", "Carrot", "Car"],
    "S": ["Sun", "Basket", "Bus"],
    "T": ["Tiger", "Water", "Cat"],
    "U": ["Umbrella", "Sun", "Menu"],
    "V": ["Van", "Seven", "TV"],
    "W": ["Whale", "Flower", "Rainbow"],
    "X": ["Xylophone", "Taxi", "Fox"],
    "Y": ["Yellow", "Crayon", "Boy"],
    "Z": ["Zebra", "Puzzle", "Quiz"],
}

# Speak initialisms as letter names, not as ordinary words.
WORD_TTS_OVERRIDES = {"DJ": "D J", "TV": "T V"}


def has_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


def validate_data() -> None:
    expected = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    if set(LETTER_NAMES) != expected or set(CURRICULUM_WORDS) != expected:
        raise ValueError("There must be exactly one entry for each A-Z letter")
    if len(LOZA_SCRIPTS) != 19 or len({s[0] for s in LOZA_SCRIPTS}) != 19:
        raise ValueError("Expected 19 unique Loza recordings")
    for letter, words in CURRICULUM_WORDS.items():
        if len(words) != 3 or not all(letter.lower() in w.lower() for w in words):
            raise ValueError(f"Invalid examples for letter {letter}")


async def synthesize_to_bytes(text: str, voice: str, rate: str = "-6%") -> bytes:
    """Request TTS, retry temporary failures, never accept an empty response."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            communicate = edge_tts.Communicate(text=text, voice=voice, rate=rate)
            chunks = bytearray()
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    chunks.extend(chunk["data"])
            if not chunks:
                raise RuntimeError(f"Empty TTS audio: {voice} / {text!r}")
            return bytes(chunks)
        except Exception as exc:
            if attempt == MAX_RETRIES:
                raise RuntimeError(
                    f"TTS failed after {MAX_RETRIES} tries: {voice} / {text!r}"
                ) from exc
            await asyncio.sleep(attempt * 1.5)
    raise AssertionError("unreachable")


def save_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    try:
        temporary.write_bytes(data)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def save_joined_mp3(path: Path, parts: list[bytes], ffmpeg_available: bool) -> None:
    """Decode, insert real silence, and encode one proper MP3 file."""
    if not parts:
        raise ValueError("No MP3 audio parts")
    if len(parts) == 1:
        save_bytes(path, parts[0])
        return

    if not ffmpeg_available:
        print(
            f"⚠ ffmpeg not found on PATH; saving first part only for {path.name} "
            f"(install ffmpeg to merge bilingual clips)"
        )
        save_bytes(path, parts[0])
        return

    audio = AudioSegment.from_file(BytesIO(parts[0]), format="mp3")
    for part in parts[1:]:
        pause = AudioSegment.silent(duration=PAUSE_MS, frame_rate=audio.frame_rate)
        audio += pause + AudioSegment.from_file(BytesIO(part), format="mp3")

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".part.mp3")
    try:
        audio.export(str(temporary), format="mp3", bitrate="128k")
        if not temporary.exists() or temporary.stat().st_size == 0:
            raise RuntimeError(f"Failed to export audio file: {path}")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


async def generate_loza_files(
    scripts, skip_existing: bool, ffmpeg_available: bool
) -> tuple[int, int]:
    created = skipped = 0
    for file_id, segments in scripts:
        out_path = AUDIO_DIR / "loza" / f"{file_id}.mp3"
        if skip_existing and out_path.is_file() and out_path.stat().st_size:
            skipped += 1
            continue
        parts = []
        for lang, text in segments:
            if lang not in {"ar", "en"}:
                raise ValueError(f"Unknown language {lang} in {file_id}")
            voice = AR_LOZA_VOICE if lang == "ar" else EN_LOZA_VOICE
            parts.append(await synthesize_to_bytes(text, voice, rate="-6%"))
        save_joined_mp3(out_path, parts, ffmpeg_available=ffmpeg_available)
        created += 1
        print(f"✓ {out_path.relative_to(AUDIO_DIR)}")
    return created, skipped


async def generate_letter_files(
    names, skip_existing: bool, force_r: bool
) -> tuple[int, int]:
    created = skipped = 0
    for letter, spoken in names.items():
        out_path = AUDIO_DIR / "letters" / f"{letter}.mp3"

        # Protect custom manual R.mp3 unless --force-r is explicitly requested.
        if letter == "R" and not force_r and out_path.is_file() and out_path.stat().st_size:
            skipped += 1
            print(f"⏭ {out_path.relative_to(AUDIO_DIR)} (protected manual file)")
            continue

        if skip_existing and out_path.is_file() and out_path.stat().st_size:
            skipped += 1
            continue

        data = await synthesize_to_bytes(spoken, EN_TEACHER_VOICE, rate="-6%")
        save_bytes(out_path, data)
        created += 1
        print(f"✓ {out_path.relative_to(AUDIO_DIR)}")
    return created, skipped


async def generate_word_files(word_groups, skip_existing: bool) -> tuple[int, int]:
    created = skipped = 0
    cache: dict[str, bytes] = {}
    for letter, words in word_groups.items():
        for word in words:
            slug = "".join(ch if ch.isalnum() else "_" for ch in word.lower())
            out_path = AUDIO_DIR / "words" / f"{letter}_{slug}.mp3"
            if skip_existing and out_path.is_file() and out_path.stat().st_size:
                skipped += 1
                continue
            spoken = WORD_TTS_OVERRIDES.get(word, word) + "."
            if spoken not in cache:
                cache[spoken] = await synthesize_to_bytes(
                    spoken, EN_TEACHER_VOICE, rate="-10%"
                )
            save_bytes(out_path, cache[spoken])
            created += 1
            print(f"✓ {out_path.relative_to(AUDIO_DIR)}")
    return created, skipped


async def main() -> None:
    parser = argparse.ArgumentParser(description="Generate learning audio")
    parser.add_argument("--sample", action="store_true", help="Only create 7 test files")
    parser.add_argument("--skip-existing", action="store_true", help="Keep existing files")
    parser.add_argument(
        "--force-r",
        action="store_true",
        help="Overwrite manual audio/letters/R.mp3 using TTS",
    )
    parser.add_argument(
        "--only", choices=["all", "loza", "letters", "words"], default="all"
    )
    args = parser.parse_args()
    validate_data()
    ffmpeg_available = has_ffmpeg()
    if not ffmpeg_available and args.only in {"all", "loza"}:
        print("ℹ Note: ffmpeg was not detected on PATH.")

    scripts = LOZA_SCRIPTS
    names = LETTER_NAMES
    word_groups = CURRICULUM_WORDS
    if args.sample:
        scripts = [item for item in scripts if item[0] in {"welcome_1", "correct_1"}]
        names = {k: v for k, v in names.items() if k in {"A", "Z"}}
        word_groups = {"A": ["Apple"], "R": ["Car"], "Z": ["Zebra"]}

    total_created = total_skipped = 0
    for section, content in [
        ("loza", scripts),
        ("letters", names),
        ("words", word_groups),
    ]:
        if args.only not in {"all", section}:
            continue
        if section == "loza":
            created, skipped = await generate_loza_files(
                content, args.skip_existing, ffmpeg_available
            )
        elif section == "letters":
            created, skipped = await generate_letter_files(
                content, args.skip_existing, args.force_r
            )
        else:
            created, skipped = await generate_word_files(content, args.skip_existing)
        total_created += created
        total_skipped += skipped

    print(f"\nDone: {total_created} generated, {total_skipped} skipped")
    print("Accent: British English (en-GB).")
    print("Note: letters/ contains NAMES, not phonics sounds.")
    print("Listen to and approve every recording before teaching children.")


if __name__ == "__main__":
    asyncio.run(main())