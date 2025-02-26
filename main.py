#!/usr/bin/env python3
import os
import sys
import subprocess
import importlib
from datetime import timedelta

# ---------------------- Dependency Check ----------------------
# List of required Python modules
required_modules = ["torch", "fairseq", "whisper", "srt", "tqdm"]

missing_modules = []
for module in required_modules:
    try:
        importlib.import_module(module)
    except ImportError:
        missing_modules.append(module)

if missing_modules:
    print("Missing Python dependencies: " + ", ".join(missing_modules))
    print("Please install them using: pip install " + " ".join(missing_modules))
    sys.exit(1)

# Now import modules after dependency check
from tqdm import tqdm
import whisper
import srt
from transformers import pipeline

# ---------------------- Check for correct Whisper installation ----------------------
if not hasattr(whisper, 'load_model'):
    print("Error: whisper.load_model not found.")
    print("It appears that you have installed the incorrect 'whisper' package.")
    print("Please uninstall it and install the official OpenAI Whisper library using:")
    print("pip uninstall whisper")
    print("pip install git+https://github.com/openai/whisper.git")
    sys.exit(1)

# ---------------------- System Command Check ----------------------
def check_command(command):
    try:
        if command == "ffsubsync":
            subprocess.run([command, "--help"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        else:
            subprocess.run([command, "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return True
    except Exception:
        return False

# Check for system dependencies
if not check_command("ffmpeg"):
    print("Missing system dependency: ffmpeg")
    sys.exit(1)
if not check_command("ffsubsync"):
    print("Missing system dependency: ffsubsync")
    sys.exit(1)

# ---------------------- Translation Setup ----------------------
def translate_text(text, src_lang, tgt_lang, translator):
    # translator: a pre-initialized transformers pipeline for translation.
    result = translator(text, src_lang=src_lang, tgt_lang=tgt_lang)
    return result[0]['translation_text']

# ---------------------- Audio Extraction ----------------------
def extract_audio(video_path, audio_path):
    command = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "44100",
        "-ac", "2",
        audio_path
    ]
    subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

# ---------------------- Main Script ----------------------
def main():
    # Step 1: Ask user for the folder containing video files
    print("Enter the path to the folder with video files:")
    # Remove surrounding quotes if present
    folder = input().strip().strip('"').strip("'")
    if not os.path.isdir(folder):
        print("The specified path is not a valid directory.")
        sys.exit(1)
    
    # List all .mp4 files in the folder
    video_files = [f for f in os.listdir(folder) if f.lower().endswith(".mp4")]
    if not video_files:
        print("No .mp4 files found in the specified directory.")
        sys.exit(1)
    
    # Step 2: Display video list and let user choose files
    print("Available video files:")
    for idx, file in enumerate(video_files, start=1):
        print(f"{idx}. {file}")
    
    while True:
        print("Enter the numbers of the files to process, separated by spaces:")
        selected = input().strip().split()
        try:
            selected_indices = [int(x) for x in selected]
        except ValueError:
            print("Invalid input. Please enter numbers separated by spaces.")
            continue
        if any(idx < 1 or idx > len(video_files) for idx in selected_indices):
            print("One or more numbers are out of range. Please try again.")
            continue
        break
    
    selected_files = [video_files[i - 1] for i in selected_indices]
    
    # Step 3: Choose source and target languages
    languages = {
        "it": "Italian",
        "en": "English",
        "fr": "French",
        "es": "Spanish",
        "de": "German"
    }
    print("Available languages:")
    for code, name in languages.items():
        print(f"{code} - {name}")
    
    while True:
        print("Enter the source language code:")
        src_lang = input().strip().lower()
        if src_lang not in languages:
            print("Invalid source language code. Please try again.")
            continue
        print("Enter the target language code:")
        tgt_lang = input().strip().lower()
        if tgt_lang not in languages:
            print("Invalid target language code. Please try again.")
            continue
        break

    # Step 3.1: Initialize the translator pipeline for SeamlessM4T using the new model.
    try:
        translator = pipeline("translation", model="facebook/seamless-m4t-v2-large")
    except Exception as e:
        print("Error initializing the translation model:", e)
        print("Please ensure that you have access to the 'facebook/seamless-m4t-v2-large' model.")
        print("If it is a private repository, login with 'huggingface-cli login'.")
        sys.exit(1)

    processed_videos = []
    generated_subtitles = []

    # Step 4: Process each selected video
    for file in selected_files:
        video_path = os.path.join(folder, file)
        base_name = os.path.splitext(file)[0]
        print(f"\nProcessing video: {file}")
        
        # Extract audio to a temporary WAV file
        audio_path = os.path.join(folder, base_name + ".wav")
        print("Extracting audio from video...")
        extract_audio(video_path, audio_path)
        
        # Transcribe audio using Whisper
        print("Transcribing audio using Whisper...")
        model_whisper = whisper.load_model("medium")
        result = model_whisper.transcribe(audio_path, verbose=False)
        transcript_segments = result["segments"]
        
        # Translate each segment using SeamlessM4T with a progress bar
        print("Translating transcript segments...")
        for segment in tqdm(transcript_segments, desc="Translating segments"):
            original_text = segment["text"].strip()
            translated_text = translate_text(original_text, src_lang, tgt_lang, translator)
            segment["text"] = translated_text
        
        # Create subtitle content in SRT format
        subtitles = []
        for idx, seg in enumerate(transcript_segments, start=1):
            sub = srt.Subtitle(
                index=idx,
                start=timedelta(seconds=seg["start"]),
                end=timedelta(seconds=seg["end"]),
                content=seg["text"]
            )
            subtitles.append(sub)
        srt_content = srt.compose(subtitles)
        
        # Determine output filenames based on naming conventions
        srt_filename = f"{base_name}.{tgt_lang}.srt"
        srt_path = os.path.join(folder, srt_filename)
        # Mapping for three-letter ISO codes for better compatibility
        iso_map = {"en": "eng", "it": "ita", "fr": "fra", "es": "spa", "de": "deu"}
        alt_lang_code = iso_map.get(tgt_lang, tgt_lang)
        alt_srt_filename = f"{base_name}.{alt_lang_code}.srt"
        alt_srt_path = os.path.join(folder, alt_srt_filename)
        
        # Save both subtitle files
        with open(srt_path, "w", encoding="utf-8") as f:
            f.write(srt_content)
        with open(alt_srt_path, "w", encoding="utf-8") as f:
            f.write(srt_content)
        
        # Remove temporary audio file
        os.remove(audio_path)
        
        processed_videos.append(file)
        generated_subtitles.append((srt_path, alt_srt_path))
    
    # Step 5: Final message
    print("\nProcessing completed.")
    print("Processed videos:")
    for video in processed_videos:
        print(video)
    
    print("\nGenerated subtitle files:")
    for paths in generated_subtitles:
        print(" and ".join(paths))
    
    print("\nSubtitles can be enabled in any media player.")

if __name__ == "__main__":
    main()
