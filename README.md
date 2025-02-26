
   # Translated Subs

   This project is a Python script that processes video files by extracting audio, transcribing speech using Whisper, translating subtitles via the Facebook SeamlessM4T model, and generating SRT files.

   ## Installation

   1. **Clone the repository (if using Git):**

      ```bash
      git clone https://github.com/markzoopark/translated-subs.git
      cd translated-subs
      ```

      *(If you prefer, you can simply download the project files from GitHub.)*

   2. **Install required Python dependencies:**

      ```bash
      pip install torch fairseq git+https://github.com/openai/whisper.git srt tqdm transformers
      ```

   3. **Ensure system dependencies are installed:**
      
      - **ffmpeg:** Download from [ffmpeg.org](https://ffmpeg.org/)
      - **ffsubsync:** Follow the instructions on its [GitHub page](https://github.com/FFsubsync/ffsubsync)

   ## Usage

   1. Run the script:

      ```bash
      python main.py
      ```

   2. Follow the on-screen prompts:
      - Enter the path to the folder containing your video files.
      - Select the video files to process.
      - Choose the source and target languages.
      - The script will generate subtitle files (`.srt`) in the same folder as your videos.

   ## Notes

   - Log in to Hugging Face CLI if the translation model requires authentication:

     ```bash
     huggingface-cli login
     ```

   - If you encounter issues with the Whisper package, uninstall any conflicting version and install the official one:

     ```bash
     pip uninstall whisper
     pip install git+https://github.com/openai/whisper.git
     ```

   Enjoy!
