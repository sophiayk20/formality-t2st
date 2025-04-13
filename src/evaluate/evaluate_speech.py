import librosa
import numpy as np
from tqdm import tqdm
import pandas as pd
from phonemizer import phonemize
from phonemizer.backend import EspeakBackend
"""
0.02 -> 2% of maximum possible energy in audio file (energy based detection)
      -> any section of audio with RMS energy below 2% of the maximum value is considered silent
0.5 -> any silent interval shorter than 0.5 seconds would be ignored, and anything longer would be recorded as a pause.

`pip install -q phonemizer`
`sudo apt-get install festival espeak-ng mbrola` or the designated phoneme background you plan on using for phonemizer
"""

def detect_pauses(audio_path, silence_threshold=0.02, min_pause_duration=0.5):
    """
    Detects pauses in the audio based on silence threshold and minimum pause duration.
    
    Parameters:
    - audio_path (str): Path to the audio file.
    - silence_threshold (float): Threshold below which audio is considered silent (lower value means more sensitivity).
    - min_pause_duration (float): Minimum duration in seconds for a pause to be considered.
    
    Returns:
    - List of pause intervals (start_time, end_time) in seconds.
    """
    # Load audio
    y, sr = librosa.load(audio_path, sr=None)

    # Compute short-term energy (root mean square energy)
    energy = librosa.feature.rms(y=y)[0]  # Root-mean-square energy
    time_intervals = librosa.times_like(energy, sr=sr)  # Time corresponding to energy values

    # Identify silent regions where energy is below the threshold
    silent = energy < silence_threshold
    pause_intervals = []

    # Group silent regions as pauses
    pause_start = None
    for i, is_silent in enumerate(silent):
        if is_silent:
            if pause_start is None:
                pause_start = time_intervals[i]
        else:
            if pause_start is not None:
                pause_end = time_intervals[i]
                if pause_end - pause_start >= min_pause_duration:
                    pause_intervals.append((pause_start, pause_end))
                pause_start = None

    # In case the last segment is a pause
    if pause_start is not None:
        pause_end = time_intervals[-1]
        if pause_end - pause_start >= min_pause_duration:
            pause_intervals.append((pause_start, pause_end))

    return pause_intervals

def extract_phonemes(transcription, lang='es'):
    backend = EspeakBackend(language=lang, preserve_punctuation=True, with_stress=False)
    phonemes = backend.phonemize([transcription])[0]
    phoneme_list = phonemes.split()  # Split the phonemes into a list
    return len(phoneme_list)

for FORMALITY in ['formal', 'informal']:
  for MODE in ["BI", "II"]:
    for LANG_CODE in ['es', 'fr', 'it']:
      TRANSCRIPTIONS_PATH = f"/content/drive/MyDrive/speech-transcriptions/{MODE}.{LANG_CODE}.{FORMALITY}.transcriptions.txt"
      FILES_PATH = f"/content/drive/MyDrive/speech-transcriptions/{MODE}.{LANG_CODE}.{FORMALITY}.files.txt"

      with open(TRANSCRIPTIONS_PATH, 'r') as f:
          transcriptions = [line.strip() for line in f.readlines()]

      with open(FILES_PATH, 'r') as f:
          file_paths = [line.strip().split()[0] for line in f.readlines()]  # just the path

      results = []
      for audio_path, transcription in tqdm(zip(file_paths, transcriptions), total=len(file_paths), desc=f"Analyzing {MODE}.{LANG_CODE}.{FORMALITY}"):
          # get audio duration with librosa
          duration = librosa.get_duration(path=audio_path)
          word_count = len(transcription.split())
          wps = word_count / duration if duration > 0 else 0

          # detect pauses
          pause_intervals = detect_pauses(audio_path)
          
          # calculate mean pause duration
          if pause_intervals:
              pause_durations = [end - start for start, end in pause_intervals]
              mean_pause_duration = np.mean(pause_durations)
          else:
              mean_pause_duration = 0

          # extract phonemes from transcription
          phoneme_lang_code = {'es': 'es', 'fr': 'fr-fr', 'de': 'de', 'it': 'it'} # supported by espeak-ng backend3 (check with: !espeak-ng --voices)
          phoneme_count = extract_phonemes(transcription, lang=phoneme_lang_code[LANG_CODE])
          
          # calculate phonemes per seconds (pps)
          pps = phoneme_count / duration if duration > 0 else 0

          results.append({
              "audio": audio_path,
              "duration_sec": duration,
              "words": word_count,
              "wps": wps,
              "phonemes": phoneme_count,
              "pps": pps,
              "pauses": pause_intervals,
              "mean_pause_duration": mean_pause_duration,
              "transcription": transcription
          })

      # convert to dataframe for average analysis
      df = pd.DataFrame(results)

      # calculate averages
      avg_duration = df["duration_sec"].mean()
      avg_words = df["words"].mean()
      avg_wps = df["wps"].mean()
      avg_phonemes = df["phonemes"].mean()
      avg_pps = df["pps"].mean()
      avg_mean_pause_duration = df["mean_pause_duration"].mean()

      print("\n--- Averages ---")
      print(f"LANG_CODE: {LANG_CODE} FORMALITY: {FORMALITY} SYSTEM: {MODE}")
      print(f"Average Duration: {avg_duration:.2f} seconds")
      print(f"Average Word Count: {avg_words:.2f} words")
      print(f"Average WPS: {avg_wps:.2f}")
      print(f"Average Phoneme Count: {avg_phonemes:.2f} phonemes")
      print(f"Average PPS: {avg_pps:.2f}")
      print(f"Average Mean Pause Duration: {avg_mean_pause_duration:.2f} seconds\n")
