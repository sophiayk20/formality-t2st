"""
    Use whisper to generate transcripts of speech files
    `pip install -q datasets`
"""

from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from tqdm import tqdm
import os
import torch

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
model_id = "openai/whisper-large-v3"
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True,
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)
LANG_CODES = ['de', 'es', 'fr', 'it']
MODES=['BF']
FORMALITIES=['formal']

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
    return_timestamps=True,
)

for LANG_CODE in LANG_CODES:
  for MODE in MODES:
    for FORMALITY in FORMALITIES:
      FOLDER_PATH=f"/content/drive/MyDrive/speech-outputs/{MODE}/{LANG_CODE}/{FORMALITY}"
      audio_files = os.listdir(FOLDER_PATH)
      ANALYSIS_PATH=f"/content/drive/MyDrive/speech-transcriptions"
      os.makedirs(ANALYSIS_PATH, exist_ok=True)
      transcription_path = f"{ANALYSIS_PATH}/{MODE}.{LANG_CODE}.{FORMALITY}.transcriptions.txt"
      files_path = f"{ANALYSIS_PATH}/{MODE}.{LANG_CODE}.{FORMALITY}.files.txt"
      audio_files.sort()

      with open(transcription_path, 'w') as f1, open(files_path, 'w') as f2:
          #for i in tqdm(range(0, len(audio_files), BATCH_SIZE)):
          for i in tqdm(range(0, len(audio_files))):
              #batch = audio_files[i:i + BATCH_SIZE]
              #batch = [f"{FOLDER_PATH}/{item}" for item in batch]
              audio_file = f"{FOLDER_PATH}/{audio_files[i]}"

              # Run batch inference
              #results = pipe(batch, batch_size=BATCH_SIZE)
              results = pipe(audio_file)

              f2.write(audio_file + '\n')
              f1.write(results['text'] + '\n')

              torch.cuda.empty_cache() # Free GPU memory

# for %M-A accuracy evaluation
def create_files(LANG_CODE, FORMALITY, MODE):
  opposite_formality = {'formal': 'informal', 'informal': 'formal'}
  FILENAME_PATH = f"/content/drive/MyDrive/speech-transcriptions/{MODE}.{LANG_CODE}.{FORMALITY}.files.txt"
  TRANSCRIPTION_PATH = f"/content/drive/MyDrive/speech-transcriptions/{MODE}.{LANG_CODE}.{FORMALITY}.transcriptions.txt"
  ANNOTATED_WRITE_PATH=f"/content/drive/MyDrive/speech-transcriptions/{MODE}.{LANG_CODE}.{FORMALITY}.annotated.references.txt"
  DETOKENIZED_WRITE_PATH=f"/content/drive/MyDrive/speech-transcriptions/{MODE}.{LANG_CODE}.{FORMALITY}.detokenized.references.txt"
  TRANSCRIPTS_WRITE_PATH=f"/content/drive/MyDrive/speech-transcriptions/{MODE}.{LANG_CODE}.{FORMALITY}.common.transcriptions.txt"
  OPPOSITE_FILENAME_PATH=f"/content/drive/MyDrive/speech-transcriptions/{MODE}.{LANG_CODE}.{opposite_formality[FORMALITY]}.files.txt"
  # ACL-IWSLT original annotated reference translation path
  ANNOTATED_REFERENCE_PATH=f"/content/drive/MyDrive/annotated-references/formality-control.test.en-{LANG_CODE}.{FORMALITY}.annotated.{LANG_CODE}"
  DETOKENIZED_REFERENCE_PATH=f"/content/drive/MyDrive/detokenized-references/formality-control.test.en-{LANG_CODE}.{FORMALITY}.{LANG_CODE}"
  OPPOSITE_ANNOTATED_REFERENCE_PATH=f"/content/drive/MyDrive/annotated-references/formality-control.test.en-{LANG_CODE}.{opposite_formality[FORMALITY]}.annotated.{LANG_CODE}"
  OPPOSITE_DETOKENIZED_REFERENCE_PATH=f"/content/drive/MyDrive/detokenized-references/formality-control.test.en-{LANG_CODE}.{opposite_formality[FORMALITY]}.{LANG_CODE}"

  filenames = []
  transcripts = []

  print(f"LANG_CODE: {LANG_CODE} formality: {FORMALITY}")

  with open(FILENAME_PATH, "r") as f:
    filenames = f.readlines()
    filenames = [line.strip() for line in filenames]
  print(f"len(filenames): {len(filenames)}")

  with open(OPPOSITE_FILENAME_PATH, "r") as f:
    opposite_filenames = f.readlines()
    opposite_filenames = [line.strip() for line in opposite_filenames]
  print(f"len(opposite_filenames): {len(opposite_filenames)}")

  opposite_indices= []
  for opposite_filename in opposite_filenames:
    filename = os.path.basename(opposite_filename)
    parts = filename.replace(".wav", "").split("-")
    i = int(parts[0])
    speaker_index = int(parts[1])
    opposite_indices.append(i)

  with open(TRANSCRIPTION_PATH, "r") as f:
    transcripts = f.readlines()
    transcripts = [line.strip() for line in transcripts]
  print(f"len(transcripts): {len(transcripts)}")

  with open(ANNOTATED_REFERENCE_PATH, "r") as f:
    annotated_references = f.readlines()
    annotated_references = [line.strip() for line in annotated_references]
  print(f"len(annotated_references): {len(annotated_references)}")

  with open(DETOKENIZED_REFERENCE_PATH, "r") as f:
    detokenized_references = f.readlines()
    detokenized_references = [line.strip() for line in detokenized_references]
  print(f"len(detokenized_references): {len(detokenized_references)}")

  with open(ANNOTATED_WRITE_PATH, 'w') as fw, open(DETOKENIZED_WRITE_PATH, 'w') as fd, open(TRANSCRIPTS_WRITE_PATH, 'w') as ft:
    for k, path in enumerate(filenames):
      # Extract filename
      filename = os.path.basename(path)  # "0-1366.wav"

      # Remove the extension and split by '-'
      parts = filename.replace(".wav", "").split("-")

      # Convert to integers
      i = int(parts[0])
      if i in opposite_indices:
        speaker_index = int(parts[1])

        fw.write(annotated_references[i] + '\n')
        fd.write(detokenized_references[i] + '\n')
        ft.write(transcripts[k] + '\n') # kth file in filenames file
      else:
        continue

LANG_CODES=['es', 'fr', 'de', 'it']
FORMALITIES=['formal', 'informal']
MODES=['BB']

for MODE in MODES:
  for FORMALITY in FORMALITIES:
    for LANG_CODE in LANG_CODES:
      create_files(LANG_CODE, FORMALITY, MODE)