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