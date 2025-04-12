from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech
from datasets import load_dataset, concatenate_datasets
from collections import defaultdict
from tqdm import tqdm
from TTS.api import TTS
import soundfile as sf
import torch
import unidecode
import os
import random


LANG_CODE='de'
FORMALITY='informal'

LANG_TO_MODEL = {'de': 'sophiayk20/speecht5_tts_voxpopuli_de', 'es': 'sophiayk20/speecht5_tts_voxpopuli_es',
                 'fr': 'sophiayk20/speecht5_tts_voxpopuli_fr', 'it': 'sophiayk20/speecht5_tts_voxpopuli_it'}

LANG_TO_FORMAL_DATASET = {'de': ['sophiayk20/de-mtformal', 'sophiayk20/topical-chat-formal-de', 'sophiayk20/telephony-formal-de'],
                          'es': ['sophiayk20/mt_formal', 'sophiayk20/topical-formal', 'sophiayk20/telephony-formal'],
                          'fr': ['sophiayk20/fr-mtformal'], 'it': ['sophiayk20/it-mtformal']}

LANG_TO_INFORMAL_DATASET = {'de': ['sophiayk20/de-mtinformal', 'sophiayk20/topical-chat-informal-de', 'sophiayk20/telephony-informal-de'],
                          'es': ['sophiayk20/mt_informal', 'sophiayk20/topical-informal', 'sophiayk20/telephony-informal'],
                          'fr': ['sophiayk20/fr-mtinformal'], 'it': ['sophiayk20/it-mtinformal']}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
tokenizer=processor.tokenizer
MODEL_NAME = LANG_TO_MODEL[LANG_CODE]
model = SpeechT5ForTextToSpeech.from_pretrained(MODEL_NAME).to(device)

DATASET_NAMES = []
if FORMALITY == 'formal':
  DATASET_NAMES = LANG_TO_FORMAL_DATASET[LANG_CODE]
else:
  DATASET_NAMES = LANG_TO_INFORMAL_DATASET[LANG_CODE]

loaded_datasets = []
# load all datasets
for dataset_name in DATASET_NAMES:
  loaded_datasets.append(load_dataset(dataset_name, split='train'))

combined_dataset = concatenate_datasets(loaded_datasets)
dataset = combined_dataset.shuffle(seed=42)
filtered_dataset = dataset.filter(lambda example: example.get(LANG_CODE) not in [None, ""])

# **Filter only text entries with leq 600 tokens -> SpeechT5 Model Restriction
def is_valid_length(example):
    text = example[LANG_CODE].strip()
    tokenized = processor(text=text, return_tensors="pt")
    return len(tokenized["input_ids"][0]) <= 600

filtered_dataset = filtered_dataset.filter(is_valid_length)
print(f"Filtered dataset length is: {len(filtered_dataset)}")

# Shuffle and select 10000 samples -> first 9000 for training, 1000 for validation
sampled_dataset = filtered_dataset.shuffle(seed=42).select(range(10000))

# Ensure no text entry is None or empty
for idx, item in enumerate(sampled_dataset):
    text = item.get(LANG_CODE, "").strip()
    assert text, f"Dataset issue at index {idx}: text is empty or None"

print("✅ All dataset entries contain valid text.")

# Print the length of the sampled dataset
print(f"Sampled dataset size: {len(sampled_dataset)}")

dataset = sampled_dataset
del sampled_dataset

def extract_all_chars(batch):
    # extract text in its plain form in target language form
    all_text = " ".join(batch[LANG_CODE])
    vocab = list(set(all_text))
    return {"vocab": [vocab], "all_text": [all_text]}

vocabs = dataset.map(
    extract_all_chars,
    batched=True,
    batch_size=-1,
    keep_in_memory=True,
    remove_columns=dataset.column_names,
)

dataset_vocab = set(vocabs["vocab"][0])
tokenizer_vocab = {k for k,_ in tokenizer.get_vocab().items()}

def cleanup_text(inputs):
  # just apply unidecode.unidecode to every character in dataset vocab
  for src in dataset_vocab:
    inputs[LANG_CODE] = inputs[LANG_CODE].replace(src, unidecode.unidecode(src) if src.isalpha() else ' ')

  return inputs

dataset = dataset.map(cleanup_text)

"""
    load XTTS model, load informal speech files whose voice features we are going to transfer over
"""
random.seed(42)
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to('cuda')
# FORMALITY='formal'
# LANG_CODE='fr'
WRITE_FOLDER=f'/content/drive/MyDrive/xtts-{FORMALITY}-{LANG_CODE}'
os.makedirs(WRITE_FOLDER,exist_ok=True)

# Create dictionary mapping each index to its corresponding speaker (train set)
indices_per_speaker=500
index_to_speaker = {index: speaker for speaker in range(18)
                    for index in range(speaker * indices_per_speaker, (speaker + 1) * indices_per_speaker)}
speakers = ['Jon', 'Lea', 'Gary', 'Jenna', 'Mike', 'Laura', 'Lauren', 'Eileen', 'Alisa', 'Karen', 'Barbara', 'Carol', 'Emily', 'Rose', 'Will', 'Patrick', 'Eric', 'Rick']
speaker_index_to_speaker = {i:speaker for i, speaker in enumerate(speakers)}
speaker_dict = {i:speaker for i, speaker in enumerate(speakers)}
parlertts_filenames = os.listdir(f'/content/drive/MyDrive/parlertts-{FORMALITY}')

# Dictionary to group files by speaker
speaker_waveform_groups = defaultdict(list)

# Process filename
for filename in parlertts_filenames:
    if filename.endswith(".wav"):
        parts = filename.split("-", 1)  # Split only at the first hyphen
        index, speaker_name = parts
        speaker_name = speaker_name.rsplit(".", 1)[0]  # Remove .wav extension
        speaker_waveform_groups[speaker_name].append(filename)

speaker_index_to_wav = {index: f'/content/drive/MyDrive/parlertts-{FORMALITY}/'+random.choice(speaker_waveform_groups[speaker_name]) for index, speaker_name in enumerate(speakers)}
print(speaker_index_to_wav)

"""
    Voice conversion, for train set
"""
dataset_texts=[]
for dataset_index in tqdm(range(9000)):
  text = dataset[dataset_index][LANG_CODE]
  dataset_texts.append(text)
  speaker_index = index_to_speaker[dataset_index]

  wav = tts.tts(
    text=text,
    speaker_wav=speaker_index_to_wav[speaker_index],
    language=LANG_CODE
  )
  sf.write(f'{WRITE_FOLDER}/{dataset_index}.wav', wav, 22050)

"""
    Test set
"""
# Validation
num_speakers = 18
start_index = 9000
end_index = 10000  # Not inclusive

total_indices = end_index - start_index  # 1000 indices

# Base indices per speaker
base_count = total_indices // num_speakers  # 55 per speaker
extra = total_indices % num_speakers  # 10 extra indices

index_to_speaker = {}
index = start_index

for speaker in range(num_speakers):
    count = base_count + (1 if speaker < extra else 0)  # First `extra` speakers get 1 more index
    for _ in range(count):
        index_to_speaker[index] = speaker
        index += 1

# Show distribution
print({s: list(index_to_speaker.values()).count(s) for s in set(index_to_speaker.values())})
print(f"Total indices assigned: {len(index_to_speaker)}")
print(min(index_to_speaker.keys()))
print(index_to_speaker[9000])
print(max(index_to_speaker.keys()))


# Also run for test set indices -> 9000 to 10000
for dataset_index in tqdm(range(9000,10000)):
    text = dataset[dataset_index][LANG_CODE]
    dataset_texts.append(text)
    speaker_index = index_to_speaker[dataset_index]

    wav = tts.tts(
      text=text,
      speaker_wav=speaker_index_to_wav[speaker_index],
      language=LANG_CODE
    )
    sf.write(f'{WRITE_FOLDER}/{dataset_index}.wav', wav, 22050) # 640
