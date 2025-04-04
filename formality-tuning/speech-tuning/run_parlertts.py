"""
    Create English speech that we will extract formal / informal prosodic styles (voice features) from./
    ParlerTTS must be installed with `pip install git+https://github.com/huggingface/parler-tts.git`
"""
from tqdm import tqdm # 1800 -> 7 hrs # 180 -> 42 mins
import soundfile as sf
from parler_tts import ParlerTTSForConditionalGeneration
import torch
from transformers import AutoTokenizer
from datasets import load_dataset, concatenate_datasets
import os

FORMALITY='informal'
BASE_FOLDER='/content/drive/MyDrive/parlertts-informal'
os.makedirs(BASE_FOLDER, exist_ok=True)
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

descriptions = {
    "formal": " delivers a formal and professional speech. The recording is of very high quality, with the speaker's voice sounding clear and very close up.",
    "informal": " delivers a casual and conversational speech. The recording is of very high quality, with the speaker's voice sounding clear and very close up.",
}

# Create dictionary mapping each index to its corresponding speaker
indices_per_speaker=10
index_to_speaker = {index: speaker for speaker in range(18)
                    for index in range(speaker * indices_per_speaker, (speaker + 1) * indices_per_speaker)}

speakers = ['Jon', 'Lea', 'Gary', 'Jenna', 'Mike', 'Laura', 'Lauren', 'Eileen', 'Alisa', 'Karen', 'Barbara', 'Carol', 'Emily', 'Rose', 'Will', 'Patrick', 'Eric', 'Rick']
speaker_dict = {i:speaker for i, speaker in enumerate(speakers)}

loaded_datasets = []
# load all datasets
for dataset_name in [f'sophiayk20/fr-mt{FORMALITY}']:
  loaded_datasets.append(load_dataset(dataset_name, split='train'))

combined_dataset = concatenate_datasets(loaded_datasets)
dataset = combined_dataset.shuffle(seed=42).select(range(180)) # 10 for each speaker

parler_model = ParlerTTSForConditionalGeneration.from_pretrained("parler-tts/parler-tts-mini-v1").to(device)
parler_tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler-tts-mini-v1")

## now that dataset is generated, generate English embeddings read aloud
for dataset_index in tqdm(range(180)):
  prompt = dataset[dataset_index]['en']
  speaker_index = index_to_speaker[dataset_index]
  speaker_name = speaker_dict[speaker_index]
  description = f"{speaker_name}" + descriptions[FORMALITY]

  input_ids = parler_tokenizer(description, return_tensors="pt").input_ids.to(device)
  prompt_input_ids = parler_tokenizer(prompt, return_tensors="pt").input_ids.to(device)

  generation = parler_model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)

  # audio arr of reading in english text, for purposes of extracting speaker embedding
  audio_arr = generation.cpu().numpy().squeeze()
  sf.write(f"{BASE_FOLDER}/{dataset_index}-{speaker_name}.wav", audio_arr, parler_model.config.sampling_rate)