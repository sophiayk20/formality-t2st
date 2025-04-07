"""
    Generates speech in designated SpeechT5 model
"""
from tqdm import tqdm
from transformers import SpeechT5ForTextToSpeech, SpeechT5HifiGan, SpeechT5Processor
from datasets import load_dataset
import os
import random
import torch
import soundfile as sf

# Choose test speakers
random.seed(42)
embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split='validation')

male_bdl_dataset = [i for i, item in enumerate(embeddings_dataset) if "bdl" in item['filename']] # male
male_rms_dataset = [i for i, item in enumerate(embeddings_dataset) if "rms" in item['filename']] # male
female_slt_dataset = [i for i, item in enumerate(embeddings_dataset) if "slt" in item['filename']] # female
female_clb_dataset = [i for i, item in enumerate(embeddings_dataset) if "clb" in item['filename']] # female

chosen_speakers=[]
for dat in [male_bdl_dataset, male_rms_dataset, female_slt_dataset, female_clb_dataset]:
  chosen_speakers.append(random.choice(dat))
print(*chosen_speakers)

speaker_indices={}
start = 0
for speaker in chosen_speakers:
  end = start + 149 # range size is 150 (150*4 == 600)
  speaker_indices[speaker] = [i for i in range(start, end+1)]
  start = end + 1
for k, v in speaker_indices.items():
  print(f"k: {k} min: {min(v)} max: {max(v)}")
  assert(len(v) == 150)

# Run example: test_speecht5("sophiayk20/speecht5_tts_formal_es", "es", "B", "F", "formal")
def test_speecht5(model_name, target_lang, text_config, speech_config, formality):
  device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model = SpeechT5ForTextToSpeech.from_pretrained(model_name)
  model = model.to(device)
  processor= SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
  vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(device) # apparently vocoder also needs to be on gpu

  with open(f"/content/drive/MyDrive/translation-outputs/{text_config}-{target_lang}-{formality}-outputs.txt") as f:
    text_translation_outputs = [x.strip() for x in f.readlines()]

  directory = f"/content/drive/MyDrive/speech-outputs/{text_config}{speech_config}/{target_lang}/{formality}"
  os.makedirs(directory, exist_ok=True)

  assert(len(text_translation_outputs) == 600)

  for speaker_index, indices in speaker_indices.items():
    speaker_embeddings = torch.tensor(embeddings_dataset[speaker_index]["xvector"]).unsqueeze(0)
    speaker_embeddings = speaker_embeddings.to(device)
    print(f"Processing speaker: {speaker_index}...")
    for k, index in enumerate(tqdm(indices)):
      # max length set to 5000 because of SpeechT5ScaledPositionalEncoding max_length being set to 5000
      try:
        inputs = processor(text=text_translation_outputs[index], return_tensors="pt")
        input_ids = inputs["input_ids"]
        #print(f"k: {k} {text_translation_outputs[index]}")
        inputs = inputs.to(device)

        speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)
        audio_arr = speech.detach().cpu().numpy().squeeze()

        output_path = f"{directory}/{index}-{speaker_index}.wav"
        sf.write(output_path, audio_arr, 16000)
      except RuntimeError:
        print(f"Speaker {speaker_index} failed to produce speech {k}")
        continue