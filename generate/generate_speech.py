from datasets import load_dataset
from transformers import SpeechT5Processor, SpeechT5HifiGan, SpeechT5ForTextToSpeech
from tqdm import tqdm
import soundfile as sf
import os
import torch

TEXT_PATH = "/content/drive/MyDrive/formality/base-formal-outputs.txt"
OUTPUT_DIR = "/content/drive/MyDrive/formality/base-base"
SPEECHT5_MODEL_PATH = "sophiayk20/speecht5_tts_voxpopuli_es"

def generate_speech(TEXT_DIR, OUTPUT_DIR, SPEECHT5_MODEL_PATH):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SpeechT5ForTextToSpeech.from_pretrained(SPEECHT5_MODEL_PATH).to(device)
    processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
    vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(device)

    embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
    speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0).to(device)

    with open(TEXT_PATH, "r") as f:
        spanish_lines = [line.rstrip() for line in f.readlines()]

    for idx, spanish_line in enumerate(tqdm(spanish_lines)):
        inputs = processor(text=spanish_line, return_tensors="pt").to(device)

        speech = model.generate_speech(inputs["input_ids"], speaker_embeddings=speaker_embeddings, vocoder=vocoder)

        sf.write(f"{OUTPUT_DIR}/{idx}.wav", speech.cpu().numpy(), samplerate=16000)

generate_speech(TEXT_PATH, OUTPUT_DIR, SPEECHT5_MODEL_PATH)
