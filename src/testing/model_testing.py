"""
    Code for checking if TTS model produces valid speech.
    Uses a random xvector from Matthijs/cmu-arctic-xvectors database.
    `pip install -q datasets`
"""

from transformers import SpeechT5ForTextToSpeech, SpeechT5Processor, SpeechT5HifiGan
from datasets import load_dataset
from IPython.display import Audio
import torch

processor=SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
model = SpeechT5ForTextToSpeech.from_pretrained(
    "sophiayk20/speecht5_tts_formal_es"
)
embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split='validation')
speaker_embeddings = torch.tensor(embeddings_dataset[2010]["xvector"]).unsqueeze(0)
text="Hola me llamo Sabrina y esto es una frase en espanol."
inputs = processor(text=text, return_tensors="pt")

vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)

Audio(speech.numpy(), rate=16000)