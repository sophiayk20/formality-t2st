"""
    Generates translation with designated M2M-100 model
"""
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
from datasets import load_dataset
from tqdm import tqdm
import torch
import os

def test_translation_model(model_name, dataset_name, target_lang, config_letter, formality):
  # Load pre-trained model and tokenizer
  model_name = model_name
  model = M2M100ForConditionalGeneration.from_pretrained(model_name)
  tokenizer = M2M100Tokenizer.from_pretrained(model_name)

  # Load dataset
  dataset = load_dataset(dataset_name, split='test')

  # Set source and target languages
  source_lang = "en"
  target_lang = target_lang

  # Prepare device (GPU or CPU)
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model = model.to(device)
  directory = "/content/drive/MyDrive/translation-outputs"
  os.makedirs(directory, exist_ok=True)
  # Open the file to write the outputs
  with open(f"{directory}/{config_letter}-{target_lang}-{formality}-outputs.txt", "w") as f:
      batch_size = 32  # Set the batch size for processing
      for idx in tqdm(range(0, len(dataset), batch_size), total=len(dataset) // batch_size, desc="Processing batches"):
          # Select a batch of data from the dataset
          batch_sentences = dataset[idx:idx+batch_size]['en']

          # Tokenize the entire batch
          tokenizer.src_lang = source_lang
          inputs = tokenizer(batch_sentences, return_tensors="pt", padding=True, truncation=True)

          # Move inputs to the same device as the model (GPU/CPU)
          inputs = {key: value.to(device) for key, value in inputs.items()}

          # Generate translations
          generated_tokens = model.generate(**inputs, forced_bos_token_id=tokenizer.get_lang_id(target_lang))

          # Decode and write translated texts to the file
          for generated_token in generated_tokens:
              translated_text = tokenizer.decode(generated_token, skip_special_tokens=True)
              f.write(translated_text + '\n')