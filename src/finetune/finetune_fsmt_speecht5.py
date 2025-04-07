"""
    Code for training formality-sensitive SpeechT5
    `pip install -q datasets soundfile speechbrain librosa`
    and huggingface login
"""

from datasets import load_dataset, Audio
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, Seq2SeqTrainingArguments, Seq2SeqTrainer
from speechbrain.pretrained import EncoderClassifier
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import os
import torch
import librosa

FORMALITY="informal"
LANG_CODE="es"

PUSH_MODEL_NAME = f"sophiayk20/speecht5_tts_{FORMALITY}_{LANG_CODE}"
DATASET_NAME = f"sophiayk20/xtts-{FORMALITY}-{LANG_CODE}"
MODEL_NAME = f"sophiayk20/speecht5_tts_voxpopuli_{LANG_CODE}"
dataset = load_dataset(DATASET_NAME)

# SpeechT5 expects audio data to have a sampling rate of 16kHz
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
tokenizer = processor.tokenizer

## Load Speaker Embedding Model
spk_model_name = "speechbrain/spkrec-xvect-voxceleb"

device = "cuda" if torch.cuda.is_available() else "cpu"
speaker_model = EncoderClassifier.from_hparams(
    source=spk_model_name,
    run_opts={"device": device},
    savedir=os.path.join("/tmp", spk_model_name),
)

def create_speaker_embedding(waveform):
    with torch.no_grad():
        speaker_embeddings = speaker_model.encode_batch(torch.tensor(waveform))
        speaker_embeddings = torch.nn.functional.normalize(speaker_embeddings, dim=2)
        speaker_embeddings = speaker_embeddings.squeeze().cpu().numpy()
    return speaker_embeddings

## Prepare dataset function for SpeechT5 Processor (will generate mel-spectrogram in labels)
## Audio preprocessing with librosa to trim trailing silence in waveforms
def prepare_dataset(example):
    y = example['audio']['array']
    sr = example['audio']['sampling_rate']

    # Trim the leading and trailing silence
    y_trimmed, _ = librosa.effects.trim(y, top_db=30) # defaults to 60, but we need more trimming
    example['audio']['array'] = y_trimmed

    audio = example["audio"]

    example["text"] = example["text"].lower()

    example = processor(
        text=example["text"],
        audio_target=audio["array"],
        sampling_rate=audio["sampling_rate"],
        return_attention_mask=False,
    )

    # strip off the batch dimension
    example["labels"] = example["labels"][0]

    # use SpeechBrain to obtain x-vector
    example["speaker_embeddings"] = create_speaker_embedding(audio["array"])

    return example

# map the function to train and test splits of dataset
dataset['train'] = dataset['train'].map(prepare_dataset, remove_columns=dataset['train'].column_names)
dataset['test'] = dataset['test'].map(prepare_dataset, remove_columns=dataset['test'].column_names)

@dataclass
class TTSDataCollatorWithPadding:
    processor: Any

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        input_ids = [{"input_ids": feature["input_ids"]} for feature in features]
        label_features = [{"input_values": feature["labels"]} for feature in features]
        speaker_features = [feature["speaker_embeddings"] for feature in features]

        # collate the inputs and targets into a batch
        batch = processor.pad(
            input_ids=input_ids, labels=label_features, return_tensors="pt"
        )

        # replace padding with -100 to ignore loss correctly
        batch["labels"] = batch["labels"].masked_fill(
            batch.decoder_attention_mask.unsqueeze(-1).ne(1), -100
        )

        # not used during fine-tuning
        del batch["decoder_attention_mask"]

        # round down target lengths to multiple of reduction factor
        if model.config.reduction_factor > 1:
            target_lengths = torch.tensor(
                [len(feature["input_values"]) for feature in label_features]
            )
            target_lengths = target_lengths.new(
                [
                    length - length % model.config.reduction_factor
                    for length in target_lengths
                ]
            )
            max_length = max(target_lengths)
            batch["labels"] = batch["labels"][:, :max_length]

        # also add in the speaker embeddings
        batch["speaker_embeddings"] = torch.tensor(speaker_features)

        return batch
    
data_collator = TTSDataCollatorWithPadding(processor=processor)
model = SpeechT5ForTextToSpeech.from_pretrained("sophiayk20/speecht5_tts_voxpopuli_es")

# disable cache during training since it's incompatible with gradient checkpointing
model.config.use_cache = False

torch.cuda.empty_cache()
training_args = Seq2SeqTrainingArguments(
    output_dir=f"./speecht5_tts_{FORMALITY}_{LANG_CODE}",  # change to a repo name of your choice
    per_device_train_batch_size=16, # 32: cuda out of memory
    gradient_accumulation_steps=2,
    learning_rate=1e-5,
    warmup_steps=500,
    max_steps=4000,
    gradient_checkpointing=True,
    fp16=True,
    evaluation_strategy="steps",
    per_device_eval_batch_size=8,
    save_steps=1000,
    eval_steps=1000,
    logging_steps=25,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    greater_is_better=False,
    label_names=["labels"],
    push_to_hub=True,
)

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    data_collator=data_collator,
    tokenizer=processor,
)

torch.cuda.empty_cache()
trainer.train() # T4: 4 hrs 4 minutes / L4: 3 hrs 33 minutes
trainer.push_to_hub()