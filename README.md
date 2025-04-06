# Controlling Formality in Text-to-Speech Translation

## Demo Page
Demo page is available at [this link](https://sophiayk20.github.io/formality-t2st).
I did not design HTML / CSS for the page and all rights for that portion of the code goes to their respectful authors.

## Source Code
Source code is in `/datasets`, `/evaluate`, `/finetune`, `/formality-tuning` and `generate`.
All are Python or shell scripts.

## Tech Stack
- transformers: AutoTokenizer, AutoModelForSeq2SeqLM
- evaluate: sacrebleu
- `fairseq2`, `stopes`, `sonar-space (SONAR)` (speech evaluation)
- `soundfile`, `pyannote`, `pydub`, `librosa` (audio silence detection & trimming)
- `ParlerTTS` (English formal and informal speech generation)
- `coqui-ai/tts` (XTTS for target voice conversion and speech synthesis)

## Models
- Text-to-Text Translation Model: `M2M-100`
- Text-to-Speech Speech Synthesis Model: `SpeechT5`
- Speech Tuning: `ParlerTTS` (English natural-language based prosody-controlled speech generation), `XTTS` (cross-lingual voice conversion and speech production)

## Formality Datasets Used
- Formality Datasets (Text-to-Text Modality)
    - `ACL-IWSLT 2022 Formality Track Datasets`
        - The ACL-IWSLT hosted the Formality Track in 2022 and 2023. They specialized in formality-sensitive translations and methods to improve them. The datasets are available through these links: [2022](https://github.com/amazon-science/contrastive-controlled-mt/tree/main/IWSLT2022) and [2023](https://github.com/amazon-science/contrastive-controlled-mt/tree/main/IWSLT2023). This project focuses on English text to Spanish speech translation, so only the 2022 dataset was used.
    - `FAME-MT Dataset`
        - `FAME-MT Dataset` was made available in June 2024 through `EAMT 2024`. See the paper: "FAME-MT Dataset: Formality Awareness Made Easy for Machine Translation Purposes." The dataset is available here: https://github.com/laniqo-public/fame-mt.
- Formal and informal datasets are concatenated and shuffled. These amount to about 40K points of training data for each formality class.

## New Datasets Released Through This Project
- 8 datasets for DE, ES, FR, IT formal / informal speech with 9K train, 1K validation for a total of 10K for each formal / informal subset of each language pair.
- `sophiayk20/xtts-{FORMALITY}-{LANG_CODE}`

Models used in the paper are also made available in the HuggingFace Hub.

## Helpful Tutorials I referenced throughout project
- [SpeechT5 Model Docs](https://huggingface.co/docs/transformers/en/model_doc/speecht5)
    - If you are interested in Text-to-Speech see the model card for `SpeechT5ForTextToSpeech`.
    - If you click expand, this will explain parameters such as `input_ids` (obtained with `SpeechT5 Tokenizer`), `decoder_input_values` (float values of input mel-spectrogram), `speaker_embeddings` (tensor containing speaker embeddings), and `labels` (float values of target mel spectrogram, which can be obtained with `SpeechT5Processor`).
